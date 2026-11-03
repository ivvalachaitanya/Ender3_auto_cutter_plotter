"""
SVG Parser, Drag Knife Geometry Caster Compensation, & Klipper G-code Generator

Implements:
- SVG parsing: per-segment flattening (exact corners, curves to a 0.01 mm chord tolerance),
  one contour per subpath, all SVG shape types.
- Blade-path noise filter (collinear points and sub-tolerance zig-zags that cause needless swivels).
- Cut ordering: inside-first (holes/inner details before their outline), nearest-neighbour,
  closed contours started on a smooth point, open paths reversed when that is shorter.
- Roland-style drag knife offset compensation with corner swivel arcs (G2/G3).
- Lead-in drag that aligns the blade before a closed contour's start, overcut that follows
  the contour through corners, optional multi-pass cutting.
- Klipper G-code with safe Z lifts and an estimated job time.

Several techniques follow the approach of PolyCut (https://github.com/IridiumIO/PolyCut);
this is an independent Python implementation, no PolyCut code is included.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import svgpathtools
except ImportError:
    svgpathtools = None

try:
    import svgelements
except ImportError:
    svgelements = None

logger = logging.getLogger(__name__)


PX_TO_MM = 25.4 / 96.0
SKIP_ID_MARKERS = ('fiducial', 'aruco', 'verification', 'touch_', 'paper_origin')


def _flatten_curve(point_fn, t0: float, t1: float, p0: complex, p1: complex,
                   tol: float, max_len: float, out: List[complex], depth: int = 0) -> None:
    """Adaptively subdivide a curve on [t0, t1] until every chord is within `tol` of the curve
    and no longer than `max_len`. Appends end points (not p0) to `out`."""
    tm = 0.5 * (t0 + t1)
    pm = point_fn(tm)
    chord = p1 - p0
    clen = abs(chord)
    if clen > 1e-12:
        dev = abs((pm - p0).real * chord.imag - (pm - p0).imag * chord.real) / clen
    else:
        dev = abs(pm - p0)
    if depth < 16 and (dev > tol or clen > max_len):
        _flatten_curve(point_fn, t0, tm, p0, pm, tol, max_len, out, depth + 1)
        _flatten_curve(point_fn, tm, t1, pm, p1, tol, max_len, out, depth + 1)
    else:
        out.append(p1)


class SVGParser:
    """Parses SVG vector paths and converts them into 2D polyline contours in mm.

    Each SVG segment is flattened on its own: straight segments keep their exact end points
    (so corners stay sharp), curves are subdivided adaptively to a chord tolerance.
    """

    def __init__(self, step_size_mm: float = 0.5, tolerance_mm: float = 0.01):
        """
        :param step_size_mm: Maximum chord length on curved segments (mm)
        :param tolerance_mm: Maximum distance between a chord and the true curve (mm)
        """
        self.step_size_mm = step_size_mm
        self.tolerance_mm = tolerance_mm
        if svgpathtools is None and svgelements is None:
            raise RuntimeError("svgelements or svgpathtools is required for SVG parsing.")

    def _flatten_segments(self, segments, is_line, start_of, end_of, point_of) -> List[complex]:
        """Flatten one continuous run of segments into a list of complex points (px)."""
        tol_px = self.tolerance_mm / PX_TO_MM
        max_px = self.step_size_mm / PX_TO_MM
        pts: List[complex] = []
        for seg in segments:
            a, b = start_of(seg), end_of(seg)
            if a is None or b is None:
                continue
            if not pts:
                pts.append(a)
            if is_line(seg):
                pts.append(b)
            else:
                # Seed with a few uniform splits so S-shaped curves are not mistaken for straight.
                n0 = 4
                ts = [i / n0 for i in range(n0 + 1)]
                ps = [a] + [point_of(seg, t) for t in ts[1:-1]] + [b]
                for i in range(n0):
                    _flatten_curve(lambda t, sg=seg: point_of(sg, t), ts[i], ts[i + 1], ps[i], ps[i + 1],
                                   tol_px, max_px, pts)
        return pts

    @staticmethod
    def _to_contour(pts: List[complex], page_height_mm: float, flip_y: bool) -> List[Tuple[float, float]]:
        contour: List[Tuple[float, float]] = []
        for p in pts:
            x_mm = p.real * PX_TO_MM
            y_raw_mm = p.imag * PX_TO_MM
            y_mm = (page_height_mm - y_raw_mm) if flip_y else y_raw_mm
            if not contour or abs(contour[-1][0] - x_mm) > 1e-4 or abs(contour[-1][1] - y_mm) > 1e-4:
                contour.append((x_mm, y_mm))
        return contour

    def parse_svg_paths(self, svg_file: str, flip_y: bool = True) -> List[List[Tuple[float, float]]]:
        """
        Parse vector shapes in an SVG file into polylines (mm), one per subpath.

        Resolves group transforms, matrix scaling and SVG units.

        :param svg_file: Path to SVG file
        :param flip_y: If True, flips Y axis (SVG top-left to Printer Bed bottom-left)
        :return: List of contours, where each contour is a list of (x, y) tuples
        """
        contours: List[List[Tuple[float, float]]] = []

        if svgelements is not None:
            svg = svgelements.SVG.parse(svg_file)
            page_height_mm = (svg.height if svg.height else 1122.52) * PX_TO_MM

            def cpx(pt):
                return None if pt is None else complex(pt.x, pt.y)

            for elem in svg.elements():
                # Accept every geometric shape (rect, circle, ellipse, line, polyline, polygon, path).
                if not isinstance(elem, svgelements.Shape):
                    continue
                elem_id = str(getattr(elem, 'id', '')).lower()
                if any(m in elem_id for m in SKIP_ID_MARKERS):
                    logger.info("Skipping non-cut verification/fiducial element ID '%s' from cut list.", elem_id)
                    continue
                path = elem if isinstance(elem, svgelements.Path) else svgelements.Path(elem)
                if len(path) == 0:
                    continue

                # Each subpath is its own contour so the blade lifts between them.
                for subpath in path.as_subpaths():
                    segs = [seg for seg in subpath if not isinstance(seg, svgelements.Move)]
                    pts = self._flatten_segments(
                        segs,
                        is_line=lambda seg: isinstance(seg, (svgelements.Line, svgelements.Close)),
                        start_of=lambda seg: cpx(seg.start),
                        end_of=lambda seg: cpx(seg.end),
                        point_of=lambda seg, t: cpx(seg.point(t)),
                    )
                    contour = self._to_contour(pts, page_height_mm, flip_y)
                    if len(contour) >= 2:
                        contours.append(contour)

            logger.info("Parsed %d path contours from SVG '%s' using svgelements.", len(contours), svg_file)
            return contours

        # Fallback to svgpathtools if svgelements is unavailable
        paths, attributes = svgpathtools.svg2paths(svg_file)
        page_height_mm = 297.0
        for path, attr in zip(paths, attributes):
            if len(path) == 0:
                continue
            elem_id = str(attr.get('id', '')).lower()
            if any(m in elem_id for m in SKIP_ID_MARKERS):
                continue
            for sub in path.continuous_subpaths():
                pts = self._flatten_segments(
                    list(sub),
                    is_line=lambda seg: isinstance(seg, svgpathtools.Line),
                    start_of=lambda seg: seg.start,
                    end_of=lambda seg: seg.end,
                    point_of=lambda seg, t: seg.point(t),
                )
                contour = self._to_contour(pts, page_height_mm, flip_y)
                if len(contour) >= 2:
                    contours.append(contour)

        return contours


Point2 = Tuple[float, float]

CLOSED_TOL_MM = 0.1           # first/last points closer than this => closed contour
CORNER_ANGLE_RAD = math.radians(45.0)   # turns at least this sharp are real corners (kept by noise filter)
SMOOTH_START_RAD = math.radians(25.0)   # a closed contour may start at a vertex turning less than this


def _turn(a: Point2, b: Point2, c: Point2) -> float:
    """Signed turn angle at b for the path a -> b -> c, in [-pi, pi]."""
    ax, ay = b[0] - a[0], b[1] - a[1]
    bx, by = c[0] - b[0], c[1] - b[1]
    return math.atan2(ax * by - ay * bx, ax * bx + ay * by)


def _dist_to_chord(p: Point2, a: Point2, b: Point2) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-12:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    return abs(dx * (p[1] - a[1]) - dy * (p[0] - a[0])) / length


def is_closed(contour: List[Point2]) -> bool:
    return len(contour) >= 4 and math.hypot(contour[0][0] - contour[-1][0], contour[0][1] - contour[-1][1]) < CLOSED_TOL_MM


def _dedupe(points: List[Point2]) -> List[Point2]:
    out: List[Point2] = []
    for p in points:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 1e-6:
            out.append((float(p[0]), float(p[1])))
    return out


def simplify_blade_path(points: List[Point2], closed: bool, tolerance_mm: float = 0.05) -> List[Point2]:
    """
    Remove vertices that would make the blade swivel for no reason, without moving the cut
    by more than `tolerance_mm` from ANY original point:
      - collinear vertices, and
      - small zig-zags (a turn that reverses its neighbour's turn), typical of traced/noisy paths.
    Real corners (>= 45 deg) and consistently curving runs are never touched.

    `points` for a closed contour is the ring WITHOUT the repeated closing point.
    """
    if tolerance_mm <= 0 or len(points) < 3:
        return list(points)
    orig = list(points)
    total = len(orig)
    kept = list(range(total))      # indices into orig that are still in the path
    min_count = 3 if closed else 2

    def span_error(i_from: int, i_to: int) -> float:
        """Max distance of the original points strictly between orig[i_from] and orig[i_to]
        from the chord joining them; inf if they do not progress along the chord."""
        a, c = orig[i_from], orig[i_to]
        dx, dy = c[0] - a[0], c[1] - a[1]
        l2 = dx * dx + dy * dy
        if l2 < 1e-18:
            return math.inf
        worst, last_t = 0.0, 0.0
        k = (i_from + 1) % total
        while k != i_to:
            px, py = orig[k][0] - a[0], orig[k][1] - a[1]
            t = (px * dx + py * dy) / l2
            if t < last_t or t > 1.0:
                return math.inf     # the span folds back on itself: not a removable wiggle
            last_t = t
            worst = max(worst, abs(dx * py - dy * px) / math.sqrt(l2))
            k = (k + 1) % total
        return worst

    def turn_at(j: int) -> float:
        m = len(kept)
        return _turn(orig[kept[j - 1]], orig[kept[j]], orig[kept[(j + 1) % m]])

    def removable(j: int) -> bool:
        m = len(kept)
        if not closed and (j == 0 or j == m - 1):
            return False
        t = turn_at(j)
        if abs(t) >= CORNER_ANGLE_RAD:
            return False
        if abs(t) >= 1e-3:
            # Not collinear: only drop it if it is a zig-zag (opposes a neighbouring turn).
            t_prev = turn_at(j - 1) if (closed or j >= 2) else 0.0
            t_next = turn_at((j + 1) % m) if (closed or j <= m - 3) else 0.0
            if not (t * t_prev < 0 or t * t_next < 0):
                return False
        return span_error(kept[j - 1], kept[(j + 1) % m]) <= tolerance_mm

    changed = True
    while changed and len(kept) > min_count:
        changed = False
        j = 0 if closed else 1
        while j < len(kept) and len(kept) > min_count:
            if removable(j):
                del kept[j]
                changed = True
            else:
                j += 1
    return [orig[k] for k in kept]


def _point_in_ring(pt: Point2, ring: np.ndarray) -> bool:
    x, y = pt
    xs, ys = ring[:, 0], ring[:, 1]
    xj, yj = np.roll(xs, 1), np.roll(ys, 1)
    straddle = (ys > y) != (yj > y)
    with np.errstate(divide='ignore', invalid='ignore'):
        x_cross = (xj - xs) * (y - ys) / (yj - ys) + xs
    return bool(np.count_nonzero(straddle & (x < x_cross)) % 2)


def _ring_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _start_candidates(ring: List[Point2], min_edge_mm: float) -> List[Tuple[int, Point2, bool]]:
    """Places a closed contour may start: smooth vertices, or the middle of a long-enough edge.
    Returns (index, point, is_midpoint); a midpoint lies on edge index -> index+1."""
    n = len(ring)
    out: List[Tuple[int, Point2, bool]] = []
    for i in range(n):
        if abs(_turn(ring[i - 1], ring[i], ring[(i + 1) % n])) < SMOOTH_START_RAD:
            out.append((i, ring[i], False))
        a, b = ring[i], ring[(i + 1) % n]
        if math.hypot(b[0] - a[0], b[1] - a[1]) >= min_edge_mm:
            out.append((i, ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0), True))
    return out


def optimize_cut_order(contours: List[List[Point2]], start_xy: Point2 = (0.0, 0.0),
                       noise_tolerance_mm: float = 0.0, min_start_edge_mm: float = 1.0,
                       align_weight_mm: float = 60.0) -> List[Tuple[List[Point2], bool]]:
    """
    Prepare contours for drag-knife cutting:
      1. clean each contour (dedupe; optional blade-noise filter, off by default: on clean
         vector designs it trades curve accuracy for fewer vertices, it is meant for traced bitmaps);
      2. order them inside-first (every contour enclosed by a closed contour is cut before it,
         so pieces don't shift once freed), choosing the nearest ready contour each time;
      3. start each closed contour at a smooth point or mid-edge (never on a corner), and pick
         the entry point and direction (closed contours may run either way, open paths may be
         reversed) by travel distance plus a penalty of up to `align_weight_mm` for starting
         against the blade's current heading.
    Returns [(points, closed)]; a closed contour's points repeat the start point at the end.
    """
    items = []
    for c in contours:
        pts = _dedupe(c)
        if len(pts) < 2:
            continue
        closed = is_closed(pts)
        if closed:
            pts = pts[:-1]
        pts = simplify_blade_path(pts, closed, noise_tolerance_mm)
        if closed and len(pts) < 3:
            continue
        arr = np.array(pts, dtype=np.float64)
        items.append({
            'pts': pts, 'closed': closed, 'arr': arr,
            'bbox': (arr[:, 0].min(), arr[:, 1].min(), arr[:, 0].max(), arr[:, 1].max()),
            'area': _ring_area(arr) if closed else 0.0,
        })

    # children[i] = contours enclosed by closed contour i (must be cut before i)
    n = len(items)
    pending_children = [0] * n
    parents: List[List[int]] = [[] for _ in range(n)]
    for i, outer in enumerate(items):
        if not outer['closed']:
            continue
        ox0, oy0, ox1, oy1 = outer['bbox']
        for j, inner in enumerate(items):
            if i == j:
                continue
            ix0, iy0, ix1, iy1 = inner['bbox']
            if ix0 < ox0 or iy0 < oy0 or ix1 > ox1 or iy1 > oy1:
                continue
            if inner['closed'] and inner['area'] >= outer['area']:
                continue
            probe = inner['pts'][len(inner['pts']) // 2]
            if _point_in_ring(inner['pts'][0], outer['arr']) and _point_in_ring(probe, outer['arr']):
                parents[j].append(i)
                pending_children[i] += 1

    def unit(a: Point2, b: Point2) -> Point2:
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = math.hypot(dx, dy) or 1.0
        return dx / d, dy / d

    def options(it):
        """Every way to cut an item: (entry point, start direction, end direction, build points)."""
        if it['closed']:
            for ring in (it['pts'], it['pts'][::-1]):   # closed shapes may be cut either way round
                m = len(ring)
                cands = _start_candidates(ring, min_start_edge_mm) or [(k, q, False) for k, q in enumerate(ring)]
                for k, q, mid in cands:
                    u = unit(ring[k], ring[(k + 1) % m])
                    if mid:
                        pts = [q] + ring[k + 1:] + ring[:k + 1] + [q]
                    else:
                        pts = ring[k:] + ring[:k] + [ring[k]]
                    # with overcut the blade finishes heading along the first edge again
                    yield q, u, u, pts
        else:
            for pts in (it['pts'], it['pts'][::-1]):
                yield pts[0], unit(pts[0], pts[1]), unit(pts[-2], pts[-1]), pts

    ordered: List[Tuple[List[Point2], bool]] = []
    done = [False] * n
    cur = start_xy
    blade_dir: Optional[Point2] = None
    for _ in range(n):
        best = None  # (cost, idx, pts, end_dir)
        ready = [i for i in range(n) if not done[i] and pending_children[i] == 0]
        if not ready:  # containment cycle (degenerate input): release the rest in file order
            ready = [next(i for i in range(n) if not done[i])]
        for i in ready:
            for q, u_start, u_end, pts in options(items[i]):
                cost = math.hypot(q[0] - cur[0], q[1] - cur[1])
                if blade_dir is not None:
                    # The blade keeps its last heading while lifted; starting in that same
                    # direction means it lands on the line instead of swinging into it.
                    cos_a = blade_dir[0] * u_start[0] + blade_dir[1] * u_start[1]
                    cost += align_weight_mm * (1.0 - cos_a) / 2.0
                if best is None or cost < best[0]:
                    best = (cost, i, pts, u_end)
        _cost, i, pts, u_end = best
        ordered.append((pts, items[i]['closed']))
        done[i] = True
        for parent in parents[i]:
            pending_children[parent] -= 1
        cur = pts[-1]
        blade_dir = u_end
    return ordered


class DragKnifePlanner:
    """Generates drag-knife toolhead trajectories with swivel corner compensation."""

    def __init__(self, blade_offset_mm: float = 0.35, swivel_angle_threshold_deg: float = 30.0,
                 overcut_mm: float = 0.5, lead_in_margin_mm: float = 0.3):
        """
        :param blade_offset_mm: Distance between toolhead pivot center and trailing blade tip (mm)
        :param swivel_angle_threshold_deg: Minimum corner direction change (deg) that gets a swivel arc
        :param overcut_mm: Distance to re-cut past the start of closed contours for clean weeding (mm)
        :param lead_in_margin_mm: Closed contours are plunged this far (plus the blade offset) behind
            their start and dragged in, so the blade is aligned before the real cut begins (mm, 0 = off)
        """
        self.blade_offset = blade_offset_mm
        self.swivel_threshold_rad = math.radians(swivel_angle_threshold_deg)
        self.overcut_mm = overcut_mm
        self.lead_in_margin_mm = lead_in_margin_mm
        # Heading of the blade after the previously planned contour. A lifted drag blade keeps
        # its heading, so contours planned in cutting order can start with an in-place swivel.
        self.blade_dir: Optional[Point2] = None

    def reset_blade_state(self) -> None:
        """Forget the blade heading (e.g. before planning a new, independent job)."""
        self.blade_dir = None

    def _entry(self, tip: Point2, from_dir: Point2, to_dir: Point2):
        """Plunge so the tip lands ON `tip` (blade still heading `from_dir`), then swivel in place
        to `to_dir`. Returns (plunge_pivot, arc_move_or_None)."""
        L = self.blade_offset
        plunge = (tip[0] + L * from_dir[0], tip[1] + L * from_dir[1])
        delta = self._angle_between(from_dir, to_dir)
        if abs(delta) < math.radians(2.0):
            return plunge, None
        return plunge, {
            'type': 'arc',
            'x': tip[0] + L * to_dir[0],
            'y': tip[1] + L * to_dir[1],
            'i': -L * from_dir[0],
            'j': -L * from_dir[1],
            'cw': delta < 0,
            'corner': tip,
        }

    @staticmethod
    def _normalize(vx: float, vy: float) -> Tuple[float, float]:
        """Normalize 2D vector."""
        mag = math.hypot(vx, vy)
        if mag == 0:
            return 0.0, 0.0
        return vx / mag, vy / mag

    @staticmethod
    def _angle_between(u: Tuple[float, float], v: Tuple[float, float]) -> float:
        """Signed angle from unit vector u to unit vector v in range [-pi, pi]."""
        dot = max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))
        det = u[0] * v[1] - u[1] * v[0]
        return math.atan2(det, dot)

    def _with_overcut(self, ring_closed: List[Point2]) -> List[Point2]:
        """Append `overcut_mm` of the contour's own start to its end, following it through corners."""
        pts = list(ring_closed)
        remaining = self.overcut_mm
        i = 0
        while remaining > 1e-9 and i < len(ring_closed) - 1:
            a, b = ring_closed[i], ring_closed[i + 1]
            seg = math.hypot(b[0] - a[0], b[1] - a[1])
            if seg >= remaining:
                f = remaining / seg
                pts.append((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
                break
            pts.append(b)
            remaining -= seg
            i += 1
        return pts

    def plan_contour(self, contour: List[Point2], closed: Optional[bool] = None
                     ) -> List[Dict[str, Union[str, float, Tuple[float, float]]]]:
        """
        Plan drag knife trajectory for a single polyline contour.

        Returns list of move instructions:
        - {'type': 'lead_in', 'start_pivot': (x, y), 'plunge_pivot': (x, y) | None, 'entry_arc': arc | None,
           'repeat_plunge_pivot': (x, y), 'repeat_entry_arc': arc | None, 'blade_tip': (x, y)}
          plunge at plunge_pivot (or start_pivot), then either swivel with entry_arc or drag
          straight to start_pivot; repeat_* is the same for extra passes over this contour.
        - {'type': 'line', 'x': x, 'y': y}
        - {'type': 'arc', 'x': end_x, 'y': end_y, 'i': offset_i, 'j': offset_j, 'cw': bool}

        :param contour: (x, y) points of the cut line; a closed contour repeats its first point at the end
        :param closed: force closed/open; None = detect from the end points
        """
        contour = _dedupe(contour)
        if len(contour) < 2:
            return []
        if closed is None:
            closed = is_closed(contour)
        if closed and self.overcut_mm > 0.0:
            contour = self._with_overcut(contour)

        moves = []
        L = self.blade_offset

        segments = []
        for i in range(len(contour) - 1):
            p1, p2 = contour[i], contour[i + 1]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                segments.append((p1, p2, (dx / dist, dy / dist), dist))
        if not segments:
            return []

        first_p1, _, u_0, _ = segments[0]
        # Pivot sits L ahead of the blade tip along the cut direction.
        start_pivot = (first_p1[0] + L * u_0[0], first_p1[1] + L * u_0[1])
        plunge_pivot, entry_arc = None, None
        if L > 0 and self.blade_dir is not None:
            # Known heading: land the tip on the start point and swivel in place (like a corner).
            plunge_pivot, entry_arc = self._entry(first_p1, self.blade_dir, u_0)
        elif closed and self.lead_in_margin_mm > 0.0:
            # Unknown heading (first cut): plunge (L + margin) behind the start pivot and drag in
            # along the first direction, so the blade has swung into line before the start point.
            back = L + self.lead_in_margin_mm
            plunge_pivot = (start_pivot[0] - back * u_0[0], start_pivot[1] - back * u_0[1])
        lead_in = {
            'type': 'lead_in',
            'start_pivot': start_pivot,
            'plunge_pivot': plunge_pivot,
            'entry_arc': entry_arc,
            'blade_tip': first_p1,
        }
        moves.append(lead_in)

        current_u = u_0
        for p1, p2, next_u, _dist in segments:
            delta_angle = self._angle_between(current_u, next_u)
            if abs(delta_angle) > self.swivel_threshold_rad and L > 0:
                # Swivel about the corner p1: pivot moves on a circle of radius L from
                # p1 + L*current_u to p1 + L*next_u. G2 (cw) for right turns, G3 for left.
                moves.append({
                    'type': 'arc',
                    'x': p1[0] + L * next_u[0],
                    'y': p1[1] + L * next_u[1],
                    'i': -L * current_u[0],
                    'j': -L * current_u[1],
                    'cw': delta_angle < 0,
                    'corner': p1,
                })
            current_u = next_u
            moves.append({
                'type': 'line',
                'x': p2[0] + L * current_u[0],
                'y': p2[1] + L * current_u[1],
            })

        # Extra passes start with the blade heading the way this contour ended.
        lead_in['repeat_plunge_pivot'], lead_in['repeat_entry_arc'] = (
            self._entry(first_p1, current_u, u_0) if L > 0 else (start_pivot, None))
        self.blade_dir = current_u
        return moves


MIN_CLEARANCE_MM = 5.0


def _arc_length(start: Point2, arc: Dict) -> float:
    """Length of a G2/G3 arc move starting at `start` (I/J relative to start)."""
    cx, cy = start[0] + arc['i'], start[1] + arc['j']
    ax, ay = start[0] - cx, start[1] - cy
    bx, by = arc['x'] - cx, arc['y'] - cy
    return math.hypot(arc['i'], arc['j']) * abs(math.atan2(ax * by - ay * bx, ax * bx + ay * by))


def safe_travel_z(z_safe: float, z_cut: float) -> float:
    """Return a travel height at least MIN_CLEARANCE_MM above the cutting plane."""
    return max(float(z_safe), float(z_cut) + MIN_CLEARANCE_MM)


class GCodeGenerator:
    """Translates planned drag-knife trajectories into Klipper G-code format."""

    def __init__(
        self,
        z_safe: float = 5.0,
        z_cut: float = 0.0,
        f_cut: float = 1016.0,
        f_travel: float = 2540.0,
        f_plunge: float = 127.0,
        dry_run: bool = False,
        z_lift_swivel: float = 0.0,
        pause_before_cut: bool = False,
        passes: int = 1,
        pass_depth_step: float = 0.0,
        use_gcode_arcs: bool = False,
    ):
        """
        Initialize G-code Generator.

        :param z_safe: Safe travel Z height in mm (absolute; raised to at least z_cut + MIN_CLEARANCE_MM)
        :param z_cut: Cutting Z depth in mm (default: 0.0 mm)
        :param f_cut: Cutting feedrate in mm/min (default: 1016 mm/min)
        :param f_travel: Rapid travel feedrate in mm/min (default: 2540 mm/min)
        :param f_plunge: Plunge feedrate in mm/min (default: 127 mm/min)
        :param dry_run: If True, sets z_cut = z_safe (no blade contact)
        :param z_lift_swivel: Lift-and-Turn height ABOVE the cut depth in mm (0.0 = disabled/in-plane)
        :param pause_before_cut: If True, inserts PAUSE command before cutting starts to wait for user confirmation
        :param passes: Number of times each contour is cut (thick material)
        :param pass_depth_step: Extra depth per pass in mm; pass k cuts at z_cut - k * pass_depth_step
        :param use_gcode_arcs: Emit swivels as G2/G3 (needs [gcode_arcs] in printer.cfg); otherwise as
            short G1 segments within 0.005 mm of the arc, which works on any firmware
        """
        # Safe height must clear the cutting plane, not just Z=0 (z_cut may be > 0).
        self.z_safe = safe_travel_z(z_safe, z_cut)
        self.z_cut = self.z_safe if dry_run else z_cut
        self.f_cut = f_cut
        self.f_travel = f_travel
        self.f_plunge = f_plunge
        self.dry_run = dry_run
        self.z_lift_swivel = z_lift_swivel
        self.pause_before_cut = pause_before_cut
        self.passes = max(1, int(passes))
        self.pass_depth_step = max(0.0, float(pass_depth_step))
        self.use_gcode_arcs = bool(use_gcode_arcs)

    def _arc_gcode(self, start: Point2, arc: Dict, feed: float, comment: str) -> List[str]:
        """G-code for a swivel arc from `start`: one G2/G3, or G1 chords within 0.005 mm of it."""
        if self.use_gcode_arcs:
            g_cmd = "G2" if arc['cw'] else "G3"
            return [f"{g_cmd} X{arc['x']:.4f} Y{arc['y']:.4f} I{arc['i']:.4f} J{arc['j']:.4f} F{feed:.1f} ; {comment}"]
        cx, cy = start[0] + arc['i'], start[1] + arc['j']
        r = math.hypot(arc['i'], arc['j'])
        a0 = math.atan2(start[1] - cy, start[0] - cx)
        a1 = math.atan2(arc['y'] - cy, arc['x'] - cx)
        sweep = a1 - a0
        if arc['cw'] and sweep > 0:
            sweep -= 2 * math.pi
        elif not arc['cw'] and sweep < 0:
            sweep += 2 * math.pi
        tol = 0.005
        max_step = 2.0 * math.acos(max(-1.0, 1.0 - tol / r)) if r > tol else math.pi
        n = max(1, int(math.ceil(abs(sweep) / max_step)))
        out = []
        for k in range(1, n + 1):
            if k == n:
                x, y = arc['x'], arc['y']
            else:
                a = a0 + sweep * k / n
                x, y = cx + r * math.cos(a), cy + r * math.sin(a)
            out.append(f"G1 X{x:.4f} Y{y:.4f} F{feed:.1f}" + (f" ; {comment}" if k == 1 else ""))
        return out

    def generate_gcode(
        self,
        planned_contours: List[List[Dict[str, Union[str, float, Tuple[float, float]]]]],
        touch_points: Optional[List[Tuple[float, float]]] = None,
        paper_origin: Optional[Tuple[float, float]] = None,
    ) -> str:
        """
        Generate complete G-code string for all parsed and planned contours.

        :param planned_contours: List of move sequences (one sequence per path contour)
        :param touch_points: Optional list of 3 (X, Y) physical bed coordinates to touch before cutting
        :param paper_origin: Optional (X, Y) physical bed coordinate of paper origin (0,0) to park at
        :return: Multi-line G-code string
        """
        lines = []

        # G-code Header
        lines.append("; ========================================================")
        lines.append("; Ender 3 Print & Cut Vinyl Cutting G-code")
        lines.append(f"; Generated by vision_cutter.py (Dry Run: {self.dry_run})")
        lines.append(f"; Parameters: Z_safe={self.z_safe:.2f}mm, Z_cut={self.z_cut:.2f}mm, Z_lift_swivel={self.z_lift_swivel:.2f}mm")
        lines.append(f"; Feedrates: Cut={self.f_cut:.0f}mm/min, Travel={self.f_travel:.0f}mm/min, Plunge={self.f_plunge:.0f}mm/min")
        lines.append("; ========================================================")
        lines.append("G21 ; Set units to millimeters")
        lines.append("G90 ; Use absolute coordinates")
        lines.append("M83 ; Relative extruder positioning (extruder disabled)")
        lines.append("BED_MESH_PROFILE LOAD=default ; Load Klipper bed warp compensation mesh")
        lines.append(f"G0 Z{self.z_safe:.3f} F{self.f_travel:.1f} ; Move to clearance level")

        # Optional Pre-Cut Touch Verification Routine
        if touch_points and len(touch_points) >= 3:
            lines.append("")
            lines.append("; ========================================================")
            lines.append("; --- Pre-Cut Touch Verification Routine ---")
            lines.append("; Moves blade to touch 3 test dots on printed sheet,")
            lines.append("; then parks at Paper Origin (0,0) and pauses for confirmation.")
            lines.append("; ========================================================")
            lines.append(f"G0 Z{self.z_safe:.3f} F{self.f_travel:.1f} ; Ensure safe travel height")
            for i, pt in enumerate(touch_points, 1):
                px, py = pt[0], pt[1]
                lines.append(f"; --- Touch Verification Dot #{i} ---")
                lines.append(f"G1 X{px:.4f} Y{py:.4f} F{self.f_travel:.1f} ; Move above Dot #{i}")
                lines.append(f"G1 Z{self.z_cut:.4f} F{self.f_plunge:.1f} ; Lower needle to touch dot")
                lines.append("G4 P1500 ; Dwell 1.5s for visual inspection")
                lines.append(f"G1 Z{self.z_safe:.4f} F{self.f_travel:.1f} ; Lift blade to safe height")
            if paper_origin is not None:
                ox, oy = paper_origin[0], paper_origin[1]
                lines.append("; --- Park at Paper Origin (0,0) ---")
                lines.append(f"G1 X{ox:.4f} Y{oy:.4f} F{self.f_travel:.1f} ; Rapid to Paper Origin (0,0)")
                hover_z = self.z_cut + 1.0
                lines.append(f"G1 Z{hover_z:.4f} F1500.0 ; Hover 1mm above Paper (0,0) for visual alignment check")
                lines.append("G4 P2000 ; Dwell 2s for visual alignment check at origin")
                lines.append(f"G0 Z{self.z_safe:.3f} F{self.f_travel:.1f} ; LIFT BLADE TO SAFE HEIGHT (>= 5mm above cut) BEFORE PAUSE TO PREVENT DRAGGING")
            else:
                lines.append(f"G0 Z{self.z_safe:.3f} F{self.f_travel:.1f} ; Ensure safe travel height (>= 5mm above cut) before pause")

            lines.append("")
            lines.append("M117 Alignment Check: Inspect dots & Resume")
            lines.append(f'RESPOND TYPE=echo MSG="Needle visited test dots & origin. Lifted to safe Z ({self.z_safe:.1f}mm). Click RESUME / RESUME_BASE to cut."')
            lines.append("PAUSE ; Pause machine safely at safe height")
            lines.append(f"G0 Z{self.z_safe:.3f} F{self.f_travel:.1f} ; Maintain safe travel height upon resume")
            lines.append("")

        if self.pause_before_cut and not touch_points:
            lines.append("")
            lines.append("; --- User Inspection Pause ---")
            lines.append(f"G0 Z{self.z_safe:.3f} F{self.f_travel:.1f} ; Ensure safe height (>= 5mm above cut) before pause")
            lines.append("M117 Check material & click Resume")
            lines.append('RESPOND TYPE=echo MSG="Job Paused! Inspect material & click RESUME to start cut."')
            lines.append("PAUSE ; Pause machine and wait for user to press Resume/OK")
            lines.append(f"G0 Z{self.z_safe:.3f} F{self.f_travel:.1f} ; Maintain safe travel height upon resume")
        lines.append("")

        contour_count = 0
        cut_len = 0.0     # mm cut with the blade down
        travel_len = 0.0  # mm of XY travel at safe height
        pos = None
        for contour_moves in planned_contours:
            if not contour_moves or contour_moves[0]['type'] != 'lead_in':
                continue
            contour_count += 1
            lead_in = contour_moves[0]
            start_pivot = lead_in['start_pivot']

            for pass_idx in range(self.passes):
                if pass_idx == 0:
                    plunge_pivot = lead_in.get('plunge_pivot') or start_pivot
                    entry_arc = lead_in.get('entry_arc')
                else:
                    plunge_pivot = lead_in.get('repeat_plunge_pivot') or start_pivot
                    entry_arc = lead_in.get('repeat_entry_arc')
                z_work = self.z_cut if self.dry_run else self.z_cut - pass_idx * self.pass_depth_step
                label = f"; --- Path #{contour_count}" + (f" (pass {pass_idx + 1}/{self.passes}, Z={z_work:.3f})" if self.passes > 1 else "") + " ---"
                lines.append(label)

                # 1. Travel to the plunge point at safe Z
                px, py = plunge_pivot
                lines.append(f"G1 X{px:.4f} Y{py:.4f} F{self.f_travel:.1f} ; Rapid to initial position")
                if pos is not None:
                    travel_len += math.hypot(px - pos[0], py - pos[1])
                pos = (px, py)

                # 2. Controlled two-stage plunge to cut depth
                if not self.dry_run:
                    approach_z = min(self.z_safe, z_work + 0.5)
                    if approach_z > z_work:
                        lines.append(f"G1 Z{approach_z:.4f} F2540.0 ; Approach surface")
                    lines.append(f"G1 Z{z_work:.4f} F{self.f_plunge:.1f} ; Plunge blade")
                else:
                    lines.append(f"G1 Z{z_work:.4f} F{self.f_travel:.1f} ; Dry run clearance")

                # 3. Align the blade before cutting: swivel in place about the start point when the
                #    blade heading is known, otherwise drag it into line along the first direction.
                if entry_arc is not None:
                    lines.extend(self._arc_gcode(pos, entry_arc, self.f_cut, "Entry swivel to align blade"))
                    cut_len += _arc_length(pos, entry_arc)
                    pos = (entry_arc['x'], entry_arc['y'])
                elif plunge_pivot != start_pivot:
                    sx, sy = start_pivot
                    lines.append(f"G1 X{sx:.4f} Y{sy:.4f} F{self.f_cut:.1f} ; Lead-in drag to align blade")
                    cut_len += math.hypot(sx - pos[0], sy - pos[1])
                    pos = (sx, sy)

                # 4. Cut the contour
                for move in contour_moves[1:]:
                    mtype = move['type']
                    if mtype == 'line':
                        lines.append(f"G1 X{move['x']:.4f} Y{move['y']:.4f} F{self.f_cut:.1f}")
                        cut_len += math.hypot(move['x'] - pos[0], move['y'] - pos[1])
                    elif mtype == 'arc':
                        cut_len += _arc_length(pos, move)
                        if self.z_lift_swivel > 0.0 and not self.dry_run:
                            lift_z = z_work + self.z_lift_swivel
                            lines.append(f"G1 Z{lift_z:.4f} F{self.f_travel:.1f} ; Lift Z {self.z_lift_swivel:.2f}mm above cut for turn")
                            lines.extend(self._arc_gcode(pos, move, self.f_travel, "Swivel arc in air"))
                            lines.append(f"G1 Z{z_work:.4f} F{self.f_plunge:.1f} ; Re-plunge blade")
                        else:
                            lines.extend(self._arc_gcode(pos, move, self.f_cut, "Swivel arc"))
                    else:
                        continue
                    pos = (move['x'], move['y'])

                # 5. Lift Z to safe clearance level at end of the pass
                lines.append(f"G1 Z{self.z_safe:.4f} F{self.f_travel:.1f} ; Lift Z to clearance level")
                lines.append("")

        plunges = contour_count * self.passes
        est_s = (cut_len / max(self.f_cut, 1e-6) + travel_len / max(self.f_travel, 1e-6)) * 60.0 \
            + plunges * (2 * abs(self.z_safe - self.z_cut) / max(self.f_travel, 1e-6) * 60.0 + 0.5 / max(self.f_plunge, 1e-6) * 60.0)
        self.stats = {"contours": contour_count, "passes": self.passes, "cut_mm": cut_len,
                      "travel_mm": travel_len, "estimated_s": est_s}
        lines.insert(5, f"; Paths: {contour_count} x {self.passes} pass(es) | Cut length: {cut_len / 1000.0:.2f} m | Estimated time: {int(est_s // 60)}m {int(est_s % 60)}s")

        # G-code Footer
        lines.append("; ========================================================")
        lines.append("; End of Cut Job")
        lines.append(f"G1 Z{self.z_safe:.4f} F{self.f_travel:.1f}")
        lines.append("G1 X0.0000 Y200.0000 F2540.0 ; Present job")
        lines.append("; ========================================================")

        return "\n".join(lines)
