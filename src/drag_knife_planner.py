"""
SVG Parser, Drag Knife Geometry Caster Compensation, & Klipper G-code Generator

Implements:
- SVG parsing & vector line discretization using svgpathtools / shapely.
- Roland-style drag knife offset caster compensation.
- Corner swivel arc generation (G2/G3) for direction changes > 30 deg.
- Klipper G-code translation with safe Z lifts and speed control.
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


class SVGParser:
    """Parses SVG vector paths and converts them into 2D polyline contours in mm."""

    def __init__(self, step_size_mm: float = 0.5):
        """
        Initialize SVG Parser.

        :param step_size_mm: Maximum segment length when sampling curves (mm)
        """
        self.step_size_mm = step_size_mm
        if svgpathtools is None and svgelements is None:
            raise RuntimeError("svgelements or svgpathtools is required for SVG parsing.")

    def parse_svg_paths(self, svg_file: str, flip_y: bool = True) -> List[List[Tuple[float, float]]]:
        """
        Parse vector paths in SVG file into discretized polylines (mm).

        Resolves all group transforms, matrix scaling, and SVG units.

        :param svg_file: Path to SVG file
        :param flip_y: If True, flips Y axis (SVG top-left to Printer Bed bottom-left)
        :return: List of contours, where each contour is a list of (x, y) tuples
        """
        contours: List[List[Tuple[float, float]]] = []
        px_to_mm = 25.4 / 96.0

        if svgelements is not None:
            svg = svgelements.SVG.parse(svg_file)
            page_height_mm = (svg.height if svg.height else 1122.52) * px_to_mm

            for elem in svg.elements():
                # Accept every geometric shape (rect, circle, ellipse, line, polyline,
                # polygon, path); convert non-path shapes to paths for uniform sampling.
                if not isinstance(elem, svgelements.Shape):
                    continue
                elem_id = str(getattr(elem, 'id', '')).lower()
                if any(s in elem_id for s in ('fiducial', 'aruco', 'verification', 'touch_', 'paper_origin')):
                    logger.info("Skipping non-cut verification/fiducial element ID '%s' from cut list.", elem_id)
                    continue
                path = elem if isinstance(elem, svgelements.Path) else svgelements.Path(elem)
                if len(path) == 0:
                    continue

                # Sample each subpath separately so the blade lifts between them
                # instead of cutting a straight line across the gap.
                for subpath in path.as_subpaths():
                    sub = svgelements.Path(subpath)
                    length_mm = sub.length() * px_to_mm
                    if length_mm < 1e-6:
                        continue
                    num_samples = max(2, int(math.ceil(length_mm / self.step_size_mm)))

                    contour: List[Tuple[float, float]] = []
                    for i in range(num_samples + 1):
                        t = i / float(num_samples)
                        pt = sub.point(t)
                        if pt is None:
                            continue
                        x_mm = pt.x * px_to_mm
                        y_raw_mm = pt.y * px_to_mm
                        y_mm = (page_height_mm - y_raw_mm) if flip_y else y_raw_mm

                        if not contour or (abs(contour[-1][0] - x_mm) > 1e-4 or abs(contour[-1][1] - y_mm) > 1e-4):
                            contour.append((x_mm, y_mm))

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
            if 'fiducial' in elem_id or 'aruco' in elem_id:
                continue
            # Sample each continuous subpath separately (no cut across gaps)
            for sub in path.continuous_subpaths():
                length_mm = sub.length() * px_to_mm
                if length_mm < 1e-6:
                    continue
                contour = []
                num_samples = max(2, int(math.ceil(length_mm / self.step_size_mm)))
                for i in range(num_samples + 1):
                    t = i / float(num_samples)
                    pt = sub.point(t)
                    x_mm = pt.real * px_to_mm
                    y_raw = pt.imag * px_to_mm
                    y_mm = (page_height_mm - y_raw) if flip_y else y_raw
                    if not contour or (abs(contour[-1][0] - x_mm) > 1e-4 or abs(contour[-1][1] - y_mm) > 1e-4):
                        contour.append((x_mm, y_mm))
                if len(contour) >= 2:
                    contours.append(contour)

        return contours


class DragKnifePlanner:
    """Generates drag-knife toolhead trajectories with swivel corner compensation."""

    def __init__(self, blade_offset_mm: float = 0.35, swivel_angle_threshold_deg: float = 30.0, overcut_mm: float = 0.5):
        """
        Initialize Drag Knife Planner.

        :param blade_offset_mm: Distance between toolhead pivot center and trailing blade tip (mm)
        :param swivel_angle_threshold_deg: Minimum corner direction change angle (degrees) to trigger swivel arc
        :param overcut_mm: Overcut distance along initial segment for closed contours to ensure clean weeding (mm)
        """
        self.blade_offset = blade_offset_mm
        self.swivel_threshold_rad = math.radians(swivel_angle_threshold_deg)
        self.overcut_mm = overcut_mm

    @staticmethod
    def _normalize(vx: float, vy: float) -> Tuple[float, float]:
        """Normalize 2D vector."""
        mag = math.hypot(vx, vy)
        if mag == 0:
            return 0.0, 0.0
        return vx / mag, vy / mag

    @staticmethod
    def _angle_between(u: Tuple[float, float], v: Tuple[float, float]) -> float:
        """
        Calculate signed angle from unit vector u to unit vector v in range [-pi, pi].
        """
        dot = max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))
        det = u[0] * v[1] - u[1] * v[0]
        return math.atan2(det, dot)

    def plan_contour(self, contour: List[Tuple[float, float]]) -> List[Dict[str, Union[str, float, Tuple[float, float]]]]:
        """
        Plan drag knife trajectory for a single polyline contour.

        Returns list of move instructions:
        - {'type': 'line', 'x': x, 'y': y}
        - {'type': 'arc', 'x': end_x, 'y': end_y, 'i': offset_i, 'j': offset_j, 'cw': bool}

        :param contour: List of (x, y) coordinates representing cut line
        :return: List of move dictionaries
        """
        if len(contour) < 2:
            return []

        moves = []
        L = self.blade_offset

        # Calculate unit direction vectors for each segment
        segments = []
        for i in range(len(contour) - 1):
            p1 = contour[i]
            p2 = contour[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                u = (dx / dist, dy / dist)
                segments.append((p1, p2, u, dist))

        if not segments:
            return []

        # Start of contour: Lead-in move to align blade along first segment direction u_0
        first_p1, _, u_0, _ = segments[0]

        # Initial pivot position for starting cut at first_p1
        # Pivot point is displaced forward along u_0 by blade offset L
        start_pivot = (first_p1[0] + L * u_0[0], first_p1[1] + L * u_0[1])

        # Pre-orient move instruction (lead-in)
        moves.append({
            'type': 'lead_in',
            'start_pivot': start_pivot,
            'blade_tip': first_p1,
        })

        current_u = u_0

        for i in range(len(segments)):
            p1, p2, next_u, dist = segments[i]

            # Calculate direction change angle
            delta_angle = self._angle_between(current_u, next_u)

            if abs(delta_angle) > self.swivel_threshold_rad and L > 0:
                # Corner swivel arc needed at vertex p1
                # Pivot current position: p1 + L * current_u
                # Pivot target position: p1 + L * next_u
                target_pivot = (p1[0] + L * next_u[0], p1[1] + L * next_u[1])

                # Arc center relative to starting pivot position:
                # Center is corner point p1.
                # Starting pivot S = p1 + L * current_u -> p1 - S = -L * current_u
                offset_i = -L * current_u[0]
                offset_j = -L * current_u[1]

                # Clockwise (G2) if delta_angle < 0, Counter-Clockwise (G3) if delta_angle > 0
                cw = delta_angle < 0

                moves.append({
                    'type': 'arc',
                    'x': target_pivot[0],
                    'y': target_pivot[1],
                    'i': offset_i,
                    'j': offset_j,
                    'cw': cw,
                    'corner': p1,
                })

            # Always update direction to current segment (prevents angular drift)
            current_u = next_u

            # Cut move along segment: pivot moves to p2 + L * current_u
            end_pivot = (p2[0] + L * current_u[0], p2[1] + L * current_u[1])
            moves.append({
                'type': 'line',
                'x': end_pivot[0],
                'y': end_pivot[1],
            })

        # Closed-loop overcut: extend cut slightly along initial segment for clean weeding
        if len(contour) >= 3 and len(segments) >= 2 and self.overcut_mm > 0.0:
            p_first = contour[0]
            p_last = contour[-1]
            if math.hypot(p_first[0] - p_last[0], p_first[1] - p_last[1]) < 0.1:
                first_u = segments[0][2]
                delta_angle = self._angle_between(current_u, first_u)
                if abs(delta_angle) > self.swivel_threshold_rad and L > 0:
                    p_close = segments[0][0]
                    target_pivot = (p_close[0] + L * first_u[0], p_close[1] + L * first_u[1])
                    moves.append({
                        'type': 'arc',
                        'x': target_pivot[0],
                        'y': target_pivot[1],
                        'i': -L * current_u[0],
                        'j': -L * current_u[1],
                        'cw': delta_angle < 0,
                        'corner': p_close,
                    })
                # Re-trace the start of the contour for overcut_mm (following its first
                # segments, not a straight line), stopping at any corner that would need a swivel.
                remaining = self.overcut_mm
                prev_u = first_u
                for p1, _p2, u, dist in segments:
                    if abs(self._angle_between(prev_u, u)) > self.swivel_threshold_rad:
                        break
                    step = min(remaining, dist)
                    tip = (p1[0] + step * u[0], p1[1] + step * u[1])
                    moves.append({
                        'type': 'line',
                        'x': tip[0] + L * u[0],
                        'y': tip[1] + L * u[1],
                    })
                    remaining -= step
                    prev_u = u
                    if remaining <= 1e-9:
                        break

        return moves


MIN_CLEARANCE_MM = 5.0


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
        for contour_moves in planned_contours:
            if not contour_moves:
                continue

            contour_count += 1
            lines.append(f"; --- Path #{contour_count} ---")

            lead_in = contour_moves[0]
            if lead_in['type'] == 'lead_in':
                start_x, start_y = lead_in['start_pivot']
                # 1. Rapid move to initial pivot location at safe Z
                lines.append(f"G1 X{start_x:.4f} Y{start_y:.4f} F{self.f_travel:.1f} ; Rapid to initial position")

                # 2. Controlled two-stage plunge to cut depth (safe relative approach)
                if not self.dry_run:
                    approach_z = min(self.z_safe, self.z_cut + 0.5)
                    if approach_z > self.z_cut:
                        lines.append(f"G1 Z{approach_z:.4f} F2540.0 ; Approach surface")
                    lines.append(f"G1 Z{self.z_cut:.4f} F{self.f_plunge:.1f} ; Plunge blade")
                else:
                    lines.append(f"G1 Z{self.z_cut:.4f} F{self.f_travel:.1f} ; Dry run clearance")

            # Execute moves along contour
            for move in contour_moves[1:]:
                mtype = move['type']
                if mtype == 'line':
                    lines.append(f"G1 X{move['x']:.4f} Y{move['y']:.4f} F{self.f_cut:.1f}")
                elif mtype == 'arc':
                    g_cmd = "G2" if move['cw'] else "G3"
                    if self.z_lift_swivel > 0.0 and not self.dry_run:
                        lift_z = self.z_cut + self.z_lift_swivel
                        lines.append(f"G1 Z{lift_z:.4f} F{self.f_travel:.1f} ; Lift Z {self.z_lift_swivel:.2f}mm above cut for turn")
                        lines.append(f"{g_cmd} X{move['x']:.4f} Y{move['y']:.4f} I{move['i']:.4f} J{move['j']:.4f} F{self.f_travel:.1f} ; Swivel arc in air")
                        lines.append(f"G1 Z{self.z_cut:.4f} F{self.f_plunge:.1f} ; Re-plunge blade")
                    else:
                        lines.append(f"{g_cmd} X{move['x']:.4f} Y{move['y']:.4f} I{move['i']:.4f} J{move['j']:.4f} F{self.f_cut:.1f} ; Swivel arc")

            # 3. Lift Z to safe clearance level at end of contour
            lines.append(f"G1 Z{self.z_safe:.4f} F{self.f_travel:.1f} ; Lift Z to clearance level")
            lines.append("")

        # G-code Footer
        lines.append("; ========================================================")
        lines.append("; End of Cut Job")
        lines.append(f"G1 Z{self.z_safe:.4f} F{self.f_travel:.1f}")
        lines.append("G1 X0.0000 Y200.0000 F2540.0 ; Present job")
        lines.append("; ========================================================")

        return "\n".join(lines)
