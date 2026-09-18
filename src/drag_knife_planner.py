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

logger = logging.getLogger(__name__)


class SVGParser:
    """Parses SVG vector paths and converts them into 2D polyline contours in mm."""

    def __init__(self, step_size_mm: float = 0.5):
        """
        Initialize SVG Parser.

        :param step_size_mm: Maximum segment length when sampling curves (mm)
        """
        self.step_size_mm = step_size_mm
        if svgpathtools is None:
            raise RuntimeError("svgpathtools is required for SVG parsing.")

    @staticmethod
    def _get_page_height_and_scale(svg_file: str) -> Tuple[float, float]:
        """
        Determine SVG page height (mm) and scale factor (25.4 / 96.0) to convert Inkscape user unit px to mm.
        """
        px_to_mm_scale = 25.4 / 96.0  # Standard 96 dpi Inkscape user unit scaling (1 px = 0.264583 mm)
        page_height_mm = 297.0  # Default A4 height (mm)

        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(svg_file)
            root = tree.getroot()

            # Check height attribute (e.g. height="297mm" or height="1122.52")
            height_str = root.get('height', '')
            if 'mm' in height_str:
                page_height_mm = float(height_str.replace('mm', '').strip())
            elif height_str:
                val = float(height_str.replace('px', '').strip())
                page_height_mm = val * px_to_mm_scale
            else:
                viewbox = root.get('viewBox')
                if viewbox:
                    parts = [float(p) for p in viewbox.replace(',', ' ').split()]
                    if len(parts) == 4:
                        page_height_mm = parts[3] * px_to_mm_scale

        except Exception as e:
            logger.warning("Could not parse SVG page height: %s. Using default 297mm.", e)

        return page_height_mm, px_to_mm_scale

    def parse_svg_paths(self, svg_file: str, flip_y: bool = True) -> List[List[Tuple[float, float]]]:
        """
        Parse vector paths in SVG file into discretized polylines (mm).

        :param svg_file: Path to SVG file
        :param flip_y: If True, flips Y axis (SVG top-left to Printer Bed bottom-left)
        :return: List of contours, where each contour is a list of (x, y) tuples
        """
        paths, attributes = svgpathtools.svg2paths(svg_file)
        page_height_mm, scale = self._get_page_height_and_scale(svg_file)

        contours: List[List[Tuple[float, float]]] = []

        for path, attr in zip(paths, attributes):
            if len(path) == 0:
                continue

            # Skip paths that belong to fiducial/aruco markers or layer 'Print'
            elem_id = str(attr.get('id', '')).lower()
            if 'fiducial' in elem_id or 'aruco' in elem_id:
                logger.info("Skipping fiducial marker path ID '%s' from cut list.", elem_id)
                continue

            contour: List[Tuple[float, float]] = []

            for segment in path:
                length_mm = segment.length() * scale
                num_samples = max(2, int(math.ceil(length_mm / self.step_size_mm)))

                for i in range(num_samples):
                    t = i / float(num_samples)
                    pt = segment.point(t)
                    x_mm = pt.real * scale
                    y_raw = pt.imag * scale
                    y_mm = (page_height_mm - y_raw) if flip_y else y_raw

                    # Avoid redundant consecutive identical points
                    if not contour or (abs(contour[-1][0] - x_mm) > 1e-4 or abs(contour[-1][1] - y_mm) > 1e-4):
                        contour.append((x_mm, y_mm))

            # Sample end point of last segment
            end_pt = path[-1].point(1.0)
            end_x = end_pt.real * scale
            end_y_raw = end_pt.imag * scale
            end_y = (page_height_mm - end_y_raw) if flip_y else end_y_raw

            if not contour or (abs(contour[-1][0] - end_x) > 1e-4 or abs(contour[-1][1] - end_y) > 1e-4):
                contour.append((end_x, end_y))

            if len(contour) >= 2:
                contours.append(contour)

        logger.info("Parsed %d path contours from SVG '%s'.", len(contours), svg_file)
        return contours


class DragKnifePlanner:
    """Generates drag-knife toolhead trajectories with swivel corner compensation."""

    def __init__(self, blade_offset_mm: float = 0.35, swivel_angle_threshold_deg: float = 30.0):
        """
        Initialize Drag Knife Planner.

        :param blade_offset_mm: Distance between toolhead pivot center and trailing blade tip (mm)
        :param swivel_angle_threshold_deg: Minimum corner direction change angle (degrees) to trigger swivel arc
        """
        self.blade_offset = blade_offset_mm
        self.swivel_threshold_rad = math.radians(swivel_angle_threshold_deg)

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

                current_u = next_u

            # Cut move along segment: pivot moves to p2 + L * current_u
            end_pivot = (p2[0] + L * current_u[0], p2[1] + L * current_u[1])
            moves.append({
                'type': 'line',
                'x': end_pivot[0],
                'y': end_pivot[1],
            })

        return moves


class GCodeGenerator:
    """Translates planned drag-knife trajectories into Klipper G-code format."""

    def __init__(
        self,
        z_safe: float = 5.0,
        z_cut: float = 0.0,
        f_cut: float = 1200.0,
        f_travel: float = 3000.0,
        dry_run: bool = False,
    ):
        """
        Initialize G-code Generator.

        :param z_safe: Safe travel Z height in mm
        :param z_cut: Cutting Z depth in mm
        :param f_cut: Cutting feedrate in mm/min
        :param f_travel: Rapid travel feedrate in mm/min
        :param dry_run: If True, sets z_cut = z_safe (no blade contact)
        """
        self.z_safe = z_safe
        self.z_cut = z_safe if dry_run else z_cut
        self.f_cut = f_cut
        self.f_travel = f_travel
        self.dry_run = dry_run

    def generate_gcode(self, planned_contours: List[List[Dict[str, Union[str, float, Tuple[float, float]]]]]) -> str:
        """
        Generate complete G-code string for all parsed and planned contours.

        :param planned_contours: List of move sequences (one sequence per path contour)
        :return: Multi-line G-code string
        """
        lines = []

        # G-code Header
        lines.append("; ========================================================")
        lines.append("; Ender 3 Print & Cut Vinyl Cutting G-code")
        lines.append(f"; Generated by vision_cutter.py (Dry Run: {self.dry_run})")
        lines.append(f"; Parameters: Z_safe={self.z_safe:.2f}mm, Z_cut={self.z_cut:.2f}mm")
        lines.append(f"; Feedrates: Cut={self.f_cut:.0f}mm/min, Travel={self.f_travel:.0f}mm/min")
        lines.append("; ========================================================")
        lines.append("G21 ; Set units to millimeters")
        lines.append("G90 ; Use absolute coordinates")
        lines.append("M83 ; Relative extruder positioning (extruder disabled)")
        lines.append(f"G0 Z{self.z_safe:.3f} F{self.f_travel:.1f} ; Raise Z to safe height")
        lines.append("")

        contour_count = 0
        for contour_moves in planned_contours:
            if not contour_moves:
                continue

            contour_count += 1
            lines.append(f"; --- Contour #{contour_count} ---")

            lead_in = contour_moves[0]
            if lead_in['type'] == 'lead_in':
                start_x, start_y = lead_in['start_pivot']
                # 1. Rapid move to initial pivot location at safe Z
                lines.append(f"G0 X{start_x:.3f} Y{start_y:.3f} F{self.f_travel:.1f}")

                # 2. Lower Z to cut depth
                lines.append(f"G1 Z{self.z_cut:.3f} F600.0 ; Lower blade to cutting depth")

            # Execute moves along contour
            for move in contour_moves[1:]:
                mtype = move['type']
                if mtype == 'line':
                    lines.append(f"G1 X{move['x']:.3f} Y{move['y']:.3f} F{self.f_cut:.1f}")
                elif mtype == 'arc':
                    g_cmd = "G2" if move['cw'] else "G3"
                    lines.append(f"{g_cmd} X{move['x']:.3f} Y{move['y']:.3f} I{move['i']:.3f} J{move['j']:.3f} F{self.f_cut:.1f} ; Swivel arc")

            # 3. Lift Z to safe height at end of contour
            lines.append(f"G0 Z{self.z_safe:.3f} F{self.f_travel:.1f} ; Lift Z")
            lines.append("")

        # G-code Footer
        lines.append("; ========================================================")
        lines.append("; End of Cut Job")
        lines.append(f"G0 Z{self.z_safe:.3f} F{self.f_travel:.1f}")
        lines.append("G0 X0.0 Y200.0 F3000.0 ; Present job")
        lines.append("; ========================================================")

        return "\n".join(lines)
