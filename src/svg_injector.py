import base64
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)

SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
XLINK_NS = "http://www.w3.org/1999/xlink"
SODIPODI_NS = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"

ET.register_namespace('', SVG_NS)
ET.register_namespace('xlink', XLINK_NS)
ET.register_namespace('inkscape', INKSCAPE_NS)
ET.register_namespace('sodipodi', SODIPODI_NS)
ET.register_namespace('dc', "http://purl.org/dc/elements/1.1/")
ET.register_namespace('cc', "http://creativecommons.org/ns#")
ET.register_namespace('rdf', "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
ET.register_namespace('osb', "http://www.openswatchbook.org/uri/2009/osb")

def sanitize_svg_namespaces_and_hrefs(svg_path: str) -> None:
    """
    Sanitize SVG XML to ensure clean browser rendering:
      - Normalizes any ns*:href to standard xlink:href
      - Strips duplicate href attributes
      - Adds standard HTML5 href alongside xlink:href
      - Ensures xmlns:xlink is declared on the root <svg> element
    """
    try:
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_text = f.read()

        # 1. Normalize ns*:href to xlink:href
        svg_text = re.sub(r'\bns\d+:href="([^"]+)"', r'xlink:href="\1"', svg_text)
        # 2. Strip any existing standalone href="..." to prevent duplicates
        svg_text = re.sub(r'\s+href="[^"]*"', '', svg_text)
        # 3. Add href="..." alongside xlink:href="..."
        svg_text = re.sub(r'xlink:href="([^"]+)"', r'xlink:href="\1" href="\1"', svg_text)
        # 4. Ensure xmlns:xlink is present on <svg>
        if 'xlink:href=' in svg_text and 'xmlns:xlink=' not in svg_text:
            svg_text = re.sub(r'(<svg\b[^>]*)', r'\1 xmlns:xlink="http://www.w3.org/1999/xlink"', svg_text, count=1)

        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_text)
    except Exception as e:
        logger.warning("Error sanitizing SVG %s: %s", svg_path, e)


def check_svg_has_fiducials(svg_input: Any) -> bool:
    """
    Check if an SVG file or ElementTree root already contains ArUco or 10x10mm corner fiducials.
    """
    if isinstance(svg_input, str):
        if not os.path.isfile(svg_input):
            return False
        try:
            tree = ET.parse(svg_input)
            root = tree.getroot()
        except Exception:
            return False
    elif isinstance(svg_input, ET.Element):
        root = svg_input
    else:
        return False

    for elem in root.iter():
        elem_id = str(elem.attrib.get('id', '')).lower()
        tag = elem.tag.split('}')[-1].lower()
        if elem_id == 'aruco_fiducials' or 'aruco_marker' in elem_id:
            return True
        if tag == 'image':
            try:
                w = float(elem.attrib.get('width', 0))
                h = float(elem.attrib.get('height', 0))
                if (abs(w - 10.0) < 1.0 and abs(h - 10.0) < 1.0) or 'aruco' in elem_id or 'fiducial' in elem_id:
                    return True
            except (ValueError, TypeError):
                pass
    return False


def check_svg_has_touch_dots(svg_input: Any) -> bool:
    """
    Check if an SVG file or ElementTree root already contains verification touch dots.
    """
    if isinstance(svg_input, str):
        if not os.path.isfile(svg_input):
            return False
        try:
            tree = ET.parse(svg_input)
            root = tree.getroot()
        except Exception:
            return False
    elif isinstance(svg_input, ET.Element):
        root = svg_input
    else:
        return False

    for elem in root.iter():
        elem_id = str(elem.attrib.get('id', '')).lower()
        if elem_id == 'verification_touch_dots' or 'touch_dot_' in elem_id or elem_id == 'paper_origin_corner':
            return True
    return False


def get_svg_dimensions(root: ET.Element) -> Tuple[float, float]:
    """Extract width and height in mm from SVG root."""
    width_mm = 210.0
    height_mm = 220.0

    viewbox = root.attrib.get('viewBox')
    if viewbox:
        parts = viewbox.replace(',', ' ').split()
        if len(parts) == 4:
            try:
                width_mm = float(parts[2])
                height_mm = float(parts[3])
                return width_mm, height_mm
            except ValueError:
                pass

    for attr, default in [('width', 210.0), ('height', 220.0)]:
        val_str = root.attrib.get(attr, '')
        val_clean = re.sub(r'[^\d.]', '', val_str)
        if val_clean:
            try:
                val = float(val_clean)
                if 'mm' in val_str:
                    val *= 1.0
                elif 'cm' in val_str:
                    val *= 10.0
                elif 'in' in val_str:
                    val *= 25.4
                elif 'pt' in val_str:
                    val *= (25.4 / 72.0)
                elif 'pc' in val_str:
                    val *= (25.4 / 6.0)
                elif 'px' in val_str:
                    val *= (25.4 / 96.0)
                elif val > 250:
                    val *= (25.4 / 96.0)

                if attr == 'width':
                    width_mm = val
                else:
                    height_mm = val
            except ValueError:
                pass

    return width_mm, height_mm


def generate_aruco_marker_b64(
    marker_id: int,
    dict_id: Optional[int] = None,
    side_px: int = 380,
    border_px: int = 40
) -> str:
    """
    Generate an ArUco marker PNG image encoded as a base64 data URI.

    Includes a 40px white quiet zone border around the marker to ensure
    sub-pixel detection and standard browser rendering compatibility.
    Bit-for-bit identical to standard Cutter fiducials (380x380 px).
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required to generate ArUco markers.")

    dict_code = dict_id if dict_id is not None else cv2.aruco.DICT_4X4_50
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_code)
    else:
        aruco_dict = cv2.aruco.Dictionary_get(dict_code)

    inner_size = max(10, side_px - 2 * border_px)
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, inner_size, borderBits=1)
    elif hasattr(cv2.aruco, "drawMarker"):
        marker_img = cv2.aruco.drawMarker(aruco_dict, marker_id, inner_size)
    else:
        raise RuntimeError("OpenCV does not support ArUco marker generation (missing generateImageMarker and drawMarker).")

    full_img = cv2.copyMakeBorder(marker_img, border_px, border_px, border_px, border_px, cv2.BORDER_CONSTANT, value=255)
    success, buf = cv2.imencode('.png', full_img)
    if not success:
        raise RuntimeError(f"Failed to encode ArUco marker {marker_id} to PNG")

    b64_str = base64.b64encode(buf).decode('ascii')
    return f"data:image/png;base64,{b64_str}"


def inject_aruco_fiducials(
    svg_path: str,
    output_path: Optional[str] = None,
    marker_size_mm: float = 10.0,
    margin_x_mm: float = 10.0,
    margin_y_mm: float = 10.0,
    right_margin_offset_mm: float = 20.0,
    config_path: Optional[str] = None,
    allow_overwrite: bool = False
) -> Dict[str, Any]:
    """
    Inject 3 standard OpenCV ArUco fiducials (DICT_4X4_50, IDs 0, 1, 2) into an SVG file.

    Standard corner placement (in bed coordinates with flipped Y):
      - Marker 0 (ID 0): Top-Left SVG (Back-Left Bed) at (margin_x_mm, margin_y_mm)
      - Marker 1 (ID 1): Top-Right SVG (Back-Right Bed) at (width - marker_size - right_margin_offset_mm, margin_y_mm)
      - Marker 2 (ID 2): Bottom-Left SVG (Front-Left Bed) at (margin_x_mm, height - marker_size - margin_y_mm)

    The right margin offset (20mm + 10mm marker = 30mm margin from edge) ensures camera clearance
    so the Ender 3 toolhead doesn't hit its physical X limit (235mm) when scanning Marker 1.

    If verification touch dots exist in the SVG, they are automatically synchronized.
    Also updates config.json 'design_fiducials' to match nominal coordinates.
    """
    if not os.path.isfile(svg_path):
        return {"status": "error", "message": f"File not found: {svg_path}"}

    # Guard: Do not allow adding fiducials if already present
    if not allow_overwrite and check_svg_has_fiducials(svg_path):
        return {
            "status": "already_exists",
            "message": f"ArUco fiducials are already present in '{os.path.basename(svg_path)}'. Adding again is not permitted to prevent duplicates."
        }

    tree = ET.parse(svg_path)
    root = tree.getroot()

    width_mm, height_mm = get_svg_dimensions(root)

    # Marker Top-Left corners (10x10mm each)
    m0_x = margin_x_mm
    m0_y = margin_y_mm

    m1_x = max(margin_x_mm, width_mm - marker_size_mm - right_margin_offset_mm)
    m1_y = margin_y_mm

    m2_x = margin_x_mm
    m2_y = max(margin_y_mm, height_mm - marker_size_mm - margin_y_mm)

    # Centers in SVG coordinates
    c0_svg = (m0_x + marker_size_mm / 2.0, m0_y + marker_size_mm / 2.0)
    c1_svg = (m1_x + marker_size_mm / 2.0, m1_y + marker_size_mm / 2.0)
    c2_svg = (m2_x + marker_size_mm / 2.0, m2_y + marker_size_mm / 2.0)

    # Centers in Bed coordinates (Bed Y = height_mm - SVG Y)
    c0_bed = (c0_svg[0], height_mm - c0_svg[1])  # Back-Left
    c1_bed = (c1_svg[0], height_mm - c1_svg[1])  # Back-Right
    c2_bed = (c2_svg[0], height_mm - c2_svg[1])  # Front-Left

    # Ensure viewBox exists on root
    if 'viewBox' not in root.attrib:
        root.attrib['viewBox'] = f"0 0 {width_mm:.3f} {height_mm:.3f}"

    # Check if verification touch dots already exist in file
    has_existing_dots = any(c.attrib.get('id') == 'verification_touch_dots' for c in root)

    # 1. Remove existing aruco_fiducials group if present
    for child in list(root):
        if child.attrib.get('id') == 'aruco_fiducials':
            root.remove(child)

    # 2. Also remove any standalone fiducial image elements to avoid duplicates
    elements_to_remove = []
    for parent in root.iter():
        for child in list(parent):
            tag = child.tag.split('}')[-1].lower()
            elem_id = str(child.attrib.get('id', '')).lower()
            if tag == 'image':
                try:
                    w = float(child.attrib.get('width', 0))
                    h = float(child.attrib.get('height', 0))
                    if (abs(w - marker_size_mm) < 1.0 and abs(h - marker_size_mm) < 1.0) or 'aruco' in elem_id:
                        elements_to_remove.append((parent, child))
                except (ValueError, TypeError):
                    pass
    for parent, child in elements_to_remove:
        try:
            parent.remove(child)
        except Exception:
            pass

    # Create Group for fiducials
    group = ET.SubElement(root, f"{{{SVG_NS}}}g", {
        "id": "aruco_fiducials",
        f"{{{INKSCAPE_NS}}}label": "ArUco Fiducials",
        "style": "display:inline;"
    })

    markers_config = [
        (0, m0_x, m0_y, "aruco_marker_0"),
        (1, m1_x, m1_y, "aruco_marker_1"),
        (2, m2_x, m2_y, "aruco_marker_2")
    ]

    for mid, mx, my, elem_id in markers_config:
        b64_href = generate_aruco_marker_b64(mid)
        ET.SubElement(group, f"{{{SVG_NS}}}image", {
            "id": elem_id,
            "x": f"{mx:.3f}",
            "y": f"{my:.3f}",
            "width": f"{marker_size_mm:.3f}",
            "height": f"{marker_size_mm:.3f}",
            "preserveAspectRatio": "none",
            "style": "display:inline;image-rendering:optimizeSpeed",
            f"{{{SODIPODI_NS}}}insensitive": "true",
            f"{{{XLINK_NS}}}href": b64_href
        })

    # Save SVG
    save_path = output_path if output_path else svg_path
    tree.write(save_path, encoding="utf-8", xml_declaration=True)

    # Post-process to guarantee browser-compatible image tags (both xlink:href and HTML5 href)
    sanitize_svg_namespaces_and_hrefs(save_path)

    # If verification touch dots were already present, re-inject them to match the new fiducials
    if has_existing_dots:
        try:
            inject_verification_dots(save_path, save_path, allow_overwrite=True)
        except Exception as de:
            logger.warning("Could not re-synchronize verification dots: %s", de)

    # Update config.json design_fiducials
    cfg_file = config_path
    if not cfg_file:
        workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg_file = os.path.join(workspace_root, "config.json")
    if os.path.isfile(cfg_file):
        try:
            with open(cfg_file, "r", encoding="utf-8") as cf:
                cfg_data = json.load(cf)
            cfg_data["design_fiducials"] = [
                [round(c0_bed[0], 3), round(c0_bed[1], 3)],  # Back-Left
                [round(c1_bed[0], 3), round(c1_bed[1], 3)],  # Back-Right
                [round(c2_bed[0], 3), round(c2_bed[1], 3)]   # Front-Left
            ]
            with open(cfg_file, "w", encoding="utf-8") as cf:
                json.dump(cfg_data, cf, indent=2)
        except Exception as ce:
            logger.warning("Could not update config.json design_fiducials: %s", ce)

    return {
        "status": "ok",
        "message": f"Successfully injected 3 ArUco fiducials (IDs 0, 1, 2) into {os.path.basename(save_path)}",
        "saved_path": save_path,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "fiducials_bed": [
            {"id": 0, "name": "Back-Left", "bed_x": round(c0_bed[0], 3), "bed_y": round(c0_bed[1], 3)},
            {"id": 1, "name": "Back-Right", "bed_x": round(c1_bed[0], 3), "bed_y": round(c1_bed[1], 3)},
            {"id": 2, "name": "Front-Left", "bed_x": round(c2_bed[0], 3), "bed_y": round(c2_bed[1], 3)}
        ],
        "fiducials_svg": [
            {"id": 0, "svg_x": round(c0_svg[0], 3), "svg_y": round(c0_svg[1], 3)},
            {"id": 1, "svg_x": round(c1_svg[0], 3), "svg_y": round(c1_svg[1], 3)},
            {"id": 2, "svg_x": round(c2_svg[0], 3), "svg_y": round(c2_svg[1], 3)}
        ]
    }

def inject_verification_dots(
    svg_path: str,
    output_path: Optional[str] = None,
    allow_overwrite: bool = False
) -> Dict[str, Any]:
    """
    Inject 3 high-contrast touch verification dots and a Paper Origin (0,0) mark into an SVG.

    Dots are placed beside each fiducial at 3 distinct angles:
      - Dot 1: Beside Fiducial 1 (Top-Left) at 0° (Right: +16mm X)
      - Dot 2: Beside Fiducial 2 (Top-Right) at 270° Bed / 90° SVG (Down: +16mm Y)
      - Dot 3: Beside Fiducial 3 (Bottom-Left) at 45° (Diagonal: +14mm X, -14mm Y in SVG)
      - Paper Origin: At bottom-left corner (0, 0) in Bed space

    :param svg_path: Path to source SVG
    :param output_path: Path to save modified SVG (if None, overwrites svg_path)
    :param allow_overwrite: If True, allows updating existing verification dots
    :return: Dictionary containing status, dot coordinates in SVG and Bed space.
    """
    if not os.path.isfile(svg_path):
        return {"status": "error", "message": f"File not found: {svg_path}"}

    # Guard: Do not allow adding test dots if already present
    if not allow_overwrite and check_svg_has_touch_dots(svg_path):
        return {
            "status": "already_exists",
            "message": f"Verification test dots are already present in '{os.path.basename(svg_path)}'. Adding again is not permitted to prevent duplicates."
        }

    tree = ET.parse(svg_path)
    root = tree.getroot()

    width_mm, height_mm = get_svg_dimensions(root)

    # Find fiducials in SVG space
    fid_svg: List[Tuple[float, float]] = []
    for elem in root.iter():
        tag = elem.tag.split('}')[-1].lower()
        elem_id = str(elem.attrib.get('id', '')).lower()
        if tag == 'image':
            try:
                w = float(elem.attrib.get('width', 0))
                h = float(elem.attrib.get('height', 0))
                x = float(elem.attrib.get('x', 0))
                y = float(elem.attrib.get('y', 0))
                if (abs(w - 10.0) < 1.0 and abs(h - 10.0) < 1.0) or 'aruco' in elem_id:
                    fid_svg.append((x + w / 2.0, y + h / 2.0))
            except (ValueError, TypeError):
                continue

    # Fallback fiducials if not detected in image tags
    if len(fid_svg) < 3:
        fid_svg = [
            (15.0, 15.0),
            (width_mm - 25.0, 15.0),
            (15.0, height_mm - 15.0)
        ]
    else:
        # Sort Top-Left, Top-Right, Bottom-Left in SVG space (Y goes down)
        median_y = np.median([p[1] for p in fid_svg])
        top_pts = sorted([p for p in fid_svg if p[1] <= median_y], key=lambda p: p[0])
        bottom_pts = sorted([p for p in fid_svg if p[1] > median_y], key=lambda p: p[0])
        sorted_fid = top_pts + bottom_pts
        if len(sorted_fid) >= 3:
            fid_svg = sorted_fid[:3]

    f1 = fid_svg[0]  # Top-Left
    f2 = fid_svg[1]  # Top-Right
    f3 = fid_svg[2]  # Bottom-Left

    # Calculate 3 dots at different angles
    # Dot 1: 0° (Right) from F1
    d1_svg = (f1[0] + 16.0, f1[1])
    # Dot 2: 90° SVG / 270° Bed (Down) from F2
    d2_svg = (f2[0], f2[1] + 16.0)
    # Dot 3: 45° Diagonal Inward from F3
    d3_svg = (f3[0] + 14.0, f3[1] - 14.0)

    # Bed space coordinates (Y inverted: Bed Y = Height - SVG Y)
    d1_bed = (d1_svg[0], height_mm - d1_svg[1])
    d2_bed = (d2_svg[0], height_mm - d2_svg[1])
    d3_bed = (d3_svg[0], height_mm - d3_svg[1])
    paper_origin_bed = (0.0, 0.0)
    paper_origin_svg = (2.0, height_mm - 2.0)

    # Remove any existing verification_touch_dots group
    for child in list(root):
        if child.attrib.get('id') == 'verification_touch_dots':
            root.remove(child)

    # Create Group for verification dots
    group = ET.SubElement(root, f"{{{SVG_NS}}}g", {
        "id": "verification_touch_dots",
        f"{{{INKSCAPE_NS}}}label": "Touch Verification Dots",
        "style": "display:inline;"
    })

    dots_data = [
        (1, d1_svg, "0 deg (Right)"),
        (2, d2_svg, "270 deg (Down)"),
        (3, d3_svg, "45 deg (Diagonal)")
    ]

    for num, pt, desc in dots_data:
        cx, cy = pt
        # Outer dashed target ring (4mm diameter)
        ET.SubElement(group, f"{{{SVG_NS}}}circle", {
            "id": f"touch_ring_{num}",
            "cx": f"{cx:.3f}",
            "cy": f"{cy:.3f}",
            "r": "2.0",
            "fill": "none",
            "stroke": "#dc2626",
            "stroke-width": "0.3",
            "stroke-dasharray": "0.8,0.4"
        })
        # Inner fine ring (1.5mm diameter)
        ET.SubElement(group, f"{{{SVG_NS}}}circle", {
            "id": f"touch_inner_ring_{num}",
            "cx": f"{cx:.3f}",
            "cy": f"{cy:.3f}",
            "r": "0.75",
            "fill": "none",
            "stroke": "#2563eb",
            "stroke-width": "0.2"
        })
        # Center touch dot (0.6mm diameter)
        ET.SubElement(group, f"{{{SVG_NS}}}circle", {
            "id": f"touch_dot_{num}",
            "cx": f"{cx:.3f}",
            "cy": f"{cy:.3f}",
            "r": "0.3",
            "fill": "#dc2626"
        })
        # Crosshair lines (6mm span)
        ET.SubElement(group, f"{{{SVG_NS}}}path", {
            "id": f"touch_crosshair_{num}",
            "d": f"M {cx-3.0:.3f} {cy:.3f} L {cx+3.0:.3f} {cy:.3f} M {cx:.3f} {cy-3.0:.3f} L {cx:.3f} {cy+3.0:.3f}",
            "stroke": "#2563eb",
            "stroke-width": "0.15"
        })
        # Text label
        text_elem = ET.SubElement(group, f"{{{SVG_NS}}}text", {
            "id": f"touch_label_{num}",
            "x": f"{cx + 3.2:.3f}",
            "y": f"{cy + 0.8:.3f}",
            "font-family": "Arial, Helvetica, sans-serif",
            "font-size": "2.2",
            "font-weight": "bold",
            "fill": "#dc2626"
        })
        text_elem.text = f"DOT {num} [{desc}]"

    # Add Paper Origin (0,0) corner bracket and marker at bottom-left
    ox, oy = paper_origin_svg
    ET.SubElement(group, f"{{{SVG_NS}}}path", {
        "id": "paper_origin_corner",
        "d": f"M {ox:.3f} {oy-8.0:.3f} L {ox:.3f} {oy:.3f} L {ox+8.0:.3f} {oy:.3f}",
        "fill": "none",
        "stroke": "#059669",
        "stroke-width": "0.4"
    })
    ET.SubElement(group, f"{{{SVG_NS}}}circle", {
        "id": "paper_origin_dot",
        "cx": f"{ox:.3f}",
        "cy": f"{oy:.3f}",
        "r": "0.4",
        "fill": "#059669"
    })
    origin_text = ET.SubElement(group, f"{{{SVG_NS}}}text", {
        "id": "paper_origin_label",
        "x": f"{ox + 1.5:.3f}",
        "y": f"{oy - 2.5:.3f}",
        "font-family": "Arial, Helvetica, sans-serif",
        "font-size": "2.0",
        "font-weight": "bold",
        "fill": "#059669"
    })
    origin_text.text = "PAPER ORIGIN (0,0)"

    # Save SVG
    save_path = output_path if output_path else svg_path
    tree.write(save_path, encoding="utf-8", xml_declaration=True)

    # Post-process to guarantee browser-compatible image tags (both xlink:href and HTML5 href)
    sanitize_svg_namespaces_and_hrefs(save_path)

    return {
        "status": "ok",
        "message": f"Successfully injected 3 verification dots and paper origin mark into {os.path.basename(save_path)}",
        "saved_path": save_path,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "dots_svg": [d1_svg, d2_svg, d3_svg],
        "dots_bed": [d1_bed, d2_bed, d3_bed],
        "paper_origin_bed": paper_origin_bed,
        "paper_origin_svg": paper_origin_svg
    }


def extract_touch_dots(
    svg_path: str,
    bed_height_mm: float = 220.0,
    nominal_fiducials: Optional[List[List[float]]] = None,
) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    """
    Extract or derive the 3 verification touch dots and Paper Origin (0,0) in Bed coordinate space.

    If the SVG has injected touch dots (<circle id="touch_dot_1">), their exact positions are extracted.
    Otherwise, they are calculated relative to nominal fiducials:
      - Dot 1: Beside F1 (Back-Left) at 0° (+16mm X)
      - Dot 2: Beside F2 (Back-Right) at 270° (-16mm Y)
      - Dot 3: Beside F3 (Front-Left) at 45° (+14mm X, +14mm Y)

    :param svg_path: Path to SVG file
    :param bed_height_mm: Bed / SVG height in mm
    :param nominal_fiducials: Optional 3x2 list of fiducial positions in bed space
    :return: (List of 3 (X, Y) touch dot positions in bed space, Paper Origin (0,0) in bed space)
    """
    dots_bed: List[Tuple[float, float]] = []
    paper_origin_bed = (0.0, 0.0)

    if os.path.isfile(svg_path):
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
            width_mm, height_mm = get_svg_dimensions(root)
            dot_map = {}
            for elem in root.iter():
                elem_id = str(elem.attrib.get("id", ""))
                if elem_id.startswith("touch_dot_"):
                    try:
                        num = int(elem_id.replace("touch_dot_", ""))
                        cx = float(elem.attrib.get("cx", 0))
                        cy = float(elem.attrib.get("cy", 0))
                        dot_map[num] = (cx, height_mm - cy)
                    except (ValueError, TypeError):
                        pass

            if len(dot_map) >= 3:
                dots_bed = [dot_map[1], dot_map[2], dot_map[3]]
                return dots_bed, paper_origin_bed
        except Exception:
            pass

    # Derive from nominal fiducials
    fids = nominal_fiducials
    if (fids is None or len(fids) == 0) and os.path.isfile(svg_path):
        from src.vision_detector import extract_svg_fiducials
        extracted = extract_svg_fiducials(svg_path, bed_height_mm=bed_height_mm)
        if extracted is not None and len(extracted) >= 3:
            fids = extracted.tolist()

    if fids is None or len(fids) < 3:
        fids = [[15.0, 205.0], [185.0, 205.0], [15.0, 15.0]]

    f1, f2, f3 = fids[0], fids[1], fids[2]
    d1 = (f1[0] + 16.0, f1[1])
    d2 = (f2[0], f2[1] - 16.0)
    d3 = (f3[0] + 14.0, f3[1] + 14.0)

    return [d1, d2, d3], paper_origin_bed
