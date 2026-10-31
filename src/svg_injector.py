import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

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
                if 'in' in val_str:
                    val *= 25.4
                elif 'px' in val_str:
                    val *= (25.4 / 96.0)
                if attr == 'width':
                    width_mm = val
                else:
                    height_mm = val
            except ValueError:
                pass

    return width_mm, height_mm

def inject_verification_dots(
    svg_path: str,
    output_path: Optional[str] = None
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
    :return: Dictionary containing status, dot coordinates in SVG and Bed space.
    """
    if not os.path.isfile(svg_path):
        return {"status": "error", "message": f"File not found: {svg_path}"}

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
    try:
        with open(save_path, "r", encoding="utf-8") as f:
            svg_text = f.read()

        svg_text = re.sub(r'\b(?:ns\d+:href|xlink:href)="([^"]+)"', r'xlink:href="\1" href="\1"', svg_text)
        if 'xlink:href=' in svg_text and 'xmlns:xlink=' not in svg_text:
            svg_text = re.sub(r'(<svg\b[^>]*)', r'\1 xmlns:xlink="http://www.w3.org/1999/xlink"', svg_text, count=1)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(svg_text)
    except Exception:
        pass

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
