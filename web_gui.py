#!/usr/bin/env python3
"""
Web-Based Cricut-Style Web GUI Server for Ender 3 Auto Cutter Pipeline.

Runs a local Web Application Server on http://localhost:5000 using ThreadingHTTPServer.
Provides:
1. Load SVG & preview cut paths / fiducials.
2. One-click 1:1 scale printing to paper/vinyl color printer.
3. Live Camera Feed & Interactive ArUco Fiducial Scanning window.
4. Auto-Scan (ArUco) or Manual Jog Mode (Crosshair Target Alignment).
5. Hardware Calibration Settings Modal (Bed size, Blade offset, Camera offsets, Z depths, Feedrates).
6. One-click G-code upload & Vinyl Cut execution on Klipper/Moonraker.

Usage:
    python web_gui.py --port 5000
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

try:
    import cv2
    import numpy as np
except ImportError as _err:
    cv2 = None
    np = None
    print(f"[WARN] OpenCV or NumPy not installed in Python environment ({sys.executable}): {_err}")
    print("[WARN] Please ensure you run Cutter Studio with Python 3.9 where packages are installed: py -3.9 web_gui.py")

PORT = 5000
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(WORKSPACE_DIR, "config.json")
sys.path.insert(0, os.path.join(WORKSPACE_DIR, "src"))


def find_python_executable() -> str:
    """Find the verified Python executable containing cv2, numpy, and svgelements."""
    candidates = [
        r"C:\Users\chait\AppData\Local\Programs\Python\Python39\python.exe",
        sys.executable,
        shutil.which("python"),
        shutil.which("python3"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            try:
                test = subprocess.run([p, "-c", "import cv2, svgelements, numpy; print('OK')"], capture_output=True, text=True, timeout=3)
                if "OK" in test.stdout:
                    return p
            except Exception:
                pass
    return sys.executable


PYTHON_EXE = find_python_executable()
print(f"[INIT] Verified Python executable for subprocesses: {PYTHON_EXE}")

DEFAULT_CONFIG = {
    "bed_width_mm": 220.0,
    "bed_height_mm": 220.0,
    "blade_offset_mm": 0.35,
    "camera_offset_x_mm": -25.0,
    "camera_offset_y_mm": 0.0,
    "mm_per_pixel": 0.05,
    "z_safe_mm": 5.0,
    "z_cut_mm": 0.0,
    "z_lift_swivel_mm": 0.0,
    "f_cut_mm_min": 1200.0,
    "f_travel_mm_min": 3000.0,
    "moonraker_url": "http://localhost:7125",
    "crowsnest_url": "http://localhost:8080/?action=stream",
    "camera_settle_seconds": 2.0,
    "search_mode": "assisted",
}


def resolve_workspace_file(filename):
    """Locate vector or toolpath file in root workspace directory or samples/ subfolder."""
    p1 = os.path.join(WORKSPACE_DIR, filename)
    if os.path.isfile(p1):
        return p1
    p2 = os.path.join(WORKSPACE_DIR, "samples", filename)
    if os.path.isfile(p2):
        return p2
    return p1


def get_available_workspace_files() -> list:
    """Scan workspace and samples/ for SVG and G-code cut files."""
    valid_exts = (".svg", ".gcode", ".nc", ".gc", ".tap")
    file_set = set()
    for d in [WORKSPACE_DIR, os.path.join(WORKSPACE_DIR, "samples")]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(valid_exts):
                    file_set.add(f)
    return sorted(
        list(file_set),
        key=lambda x: (
            0 if "cut_size" in x.lower() else (1 if "cut.svg" in x.lower() else (2 if x.lower().endswith(".svg") else 3)),
            x.lower(),
        ),
    )


def get_file_bounding_box(filepath):
    """Calculate min_x, max_x, min_y, max_y bounding box for SVG or G-code file."""
    min_x, max_x, min_y, max_y = 0.0, 0.0, 0.0, 0.0
    if filepath.lower().endswith((".gcode", ".nc", ".gc", ".tap")):
        cur_x, cur_y = 0.0, 0.0
        found = False
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip().split(";")[0]
                if not line:
                    continue
                parts = line.split()
                cmd = parts[0].upper()
                if cmd in ("G0", "G1", "G00", "G01", "G2", "G3", "G02", "G03"):
                    for p in parts[1:]:
                        if p.upper().startswith("X"):
                            try:
                                cur_x = float(p[1:])
                                found = True
                            except ValueError:
                                pass
                        elif p.upper().startswith("Y"):
                            try:
                                cur_y = float(p[1:])
                                found = True
                            except ValueError:
                                pass
                    if found:
                        if min_x == 0.0 and max_x == 0.0:
                            min_x, max_x = cur_x, cur_x
                            min_y, max_y = cur_y, cur_y
                        else:
                            min_x = min(min_x, cur_x)
                            max_x = max(max_x, cur_x)
                            min_y = min(min_y, cur_y)
                            max_y = max(max_y, cur_y)
    else:
        try:
            sys.path.insert(0, os.path.join(WORKSPACE_DIR, "src"))
            from drag_knife_planner import SVGParser
            parser_obj = SVGParser(step_size_mm=1.0)
            contours = parser_obj.parse_svg_paths(filepath, flip_y=False)
            if contours:
                all_pts = [pt for contour in contours for pt in contour]
                min_x = min(p[0] for p in all_pts)
                max_x = max(p[0] for p in all_pts)
                min_y = min(p[1] for p in all_pts)
                max_y = max(p[1] for p in all_pts)
        except Exception as e:
            print(f"[WARN] Failed to extract SVG bounding box: {e}")

    if max_x == 0.0 and max_y == 0.0:
        min_x, max_x, min_y, max_y = 0.0, 210.0, 0.0, 297.0

    return min_x, max_x, min_y, max_y


def load_config():
    """Load configuration from config.json or return defaults."""
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                config = DEFAULT_CONFIG.copy()
                config.update(data)
                return config
        except Exception as e:
            print("[WARN] Failed to load config.json:", e)
    return DEFAULT_CONFIG.copy()


def recalculate_affine_from_settings(cfg):
    """
    If nominal_fiducials (measured camera gantry coordinates) and affine_matrix.json exist,
    recalculate the 2D rigid affine matrix M using updated camera offsets.
    This allows live recalibration of touch verification dots without re-running Step 3.
    """
    affine_path = os.path.join(WORKSPACE_DIR, "affine_matrix.json")
    if not os.path.isfile(affine_path):
        return

    gantry_fids = cfg.get("nominal_fiducials")
    if not gantry_fids or len(gantry_fids) < 3:
        return

    try:
        with open(affine_path, "r", encoding="utf-8") as f:
            af_data = json.load(f)

        design_pts = af_data.get("design_points")
        if not design_pts or len(design_pts) < 3:
            design_pts = cfg.get("design_fiducials", [[15.0, 205.0], [185.0, 205.0], [15.0, 15.0]])

        cam_off_x = float(cfg.get("camera_offset_x_mm", 0.0))
        cam_off_y = float(cfg.get("camera_offset_y_mm", 0.0))

        # Recompute measured bed points from gantry coordinates + new camera offsets
        new_bed_pts = []
        for g in gantry_fids:
            new_bed_pts.append([float(g[0]) + cam_off_x, float(g[1]) + cam_off_y])

        from src.vision_detector import TransformEstimator
        import math
        design_arr = np.array(design_pts, dtype=np.float64)
        bed_arr = np.array(new_bed_pts, dtype=np.float64)

        M = TransformEstimator.compute_affine_matrix(design_arr, bed_arr)
        a, b, tx = float(M[0, 0]), float(M[0, 1]), float(M[0, 2])
        scale = float(math.sqrt(a * a + b * b))
        angle_deg = float(math.degrees(math.atan2(b, a)))
        ty = float(M[1, 2])

        af_data["matrix"] = M.tolist()
        af_data["scale"] = scale
        af_data["rotation_deg"] = angle_deg
        af_data["translation_mm"] = [tx, ty]
        af_data["measured_bed_points"] = new_bed_pts
        af_data["camera_offset"] = [cam_off_x, cam_off_y]
        af_data["timestamp"] = time.time()

        with open(affine_path, "w", encoding="utf-8") as f:
            json.dump(af_data, f, indent=2)

        print(f"[SETTINGS] Live updated affine_matrix.json with new camera offset ({cam_off_x}, {cam_off_y}): Tx={tx:.2f}, Ty={ty:.2f}")
    except Exception as e:
        print("[SETTINGS] Could not recalculate affine matrix from settings:", e)


def save_config(new_config):
    """Save configuration dictionary to config.json."""
    config = load_config()
    config.update(new_config)
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        if "camera_offset_x_mm" in new_config or "camera_offset_y_mm" in new_config:
            recalculate_affine_from_settings(config)

        return True
    except Exception as e:
        print("[ERROR] Failed to save config.json:", e)
        return False


def render_gcode_preview_html(filepath, bed_w=210.0, bed_h=297.0):
    """Parse G-code moves and return rendered HTML bed mat preview with travel/cut paths."""
    cur_x, cur_y = 0.0, 0.0
    min_x, max_x = 0.0, 0.0
    min_y, max_y = 0.0, 0.0
    travel_moves = []
    cut_moves = []

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip().split(";")[0]
                if not line:
                    continue
                parts = line.split()
                cmd = parts[0].upper()
                if cmd in ("G0", "G1", "G00", "G01", "G2", "G3", "G02", "G03"):
                    new_x, new_y = cur_x, cur_y
                    is_move = False
                    for p in parts[1:]:
                        if p.upper().startswith("X"):
                            try:
                                new_x = float(p[1:])
                                is_move = True
                            except ValueError:
                                pass
                        elif p.upper().startswith("Y"):
                            try:
                                new_y = float(p[1:])
                                is_move = True
                            except ValueError:
                                pass
                    if is_move:
                        if cmd in ("G0", "G00"):
                            travel_moves.append((cur_x, cur_y, new_x, new_y))
                        else:
                            cut_moves.append((cur_x, cur_y, new_x, new_y))
                        cur_x, cur_y = new_x, new_y
                        min_x = min(min_x, cur_x)
                        max_x = max(max_x, cur_x)
                        min_y = min(min_y, cur_y)
                        max_y = max(max_y, cur_y)
    except Exception as e:
        print(f"[WARN] G-code preview parse error: {e}")

    gc_w = max(max_x, 10.0)
    gc_h = max(max_y, 10.0)

    total_w = max(bed_w, gc_w)
    total_h = max(bed_h, gc_h)

    bed_w_pct = (bed_w / total_w) * 100.0
    bed_h_pct = (bed_h / total_h) * 100.0
    sheet_w_pct = (gc_w / total_w) * 100.0
    sheet_h_pct = (gc_h / total_h) * 100.0

    exceeds_w = gc_w > bed_w
    exceeds_h = gc_h > bed_h
    exceeds = exceeds_w or exceeds_h
    diff_h = max(0.0, gc_h - bed_h)
    diff_w = max(0.0, gc_w - bed_w)

    travel_d = " ".join(f"M {x1:.2f},{y1:.2f} L {x2:.2f},{y2:.2f}" for x1, y1, x2, y2 in travel_moves)
    cut_d = " ".join(f"M {x1:.2f},{y1:.2f} L {x2:.2f},{y2:.2f}" for x1, y1, x2, y2 in cut_moves)

    svg_content = f'''<svg viewBox="0 0 {gc_w:.2f} {gc_h:.2f}" style="width:100%; height:100%; max-width:100%; max-height:100%;">
      <g transform="scale(1, -1) translate(0, -{gc_h:.2f})">
        <path d="{travel_d}" stroke="#38bdf8" stroke-width="0.5" stroke-dasharray="2,2" fill="none" opacity="0.6"/>
        <path d="{cut_d}" stroke="#10b981" stroke-width="1.2" fill="none"/>
      </g>
    </svg>'''

    badge_exceeds = ""
    if exceeds:
        ex_msg = []
        if exceeds_w:
            ex_msg.append(f"+{diff_w:.0f}mm X")
        if exceeds_h:
            ex_msg.append(f"+{diff_h:.0f}mm Y")
        badge_exceeds = f"<span style='padding:2px 8px; border-radius:4px; background:rgba(239, 68, 68, 0.25); color:#f87171; border:1px solid rgba(239, 68, 68, 0.4); font-size:10px; font-weight:bold;'>⚠️ Exceeds Bed ({', '.join(ex_msg)})</span>"

    preview_html = f'''<div style="width:100%; height:100%; display:flex; flex-direction:column; justify-content:center; align-items:center; padding:4px; background:#090d16; user-select:none;">
      <div style="display:flex; align-items:center; gap:8px; font-size:11px; font-family:monospace; margin-bottom:6px; z-index:10;">
        <span style="padding:2px 8px; border-radius:4px; background:rgba(56, 189, 248, 0.15); color:#38bdf8; border:1px solid rgba(56, 189, 248, 0.3); font-weight:600;">
          🛏️ Bed: {bed_w:.0f}×{bed_h:.0f} mm
        </span>
        <span style="padding:2px 8px; border-radius:4px; background:rgba(16, 185, 129, 0.15); color:#34d399; border:1px solid rgba(16, 185, 129, 0.3); font-weight:600;">
          ⚙️ PolyCut G-code: {gc_w:.0f}×{gc_h:.0f} mm
        </span>
        {badge_exceeds}
      </div>

      <div style="aspect-ratio:{total_w}/{total_h}; width:100%; max-width:92%; max-height:82%; background:#050811; border:1px solid #1e293b; border-radius:8px; box-shadow:0 8px 24px rgba(0,0,0,0.7); position:relative; overflow:hidden; display:flex; justify-content:flex-start; align-items:flex-start; padding:0;">
        <div style="position:absolute; top:0; left:0; width:{bed_w_pct:.2f}%; height:{bed_h_pct:.2f}%; background:#111827; border:2px solid #3b82f6; border-radius:4px; box-sizing:border-box; z-index:1; overflow:hidden;">
          <div style="position:absolute; inset:0; background-image:linear-gradient(to right, rgba(59, 130, 246, 0.15) 1px, transparent 1px), linear-gradient(to bottom, rgba(59, 130, 246, 0.15) 1px, transparent 1px); background-size: 20px 20px; pointer-events:none;"></div>
          <div style="position:absolute; bottom:4px; left:6px; font-size:9px; font-family:monospace; color:#60a5fa; background:rgba(15,23,42,0.85); padding:1px 4px; border-radius:2px; pointer-events:none; z-index:5;">(0,0) Bed Origin</div>
        </div>

        <div style="position:absolute; top:0; left:0; width:{sheet_w_pct:.2f}%; height:{sheet_h_pct:.2f}%; background:#0b132b; box-shadow:0 4px 16px rgba(0,0,0,0.6); z-index:2; display:flex; justify-content:center; align-items:center; border:1px solid #334155; overflow:hidden;">
          {svg_content}
          {"<div style='position:absolute; top:" + f"{(bed_h_pct / sheet_h_pct * 100.0):.2f}" + "%; bottom:0; left:0; right:0; background:rgba(239, 68, 68, 0.35); border-top:2px dashed #ef4444; backdrop-filter:blur(1px); display:flex; align-items:center; justify-content:center; color:#ffffff; font-size:10px; font-family:monospace; font-weight:bold; text-shadow:0 1px 3px rgba(0,0,0,0.8); z-index:10; pointer-events:none;'>⛔ OUTSIDE PRINTABLE BED (+" + f"{diff_h:.0f}mm EXCEEDED)</div>" if exceeds_h else ""}
          {"<div style='position:absolute; left:" + f"{(bed_w_pct / sheet_w_pct * 100.0):.2f}" + "%; right:0; top:0; bottom:0; background:rgba(239, 68, 68, 0.35); border-left:2px dashed #ef4444; backdrop-filter:blur(1px); display:flex; align-items:center; justify-content:center; color:#ffffff; font-size:10px; font-family:monospace; font-weight:bold; text-shadow:0 1px 3px rgba(0,0,0,0.8); z-index:10; pointer-events:none;'>⛔ OUTSIDE BED</div>" if (exceeds_w and not exceeds_h) else ""}
        </div>
      </div>
    </div>'''

    return len(cut_moves), preview_html


def generate_bed_test_pattern(grid_count=16, box_size=20.0, x_size=5.0, margin=15.0):
    """
    Generate an SVG test pattern for Bed Mesh / Z-height calibration.
    Creates a grid of square check-boxes with a small 5x5mm 'X' mark at the center of each box.
    """
    cfg = load_config()
    bed_w = float(cfg.get("bed_width_mm", 220.0))
    bed_h = float(cfg.get("bed_height_mm", 220.0))

    cols = max(1, int(math.sqrt(grid_count)))
    rows = cols
    if cols * rows < grid_count:
        rows += 1

    usable_w = max(10.0, bed_w - (2.0 * margin))
    usable_h = max(10.0, bed_h - (2.0 * margin))

    dx = usable_w / (cols - 1) if cols > 1 else 0.0
    dy = usable_h / (rows - 1) if rows > 1 else 0.0

    path_elements = []

    for r in range(rows):
        y_center = margin + (r * dy) if rows > 1 else (bed_h / 2.0)
        for c in range(cols):
            x_center = margin + (c * dx) if cols > 1 else (bed_w / 2.0)

            # Outer box
            x_min = x_center - (box_size / 2.0)
            x_max = x_center + (box_size / 2.0)
            y_min = y_center - (box_size / 2.0)
            y_max = y_center + (box_size / 2.0)

            box_d = f"M {x_min:.2f},{y_min:.2f} L {x_max:.2f},{y_min:.2f} L {x_max:.2f},{y_max:.2f} L {x_min:.2f},{y_max:.2f} Z"
            path_elements.append(f'    <path d="{box_d}" stroke="#10b981" stroke-width="1" fill="none"/>')

            # Center X mark
            hx = x_size / 2.0
            x_d = f"M {x_center - hx:.2f},{y_center - hx:.2f} L {x_center + hx:.2f},{y_center + hx:.2f} M {x_center - hx:.2f},{y_center + hx:.2f} L {x_center + hx:.2f},{y_center - hx:.2f}"
            path_elements.append(f'    <path d="{x_d}" stroke="#f43f5e" stroke-width="1.2" fill="none"/>')

    svg_content = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{bed_w:.1f}mm" height="{bed_h:.1f}mm" viewBox="0 0 {bed_w:.1f} {bed_h:.1f}">
  <g id="bed_test_grid">
{os.linesep.join(path_elements)}
  </g>
</svg>'''

    svg_filename = f"bed_mesh_test_{grid_count}.svg"
    samples_dir = os.path.join(WORKSPACE_DIR, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    svg_filepath = os.path.join(samples_dir, svg_filename)
    with open(svg_filepath, "w", encoding="utf-8") as f:
        f.write(svg_content)

    return svg_filename


class CameraStreamer:
    """Helper class to enumerate, capture, annotate, and stream camera feeds."""

    def __init__(self):
        self.detector = None
        self._frame_count = 0
        self._last_dets = {}
        if cv2 is not None:
            try:
                from vision_detector import FiducialDetector
                self.detector = FiducialDetector(fiducial_type="aruco")
            except Exception as e:
                print("[WARN] Could not initialize FiducialDetector for camera streamer:", e)

    def enumerate_cameras(self):
        """Enumerate local USB webcams and network camera streams without blocking."""
        cameras = []
        cfg = load_config()
        crowsnest_url = cfg.get("crowsnest_url", "http://localhost:8080/?action=stream")
        cameras.append({
            "id": "crowsnest",
            "name": f"Klipper Crowsnest Stream ({crowsnest_url})",
            "type": "mjpeg",
            "url": crowsnest_url
        })
        cameras.append({
            "id": "0",
            "name": "USB Camera 0 (Local)",
            "type": "opencv",
            "index": 0
        })
        cameras.append({
            "id": "mock",
            "name": "Mock Virtual Camera (Test Feed)",
            "type": "mock"
        })
        return cameras

    def fetch_crowsnest_frame(self):
        """Fetch live frame from Klipper Crowsnest snapshot endpoint or fast MJPEG delimiter extraction."""
        cfg = load_config()
        crowsnest_url = cfg.get("crowsnest_url", "http://localhost:8080/?action=stream")
        urls_to_try = []
        if "action=stream" in crowsnest_url:
            urls_to_try.append(crowsnest_url.replace("action=stream", "action=snapshot"))
            urls_to_try.append(crowsnest_url)
        elif "action=snapshot" in crowsnest_url:
            urls_to_try.append(crowsnest_url)
            urls_to_try.append(crowsnest_url.replace("action=snapshot", "action=stream"))
        else:
            urls_to_try.append(crowsnest_url)

        for url in urls_to_try:
            try:
                if "action=stream" in url:
                    req = urllib.request.Request(url, headers={"User-Agent": "CutterStudio/1.0"})
                    with urllib.request.urlopen(req, timeout=1.5) as stream_resp:
                        buf = b""
                        for _ in range(35):
                            chunk = stream_resp.read(2048)
                            if not chunk:
                                break
                            buf += chunk

                        starts = [i for i in range(len(buf) - 1) if buf[i:i+2] == b"\xff\xd8"]
                        ends = [i for i in range(len(buf) - 1) if buf[i:i+2] == b"\xff\xd9"]

                        for s in reversed(starts):
                            valid_ends = [e for e in ends if e > s]
                            if valid_ends:
                                e = valid_ends[0]
                                jpg = buf[s : e + 2]
                                if cv2 is not None and np is not None:
                                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                                    if frame is not None and frame.size > 0:
                                        return frame
                else:
                    req = urllib.request.Request(url, headers={"User-Agent": "CutterStudio/1.0"})
                    with urllib.request.urlopen(req, timeout=1.0) as resp:
                        data = resp.read()
                        if data and len(data) > 0 and cv2 is not None and np is not None:
                            img_array = np.frombuffer(data, dtype=np.uint8)
                            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            if frame is not None and frame.size > 0:
                                return frame
            except Exception:
                pass
        return None

    def generate_mock_frame(self, frame_num: int):
        """Generate a 640x480 synthetic camera frame with grid, reticle, and ArUco markers."""
        if cv2 is None or np is None:
            return None

        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (20, 24, 33)  # Dark slate background

        # Draw grid
        for x in range(0, w, 40):
            cv2.line(frame, (x, 0), (x, h), (35, 42, 56), 1)
        for y in range(0, h, 40):
            cv2.line(frame, (0, y), (w, y), (35, 42, 56), 1)

        # Draw optical center target reticle
        cu, cv = w // 2, h // 2
        cv2.line(frame, (cu - 25, cv), (cu + 25, cv), (0, 255, 255), 1)
        cv2.line(frame, (cu, cv - 25), (cu, cv + 25), (0, 255, 255), 1)
        cv2.circle(frame, (cu, cv), 6, (0, 255, 255), 1)
        cv2.circle(frame, (cu, cv), 2, (0, 0, 255), -1)

        # Simulate small camera pan/jitter
        dx = int(5 * math.sin(frame_num * 0.1))
        dy = int(3 * math.cos(frame_num * 0.1))

        # Simulated ArUco marker #0 (Top-Left)
        m0_u, m0_v = 150 + dx, 120 + dy
        cv2.rectangle(frame, (m0_u - 30, m0_v - 30), (m0_u + 30, m0_v + 30), (0, 255, 128), 2)
        cv2.putText(frame, "ArUco ID: 0", (m0_u - 40, m0_v - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1)
        cv2.circle(frame, (m0_u, m0_v), 3, (0, 0, 255), -1)

        # Simulated ArUco marker #1 (Top-Right)
        m1_u, m1_v = 480 + dx, 120 + dy
        cv2.rectangle(frame, (m1_u - 30, m1_v - 30), (m1_u + 30, m1_v + 30), (0, 255, 128), 2)
        cv2.putText(frame, "ArUco ID: 1", (m1_u - 40, m1_v - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1)
        cv2.circle(frame, (m1_u, m1_v), 3, (0, 0, 255), -1)

        # Status text overlay
        cv2.putText(frame, "SIMULATED CAMERA FEED (CROSSHAIR TARGET ON)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)
        cv2.putText(frame, f"Frame: #{frame_num} | Focus Score: 142.5 VAR", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return frame

    def annotate_frame(self, frame):
        """Annotate camera frame with reticle, focus score, and ArUco marker bounding boxes."""
        if cv2 is None or frame is None or frame.size == 0:
            return frame

        h, w = frame.shape[:2]
        cu, cv = w // 2, h // 2

        # Optical center target reticle (High Contrast Yellow/Red)
        cv2.line(frame, (cu - 25, cv), (cu + 25, cv), (0, 255, 255), 1)
        cv2.line(frame, (cu, cv - 25), (cu, cv + 25), (0, 255, 255), 1)
        cv2.circle(frame, (cu, cv), 6, (0, 255, 255), 1)
        cv2.circle(frame, (cu, cv), 2, (0, 0, 255), -1)

        # Focus score
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        focus_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        self._frame_count += 1
        if self.detector is not None:
            try:
                # Detect ArUco markers every 4th frame without warning spam to keep stream buttery smooth
                if self._frame_count % 4 == 0 or not self._last_dets:
                    self._last_dets = self.detector.detect_aruco_markers(frame, log_warnings=False)
                dets = self._last_dets
                for marker_id, (u, v) in dets.items():
                    u_int, v_int = int(u), int(v)
                    cv2.circle(frame, (u_int, v_int), 5, (0, 0, 255), -1)
                    cv2.rectangle(frame, (u_int - 25, v_int - 25), (u_int + 25, v_int + 25), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"ArUco ID:{marker_id} ({u:.1f}, {v:.1f})",
                        (u_int - 45, v_int - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1,
                    )
            except Exception:
                pass

        cv2.putText(frame, f"LIVE CAMERA FEED | Focus Score: {focus_score:.1f}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame


camera_streamer = CameraStreamer()

# Shared persistent Moonraker client so we reuse TCP connections (HTTP keep-alive)
# instead of creating a new requests.Session / TCP connection on every button press.
_moonraker_client = None
_moonraker_client_url = None


def get_moonraker_client():
    """Return a shared MoonrakerClient instance, recreating it only if the URL changed."""
    global _moonraker_client, _moonraker_client_url
    cfg = load_config()
    moon_url = cfg.get("moonraker_url", "http://localhost:7125")
    if _moonraker_client is None or _moonraker_client_url != moon_url:
        sys.path.insert(0, os.path.join(WORKSPACE_DIR, "src"))
        from moonraker_client import MoonrakerClient
        _moonraker_client = MoonrakerClient(base_url=moon_url, timeout=5.0)
        _moonraker_client_url = moon_url
    return _moonraker_client

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Ender 3 Auto-Cutter Studio</title>
  <script>
    // Bulletproof Early Error Trap: Captures ANY unhandled JS crash or promise rejection
    window.__earlyErrors = [];
    window.onerror = function(msg, url, line, col, error) {
      if (String(msg).indexOf('ResizeObserver') !== -1) return false;
      var errText = '[CRASH] ' + msg + (line ? ' (line ' + line + ')' : '');
      window.__earlyErrors.push(errText);
      var box = document.getElementById('log-box');
      if (box) {
        var d = document.createElement('div');
        d.style.color = '#f87171';
        d.style.fontWeight = 'bold';
        d.textContent = '[' + new Date().toLocaleTimeString() + '] ❌ ' + errText;
        box.appendChild(d);
        box.scrollTop = box.scrollHeight;
      }
      return false;
    };
    window.addEventListener('unhandledrejection', function(event) {
      var reason = event.reason;
      var msg = reason && (reason.message || reason.stack) ? (reason.message || reason) : reason;
      var errText = '[ASYNC ERROR] ' + msg;
      window.__earlyErrors.push(errText);
      var box = document.getElementById('log-box');
      if (box) {
        var d = document.createElement('div');
        d.style.color = '#fbbf24';
        d.style.fontWeight = 'bold';
        d.textContent = '[' + new Date().toLocaleTimeString() + '] ⚠️ ' + errText;
        box.appendChild(d);
        box.scrollTop = box.scrollHeight;
      }
    });
  </script>
  <!-- Load Tailwind asynchronously so it NEVER blocks page rendering if offline -->
  <script src="https://cdn.tailwindcss.com" defer></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background-color: #0b0f19; color: #f1f5f9; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
    .card { background-color: #111827; border: 1px solid #1f2937; border-radius: 0.75rem; padding: 1rem; }
    .btn-primary { background-color: #10b981; color: white; border-radius: 0.5rem; font-weight: 600; }
    .btn-primary:hover { background-color: #059669; }
    #svg-container svg {
      max-width: 100%;
      max-height: 100%;
      width: auto !important;
      height: auto !important;
      display: block;
      margin: auto;
    }
    /* Complete Standalone Offline CSS Fallbacks (Zero Internet Required) */
    .flex { display: flex; }
    .inline-flex { display: inline-flex; }
    .flex-col { flex-direction: column; }
    .items-center { align-items: center; }
    .items-start { align-items: flex-start; }
    .justify-between { justify-content: space-between; }
    .justify-center { justify-content: center; }
    .justify-end { justify-content: flex-end; }
    .space-x-1 > * + * { margin-left: 0.25rem; }
    .space-x-2 > * + * { margin-left: 0.5rem; }
    .space-x-3 > * + * { margin-left: 0.75rem; }
    .space-x-4 > * + * { margin-left: 1rem; }
    .space-y-1 > * + * { margin-top: 0.25rem; }
    .space-y-2 > * + * { margin-top: 0.5rem; }
    .space-y-3 > * + * { margin-top: 0.75rem; }
    .space-y-4 > * + * { margin-top: 1rem; }
    .space-y-6 > * + * { margin-top: 1.5rem; }
    .grid { display: grid; }
    .grid-cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .grid-cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .gap-1 { gap: 0.25rem; }
    .gap-2 { gap: 0.5rem; }
    .gap-3 { gap: 0.75rem; }
    .gap-4 { gap: 1rem; }
    .gap-6 { gap: 1.5rem; }
    .w-full { width: 100%; }
    .max-w-7xl { max-width: 80rem; margin-left: auto; margin-right: auto; }
    .max-w-2xl { max-width: 42rem; }
    .h-full { height: 100%; }
    .h-\[450px\] { height: 450px; }
    .max-h-\[90vh\] { max-height: 90vh; }
    .rounded { border-radius: 0.375rem; }
    .rounded-md { border-radius: 0.375rem; }
    .rounded-lg { border-radius: 0.5rem; }
    .rounded-xl { border-radius: 0.75rem; }
    .rounded-full { border-radius: 9999px; }
    .p-1 { padding: 0.25rem; }
    .p-1\.5 { padding: 0.375rem; }
    .p-2 { padding: 0.5rem; }
    .p-3 { padding: 0.75rem; }
    .p-4 { padding: 1rem; }
    .p-6 { padding: 1.5rem; }
    .px-1\.5 { padding-left: 0.375rem; padding-right: 0.375rem; }
    .px-2 { padding-left: 0.5rem; padding-right: 0.5rem; }
    .px-3 { padding-left: 0.75rem; padding-right: 0.75rem; }
    .px-4 { padding-left: 1rem; padding-right: 1rem; }
    .py-0\.5 { padding-top: 0.125rem; padding-bottom: 0.125rem; }
    .py-1 { padding-top: 0.25rem; padding-bottom: 0.25rem; }
    .py-1\.5 { padding-top: 0.375rem; padding-bottom: 0.375rem; }
    .py-2 { padding-top: 0.5rem; padding-bottom: 0.5rem; }
    .py-3 { padding-top: 0.75rem; padding-bottom: 0.75rem; }
    .pb-3 { padding-bottom: 0.75rem; }
    .pb-4 { padding-bottom: 1rem; }
    .pt-2 { padding-top: 0.5rem; }
    .mb-1 { margin-bottom: 0.25rem; }
    .mb-2 { margin-bottom: 0.5rem; }
    .mb-3 { margin-bottom: 0.75rem; }
    .mr-1 { margin-right: 0.25rem; }
    .mr-2 { margin-right: 0.5rem; }
    .bg-black { background-color: #000000; }
    .bg-slate-700 { background-color: #334155; }
    .bg-slate-800 { background-color: #1e293b; }
    .bg-slate-900 { background-color: #0f172a; }
    .bg-slate-950 { background-color: #020617; }
    .bg-emerald-500 { background-color: #10b981; }
    .bg-emerald-600 { background-color: #059669; }
    .bg-emerald-700 { background-color: #047857; }
    .bg-emerald-900 { background-color: #064e3b; }
    .bg-purple-600 { background-color: #9333ea; }
    .bg-purple-700 { background-color: #7e22ce; }
    .bg-purple-900 { background-color: #581c87; }
    .bg-rose-500 { background-color: #f43f5e; }
    .bg-rose-600 { background-color: #e11d48; }
    .bg-blue-600 { background-color: #2563eb; }
    .bg-blue-950\/60 { background-color: rgba(23, 37, 84, 0.6); }
    .bg-amber-600 { background-color: #d97706; }
    .bg-amber-900\/60 { background-color: rgba(120, 53, 15, 0.6); }
    .bg-emerald-500\/20 { background-color: rgba(16, 185, 129, 0.2); }
    .bg-rose-500\/20 { background-color: rgba(244, 63, 94, 0.2); }
    .bg-purple-500\/20 { background-color: rgba(168, 85, 247, 0.2); }
    .bg-slate-950\/80 { background-color: rgba(2, 6, 23, 0.8); }
    .text-white { color: #ffffff; }
    .text-black { color: #000000; }
    .text-slate-200 { color: #e2e8f0; }
    .text-slate-300 { color: #cbd5e1; }
    .text-slate-400 { color: #94a3b8; }
    .text-slate-500 { color: #64748b; }
    .text-emerald-200 { color: #a7f3d0; }
    .text-emerald-300 { color: #6ee7b7; }
    .text-emerald-400 { color: #34d399; }
    .text-purple-200 { color: #e9d5ff; }
    .text-purple-400 { color: #c084fc; }
    .text-amber-300 { color: #fcd34d; }
    .text-amber-400 { color: #fbbf24; }
    .text-rose-400 { color: #fb7185; }
    .text-blue-400 { color: #60a5fa; }
    .border { border-width: 1px; border-style: solid; }
    .border-b { border-bottom-width: 1px; border-bottom-style: solid; }
    .border-slate-700 { border-color: #334155; }
    .border-slate-800 { border-color: #1e293b; }
    .border-emerald-500\/30 { border-color: rgba(16, 185, 129, 0.3); }
    .border-rose-500\/30 { border-color: rgba(244, 63, 94, 0.3); }
    .border-purple-500\/40 { border-color: rgba(168, 85, 247, 0.4); }
    .border-blue-800\/80 { border-color: rgba(30, 64, 175, 0.8); }
    .border-amber-600\/50 { border-color: rgba(217, 119, 6, 0.5); }
    .font-mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    .font-bold { font-weight: 700; }
    .font-semibold { font-weight: 600; }
    .text-\[10px\] { font-size: 10px; line-height: 14px; }
    .text-\[11px\] { font-size: 11px; line-height: 15px; }
    .text-xs { font-size: 0.75rem; line-height: 1rem; }
    .text-sm { font-size: 0.875rem; line-height: 1.25rem; }
    .text-lg { font-size: 1.125rem; line-height: 1.75rem; }
    .text-xl { font-size: 1.25rem; line-height: 1.75rem; }
    .text-2xl { font-size: 1.5rem; line-height: 2rem; }
    .text-center { text-align: center; }
    .overflow-hidden { overflow: hidden; }
    .overflow-y-auto { overflow-y: auto; }
    .relative { position: relative; }
    .fixed { position: fixed; }
    .inset-0 { top: 0; right: 0; bottom: 0; left: 0; }
    .z-50 { z-index: 50; }
    .hidden { display: none !important; }
    .cursor-pointer { cursor: pointer; }
    .cursor-not-allowed { cursor: not-allowed; }
    .shadow { box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06); }
    .shadow-md { box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); }
    .shadow-lg { box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); }
    .shadow-2xl { box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25); }
    .object-contain { object-fit: contain; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .4; } }
    .animate-pulse { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
    select, input, button { font-family: inherit; font-size: inherit; }
    select:focus, input:focus, button:focus { outline: none; border-color: #10b981; }
  </style>
</head>
<body class="p-6">
  <div class="max-w-7xl mx-auto space-y-6">
    <!-- Header -->
    <div class="flex justify-between items-center pb-4 border-b border-slate-700">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 bg-emerald-500/20 text-emerald-400 flex items-center justify-center rounded-xl text-xl font-bold">✂️</div>
        <div>
          <h1 class="text-2xl font-bold">Ender 3 Cutter Studio</h1>
          <p class="text-xs text-slate-400">Autonomous Computer-Vision Print & Cut Web Application</p>
        </div>
      </div>
      <div class="flex items-center space-x-3">
        <span id="klipper-status-badge" onclick="checkConnection()" title="Click to check server connection" class="cursor-pointer px-3 py-1 rounded-full text-xs bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">● Cutter Studio Online (Port 5000)</span>
        <button onclick="openBedTestModal()" class="px-3 py-1.5 rounded-lg text-xs bg-amber-900/60 hover:bg-amber-800 text-amber-300 border border-amber-600/50 font-semibold flex items-center space-x-1.5 shadow transition-all hover:border-amber-400">
          <span>🏁 Bed Mesh Test</span>
        </button>
        <button onclick="openSettingsModal()" class="px-3 py-1.5 rounded-lg text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-semibold flex items-center space-x-1.5 shadow transition-all hover:border-slate-500">
          <span>⚙️ Hardware Settings</span>
        </button>
      </div>
    </div>

    <!-- Workflow Steps Grid -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
      <!-- Step 1 -->
      <div class="card p-4 space-y-3">
        <div class="flex items-center space-x-2">
          <span class="w-6 h-6 rounded-full bg-emerald-500 text-white flex items-center justify-center text-xs font-bold">1</span>
          <h3 class="font-semibold text-sm">Select Cut File</h3>
        </div>
        <select id="svg-file" onchange="onFileSelectionChanged()" class="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-emerald-400 font-mono">
          <option value="Cut.svg">Cut.svg</option>
        </select>
        <div class="grid grid-cols-2 gap-2 text-[10px]">
          <button onclick="loadFileList()" class="bg-slate-800 hover:bg-slate-700 text-slate-300 py-1.5 rounded border border-slate-700 font-semibold flex items-center justify-center space-x-1 shadow">
            <span>📁 Workspace</span>
          </button>
          <button onclick="document.getElementById('file-upload-input').click()" class="bg-emerald-700 hover:bg-emerald-600 text-white py-1.5 rounded font-semibold flex items-center justify-center space-x-1 shadow">
            <span>📤 Choose File</span>
          </button>
          <input type="file" id="file-upload-input" onchange="uploadSelectedFile()" accept=".svg,.gcode,.nc,.gc,.tap" class="hidden">
        </div>
        <button id="load-preview-btn" onclick="loadPreview()" class="w-full btn-primary py-1.5 text-xs shadow">Load File Preview</button>
        <button id="btn-inject-fiducials" onclick="injectFiducialsToSvg()" class="w-full bg-emerald-700 hover:bg-emerald-600 text-white font-semibold py-1.5 rounded text-xs shadow flex items-center justify-center space-x-1">
          <span>🏁 Add ArUco Fiducials to SVG</span>
        </button>
        <button id="btn-inject-dots" onclick="injectTouchDotsToSvg()" class="w-full bg-blue-700 hover:bg-blue-600 text-white font-semibold py-1.5 rounded text-xs shadow flex items-center justify-center space-x-1">
          <span>🎯 Add 3 Test Dots to SVG</span>
        </button>
        <button onclick="downloadSvg()" class="w-full bg-slate-800 hover:bg-slate-700 text-slate-300 py-1 rounded border border-slate-700 text-[11px] font-semibold flex items-center justify-center space-x-1 shadow">
          <span>🖨️ Download / Export SVG</span>
        </button>
      </div>

      <!-- Step 2 -->
      <div class="card p-4 space-y-3">
        <div class="flex items-center space-x-2">
          <span class="w-6 h-6 rounded-full bg-blue-500 text-white flex items-center justify-center text-xs font-bold">2</span>
          <h3 class="font-semibold text-sm">Print / Prep Sheet</h3>
        </div>
        <div id="step2-svg-mode" class="space-y-2">
          <p class="text-xs text-slate-400">Send 1:1 scale design with fiducials to color printer.</p>
          <button onclick="openPrintWindow()" class="w-full bg-blue-600 hover:bg-blue-500 text-white font-semibold py-2 rounded text-xs shadow">🖨️ Print Sheet (1:1)</button>
        </div>
        <div id="step2-gcode-mode" class="space-y-2 hidden">
          <div class="bg-blue-950/60 border border-blue-800/80 p-2 rounded text-xs space-y-1">
            <div class="font-semibold text-blue-300 flex items-center space-x-1">
              <span>⚙️ PolyCut G-code Ready</span>
            </div>
            <p class="text-[10px] text-slate-400">Toolpath file selected. Proceed to camera alignment or cutting.</p>
          </div>
        </div>
      </div>

      <!-- Step 3 -->
      <div class="card p-4 space-y-3">
        <div class="flex items-center justify-between">
          <div class="flex items-center space-x-2">
            <span class="w-6 h-6 rounded-full bg-purple-500 text-white flex items-center justify-center text-xs font-bold">3</span>
            <h3 class="font-semibold text-sm">Scan & Align</h3>
          </div>
          <!-- Alignment Mode Toggle -->
          <div class="flex rounded-lg bg-slate-900 p-0.5 border border-slate-700 text-[10px]">
            <button id="mode-auto-btn" onclick="setAlignMode('auto')" class="px-2 py-0.5 rounded font-semibold bg-purple-600 text-white shadow">⚡ Auto</button>
            <button id="mode-manual-btn" onclick="setAlignMode('manual')" class="px-2 py-0.5 rounded font-semibold text-slate-400 hover:text-white">🎮 Manual</button>
          </div>
        </div>

        <div>
          <div class="flex justify-between items-center mb-1">
            <label class="text-[11px] text-slate-400">Camera Source:</label>
            <button onclick="refreshCameras()" class="text-[10px] text-purple-400 hover:underline">🔄 Refresh</button>
          </div>
          <select id="camera-select" onchange="switchCameraStream()" class="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-purple-300 font-mono">
            <option value="crowsnest" selected>Klipper Crowsnest Stream</option>
            <option value="0">USB Camera 0 (Primary)</option>
            <option value="1">USB Camera 1 (Secondary)</option>
            <option value="mock">Mock Virtual Feed</option>
          </select>
        </div>

        <!-- Auto Mode Controls -->
        <div id="auto-align-controls" class="space-y-2">
          <div>
            <label class="text-[11px] text-slate-400 block mb-1">Search Strategy & Speed Profile:</label>
            <select id="search-mode-select" class="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-purple-300 font-mono">
              <option value="assisted" selected>🎯 Assisted Align (Wait for Paper Adjust ➔ Auto 2 & 3)</option>
              <option value="auto">⚡ Auto-Cascade (Fastest ➔ Spiral ➔ Deep)</option>
              <option value="fast">🚀 Turbo / Direct (Fastest ~5s, exact hit)</option>
              <option value="spiral">🌀 Adaptive Spiral (Standard ~15s, local expansion)</option>
              <option value="exhaustive">🔍 Exhaustive Bed Sweep (Deep ~45s, arbitrary placement)</option>
            </select>
            <p class="text-[10px] text-slate-400 mt-1 leading-relaxed" id="search-mode-hint">
              <span class="text-purple-400 font-semibold">Tip:</span> Parks at Marker 1 & dwells 2s for camera stabilization. Adjust paper under camera; once Marker 1 is spotted, it auto-locks Markers 2 & 3.
            </p>
          </div>
          <button id="run-scan-btn" onclick="runScan()" class="w-full bg-purple-600 hover:bg-purple-500 text-white font-semibold py-2 rounded text-xs shadow-md">📷 Run Autonomous Scan & Align</button>
          <button id="verify-dots-btn" onclick="runTouchVerification()" class="w-full bg-purple-800 hover:bg-purple-700 text-white font-semibold py-1.5 rounded text-xs shadow flex items-center justify-center space-x-1 mt-1 border border-purple-500/40">
            <span>📍 Touch-Off Test (3 Dots & Park)</span>
          </button>
          <label class="flex items-center space-x-1.5 text-[11px] text-slate-400 cursor-pointer pt-0.5">
            <input type="checkbox" id="force-home-check" class="rounded bg-slate-800 border-slate-700 text-purple-600 focus:ring-0">
            <span>Force G28 homing (uncheck to skip if already homed)</span>
          </label>
        </div>

        <!-- Manual Mode Controls (Hidden by Default) -->
        <div id="manual-align-controls" class="space-y-2.5 hidden">
          <div class="bg-slate-900/90 p-2 rounded border border-slate-800 space-y-2">
            <div class="text-[11px] font-semibold text-purple-300 text-center">🎮 Manual D-Pad Jogging</div>
            <!-- D-Pad -->
            <div class="grid grid-cols-3 gap-1 w-32 mx-auto text-center">
              <div></div>
              <button onclick="jog('Y', 1)" class="bg-slate-800 hover:bg-slate-700 text-purple-300 py-1 rounded text-xs font-bold shadow">▲ Y+</button>
              <div></div>
              <button onclick="jog('X', -1)" class="bg-slate-800 hover:bg-slate-700 text-purple-300 py-1 rounded text-xs font-bold shadow">◀ X-</button>
              <button onclick="jogHome()" class="bg-slate-700 hover:bg-slate-600 text-amber-400 py-1 rounded text-[10px] font-bold shadow" title="Home X/Y Gantry">🏠</button>
              <button onclick="jog('X', 1)" class="bg-slate-800 hover:bg-slate-700 text-purple-300 py-1 rounded text-xs font-bold shadow">X+ ▶</button>
              <div></div>
              <button onclick="jog('Y', -1)" class="bg-slate-800 hover:bg-slate-700 text-purple-300 py-1 rounded text-xs font-bold shadow">▼ Y-</button>
              <div></div>
            </div>
            <!-- Step Size -->
            <div class="flex justify-center space-x-1 text-[10px] items-center pt-1">
              <span class="text-slate-400">Step:</span>
              <button onclick="setStep(0.1)" id="step-01" class="px-1.5 py-0.5 rounded bg-purple-600 text-white font-mono">0.1mm</button>
              <button onclick="setStep(1.0)" id="step-1" class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">1.0mm</button>
              <button onclick="setStep(10.0)" id="step-10" class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">10mm</button>
            </div>
          </div>

          <!-- 4-Corner Bed Leveling (Tramming) Helper -->
          <div class="bg-slate-900/90 p-2 rounded border border-slate-800 space-y-1">
            <div class="text-[10px] font-semibold text-amber-300 text-center">📐 Bed Tramming (4-Corner Knobs)</div>
            <div class="grid grid-cols-2 gap-1 text-[10px]">
              <button onclick="jogToCorner('FL')" class="bg-slate-800 hover:bg-amber-900/60 text-amber-300 py-1 rounded border border-slate-700 font-mono">📍 Front-Left (25, 20)</button>
              <button onclick="jogToCorner('FR')" class="bg-slate-800 hover:bg-amber-900/60 text-amber-300 py-1 rounded border border-slate-700 font-mono">📍 Front-Right (195, 20)</button>
              <button onclick="jogToCorner('RL')" class="bg-slate-800 hover:bg-amber-900/60 text-amber-300 py-1 rounded border border-slate-700 font-mono">📍 Rear-Left (25, 210)</button>
              <button onclick="jogToCorner('RR')" class="bg-slate-800 hover:bg-amber-900/60 text-amber-300 py-1 rounded border border-slate-700 font-mono">📍 Rear-Right (195, 210)</button>
            </div>
          </div>

          <!-- Lock Fiducial Marks -->
          <div class="space-y-1 text-[10px]">
            <button onclick="lockMark(0)" id="mark-0-btn" class="w-full bg-slate-800 hover:bg-slate-700 text-emerald-400 py-1 rounded border border-slate-700 font-mono">📌 Lock Mark #0 (Top-Left)</button>
            <button onclick="lockMark(1)" id="mark-1-btn" class="w-full bg-slate-800 hover:bg-slate-700 text-emerald-400 py-1 rounded border border-slate-700 font-mono">📌 Lock Mark #1 (Top-Right)</button>
            <button onclick="lockMark(2)" id="mark-2-btn" class="w-full bg-slate-800 hover:bg-slate-700 text-emerald-400 py-1 rounded border border-slate-700 font-mono">📌 Lock Mark #2 (Bottom-Left)</button>
          </div>

          <button onclick="solveManualAlignment()" class="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-1.5 rounded text-xs shadow">📐 Solve & Save 2D Affine Matrix</button>
        </div>
      </div>

      <!-- Step 4 -->
      <div class="card p-4 space-y-3">
        <div class="flex items-center space-x-2">
          <span class="w-6 h-6 rounded-full bg-rose-500 text-white flex items-center justify-center text-xs font-bold">4</span>
          <h3 class="font-semibold text-sm">Cut Vinyl</h3>
        </div>
        <div class="flex items-center justify-between text-xs">
          <span class="text-slate-400">Dry Run (Air Cut):</span>
          <input type="checkbox" id="dry-run" class="accent-emerald-500">
        </div>
        <div class="flex items-center justify-between text-xs">
          <span class="text-slate-400 font-medium">Verify Dots & Pause:</span>
          <input type="checkbox" id="verify-touch-check" class="accent-purple-500" checked title="Touch 3 test dots and pause at Paper (0,0) before cutting">
        </div>
        <div class="flex items-center justify-between text-xs">
          <span class="text-slate-400 font-medium">Cut Z Depth:</span>
          <div class="flex items-center space-x-1">
            <input type="number" id="quick-z-cut" step="0.05" onchange="syncQuickZCut(this.value)" class="w-16 bg-slate-950 border border-slate-700 rounded px-1.5 py-0.5 text-slate-200 text-right font-mono text-xs" value="0.0">
            <span class="text-slate-500 text-[10px]">mm</span>
          </div>
        </div>
        <div class="grid grid-cols-2 gap-2 text-xs">
          <button id="run-frame-btn" onclick="runBoundingBoxFrame()" class="bg-slate-800 hover:bg-slate-700 text-purple-300 py-1.5 rounded border border-purple-500/40 text-[11px] font-semibold shadow transition-all hover:border-purple-400 flex items-center justify-center space-x-1">
            <span>🔲 Frame Bounds</span>
          </button>
          <button onclick="downloadGcode()" class="bg-purple-700 hover:bg-purple-600 text-white py-1.5 rounded text-[11px] font-semibold shadow transition-all flex items-center justify-center space-x-1">
            <span>💾 Save G-code</span>
          </button>
        </div>
        <button id="run-cut-btn" onclick="runCut()" class="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-2 rounded text-xs shadow-lg">🚀 START VINYL CUT</button>
        <div class="grid grid-cols-2 gap-2 pt-1">
          <button id="resume-cut-btn" onclick="resumeCut()" class="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-1.5 rounded text-xs shadow flex items-center justify-center space-x-1" title="Resume paused print (sends RESUME_BASE to bypass cold extruder check)">
            <span>▶ RESUME CUT</span>
          </button>
          <button id="cancel-cut-btn" onclick="cancelCut()" class="w-full bg-rose-800 hover:bg-rose-700 text-white font-semibold py-1.5 rounded text-xs shadow flex items-center justify-center space-x-1" title="Cancel active print job">
            <span>⏹ CANCEL</span>
          </button>
        </div>
      </div>
    </div>

    <!-- Main Display Section (3 Columns) -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
      <!-- Col 1: SVG Preview -->
      <div class="card p-4 space-y-3">
        <h3 class="text-sm font-semibold">SVG Vector & Cut-Path Preview</h3>
        <div id="svg-container" class="w-full h-[450px] bg-slate-950 rounded border border-slate-800 flex items-center justify-center p-3 overflow-hidden">
          <p class="text-slate-500 text-xs">Click 'Load SVG Preview' to render vector cut paths</p>
        </div>
      </div>

      <!-- Col 2: Live Camera & Scanner View -->
      <div class="card p-4 space-y-3">
        <div class="flex justify-between items-center">
          <h3 class="text-sm font-semibold">Live Camera & Fiducial View</h3>
          <span id="cam-status" class="text-[10px] text-emerald-400 font-mono">● LIVE (30 FPS)</span>
        </div>
        <div id="camera-container" class="w-full h-[450px] bg-black rounded border border-slate-800 flex items-center justify-center overflow-hidden relative">
          <img id="camera-feed" src="/api/camera/stream?source=crowsnest" class="w-full h-full object-contain">
        </div>
      </div>

      <!-- Col 3: Real-time Pipeline Log -->
      <div class="card p-4 space-y-3 flex flex-col justify-between">
        <div>
          <div class="flex justify-between items-center mb-2">
            <h3 class="text-sm font-semibold">Real-time Pipeline Log</h3>
            <div class="flex space-x-2">
              <button onclick="copyLog()" class="text-[10px] text-slate-400 hover:text-white px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700" title="Copy entire log to clipboard">📋 Copy Log</button>
              <button onclick="clearLog()" class="text-[10px] text-slate-400 hover:text-white px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700" title="Clear log window">🧹 Clear</button>
            </div>
          </div>
          <div id="log-box" class="w-full h-[450px] bg-slate-950 rounded p-3 text-[11px] font-mono text-emerald-400 overflow-y-auto space-y-1 border border-slate-800">
            <div>[INIT] Cutter Studio Web Server ready on port 5000</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Alignment Touch Verification Modal -->
  <div id="touch-verify-modal" style="display: none;" class="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
    <div class="card max-w-md w-full p-6 space-y-4 border border-purple-500/50 shadow-2xl">
      <div class="flex items-center space-x-3 text-purple-400">
        <span class="text-2xl">🎯</span>
        <div>
          <h3 class="text-base font-bold text-white">Touch-Off Alignment Complete</h3>
          <p class="text-xs text-slate-400">Physical accuracy verification routine</p>
        </div>
      </div>
      <div class="bg-slate-900/80 p-3 rounded-lg border border-slate-800 text-xs space-y-2 text-slate-300">
        <p>The cutter visited the 3 test target dots on your sheet:</p>
        <ul class="list-disc list-inside space-y-1 font-mono text-[11px] text-purple-300">
          <li>Dot 1: Beside Marker 1 at 0° (Right)</li>
          <li>Dot 2: Beside Marker 2 at 270° (Down)</li>
          <li>Dot 3: Beside Marker 3 at 45° (Diagonal)</li>
        </ul>
        <p class="pt-1 text-emerald-400 font-semibold">📍 Blade is parked hovering at Paper Origin (0, 0).</p>
      </div>
      <div class="text-xs text-slate-300 font-medium">
        Did the needle touch all 3 target dots accurately on your printed sheet?
      </div>
      <div class="grid grid-cols-2 gap-2 pt-2">
        <button onclick="confirmTouchAndCut()" class="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-2 rounded text-xs shadow-lg flex items-center justify-center space-x-1">
          <span>🚀 Yes, Start Vinyl Cut</span>
        </button>
        <button onclick="adjustOffsetsFromVerify()" class="w-full bg-slate-800 hover:bg-slate-700 text-amber-300 font-semibold py-2 rounded text-xs border border-amber-500/40 flex items-center justify-center space-x-1">
          <span>⚙️ Adjust Offsets</span>
        </button>
      </div>
      <button onclick="closeTouchVerifyModal()" class="w-full text-slate-500 hover:text-slate-300 text-[11px] pt-1">Dismiss & Keep Parked</button>
    </div>
  </div>

  <!-- Settings Calibration Modal -->
  <div id="settings-modal" style="display: none;" class="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4 hidden">
    <div class="card max-w-2xl w-full p-6 space-y-6 border-slate-700 shadow-2xl max-h-[90vh] overflow-y-auto">
      <div class="flex justify-between items-center border-b border-slate-700 pb-3">
        <div class="flex items-center space-x-2">
          <span class="text-xl">⚙️</span>
          <h2 class="text-lg font-bold">Hardware Calibration & Settings</h2>
        </div>
        <button onclick="closeSettingsModal()" class="text-slate-400 hover:text-white text-lg font-bold">✕</button>
      </div>

      <!-- Form Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
        <!-- Bed Calibration -->
        <div class="space-y-3 bg-slate-900/60 p-3 rounded-lg border border-slate-800">
          <h4 class="font-semibold text-emerald-400 border-b border-slate-800 pb-1">🛏️ Printable Bed Size</h4>
          <div>
            <label class="text-slate-400 block mb-1">Bed Width (mm):</label>
            <input type="number" id="cfg-bed-width" step="1" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200">
          </div>
          <div>
            <label class="text-slate-400 block mb-1">Bed Height (mm):</label>
            <input type="number" id="cfg-bed-height" step="1" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200">
          </div>
        </div>

        <!-- Drag Knife Calibration -->
        <div class="space-y-3 bg-slate-900/60 p-3 rounded-lg border border-slate-800">
          <h4 class="font-semibold text-rose-400 border-b border-slate-800 pb-1">🔪 Drag Knife Blade</h4>
          <div>
            <label class="text-slate-400 block mb-1">Roland Blade Offset (mm):</label>
            <input type="number" id="cfg-blade-offset" step="0.01" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200">
            <span class="text-[10px] text-slate-500">Standard Roland 45° blade offset is 0.35 mm</span>
          </div>
        </div>

        <!-- Camera Calibration -->
        <div class="space-y-3 bg-slate-900/60 p-3 rounded-lg border border-slate-800">
          <h4 class="font-semibold text-purple-400 border-b border-slate-800 pb-1">📷 Camera Offsets (Rel to Knife Tip)</h4>
          <div>
            <label class="text-slate-400 block mb-1">Camera Offset X (mm):</label>
            <input type="number" id="cfg-cam-offset-x" step="0.1" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200">
          </div>
          <div>
            <label class="text-slate-400 block mb-1">Camera Offset Y (mm):</label>
            <input type="number" id="cfg-cam-offset-y" step="0.1" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200">
          </div>
          <div>
            <label class="text-slate-400 block mb-1">Camera Scale (mm / pixel):</label>
            <input type="number" id="cfg-mm-px" step="0.001" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200">
          </div>
          <div>
            <label class="text-slate-400 block mb-1">Camera Settle Dwell (seconds):</label>
            <input type="number" id="cfg-camera-settle" step="0.5" min="0.5" max="10.0" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200">
            <span class="text-[10px] text-slate-500">Wait time after gantry move to stabilize video feed (default: 2.0s)</span>
          </div>
        </div>

        <!-- Z Depths & Feedrates -->
        <div class="space-y-3 bg-slate-900/60 p-3 rounded-lg border border-slate-800">
          <h4 class="font-semibold text-blue-400 border-b border-slate-800 pb-1">⚡ Motion & Z Depths</h4>
          <div>
            <label class="text-slate-400 block mb-1">Safe Z Travel Height (mm):</label>
            <input type="number" id="cfg-z-safe" step="0.5" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200">
          </div>
          <div>
            <label class="text-slate-400 block mb-1">Cut Z Depth (mm):</label>
            <input type="number" id="cfg-z-cut" step="0.1" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200">
          </div>
          <div>
            <label class="text-slate-400 block mb-1">Lift-and-Turn Swivel Height (mm):</label>
            <input type="number" id="cfg-z-lift-swivel" step="0.1" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200">
            <span class="text-[10px] text-slate-500">0.0 = in-plane (vinyl); 0.5 = lift-and-turn (paper)</span>
          </div>
          <div>
            <label class="text-slate-400 block mb-1">Cutting Feedrate (mm/min):</label>
            <input type="number" id="cfg-f-cut" step="50" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200">
          </div>
          <div>
            <label class="text-slate-400 block mb-1">Travel Feedrate (mm/min):</label>
            <input type="number" id="cfg-f-travel" step="100" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200">
          </div>
        </div>

        <!-- Connections -->
        <div class="md:col-span-2 space-y-3 bg-slate-900/60 p-3 rounded-lg border border-slate-800">
          <h4 class="font-semibold text-amber-400 border-b border-slate-800 pb-1">🌐 Network Connections</h4>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label class="text-slate-400 block mb-1">Moonraker Server URL:</label>
              <input type="text" id="cfg-moonraker-url" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200 font-mono text-[11px]">
            </div>
            <div>
              <label class="text-slate-400 block mb-1">Crowsnest Stream URL:</label>
              <input type="text" id="cfg-crowsnest-url" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200 font-mono text-[11px]">
            </div>
          </div>
        </div>
      </div>

      <!-- Modal Footer -->
      <div class="flex justify-end space-x-3 border-t border-slate-700 pt-4">
        <button onclick="closeSettingsModal()" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-semibold">Cancel</button>
        <button onclick="saveSettings()" class="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-bold shadow-lg">💾 Save Settings & Apply</button>
      </div>
    </div>
  </div>

  <!-- Bed Mesh Test Generator Modal -->
  <div id="bed-test-modal" style="display: none;" class="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4 hidden">
    <div class="card w-full max-w-md p-6 space-y-4 shadow-2xl border border-slate-700">
      <div class="flex justify-between items-center border-b border-slate-800 pb-3">
        <h3 class="text-base font-bold text-amber-400 flex items-center space-x-2">
          <span>🏁 Bed Mesh Z-Cut Test Pattern Generator</span>
        </h3>
        <button onclick="closeBedTestModal()" class="text-slate-400 hover:text-white text-lg font-bold">✕</button>
      </div>

      <p class="text-xs text-slate-300 leading-relaxed">
        Generate a grid of check-boxes across your printable bed with a 5×5mm "X" mark in the center of each box. 
        Cut this pattern on paper or vinyl to visually verify your Klipper Bed Mesh Z compensation depth across the entire bed plate!
      </p>

      <div class="space-y-3 text-xs">
        <div>
          <label class="text-slate-400 block mb-1 font-semibold">Number of Test Boxes (Grid):</label>
          <select id="cfg-bed-grid" class="w-full bg-slate-950 border border-slate-700 rounded px-2.5 py-1.5 text-amber-300 font-semibold text-xs">
            <option value="1">1 Box (1×1 - Center Bed Check)</option>
            <option value="4">4 Boxes (2×2 - 4 Bed Corners)</option>
            <option value="16" selected>16 Boxes (4×4 - Standard Bed Mesh Test)</option>
            <option value="64">64 Boxes (8×8 - Fine Mesh Verification)</option>
          </select>
        </div>

        <div class="grid grid-cols-2 gap-3">
          <div>
            <label class="text-slate-400 block mb-1 font-semibold">Outer Box Size (mm):</label>
            <input type="number" id="cfg-bed-box-size" value="20" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200 font-mono">
          </div>
          <div>
            <label class="text-slate-400 block mb-1 font-semibold">Center "X" Size (mm):</label>
            <input type="number" id="cfg-bed-x-size" value="5" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200 font-mono">
          </div>
        </div>

        <div>
          <label class="text-slate-400 block mb-1 font-semibold">Bed Margin (mm):</label>
          <input type="number" id="cfg-bed-margin" value="15" class="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200 font-mono">
        </div>
      </div>

      <!-- Modal Footer -->
      <div class="flex justify-end space-x-3 border-t border-slate-700 pt-4">
        <button onclick="closeBedTestModal()" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-semibold">Cancel</button>
        <button onclick="generateBedTestPattern()" class="px-4 py-2 bg-amber-600 hover:bg-amber-500 text-white rounded-lg text-xs font-bold shadow-lg">🎨 Generate & Load Preview</button>
      </div>
    </div>
  </div>

  <script>
    let currentStepSize = 1.0;
    let manualLockedPoints = [null, null, null];

    function copyLog() {
      const box = document.getElementById('log-box');
      if (!box) return;
      const text = box.innerText || box.textContent;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(() => {
          addLog('[INFO] Log copied to clipboard! Ready to paste.');
        }).catch(() => {
          alert(["Log text (Ctrl+C to copy):", text].join(String.fromCharCode(10) + String.fromCharCode(10)));
        });
      } else {
        alert(["Log text (Ctrl+C to copy):", text].join(String.fromCharCode(10) + String.fromCharCode(10)));
      }
    }

    function clearLog() {
      const box = document.getElementById('log-box');
      if (box) box.innerHTML = '<div>[LOG CLEARED] Ready.</div>';
    }

    // Flush any early crashes captured before DOM was ready
    document.addEventListener('DOMContentLoaded', function() {
      const box = document.getElementById('log-box');
      if (box && window.__earlyErrors && window.__earlyErrors.length > 0) {
        window.__earlyErrors.forEach(err => {
          const d = document.createElement('div');
          d.style.color = '#f87171';
          d.style.fontWeight = 'bold';
          d.textContent = '[' + new Date().toLocaleTimeString() + '] ❌ ' + err;
          box.appendChild(d);
        });
        box.scrollTop = box.scrollHeight;
      }
    });

    function addLog(msg) {
      const box = document.getElementById('log-box');
      if (!box || !msg) return;
      const lines = String(msg).split(String.fromCharCode(10)).map(l => l.replace(String.fromCharCode(13), ''));
      const timeStr = new Date().toLocaleTimeString();
      lines.forEach(l => {
        const trimmed = l.trim();
        if (trimmed) {
          box.innerHTML += `<div>[${timeStr}] ${trimmed}</div>`;
        }
      });
      box.scrollTop = box.scrollHeight;
    }

    function setAlignMode(mode) {
      const autoBtn = document.getElementById('mode-auto-btn');
      const manualBtn = document.getElementById('mode-manual-btn');
      const autoCtrl = document.getElementById('auto-align-controls');
      const manualCtrl = document.getElementById('manual-align-controls');

      if (mode === 'auto') {
        autoBtn.className = "px-2 py-0.5 rounded font-semibold bg-purple-600 text-white shadow";
        manualBtn.className = "px-2 py-0.5 rounded font-semibold text-slate-400 hover:text-white";
        autoCtrl.classList.remove('hidden');
        manualCtrl.classList.add('hidden');
        addLog('Switched to Automatic (ArUco) Alignment Mode.');
      } else {
        manualBtn.className = "px-2 py-0.5 rounded font-semibold bg-purple-600 text-white shadow";
        autoBtn.className = "px-2 py-0.5 rounded font-semibold text-slate-400 hover:text-white";
        manualCtrl.classList.remove('hidden');
        autoCtrl.classList.add('hidden');
        addLog('Switched to Manual Camera Jog (Crosshair Target) Alignment Mode.');
      }
    }

    function setStep(val) {
      currentStepSize = val;
      ['01', '1', '10'].forEach(id => {
        const btn = document.getElementById('step-' + id);
        btn.className = "px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono";
      });
      const activeId = val === 0.1 ? '01' : (val === 1.0 ? '1' : '10');
      document.getElementById('step-' + activeId).className = "px-1.5 py-0.5 rounded bg-purple-600 text-white font-mono";
    }

    let jogBusy = false;  // Debounce flag to prevent overlapping jogs

    function jog(axis, dir) {
      if (jogBusy) {
        addLog('[JOG] Previous jog still in progress, please wait...');
        return;
      }
      jogBusy = true;
      const dist = dir * currentStepSize;
      addLog(`[JOG] Moving ${axis} by ${dist > 0 ? '+' : ''}${dist} mm...`);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      fetch(`/api/jog?axis=${axis}&dist=${dist}`, { method: 'POST', signal: controller.signal })
        .then(r => r.json())
        .then(data => {
          addLog(data.message);
          if (data.status === 'error') {
            addLog('[ERROR] Jog command failed on printer.');
          }
        })
        .catch(err => addLog(`[ERROR] Jog request failed: ${err.message}`))
        .finally(() => {
          jogBusy = false;
          clearTimeout(timeoutId);
        });
    }

    function jogHome() {
      if (jogBusy) {
        addLog('[JOG] Previous jog still in progress, cannot home.');
        return;
      }
      jogBusy = true;
      addLog('[JOG] Homing X and Y gantries...');
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000);
      fetch(`/api/jog?axis=HOME&dist=0`, { method: 'POST', signal: controller.signal })
        .then(r => r.json())
        .then(data => {
          addLog(data.message);
          if (data.status === 'error') {
            addLog('[ERROR] Homing failed on printer.');
          }
        })
        .catch(err => addLog(`[ERROR] Homing request failed: ${err.message}`))
        .finally(() => {
          clearTimeout(timeoutId);
          jogBusy = false;
        });
    }

    function jogToCorner(corner) {
      let x = 25, y = 20;
      if (corner === 'FL') { x = 25; y = 20; }
      else if (corner === 'FR') { x = 195; y = 20; }
      else if (corner === 'RL') { x = 25; y = 210; }
      else if (corner === 'RR') { x = 195; y = 210; }

      addLog(`[TRAMMING] Moving cutter toolhead to ${corner} corner (X=${x}, Y=${y} mm)...`);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      fetch(`/api/jog_absolute?x=${x}&y=${y}&z=0.0`, { signal: controller.signal })
        .then(r => r.json())
        .then(data => addLog(data.message))
        .catch(err => addLog(`[ERROR] Corner jog failed: ${err.message}`))
        .finally(() => clearTimeout(timeoutId));
    }

    function lockMark(idx) {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      fetch('/api/toolhead_position', { signal: controller.signal })
        .then(r => r.json())
        .then(data => {
          const pos = data.position || [10.0 + idx * 90.0, 10.0, 5.0];
          manualLockedPoints[idx] = [pos[0], pos[1]];
          const btn = document.getElementById(`mark-${idx}-btn`);
          btn.className = "w-full bg-emerald-950/80 text-emerald-400 py-1 rounded border border-emerald-500/50 font-mono font-bold";
          btn.textContent = `✅ Mark #${idx} Locked (${pos[0].toFixed(1)}, ${pos[1].toFixed(1)})`;
          addLog(`[MANUAL] Locked Registration Mark #${idx} at X=${pos[0].toFixed(2)} mm, Y=${pos[1].toFixed(2)} mm.`);
        })
        .catch(err => addLog(`[ERROR] Failed to lock mark #${idx}: ${err.message}`))
        .finally(() => clearTimeout(timeoutId));
    }

    function solveManualAlignment() {
      if (!manualLockedPoints[0] || !manualLockedPoints[1] || !manualLockedPoints[2]) {
        addLog('[ERROR] Please jog to and lock all 3 registration marks before solving alignment.');
        return;
      }
      addLog('[MANUAL] Solving 2D Affine Transformation Matrix from 3 locked points...');
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);
      fetch('/api/manual_lock', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          actual: manualLockedPoints,
          nominal: [[15.0, 15.0], [195.0, 15.0], [15.0, 282.0]]
        }),
        signal: controller.signal
      })
      .then(r => r.json())
      .then(data => {
        addLog(data.message);
        if (data.status === 'error') {
          addLog('[ERROR] Alignment solve failed. Check that all 3 marks are locked correctly.');
        }
      })
      .catch(err => addLog(`[ERROR] Alignment request failed: ${err.message}`))
      .finally(() => clearTimeout(timeoutId));
    }

    function openSettingsModal() {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      fetch('/api/settings', { signal: controller.signal })
        .then(r => r.json())
        .then(data => {
          if (data.status === 'ok') {
            const cfg = data.settings;
            document.getElementById('cfg-bed-width').value = cfg.bed_width_mm;
            document.getElementById('cfg-bed-height').value = cfg.bed_height_mm;
            document.getElementById('cfg-blade-offset').value = cfg.blade_offset_mm;
            document.getElementById('cfg-cam-offset-x').value = cfg.camera_offset_x_mm;
            document.getElementById('cfg-cam-offset-y').value = cfg.camera_offset_y_mm;
            document.getElementById('cfg-mm-px').value = cfg.mm_per_pixel;
            if (document.getElementById('cfg-camera-settle')) {
              document.getElementById('cfg-camera-settle').value = cfg.camera_settle_seconds !== undefined ? cfg.camera_settle_seconds : 2.0;
            }
            if (document.getElementById('search-mode-select') && cfg.search_mode) {
              document.getElementById('search-mode-select').value = cfg.search_mode;
            }
            document.getElementById('cfg-z-safe').value = cfg.z_safe_mm;
            document.getElementById('cfg-z-cut').value = cfg.z_cut_mm;
            document.getElementById('cfg-z-lift-swivel').value = cfg.z_lift_swivel_mm !== undefined ? cfg.z_lift_swivel_mm : 0.0;
            document.getElementById('cfg-f-cut').value = cfg.f_cut_mm_min;
            document.getElementById('cfg-f-travel').value = cfg.f_travel_mm_min;
            document.getElementById('cfg-moonraker-url').value = cfg.moonraker_url;
            document.getElementById('cfg-crowsnest-url').value = cfg.crowsnest_url;
            const sm = document.getElementById('settings-modal');
            sm.style.display = 'flex';
            sm.classList.remove('hidden');
          }
        })
        .catch(err => addLog(`[ERROR] Settings load failed: ${err.message}`))
        .finally(() => clearTimeout(timeoutId));
    }

    function closeSettingsModal() {
      const sm = document.getElementById('settings-modal');
      sm.style.display = 'none';
      sm.classList.add('hidden');
    }

    function openBedTestModal() {
      const bm = document.getElementById('bed-test-modal');
      bm.style.display = 'flex';
      bm.classList.remove('hidden');
    }

    function closeBedTestModal() {
      const bm = document.getElementById('bed-test-modal');
      bm.style.display = 'none';
      bm.classList.add('hidden');
    }

    function generateBedTestPattern() {
      const grid = parseInt(document.getElementById('cfg-bed-grid').value);
      const boxSize = parseFloat(document.getElementById('cfg-bed-box-size').value);
      const xSize = parseFloat(document.getElementById('cfg-bed-x-size').value);
      const margin = parseFloat(document.getElementById('cfg-bed-margin').value);

      addLog(`[BED TEST] Generating ${grid}-box mesh calibration test pattern...`);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);

      fetch('/api/generate_bed_test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ grid_count: grid, box_size: boxSize, x_size: xSize, margin: margin }),
        signal: controller.signal
      })
      .then(r => r.json())
      .then(data => {
        if (data.status === 'ok') {
          addLog(`[BED TEST] ${data.message} Loading file preview...`);
          closeBedTestModal();
          loadFileList();
          setTimeout(() => {
            const select = document.getElementById('svg-file');
            select.value = data.filename;
            onFileSelectionChanged();
            loadPreview();
          }, 300);
        } else {
          addLog(`[ERROR] Bed test pattern generation failed: ${data.message}`);
        }
      })
      .catch(err => addLog(`[ERROR] Failed to generate bed test pattern: ${err.message}`))
      .finally(() => clearTimeout(timeoutId));
    }

    function syncQuickZCut(val) {
      const v = parseFloat(val);
      if (isNaN(v)) return;
      const setInp = document.getElementById('cfg-z-cut');
      if (setInp) setInp.value = v;
      fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ z_cut_mm: v })
      }).then(r => r.json()).then(d => {
        addLog(`[SETTINGS] Cut Z depth updated to ${v.toFixed(2)} mm`);
      }).catch(() => {});
    }

    function saveSettings() {
      const newCfg = {
        bed_width_mm: parseFloat(document.getElementById('cfg-bed-width').value),
        bed_height_mm: parseFloat(document.getElementById('cfg-bed-height').value),
        blade_offset_mm: parseFloat(document.getElementById('cfg-blade-offset').value),
        camera_offset_x_mm: parseFloat(document.getElementById('cfg-cam-offset-x').value),
        camera_offset_y_mm: parseFloat(document.getElementById('cfg-cam-offset-y').value),
        mm_per_pixel: parseFloat(document.getElementById('cfg-mm-px').value),
        camera_settle_seconds: parseFloat(document.getElementById('cfg-camera-settle')?.value) || 2.0,
        z_safe_mm: parseFloat(document.getElementById('cfg-z-safe').value),
        z_cut_mm: parseFloat(document.getElementById('cfg-z-cut').value),
        z_lift_swivel_mm: parseFloat(document.getElementById('cfg-z-lift-swivel').value) || 0.0,
        f_cut_mm_min: parseFloat(document.getElementById('cfg-f-cut').value),
        f_travel_mm_min: parseFloat(document.getElementById('cfg-f-travel').value),
        moonraker_url: document.getElementById('cfg-moonraker-url').value,
        crowsnest_url: document.getElementById('cfg-crowsnest-url').value,
      };

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newCfg),
        signal: controller.signal
      })
      .then(r => r.json())
      .then(data => {
        if (data.status === 'ok') {
          if (document.getElementById('quick-z-cut') && document.getElementById('cfg-z-cut')) {
            document.getElementById('quick-z-cut').value = document.getElementById('cfg-z-cut').value;
          }
          addLog('[SETTINGS] Calibration parameters updated and saved to config.json!');
          closeSettingsModal();
          loadPreview();
        } else {
          addLog('[ERROR] Failed to save settings: ' + data.message);
        }
      })
      .catch(err => addLog(`[ERROR] Settings save failed: ${err.message}`))
      .finally(() => clearTimeout(timeoutId));
    }

    function refreshCameras() {
      addLog('Scanning attached camera devices...');
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000);
      fetch('/api/cameras', { signal: controller.signal })
        .then(r => r.json())
        .then(data => {
          if (data.status === 'ok' && data.cameras.length > 0) {
            const select = document.getElementById('camera-select');
            select.innerHTML = '';
            data.cameras.forEach(cam => {
              const opt = document.createElement('option');
              opt.value = cam.id;
              opt.textContent = cam.name;
              select.appendChild(opt);
            });
            addLog(`Found ${data.cameras.length} camera sources.`);
            switchCameraStream();
          }
        })
        .catch(err => addLog(`[ERROR] Camera scan failed: ${err.message}`))
        .finally(() => clearTimeout(timeoutId));
    }

    function switchCameraStream() {
      const cam = document.getElementById('camera-select').value;
      const feed = document.getElementById('camera-feed');
      addLog(`Switching live camera stream to source '${cam}'...`);
      feed.src = `/api/camera/stream?source=${cam}&t=${Date.now()}`;
    }

    function loadPreview() {
      const select = document.getElementById('svg-file');
      if (!select || !select.value) {
        addLog('[WARN] No file selected to preview.');
        return;
      }
      const file = select.value;
      const btn = document.getElementById('load-preview-btn');
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = "⏳ Loading Preview...";
      }
      addLog(`Loading SVG preview for '${file}'...`);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);
      fetch(`/api/preview?file=${encodeURIComponent(file)}`, { signal: controller.signal })
        .then(r => r.json())
        .then(data => {
          if (data.status === 'ok') {
            const container = document.getElementById('svg-container');
            if (container) container.innerHTML = data.svg_content;
            addLog(`Rendered SVG (${data.contours_count} contours found).`);
            updateInjectionButtonStates(data.has_fiducials, data.has_touch_dots);
          } else {
            addLog(`Error: ${data.message}`);
          }
        })
        .catch(err => addLog(`[ERROR] Failed to load preview: ${err.message}`))
        .finally(() => {
          clearTimeout(timeoutId);
          if (btn) {
            btn.disabled = false;
            btn.innerHTML = "Load File Preview";
          }
        });
    }

    function openPrintWindow() {
      const select = document.getElementById('svg-file');
      if (!select || !select.value) {
        alert("Please select a file to print first.");
        return;
      }
      const file = select.value;
      addLog(`[PRINT] Sending full color artwork + ArUco fiducials of '${file}' to 2D Paper/Vinyl Printer at 1:1 scale...`);
      window.open(`/print-artwork?file=${encodeURIComponent(file)}`, '_blank');
    }

    function runScan() {
      const select = document.getElementById('svg-file');
      if (!select || !select.value) {
        alert("Please select a cut file in Step 1 first.");
        addLog("[ERROR] No file selected in Step 1.");
        return;
      }
      const file = select.value;
      const camSelect = document.getElementById('camera-select');
      const cam = camSelect ? camSelect.value : 'crowsnest';
      const searchModeSelect = document.getElementById('search-mode-select');
      const searchMode = searchModeSelect ? searchModeSelect.value : 'auto';
      const forceHomeCheck = document.getElementById('force-home-check');
      const forceHome = forceHomeCheck ? forceHomeCheck.checked : false;

      const btn = document.getElementById('run-scan-btn');
      if (btn) {
        btn.disabled = true;
        btn.className = "w-full bg-purple-900 text-purple-200 font-semibold py-2 rounded text-xs shadow-md cursor-not-allowed animate-pulse";
        if (searchMode === 'assisted' || searchMode === 'manual') {
          btn.innerHTML = "🎯 Assisted Align: Adjust paper under camera...";
        } else {
          btn.innerHTML = "⏳ Scanning & Centering Fiducials... (Please Wait)";
        }
      }

      addLog(`[VISION] Starting fiducial scan for '${file}' using camera '${cam}' (Strategy: ${searchMode.toUpperCase()}, Force Home: ${forceHome})...`);
      if (searchMode === 'assisted' || searchMode === 'manual') {
        addLog('[ASSISTED ALIGN] Gantry will move to Marker 1 setup point and wait. Watch live camera feed and adjust paper under camera lens!');
      }
      const controller = new AbortController();
      const scanTimeoutMs = (searchMode === 'assisted' || searchMode === 'manual') ? 240000 : 180000;
      const timeoutId = setTimeout(() => controller.abort(), scanTimeoutMs);
      fetch(`/api/scan?file=${encodeURIComponent(file)}&camera=${encodeURIComponent(cam)}&force_home=${forceHome ? '1' : '0'}&search_mode=${encodeURIComponent(searchMode)}`, { method: 'POST', signal: controller.signal })
        .then(r => r.json())
        .then(data => {
          addLog(data.message);
          if (data.status === 'error') {
            addLog('[ERROR] Vision scan failed.');
          } else {
            addLog('🎉 [SUCCESS] Fiducials located & centered! 2D Affine Matrix solved. Ready to cut in Step 4.');
          }
        })
        .catch(err => addLog(`[ERROR] Scan request failed: ${err.message}`))
        .finally(() => {
          clearTimeout(timeoutId);
          if (btn) {
            btn.disabled = false;
            btn.className = "w-full bg-purple-600 hover:bg-purple-500 text-white font-semibold py-2 rounded text-xs shadow-md";
            btn.innerHTML = "📷 Run Autonomous Scan & Align";
          }
        });
    }

    function onFileSelectionChanged() {
      const select = document.getElementById('svg-file');
      if (!select || !select.value) return;
      const file = select.value;
      const isGcode = Boolean(file.toLowerCase().match(/\.(gcode|nc|gc|tap)$/));
      const svgMode = document.getElementById('step2-svg-mode');
      const gcodeMode = document.getElementById('step2-gcode-mode');
      if (isGcode) {
        if (svgMode) svgMode.classList.add('hidden');
        if (gcodeMode) gcodeMode.classList.remove('hidden');
      } else {
        if (svgMode) svgMode.classList.remove('hidden');
        if (gcodeMode) gcodeMode.classList.add('hidden');
      }
      loadPreview();
    }

    function loadFileList() {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);
      fetch('/api/files', { signal: controller.signal })
        .then(r => r.json())
        .then(data => {
          if (data.status === 'ok' && data.files && data.files.length > 0) {
            const select = document.getElementById('svg-file');
            if (!select) return;
            const currentVal = select.value;
            select.innerHTML = '';
            data.files.forEach(f => {
              const opt = document.createElement('option');
              opt.value = f;
              opt.textContent = f;
              select.appendChild(opt);
            });
            if (currentVal && data.files.includes(currentVal)) {
              select.value = currentVal;
            } else {
              select.value = data.files[0];
            }
            addLog(`Loaded ${data.files.length} vector/gcode files from workspace.`);
            onFileSelectionChanged();
          } else {
            addLog('[WARN] No vector or G-code files found in workspace.');
          }
        })
        .catch(err => addLog(`[ERROR] File list failed: ${err.message}`))
        .finally(() => clearTimeout(timeoutId));
    }

    function uploadSelectedFile() {
      const input = document.getElementById('file-upload-input');
      if (!input.files || input.files.length === 0) return;
      const file = input.files[0];
      const formData = new FormData();
      formData.append('file', file);

      addLog(`Uploading '${file.name}' to workspace...`);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);
      fetch('/api/upload', {
        method: 'POST',
        body: formData,
        signal: controller.signal
      })
      .then(r => r.json())
      .then(data => {
        if (data.status === 'ok') {
          addLog(`[UPLOAD] ${data.message}`);
          loadFileList();
        } else {
          addLog(`[ERROR] Upload failed: ${data.message}`);
        }
      })
      .catch(err => addLog(`[ERROR] Upload failed: ${err.message}`))
      .finally(() => clearTimeout(timeoutId));
    }

    function runBoundingBoxFrame() {
      const select = document.getElementById('svg-file');
      if (!select || !select.value) {
        alert("Please select a file in Step 1 first.");
        return;
      }
      const file = select.value;
      const btn = document.getElementById('run-frame-btn');
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = "<span>⏳ Tracing Frame...</span>";
      }
      addLog(`[FRAME] Tracing outer 4-corner bounding box for '${file}' at Safe Z...`);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);
      fetch(`/api/frame?file=${encodeURIComponent(file)}`, { method: 'POST', signal: controller.signal })
        .then(r => r.json())
        .then(data => {
          addLog(data.message);
          if (data.status === 'error') {
            addLog('[ERROR] Framing failed on printer.');
          }
        })
        .catch(err => addLog(`[ERROR] Framing request failed: ${err.message}`))
        .finally(() => {
          clearTimeout(timeoutId);
          if (btn) {
            btn.disabled = false;
            btn.innerHTML = "<span>🔲 Frame Bounds</span>";
          }
        });
    }

    function updateInjectionButtonStates(hasFids, hasDots) {
      const fidsBtn = document.getElementById('btn-inject-fiducials');
      const dotsBtn = document.getElementById('btn-inject-dots');

      if (fidsBtn) {
        if (hasFids) {
          fidsBtn.disabled = true;
          fidsBtn.className = "w-full bg-slate-800/90 text-emerald-400 font-semibold py-1.5 rounded text-xs shadow border border-emerald-800/60 cursor-not-allowed flex items-center justify-center space-x-1 opacity-80";
          fidsBtn.innerHTML = "<span>✅ ArUco Fiducials Already Added</span>";
          fidsBtn.title = "ArUco fiducials are already present in this file. Adding again is disabled.";
        } else {
          fidsBtn.disabled = false;
          fidsBtn.className = "w-full bg-emerald-700 hover:bg-emerald-600 text-white font-semibold py-1.5 rounded text-xs shadow flex items-center justify-center space-x-1";
          fidsBtn.innerHTML = "<span>🏁 Add ArUco Fiducials to SVG</span>";
          fidsBtn.title = "Add 3 ArUco fiducials (DICT_4X4_50, 10x10mm) to sheet corners";
        }
      }

      if (dotsBtn) {
        if (hasDots) {
          dotsBtn.disabled = true;
          dotsBtn.className = "w-full bg-slate-800/90 text-blue-400 font-semibold py-1.5 rounded text-xs shadow border border-blue-800/60 cursor-not-allowed flex items-center justify-center space-x-1 opacity-80";
          dotsBtn.innerHTML = "<span>✅ 3 Test Dots Already Added</span>";
          dotsBtn.title = "Verification test dots are already present in this file. Adding again is disabled.";
        } else {
          dotsBtn.disabled = false;
          dotsBtn.className = "w-full bg-blue-700 hover:bg-blue-600 text-white font-semibold py-1.5 rounded text-xs shadow flex items-center justify-center space-x-1";
          dotsBtn.innerHTML = "<span>🎯 Add 3 Test Dots to SVG</span>";
          dotsBtn.title = "Add 3 physical verification touch-off dots beside fiducials";
        }
      }
    }

    function injectFiducialsToSvg() {
      const select = document.getElementById('svg-file');
      if (!select || !select.value) {
        alert("Please select an SVG file in Step 1 first.");
        return;
      }
      const file = select.value;
      const btn = document.getElementById('btn-inject-fiducials');
      if (btn && btn.disabled) {
        alert(`ArUco fiducials are already present in '${file}'. You cannot add them again.`);
        return;
      }
      addLog(`[SVG] Injecting 3 ArUco fiducials into '${file}'...`);

      fetch(`/api/inject_fiducials?file=${encodeURIComponent(file)}`, { method: 'POST' })
        .then(r => r.json())
        .then(data => {
          addLog(data.message);
          if (data.status === 'ok') {
            loadPreview();
            alert(`ArUco Fiducials successfully added to '${file}'!\n\nAdded 3 markers (DICT_4X4_50, 10x10mm):\n- Marker 1 (ID 0): Top-Left (Back-Left)\n- Marker 2 (ID 1): Top-Right (Back-Right)\n- Marker 3 (ID 2): Bottom-Left (Front-Left)\n\nPreview reloaded. Use 'Print Sheet (1:1)' in Step 2 to print!`);
          } else if (data.status === 'already_exists') {
            loadPreview();
            alert(`Notice: ${data.message}`);
          } else {
            alert("Failed to inject fiducials: " + data.message);
          }
        })
        .catch(err => addLog(`[ERROR] Fiducial injection failed: ${err.message}`));
    }

    function injectTouchDotsToSvg() {
      const select = document.getElementById('svg-file');
      if (!select || !select.value) {
        alert("Please select an SVG file in Step 1 first.");
        return;
      }
      const file = select.value;
      const btn = document.getElementById('btn-inject-dots');
      if (btn && btn.disabled) {
        alert(`Verification test dots are already present in '${file}'. You cannot add them again.`);
        return;
      }
      addLog(`[SVG] Injecting 3 verification touch dots and Paper (0,0) mark into '${file}'...`);

      fetch(`/api/inject_touch_dots?file=${encodeURIComponent(file)}`, { method: 'POST' })
        .then(r => r.json())
        .then(data => {
          addLog(data.message);
          if (data.status === 'ok') {
            loadPreview();
            alert(`Verification dots successfully added to '${file}'!\n\n3 Target dots have been positioned beside fiducials:\n- Dot 1: 0° (Right)\n- Dot 2: 270° (Down)\n- Dot 3: 45° (Diagonal)\n- Paper Origin (0,0): Bottom-left corner\n\nPreview reloaded. Use 'Print Sheet (1:1)' in Step 2 to print!`);
          } else if (data.status === 'already_exists') {
            loadPreview();
            alert(`Notice: ${data.message}`);
          } else {
            alert("Failed to inject dots: " + data.message);
          }
        })
        .catch(err => addLog(`[ERROR] SVG injection failed: ${err.message}`));
    }

    function downloadSvg() {
      const select = document.getElementById('svg-file');
      if (!select || !select.value) {
        alert("Please select a file first.");
        return;
      }
      const file = select.value;
      window.location.href = `/api/download_svg?file=${encodeURIComponent(file)}`;
    }

    function runTouchVerification() {
      const select = document.getElementById('svg-file');
      if (!select || !select.value) {
        alert("Please select a file in Step 1 first.");
        return;
      }
      const file = select.value;
      const quickZInp = document.getElementById('quick-z-cut');
      const zCut = quickZInp ? quickZInp.value : '0.0';

      const btn = document.getElementById('verify-dots-btn');
      if (btn) {
        btn.disabled = true;
        btn.className = "w-full bg-purple-950 text-purple-300 font-semibold py-1.5 rounded text-xs shadow cursor-not-allowed animate-pulse flex items-center justify-center space-x-1 mt-1 border border-purple-500/40";
        btn.innerHTML = "<span>⏳ Visiting 3 Test Dots & Parking at (0,0)...</span>";
      }

      addLog(`[VERIFY] Initiating physical touch-off test for '${file}'...`);
      addLog(`[VERIFY] Blade will touch Dot 1 (0°), Dot 2 (270°), Dot 3 (45°), then park at Paper (0,0).`);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 120000);

      fetch(`/api/verify_touch_dots?file=${encodeURIComponent(file)}&z_cut=${encodeURIComponent(zCut)}`, {
        method: 'POST',
        signal: controller.signal
      })
      .then(r => r.json())
      .then(data => {
        addLog(data.message);
        if (data.status === 'ok') {
          openTouchVerifyModal();
        } else {
          addLog('[ERROR] Touch verification failed: ' + data.message);
          alert('Verification move failed: ' + data.message);
        }
      })
      .catch(err => {
        addLog(`[ERROR] Verification request failed: ${err.message}`);
      })
      .finally(() => {
        clearTimeout(timeoutId);
        if (btn) {
          btn.disabled = false;
          btn.className = "w-full bg-purple-800 hover:bg-purple-700 text-white font-semibold py-1.5 rounded text-xs shadow flex items-center justify-center space-x-1 mt-1 border border-purple-500/40";
          btn.innerHTML = "<span>📍 Touch-Off Test (3 Dots & Park)</span>";
        }
      });
    }

    function openTouchVerifyModal() {
      const m = document.getElementById('touch-verify-modal');
      if (m) m.style.display = 'flex';
    }

    function closeTouchVerifyModal() {
      const m = document.getElementById('touch-verify-modal');
      if (m) m.style.display = 'none';
    }

    function confirmTouchAndCut() {
      closeTouchVerifyModal();
      addLog("[VERIFY] User confirmed needle alignment! Starting vinyl cut...");
      runCut();
    }

    function adjustOffsetsFromVerify() {
      closeTouchVerifyModal();
      openSettingsModal();
    }

    function downloadGcode() {
      const select = document.getElementById('svg-file');
      if (!select || !select.value) {
        alert("Please select a file in Step 1 first.");
        return;
      }
      const file = select.value;
      const dryCheck = document.getElementById('dry-run');
      const dry = (dryCheck && dryCheck.checked) ? '1' : '0';
      const quickZInp = document.getElementById('quick-z-cut');
      const zCutVal = quickZInp ? quickZInp.value : '';
      addLog(`[GCODE] Generating & downloading Klipper G-code for '${file}'...`);
      const verifyCheck = document.getElementById('verify-touch-check');
      const verifyTouch = (verifyCheck && verifyCheck.checked) ? '1' : '0';
      window.location.href = `/api/download_gcode?file=${encodeURIComponent(file)}&dry=${dry}&z_cut=${encodeURIComponent(zCutVal)}&verify_touch=${verifyTouch}`;
    }

    function runCut() {
      const select = document.getElementById('svg-file');
      if (!select || !select.value) {
        alert("Please select a cut file in Step 1 first.");
        addLog("[ERROR] No file selected in Step 1.");
        return;
      }
      const file = select.value;
      const dryCheck = document.getElementById('dry-run');
      const dry = (dryCheck && dryCheck.checked) ? '--dry-run' : '';
      const quickZInp = document.getElementById('quick-z-cut');
      const zCutVal = quickZInp ? quickZInp.value : '';

      const btn = document.getElementById('run-cut-btn');
      if (btn) {
        btn.disabled = true;
        btn.className = "w-full bg-emerald-900 text-emerald-200 font-bold py-2 rounded text-xs shadow-lg cursor-not-allowed animate-pulse";
        btn.innerHTML = "⏳ Preparing & Sending Job to Cutter...";
      }
      addLog(`[CUTTER] Processing '${file}' & sending G-code to Ender 3 cutter...`);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 180000);
      const verifyCheck = document.getElementById('verify-touch-check');
      const verifyTouch = (verifyCheck && verifyCheck.checked) ? '1' : '0';
      fetch(`/api/cut?file=${encodeURIComponent(file)}&dry=${encodeURIComponent(dry)}&z_cut=${encodeURIComponent(zCutVal)}&verify_touch=${verifyTouch}`, { method: 'POST', signal: controller.signal })
        .then(r => r.json())
        .then(data => {
          addLog(data.message);
          if (data.status === 'error') {
            addLog('[ERROR] Cut job failed to start.');
          } else {
            addLog('🎉 [SUCCESS] Cut job sent to Klipper! Cutter is in motion.');
          }
        })
        .catch(err => addLog(`[ERROR] Cut request failed: ${err.message}`))
        .finally(() => {
          clearTimeout(timeoutId);
          if (btn) {
            btn.disabled = false;
            btn.className = "w-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-2 rounded text-xs shadow-lg";
            btn.innerHTML = "🚀 START VINYL CUT";
          }
        });
    }

    // Connection health monitor
    function resumeCut() {
      addLog("[RESUME] Sending resume command (RESUME_BASE to bypass extruder temperature checks)...");
      const btn = document.getElementById('resume-cut-btn');
      if (btn) { btn.disabled = true; btn.innerText = "⏳ Resuming..."; }
      fetch('/api/resume', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
          addLog(`[RESUME] ${data.message}`);
          if (data.status !== 'ok') {
            alert(["Resume failed: " + data.message, "Tip: In Mainsail console, type RESUME_BASE or set min_extrude_temp: 0 in printer.cfg."].join(String.fromCharCode(10) + String.fromCharCode(10)));
          }
        })
        .catch(err => addLog(`[ERROR] Resume request failed: ${err.message}`))
        .finally(() => {
          if (btn) { btn.disabled = false; btn.innerHTML = "<span>▶ RESUME CUT</span>"; }
        });
    }

    function cancelCut() {
      if (!confirm("Are you sure you want to cancel the active print job?")) return;
      addLog("[CANCEL] Canceling active print job...");
      fetch('/api/cancel', { method: 'POST' })
        .then(r => r.json())
        .then(data => addLog(`[CANCEL] ${data.message}`))
        .catch(err => addLog(`[ERROR] Cancel failed: ${err.message}`));
    }

    function checkConnection() {
      fetch('/api/settings', { cache: 'no-store' })
        .then(r => r.json())
        .then(data => {
          const badge = document.getElementById('klipper-status-badge');
          if (badge) {
            badge.className = "cursor-pointer px-3 py-1 rounded-full text-xs bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
            badge.innerHTML = "● Cutter Studio Online (Port 5000)";
          }
        })
        .catch(() => {
          const badge = document.getElementById('klipper-status-badge');
          if (badge) {
            badge.className = "cursor-pointer px-3 py-1 rounded-full text-xs bg-rose-500/20 text-rose-400 border border-rose-500/30 animate-pulse";
            badge.innerHTML = "⚠️ Cutter Studio Disconnected - Refreshing...";
          }
        });
    }

    // Auto-refresh cameras & load file list on page load
    function initStudio() {
      refreshCameras();
      loadFileList();
      fetch('/api/settings').then(r => r.json()).then(d => {
        if (d.settings && d.settings.z_cut_mm !== undefined) {
          const qz = document.getElementById('quick-z-cut');
          if (qz) qz.value = d.settings.z_cut_mm;
        }
      }).catch(() => {});
      setInterval(checkConnection, 5000);
      setTimeout(onFileSelectionChanged, 100);
    }

    if (document.readyState === 'loading') {
      window.addEventListener('DOMContentLoaded', initStudio);
    } else {
      initStudio();
    }
  </script>
</body>
</html>
"""


class WebRequestHandler(BaseHTTPRequestHandler):
    """HTTP Handler for Cricut Web GUI with MJPEG Camera Streaming, Settings, and Manual Jog Alignment."""

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            files = get_available_workspace_files()
            options_html = "\n".join(f'<option value="{f}">{f}</option>' for f in files) if files else '<option value="Cut_size.svg">Cut_size.svg</option>'
            page_content = HTML_PAGE.replace('<option value="Cut.svg">Cut.svg</option>', options_html)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page_content.encode("utf-8"))
        elif parsed.path == "/api/files":
            files = get_available_workspace_files()
            resp = {"status": "ok", "files": files}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        elif parsed.path == "/api/settings":
            cfg = load_config()
            resp = {"status": "ok", "settings": cfg}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        elif parsed.path == "/api/cameras":
            cams = camera_streamer.enumerate_cameras()
            resp = {"status": "ok", "cameras": cams}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        elif parsed.path == "/api/toolhead_position":
            try:
                client = get_moonraker_client()
                pos = client.get_toolhead_position()
                resp = {"status": "ok", "position": list(pos)}
            except Exception:
                resp = {"status": "ok", "position": [15.0, 15.0, 5.0]}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        elif parsed.path == "/api/camera/stream":
            query = parse_qs(parsed.query)
            source_str = query.get("source", ["crowsnest"])[0]

            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            frame_count = 0
            cap = None

            if source_str.isdigit() and cv2 is not None:
                try:
                    cam_idx = int(source_str)
                    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(cam_idx)
                except Exception as e:
                    logger.warning("Could not open USB camera %s: %s", source_str, e)
                    cap = None

            try:
                while True:
                    frame_count += 1
                    frame = None

                    if source_str == "crowsnest" or (not source_str.isdigit() and source_str != "mock"):
                        raw_frame = camera_streamer.fetch_crowsnest_frame()
                        if raw_frame is not None:
                            frame = camera_streamer.annotate_frame(raw_frame)

                    if frame is None and cap is not None and cap.isOpened():
                        ret, raw_frame = cap.read()
                        if ret and raw_frame is not None:
                            frame = camera_streamer.annotate_frame(raw_frame)

                    if frame is None:
                        frame = camera_streamer.generate_mock_frame(frame_count)

                    if frame is not None and cv2 is not None:
                        _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        jpeg_bytes = jpeg.tobytes()

                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpeg_bytes)))
                        self.end_headers()
                        self.wfile.write(jpeg_bytes)
                        self.wfile.write(b"\r\n")

                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError, Exception):
                pass
            finally:
                if cap is not None:
                    cap.release()

        elif parsed.path == "/api/preview":
            query = parse_qs(parsed.query)
            filename = query.get("file", ["Cut.svg"])[0]
            filepath = resolve_workspace_file(filename)

            if os.path.isfile(filepath):
                try:
                    cfg = load_config()
                    bed_w = cfg.get("bed_width_mm", 210.0)
                    bed_h = cfg.get("bed_height_mm", 297.0)

                    if filename.lower().endswith((".gcode", ".nc", ".gc", ".tap")):
                        count, preview_html = render_gcode_preview_html(filepath, bed_w, bed_h)
                        resp = {"status": "ok", "contours_count": count, "svg_content": preview_html}
                    else:
                        with open(filepath, "r", encoding="utf-8") as f:
                            svg_code = f.read()

                        # Sanitize SVG: ensure any display:none on artwork images or layers is enabled on-the-fly
                        import re
                        svg_code_sanitized = svg_code.replace("display:none;", "display:inline;").replace("display:none", "display:inline")
                        # Restore any ns*:href to xlink:href and add HTML5 href for reliable browser image rendering
                        svg_code_sanitized = re.sub(r'\b(?:ns\d+:href|xlink:href)="([^"]+)"', r'xlink:href="\1" href="\1"', svg_code_sanitized)
                        if 'xlink:href=' in svg_code_sanitized and 'xmlns:xlink=' not in svg_code_sanitized:
                            svg_code_sanitized = re.sub(r'(<svg\b[^>]*)', r'\1 xmlns:xlink="http://www.w3.org/1999/xlink"', svg_code_sanitized, count=1)

                        import re
                        w_match = re.search(r'width=["\']([0-9.]+)(mm|px)?["\']', svg_code)
                        h_match = re.search(r'height=["\']([0-9.]+)(mm|px)?["\']', svg_code)
                        vb_match = re.search(r'viewBox=["\']([0-9.\s,-]+)["\']', svg_code)

                        svg_w = float(w_match.group(1)) if w_match else 210.0
                        svg_h = float(h_match.group(1)) if h_match else 297.0

                        if vb_match:
                            vb_parts = [float(x) for x in re.split(r'[\s,]+', vb_match.group(1).strip()) if x]
                            if len(vb_parts) == 4:
                                if not w_match:
                                    svg_w = vb_parts[2]
                                if not h_match:
                                    svg_h = vb_parts[3]

                        # Bounding total viewport encompassing BOTH Bed and Sheet at 1:1 physical scale
                        total_w = max(bed_w, svg_w)
                        total_h = max(bed_h, svg_h)

                        bed_w_pct = (bed_w / total_w) * 100.0
                        bed_h_pct = (bed_h / total_h) * 100.0

                        sheet_w_pct = (svg_w / total_w) * 100.0
                        sheet_h_pct = (svg_h / total_h) * 100.0

                        exceeds_w = svg_w > bed_w
                        exceeds_h = svg_h > bed_h
                        exceeds = exceeds_w or exceeds_h

                        diff_h = max(0.0, svg_h - bed_h)
                        diff_w = max(0.0, svg_w - bed_w)

                        badge_exceeds = ""
                        if exceeds:
                            ex_msg = []
                            if exceeds_w:
                                ex_msg.append(f"+{diff_w:.0f}mm X")
                            if exceeds_h:
                                ex_msg.append(f"+{diff_h:.0f}mm Y")
                            badge_exceeds = f"<span style='padding:2px 8px; border-radius:4px; background:rgba(239, 68, 68, 0.25); color:#f87171; border:1px solid rgba(239, 68, 68, 0.4); font-size:10px; font-weight:bold;'>⚠️ Exceeds Bed ({', '.join(ex_msg)})</span>"

                        preview_html = f'''<div style="width:100%; height:100%; display:flex; flex-direction:column; justify-content:center; align-items:center; padding:4px; background:#090d16; user-select:none;">
                          <!-- Dimension Status Header -->
                          <div style="display:flex; align-items:center; gap:8px; font-size:11px; font-family:monospace; margin-bottom:6px; z-index:10;">
                            <span style="padding:2px 8px; border-radius:4px; background:rgba(56, 189, 248, 0.15); color:#38bdf8; border:1px solid rgba(56, 189, 248, 0.3); font-weight:600;">
                              🛏️ Bed: {bed_w:.0f}×{bed_h:.0f} mm
                            </span>
                            <span style="padding:2px 8px; border-radius:4px; background:rgba(16, 185, 129, 0.15); color:#34d399; border:1px solid rgba(16, 185, 129, 0.3); font-weight:600;">
                              📄 Sheet: {svg_w:.0f}×{svg_h:.0f} mm
                            </span>
                            {badge_exceeds}
                          </div>

                          <!-- Total Viewport Container holding Bed + Sheet at 1:1 Physical Scale -->
                          <div style="aspect-ratio:{total_w}/{total_h}; width:100%; max-width:92%; max-height:82%; background:#050811; border:1px solid #1e293b; border-radius:8px; box-shadow:0 8px 24px rgba(0,0,0,0.7); position:relative; overflow:hidden; display:flex; justify-content:flex-start; align-items:flex-start; padding:0;">

                            <!-- 1. Physical Printer Bed Box -->
                            <div style="position:absolute; top:0; left:0; width:{bed_w_pct:.2f}%; height:{bed_h_pct:.2f}%; background:#111827; border:2px solid #3b82f6; border-radius:4px; box-sizing:border-box; z-index:1; overflow:hidden;">
                              <!-- Grid Pattern Overlay -->
                              <div style="position:absolute; inset:0; background-image:linear-gradient(to right, rgba(59, 130, 246, 0.15) 1px, transparent 1px), linear-gradient(to bottom, rgba(59, 130, 246, 0.15) 1px, transparent 1px); background-size: 20px 20px; pointer-events:none;"></div>
                              <div style="position:absolute; bottom:4px; left:6px; font-size:9px; font-family:monospace; color:#60a5fa; background:rgba(15,23,42,0.85); padding:1px 4px; border-radius:2px; pointer-events:none; z-index:5;">(0,0) Bed Origin</div>
                            </div>

                            <!-- 2. SVG Paper Sheet (True 1:1 Scale) -->
                            <div style="position:absolute; top:0; left:0; width:{sheet_w_pct:.2f}%; height:{sheet_h_pct:.2f}%; background:#ffffff; box-shadow:0 4px 16px rgba(0,0,0,0.6); z-index:2; display:flex; justify-content:center; align-items:center; border:1px solid #cbd5e1; overflow:hidden;">
                              {svg_code_sanitized}

                              <!-- Red Hazard Overlay for portion exceeding bed Y/X limit -->
                              {"<div style='position:absolute; top:" + f"{(bed_h_pct / sheet_h_pct * 100.0):.2f}" + "%; bottom:0; left:0; right:0; background:rgba(239, 68, 68, 0.35); border-top:2px dashed #ef4444; backdrop-filter:blur(1px); display:flex; align-items:center; justify-content:center; color:#ffffff; font-size:10px; font-family:monospace; font-weight:bold; text-shadow:0 1px 3px rgba(0,0,0,0.8); z-index:10; pointer-events:none;'>⛔ OUTSIDE PRINTABLE BED (+" + f"{diff_h:.0f}mm EXCEEDED)</div>" if exceeds_h else ""}
                              {"<div style='position:absolute; left:" + f"{(bed_w_pct / sheet_w_pct * 100.0):.2f}" + "%; right:0; top:0; bottom:0; background:rgba(239, 68, 68, 0.35); border-left:2px dashed #ef4444; backdrop-filter:blur(1px); display:flex; align-items:center; justify-content:center; color:#ffffff; font-size:10px; font-family:monospace; font-weight:bold; text-shadow:0 1px 3px rgba(0,0,0,0.8); z-index:10; pointer-events:none;'>⛔ OUTSIDE BED</div>" if (exceeds_w and not exceeds_h) else ""}
                            </div>

                          </div>
                        </div>'''

                        sys.path.insert(0, os.path.join(WORKSPACE_DIR, "src"))
                        from drag_knife_planner import SVGParser
                        parser_obj = SVGParser(step_size_mm=1.0)
                        contours = parser_obj.parse_svg_paths(filepath, flip_y=False)
                        from src.svg_injector import check_svg_has_fiducials, check_svg_has_touch_dots
                        has_fids = check_svg_has_fiducials(filepath)
                        has_dots = check_svg_has_touch_dots(filepath)
                        resp = {
                            "status": "ok",
                            "contours_count": len(contours),
                            "svg_content": preview_html,
                            "has_fiducials": has_fids,
                            "has_touch_dots": has_dots
                        }
                except Exception as e:
                    resp = {"status": "error", "message": str(e)}
            else:
                resp = {"status": "error", "message": f"File '{filename}' not found."}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif parsed.path == "/print-artwork":
            query = parse_qs(parsed.query)
            filename = query.get("file", ["Cut.svg"])[0]
            filepath = resolve_workspace_file(filename)

            if os.path.isfile(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        svg_code = f.read()
                    import re
                    svg_code_sanitized = svg_code.replace("display:none;", "display:inline;").replace("display:none", "display:inline")
                    svg_code_sanitized = re.sub(r'\b(?:ns\d+:href|xlink:href)="([^"]+)"', r'xlink:href="\1" href="\1"', svg_code_sanitized)
                    if 'xlink:href=' in svg_code_sanitized and 'xmlns:xlink=' not in svg_code_sanitized:
                        svg_code_sanitized = re.sub(r'(<svg\b[^>]*)', r'\1 xmlns:xlink="http://www.w3.org/1999/xlink"', svg_code_sanitized, count=1)
                    print_html = f"""<!DOCTYPE html>
<html>
<head>
  <title>Print Sheet - {filename}</title>
  <style>
    @page {{ size: A4 portrait; margin: 0; }}
    html, body {{ margin: 0; padding: 0; width: 210mm; height: 297mm; background: white; display: flex; justify-content: center; align-items: center; }}
    svg {{ width: 210mm; height: 297mm; display: block; }}
  </style>
</head>
<body onload="setTimeout(function(){{ window.print(); }}, 600)">
  {svg_code_sanitized}
</body>
</html>"""
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(print_html.encode("utf-8"))
                    return
                except Exception as e:
                    resp = {"status": "error", "message": str(e)}
            else:
                resp = {"status": "error", "message": f"File '{filename}' not found."}

        elif parsed.path == "/api/download_svg":
            query = parse_qs(parsed.query)
            filename = query.get("file", ["Cut.svg"])[0]
            filepath = resolve_workspace_file(filename)
            if os.path.isfile(filepath):
                with open(filepath, "rb") as f:
                    svg_bytes = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("Content-Length", str(len(svg_bytes)))
                self.end_headers()
                self.wfile.write(svg_bytes)
                return
            else:
                self.send_response(404)
                self.end_headers()
                return

        elif parsed.path == "/api/download_gcode":
            query = parse_qs(parsed.query)
            filename = query.get("file", ["Cut.svg"])[0]
            dry_flag = query.get("dry", ["0"])[0]
            filepath = resolve_workspace_file(filename)

            if os.path.isfile(filepath):
                base_name = os.path.splitext(filename)[0]
                out_filename = f"{base_name}_cutter.gcode"
                out_filepath = os.path.join(WORKSPACE_DIR, out_filename)

                cmd = [PYTHON_EXE, "vision_cutter.py", "--svg", filename, "--output", out_filename, "--mock"]
                affine_file = os.path.join(WORKSPACE_DIR, "affine_matrix.json")
                if os.path.isfile(affine_file):
                    try:
                        with open(affine_file, "r", encoding="utf-8") as f:
                            af_data = json.load(f)
                        if af_data.get("file") == filename or (time.time() - af_data.get("timestamp", 0)) < 3600:
                            cmd.append("--use-cached-alignment")
                    except Exception:
                        pass
                cfg = load_config()
                z_lift_swivel = cfg.get("z_lift_swivel_mm", 0.0)
                if z_lift_swivel > 0.0:
                    cmd.extend(["--z-lift-swivel", str(z_lift_swivel)])

                z_cut_param = query.get("z_cut", [""])[0]
                if z_cut_param != "":
                    try:
                        cmd.extend(["--z-cut", str(float(z_cut_param))])
                    except ValueError:
                        pass

                verify_touch = query.get("verify_touch", ["1"])[0]
                if verify_touch == "1" or verify_touch == "true":
                    cmd.append("--verify-touch")

                if dry_flag == "1" or dry_flag == "true":
                    cmd.append("--dry-run")

                try:
                    res = subprocess.run(cmd, cwd=WORKSPACE_DIR, capture_output=True, text=True, timeout=30)
                    if res.returncode == 0 and os.path.isfile(out_filepath):
                        with open(out_filepath, "rb") as f:
                            gcode_bytes = f.read()

                        self.send_response(200)
                        self.send_header("Content-Type", "text/x-gcode; charset=utf-8")
                        self.send_header("Content-Disposition", f'attachment; filename="{out_filename}"')
                        self.send_header("Content-Length", str(len(gcode_bytes)))
                        self.end_headers()
                        self.wfile.write(gcode_bytes)
                        return
                    else:
                        resp = {"status": "error", "message": res.stderr or res.stdout or "G-code generation failed."}
                except Exception as err:
                    resp = {"status": "error", "message": str(err)}
            else:
                resp = {"status": "error", "message": f"File '{filename}' not found."}

            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/api/upload":
            content_type = self.headers.get("Content-Type", "")
            content_length = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_length)

            filename = None
            file_bytes = None

            if "multipart/form-data" in content_type:
                import re
                boundary_match = re.search(r'boundary=([^\s;]+)', content_type)
                if boundary_match:
                    boundary = boundary_match.group(1).encode()
                    parts = post_body.split(b"--" + boundary)
                    for part in parts:
                        if b'filename="' in part:
                            header_part, content = part.split(b"\r\n\r\n", 1)
                            content = content.rsplit(b"\r\n", 1)[0]
                            filename_match = re.search(r'filename="([^"]+)"', header_part.decode("utf-8", errors="ignore"))
                            if filename_match:
                                filename = os.path.basename(filename_match.group(1))
                                file_bytes = content
                                break
            elif query.get("filename"):
                filename = os.path.basename(query.get("filename")[0])
                file_bytes = post_body

            if filename and file_bytes is not None:
                save_path = os.path.join(WORKSPACE_DIR, filename)
                # If SVG, normalize any ns*:href attributes so images render in browser
                if filename.lower().endswith(".svg"):
                    try:
                        svg_text = file_bytes.decode("utf-8", errors="ignore")
                        import re
                        svg_text = re.sub(r'\b(?:ns\d+:href|xlink:href)="([^"]+)"', r'xlink:href="\1" href="\1"', svg_text)
                        if 'xlink:href=' in svg_text and 'xmlns:xlink=' not in svg_text:
                            svg_text = re.sub(r'(<svg\b[^>]*)', r'\1 xmlns:xlink="http://www.w3.org/1999/xlink"', svg_text, count=1)
                        file_bytes = svg_text.encode("utf-8")
                    except Exception:
                        pass
                with open(save_path, "wb") as f:
                    f.write(file_bytes)
                resp = {"status": "ok", "filename": filename, "message": f"Successfully uploaded '{filename}'."}
            else:
                resp = {"status": "error", "message": "Failed to extract uploaded file."}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif parsed.path == "/api/frame":
            filename = query.get("file", ["Cut.svg"])[0]
            filepath = resolve_workspace_file(filename)

            if os.path.isfile(filepath):
                cfg = load_config()
                z_safe = cfg.get("z_safe_mm", 5.0)
                f_travel = cfg.get("f_travel_mm_min", 3000.0)
                moon_url = cfg.get("moonraker_url", "http://localhost:7125")

                min_x, max_x, min_y, max_y = get_file_bounding_box(filepath)

                gcode_lines = [
                    "G90",
                    f"G0 Z{z_safe:.2f} F{f_travel:.0f}",
                    f"G0 X{min_x:.2f} Y{min_y:.2f} F{f_travel:.0f}",
                    f"G0 X{max_x:.2f} Y{min_y:.2f} F{f_travel:.0f}",
                    f"G0 X{max_x:.2f} Y{max_y:.2f} F{f_travel:.0f}",
                    f"G0 X{min_x:.2f} Y{max_y:.2f} F{f_travel:.0f}",
                    f"G0 X{min_x:.2f} Y{min_y:.2f} F{f_travel:.0f}",
                ]
                gcode_str = "\n".join(gcode_lines)

                try:
                    sys.path.insert(0, os.path.join(WORKSPACE_DIR, "src"))
                    from moonraker_client import MoonrakerClient
                    client = MoonrakerClient(base_url=moon_url)
                    if client.is_connected():
                        success = client.send_gcode(gcode_str)
                        msg = f"[FRAME] Traced 4-corner bounding box on printer! X=[{min_x:.1f}, {max_x:.1f}] mm, Y=[{min_y:.1f}, {max_y:.1f}] mm."
                    else:
                        msg = f"[MOCK FRAME] Traced outer bounding box: X=[{min_x:.1f}, {max_x:.1f}] mm, Y=[{min_y:.1f}, {max_y:.1f}] mm at Z={z_safe}mm."
                    resp = {"status": "ok", "message": msg, "bounds": [min_x, max_x, min_y, max_y]}
                except Exception as err:
                    resp = {"status": "ok", "message": f"[MOCK FRAME] Traced outer bounding box: X=[{min_x:.1f}, {max_x:.1f}] mm, Y=[{min_y:.1f}, {max_y:.1f}] mm."}
            else:
                resp = {"status": "error", "message": f"File '{filename}' not found."}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif parsed.path == "/api/settings":
            content_length = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_length)
            try:
                data = json.loads(post_body.decode("utf-8"))
                if save_config(data):
                    resp = {"status": "ok", "message": "Settings saved successfully."}
                else:
                    resp = {"status": "error", "message": "Failed to save settings file."}
            except Exception as e:
                resp = {"status": "error", "message": str(e)}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif parsed.path == "/api/generate_bed_test":
            content_length = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_length)
            try:
                data = json.loads(post_body.decode("utf-8")) if post_body else {}
                grid_count = int(data.get("grid_count", 16))
                box_size = float(data.get("box_size", 20.0))
                x_size = float(data.get("x_size", 5.0))
                margin = float(data.get("margin", 15.0))

                svg_filename = generate_bed_test_pattern(grid_count, box_size, x_size, margin)
                resp = {
                    "status": "ok",
                    "message": f"Generated {grid_count}-box Bed Mesh Test pattern ('{svg_filename}').",
                    "filename": svg_filename,
                }
            except Exception as err:
                resp = {"status": "error", "message": str(err)}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif parsed.path == "/api/jog":
            axis = query.get("axis", ["X"])[0].upper()
            dist = float(query.get("dist", ["1.0"])[0])

            try:
                client = get_moonraker_client()
                if axis == "HOME":
                    success = client.send_gcode("G28 X Y")
                    msg = "Homed X and Y gantries."
                else:
                    gcode = f"G91\nG0 {axis}{dist:+.3f} F3000\nG90"
                    success = client.send_gcode(gcode)
                    msg = f"Jogged {axis} {dist:+.1f}mm."

                resp = {"status": "ok" if success else "error", "message": msg}
            except Exception as e:
                resp = {"status": "error", "message": f"Jog failed: {e}"}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif parsed.path == "/api/jog_absolute":
            x = float(query.get("x", ["25"])[0])
            y = float(query.get("y", ["20"])[0])
            z = float(query.get("z", ["0"])[0])

            try:
                client = get_moonraker_client()
                gcode = f"G90\nG0 X{x:.1f} Y{y:.1f} Z{z:.1f} F3000"
                success = client.send_gcode(gcode)
                msg = f"Jogged toolhead to absolute corner X={x:.1f}, Y={y:.1f}, Z={z:.1f} mm."
            except Exception:
                msg = f"[MOCK] Simulated jog to absolute corner X={x:.1f}, Y={y:.1f}, Z={z:.1f} mm."

            resp = {"status": "ok", "message": msg}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif parsed.path == "/api/manual_lock":
            content_length = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_length)
            try:
                cfg = load_config()
                default_nom = cfg.get("nominal_fiducials", [[0.0, 210.0], [170.0, 210.0], [0.0, 20.0]])
                actual_pts = np.array(data.get("actual", []), dtype=np.float64)
                nominal_pts = np.array(data.get("nominal", default_nom), dtype=np.float64)

                if len(actual_pts) < 3:
                    resp = {"status": "error", "message": "Please lock all 3 registration marks before solving alignment."}
                elif cv2 is not None:
                    matrix, inliers = cv2.estimateAffinePartial2D(nominal_pts, actual_pts)
                    if matrix is not None:
                        a, b, tx = matrix[0, 0], matrix[0, 1], matrix[0, 2]
                        scale = math.sqrt(a * a + b * b)
                        angle_rad = math.atan2(b, a)
                        angle_deg = math.degrees(angle_rad)
                        ty = matrix[1, 2]

                        affine_file = os.path.join(WORKSPACE_DIR, "affine_matrix.json")
                        with open(affine_file, "w", encoding="utf-8") as f:
                            json.dump({
                                "matrix": matrix.tolist(),
                                "scale": scale,
                                "rotation_deg": angle_deg,
                                "translation_mm": [tx, ty]
                            }, f, indent=2)

                        msg = f"[MANUAL ALIGNMENT] Solved 2D Affine Matrix! Scale={scale:.4f}, Rotation={angle_deg:+.2f}°, Tx={tx:+.2f}mm, Ty={ty:+.2f}mm."
                        resp = {"status": "ok", "message": msg}
                    else:
                        resp = {"status": "error", "message": "Failed to compute 2D Affine Matrix from manual points."}
                else:
                    resp = {"status": "ok", "message": "Manual points locked."}
            except Exception as err:
                resp = {"status": "error", "message": str(err)}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif parsed.path == "/api/inject_fiducials":
            query = parse_qs(parsed.query)
            filename = query.get("file", ["Cut.svg"])[0]
            filepath = resolve_workspace_file(filename)
            try:
                from src.svg_injector import inject_aruco_fiducials
                res = inject_aruco_fiducials(filepath)
            except Exception as e:
                res = {"status": "error", "message": str(e)}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))

        elif parsed.path == "/api/inject_touch_dots":
            query = parse_qs(parsed.query)
            filename = query.get("file", ["Cut.svg"])[0]
            filepath = resolve_workspace_file(filename)
            try:
                from src.svg_injector import inject_verification_dots
                res = inject_verification_dots(filepath)
            except Exception as e:
                res = {"status": "error", "message": str(e)}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))

        elif parsed.path == "/api/resume":
            try:
                cfg = load_config()
                moonraker_url = cfg.get("moonraker_url", "http://192.168.1.20:7125")
                from src.moonraker_client import MoonrakerClient
                mr = MoonrakerClient(base_url=moonraker_url)
                if not mr.is_connected():
                    resp = {"status": "error", "message": f"Cannot connect to Moonraker at {moonraker_url}."}
                else:
                    # Send RESUME_BASE first to bypass temperature checks; fall back to RESUME
                    success = mr.send_gcode("RESUME_BASE")
                    if not success:
                        success = mr.send_gcode("RESUME")
                    if success:
                        resp = {"status": "ok", "message": "Cut resumed successfully via RESUME_BASE."}
                    else:
                        resp = {"status": "error", "message": "Failed to send resume command to Moonraker."}
            except Exception as e:
                resp = {"status": "error", "message": f"Resume failed: {e}"}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif parsed.path == "/api/cancel":
            try:
                cfg = load_config()
                moonraker_url = cfg.get("moonraker_url", "http://192.168.1.20:7125")
                from src.moonraker_client import MoonrakerClient
                mr = MoonrakerClient(base_url=moonraker_url)
                if not mr.is_connected():
                    resp = {"status": "error", "message": f"Cannot connect to Moonraker at {moonraker_url}."}
                else:
                    success = mr.send_gcode("CANCEL_PRINT")
                    if success:
                        resp = {"status": "ok", "message": "Print job canceled."}
                    else:
                        resp = {"status": "error", "message": "Failed to cancel print job."}
            except Exception as e:
                resp = {"status": "error", "message": f"Cancel failed: {e}"}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif parsed.path == "/api/verify_touch_dots":
            query = parse_qs(parsed.query)
            filename = query.get("file", ["Cut.svg"])[0]
            z_cut_param = query.get("z_cut", ["0.0"])[0]
            try:
                z_cut = float(z_cut_param)
            except ValueError:
                z_cut = 0.0

            filepath = resolve_workspace_file(filename)
            affine_file = os.path.join(WORKSPACE_DIR, "affine_matrix.json")
            if not os.path.isfile(affine_file):
                resp = {"status": "error", "message": "No affine alignment matrix found. Please run Step 3 'Run Autonomous Scan & Align' first."}
            else:
                try:
                    with open(affine_file, "r", encoding="utf-8") as f:
                        af_data = json.load(f)
                    cfg = load_config()

                    # Live sync: verify affine_matrix.json matches current camera offsets
                    cam_off_x = float(cfg.get("camera_offset_x_mm", 0.0))
                    cam_off_y = float(cfg.get("camera_offset_y_mm", 0.0))
                    af_cam_off = af_data.get("camera_offset")
                    if af_cam_off is None or af_cam_off != [cam_off_x, cam_off_y]:
                        recalculate_affine_from_settings(cfg)
                        with open(affine_file, "r", encoding="utf-8") as f:
                            af_data = json.load(f)

                    M = np.array(af_data["matrix"], dtype=np.float64)

                    from src.svg_injector import extract_touch_dots
                    from src.vision_detector import TransformEstimator
                    bed_height = cfg.get("bed_height_mm", 220.0)
                    design_fids = cfg.get("design_fiducials", None)
                    dots_nom, origin_nom = extract_touch_dots(filepath, bed_height_mm=bed_height, design_fiducials=design_fids)

                    dots_arr = np.array(dots_nom, dtype=np.float64)
                    trans_dots = TransformEstimator.transform_points(dots_arr, M)
                    touch_pts_bed = [(float(p[0]), float(p[1])) for p in trans_dots]

                    orig_arr = np.array([[origin_nom[0], origin_nom[1]]], dtype=np.float64)
                    trans_orig = TransformEstimator.transform_points(orig_arr, M)
                    paper_origin_bed = (float(trans_orig[0][0]), float(trans_orig[0][1]))

                    moonraker_url = cfg.get("moonraker_url", "http://192.168.1.20:7125")
                    z_safe = cfg.get("z_safe_mm", 5.0)
                    f_travel = cfg.get("f_travel_mm_min", 3000.0)
                    f_plunge = 127.0

                    from src.moonraker_client import MoonrakerClient
                    mr = MoonrakerClient(base_url=moonraker_url)
                    if not mr.is_connected():
                        resp = {"status": "error", "message": f"Cannot connect to Moonraker at {moonraker_url}."}
                    else:
                        safe_z = max(float(z_safe), 5.0)
                        mr.jog_to(z=safe_z, feedrate=f_travel, wait_finish=True)
                        for dx, dy in touch_pts_bed:
                            mr.jog_to(x=dx, y=dy, z=safe_z, feedrate=f_travel, wait_finish=True)
                            mr.jog_to(z=z_cut, feedrate=f_plunge, wait_finish=True)
                            time.sleep(1.5)
                            mr.jog_to(z=safe_z, feedrate=f_travel, wait_finish=True)

                        mr.jog_to(x=paper_origin_bed[0], y=paper_origin_bed[1], z=safe_z, feedrate=f_travel, wait_finish=True)
                        mr.jog_to(z=1.0, feedrate=1500.0, wait_finish=True)
                        time.sleep(2.0)
                        mr.jog_to(z=safe_z, feedrate=f_travel, wait_finish=True)

                        resp = {
                            "status": "ok",
                            "message": f"Blade touched 3 verification dots and origin, then lifted to safe height ({safe_z:.1f}mm).",
                            "dots_bed": touch_pts_bed,
                            "paper_origin_bed": paper_origin_bed
                        }
                except Exception as e:
                    resp = {"status": "error", "message": f"Verification move failed: {e}"}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif parsed.path == "/api/scan" or parsed.path == "/api/cut":
            is_scan = parsed.path == "/api/scan"
            filename = query.get("file", ["Cut1.svg"])[0]
            cam_source = query.get("camera", ["crowsnest"])[0]
            dry_flag = query.get("dry", [""])[0]
            force_home = query.get("force_home", ["0"])[0]
            search_mode = query.get("search_mode", ["auto"])[0]

            cmd = [PYTHON_EXE, "vision_cutter.py", "--svg", filename]

            if is_scan:
                cmd.append("--no-upload")
            else:
                # /api/cut: use cached alignment from Step 3 if available, and trigger print
                affine_file = os.path.join(WORKSPACE_DIR, "affine_matrix.json")
                if os.path.isfile(affine_file):
                    try:
                        with open(affine_file, "r", encoding="utf-8") as f:
                            af_data = json.load(f)
                        # Reuse if for same file or generated within last hour
                        if af_data.get("file") == filename or (time.time() - af_data.get("timestamp", 0)) < 3600:
                            cmd.append("--use-cached-alignment")
                    except Exception:
                        pass
                cmd.append("--auto-start")

            if force_home == "1" or force_home == "true":
                cmd.append("--force-home")

            cmd.extend(["--search-mode", search_mode])

            cfg = load_config()
            settle_secs = cfg.get("camera_settle_seconds", 2.0)
            cmd.extend(["--settle-seconds", str(settle_secs)])
            cmd.extend(["--camera-offset-x", str(cfg.get("camera_offset_x_mm", 0.0))])
            cmd.extend(["--camera-offset-y", str(cfg.get("camera_offset_y_mm", 0.0))])
            moonraker_url = cfg.get("moonraker_url", "http://localhost:7125")
            crowsnest_url = cfg.get("crowsnest_url", "http://localhost:8080/?action=stream")

            cmd.extend(["--moonraker-url", moonraker_url])
            cmd.extend(["--crowsnest-url", crowsnest_url])

            if cam_source.isdigit():
                cmd.extend(["--camera-index", cam_source])
            elif cam_source == "mock":
                cmd.append("--mock")

            z_scan = cfg.get("z_scan_mm", 20.0)
            cmd.extend(["--z-scan", str(z_scan)])

            z_lift_swivel = cfg.get("z_lift_swivel_mm", 0.0)
            if z_lift_swivel > 0.0:
                cmd.extend(["--z-lift-swivel", str(z_lift_swivel)])

            z_cut_param = query.get("z_cut", [""])[0]
            if z_cut_param != "":
                try:
                    cmd.extend(["--z-cut", str(float(z_cut_param))])
                    cfg["z_cut_mm"] = float(z_cut_param)
                    save_config(cfg)
                except ValueError:
                    pass

            verify_touch = query.get("verify_touch", ["1"])[0]
            if not is_scan and (verify_touch == "1" or verify_touch == "true"):
                cmd.append("--verify-touch")

            if dry_flag:
                cmd.append(dry_flag)

            try:
                res = subprocess.run(cmd, cwd=WORKSPACE_DIR, capture_output=True, text=True, timeout=180)
                stdout_text = res.stdout.strip()
                stderr_text = res.stderr.strip()
                combined_msg = stdout_text or stderr_text or ("Pipeline executed successfully." if res.returncode == 0 else "Pipeline execution failed.")

                resp = {
                    "status": "ok" if res.returncode == 0 else "error",
                    "message": combined_msg,
                }
            except Exception as err:
                resp = {"status": "error", "message": str(err)}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_error(404, "Unknown Endpoint")


def main():
    parser = argparse.ArgumentParser(description="Start Cutter Studio Web GUI Server.")
    parser.add_argument("--port", type=int, default=5000, help="Web server port (default: 5000).")
    args = parser.parse_args()

    server_address = ("", args.port)
    httpd = ThreadingHTTPServer(server_address, WebRequestHandler)
    print("======================================================", flush=True)
    print("Ender 3 Cutter Studio Web Server Started!", flush=True)
    print(f"Open in your browser: http://localhost:{args.port}", flush=True)
    print("======================================================", flush=True)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Web Server.", flush=True)
        httpd.server_close()


if __name__ == "__main__":
    main()
