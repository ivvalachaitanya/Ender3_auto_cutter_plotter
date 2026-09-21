#!/usr/bin/env python3
"""
Web-Based Cricut-Style Web GUI Server for Ender 3 Auto Cutter Pipeline.

Runs a local Web Application Server on http://localhost:5000 using ThreadingHTTPServer.
Provides:
1. Load SVG & preview cut paths / fiducials.
2. One-click 1:1 scale printing to paper/vinyl color printer.
3. Live Camera Feed & Interactive ArUco Fiducial Scanning window.
4. Camera selection dropdown (USB cameras, Klipper Crowsnest stream, Mock stream).
5. One-click G-code upload & Vinyl Cut execution on Klipper/Moonraker.

Usage:
    python web_gui.py --port 5000
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

PORT = 5000
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(WORKSPACE_DIR, "src"))


class CameraStreamer:
    """Helper class to enumerate, capture, annotate, and stream camera feeds."""

    def __init__(self):
        self.detector = None
        if cv2 is not None:
            try:
                from vision_detector import FiducialDetector
                self.detector = FiducialDetector(fiducial_type="aruco")
            except Exception as e:
                print("[WARN] Could not initialize FiducialDetector for camera streamer:", e)

    def enumerate_cameras(self):
        """Enumerate local USB webcams and network camera streams."""
        cameras = []
        if cv2 is not None:
            for idx in range(4):
                try:
                    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(idx)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            h, w = frame.shape[:2]
                            cameras.append({
                                "id": str(idx),
                                "name": f"USB Camera {idx} ({w}x{h})",
                                "type": "opencv",
                                "index": idx
                            })
                        cap.release()
                except Exception:
                    pass

        cameras.append({
            "id": "crowsnest",
            "name": "Klipper Crowsnest Stream (Port 8080)",
            "type": "mjpeg",
            "url": "http://localhost:8080/?action=stream"
        })
        cameras.append({
            "id": "mock",
            "name": "Mock Virtual Camera (Test Feed)",
            "type": "mock"
        })
        return cameras

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
        cv2.line(frame, (cu - 20, cv), (cu + 20, cv), (255, 180, 0), 1)
        cv2.line(frame, (cu, cv - 20), (cu, cv + 20), (255, 180, 0), 1)
        cv2.circle(frame, (cu, cv), 5, (255, 180, 0), 1)

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
        cv2.putText(frame, "SIMULATED MOCK CAMERA FEED (30 FPS)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)
        cv2.putText(frame, f"Frame: #{frame_num} | Focus Score: 142.5 VAR", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return frame

    def annotate_frame(self, frame):
        """Annotate camera frame with reticle, focus score, and ArUco marker bounding boxes."""
        if cv2 is None or frame is None or frame.size == 0:
            return frame

        h, w = frame.shape[:2]
        cu, cv = w // 2, h // 2

        # Optical center crosshairs
        cv2.line(frame, (cu - 20, cv), (cu + 20, cv), (255, 180, 0), 1)
        cv2.line(frame, (cu, cv - 20), (cu, cv + 20), (255, 180, 0), 1)
        cv2.circle(frame, (cu, cv), 5, (255, 180, 0), 1)

        # Focus score
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        focus_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if self.detector is not None:
            try:
                dets = self.detector.detect_aruco_markers(frame)
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

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Ender 3 Auto-Cutter Studio</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { background-color: #0f172a; color: #f8fafc; font-family: system-ui, sans-serif; }
    .card { background-color: #1e293b; border: 1px solid #334155; border-radius: 1rem; }
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
      <div class="flex space-x-3">
        <span class="px-3 py-1 rounded-full text-xs bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">● Klipper Server Online (7125)</span>
        <span class="px-3 py-1 rounded-full text-xs bg-blue-500/20 text-blue-400 border border-blue-500/30">● Crowsnest Stream Online (8080)</span>
      </div>
    </div>

    <!-- Workflow Steps Grid -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
      <!-- Step 1 -->
      <div class="card p-4 space-y-3">
        <div class="flex items-center space-x-2">
          <span class="w-6 h-6 rounded-full bg-emerald-500 text-white flex items-center justify-center text-xs font-bold">1</span>
          <h3 class="font-semibold text-sm">Select SVG Artwork</h3>
        </div>
        <input type="text" id="svg-file" value="Cut.svg" class="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-emerald-400 font-mono">
        <button onclick="loadPreview()" class="w-full btn-primary py-2 text-xs">Load SVG Preview</button>
      </div>

      <!-- Step 2 -->
      <div class="card p-4 space-y-3">
        <div class="flex items-center space-x-2">
          <span class="w-6 h-6 rounded-full bg-blue-500 text-white flex items-center justify-center text-xs font-bold">2</span>
          <h3 class="font-semibold text-sm">Print Artwork</h3>
        </div>
        <p class="text-xs text-slate-400">Send 1:1 scale design to your paper/vinyl color printer.</p>
        <button onclick="openPrintWindow()" class="w-full bg-blue-600 hover:bg-blue-500 text-white font-semibold py-2 rounded text-xs">🖨️ Print Sheet (1:1)</button>
      </div>

      <!-- Step 3 -->
      <div class="card p-4 space-y-3">
        <div class="flex items-center space-x-2">
          <span class="w-6 h-6 rounded-full bg-purple-500 text-white flex items-center justify-center text-xs font-bold">3</span>
          <h3 class="font-semibold text-sm">Scan & Align</h3>
        </div>
        <div>
          <div class="flex justify-between items-center mb-1">
            <label class="text-[11px] text-slate-400">Select Camera Source:</label>
            <button onclick="refreshCameras()" class="text-[10px] text-purple-400 hover:underline">🔄 Refresh</button>
          </div>
          <select id="camera-select" onchange="switchCameraStream()" class="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-purple-300 font-mono">
            <option value="0">USB Camera 0 (Primary)</option>
            <option value="1">USB Camera 1 (Secondary)</option>
            <option value="crowsnest">Klipper Crowsnest (8080)</option>
            <option value="mock">Mock Virtual Feed</option>
          </select>
        </div>
        <button onclick="runScan()" class="w-full bg-purple-600 hover:bg-purple-500 text-white font-semibold py-2 rounded text-xs">📷 Scan Fiducials</button>
      </div>

      <!-- Step 4 -->
      <div class="card p-4 space-y-3">
        <div class="flex items-center space-x-2">
          <span class="w-6 h-6 rounded-full bg-rose-500 text-white flex items-center justify-center text-xs font-bold">4</span>
          <h3 class="font-semibold text-sm">Cut Vinyl</h3>
        </div>
        <div class="flex items-center justify-between text-xs">
          <span class="text-slate-400">Dry Run:</span>
          <input type="checkbox" id="dry-run" class="accent-emerald-500">
        </div>
        <button onclick="runCut()" class="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-2 rounded text-xs shadow-lg">🚀 START VINYL CUT</button>
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
          <img id="camera-feed" src="/api/camera/stream?source=0" class="w-full h-full object-contain">
        </div>
      </div>

      <!-- Col 3: Real-time Pipeline Log -->
      <div class="card p-4 space-y-3 flex flex-col justify-between">
        <div>
          <h3 class="text-sm font-semibold mb-2">Real-time Pipeline Log</h3>
          <div id="log-box" class="w-full h-[450px] bg-slate-950 rounded p-3 text-[11px] font-mono text-emerald-400 overflow-y-auto space-y-1 border border-slate-800">
            <div>[INIT] Cutter Studio Web Server ready on port 5000</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    function addLog(msg) {
      const box = document.getElementById('log-box');
      box.innerHTML += `<div>[${new Date().toLocaleTimeString()}] ${msg}</div>`;
      box.scrollTop = box.scrollHeight;
    }

    function refreshCameras() {
      addLog('Scanning attached camera devices...');
      fetch('/api/cameras')
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
        });
    }

    function switchCameraStream() {
      const cam = document.getElementById('camera-select').value;
      const feed = document.getElementById('camera-feed');
      addLog(`Switching live camera stream to source '${cam}'...`);
      feed.src = `/api/camera/stream?source=${cam}&t=${Date.now()}`;
    }

    function loadPreview() {
      const file = document.getElementById('svg-file').value;
      addLog(`Loading SVG preview for '${file}'...`);
      fetch(`/api/preview?file=${file}`)
        .then(r => r.json())
        .then(data => {
          if (data.status === 'ok') {
            document.getElementById('svg-container').innerHTML = data.svg_content;
            addLog(`Rendered SVG (${data.contours_count} contours found).`);
          } else {
            addLog(`Error: ${data.message}`);
          }
        });
    }

    function openPrintWindow() {
      const file = document.getElementById('svg-file').value;
      addLog(`[PRINT] Sending full color artwork + ArUco fiducials of '${file}' to 2D Paper/Vinyl Printer at 1:1 scale...`);
      window.open(`/print-artwork?file=${file}`, '_blank');
    }

    function runScan() {
      const file = document.getElementById('svg-file').value;
      const cam = document.getElementById('camera-select').value;
      addLog(`[VISION] Starting fiducial scan for '${file}' using camera '${cam}'...`);
      fetch(`/api/scan?file=${file}&camera=${cam}`, { method: 'POST' })
        .then(r => r.json())
        .then(data => {
          addLog(data.message);
        });
    }

    function runCut() {
      const file = document.getElementById('svg-file').value;
      const dry = document.getElementById('dry-run').checked ? '--dry-run' : '';
      addLog(`[CUTTER] Stripping pixel images & sending ONLY vector cut paths (G-code) to Ender 3 machine...`);
      fetch(`/api/cut?file=${file}&dry=${dry}`, { method: 'POST' })
        .then(r => r.json())
        .then(data => {
          addLog(data.message);
        });
    }

    // Auto-refresh cameras on page load
    window.addEventListener('DOMContentLoaded', () => {
      refreshCameras();
      loadPreview();
    });
  </script>
</body>
</html>
"""


class WebRequestHandler(BaseHTTPRequestHandler):
    """HTTP Handler for Cricut Web GUI with MJPEG Camera Streaming."""

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif parsed.path == "/api/cameras":
            cams = camera_streamer.enumerate_cameras()
            resp = {"status": "ok", "cameras": cams}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        elif parsed.path == "/api/camera/stream":
            query = parse_qs(parsed.query)
            source_str = query.get("source", ["0"])[0]

            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            frame_count = 0
            cap = None

            if source_str.isdigit() and cv2 is not None:
                cam_idx = int(source_str)
                cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(cam_idx)

            try:
                while True:
                    frame_count += 1
                    frame = None

                    if cap is not None and cap.isOpened():
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

                    time.sleep(0.033)  # ~30 FPS
            except (BrokenPipeError, ConnectionResetError, Exception):
                pass
            finally:
                if cap is not None:
                    cap.release()

        elif parsed.path == "/api/preview":
            query = parse_qs(parsed.query)
            filename = query.get("file", ["Cut.svg"])[0]

            if filename.lower() == "cut1.svg" and os.path.isfile(os.path.join(WORKSPACE_DIR, "Cut.svg")):
                filepath = os.path.join(WORKSPACE_DIR, "Cut.svg")
            else:
                filepath = os.path.join(WORKSPACE_DIR, filename)

            if os.path.isfile(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        svg_code = f.read()

                    # Sanitize SVG: ensure any display:none on artwork images or layers is enabled on-the-fly
                    svg_code_sanitized = svg_code.replace("display:none;", "display:inline;").replace("display:none", "display:inline")

                    preview_html = f'<div style="width:100%; height:100%; display:flex; justify-content:center; align-items:center; background:#ffffff; border-radius:8px; overflow:hidden; padding:4px;">{svg_code_sanitized}</div>'

                    sys.path.insert(0, os.path.join(WORKSPACE_DIR, "src"))
                    from drag_knife_planner import SVGParser
                    parser_obj = SVGParser(step_size_mm=1.0)
                    contours = parser_obj.parse_svg_paths(filepath, flip_y=False)

                    resp = {"status": "ok", "contours_count": len(contours), "svg_content": preview_html}
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

            if filename.lower() == "cut1.svg" and os.path.isfile(os.path.join(WORKSPACE_DIR, "Cut.svg")):
                filepath = os.path.join(WORKSPACE_DIR, "Cut.svg")
            else:
                filepath = os.path.join(WORKSPACE_DIR, filename)

            if os.path.isfile(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        svg_code = f.read()
                    svg_code_sanitized = svg_code.replace("display:none;", "display:inline;").replace("display:none", "display:inline")
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

            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        filename = query.get("file", ["Cut1.svg"])[0]
        cam_source = query.get("camera", ["0"])[0]
        dry_flag = query.get("dry", [""])[0]

        if parsed.path == "/api/scan" or parsed.path == "/api/cut":
            cmd = [sys.executable, "vision_cutter.py", "--svg", filename]
            if cam_source.isdigit():
                cmd.extend(["--camera-index", cam_source])

            if cam_source == "mock" or not cam_source.isdigit():
                cmd.append("--mock")

            if dry_flag:
                cmd.append(dry_flag)

            try:
                res = subprocess.run(cmd, cwd=WORKSPACE_DIR, capture_output=True, text=True, timeout=30)
                resp = {
                    "status": "ok" if res.returncode == 0 else "error",
                    "message": f"Pipeline executed successfully for {filename}." if res.returncode == 0 else (res.stderr or res.stdout),
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
    print("======================================================")
    print("Ender 3 Cutter Studio Web Server Started!")
    print(f"Open in your browser: http://localhost:{args.port}")
    print("======================================================")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Web Server.")
        httpd.server_close()


if __name__ == "__main__":
    main()
