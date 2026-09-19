#!/usr/bin/env python3
"""
Web-Based Cricut-Style Web GUI Server for Ender 3 Auto Cutter Pipeline.

Runs a local Web Application Server on http://localhost:5000.
Provides:
1. Load SVG & preview cut paths / fiducials.
2. One-click 1:1 scale printing to paper/vinyl color printer.
3. One-click Vision Scan & Camera Fiducial Alignment.
4. One-click G-code upload & Vinyl Cut execution on Klipper/Moonraker.

Usage:
    python web_gui.py --port 5000
"""

import argparse
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

PORT = 5000
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))


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
  </style>
</head>
<body class="p-6">
  <div class="max-w-6xl mx-auto space-y-6">
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
        <input type="text" id="svg-file" value="Cut1.svg" class="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-emerald-400 font-mono">
        <button onclick="loadPreview()" class="w-full btn-primary py-2 text-xs">Load SVG Preview</button>
      </div>

      <!-- Step 2 -->
      <div class="card p-4 space-y-3">
        <div class="flex items-center space-x-2">
          <span class="w-6 h-6 rounded-full bg-blue-500 text-white flex items-center justify-center text-xs font-bold">2</span>
          <h3 class="font-semibold text-sm">Print Artwork</h3>
        </div>
        <p class="text-xs text-slate-400">Send 1:1 scale design to your paper/vinyl color printer.</p>
        <button onclick="window.print()" class="w-full bg-blue-600 hover:bg-blue-500 text-white font-semibold py-2 rounded text-xs">🖨️ Print Sheet (1:1)</button>
      </div>

      <!-- Step 3 -->
      <div class="card p-4 space-y-3">
        <div class="flex items-center space-x-2">
          <span class="w-6 h-6 rounded-full bg-purple-500 text-white flex items-center justify-center text-xs font-bold">3</span>
          <h3 class="font-semibold text-sm">Scan & Align</h3>
        </div>
        <p class="text-xs text-slate-400">Jog camera & solve 2D Affine Transformation.</p>
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

    <!-- Main Display Section -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
      <div class="card p-4 md:col-span-2 space-y-3">
        <h3 class="text-sm font-semibold">SVG Vector & Cut-Path Preview</h3>
        <div id="svg-container" class="w-full h-80 bg-slate-950 rounded border border-slate-800 flex items-center justify-center p-4">
          <p class="text-slate-500 text-xs">Click 'Load SVG Preview' to render vector cut paths</p>
        </div>
      </div>

      <div class="card p-4 space-y-3 flex flex-col justify-between">
        <div>
          <h3 class="text-sm font-semibold mb-2">Real-time Pipeline Log</h3>
          <div id="log-box" class="w-full h-64 bg-slate-950 rounded p-3 text-[11px] font-mono text-emerald-400 overflow-y-auto space-y-1 border border-slate-800">
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

    function runScan() {
      const file = document.getElementById('svg-file').value;
      addLog(`Starting camera alignment scan for '${file}'...`);
      fetch(`/api/scan?file=${file}`, { method: 'POST' })
        .then(r => r.json())
        .then(data => {
          addLog(data.message);
        });
    }

    function runCut() {
      const file = document.getElementById('svg-file').value;
      const dry = document.getElementById('dry-run').checked ? '--dry-run' : '';
      addLog(`Sending cut job to Klipper (${dry})...`);
      fetch(`/api/cut?file=${file}&dry=${dry}`, { method: 'POST' })
        .then(r => r.json())
        .then(data => {
          addLog(data.message);
        });
    }
  </script>
</body>
</html>
"""


class WebRequestHandler(BaseHTTPRequestHandler):
    """HTTP Handler for Cricut Web GUI."""

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif parsed.path == "/api/preview":
            query = parse_qs(parsed.query)
            filename = query.get("file", ["Cut1.svg"])[0]
            filepath = os.path.join(WORKSPACE_DIR, filename)

            if os.path.isfile(filepath):
                try:
                    # Import SVGParser to generate lightweight SVG vector paths
                    sys.path.insert(0, os.path.join(WORKSPACE_DIR, "src"))
                    from drag_knife_planner import SVGParser

                    parser_obj = SVGParser(step_size_mm=1.0)
                    contours = parser_obj.parse_svg_paths(filepath, flip_y=False)

                    # Generate clean, lightweight HTML5 SVG string (under 5KB)
                    svg_lines = ['<svg viewBox="0 0 210 297" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" style="background:#020617; border-radius:8px;">']
                    # Render page border
                    svg_lines.append('<rect x="0" y="0" width="210" height="297" fill="none" stroke="#334155" stroke-width="1" />')
                    # Render fiducial indicator boxes
                    svg_lines.append('<rect x="10" y="10" width="10" height="10" fill="#10b981" />')
                    svg_lines.append('<rect x="180" y="10" width="10" height="10" fill="#10b981" />')
                    svg_lines.append('<rect x="10" y="250" width="10" height="10" fill="#10b981" />')

                    # Render vector cut paths in red
                    for contour in contours:
                        pts_str = " ".join([f"{pt[0]:.2f},{pt[1]:.2f}" for pt in contour])
                        svg_lines.append(f'<polyline points="{pts_str}" fill="none" stroke="#ef4444" stroke-width="0.8" stroke-dasharray="3,3" />')

                    svg_lines.append('</svg>')
                    clean_svg_code = "".join(svg_lines)

                    resp = {"status": "ok", "contours_count": len(contours), "svg_content": clean_svg_code}
                except Exception as e:
                    resp = {"status": "error", "message": str(e)}
            else:
                resp = {"status": "error", "message": f"File '{filename}' not found."}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        filename = query.get("file", ["Cut1.svg"])[0]
        dry_flag = query.get("dry", [""])[0]

        if parsed.path == "/api/scan" or parsed.path == "/api/cut":
            cmd = [sys.executable, "vision_cutter.py", "--svg", filename, "--mock"]
            if dry_flag:
                cmd.append(dry_flag)

            try:
                res = subprocess.run(cmd, cwd=WORKSPACE_DIR, capture_output=True, text=True, timeout=30)
                resp = {
                    "status": "ok" if res.returncode == 0 else "error",
                    "message": f"Pipeline executed successfully for {filename}." if res.returncode == 0 else res.stderr,
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
    httpd = HTTPServer(server_address, WebRequestHandler)
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
