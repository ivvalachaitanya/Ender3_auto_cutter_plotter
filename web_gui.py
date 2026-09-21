#!/usr/bin/env python3
"""
Web-Based Cricut-Style Web GUI Server for Ender 3 Auto Cutter Pipeline.

Runs a local Web Application Server on http://localhost:5000 using ThreadingHTTPServer.
Provides:
1. Load SVG & preview cut paths / fiducials.
2. One-click 1:1 scale printing to paper/vinyl color printer.
3. Live Camera Feed & Interactive ArUco Fiducial Scanning window.
4. Camera selection dropdown (USB cameras, Klipper Crowsnest stream, Mock stream).
5. Hardware Calibration Settings Modal (Bed size, Blade offset, Camera offsets, Z depths, Feedrates).
6. One-click G-code upload & Vinyl Cut execution on Klipper/Moonraker.

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
CONFIG_PATH = os.path.join(WORKSPACE_DIR, "config.json")
sys.path.insert(0, os.path.join(WORKSPACE_DIR, "src"))

DEFAULT_CONFIG = {
    "bed_width_mm": 210.0,
    "bed_height_mm": 297.0,
    "blade_offset_mm": 0.35,
    "camera_offset_x_mm": -25.0,
    "camera_offset_y_mm": 0.0,
    "mm_per_pixel": 0.05,
    "z_safe_mm": 5.0,
    "z_cut_mm": 0.0,
    "f_cut_mm_min": 1200.0,
    "f_travel_mm_min": 3000.0,
    "moonraker_url": "http://localhost:7125",
    "crowsnest_url": "http://localhost:8080/?action=stream",
}


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


def save_config(new_config):
    """Save configuration dictionary to config.json."""
    config = load_config()
    config.update(new_config)
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print("[ERROR] Failed to save config.json:", e)
        return False


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

        cfg = load_config()
        crowsnest_url = cfg.get("crowsnest_url", "http://localhost:8080/?action=stream")
        cameras.append({
            "id": "crowsnest",
            "name": f"Klipper Crowsnest Stream ({crowsnest_url})",
            "type": "mjpeg",
            "url": crowsnest_url
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
      <div class="flex items-center space-x-3">
        <span class="px-3 py-1 rounded-full text-xs bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">● Klipper Online (7125)</span>
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

  <!-- Settings Calibration Modal -->
  <div id="settings-modal" class="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4 hidden">
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

  <script>
    function addLog(msg) {
      const box = document.getElementById('log-box');
      box.innerHTML += `<div>[${new Date().toLocaleTimeString()}] ${msg}</div>`;
      box.scrollTop = box.scrollHeight;
    }

    function openSettingsModal() {
      fetch('/api/settings')
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
            document.getElementById('cfg-z-safe').value = cfg.z_safe_mm;
            document.getElementById('cfg-z-cut').value = cfg.z_cut_mm;
            document.getElementById('cfg-f-cut').value = cfg.f_cut_mm_min;
            document.getElementById('cfg-f-travel').value = cfg.f_travel_mm_min;
            document.getElementById('cfg-moonraker-url').value = cfg.moonraker_url;
            document.getElementById('cfg-crowsnest-url').value = cfg.crowsnest_url;
            document.getElementById('settings-modal').classList.remove('hidden');
          }
        });
    }

    function closeSettingsModal() {
      document.getElementById('settings-modal').classList.add('hidden');
    }

    function saveSettings() {
      const newCfg = {
        bed_width_mm: parseFloat(document.getElementById('cfg-bed-width').value),
        bed_height_mm: parseFloat(document.getElementById('cfg-bed-height').value),
        blade_offset_mm: parseFloat(document.getElementById('cfg-blade-offset').value),
        camera_offset_x_mm: parseFloat(document.getElementById('cfg-cam-offset-x').value),
        camera_offset_y_mm: parseFloat(document.getElementById('cfg-cam-offset-y').value),
        mm_per_pixel: parseFloat(document.getElementById('cfg-mm-px').value),
        z_safe_mm: parseFloat(document.getElementById('cfg-z-safe').value),
        z_cut_mm: parseFloat(document.getElementById('cfg-z-cut').value),
        f_cut_mm_min: parseFloat(document.getElementById('cfg-f-cut').value),
        f_travel_mm_min: parseFloat(document.getElementById('cfg-f-travel').value),
        moonraker_url: document.getElementById('cfg-moonraker-url').value,
        crowsnest_url: document.getElementById('cfg-crowsnest-url').value,
      };

      fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newCfg)
      })
      .then(r => r.json())
      .then(data => {
        if (data.status === 'ok') {
          addLog('[SETTINGS] Calibration parameters updated and saved to config.json!');
          closeSettingsModal();
          loadPreview();
        } else {
          addLog('[ERROR] Failed to save settings: ' + data.message);
        }
      });
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
    """HTTP Handler for Cricut Web GUI with MJPEG Camera Streaming and Settings Management."""

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
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

                    cfg = load_config()
                    bed_w = cfg.get("bed_width_mm", 210.0)
                    bed_h = cfg.get("bed_height_mm", 297.0)

                    preview_html = f'''<div style="width:100%; height:100%; display:flex; flex-direction:column; justify-content:center; align-items:center; padding:2px;">
                      <div style="font-size:11px; font-family:monospace; color:#38bdf8; margin-bottom:4px; font-weight:600;">
                        🛏️ Printable Bed Canvas: {bed_w:.0f}mm × {bed_h:.0f}mm
                      </div>
                      <div style="aspect-ratio:{bed_w}/{bed_h}; max-width:95%; max-height:85%; background:#ffffff; border:2px solid #0284c7; border-radius:6px; box-shadow:0 4px 12px rgba(0,0,0,0.5); overflow:hidden; display:flex; justify-content:center; align-items:center; padding:2px;">
                        {svg_code_sanitized}
                      </div>
                    </div>'''

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

        if parsed.path == "/api/settings":
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

        elif parsed.path == "/api/scan" or parsed.path == "/api/cut":
            filename = query.get("file", ["Cut1.svg"])[0]
            cam_source = query.get("camera", ["0"])[0]
            dry_flag = query.get("dry", [""])[0]

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
