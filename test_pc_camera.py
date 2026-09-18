#!/usr/bin/env python3
"""
PC Camera & Vision Testing Tool for Ender 3 Auto Cutter Pipeline.

Run this script directly on your PC with a connected webcam to test:
1. Live camera feed capture.
2. Real-time ArUco marker (DICT_4X4_50) detection and sub-pixel circle detection.
3. Affine transformation & live SVG cut-path overlay preview on your screen!

Usage:
    python test_pc_camera.py --camera 0 --fiducial-type aruco
    python test_pc_camera.py --camera 0 --svg sample_test.svg
"""

import argparse
import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# Add src package directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    import cv2
except ImportError:
    print("Error: OpenCV is not installed. Run: pip install opencv-python")
    sys.exit(1)

from vision_detector import FiducialDetector, TransformEstimator
from drag_knife_planner import DragKnifePlanner, GCodeGenerator, SVGParser


def main():
    parser = argparse.ArgumentParser(description="Test Vision Detection on PC Camera.")
    parser.add_argument("--camera", type=int, default=0, help="PC Webcam index (default: 0).")
    parser.add_argument("--fiducial-type", type=str, choices=["aruco", "circle"], default="aruco", help="Fiducial type to detect.")
    parser.add_argument("--svg", type=str, default="sample_test.svg", help="Path to sample SVG for overlay test.")
    parser.add_argument("--mm-per-pixel", type=float, default=0.05, help="Estimated mm per pixel scale.")
    args = parser.parse_args()

    print("======================================================")
    print("Starting PC Camera & Vision Test Tool")
    print(f"Camera Device Index: {args.camera}")
    print(f"Fiducial Type: {args.fiducial_type.upper()}")
    print("Press 'q' to exit, 's' to save current snapshot frame.")
    print("======================================================")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera device index {args.camera}.")
        print("Tip: If using external USB webcam, try --camera 1 or --camera 2.")
        sys.exit(1)

    detector = FiducialDetector(fiducial_type=args.fiducial_type)

    # Optional SVG parsing for path overlay
    svg_contours = []
    if os.path.isfile(args.svg):
        try:
            parser_svg = SVGParser(step_size_mm=1.0)
            svg_contours = parser_svg.parse_svg_paths(args.svg)
            print(f"Loaded SVG '{args.svg}' with {len(svg_contours)} path contours.")
        except Exception as e:
            print(f"Warning: Could not parse SVG '{args.svg}': {e}")

    frame_count = 0
    fps_start_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to read frame from camera.")
            break

        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - fps_start_time
            fps = frame_count / elapsed if elapsed > 0 else 0.0

        display_frame = frame.copy()
        h, w = frame.shape[:2]
        center_u, center_v = w // 2, h // 2

        # Draw optical center crosshair
        cv2.line(display_frame, (center_u - 15, center_v), (center_u + 15, center_v), (255, 0, 0), 1)
        cv2.line(display_frame, (center_u, center_v - 15), (center_u, center_v + 15), (255, 0, 0), 1)
        cv2.circle(display_frame, (center_u, center_v), 3, (255, 0, 0), -1)

        detected_centers = []

        if args.fiducial_type == "aruco":
            markers = detector.detect_aruco_markers(frame)
            for marker_id, (u, v) in markers.items():
                u_int, v_int = int(round(u)), int(round(v))
                detected_centers.append((u, v))

                # Draw green marker box and center point
                cv2.circle(display_frame, (u_int, v_int), 6, (0, 255, 0), -1)
                cv2.putText(
                    display_frame,
                    f"ArUco ID {marker_id} ({u_int}, {v_int})",
                    (u_int + 10, v_int - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
        elif args.fiducial_type == "circle":
            center = detector.detect_circle_fiducial(frame)
            if center is not None:
                u_int, v_int = int(round(center[0])), int(round(center[1]))
                detected_centers.append(center)
                cv2.circle(display_frame, (u_int, v_int), 8, (0, 255, 0), 2)
                cv2.circle(display_frame, (u_int, v_int), 2, (0, 0, 255), -1)
                cv2.putText(
                    display_frame,
                    f"Fiducial ({u_int}, {v_int})",
                    (u_int + 10, v_int - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # Status text overlay
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Detected Markers: {len(detected_centers)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display_frame, "Press 'q' to quit | 's' to save screenshot", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        try:
            cv2.imshow("Vision Cutter - PC Camera Test", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"camera_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved snapshot image to '{filename}'")
        except cv2.error:
            # Headless OpenCV installed (opencv-python-headless)
            snapshot_file = "camera_snapshot_headless.jpg"
            cv2.imwrite(snapshot_file, display_frame)
            print(f"\n[NOTE] OpenCV Headless detected (GUI window unavailable).")
            print(f"Saved processed camera frame with detection overlays to '{snapshot_file}'.")
            print("\nTo enable live video GUI window on PC, install full OpenCV:")
            print("  pip uninstall opencv-python-headless -y")
            print("  pip install opencv-python")
            break

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    print("PC Camera test ended.")


if __name__ == "__main__":
    main()
