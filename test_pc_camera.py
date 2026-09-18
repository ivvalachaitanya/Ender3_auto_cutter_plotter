#!/usr/bin/env python3
"""
PC Camera & Vision Testing Tool with Focus Control & Pixel Coordinate Capture.

Features:
1. Live camera feed capture with sub-pixel fiducial detection (ArUco or Circle).
2. Save annotated screenshots rendered with exact (u, v) pixel coordinates + JSON metadata sidecar!
3. Hardware camera focus control (autofocus lock, manual focus adjustment).

Usage:
    python test_pc_camera.py --camera 0 --fiducial-type aruco
    python test_pc_camera.py --camera 0 --auto-lock-focus
    python test_pc_camera.py --camera 0 --manual-focus 50
"""

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Test Vision Detection on PC Camera with Focus Control.")
    parser.add_argument("--camera", type=int, default=0, help="PC Webcam index (default: 0).")
    parser.add_argument("--fiducial-type", type=str, choices=["aruco", "circle"], default="aruco", help="Fiducial type to detect.")
    parser.add_argument("--svg", type=str, default="sample_test.svg", help="Path to sample SVG for overlay test.")
    parser.add_argument("--auto-lock-focus", action="store_true", help="Automatically lock camera focus when a fiducial is found.")
    parser.add_argument("--manual-focus", type=float, default=None, help="Set manual focus value (0 to 255).")
    args = parser.parse_args()

    print("======================================================")
    print("Starting PC Camera & Vision Test Tool")
    print(f"Camera Index: {args.camera} | Fiducial Type: {args.fiducial_type.upper()}")
    print("Controls:")
    print("  's' : Save screenshot image + JSON pixel coordinates")
    print("  'f' : Toggle Auto-Focus / Lock Focus")
    print("  '[' / ']' : Decrease / Increase Manual Focus")
    print("  'q' : Quit")
    print("======================================================")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera device index {args.camera}.")
        sys.exit(1)

    # Initial focus configuration
    autofocus_enabled = True
    current_focus = 50.0

    if args.manual_focus is not None:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, args.manual_focus)
        autofocus_enabled = False
        current_focus = args.manual_focus
        print(f"Set initial manual focus to {current_focus}")

    detector = FiducialDetector(fiducial_type=args.fiducial_type)

    focus_locked = False
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

        detected_records = []

        if args.fiducial_type == "aruco":
            markers = detector.detect_aruco_markers(frame)
            for marker_id, (u, v) in markers.items():
                u_int, v_int = int(round(u)), int(round(v))
                detected_records.append({
                    "id": marker_id,
                    "pixel_u": round(u, 3),
                    "pixel_v": round(v, 3),
                })

                # Draw bounding marker indicators and pixel coordinates on frame
                cv2.circle(display_frame, (u_int, v_int), 6, (0, 255, 0), -1)
                label = f"ArUco #{marker_id} U:{u:.1f} V:{v:.1f}"
                cv2.putText(display_frame, label, (u_int + 10, v_int - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif args.fiducial_type == "circle":
            center = detector.detect_circle_fiducial(frame)
            if center is not None:
                u, v = center
                u_int, v_int = int(round(u)), int(round(v))
                detected_records.append({
                    "id": 0,
                    "pixel_u": round(u, 3),
                    "pixel_v": round(v, 3),
                })
                cv2.circle(display_frame, (u_int, v_int), 8, (0, 255, 0), 2)
                cv2.circle(display_frame, (u_int, v_int), 2, (0, 0, 255), -1)
                label = f"Fiducial U:{u:.1f} V:{v:.1f}"
                cv2.putText(display_frame, label, (u_int + 10, v_int - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate image sharpness score (Laplacian Variance)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness_score = float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())
        if sharpness_score < 100:
            sharp_text = f"Sharpness: {sharpness_score:.1f} (BLURRY - Move object 10-20 cm further away)"
            sharp_color = (0, 0, 255)  # Red for blurry
        elif sharpness_score < 250:
            sharp_text = f"Sharpness: {sharpness_score:.1f} (OK)"
            sharp_color = (0, 255, 255)  # Yellow
        else:
            sharp_text = f"Sharpness: {sharpness_score:.1f} (SHARP)"
            sharp_color = (0, 255, 0)  # Green

        # Overlay status bar
        focus_status = "LOCKED" if focus_locked else ("AUTO" if autofocus_enabled else f"MANUAL ({current_focus:.0f})")
        cv2.putText(display_frame, f"FPS: {fps:.1f} | Focus: {focus_status}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(display_frame, sharp_text, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sharp_color, 2)
        cv2.putText(display_frame, f"Detected Fiducials: {len(detected_records)}", (10, 71), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(display_frame, "Keys: 's'=Save screenshot+coords | 'f'=Focus Lock | '['/']'=Focus Adj | 'q'=Quit", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        try:
            cv2.imshow("Vision Cutter - PC Camera & Focus Test", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = int(time.time())
                img_name = f"snapshot_coords_{ts}.jpg"
                json_name = f"snapshot_coords_{ts}.json"

                # 1. Save annotated frame with rendered pixel coordinates
                cv2.imwrite(img_name, display_frame)

                # 2. Save JSON sidecar with exact sub-pixel coordinates
                metadata = {
                    "timestamp": ts,
                    "frame_width": w,
                    "frame_height": h,
                    "fiducial_type": args.fiducial_type,
                    "focus_status": focus_status,
                    "detections": detected_records,
                }
                with open(json_name, "w") as f:
                    json.dump(metadata, f, indent=2)

                print(f"\n[SAVED] Image: '{img_name}'")
                print(f"[SAVED] Coordinates Metadata: '{json_name}'")
                print(json.dumps(metadata, indent=2))

            elif key == ord('f'):
                autofocus_enabled = not autofocus_enabled
                focus_locked = not autofocus_enabled
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus_enabled else 0)
                print(f"[CAMERA] Autofocus set to {autofocus_enabled}")

            elif key == ord('['):
                current_focus = max(0.0, current_focus - 5.0)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                cap.set(cv2.CAP_PROP_FOCUS, current_focus)
                autofocus_enabled = False
                focus_locked = True
                print(f"[CAMERA] Manual focus adjusted to {current_focus}")

            elif key == ord(']'):
                current_focus = min(255.0, current_focus + 5.0)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                cap.set(cv2.CAP_PROP_FOCUS, current_focus)
                autofocus_enabled = False
                focus_locked = True
                print(f"[CAMERA] Manual focus adjusted to {current_focus}")

        except cv2.error:
            # Headless fallback
            ts = int(time.time())
            img_name = f"snapshot_coords_headless_{ts}.jpg"
            json_name = f"snapshot_coords_headless_{ts}.json"
            cv2.imwrite(img_name, display_frame)
            metadata = {
                "timestamp": ts,
                "frame_width": w,
                "frame_height": h,
                "fiducial_type": args.fiducial_type,
                "detections": detected_records,
            }
            with open(json_name, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved snapshot '{img_name}' and metadata '{json_name}'.")
            break

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    print("PC Camera test ended.")


if __name__ == "__main__":
    main()
