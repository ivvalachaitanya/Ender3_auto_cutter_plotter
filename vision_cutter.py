#!/usr/bin/env python3
"""
Autonomous Computer-Vision "Print & Cut" Vinyl Cutting Pipeline for Ender 3 / Klipper.

Usage:
    python vision_cutter.py --svg path/to/file.svg --blade-offset 0.35 --moonraker-url http://192.168.1.50:7125
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import List, Tuple
import numpy as np

# Add src package directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from moonraker_client import CameraClient, MoonrakerClient
from vision_detector import FiducialDetector, FiducialSearcher, TransformEstimator, extract_svg_fiducials
from drag_knife_planner import DragKnifePlanner, GCodeGenerator, SVGParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vision_cutter")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Autonomous Computer-Vision Print & Cut Vinyl Cutting Pipeline for Ender 3 (Klipper)."
    )

    # Load defaults from config.json if available
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    cfg = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            pass

    # Required / Input files
    parser.add_argument("--svg", type=str, required=True, help="Path to input SVG file containing cut paths.")
    parser.add_argument("--output", type=str, default="print_and_cut_output.gcode", help="Output G-code file path.")

    # Hardware & Network Settings
    parser.add_argument("--moonraker-url", type=str, default=cfg.get("moonraker_url", "http://localhost:7125"), help="Moonraker server URL.")
    parser.add_argument("--crowsnest-url", type=str, default=cfg.get("crowsnest_url", "http://localhost:8080/?action=stream"), help="Crowsnest camera snapshot URL.")
    parser.add_argument("--camera-index", type=int, default=0, help="Local OpenCV camera index fallback.")

    # Camera & Tool Calibration Parameters
    parser.add_argument("--camera-offset-x", type=float, default=cfg.get("camera_offset_x_mm", -25.0), help="Camera X offset relative to knife tip (mm).")
    parser.add_argument("--camera-offset-y", type=float, default=cfg.get("camera_offset_y_mm", 0.0), help="Camera Y offset relative to knife tip (mm).")
    parser.add_argument("--mm-per-pixel", type=float, default=cfg.get("mm_per_pixel", 0.0293), help="Camera scale calibration factor (mm/pixel).")
    parser.add_argument("--blade-offset", type=float, default=cfg.get("blade_offset_mm", 0.35), help="Drag knife blade offset (mm).")

    # Vision & Fiducials
    default_nom = json.dumps(cfg.get("nominal_fiducials", [[0.0, 210.0], [170.0, 210.0], [0.0, 20.0]]))
    parser.add_argument("--fiducial-type", type=str, choices=["aruco", "circle"], default="aruco", help="Type of fiducial markers.")
    parser.add_argument(
        "--nominal-fiducials",
        type=str,
        default=default_nom,
        help="JSON string of nominal SVG fiducial coordinates e.g. '[[0,210], [170,210], [0,20]]'",
    )

    # Cutting & Scanning Parameters
    parser.add_argument("--z-scan", type=float, default=cfg.get("z_scan_mm", 20.0), help="Camera focal scanning Z height in mm (default: 20.0).")
    parser.add_argument("--z-safe", type=float, default=cfg.get("z_safe_mm", 5.0), help="Safe travel Z height (mm).")
    parser.add_argument("--z-cut", type=float, default=cfg.get("z_cut_mm", 0.0), help="Cutting Z depth (mm).")
    parser.add_argument("--z-lift-swivel", type=float, default=cfg.get("z_lift_swivel_mm", 0.0), help="Z height for Lift-and-Turn sharp corner swiveling in mm (0.0 = in-plane).")
    parser.add_argument("--f-cut", type=float, default=cfg.get("f_cut_mm_min", 1200.0), help="Cutting feedrate (mm/min).")
    parser.add_argument("--f-travel", type=float, default=cfg.get("f_travel_mm_min", 3000.0), help="Travel feedrate (mm/min).")

    # Execution flags
    parser.add_argument("--no-auto-search", action="store_true", help="Disable autonomous spiral search and rigid geometric projection.")
    parser.add_argument("--force-home", action="store_true", help="Force printer G28 homing even if axes are already homed.")
    parser.add_argument("--dry-run", action="store_true", help="Perform alignment check & generate G-code at safe Z without cutting.")
    parser.add_argument("--auto-start", action="store_true", help="Automatically trigger print job via Moonraker after upload.")
    parser.add_argument("--mock", action="store_true", help="Run in mock offline mode (simulates Klipper & camera responses).")

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("======================================================")
    logger.info("Starting Vision Cutter Pipeline")
    logger.info("SVG File: %s", args.svg)
    logger.info("Blade Offset: %.3f mm | Dry Run: %s", args.blade_offset, args.dry_run)
    logger.info("======================================================")

    # Validate SVG existence
    if not os.path.isfile(args.svg):
        samples_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", args.svg)
        if os.path.isfile(samples_path):
            args.svg = samples_path
        else:
            logger.error("Specified SVG/Gcode file '%s' does not exist.", args.svg)
            sys.exit(1)

    # Parse nominal fiducial locations
    try:
        nominal_fiducials = json.loads(args.nominal_fiducials)
        nominal_fiducials = np.array(nominal_fiducials, dtype=np.float64)
        if len(nominal_fiducials) < 3:
            raise ValueError("At least 3 nominal fiducial points are required.")
    except Exception as e:
        logger.error("Failed to parse --nominal-fiducials JSON parameter: %s", e)
        sys.exit(1)

    # Step 1: Connect to Moonraker & Camera Clients
    moonraker = MoonrakerClient(base_url=args.moonraker_url)
    camera = CameraClient(snapshot_url=args.crowsnest_url, camera_index=args.camera_index)

    if not args.mock and not moonraker.is_connected():
        logger.error("Cannot reach Moonraker server at '%s'. Check connection or use --mock.", args.moonraker_url)
        sys.exit(1)

    # Step 2: Printer Homing
    if not args.mock:
        if args.force_home or not moonraker.is_homed("xyz"):
            logger.info("Homing printer gantry (G28)...")
            if not moonraker.home_gantry(force=True):
                logger.error("Homing sequence failed!")
                sys.exit(1)
        else:
            logger.info("Printer axes are already homed (XYZ). Skipping redundant G28 homing sequence.")
    else:
        logger.info("[MOCK] Printer gantry homed.")

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    cfg = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            pass

    bed_max_x = cfg.get("bed_width_mm", 235.0)
    bed_max_y = cfg.get("bed_height_mm", 235.0)

    # Determine reference nominal points in design space
    design_nominal_points = None
    if not args.svg.lower().endswith((".gcode", ".nc", ".gc", ".tap")):
        design_nominal_points = extract_svg_fiducials(args.svg, bed_height_mm=bed_max_y)
        if design_nominal_points is not None:
            logger.info("Extracted %d nominal fiducials from SVG design: %s", len(design_nominal_points), design_nominal_points.tolist())

    if design_nominal_points is None:
        cfg_design = cfg.get("design_fiducials")
        if cfg_design and len(cfg_design) >= 3:
            design_nominal_points = np.array(cfg_design, dtype=np.float64)
            logger.info("Using design_fiducials from config: %s", design_nominal_points.tolist())
        else:
            design_nominal_points = nominal_fiducials

    # Step 3: Discover & Detect Physical Fiducials
    detector = FiducialDetector(fiducial_type=args.fiducial_type)

    if not args.mock and not args.no_auto_search:
        logger.info("Engaging Autonomous Fiducial Discovery & Visual Servoing...")
        measured_bed_points, measured_gantry_points = FiducialSearcher.discover_all_fiducials(
            moonraker_client=moonraker,
            camera_client=camera,
            detector=detector,
            design_fiducials=design_nominal_points,
            nominal_fiducials=nominal_fiducials,
            z_scan=args.z_scan,
            bed_max_x=bed_max_x,
            bed_max_y=bed_max_y,
            camera_offset_x=args.camera_offset_x,
            camera_offset_y=args.camera_offset_y,
            mm_per_pixel=args.mm_per_pixel,
        )
        try:
            cfg["nominal_fiducials"] = measured_gantry_points.tolist()
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            logger.info("Updated nominal_fiducials in config.json with newly centered coordinates.")
        except Exception as e:
            logger.warning("Could not persist updated nominal fiducials: %s", e)

        moonraker.jog_to(z=args.z_safe, feedrate=args.f_travel, wait_finish=True)
    elif not args.mock:
        # Legacy fixed-coordinate single-shot detection
        measured_bed_points = []
        for idx, nominal_pt in enumerate(nominal_fiducials):
            nom_x, nom_y = nominal_pt[0], nominal_pt[1]
            target_cam_x = max(0.0, min(bed_max_x, nom_x))
            target_cam_y = max(0.0, min(bed_max_y, nom_y))

            logger.info("Jogging camera over Fiducial #%d (Search Point: %.2f, %.2f) -> Target Gantry (%.2f, %.2f) at Z=%.1f...", idx + 1, nom_x, nom_y, target_cam_x, target_cam_y, args.z_scan)
            moonraker.jog_to(x=target_cam_x, y=target_cam_y, z=args.z_scan, feedrate=args.f_travel, wait_finish=True)
            time.sleep(0.5)

            try:
                frame = camera.get_frame()
            except Exception as err:
                logger.error("Failed to capture frame at fiducial #%d: %s", idx + 1, err)
                sys.exit(1)

            detected_px = detector.detect(frame)
            if detected_px is None:
                logger.error("Failed to detect fiducial marker #%d in camera frame!", idx + 1)
                sys.exit(1)

            frame_h, frame_w = frame.shape[:2]
            current_x, current_y, _ = moonraker.get_toolhead_position()

            bed_x, bed_y = TransformEstimator.pixel_to_bed_space(
                pixel_u=detected_px[0],
                pixel_v=detected_px[1],
                frame_width=frame_w,
                frame_height=frame_h,
                camera_gantry_x=current_x,
                camera_gantry_y=current_y,
                mm_per_pixel=args.mm_per_pixel,
                camera_offset_x=args.camera_offset_x,
                camera_offset_y=args.camera_offset_y,
            )
            measured_bed_points.append([bed_x, bed_y])
            logger.info("Fiducial #%d Measured Bed Coordinate: (X=%.3f, Y=%.3f)", idx + 1, bed_x, bed_y)

        moonraker.jog_to(z=args.z_safe, feedrate=args.f_travel, wait_finish=True)
        measured_bed_points = np.array(measured_bed_points, dtype=np.float64)
    else:
        # Mock mode: add camera offset and minor simulated translation
        measured_bed_points = []
        for idx, nominal_pt in enumerate(nominal_fiducials):
            bed_x = nominal_pt[0] + args.camera_offset_x + 0.2
            bed_y = nominal_pt[1] + args.camera_offset_y + 0.1
            logger.info("[MOCK] Simulated measured bed coordinate for Fiducial #%d: (%.3f, %.3f)", idx + 1, bed_x, bed_y)
            measured_bed_points.append([bed_x, bed_y])
        measured_bed_points = np.array(measured_bed_points, dtype=np.float64)

    # Step 4: Compute 2D Rigid Affine Matrix M
    logger.info("Computing 2D Rigid Affine Transformation matrix...")
    M = TransformEstimator.compute_affine_matrix(design_nominal_points, measured_bed_points)

    # Step 5: Check if input is SVG or PolyCut/JScut G-code File
    if args.svg.lower().endswith((".gcode", ".nc", ".gc", ".tap")):
        logger.info("Input file '%s' detected as G-code (PolyCut / JScut / Inkscape).", args.svg)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
        from gcode_transformer import GCodeTransformer

        transformer = GCodeTransformer(matrix=M)
        success = transformer.transform_gcode_file(args.svg, args.output)
        if not success:
            logger.error("Failed to apply fiducial compensation to PolyCut G-code file!")
            sys.exit(1)
        logger.info("Successfully applied 2D Affine Camera Fiducial Compensation to PolyCut G-code!")
    else:
        # Parse SVG & Apply Affine Transformation
        logger.info("Parsing SVG cut paths from '%s'...", args.svg)
        svg_parser = SVGParser(step_size_mm=0.5)
        raw_contours = svg_parser.parse_svg_paths(args.svg)

        if not raw_contours:
            logger.error("No valid vector contours found in SVG file!")
            sys.exit(1)

        transformed_contours = []
        for contour in raw_contours:
            pts_arr = np.array(contour, dtype=np.float64)
            transformed_pts = TransformEstimator.transform_points(pts_arr, M)
            transformed_contours.append([(pt[0], pt[1]) for pt in transformed_pts])

        # Step 6: Drag Knife Swivel Arc Compensation & Trajectory Planning
        logger.info("Planning drag-knife trajectories (Blade Offset: %.3f mm)...", args.blade_offset)
        knife_planner = DragKnifePlanner(blade_offset_mm=args.blade_offset, swivel_angle_threshold_deg=30.0)

        planned_contours = []
        for contour in transformed_contours:
            planned_moves = knife_planner.plan_contour(contour)
            if planned_moves:
                planned_contours.append(planned_moves)

        # Step 7: Generate Klipper G-code
        gcode_gen = GCodeGenerator(
            z_safe=args.z_safe,
            z_cut=args.z_cut,
            f_cut=args.f_cut,
            f_travel=args.f_travel,
            dry_run=args.dry_run,
            z_lift_swivel=args.z_lift_swivel,
        )
        final_gcode = gcode_gen.generate_gcode(planned_contours)

        # Write G-code to local file
        output_filename = args.output
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(final_gcode)
        logger.info("Saved final G-code file to '%s'.", output_filename)

    # Step 8: Upload to Moonraker & Start Job
    if not args.mock:
        try:
            remote_path = moonraker.upload_gcode(output_filename)
            logger.info("Successfully uploaded G-code to Moonraker path: '%s'", remote_path)

            if args.auto_start:
                logger.info("Triggering auto-start print job...")
                moonraker.start_print(output_filename)
        except Exception as err:
            logger.error("Failed to upload/start print job on Moonraker: %s", err)
            sys.exit(1)
    else:
        logger.info("[MOCK] Upload and print trigger step completed.")

    logger.info("======================================================")
    logger.info("Vision Cutter Pipeline Completed Successfully!")
    logger.info("======================================================")


if __name__ == "__main__":
    main()
