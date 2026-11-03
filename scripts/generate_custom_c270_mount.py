#!/usr/bin/env python3
"""
Custom 3D STL Generator for Ender 3 Drag Knife Support + Logitech C270 Camera Mount.

Combines SupportoDragKnife.STL with a parametric Logitech C270 webcam mounting arm.

Usage:
    python generate_custom_c270_mount.py --offset-x -25.0 --screw-dia 3.2 --hinge-width 6.2
"""

import argparse
import math
import os
import sys
import numpy as np

try:
    import trimesh
except ImportError:
    print("Error: trimesh is required. Run: pip install trimesh manifold3d")
    sys.exit(1)

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
MECHANICAL_DIR = os.path.join(WORKSPACE_DIR, "..", "mechanical")
INPUT_STL = os.path.join(MECHANICAL_DIR, "SupportoDragKnife.STL")
OUTPUT_STL = os.path.join(MECHANICAL_DIR, "SupportoDragKnife_with_C270_CameraMount.stl")


def generate_mount(offset_x=-25.0, offset_y=0.0, screw_dia=3.2, hinge_width=6.2, side="left"):
    """
    Generate 3D combined STL model.

    :param offset_x: Camera X offset relative to knife tip (mm)
    :param offset_y: Camera Y offset relative to knife tip (mm)
    :param screw_dia: Hinge screw hole diameter in mm (default: 3.2mm for M3)
    :param hinge_width: C270 hinge slot width in mm (default: 6.2mm)
    :param side: 'left' or 'right'
    """
    if not os.path.isfile(INPUT_STL):
        raise FileNotFoundError(f"Input file '{INPUT_STL}' not found.")

    print("======================================================")
    print("Generating Custom 3D STL with Logitech C270 Camera Mount")
    print(f"Base STL: {os.path.basename(INPUT_STL)}")
    print(f"Camera Offset X: {offset_x:.1f} mm | Offset Y: {offset_y:.1f} mm")
    print(f"Hinge Screw Hole: {screw_dia:.1f} mm (M3/M4) | Hinge Width: {hinge_width:.1f} mm")
    print("======================================================")

    base_mesh = trimesh.load(INPUT_STL)

    # Base mesh bounds
    b_min, b_max = base_mesh.bounds[0], base_mesh.bounds[1]
    base_center_y = (b_min[1] + b_max[1]) / 2.0  # 22.5 mm

    # Arm dimensions
    arm_length = abs(offset_x) + 2.0
    arm_w = 14.0
    arm_h = 8.0

    # Create Extension Arm Box
    arm = trimesh.creation.box(extents=[arm_length, arm_w, arm_h])
    if side == "left":
        arm.apply_translation([-arm_length / 2.0 + 1.0, base_center_y + offset_y, arm_h / 2.0])
    else:
        arm.apply_translation([b_max[0] + arm_length / 2.0 - 1.0, base_center_y + offset_y, arm_h / 2.0])

    # Logitech C270 Hinge Tab Parameters
    hinge_outer_r = 6.0
    hinge_inner_r = screw_dia / 2.0

    hinge_outer = trimesh.creation.cylinder(radius=hinge_outer_r, height=hinge_width)
    hinge_outer.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))

    hinge_hole = trimesh.creation.cylinder(radius=hinge_inner_r, height=hinge_width + 4.0)
    hinge_hole.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))

    if side == "left":
        mount_x = offset_x
    else:
        mount_x = b_max[0] + abs(offset_x)

    hinge_outer.apply_translation([mount_x, base_center_y + offset_y, arm_h / 2.0])
    hinge_hole.apply_translation([mount_x, base_center_y + offset_y, arm_h / 2.0])

    # Boolean difference for screw hole
    hinge_tab = trimesh.boolean.difference([hinge_outer, hinge_hole], engine="manifold")

    # Union all parts into a single 3D watertight mesh
    combined = trimesh.boolean.union([base_mesh, arm, hinge_tab], engine="manifold")

    # Export STL
    combined.export(OUTPUT_STL)
    print("[OK] Exported watertight STL model:", OUTPUT_STL)
    dims = combined.bounds[1] - combined.bounds[0]
    print(f"Final Model Dimensions: {dims[0]:.2f}mm (X) x {dims[1]:.2f}mm (Y) x {dims[2]:.2f}mm (Z)")
    return OUTPUT_STL


def main():
    parser = argparse.ArgumentParser(description="Generate 3D STL for Drag Knife + Logitech C270 Mount.")
    parser.add_argument("--offset-x", type=float, default=-25.0, help="Camera X offset (default: -25.0 mm).")
    parser.add_argument("--offset-y", type=float, default=0.0, help="Camera Y offset (default: 0.0 mm).")
    parser.add_argument("--screw-dia", type=float, default=3.2, help="Hinge screw hole diameter (default: 3.2 mm for M3).")
    parser.add_argument("--hinge-width", type=float, default=6.2, help="C270 hinge slot width (default: 6.2 mm).")
    parser.add_argument("--side", type=str, choices=["left", "right"], default="left", help="Mount side (default: left).")
    args = parser.parse_args()

    generate_mount(
        offset_x=args.offset_x,
        offset_y=args.offset_y,
        screw_dia=args.screw_dia,
        hinge_width=args.hinge_width,
        side=args.side,
    )


if __name__ == "__main__":
    main()
