"""
Ender 3 Auto Cutter Plotter - Core Package
"""

__version__ = "1.0.0"

from .drag_knife_planner import DragKnifePlanner, GCodeGenerator, SVGParser
from .gcode_transformer import GCodeTransformer
from .moonraker_client import CameraClient, MoonrakerClient
from .svg_injector import (
    check_svg_has_fiducials,
    check_svg_has_touch_dots,
    extract_touch_dots,
    inject_aruco_fiducials,
    inject_verification_dots,
)
from .vision_detector import (
    FiducialDetector,
    FiducialSearcher,
    TransformEstimator,
    extract_svg_fiducials,
)

__all__ = [
    "DragKnifePlanner",
    "GCodeGenerator",
    "SVGParser",
    "GCodeTransformer",
    "CameraClient",
    "MoonrakerClient",
    "FiducialDetector",
    "FiducialSearcher",
    "TransformEstimator",
    "extract_svg_fiducials",
    "inject_aruco_fiducials",
    "inject_verification_dots",
    "extract_touch_dots",
    "check_svg_has_fiducials",
    "check_svg_has_touch_dots",
]
