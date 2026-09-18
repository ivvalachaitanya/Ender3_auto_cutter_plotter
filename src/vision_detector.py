"""
Computer Vision & Coordinate Transformation Module

Handles:
- Detection of ArUco markers (DICT_4X4_50) and circular fiducials with sub-pixel precision.
- Mapping image pixel coordinates to physical printer bed coordinates using camera offset and scale.
- 2D Rigid Affine Transformation estimation (cv2.estimateAffinePartial2D) between nominal SVG points and physical bed points.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


class FiducialDetector:
    """Detects ArUco markers or sub-pixel circular fiducials in camera frames."""

    def __init__(self, fiducial_type: str = "aruco", dictionary_id: int = None):
        """
        Initialize fiducial detector.

        :param fiducial_type: 'aruco' or 'circle'
        :param dictionary_id: OpenCV ArUco dictionary ID (default: cv2.aruco.DICT_4X4_50)
        """
        self.fiducial_type = fiducial_type.lower()
        if cv2 is None:
            raise RuntimeError("OpenCV is required for FiducialDetector.")

        if self.fiducial_type == "aruco":
            dict_id = dictionary_id if dictionary_id is not None else cv2.aruco.DICT_4X4_50
            if hasattr(cv2.aruco, "getPredefinedDictionary"):
                self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            else:
                self.aruco_dict = cv2.aruco.Dictionary_get(dict_id)

            if hasattr(cv2.aruco, "DetectorParameters"):
                self.aruco_params = cv2.aruco.DetectorParameters()
            else:
                self.aruco_params = cv2.aruco.DetectorParameters_create()

            # OpenCV 4.7+ support
            if hasattr(cv2.aruco, "ArucoDetector"):
                self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            else:
                self.aruco_detector = None

    def detect_aruco_markers(self, frame: np.ndarray) -> Dict[int, Tuple[float, float]]:
        """
        Detect ArUco markers in frame.

        :param frame: BGR or Grayscale frame image
        :return: Dictionary mapping marker ID to center pixel coordinate (u, v)
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()

        if self.aruco_detector is not None:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        results = {}
        if ids is not None and len(ids) > 0:
            for marker_id, corner in zip(ids.flatten(), corners):
                pts = corner[0]  # 4 corner points
                center_u = float(np.mean(pts[:, 0]))
                center_v = float(np.mean(pts[:, 1]))
                results[int(marker_id)] = (center_u, center_v)
                logger.info("Detected ArUco ID %d at pixel (%.2f, %.2f)", marker_id, center_u, center_v)
        else:
            logger.warning("No ArUco markers detected in frame.")

        return results

    def detect_circle_fiducial(self, frame: np.ndarray, min_area: float = 100.0, max_area: float = 50000.0, min_circularity: float = 0.75) -> Optional[Tuple[float, float]]:
        """
        Detect sub-pixel circular fiducial centroid in frame.

        :param frame: BGR or Grayscale frame image
        :param min_area: Minimum contour area filter
        :param max_area: Maximum contour area filter
        :param min_circularity: Minimum circularity filter (4 * pi * area / perimeter^2)
        :return: Tuple (u, v) pixel coordinate of center, or None if not found
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_center = None
        best_circularity = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity >= min_circularity and circularity > best_circularity:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    best_center = (cx, cy)
                    best_circularity = circularity

        if best_center is not None:
            # Refine centroid with sub-pixel precision
            initial_pt = np.array([[[best_center[0], best_center[1]]]], dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            refined_pt = cv2.cornerSubPix(gray, initial_pt, (5, 5), (-1, -1), criteria)
            refined_u = float(refined_pt[0][0][0])
            refined_v = float(refined_pt[0][0][1])
            logger.info("Detected circular fiducial at sub-pixel (%.2f, %.2f) with circularity %.3f", refined_u, refined_v, best_circularity)
            return refined_u, refined_v

        logger.warning("No circular fiducial matching criteria detected.")
        return None

    def detect(self, frame: np.ndarray, expected_id: Optional[int] = None) -> Optional[Tuple[float, float]]:
        """
        General detect call for selected fiducial type.

        :param frame: Image frame
        :param expected_id: Optional ArUco marker ID expected
        :return: Tuple (u, v) of center pixel coordinates or None
        """
        if self.fiducial_type == "aruco":
            markers = self.detect_aruco_markers(frame)
            if not markers:
                return None
            if expected_id is not None:
                return markers.get(expected_id)
            # Default to first detected marker
            return list(markers.values())[0]
        elif self.fiducial_type == "circle":
            return self.detect_circle_fiducial(frame)
        else:
            raise ValueError(f"Unsupported fiducial type: {self.fiducial_type}")


class TransformEstimator:
    """Computes coordinate transformations from image pixels to bed space and 2D affine mapping."""

    @staticmethod
    def pixel_to_bed_space(
        pixel_u: float,
        pixel_v: float,
        frame_width: int,
        frame_height: int,
        camera_gantry_x: float,
        camera_gantry_y: float,
        mm_per_pixel: float,
        camera_offset_x: float = 0.0,
        camera_offset_y: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Convert frame pixel coordinate (u, v) to physical printer bed coordinate (X, Y).

        :param pixel_u: Image column coordinate (0..width)
        :param pixel_v: Image row coordinate (0..height)
        :param frame_width: Image width in pixels
        :param frame_height: Image height in pixels
        :param camera_gantry_x: Gantry position X when image was taken
        :param camera_gantry_y: Gantry position Y when image was taken
        :param mm_per_pixel: Calibration factor (mm per pixel)
        :param camera_offset_x: Camera offset X relative to knife (Cam_X - Knife_X)
        :param camera_offset_y: Camera offset Y relative to knife (Cam_Y - Knife_Y)
        :return: (X_bed, Y_bed) physical coordinates
        """
        # Center of frame
        center_u = frame_width / 2.0
        center_v = frame_height / 2.0

        # Pixel offsets from optical center
        du_px = pixel_u - center_u
        dv_px = center_v - pixel_v  # Invert Y axis: image Y goes down, bed Y goes up

        # Physical position under camera optical axis
        fiducial_cam_x = camera_gantry_x + (du_px * mm_per_pixel)
        fiducial_cam_y = camera_gantry_y + (dv_px * mm_per_pixel)

        # Subtract camera offset to get bed coordinates relative to toolhead origin
        bed_x = fiducial_cam_x - camera_offset_x
        bed_y = fiducial_cam_y - camera_offset_y

        return bed_x, bed_y

    @staticmethod
    def compute_affine_matrix(svg_points: np.ndarray, bed_points: np.ndarray) -> np.ndarray:
        """
        Compute partial 2D rigid affine matrix M (2x3) mapping SVG coordinates to physical Bed coordinates.

        Uses cv2.estimateAffinePartial2D to restrict transformation to translation, rotation, and uniform scaling.

        :param svg_points: Nx2 array of nominal SVG points
        :param bed_points: Nx2 array of measured physical bed points
        :return: 2x3 Affine transformation matrix M
        """
        if cv2 is None:
            raise RuntimeError("OpenCV is required for compute_affine_matrix.")

        svg_pts = np.asarray(svg_points, dtype=np.float32).reshape(-1, 2)
        bed_pts = np.asarray(bed_points, dtype=np.float32).reshape(-1, 2)

        if len(svg_pts) < 3 or len(bed_pts) < 3:
            raise ValueError(f"At least 3 point pairs are required for 2D rigid affine estimation. Provided {len(svg_pts)} points.")

        M, inliers = cv2.estimateAffinePartial2D(svg_pts, bed_pts)

        if M is None:
            raise RuntimeError("Failed to estimate 2D rigid affine transformation matrix from points.")

        logger.info("Computed 2D Affine Transformation Matrix:\n%s", M)
        return M

    @staticmethod
    def transform_points(points: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Apply 2D affine transformation matrix M to an Nx2 array of points.

        :param points: Nx2 array of coordinates
        :param M: 2x3 affine matrix
        :return: Nx2 array of transformed coordinates
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        ones = np.ones((len(pts), 1), dtype=np.float64)
        pts_homogeneous = np.hstack([pts, ones])  # Nx3

        transformed = pts_homogeneous @ M.T  # Nx2
        return transformed
