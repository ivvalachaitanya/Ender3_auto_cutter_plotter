"""
Computer Vision & Coordinate Transformation Module

Handles:
- Detection of ArUco markers (DICT_4X4_50) and circular fiducials with sub-pixel precision.
- Mapping image pixel coordinates to physical printer bed coordinates using camera offset and scale.
- 2D Rigid Affine Transformation estimation (cv2.estimateAffinePartial2D) between nominal SVG points and physical bed points.
"""

import logging
import os
import time
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

    def detect_aruco_markers(self, frame: np.ndarray, log_warnings: bool = True) -> Dict[int, Tuple[float, float]]:
        """
        Detect ArUco markers in frame.

        :param frame: BGR or Grayscale frame image
        :param log_warnings: Whether to log marker detections/warnings (set False for high-FPS video streaming)
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
                if log_warnings:
                    logger.info("Detected ArUco ID %d at pixel (%.2f, %.2f)", marker_id, center_u, center_v)
        else:
            if log_warnings:
                logger.warning("No ArUco markers detected in frame.")

        return results

    def detect_all_circle_fiducials(self, frame: np.ndarray, min_area: float = 15.0, max_area: float = 50000.0, min_circularity: float = 0.55) -> List[Tuple[float, float]]:
        """
        Detect ALL sub-pixel circular fiducial centroids in frame.

        :param frame: BGR or Grayscale frame image
        :param min_area: Minimum contour area filter
        :param max_area: Maximum contour area filter
        :param min_circularity: Minimum circularity filter
        :return: List of (u, v) sub-pixel pixel coordinate tuples for all detected circles
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity >= min_circularity:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    centers.append(((cx, cy), circularity))

        if not centers:
            logger.warning("No circular fiducials matching criteria detected.")
            return []

        # Sort by circularity descending
        centers.sort(key=lambda item: item[1], reverse=True)

        refined_centers = []
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        for (cx, cy), circ in centers:
            initial_pt = np.array([[[cx, cy]]], dtype=np.float32)
            refined_pt = cv2.cornerSubPix(gray, initial_pt, (5, 5), (-1, -1), criteria)
            refined_u = float(refined_pt[0][0][0])
            refined_v = float(refined_pt[0][0][1])
            refined_centers.append((refined_u, refined_v))
            logger.info("Detected circular fiducial at sub-pixel (%.2f, %.2f) circularity=%.3f", refined_u, refined_v, circ)

        return refined_centers

    def detect_circle_fiducial(self, frame: np.ndarray, min_area: float = 15.0, max_area: float = 50000.0, min_circularity: float = 0.55) -> Optional[Tuple[float, float]]:
        """
        Detect single best sub-pixel circular fiducial centroid in frame.
        """
        all_centers = self.detect_all_circle_fiducials(frame, min_area=min_area, max_area=max_area, min_circularity=min_circularity)
        return all_centers[0] if all_centers else None

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

        Calibrated for Ender 3 camera mount:
        - Moving gantry +X increases image u; to center (reduce u), gantry moves -X.
        - Moving gantry +Y decreases image v; to center (reduce v), gantry moves +Y.

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
        center_u = frame_width / 2.0
        center_v = frame_height / 2.0

        du_px = pixel_u - center_u
        dv_px = pixel_v - center_v

        bed_x = camera_gantry_x + camera_offset_x - (du_px * mm_per_pixel)
        bed_y = camera_gantry_y + camera_offset_y + (dv_px * mm_per_pixel)

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


def extract_svg_fiducials(svg_path: str, bed_height_mm: float = 220.0) -> Optional[np.ndarray]:
    """
    Extract nominal fiducial positions from SVG file (in bed coordinate space with flipped Y).
    Looks for 10x10mm marker image elements or elements with 'aruco'/'fiducial' in id.

    Returns Nx2 array of points sorted: Back-Left, Back-Right, Front-Left.
    """
    if not os.path.isfile(svg_path):
        return None
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(svg_path)
        root = tree.getroot()

        height = bed_height_mm
        vb = root.attrib.get('viewBox')
        if vb:
            parts = vb.replace(',', ' ').split()
            if len(parts) == 4:
                try:
                    height = float(parts[3])
                except ValueError:
                    pass

        markers = []
        for elem in root.iter():
            tag = elem.tag.split('}')[-1].lower()
            elem_id = str(elem.attrib.get('id', '')).lower()
            if tag == 'image':
                try:
                    w = float(elem.attrib.get('width', 0))
                    h = float(elem.attrib.get('height', 0))
                    x = float(elem.attrib.get('x', 0))
                    y = float(elem.attrib.get('y', 0))
                except (ValueError, TypeError):
                    continue

                if (abs(w - 10.0) < 1.0 and abs(h - 10.0) < 1.0) or 'aruco' in elem_id or 'fiducial' in elem_id:
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    bed_y = height - cy
                    markers.append((cx, bed_y))

        if len(markers) >= 3:
            pts = np.array(markers[:3], dtype=np.float64)
            median_y = np.median(pts[:, 1])
            back_pts = sorted([p for p in pts if p[1] >= median_y], key=lambda p: p[0])
            front_pts = sorted([p for p in pts if p[1] < median_y], key=lambda p: p[0])
            sorted_markers = back_pts + front_pts
            if len(sorted_markers) >= 3:
                return np.array(sorted_markers[:3], dtype=np.float64)
            return pts
    except Exception as e:
        logger.warning("Could not extract SVG fiducials from %s: %s", svg_path, e)
    return None


class FiducialSearcher:
    """
    Autonomous Fiducial Search and Visual Servoing Centering Engine.

    Features:
    1. Sub-pixel visual servo auto-centering (iterative micro-jog).
    2. Adaptive expanding square spiral search for marker acquisition.
    3. Rigid-body geometric vector and rotation/scale projection for instant lock
       of remaining markers without blind scanning.
    """

    @staticmethod
    def center_marker(
        moonraker_client,
        camera_client,
        detector: FiducialDetector,
        gantry_x: float,
        gantry_y: float,
        z_scan: float = 20.0,
        bed_max_x: float = 235.0,
        bed_max_y: float = 235.0,
        mm_per_pixel: float = 0.0293,
        max_iters: int = 2,
        settle_seconds: float = 2.0,
        expected_id: Optional[int] = None,
    ) -> Optional[Tuple[float, float, Tuple[float, float]]]:
        """
        Perform iterative visual servo micro-jogging to bring marker to exact optical center.

        :return: (actual_x, actual_y, (pixel_u, pixel_v)) or None
        """
        curr_x = gantry_x
        curr_y = gantry_y

        moonraker_client.jog_to(x=curr_x, y=curr_y, z=z_scan, feedrate=3000.0, wait_finish=True)
        time.sleep(settle_seconds)

        frame = camera_client.get_frame()
        pt = detector.detect(frame, expected_id=expected_id)
        if pt is None:
            return None

        h, w = frame.shape[:2]
        center_u = w / 2.0
        center_v = h / 2.0

        for it in range(max_iters):
            du = pt[0] - center_u
            dv = pt[1] - center_v

            if abs(du) < 2.0 and abs(dv) < 2.0:
                logger.info("Fiducial already centered to sub-pixel accuracy (offset: du=%.1f, dv=%.1f px).", du, dv)
                break

            adj_x = max(0.0, min(bed_max_x, curr_x - (du * mm_per_pixel)))
            adj_y = max(0.0, min(bed_max_y, curr_y + (dv * mm_per_pixel)))

            logger.info("Visual servo centering iteration %d: micro-jogging from (%.2f, %.2f) to (%.2f, %.2f)...", it + 1, curr_x, curr_y, adj_x, adj_y)
            moonraker_client.jog_to(x=adj_x, y=adj_y, z=z_scan, feedrate=2000.0, wait_finish=True)
            time.sleep(settle_seconds)

            curr_x, curr_y = adj_x, adj_y
            try:
                new_frame = camera_client.get_frame()
                new_pt = detector.detect(new_frame, expected_id=expected_id)
                if new_pt is not None:
                    pt = new_pt
            except Exception:
                pass

        actual_x, actual_y, _ = moonraker_client.get_toolhead_position()
        logger.info("Fiducial locked & centered at gantry (X=%.3f, Y=%.3f), pixel (%.1f, %.1f).", actual_x, actual_y, pt[0], pt[1])
        return actual_x, actual_y, pt

    @staticmethod
    def search_and_center(
        moonraker_client,
        camera_client,
        detector: FiducialDetector,
        start_x: float,
        start_y: float,
        z_scan: float = 20.0,
        bed_max_x: float = 235.0,
        bed_max_y: float = 235.0,
        mm_per_pixel: float = 0.0293,
        step_mm: float = 14.0,
        max_rings: int = 3,
        settle_seconds: float = 2.0,
        expected_id: Optional[int] = None,
    ) -> Optional[Tuple[float, float, Tuple[float, float]]]:
        """
        Search for a fiducial starting at (start_x, start_y).
        If visible immediately, centers it. If not, spirals outward ring by ring.

        :return: (actual_x, actual_y, (pixel_u, pixel_v)) or None
        """
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        curr_x, curr_y = start_x, start_y
        step_len = 1
        dir_idx = 0
        steps_taken = 0
        total_steps = (2 * max_rings + 1) ** 2
        last_gantry = None

        logger.info("Initiating autonomous fiducial search around (X=%.1f, Y=%.1f, Z=%.1f)...", start_x, start_y, z_scan)

        for step in range(total_steps):
            target_x = max(0.0, min(bed_max_x, curr_x))
            target_y = max(0.0, min(bed_max_y, curr_y))

            # Skip redundant moves caused by bounding box clamping
            if last_gantry is None or abs(target_x - last_gantry[0]) > 0.5 or abs(target_y - last_gantry[1]) > 0.5:
                moonraker_client.jog_to(x=target_x, y=target_y, z=z_scan, feedrate=3000.0, wait_finish=True)
                time.sleep(settle_seconds)
                last_gantry = (target_x, target_y)

                try:
                    frame = camera_client.get_frame()
                except Exception as e:
                    logger.warning("Failed to grab camera frame during search: %s", e)
                    continue

                pt = detector.detect(frame, expected_id=expected_id)
                if pt is not None:
                    logger.info("Fiducial marker (ID=%s) spotted at gantry (X=%.2f, Y=%.2f)! Centering...", expected_id, target_x, target_y)
                    return FiducialSearcher.center_marker(
                        moonraker_client=moonraker_client,
                        camera_client=camera_client,
                        detector=detector,
                        gantry_x=target_x,
                        gantry_y=target_y,
                        z_scan=z_scan,
                        bed_max_x=bed_max_x,
                        bed_max_y=bed_max_y,
                        mm_per_pixel=mm_per_pixel,
                        settle_seconds=settle_seconds,
                        expected_id=expected_id,
                    )

            # Advance spiral to next hop
            curr_x += dx[dir_idx] * step_mm
            curr_y += dy[dir_idx] * step_mm
            steps_taken += 1
            if steps_taken == step_len:
                steps_taken = 0
                dir_idx = (dir_idx + 1) % 4
                if dir_idx % 2 == 0:
                    step_len += 1

        logger.warning("Autonomous fiducial search exhausted after %d hops without finding marker.", total_steps)
        return None

    @staticmethod
    def sweep_bed_quadrants(
        moonraker_client,
        camera_client,
        detector: FiducialDetector,
        z_scan: float = 20.0,
        bed_max_x: float = 235.0,
        bed_max_y: float = 235.0,
        mm_per_pixel: float = 0.0293,
        settle_seconds: float = 2.0,
    ) -> Optional[Tuple[int, float, float, Tuple[float, float]]]:
        """
        Sweep strategic bed vantage points across all quadrants until any fiducial is spotted.

        :return: (marker_id, actual_x, actual_y, (u, v)) or None
        """
        sweep_points = [
            # Quadrant 1: Rear-Left (most standard placement)
            (5.0, 220.0),
            (35.0, 220.0),
            (20.0, 185.0),
            # Quadrant 2: Rear-Right
            (173.0, 220.0),
            (195.0, 220.0),
            (170.0, 185.0),
            # Quadrant 3: Front-Left
            (5.0, 35.0),
            (35.0, 35.0),
            (20.0, 70.0),
            # Quadrant 4: Front-Right
            (173.0, 35.0),
            (195.0, 35.0),
            # Center Bed
            (115.0, 115.0),
            (115.0, 185.0),
            (115.0, 45.0),
        ]

        logger.info("[EXHAUSTIVE SWEEP] Probing %d strategic bed vantage points across all quadrants...", len(sweep_points))
        for idx, (probe_x, probe_y) in enumerate(sweep_points):
            target_x = max(0.0, min(bed_max_x, probe_x))
            target_y = max(0.0, min(bed_max_y, probe_y))

            moonraker_client.jog_to(x=target_x, y=target_y, z=z_scan, feedrate=3000.0, wait_finish=True)
            time.sleep(settle_seconds)

            try:
                frame = camera_client.get_frame()
            except Exception as e:
                logger.warning("Frame grab failed during bed sweep: %s", e)
                continue

            markers = detector.detect_aruco_markers(frame)
            if markers:
                marker_id, (u, v) = next(iter(markers.items()))
                logger.info("[EXHAUSTIVE SWEEP] Spotted ArUco ID %d at gantry (%.1f, %.1f)! Centering...", marker_id, target_x, target_y)
                centered = FiducialSearcher.center_marker(
                    moonraker_client=moonraker_client,
                    camera_client=camera_client,
                    detector=detector,
                    gantry_x=target_x,
                    gantry_y=target_y,
                    z_scan=z_scan,
                    bed_max_x=bed_max_x,
                    bed_max_y=bed_max_y,
                    mm_per_pixel=mm_per_pixel,
                    settle_seconds=settle_seconds,
                )
                if centered is not None:
                    act_x, act_y, pt = centered
                    return marker_id, act_x, act_y, pt

        logger.warning("[EXHAUSTIVE SWEEP] Bed quadrant sweep completed with 0 detections.")
        return None

    @staticmethod
    def discover_all_fiducials(
        moonraker_client,
        camera_client,
        detector: FiducialDetector,
        design_fiducials: np.ndarray,
        nominal_fiducials: Optional[np.ndarray] = None,
        z_scan: float = 20.0,
        bed_max_x: float = 235.0,
        bed_max_y: float = 235.0,
        camera_offset_x: float = 25.0,
        camera_offset_y: float = 0.0,
        mm_per_pixel: float = 0.0293,
        search_mode: str = "assisted",
        settle_seconds: float = 2.0,
        wait_for_user_timeout: float = 180.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Autonomous 3-Fiducial Discovery & Visual Servoing Alignment Pipeline with Speed Preferences.

        Search Modes:
        - 'fast': Preference 1 (Turbo / Direct) - 0-hop single-shot hit at cached positions (~5s).
        - 'spiral': Preference 2 (Adaptive Spiral) - Local expanding spiral search + rigid projection (~15s).
        - 'exhaustive': Preference 3 (Exhaustive Sweep) - Global bed quadrant sweep for arbitrary placement (~45s).
        - 'auto': Cascading Speed Priority: Tries Preference 1 -> if missed, escalates to Preference 2 -> if missed, escalates to Preference 3.

        :return: (measured_bed_points (Nx2), measured_gantry_points (Nx2))
        """
        import math
        mode = (search_mode or "auto").lower().strip()

        # Determine actual camera frame resolution (avoid hardcoding 640x480)
        try:
            _probe_frame = camera_client.get_frame()
            _cam_h, _cam_w = _probe_frame.shape[:2]
        except Exception:
            _cam_w, _cam_h = 640, 480  # Safe fallback

        logger.info("======================================================")
        logger.info("Autonomous Fiducial Engine | Strategy: '%s'", mode.upper())
        logger.info("Design Fiducials: %s", design_fiducials.tolist())
        logger.info("======================================================")

        d1 = design_fiducials[0]
        d2 = design_fiducials[1]
        d3 = design_fiducials[2]

        def try_lock_three_markers(
            start_x1: float,
            start_y1: float,
            max_rings_m1: int,
            max_rings_m2: int,
            max_rings_m3: int,
        ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
            """Helper to attempt 3-marker lock with given spiral search tolerances."""
            exp_id_1 = 0 if detector.fiducial_type == "aruco" else None
            exp_id_2 = 1 if detector.fiducial_type == "aruco" else None
            exp_id_3 = 2 if detector.fiducial_type == "aruco" else None

            # Step 1: Discover & Center Fiducial #1 (Rear-Left)
            res1 = FiducialSearcher.search_and_center(
                moonraker_client=moonraker_client,
                camera_client=camera_client,
                detector=detector,
                start_x=start_x1,
                start_y=start_y1,
                z_scan=z_scan,
                bed_max_x=bed_max_x,
                bed_max_y=bed_max_y,
                mm_per_pixel=mm_per_pixel,
                max_rings=max_rings_m1,
                settle_seconds=settle_seconds,
                expected_id=exp_id_1,
            )
            if res1 is None:
                return None

            gantry_x1, gantry_y1, pt1 = res1
            bed_x1, bed_y1 = TransformEstimator.pixel_to_bed_space(
                pixel_u=pt1[0], pixel_v=pt1[1],
                frame_width=_cam_w, frame_height=_cam_h,
                camera_gantry_x=gantry_x1, camera_gantry_y=gantry_y1,
                mm_per_pixel=mm_per_pixel,
                camera_offset_x=camera_offset_x, camera_offset_y=camera_offset_y,
            )
            logger.info("Fiducial #1 LOCKED at Bed (X=%.3f, Y=%.3f)", bed_x1, bed_y1)

            # Step 2: Predict & Center Fiducial #2 (Rear-Right)
            d12_x = d2[0] - d1[0]
            d12_y = d2[1] - d1[1]
            pred_gantry_x2 = max(0.0, min(bed_max_x, gantry_x1 + d12_x))
            pred_gantry_y2 = max(0.0, min(bed_max_y, gantry_y1 + d12_y))

            res2 = FiducialSearcher.search_and_center(
                moonraker_client=moonraker_client,
                camera_client=camera_client,
                detector=detector,
                start_x=pred_gantry_x2,
                start_y=pred_gantry_y2,
                z_scan=z_scan,
                bed_max_x=bed_max_x,
                bed_max_y=bed_max_y,
                mm_per_pixel=mm_per_pixel,
                max_rings=max_rings_m2,
                settle_seconds=settle_seconds,
                expected_id=exp_id_2,
            )
            if res2 is None:
                return None

            gantry_x2, gantry_y2, pt2 = res2
            bed_x2, bed_y2 = TransformEstimator.pixel_to_bed_space(
                pixel_u=pt2[0], pixel_v=pt2[1],
                frame_width=_cam_w, frame_height=_cam_h,
                camera_gantry_x=gantry_x2, camera_gantry_y=gantry_y2,
                mm_per_pixel=mm_per_pixel,
                camera_offset_x=camera_offset_x, camera_offset_y=camera_offset_y,
            )
            logger.info("Fiducial #2 LOCKED at Bed (X=%.3f, Y=%.3f)", bed_x2, bed_y2)

            # Step 3: Rigid Projection & Centering for Fiducial #3 (Front-Left)
            P12_x = gantry_x2 - gantry_x1
            P12_y = gantry_y2 - gantry_y1
            dist_d12 = math.hypot(d12_x, d12_y)
            dist_p12 = math.hypot(P12_x, P12_y)

            scale_factor = dist_p12 / max(1.0, dist_d12)
            theta_d = math.atan2(d12_y, d12_x)
            theta_p = math.atan2(P12_y, P12_x)
            d_theta = theta_p - theta_d

            logger.info("Sheet Geometry Solved: Scale=%.4f, Rotation=%.2f°", scale_factor, math.degrees(d_theta))

            d13_x = d3[0] - d1[0]
            d13_y = d3[1] - d1[1]

            proj_dx = scale_factor * (d13_x * math.cos(d_theta) - d13_y * math.sin(d_theta))
            proj_dy = scale_factor * (d13_x * math.sin(d_theta) + d13_y * math.cos(d_theta))

            pred_gantry_x3 = max(0.0, min(bed_max_x, gantry_x1 + proj_dx))
            pred_gantry_y3 = max(0.0, min(bed_max_y, gantry_y1 + proj_dy))

            res3 = FiducialSearcher.search_and_center(
                moonraker_client=moonraker_client,
                camera_client=camera_client,
                detector=detector,
                start_x=pred_gantry_x3,
                start_y=pred_gantry_y3,
                z_scan=z_scan,
                bed_max_x=bed_max_x,
                bed_max_y=bed_max_y,
                mm_per_pixel=mm_per_pixel,
                max_rings=max_rings_m3,
                settle_seconds=settle_seconds,
                expected_id=exp_id_3,
            )
            if res3 is None:
                return None

            gantry_x3, gantry_y3, pt3 = res3
            bed_x3, bed_y3 = TransformEstimator.pixel_to_bed_space(
                pixel_u=pt3[0], pixel_v=pt3[1],
                frame_width=_cam_w, frame_height=_cam_h,
                camera_gantry_x=gantry_x3, camera_gantry_y=gantry_y3,
                mm_per_pixel=mm_per_pixel,
                camera_offset_x=camera_offset_x, camera_offset_y=camera_offset_y,
            )
            logger.info("Fiducial #3 LOCKED at Bed (X=%.3f, Y=%.3f)", bed_x3, bed_y3)

            measured_bed = np.array([[bed_x1, bed_y1], [bed_x2, bed_y2], [bed_x3, bed_y3]], dtype=np.float64)
            measured_gantry = np.array([[gantry_x1, gantry_y1], [gantry_x2, gantry_y2], [gantry_x3, gantry_y3]], dtype=np.float64)
            return measured_bed, measured_gantry

        # Establish initial seed coordinate for Marker 1
        if nominal_fiducials is not None and len(nominal_fiducials) >= 1:
            seed_x1 = nominal_fiducials[0][0]
            seed_y1 = nominal_fiducials[0][1]
        else:
            seed_x1 = max(0.0, min(bed_max_x, d1[0] - camera_offset_x))
            seed_y1 = max(0.0, min(bed_max_y, d1[1] - camera_offset_y))

        # -------------------------------------------------------------
        # Preference 0: Assisted / Manual Paper Adjust (Wait at Marker 1)
        # Gantry moves to Marker 1 seed and pauses for user paper adjustment.
        # As soon as Marker 1 is spotted, it centers #1 and auto-locks #2 & #3.
        # -------------------------------------------------------------
        if mode in ("assisted", "manual", "paper_adjust"):
            logger.info("======================================================")
            logger.info("🎯 [ASSISTED ALIGN] Manual Paper Adjustment Mode Active")
            logger.info("Jogging camera to Marker 1 setup point (X=%.2f, Y=%.2f)...", seed_x1, seed_y1)
            logger.info("======================================================")
            moonraker_client.jog_to(x=seed_x1, y=seed_y1, z=z_scan, feedrate=3000.0, wait_finish=True)
            time.sleep(settle_seconds)

            exp_id_1 = 0 if detector.fiducial_type == "aruco" else None
            exp_id_2 = 1 if detector.fiducial_type == "aruco" else None
            exp_id_3 = 2 if detector.fiducial_type == "aruco" else None

            logger.info("📷 Camera is parked at Marker 1 setup point.")
            logger.info("👉 Please slide your paper sheet on the bed so Marker #1 (ArUco ID %s) appears in the live camera feed.", exp_id_1 if exp_id_1 is not None else "any")
            logger.info("⏳ Waiting for paper placement (Timeout: %.0fs)...", wait_for_user_timeout)

            start_wait = time.time()
            consecutive_hits = 0
            res1 = None

            while (time.time() - start_wait) < wait_for_user_timeout:
                elapsed = int(time.time() - start_wait)
                try:
                    frame = camera_client.get_frame()
                    pt = detector.detect(frame, expected_id=exp_id_1)
                except Exception as e:
                    logger.debug("Frame capture while waiting: %s", e)
                    pt = None

                if pt is not None:
                    consecutive_hits += 1
                    logger.info("👁️ Marker #1 spotted in camera frame! Verifying stability (%d/2)...", consecutive_hits)
                    if consecutive_hits >= 2:
                        logger.info("✅ Marker #1 steady! Centering with visual servoing...")
                        time.sleep(0.5)
                        res1 = FiducialSearcher.center_marker(
                            moonraker_client=moonraker_client,
                            camera_client=camera_client,
                            detector=detector,
                            gantry_x=seed_x1,
                            gantry_y=seed_y1,
                            z_scan=z_scan,
                            bed_max_x=bed_max_x,
                            bed_max_y=bed_max_y,
                            mm_per_pixel=mm_per_pixel,
                            settle_seconds=settle_seconds,
                            expected_id=exp_id_1,
                        )
                        break
                else:
                    consecutive_hits = 0
                    if elapsed > 0 and elapsed % 5 == 0:
                        logger.info("⏳ [ASSISTED ALIGN] Waiting for paper adjustment (%ds / %.0fs)... Slide sheet until Fiducial #1 is visible in camera.", elapsed, wait_for_user_timeout)

                time.sleep(0.7)

            if res1 is None:
                raise RuntimeError(f"Assisted alignment timed out after {wait_for_user_timeout:.0f} seconds without detecting Fiducial #1.")

            gantry_x1, gantry_y1, pt1 = res1
            bed_x1, bed_y1 = TransformEstimator.pixel_to_bed_space(
                pixel_u=pt1[0], pixel_v=pt1[1],
                frame_width=_cam_w, frame_height=_cam_h,
                camera_gantry_x=gantry_x1, camera_gantry_y=gantry_y1,
                mm_per_pixel=mm_per_pixel,
                camera_offset_x=camera_offset_x, camera_offset_y=camera_offset_y,
            )
            logger.info("✅ Fiducial #1 LOCKED at Bed (X=%.3f, Y=%.3f), Gantry (X=%.3f, Y=%.3f)", bed_x1, bed_y1, gantry_x1, gantry_y1)

            # Step 2: Auto-jog to predicted Fiducial #2
            d12_x = d2[0] - d1[0]
            d12_y = d2[1] - d1[1]
            pred_gantry_x2 = max(0.0, min(bed_max_x, gantry_x1 + d12_x))
            pred_gantry_y2 = max(0.0, min(bed_max_y, gantry_y1 + d12_y))

            logger.info("➡️ Auto-jogging to Fiducial #2 at Gantry (X=%.2f, Y=%.2f)...", pred_gantry_x2, pred_gantry_y2)
            res2 = FiducialSearcher.search_and_center(
                moonraker_client=moonraker_client,
                camera_client=camera_client,
                detector=detector,
                start_x=pred_gantry_x2,
                start_y=pred_gantry_y2,
                z_scan=z_scan,
                bed_max_x=bed_max_x,
                bed_max_y=bed_max_y,
                mm_per_pixel=mm_per_pixel,
                max_rings=2,
                settle_seconds=settle_seconds,
                expected_id=exp_id_2,
            )
            if res2 is None:
                raise RuntimeError("Failed to lock Fiducial #2. Please check that paper is not rotated excessively.")

            gantry_x2, gantry_y2, pt2 = res2
            bed_x2, bed_y2 = TransformEstimator.pixel_to_bed_space(
                pixel_u=pt2[0], pixel_v=pt2[1],
                frame_width=_cam_w, frame_height=_cam_h,
                camera_gantry_x=gantry_x2, camera_gantry_y=gantry_y2,
                mm_per_pixel=mm_per_pixel,
                camera_offset_x=camera_offset_x, camera_offset_y=camera_offset_y,
            )
            logger.info("✅ Fiducial #2 LOCKED at Bed (X=%.3f, Y=%.3f), Gantry (X=%.3f, Y=%.3f)", bed_x2, bed_y2, gantry_x2, gantry_y2)

            # Step 3: Rigid Projection & Centering for Fiducial #3
            P12_x = gantry_x2 - gantry_x1
            P12_y = gantry_y2 - gantry_y1
            dist_d12 = math.hypot(d12_x, d12_y)
            dist_p12 = math.hypot(P12_x, P12_y)

            scale_factor = dist_p12 / max(1.0, dist_d12)
            theta_d = math.atan2(d12_y, d12_x)
            theta_p = math.atan2(P12_y, P12_x)
            d_theta = theta_p - theta_d

            logger.info("Sheet Geometry Solved: Scale=%.4f, Rotation=%.2f°", scale_factor, math.degrees(d_theta))

            d13_x = d3[0] - d1[0]
            d13_y = d3[1] - d1[1]

            proj_dx = scale_factor * (d13_x * math.cos(d_theta) - d13_y * math.sin(d_theta))
            proj_dy = scale_factor * (d13_x * math.sin(d_theta) + d13_y * math.cos(d_theta))

            pred_gantry_x3 = max(0.0, min(bed_max_x, gantry_x1 + proj_dx))
            pred_gantry_y3 = max(0.0, min(bed_max_y, gantry_y1 + proj_dy))

            logger.info("➡️ Auto-jogging to Fiducial #3 at Gantry (X=%.2f, Y=%.2f)...", pred_gantry_x3, pred_gantry_y3)
            res3 = FiducialSearcher.search_and_center(
                moonraker_client=moonraker_client,
                camera_client=camera_client,
                detector=detector,
                start_x=pred_gantry_x3,
                start_y=pred_gantry_y3,
                z_scan=z_scan,
                bed_max_x=bed_max_x,
                bed_max_y=bed_max_y,
                mm_per_pixel=mm_per_pixel,
                max_rings=2,
                settle_seconds=settle_seconds,
                expected_id=exp_id_3,
            )
            if res3 is None:
                raise RuntimeError("Failed to lock Fiducial #3. Please check that paper is flat on bed.")

            gantry_x3, gantry_y3, pt3 = res3
            bed_x3, bed_y3 = TransformEstimator.pixel_to_bed_space(
                pixel_u=pt3[0], pixel_v=pt3[1],
                frame_width=_cam_w, frame_height=_cam_h,
                camera_gantry_x=gantry_x3, camera_gantry_y=gantry_y3,
                mm_per_pixel=mm_per_pixel,
                camera_offset_x=camera_offset_x, camera_offset_y=camera_offset_y,
            )
            logger.info("✅ Fiducial #3 LOCKED at Bed (X=%.3f, Y=%.3f), Gantry (X=%.3f, Y=%.3f)", bed_x3, bed_y3, gantry_x3, gantry_y3)

            measured_bed = np.array([[bed_x1, bed_y1], [bed_x2, bed_y2], [bed_x3, bed_y3]], dtype=np.float64)
            measured_gantry = np.array([[gantry_x1, gantry_y1], [gantry_x2, gantry_y2], [gantry_x3, gantry_y3]], dtype=np.float64)
            logger.info("🎉 [ASSISTED ALIGN] Complete! All 3 fiducials successfully acquired and locked.")
            return measured_bed, measured_gantry

        # -------------------------------------------------------------
        # Preference 1: Turbo / Fast (Direct Hits Only, ~5s)
        # -------------------------------------------------------------
        if mode in ("auto", "fast"):
            logger.info("🚀 [PREFERENCE 1: TURBO/FAST] Probing direct nominal coordinates...")
            res_fast = try_lock_three_markers(seed_x1, seed_y1, max_rings_m1=0, max_rings_m2=0, max_rings_m3=0)
            if res_fast is not None:
                logger.info("🚀 [PREFERENCE 1: TURBO/FAST] Succeeded! All 3 fiducials locked in 3 direct shots (~5s)!")
                return res_fast
            elif mode == "fast":
                raise RuntimeError("Turbo/Fast search failed: fiducial markers not visible at direct nominal positions.")
            logger.info("[PREFERENCE 1: TURBO/FAST] Direct hit missed. Escalating to Preference 2 (Adaptive Spiral)...")

        # -------------------------------------------------------------
        # Preference 2: Adaptive Spiral (Local Expansion, ~15s)
        # -------------------------------------------------------------
        if mode in ("auto", "spiral"):
            logger.info("🌀 [PREFERENCE 2: ADAPTIVE SPIRAL] Expanding spiral search around seed (X=%.1f, Y=%.1f)...", seed_x1, seed_y1)
            res_spiral = try_lock_three_markers(seed_x1, seed_y1, max_rings_m1=3, max_rings_m2=2, max_rings_m3=1)
            if res_spiral is not None:
                logger.info("⚡ [PREFERENCE 2: ADAPTIVE SPIRAL] Succeeded! Spiral discovery & rigid lock completed in ~15s!")
                return res_spiral
            elif mode == "spiral":
                raise RuntimeError("Adaptive Spiral search exhausted without locating fiducials in local region.")
            logger.info("[PREFERENCE 2: ADAPTIVE SPIRAL] Spiral search exhausted. Escalating to Preference 3 (Exhaustive Sweep)...")

        # -------------------------------------------------------------
        # Preference 3: Exhaustive Bed Sweep (Deep Discovery, ~45s)
        # -------------------------------------------------------------
        if mode in ("auto", "exhaustive"):
            logger.info("🔍 [PREFERENCE 3: EXHAUSTIVE SWEEP] Sweeping bed quadrants for unconstrained placement...")
            sweep_res = FiducialSearcher.sweep_bed_quadrants(
                moonraker_client=moonraker_client,
                camera_client=camera_client,
                detector=detector,
                z_scan=z_scan,
                bed_max_x=bed_max_x,
                bed_max_y=bed_max_y,
                mm_per_pixel=mm_per_pixel,
                settle_seconds=settle_seconds,
            )
            if sweep_res is not None:
                marker_id, act_x, act_y, pt = sweep_res
                # Reverse project Marker 1 seed from whichever marker was spotted
                if marker_id == 1:
                    cand_x1 = act_x - (d2[0] - d1[0])
                    cand_y1 = act_y - (d2[1] - d1[1])
                elif marker_id == 2:
                    cand_x1 = act_x - (d3[0] - d1[0])
                    cand_y1 = act_y - (d3[1] - d1[1])
                else:
                    cand_x1 = act_x
                    cand_y1 = act_y

                cand_x1 = max(0.0, min(bed_max_x, cand_x1))
                cand_y1 = max(0.0, min(bed_max_y, cand_y1))

                logger.info("🔍 [PREFERENCE 3] Anchor acquired (ID %d). Locking all 3 markers from (X=%.1f, Y=%.1f)...", marker_id, cand_x1, cand_y1)
                res_deep = try_lock_three_markers(cand_x1, cand_y1, max_rings_m1=2, max_rings_m2=2, max_rings_m3=1)
                if res_deep is not None:
                    logger.info("🎯 [PREFERENCE 3: EXHAUSTIVE SWEEP] Succeeded! Full bed recovery complete!")
                    return res_deep

        raise RuntimeError(f"Autonomous fiducial search failed under mode '{mode}'. No markers could be locked on bed.")



