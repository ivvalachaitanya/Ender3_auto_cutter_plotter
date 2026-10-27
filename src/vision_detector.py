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
        settle_seconds: float = 0.35,
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
        pt = detector.detect(frame)
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
                new_pt = detector.detect(new_frame)
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
        settle_seconds: float = 0.35,
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
                moonraker_client.jog_to(x=target_x, y=target_y, z=z_scan, feedrate=4000.0, wait_finish=True)
                time.sleep(settle_seconds)
                last_gantry = (target_x, target_y)

                try:
                    frame = camera_client.get_frame()
                except Exception as e:
                    logger.warning("Failed to grab camera frame during search: %s", e)
                    continue

                pt = detector.detect(frame)
                if pt is not None:
                    logger.info("Fiducial marker spotted at gantry (X=%.2f, Y=%.2f)! Centering...", target_x, target_y)
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full Autonomous 3-Fiducial Discovery & Visual Servoing Alignment Pipeline.

        1. Discovers & centers Fiducial #1 (Rear-Left) via spiral search around seed.
        2. Projects design vector D12 to predict Fiducial #2 (Rear-Right), searches & centers.
        3. Computes physical sheet scale and rotation angle from P12 vector.
        4. Projects exact rigid position for Fiducial #3 (Front-Left), centers with 0 hops.

        :return: (measured_bed_points (Nx2), measured_gantry_points (Nx2))
        """
        import math
        logger.info("======================================================")
        logger.info("Starting Autonomous Fiducial Discovery & Alignment")
        logger.info("Design Fiducials: %s", design_fiducials.tolist())
        logger.info("======================================================")

        d1 = design_fiducials[0]
        d2 = design_fiducials[1]
        d3 = design_fiducials[2]

        # -------------------------------------------------------------
        # Step 1: Discover & Center Fiducial #1 (Rear-Left)
        # -------------------------------------------------------------
        if nominal_fiducials is not None and len(nominal_fiducials) >= 1:
            start_x1 = nominal_fiducials[0][0]
            start_y1 = nominal_fiducials[0][1]
        else:
            start_x1 = max(0.0, min(bed_max_x, d1[0] - camera_offset_x))
            start_y1 = max(0.0, min(bed_max_y, d1[1] - camera_offset_y))

        logger.info("[AUTONOMOUS SEARCH] Step 1/3: Locating Fiducial #1 around (X=%.2f, Y=%.2f)...", start_x1, start_y1)
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
            max_rings=3,
        )
        if res1 is None:
            raise RuntimeError("Autonomous search failed to locate Fiducial #1 on bed.")

        gantry_x1, gantry_y1, pt1 = res1
        bed_x1, bed_y1 = TransformEstimator.pixel_to_bed_space(
            pixel_u=pt1[0], pixel_v=pt1[1],
            frame_width=640, frame_height=480,
            camera_gantry_x=gantry_x1, camera_gantry_y=gantry_y1,
            mm_per_pixel=mm_per_pixel,
            camera_offset_x=camera_offset_x, camera_offset_y=camera_offset_y,
        )
        logger.info("[AUTONOMOUS SEARCH] -> Fiducial #1 LOCKED at Bed (X=%.3f, Y=%.3f)", bed_x1, bed_y1)

        # -------------------------------------------------------------
        # Step 2: Predict, Search & Center Fiducial #2 (Rear-Right)
        # -------------------------------------------------------------
        d12_x = d2[0] - d1[0]
        d12_y = d2[1] - d1[1]
        pred_gantry_x2 = max(0.0, min(bed_max_x, gantry_x1 + d12_x))
        pred_gantry_y2 = max(0.0, min(bed_max_y, gantry_y1 + d12_y))

        logger.info("[AUTONOMOUS SEARCH] Step 2/3: Predicting Fiducial #2 at gantry (X=%.2f, Y=%.2f)...", pred_gantry_x2, pred_gantry_y2)
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
        )
        if res2 is None:
            raise RuntimeError("Autonomous search failed to locate Fiducial #2 on bed.")

        gantry_x2, gantry_y2, pt2 = res2
        bed_x2, bed_y2 = TransformEstimator.pixel_to_bed_space(
            pixel_u=pt2[0], pixel_v=pt2[1],
            frame_width=640, frame_height=480,
            camera_gantry_x=gantry_x2, camera_gantry_y=gantry_y2,
            mm_per_pixel=mm_per_pixel,
            camera_offset_x=camera_offset_x, camera_offset_y=camera_offset_y,
        )
        logger.info("[AUTONOMOUS SEARCH] -> Fiducial #2 LOCKED at Bed (X=%.3f, Y=%.3f)", bed_x2, bed_y2)

        # -------------------------------------------------------------
        # Step 3: Solve Rigid Body Transform & Predict Fiducial #3 (Front-Left)
        # -------------------------------------------------------------
        P12_x = gantry_x2 - gantry_x1
        P12_y = gantry_y2 - gantry_y1
        dist_d12 = math.hypot(d12_x, d12_y)
        dist_p12 = math.hypot(P12_x, P12_y)

        scale_factor = dist_p12 / max(1.0, dist_d12)
        theta_d = math.atan2(d12_y, d12_x)
        theta_p = math.atan2(P12_y, P12_x)
        d_theta = theta_p - theta_d

        logger.info("[AUTONOMOUS SEARCH] Sheet Geometry Solved: Scale=%.4f, Rotation=%.2f°", scale_factor, math.degrees(d_theta))

        # Project D13 vector with exact scale and rotation
        d13_x = d3[0] - d1[0]
        d13_y = d3[1] - d1[1]

        proj_dx = scale_factor * (d13_x * math.cos(d_theta) - d13_y * math.sin(d_theta))
        proj_dy = scale_factor * (d13_x * math.sin(d_theta) + d13_y * math.cos(d_theta))

        pred_gantry_x3 = max(0.0, min(bed_max_x, gantry_x1 + proj_dx))
        pred_gantry_y3 = max(0.0, min(bed_max_y, gantry_y1 + proj_dy))

        logger.info("[AUTONOMOUS SEARCH] Step 3/3: Rigid Projection for Fiducial #3 -> Target Gantry (X=%.2f, Y=%.2f)...", pred_gantry_x3, pred_gantry_y3)
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
            max_rings=1,
        )
        if res3 is None:
            raise RuntimeError("Autonomous search failed to locate Fiducial #3 on bed.")

        gantry_x3, gantry_y3, pt3 = res3
        bed_x3, bed_y3 = TransformEstimator.pixel_to_bed_space(
            pixel_u=pt3[0], pixel_v=pt3[1],
            frame_width=640, frame_height=480,
            camera_gantry_x=gantry_x3, camera_gantry_y=gantry_y3,
            mm_per_pixel=mm_per_pixel,
            camera_offset_x=camera_offset_x, camera_offset_y=camera_offset_y,
        )
        logger.info("[AUTONOMOUS SEARCH] -> Fiducial #3 LOCKED at Bed (X=%.3f, Y=%.3f)", bed_x3, bed_y3)

        measured_bed_pts = np.array([
            [bed_x1, bed_y1],
            [bed_x2, bed_y2],
            [bed_x3, bed_y3],
        ], dtype=np.float64)

        measured_gantry_pts = np.array([
            [gantry_x1, gantry_y1],
            [gantry_x2, gantry_y2],
            [gantry_x3, gantry_y3],
        ], dtype=np.float64)

        return measured_bed_pts, measured_gantry_pts



