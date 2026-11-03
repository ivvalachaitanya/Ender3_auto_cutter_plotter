"""
Computer Vision & Coordinate Transformation Module

Handles:
- Detection of ArUco markers (DICT_4X4_50) and circular fiducials with sub-pixel precision.
- Mapping image pixel coordinates to physical printer bed coordinates using camera offset and scale.
- 2D Rigid Affine Transformation estimation (cv2.estimateAffinePartial2D) between nominal SVG points and physical bed points.
"""

import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)

# Black square of the 10 mm marker image svg_injector makes (300 px marker + 2 x 40 px white border).
INNER_MARKER_MM = 10.0 * 300.0 / 380.0


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

    def detect_detailed(self, frame: np.ndarray) -> Dict:
        """
        Detect fiducials with geometry. Returns {id: {'center': (u, v), 'corners': 4x2 | None, 'side_px': float | None}}.
        Circle fiducials have no ID: the single best one is returned under key None.
        """
        if self.fiducial_type != "aruco":
            c = self.detect_circle_fiducial(frame)
            return {} if c is None else {None: {'center': c, 'corners': None, 'side_px': None}}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        if self.aruco_detector is not None:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        out = {}
        if ids is not None:
            for mid, c in zip(ids.flatten(), corners):
                pts = c[0].astype(np.float64)
                # sub-pixel corners make the heading (sheet rotation) and size (mm/px) precise
                try:
                    refined = cv2.cornerSubPix(gray, pts.astype(np.float32).reshape(-1, 1, 2), (5, 5), (-1, -1),
                                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
                    pts = refined.reshape(-1, 2).astype(np.float64)
                except cv2.error:
                    pass
                side = float(np.mean(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)))
                out[int(mid)] = {'center': (float(pts[:, 0].mean()), float(pts[:, 1].mean())), 'corners': pts, 'side_px': side}
        return out

    def find_marker_cue(self, frame: np.ndarray, marker_px: float) -> Optional[Tuple[float, float]]:
        """
        Estimate the centre (px, may lie outside the frame) of a marker that is only partly visible
        (cut off by the frame edge) or visible but not decodable. Returns None if nothing marker-like.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        if float(gray.std()) < 12.0:
            return None  # blank view: nothing to follow
        h, w = gray.shape[:2]
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_area = None, 0.0
        m2 = marker_px * marker_px
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = float(cv2.contourArea(cnt))
            if area < 0.06 * m2 or area > 1.25 * m2:
                continue
            if bw > 1.2 * marker_px or bh > 1.2 * marker_px:
                continue  # larger than a marker (bed edge, artwork)
            left, top, right, bottom = x <= 1, y <= 1, x + bw >= w - 1, y + bh >= h - 1
            if left + top + right + bottom >= 3:
                continue
            if area < 0.6 * bw * bh:
                continue  # not a filled square-ish blob (line art)
            if not (left or right) and abs(bw - marker_px) > 0.35 * marker_px:
                continue
            if not (top or bottom) and abs(bh - marker_px) > 0.35 * marker_px:
                continue
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            if left and not right:
                cx = x + bw - marker_px / 2.0
            elif right and not left:
                cx = x + marker_px / 2.0
            if top and not bottom:
                cy = y + bh - marker_px / 2.0
            elif bottom and not top:
                cy = y + marker_px / 2.0
            if area > best_area:
                best, best_area = (cx, cy), area
        return best

    def find_touch_target(self, frame: np.ndarray, mm_per_pixel: float) -> Optional[Tuple[float, float]]:
        """
        Find the centre (px, sub-pixel) of a printed touch-dot target: a 0.75 mm ring around a filled
        0.3 mm dot (as inject_verification_dots draws them). Picks the target nearest the image centre.
        Returns None when no target is in view.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h, w = gray.shape[:2]
        blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
        r_ring = 0.75 / mm_per_pixel
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=r_ring * 2, param1=80, param2=18,
                                   minRadius=int(r_ring * 0.7), maxRadius=int(r_ring * 1.35))
        if circles is None:
            return None
        background = float(np.median(gray))
        best = None
        for x, y, _r in circles[0]:
            xi, yi = int(round(x)), int(round(y))
            if not (3 <= xi < w - 3 and 3 <= yi < h - 3):
                continue
            if float(gray[yi - 2:yi + 3, xi - 2:xi + 3].mean()) > background - 25:
                continue  # no filled centre dot: some other circle, not a touch target
            d = math.hypot(x - w / 2.0, y - h / 2.0)
            if best is None or d < best[0]:
                best = (d, float(x), float(y))
        if best is None:
            return None
        _, x, y = best
        # Refine with the intensity-weighted centroid of the filled centre dot.
        rr = int(max(4, 0.5 / mm_per_pixel))
        x0, y0 = max(0, int(round(x)) - rr), max(0, int(round(y)) - rr)
        patch = gray[y0:y0 + 2 * rr + 1, x0:x0 + 2 * rr + 1].astype(np.float64)
        if patch.size:
            thr = (patch.min() + background) / 2.0
            wgt = np.clip(thr - patch, 0, None)
            yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
            wgt[np.hypot(xx + x0 - x, yy + y0 - y) > 0.45 / mm_per_pixel] = 0  # the dot only, not the ring
            if wgt.sum() > 0:
                x = float((wgt * (xx + x0)).sum() / wgt.sum())
                y = float((wgt * (yy + y0)).sum() / wgt.sum())
        return x, y

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
    Fiducial search and visual-servo centering.

    How it finds the three registration marks quickly:
      1. Search steps come from what the camera actually sees: a marker is only decodable when its
         whole black square is in frame, so the step is (field of view - marker size), leaving no gaps.
      2. Partially visible markers are followed: a dark square cut off by the frame edge gives a
         jump straight to the marker instead of continuing a blind pattern.
      3. Any marker ID can be the anchor; the others are predicted from it.
      4. The anchor's corners give the paper rotation (and the true mm/px from the known marker size),
         so the 170-180 mm jumps to the other marks land on target even when the sheet is skewed.
      5. After two marks, scale and rotation come from the marks themselves for the third.
    Statistics of the last run (frames, measured mm/px, camera twist) are kept in `last_run`.
    """

    last_run: Dict = {}

    class _Scan:
        def __init__(self, moonraker_client, camera_client, detector, z_scan, bed_max_x, bed_max_y,
                     camera_offset_x, camera_offset_y, mm_per_pixel, settle_seconds, marker_size_mm,
                     camera_rotation_deg):
            self.mr = moonraker_client
            self.cam = camera_client
            self.det = detector
            self.z = z_scan
            self.bx, self.by = bed_max_x, bed_max_y
            self.off = np.array([camera_offset_x, camera_offset_y], dtype=np.float64)
            self.mpp = mm_per_pixel
            self.settle = settle_seconds
            self.marker_mm = marker_size_mm
            self.cam_rot = math.radians(camera_rotation_deg)
            self.frames = 0
            self.mpp_samples: List[float] = []
            try:
                probe = self.cam.get_frame()
                self.h, self.w = probe.shape[:2]
            except Exception:
                self.w, self.h = 640, 480
            fov_w, fov_h = self.w * self.mpp, self.h * self.mpp
            # Half-width of the region where a marker centre still gives a decodable full marker.
            self.win = np.array([max(0.75, (fov_w - self.marker_mm) / 2.0 - 0.4),
                                 max(0.75, (fov_h - self.marker_mm) / 2.0 - 0.4)])
            # Half-width where at least part of the marker is in frame (a cue can be followed).
            self.cue_win = np.array([(fov_w + self.marker_mm) / 2.0 - 1.0, (fov_h + self.marker_mm) / 2.0 - 1.0])
            logger.info("Camera view %.1f x %.1f mm (%dx%d px @ %.4f mm/px); marker %.1f mm -> decode window +/-%.1f x +/-%.1f mm.",
                        fov_w, fov_h, self.w, self.h, self.mpp, self.marker_mm, self.win[0], self.win[1])

        # ---- geometry ----
        def img_vec_to_bed(self, du: float, dv: float, mpp: Optional[float] = None) -> np.ndarray:
            """Image displacement (px) -> bed displacement (mm). Matches pixel_to_bed_space (+ camera twist)."""
            m = self.mpp if mpp is None else mpp
            ix, iy = -du * m, dv * m
            c, s = math.cos(self.cam_rot), math.sin(self.cam_rot)
            return np.array([c * ix - s * iy, s * ix + c * iy])

        def pix_to_bed(self, u: float, v: float, gantry: np.ndarray, mpp: Optional[float] = None) -> np.ndarray:
            return gantry + self.off + self.img_vec_to_bed(u - self.w / 2.0, v - self.h / 2.0, mpp)

        def gantry_for(self, bed_pt: np.ndarray) -> np.ndarray:
            g = np.asarray(bed_pt, dtype=np.float64) - self.off
            return np.array([min(max(g[0], 0.0), self.bx), min(max(g[1], 0.0), self.by)])

        def reach_shortfall(self, bed_pt: np.ndarray) -> np.ndarray:
            """How far (mm, signed per axis) a bed point lies outside what the camera can be centred on."""
            g = np.asarray(bed_pt, dtype=np.float64) - self.off
            return g - self.gantry_for(bed_pt)

        def safe_anchor_box(self, design: np.ndarray, max_rot_deg: float = 4.0, margin: float = 1.5):
            """Bed region for mark #0 such that all marks stay within camera reach for sheet
            rotations up to +/-max_rot_deg. Returns (lo, hi) or None if no such region exists."""
            lo = np.array([-np.inf, -np.inf])
            hi = np.array([np.inf, np.inf])
            reach_lo = self.off + margin
            reach_hi = np.array([self.bx, self.by]) + self.off - margin
            for rot in np.radians(np.linspace(-max_rot_deg, max_rot_deg, 9)):
                c, s_ = math.cos(rot), math.sin(rot)
                for d in design:
                    rel = d - design[0]
                    r = np.array([c * rel[0] - s_ * rel[1], s_ * rel[0] + c * rel[1]])
                    lo = np.maximum(lo, reach_lo - r)
                    hi = np.minimum(hi, reach_hi - r)
            return (lo, hi) if np.all(lo <= hi) else None

        def heading(self, det: Dict) -> Optional[float]:
            """Marker rotation on the bed (rad) from its top edge, or None (circle fiducials)."""
            c = det.get('corners')
            if c is None:
                return None
            v = self.img_vec_to_bed(float(c[1][0] - c[0][0]), float(c[1][1] - c[0][1]))
            return math.atan2(v[1], v[0])

        def marker_mpp(self, det: Dict) -> Optional[float]:
            side = det.get('side_px')
            if not side:
                return None
            m = self.marker_mm / side
            return m if 0.85 * self.mpp < m < 1.15 * self.mpp else None

        def eff_mpp(self) -> float:
            return float(np.median(self.mpp_samples)) if self.mpp_samples else self.mpp

        # ---- camera ----
        def look(self, gantry: np.ndarray):
            self.mr.jog_to(x=float(gantry[0]), y=float(gantry[1]), z=self.z, feedrate=3000.0, wait_finish=True)
            time.sleep(self.settle)
            try:
                frame = self.cam.get_frame()
            except Exception as e:
                logger.warning("Frame grab failed: %s", e)
                return None, {}, gantry
            self.frames += 1
            try:
                x, y, _ = self.mr.get_toolhead_position()
                gantry = np.array([x, y], dtype=np.float64)
            except Exception:
                pass
            return frame, self.det.detect_detailed(frame), gantry

        # ---- search ----
        def search(self, target_bed: np.ndarray, radius_mm: float, want: Optional[set] = None,
                   coarse: bool = False):
            """
            Look around `target_bed` (nearest first) until a marker decodes.
            Returns (dets, gantry) or None. `coarse` uses cue-sized steps (relies on partial-marker cues).
            """
            half = self.cue_win * 0.9 if coarse else self.win * 0.95
            step = 2.0 * half
            nx = int(math.ceil(radius_mm / step[0]))
            ny = int(math.ceil(radius_mm / step[1]))
            offsets = [(i * step[0], j * step[1]) for i in range(-nx, nx + 1) for j in range(-ny, ny + 1)
                       if math.hypot(i * step[0], j * step[1]) <= radius_mm + 1e-6 or (i == 0 and j == 0)]
            offsets.sort(key=lambda o: math.hypot(o[0], o[1]))
            visited: List[np.ndarray] = []
            marker_px = self.marker_mm / self.mpp
            for ox, oy in offsets:
                g = self.gantry_for(np.asarray(target_bed) + np.array([ox, oy]))
                if any(np.all(np.abs(g - p) < half * 0.5) for p in visited):
                    continue  # clamped at the bed edge onto a spot already seen
                visited.append(g)
                frame, dets, g_act = self.look(g)
                if frame is None:
                    continue
                if dets and (want is None or want & set(dets) or None in dets):
                    return dets, g_act
                # Follow a partially visible marker (up to 2 hops).
                for _hop in range(2):
                    cue = self.det.find_marker_cue(frame, marker_px)
                    if cue is None:
                        break
                    cue_bed = self.pix_to_bed(cue[0], cue[1], g_act)
                    if np.linalg.norm(cue_bed - np.asarray(target_bed)) > radius_mm + float(np.max(self.cue_win)) + 2.0:
                        break
                    g2 = self.gantry_for(cue_bed)
                    visited.append(g2)
                    frame, dets, g_act = self.look(g2)
                    if frame is None:
                        break
                    if dets and (want is None or want & set(dets) or None in dets):
                        logger.info("Followed a partially visible marker to (%.1f, %.1f).", cue_bed[0], cue_bed[1])
                        return dets, g_act
            return None

        def center_on(self, mid, gantry: np.ndarray, dets: Dict, max_iters: int = 3) -> Optional[Dict]:
            """Servo the camera onto marker `mid`; returns its bed/gantry position and measurements."""
            det = dets[mid]
            m = self.marker_mpp(det)
            if m:
                self.mpp_samples.append(m)
            head = self.heading(det)
            for _ in range(max_iters):
                du, dv = det['center'][0] - self.w / 2.0, det['center'][1] - self.h / 2.0
                if abs(du) < 1.5 and abs(dv) < 1.5:
                    break
                bed = self.pix_to_bed(det['center'][0], det['center'][1], gantry, self.eff_mpp())
                target = self.gantry_for(bed)
                if np.linalg.norm(target - gantry) < 0.02:
                    break  # at the bed limit: cannot centre further
                frame, new, gantry = self.look(target)
                if mid not in new:
                    logger.warning("Lost marker %s while centring; using last good view.", mid)
                    frame, new, gantry = self.look(gantry)
                    if mid not in new:
                        break
                det = new[mid]
                m = self.marker_mpp(det)
                if m:
                    self.mpp_samples.append(m)
                h2 = self.heading(det)
                head = h2 if h2 is not None else head
            bed = self.pix_to_bed(det['center'][0], det['center'][1], gantry, self.eff_mpp())
            return {'bed': bed, 'gantry': np.array(gantry, dtype=np.float64), 'heading': head, 'pixel': det['center']}

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
        marker_size_mm: float = INNER_MARKER_MM,
        camera_rotation_deg: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find, centre and measure the three registration marks.

        Search modes (all lock the 2nd/3rd marks by prediction from the first one found):
        - 'assisted' / 'manual': park at the expected spot of mark #1 and wait for the sheet to be
          slid under the camera (any of the three marks is accepted).
        - 'fast': one look at the expected spot (plus following a partly visible mark).
        - 'spiral': gap-free search within 25 mm of the expected spot.
        - 'exhaustive': nearest-first sweep of the whole bed.
        - 'auto': 'spiral', then 'exhaustive'.

        :return: (measured_bed_points (3x2, mark order 0,1,2), measured_gantry_points (3x2))
        """
        mode = (search_mode or "auto").lower().strip()
        design = np.asarray(design_fiducials, dtype=np.float64)[:3]
        scan = FiducialSearcher._Scan(moonraker_client, camera_client, detector, z_scan, bed_max_x, bed_max_y,
                                      camera_offset_x, camera_offset_y, mm_per_pixel, settle_seconds,
                                      marker_size_mm, camera_rotation_deg)
        circle_mode = detector.fiducial_type != "aruco"
        all_ids = [0, 1, 2]

        logger.info("======================================================")
        logger.info("Fiducial search | mode '%s' | design marks %s", mode.upper(), design.tolist())
        logger.info("======================================================")

        if nominal_fiducials is not None and len(nominal_fiducials) >= 1:
            seed_bed = np.asarray(nominal_fiducials[0], dtype=np.float64) + scan.off
        else:
            seed_bed = design[0].copy()

        found: Dict[int, Dict] = {}

        def record(dets: Dict, gantry: np.ndarray, expected: int):
            """Centre on the most useful marker in `dets` and store it."""
            if circle_mode:
                mid_key, mid = None, expected
            else:
                mids = [k for k in dets if k in all_ids and k not in found]
                if not mids:
                    return None
                mid = expected if expected in mids else min(mids)
                mid_key = mid
            res = scan.center_on(mid_key, gantry, dets)
            if res is None:
                return None
            found[mid] = res
            logger.info("Mark #%d LOCKED at bed (X=%.3f, Y=%.3f), gantry (X=%.3f, Y=%.3f)%s",
                        mid, res['bed'][0], res['bed'][1], res['gantry'][0], res['gantry'][1],
                        "" if res['heading'] is None else f", sheet rotation {math.degrees(res['heading']):+.2f} deg")
            return mid

        # ---------------- 1. find the first (anchor) mark ----------------
        anchor = None
        if mode in ("assisted", "manual", "paper_adjust"):
            # The sheet is slid to wherever the camera waits, so wait at a spot from which all three
            # marks remain reachable even if the sheet ends up a few degrees skewed.
            box = scan.safe_anchor_box(design)
            if box is None:
                logger.warning("[ASSISTED] The marks are too far apart for the camera to reach all of them; check design_fiducials.")
            else:
                safe = np.minimum(np.maximum(seed_bed, box[0]), box[1])
                if np.linalg.norm(safe - seed_bed) > 0.5:
                    logger.info("[ASSISTED] Moved the waiting spot from bed (%.1f, %.1f) to (%.1f, %.1f) so all marks stay within camera reach.",
                                seed_bed[0], seed_bed[1], safe[0], safe[1])
                seed_bed = safe
            g = scan.gantry_for(seed_bed)
            logger.info("[ASSISTED] Camera parked at the expected position of mark #1 (gantry X=%.1f, Y=%.1f).", g[0], g[1])
            logger.info("[ASSISTED] Slide the sheet until a registration mark is in the camera view (timeout %.0fs)...", wait_for_user_timeout)
            moonraker_client.jog_to(x=float(g[0]), y=float(g[1]), z=z_scan, feedrate=3000.0, wait_finish=True)
            time.sleep(settle_seconds)
            start = time.time()
            last_ids, hits, last_log = None, 0, -1
            while time.time() - start < wait_for_user_timeout:
                frame = None
                try:
                    frame = camera_client.get_frame()
                    scan.frames += 1
                    dets = detector.detect_detailed(frame)
                except Exception as e:
                    logger.debug("Frame capture while waiting: %s", e)
                    dets = {}
                ids = frozenset(dets)
                if dets and ids == last_ids:
                    hits += 1
                else:
                    hits = 1 if dets else 0
                last_ids = ids
                if hits >= 2:
                    logger.info("[ASSISTED] Mark visible and steady, locking...")
                    anchor = record(dets, g, 0)
                    break
                if not dets:
                    # Part of a mark is in view but not decodable: move the camera onto it
                    # (only near the waiting spot, so the camera never wanders off).
                    cue = detector.find_marker_cue(frame, scan.marker_mm / scan.mpp) if frame is not None else None
                    if cue is not None:
                        cue_bed = scan.pix_to_bed(cue[0], cue[1], g)
                        if np.linalg.norm(cue_bed - seed_bed) < 15.0:
                            g_new = scan.gantry_for(cue_bed)
                            if np.linalg.norm(g_new - g) > 0.5:
                                logger.info("[ASSISTED] Part of a mark is visible; moving the camera onto it.")
                                moonraker_client.jog_to(x=float(g_new[0]), y=float(g_new[1]), z=z_scan, feedrate=3000.0, wait_finish=True)
                                time.sleep(settle_seconds)
                                g = g_new
                                continue
                elapsed = int(time.time() - start)
                if elapsed % 5 == 0 and elapsed != last_log:
                    last_log = elapsed
                    logger.info("[ASSISTED] Waiting for the sheet (%ds / %.0fs)...", elapsed, wait_for_user_timeout)
                time.sleep(0.7)
            if anchor is None:
                raise RuntimeError(f"Assisted alignment timed out after {wait_for_user_timeout:.0f}s without seeing a registration mark.")
        else:
            res = None
            if mode in ("fast",):
                res = scan.search(seed_bed, 0.0)
            elif mode in ("spiral", "auto"):
                res = scan.search(seed_bed, 25.0)
            if res is None and mode in ("auto", "exhaustive"):
                logger.info("Not near the expected spot: sweeping the whole bed (nearest first)...")
                corners = [np.array([0.0, 0.0]), np.array([bed_max_x, 0.0]), np.array([0.0, bed_max_y]), np.array([bed_max_x, bed_max_y])]
                reach = max(float(np.linalg.norm(c - seed_bed)) for c in corners)
                res = scan.search(seed_bed, reach, coarse=True)
            if res is None:
                raise RuntimeError(f"Fiducial search ('{mode}') found no registration mark. "
                                   f"Check the sheet is under the camera near bed (X={seed_bed[0]:.0f}, Y={seed_bed[1]:.0f}) and the marks are printed sharp.")
            anchor = record(res[0], res[1], 0)
            if anchor is None:
                raise RuntimeError("A registration mark was seen but could not be centred.")

        # ---------------- 2. predict and lock the remaining marks ----------------
        def predict(mid: int) -> Tuple[np.ndarray, float]:
            """Predicted bed position of mark `mid` and a search radius."""
            known = sorted(found)
            if len(known) >= 2:
                a, b = known[0], known[1]
                da, db = design[a], design[b]
                pa, pb = found[a]['bed'], found[b]['bed']
                vd, vp = db - da, pb - pa
                ang = math.atan2(vp[1], vp[0]) - math.atan2(vd[1], vd[0])
                sc = float(np.linalg.norm(vp) / max(np.linalg.norm(vd), 1e-9))
                c, s = math.cos(ang), math.sin(ang)
                rel = design[mid] - da
                return pa + sc * np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]]), 6.0
            k = known[0]
            head = found[k]['heading']
            ang = head if head is not None else 0.0
            c, s = math.cos(ang), math.sin(ang)
            rel = design[mid] - design[k]
            radius = 8.0 if head is not None else 25.0
            return found[k]['bed'] + np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]]), radius

        for _ in range(4):
            missing = [m for m in all_ids if m not in found]
            if not missing:
                break
            # nearest predicted first keeps travel short
            preds = {m: predict(m) for m in missing}
            mid = min(missing, key=lambda m: float(np.linalg.norm(preds[m][0] - found[max(found)]['bed'])))
            target, radius = preds[mid]
            short = scan.reach_shortfall(target)
            if np.any(np.abs(short) > scan.win):
                moves = []
                if short[0] < 0:
                    moves.append(f"{-short[0]:.0f} mm to the right (+X)")
                elif short[0] > 0:
                    moves.append(f"{short[0]:.0f} mm to the left (-X)")
                if short[1] < 0:
                    moves.append(f"{-short[1]:.0f} mm towards the back (+Y)")
                elif short[1] > 0:
                    moves.append(f"{short[1]:.0f} mm towards the front (-Y)")
                raise RuntimeError(f"Registration mark #{mid} is at bed (X={target[0]:.1f}, Y={target[1]:.1f}), outside the camera's reach. "
                                   f"Move the sheet {' and '.join(moves)} (or straighten it) and scan again.")
            logger.info("Mark #%d predicted at bed (X=%.2f, Y=%.2f); searching within %.0f mm...", mid, target[0], target[1], radius)
            res = scan.search(target, radius, want=None if circle_mode else set(missing))
            if res is None and radius < 20.0:
                logger.info("Mark #%d not at the predicted spot; widening search to 20 mm...", mid)
                res = scan.search(target, 20.0, want=None if circle_mode else set(missing))
            if res is None:
                raise RuntimeError(f"Could not find registration mark #{mid} near bed (X={target[0]:.1f}, Y={target[1]:.1f}). "
                                   "Check the sheet is flat and the mark is printed and not covered.")
            record(res[0], res[1], mid)

        missing = [m for m in all_ids if m not in found]
        if missing:
            raise RuntimeError(f"Registration marks {missing} could not be locked.")

        bed_pts = np.array([found[m]['bed'] for m in all_ids], dtype=np.float64)
        gantry_pts = np.array([found[m]['gantry'] for m in all_ids], dtype=np.float64)

        # Self-calibration hints: camera twist (anchor heading vs. the true sheet rotation) and mm/px.
        stats = {'frames': scan.frames, 'mode': mode, 'measured_mm_per_pixel': scan.eff_mpp() if scan.mpp_samples else None}
        vd, vp = design[1] - design[0], bed_pts[1] - bed_pts[0]
        sheet_rot = math.atan2(vp[1], vp[0]) - math.atan2(vd[1], vd[0])
        heads = [found[m]['heading'] for m in all_ids if found[m]['heading'] is not None]
        if heads:
            twist = math.degrees(math.atan2(np.mean([math.sin(h - sheet_rot) for h in heads]),
                                            np.mean([math.cos(h - sheet_rot) for h in heads])))
            stats['camera_rotation_deg'] = camera_rotation_deg + twist
        stats['sheet_rotation_deg'] = math.degrees(sheet_rot)
        FiducialSearcher.last_run = stats
        logger.info("All 3 marks locked in %d camera frames. Sheet rotation %+.2f deg.%s%s", scan.frames, math.degrees(sheet_rot),
                    f" Measured {stats['measured_mm_per_pixel']:.5f} mm/px (config {mm_per_pixel:.5f})." if stats['measured_mm_per_pixel'] else "",
                    f" Camera twist {stats['camera_rotation_deg']:+.2f} deg." if 'camera_rotation_deg' in stats else "")
        return bed_pts, gantry_pts
