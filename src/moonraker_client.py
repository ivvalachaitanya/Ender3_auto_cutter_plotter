"""
Moonraker & Crowsnest HTTP Client Module

Provides interface for:
- Moonraker REST API (G-code execution, positioning, G-code file upload, print execution)
- Camera frame acquisition (Crowsnest HTTP snapshot stream with local OpenCV VideoCapture fallback)
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import requests

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


class MoonrakerClient:
    """Client for interacting with Klipper via Moonraker REST API."""

    def __init__(self, base_url: str = "http://localhost:7125", timeout: float = 10.0):
        """
        Initialize Moonraker HTTP client.

        :param base_url: Base URL of Moonraker server (e.g. http://192.168.1.50:7125)
        :param timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def is_connected(self) -> bool:
        """Check if Moonraker API is reachable."""
        try:
            resp = self.session.get(f"{self.base_url}/printer/info", timeout=self.timeout)
            return resp.status_code == 200
        except requests.RequestException as e:
            logger.warning("Moonraker connection check failed: %s", e)
            return False

    def send_gcode(self, script: str) -> bool:
        """
        Execute a G-code script on Klipper.

        :param script: G-code command string or multi-line block
        :return: True if successful, False otherwise
        """
        url = f"{self.base_url}/printer/gcode/script"
        try:
            resp = self.session.post(url, json={"script": script}, timeout=self.timeout)
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error("Failed to send G-code '%s': %s", script, e)
            return False

    def get_toolhead_position(self) -> Tuple[float, float, float]:
        """
        Query current toolhead position (X, Y, Z).

        :return: Tuple of (X, Y, Z) coordinates in mm
        """
        url = f"{self.base_url}/printer/objects/query?toolhead"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            pos = data["result"]["status"]["toolhead"]["position"]
            return float(pos[0]), float(pos[1]), float(pos[2])
        except (requests.RequestException, KeyError, IndexError, ValueError) as e:
            logger.error("Failed to query toolhead position: %s", e)
            raise RuntimeError(f"Could not retrieve toolhead position from Moonraker: {e}")

    def home_gantry(self, wait_seconds: float = 10.0) -> bool:
        """
        Execute homing sequence (G28).

        :param wait_seconds: Time to wait after homing command
        :return: True if successful
        """
        logger.info("Executing printer homing sequence (G28)...")
        if not self.send_gcode("G28"):
            return False
        time.sleep(wait_seconds)
        return True

    def jog_to(self, x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None, feedrate: float = 3000.0) -> bool:
        """
        Jog gantry to absolute coordinates.

        :param x: Target X position in mm
        :param y: Target Y position in mm
        :param z: Target Z position in mm
        :param feedrate: Travel speed in mm/min
        :return: True if successful
        """
        gcode_parts = ["G90", f"G0 F{feedrate:.1f}"]
        move_args = []
        if x is not None:
            move_args.append(f"X{x:.3f}")
        if y is not None:
            move_args.append(f"Y{y:.3f}")
        if z is not None:
            move_args.append(f"Z{z:.3f}")

        if not move_args:
            return True

        gcode_parts.append("G0 " + " ".join(move_args))
        script = "\n".join(gcode_parts)
        return self.send_gcode(script)

    def upload_gcode(self, file_path: str, target_filename: Optional[str] = None) -> str:
        """
        Upload G-code file to Moonraker gcodes directory.

        :param file_path: Local path to G-code file
        :param target_filename: Optional override for target filename in Moonraker
        :return: Moonraker internal filename
        """
        url = f"{self.base_url}/server/files/upload"
        filename = target_filename or file_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        logger.info("Uploading G-code file %s to Moonraker...", filename)

        try:
            with open(file_path, "rb") as f:
                files = {"file": (filename, f, "application/octet-stream")}
                resp = self.session.post(url, files=files, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                return data["item"]["path"]
        except (requests.RequestException, KeyError) as e:
            logger.error("Failed to upload G-code file to Moonraker: %s", e)
            raise RuntimeError(f"G-code upload failed: {e}")

    def start_print(self, filename: str) -> bool:
        """
        Start executing a print job on Klipper.

        :param filename: Filename in Moonraker gcodes storage
        :return: True if successfully started
        """
        url = f"{self.base_url}/printer/print/start"
        logger.info("Starting print job for '%s'...", filename)
        try:
            resp = self.session.post(url, json={"filename": filename}, timeout=self.timeout)
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error("Failed to start print job: %s", e)
            return False


class CameraClient:
    """Client for pulling frame snapshots from Crowsnest stream or USB camera."""

    def __init__(self, snapshot_url: str = "http://localhost:8080/?action=snapshot", camera_index: int = 0, timeout: float = 5.0):
        """
        Initialize camera frame client.

        :param snapshot_url: Crowsnest / MJPG-streamer snapshot URL
        :param camera_index: OpenCV local camera index if HTTP fallback is needed
        :param timeout: HTTP request timeout in seconds
        """
        self.snapshot_url = snapshot_url
        self.camera_index = camera_index
        self.timeout = timeout
        self.session = requests.Session()

    def get_frame(self) -> np.ndarray:
        """
        Fetch current image frame from camera.

        Tries snapshot URL first; falls back to local OpenCV VideoCapture if URL fails.
        :return: BGR image numpy array
        """
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is not installed. Required for image processing.")

        # Attempt 1: HTTP Snapshot stream
        if self.snapshot_url:
            try:
                resp = self.session.get(self.snapshot_url, timeout=self.timeout)
                if resp.status_code == 200 and len(resp.content) > 0:
                    img_array = np.frombuffer(resp.content, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is not None and frame.size > 0:
                        return frame
            except requests.RequestException as e:
                logger.warning("Crowsnest HTTP snapshot request failed (%s). Trying local camera index %d...", e, self.camera_index)

        # Attempt 2: Local OpenCV VideoCapture fallback
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open local camera device at index {self.camera_index} and HTTP snapshot failed.")

        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError(f"Failed to read frame from OpenCV camera index {self.camera_index}.")
            return frame
        finally:
            cap.release()

    def set_camera_autofocus(self, enable: bool) -> bool:
        """
        Enable or disable hardware camera autofocus.

        :param enable: True to enable autofocus, False to disable/lock focus
        :return: True if property set successfully
        """
        if cv2 is None:
            return False

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            return False

        try:
            val = 1.0 if enable else 0.0
            success = cap.set(cv2.CAP_PROP_AUTOFOCUS, val)
            logger.info("Camera autofocus set to %s (success=%s)", enable, success)
            return success
        finally:
            cap.release()

    def set_camera_focus(self, focus_value: float) -> bool:
        """
        Set manual camera focus value.

        :param focus_value: Manual focus setting (typically 0.0 to 255.0 depending on driver)
        :return: True if property set successfully
        """
        if cv2 is None:
            return False

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            return False

        try:
            # Disable autofocus first for manual setting to take effect
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0)
            success = cap.set(cv2.CAP_PROP_FOCUS, focus_value)
            logger.info("Camera manual focus set to %.1f (success=%s)", focus_value, success)
            return success
        finally:
            cap.release()

    def trigger_autofocus_lock(self, settle_time: float = 1.0) -> bool:
        """
        Temporarily enable autofocus to let the lens settle on the fiducial, then lock focus.

        :param settle_time: Time in seconds to allow autofocus to adjust before locking
        :return: True if successful
        """
        logger.info("Triggering autofocus lock sequence...")
        self.set_camera_autofocus(True)
        time.sleep(settle_time)
        return self.set_camera_autofocus(False)

