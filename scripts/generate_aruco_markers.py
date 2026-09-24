#!/usr/bin/env python3
"""
Generate Printable / Displayable ArUco Markers (DICT_4X4_50).

Generates PNG images for ArUco Marker ID 0, 1, and 2 so you can display them
on your phone screen or print them out on paper to test camera detection!
"""

import sys
import numpy as np

try:
    import cv2
except ImportError:
    print("Error: OpenCV is not installed.")
    sys.exit(1)


def main():
    print("Generating ArUco Markers (DICT_4X4_50)...")

    # Get DICT_4X4_50 dictionary
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    else:
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

    for marker_id in range(3):
        if hasattr(cv2.aruco, "generateImageMarker"):
            img = cv2.aruco.generateImageMarker(dictionary, marker_id, 300)
        else:
            img = cv2.aruco.drawMarker(dictionary, marker_id, 300)

        # Add white border around marker for high contrast
        bordered_img = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)

        filename = f"aruco_marker_id_{marker_id}.png"
        cv2.imwrite(filename, bordered_img)
        print(f"Saved '{filename}'")

    print("\nDone! Open 'aruco_marker_id_0.png' on your phone or print it to test camera detection.")


if __name__ == "__main__":
    main()
