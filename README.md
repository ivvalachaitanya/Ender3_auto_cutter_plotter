# Ender 3 Autonomous Computer-Vision "Print & Cut" Vinyl Cutting Pipeline

An autonomous computer-vision alignment and drag-knife vinyl cutting pipeline for Creality Ender 3 (or standard Cartesian 3D printers) running Klipper firmware, Moonraker, and Crowsnest on a Raspberry Pi.

---

## 🌟 System Overview & Features

- **Moonraker & Klipper HTTP Integration:** Communicates over Moonraker REST API (`port 7125`) for gantry homing, coordinate queries, G-code file upload, and automatic print job triggering.
- **Crowsnest Vision Integration:** Captures live camera snapshots directly from Crowsnest HTTP stream (`port 8080`) or local OpenCV webcam index fallback.
- **Sub-Pixel Vision Fiducial Detection:** Detects ArUco markers (`DICT_4X4_50`) or sub-pixel circular fiducial centroids using contour moment analysis and `cv2.cornerSubPix`.
- **2D Rigid Affine Transformation:** Computes 2D partial rigid affine matrices (`cv2.estimateAffinePartial2D`) between SVG nominal space and physical printer bed space to correct for scale skew, rotation, and translation.
- **Roland Drag Knife Geometry Compensation:** Implements drag-knife caster compensation by generating corner swivel arcs (`G2`/`G3`) whenever direction changes sharply (> 30°), preventing vinyl tearing or rounded corners.
- **Comprehensive CLI Interface:** Configurable flags (`--svg`, `--blade-offset`, `--dry-run`, `--moonraker-url`, `--auto-start`, `--mock`).

---

## 🏗️ Repository Architecture

```
Ender3_auto_cutter_plotter/
├── vision_cutter.py          # Main CLI Entrypoint & Pipeline Controller
├── sample_test.svg           # Sample test SVG with fiducials & cut paths
├── requirements.txt          # Python dependencies
├── README.md                 # Complete documentation & calibration guide
└── src/                      # Core Package Modules
    ├── __init__.py           # Package initializer
    ├── moonraker_client.py   # Moonraker REST API & Crowsnest Camera Client
    ├── vision_detector.py    # Vision fiducial detection & 2D Affine Transform
    └── drag_knife_planner.py # SVG parser, Drag Knife Swivel Arc Planner, & G-code Generator
```

---

## 🚀 Installation & Dependencies

On your Raspberry Pi (or host PC connected to printer network):

```bash
git clone https://github.com/ivvalachaitanya/Ender3_auto_cutter_plotter.git
cd Ender3_auto_cutter_plotter
pip install -r requirements.txt
```

### Dependencies:
- `python >= 3.8`
- `opencv-python-headless`
- `numpy`
- `requests`
- `svgpathtools`
- `shapely`

---

## 📐 Step-by-Step Camera-to-Blade Offset Calibration Guide

To achieve precise alignment between what the camera sees and where the drag knife cuts, perform the following calibration steps once during setup:

```
                  +--------------------------+
                  |    Gantry Toolhead       |
                  |                          |
                  |   [Camera]    [Knife]    |
                  +------|-----------|-------+
                         |<--Offset->|
                         |   X, Y    |
                         v           v
                    (Camera Axis) (Blade Tip)
```

### 1. Calibrate Camera Scale (`--mm-per-pixel`)
1. Place a fine mm ruler on the bed directly under the camera at focus height.
2. Capture a frame snapshot using `python vision_cutter.py --mock ...` or via Crowsnest browser UI (`http://<pi_ip>:8080/?action=snapshot`).
3. Measure the pixel distance between two ruler markings (e.g. 10 mm apart).
4. Compute:
   $$\text{MM\_PER\_PIXEL} = \frac{\text{Distance in mm}}{\text{Distance in pixels}}$$
   *Example:* If $10\text{ mm} = 200\text{ pixels}$, then `--mm-per-pixel 0.05`.

### 2. Calibrate Camera Offset (`--camera-offset-x`, `--camera-offset-y`)
1. Mount vinyl paper on the printer bed and home the printer (`G28`).
2. Lower Z and perform a small test pinprick or dot cut with the drag knife at a known gantry coordinate, e.g., $(X_0 = 100.0, Y_0 = 100.0)$.
3. Raise Z to safe travel height ($Z = 5.0$).
4. Use Mainsail/Fluidd gantry controls to jog the carriage until the test mark is centered in the camera image frame.
5. Read the new printer gantry position $(X_{\text{cam}}, Y_{\text{cam}})$ from Moonraker/Mainsail.
6. Compute offset values:
   $$\text{CAMERA\_OFFSET\_X} = X_{\text{cam}} - X_0$$
   $$\text{CAMERA\_OFFSET\_Y} = Y_{\text{cam}} - Y_0$$
   *Example:* If knife prick was at $(100, 100)$ and camera center aligns with prick at gantry $(75.0, 100.0)$, then `--camera-offset-x -25.0`.

### 3. Calibrate Drag Knife Blade Offset (`--blade-offset`)
Roland drag knives have a small offset (typically $0.25\text{ mm}$ to $0.45\text{ mm}$) between the pivot axis and the trailing cutting edge.
1. Run a test cut of a $20\text{ mm} \times 20\text{ mm}$ square (`sample_test.svg`) using `--blade-offset 0.35`.
2. Inspect the corners:
   - **Rounded Corners:** The blade offset is set too small. Increase `--blade-offset` in steps of $0.05\text{ mm}$.
   - **Ears / Loops at Corners:** The blade offset is set too large. Decrease `--blade-offset` in steps of $0.05\text{ mm}$.
   - **Sharp 90° Corners:** Perfect calibration!

---

## 💻 CLI Usage Examples

### 1. Dry Run (Alignment Check without Cutting)
Runs camera alignment, detects fiducials, computes affine transform, and saves G-code with Z raised to safe height ($Z = 5.0\text{ mm}$):
```bash
python vision_cutter.py \
  --svg sample_test.svg \
  --moonraker-url http://192.168.1.50:7125 \
  --crowsnest-url http://192.168.1.50:8080/?action=snapshot \
  --blade-offset 0.35 \
  --camera-offset-x -25.0 \
  --camera-offset-y 0.0 \
  --mm-per-pixel 0.05 \
  --dry-run
```

### 2. Full Auto Production Run
Executes alignment, generates swivel-compensated G-code, uploads to Moonraker, and triggers the print job:
```bash
python vision_cutter.py \
  --svg my_sticker_sheet.svg \
  --moonraker-url http://192.168.1.50:7125 \
  --blade-offset 0.35 \
  --nominal-fiducials "[[10.0, 10.0], [100.0, 10.0], [10.0, 100.0]]" \
  --z-cut 0.0 \
  --z-safe 5.0 \
  --f-cut 1200 \
  --auto-start
```

### 3. Offline Mock Mode (For Testing without Printer/Camera)
Simulates Klipper homing, camera frames, and fiducial detection:
```bash
python vision_cutter.py --svg sample_test.svg --mock
```

---

## ⚙️ Klipper Configuration Recommendations

In your `printer.cfg`, ensure the toolhead bounds allow probing/jogging with camera offsets:

```ini
[stepper_x]
position_min: -30
position_max: 235

[stepper_y]
position_min: -10
position_max: 235

[gcode_arcs]
resolution: 0.1
```
*Note: Enabling `[gcode_arcs]` in Klipper ensures smooth execution of `G2` and `G3` drag-knife corner swivel moves.*
