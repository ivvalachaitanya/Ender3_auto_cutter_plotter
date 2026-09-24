# Ender 3 Autonomous Computer-Vision "Print & Cut" Vinyl Cutting Pipeline

An autonomous computer-vision alignment and drag-knife vinyl cutting system for Creality Ender 3 (and standard Cartesian 3D printers) running Klipper firmware, Moonraker, and Crowsnest. Features a **Cricut-style Web GUI Studio (`web_gui.py`)**, **Sub-Pixel Vision Fiducial Alignment**, **Dual Auto ArUco & Manual Crosshair Jog Modes**, **PolyCut / JScut G-code Affine Transformation**, and **Roland Drag Knife Corner Swivel Planning**.

---

## 🌟 Key Features

- **🎨 Web Studio GUI (`web_gui.py`):** Interactive web application running on local port `5000` with Tailwind CSS, true 1:1 physical bed mat rendering, live 30 FPS MJPEG webcam stream, workspace file manager, and file upload modal.
- **⚡ Dual Alignment Modes:**
  - **⚡ Automatic (ArUco Fiducials):** Sub-pixel detection of OpenCV ArUco markers (`DICT_4X4_50`) with automated gantry positioning.
  - **🎮 Manual (Camera Crosshair Jog):** Interactive D-pad jog controls with live camera optical target reticle to manually lock 3 registration marks and solve the 2D Affine transformation.
- **📄 SVG & G-code Support:** Seamlessly load SVG artwork vector cut paths or pre-generated G-code toolpaths (from PolyCut, JScut, Inkscape).
- **📐 2D Rigid Affine Transformation ($S, \theta, T_x, T_y$):** Computes 2D partial rigid affine matrices (`cv2.estimateAffinePartial2D`) to compensate for sheet rotation, translation, and scale skew on the bed.
- **🔪 Roland Drag Knife Geometry Compensation:** Implements corner swivel arc compensation (`G2`/`G3`) for drag-knife blade caster offset, preventing vinyl tearing or rounded corners.
- **⚙️ Persistent Configuration (`config.json`):** Web GUI calibration modal to configure printable bed size, blade offset, camera offsets ($X, Y$), Z travel heights, and feedrates.
- **🔧 3D Printable Mechanical Hardware:** Custom Logitech C270 webcam mount integrated directly into the Ender 3 toolhead drag knife bracket ($X_{\text{offset}} = -25.0\text{ mm}$).

---

## 🏗️ Repository Architecture

```
Ender3_auto_cutter_plotter/
├── config.json                     # Persistent hardware calibration settings
├── requirements.txt                # Python package dependencies
├── README.md                       # Documentation & setup guide
├── vision_cutter.py                # Main CLI Print & Cut pipeline execution tool
├── web_gui.py                      # Cricut-style Web Application Server (Port 5000)
│
├── src/                            # Core Pipeline Python Modules
│   ├── __init__.py                 # Package initializer
│   ├── drag_knife_planner.py       # SVG parsing, swivel corner planning, Klipper G-code generator
│   ├── gcode_transformer.py        # 2D Affine transformation matrix solver for G-code toolpaths
│   ├── moonraker_client.py         # Klipper / Moonraker API & camera snapshot client
│   └── vision_detector.py          # ArUco fiducial detector & 2D Affine pose estimator
│
├── mechanical/                     # 3D Printable CAD Models & Assemblies
│   ├── Ender3VinylCutter.3mf       # 3MF Bambu / Prusa slicer assembly file
│   ├── ender3vinylcutter.stl       # Assembly STL model
│   ├── SupportoDragKnife.SLDPRT    # SolidWorks original toolhead part
│   ├── SupportoDragKnife.STL       # Standard Ender 3 drag knife bracket
│   ├── SupportoDragKnife_with_C270_CameraMount.stl # Drag knife bracket + C270 camera mount
│   ├── LogiC270_wheel.stl          # Focus wheel modification for C270
│   ├── Logitech C270 Focus inner.stl
│   └── Logitech C270 Focus outer.stl
│
├── scripts/                        # Utility Scripts & Generators
│   ├── fix_svg.py                  # SVG element sanitizer utility
│   ├── generate_aruco_markers.py   # ArUco fiducial marker PNG generator
│   ├── generate_custom_c270_mount.py # Manifold3D parametric CAD mount generator
│   └── test_pc_camera.py           # OpenCV camera enumeration & focus test utility
│
├── samples/                        # Sample SVG Artwork & Toolpath Files
│   ├── Cut.svg                     # Sample sticker sheet with ArUco markers & cut contours
│   ├── Cut1.svg
│   ├── Cut_size.svg
│   ├── cut.gcode                   # Sample PolyCut toolpath file
│   ├── sample_test.svg
│   └── settings.jscut              # JSCut toolpath project settings
│
└── assets/                         # Visual Assets & Calibration Images
    ├── markers/                    # High-resolution ArUco marker PNG files
    ├── images/                     # Mechanical assembly documentation diagrams
    └── snapshots/                  # Camera snapshot test logs & coordinate JSON files
```

---

## 🚀 Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ivvalachaitanya/Ender3_auto_cutter_plotter.git
   cd Ender3_auto_cutter_plotter
   ```

2. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Dependencies:**
   - `python >= 3.8`
   - `opencv-python`
   - `numpy`
   - `requests`
   - `svgelements`
   - `trimesh` & `manifold3d` (optional, for CAD generation)

---

## 🌐 Running the Cutter Studio Web Application

Start the local web GUI server:
```bash
python web_gui.py --port 5000
```
Open **`http://localhost:5000`** in your web browser.

### Web GUI Workflow:
1. **Step 1: Select or Upload Cut File:** Choose an existing `.svg` or `.gcode` file from the workspace dropdown, or click **`📤 Choose File`** to upload your design.
2. **Step 2: Print Sheet (SVG Mode):** Click **`🖨️ Print Sheet (1:1)`** to send your SVG artwork with ArUco fiducials to your 2D paper/vinyl color printer at 1:1 scale. (Skipped automatically if PolyCut G-code is selected).
3. **Step 3: Scan & Align:**
   - **⚡ Auto Mode:** Click **`📷 Auto-Scan ArUco Fiducials`** for automatic vision detection.
   - **🎮 Manual Mode:** Use the interactive D-Pad to jog the camera crosshairs over registration marks #0, #1, #2, click **`📌 Lock Mark`**, and solve the 2D Affine Matrix.
4. **Step 4: Cut Vinyl & Bounding Box Framing:** Click **`🔲 Trace Bounding Box (Frame)`** to trace the 4 outer corners at Safe Z ($Z=5.0\text{ mm}$) to verify material placement, set your **Dry Run** preference, and click **`🚀 START VINYL CUT`** to execute the job on Klipper!

---

## 💻 CLI Usage

You can also run the pipeline headlessly via command line:

### 1. Dry Run (Alignment Check Without Cutting)
```bash
python vision_cutter.py \
  --svg samples/Cut.svg \
  --moonraker-url http://192.168.1.50:7125 \
  --crowsnest-url http://192.168.1.50:8080/?action=snapshot \
  --blade-offset 0.35 \
  --camera-offset-x -25.0 \
  --camera-offset-y 0.0 \
  --mm-per-pixel 0.05 \
  --dry-run
```

### 2. PolyCut G-code Fiducial Compensation
```bash
python vision_cutter.py \
  --svg samples/cut.gcode \
  --moonraker-url http://192.168.1.50:7125 \
  --auto-start
```

### 3. Offline Mock Mode (Testing Without Hardware)
```bash
python vision_cutter.py --svg samples/Cut.svg --mock
```

---

## 📐 Hardware Calibration Guide

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

1. **Camera Scale (`--mm-per-pixel`):**
   Place a ruler on the bed under the camera. Measure pixel distance between 10 mm marks.  
   $$\text{mm\_per\_pixel} = \frac{10\text{ mm}}{\text{distance in pixels}}$$

2. **Camera Offsets (`--camera-offset-x`, `--camera-offset-y`):**
   Make a small pinprick cut with the drag knife at $(100.0, 100.0)$. Jog gantry until the camera target reticle centers over the mark. Read gantry position $(X_{\text{cam}}, Y_{\text{cam}})$:
   $$X_{\text{offset}} = X_{\text{cam}} - 100.0$$
   $$Y_{\text{offset}} = Y_{\text{cam}} - 100.0$$

3. **Drag Knife Blade Offset (`--blade-offset`):**
   Roland 45° blades feature a $0.35\text{ mm}$ offset between pivot axis and cutting tip.  
   - **Rounded Corners:** Increase blade offset ($+0.05\text{ mm}$).
   - **Looping Corners:** Decrease blade offset ($-0.05\text{ mm}$).

---

## ⚙️ Recommended Klipper `printer.cfg` Settings

Ensure `printer.cfg` includes arc support for smooth drag-knife swivel moves:

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

---

## 📜 License

MIT License - Open Source for Ender 3 & Klipper vinyl cutting enthusiasts.
