# Ender 3 Print & Cut: Computer-Vision Vinyl Cutter

Turns a Creality Ender 3 running **Klipper + Moonraker + Crowsnest** into a print-and-cut vinyl
cutter. Print your design with three ArUco registration marks, lay the sheet on the bed, and a
toolhead camera finds the marks, works out exactly where and how skewed the sheet is, and a drag
knife cuts along the printed outlines.

Everything runs from a browser-based **Cutter Studio** (`web_gui.py`) or the command line
(`vision_cutter.py`).

## Features

- **Fast, reliable mark finding:** search steps sized to the camera's view, partially visible
  marks followed, sheet rotation read from the first mark, reach-aware placement help.
  Auto scan locks all three marks in about 25–35 s. → [details](docs/registration-mark-search.md)
- **Accurate alignment:** rotation + scale + offset solved from the marks; on the test machine the
  aligned positions are within 0.1 mm (Camera Accuracy Check).
- **Drag-knife cut paths:** exact corners, inside-first cut order, swivel compensation at corners
  and at every plunge, overcut, multi-pass. Techniques follow
  [PolyCut](https://github.com/IridiumIO/PolyCut). → [details](docs/cut-path-pipeline.md)
- **Built-in checks:** Camera Accuracy Check (no blade contact), Touch-Off Test, Frame Bounds,
  dry run. → [workflow](docs/workflow.md)
- **Bed-levelling tools:** corner measurement script and a dedicated cutting mesh profile.
  → [details](docs/bed-levelling.md)
- **SVG or existing G-code:** generate from SVG, or re-align PolyCut/JScut/Inkscape toolpaths to the
  printed sheet.
- **Safe by design:** alignments only reused for the same file within an hour, travel height always
  above the cut plane, swivels as plain `G1` moves (no `[gcode_arcs]` needed).
  → [details](docs/alignment-and-safety.md)

## Documentation

| | |
|---|---|
| [Workflow](docs/workflow.md) | Using Cutter Studio from file to finished cut |
| [Configuration](docs/configuration.md) | `config.json` keys, CLI flags, Klipper `printer.cfg` |
| [Calibration](docs/calibration.md) | Camera offset, mm/px, camera twist, blade offset, cut depth |
| [Bed levelling](docs/bed-levelling.md) | Levelling with the BLTouch and the `cutter` mesh |
| [Registration-mark search](docs/registration-mark-search.md) | How the marks are found |
| [Cut-path pipeline](docs/cut-path-pipeline.md) | How SVG becomes drag-knife G-code |
| [Alignment & safety rules](docs/alignment-and-safety.md) | Alignment reuse, Z rules, transformer |
| [Testing](docs/testing.md) | Simulators, live results, hardware test checklist |
| [Changelog](docs/changelog.md) | What changed on the current branch |

## Hardware

- Creality Ender 3 (tested: Melzi ATmega1284P board) with Klipper, Moonraker, Mainsail, Crowsnest
- BLTouch (tested 43 mm left of the knife) for homing and the bed mesh
- Spring-loaded drag-knife holder with a Roland-style 45° blade, in place of the hotend
- Logitech C270 webcam on the toolhead (tested 29.6 mm right of the blade), refocused for close range
- Printable parts in `mechanical/`: knife bracket (`SupportoDragKnife.STL` / `.SLDPRT`), C270 focus
  wheel, and the full toolhead assembly (`Ender3VinylCutter.3mf`, `ender3vinylcutter.stl`)

## Installation

```bash
git clone https://github.com/ivvalachaitanya/Ender3_auto_cutter_plotter.git
cd Ender3_auto_cutter_plotter
pip install -r requirements.txt     # Python 3.8+: opencv-python, numpy<2, requests, svgelements, svgpathtools, shapely
```

Then set your printer's address and calibration in `config.json` (or in the GUI's Settings) and
work through [Calibration](docs/calibration.md) once.

## Quick start

**Web GUI**

```bash
python web_gui.py --port 5000       # or start_web_gui.bat
```

Open `http://localhost:5000`, then:

1. **Select Cut File**: choose or upload an SVG.
2. **Print / Prep Sheet**: add the ArUco marks and test dots, then print at 1:1 scale.
3. **Scan & Align**: lay the sheet on the bed and run the scan (Assisted mode: slide the sheet
   under the camera).
4. **Cut Vinyl**: Camera Accuracy Check → Touch-Off Test → dry run → **START VINYL CUT**.

Full walkthrough: [docs/workflow.md](docs/workflow.md).

**Command line**

```bash
# scan the marks only
python vision_cutter.py --svg samples/Cut_default.svg --no-upload --search-mode auto

# dry run from the saved alignment, upload and start
python vision_cutter.py --svg samples/Cut_default.svg --use-cached-alignment --dry-run --auto-start

# offline test, no printer needed
python vision_cutter.py --svg samples/Cut_default.svg --mock --no-upload
```

All flags: [docs/configuration.md](docs/configuration.md#vision_cutterpy-command-line).

**Bed levelling** (blade out, sheet off)

```bash
python scripts/measure_bed_corners.py
```

## Repository layout

```
Ender3_auto_cutter_plotter/
├── web_gui.py                  Cutter Studio web server (port 5000)
├── vision_cutter.py            Command-line pipeline: scan → align → plan → G-code → upload
├── config.json                 Machine calibration and cutting settings
├── requirements.txt
├── start_web_gui.bat, start_cutter_studio.bat
├── src/
│   ├── drag_knife_planner.py   SVG parsing, cut ordering, drag-knife planning, G-code generator
│   ├── vision_detector.py      ArUco/target detection, mark search, affine solver
│   ├── gcode_transformer.py    Re-aligns existing G-code toolpaths
│   ├── svg_injector.py         Adds ArUco marks and touch-off test dots to an SVG
│   └── moonraker_client.py     Moonraker API and camera client
├── scripts/
│   ├── measure_bed_corners.py  Bed levelling helper (BLTouch)
│   ├── generate_aruco_markers.py, fix_svg.py, test_pc_camera.py
│   └── generate_custom_c270_mount.py
├── docs/                       Detailed documentation
├── samples/                    Example designs, PolyCut G-code, Ender 3 PolyCut profile
├── mechanical/                 Printable knife bracket, C270 focus parts, toolhead assembly
└── assets/                     Marker images, part renders, camera snapshots
```

## PolyCut / JScut

To cut existing toolpaths instead of generating them, pass the `.gcode` file as `--svg` (or pick
it in the GUI). The scan aligns it to the printed sheet the same way. A PolyCut profile for the
Ender 3 is in [`samples/ender3_gen1_polycut_profile.json`](samples/ender3_gen1_polycut_profile.json).

## License

MIT. Several cut-path techniques follow [PolyCut](https://github.com/IridiumIO/PolyCut) (GPL-3);
they are independently reimplemented here and no PolyCut code is included.
