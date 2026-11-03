# Configuration

## `config.json`

Read by both `web_gui.py` and `vision_cutter.py`, and edited from the GUI's Settings modal.
Values in the "Default" column apply when a key is missing.

### Machine and camera

| Key | Default | Meaning |
|---|---|---|
| `bed_width_mm`, `bed_height_mm` | 235 | Travel limits used for clamping moves |
| `moonraker_url` | `http://localhost:7125` | Moonraker API |
| `crowsnest_url` | `http://localhost:8080/?action=stream` | Camera stream; `action=snapshot` is tried automatically |
| `camera_offset_x_mm`, `camera_offset_y_mm` | −25, 0 | Camera centre minus knife tip, in bed mm (see [Calibration](calibration.md)) |
| `mm_per_pixel` | 0.05 | Camera scale at the scan height |
| `camera_rotation_deg` | 0 | Camera mounting twist; learned automatically by each auto scan |
| `marker_size_mm` | 7.895 | Size of the ArUco black square (10 mm image with a white border) |
| `z_scan_mm` | 20 | Z height for camera work |
| `camera_settle_seconds` | 2 | Wait after each move before grabbing a frame |
| `search_mode` | `assisted` | Default registration-mark search mode |
| `design_fiducials` | — | Mark centres in the design (fallback when the SVG has none) |
| `nominal_fiducials` | — | Gantry positions of the marks from the last scan (search seed); updated by each scan |

### Cutting

| Key | Default | Meaning |
|---|---|---|
| `z_cut_mm` | 0 | Blade depth (Z) for cutting. Check with the Touch-Off Test |
| `z_safe_mm` | 5 | Travel height; always raised to at least `z_cut_mm + 5` |
| `z_lift_swivel_mm` | 0 | Lift-and-turn height **above `z_cut_mm`** at corners (0 = swivel in the material) |
| `f_cut_mm_min`, `f_travel_mm_min` | 1200, 3000 | Feedrates |
| `blade_offset_mm` | 0.35 | Drag-knife offset between holder axis and blade tip |
| `overcut_mm` | 0.5 | Re-cut past the start of closed shapes |
| `lead_in_margin_mm` | 0.3 | First cut of a job: plunge `blade_offset + margin` behind the start |
| `path_noise_tolerance_mm` | 0 | Noise filter for traced bitmaps (≈0.05); leave 0 for clean vectors |
| `optimize_cut_order` | true | Inside-first, nearest-neighbour, blade-aware order |
| `passes`, `pass_depth_step_mm` | 1, 0 | Cut each contour `passes` times, each pass this much deeper |
| `use_gcode_arcs` | false | Write swivels as `G2`/`G3` (needs `[gcode_arcs]`) instead of short `G1` moves |
| `bed_mesh_profile` | `default` | Klipper mesh loaded at job start (this machine: `cutter`) |

## `vision_cutter.py` command line

Every option defaults to the matching `config.json` value.

```
--svg FILE                 SVG design or PolyCut/JScut G-code (required)
--output FILE              Output G-code (default print_and_cut_output.gcode)
--moonraker-url, --crowsnest-url, --camera-index
--camera-offset-x, --camera-offset-y, --mm-per-pixel
--search-mode {assisted,manual,auto,fast,spiral,exhaustive}
--settle-seconds S         --z-scan Z         --fiducial-type {aruco,circle}
--nominal-fiducials JSON   --no-auto-search   --force-home
--use-cached-alignment     reuse affine_matrix.json instead of scanning
--blade-offset, --overcut, --lead-in, --noise-tolerance, --no-optimize-order
--passes N, --pass-step MM, --gcode-arcs, --mesh-profile NAME
--z-safe, --z-cut, --z-lift-swivel, --f-cut, --f-travel
--dry-run                  everything at safe height (no blade contact)
--verify-touch             touch the 3 test dots and pause before cutting
--pause-before-cut         pause before the first cut (when not verifying)
--no-upload                scan/generate only     --auto-start   start the job after upload
--mock                     offline test with a simulated printer and camera
```

Examples:

```bash
# scan only (writes affine_matrix.json, no G-code upload)
python vision_cutter.py --svg samples/Cut_default.svg --no-upload --search-mode auto

# generate G-code from the saved alignment, dry run, upload and start
python vision_cutter.py --svg samples/Cut_default.svg --use-cached-alignment --dry-run --auto-start

# re-align a PolyCut toolpath to the printed sheet
python vision_cutter.py --svg samples/cut.gcode --auto-start
```

`--mock` never saves its simulated alignment, so it cannot be reused for a real cut.

## Klipper `printer.cfg` (this machine)

Settings tuned for cutting on the Ender 3 (Melzi board, BLTouch 43 mm left of the knife).
See [Bed levelling](bed-levelling.md) for why.

```ini
[bltouch]
samples: 5                      # median of 5 ignores occasional bad touches
samples_result: median
samples_tolerance: 0.150
samples_tolerance_retries: 5

[bed_mesh]
mesh_min: 5, 10                 # measured plate edges
mesh_max: 190, 228              # probe reach limits
probe_count: 7, 7
algorithm: bicubic
fade_start: 1
fade_end: 0                     # no fade: full correction at the cut height (Z≈3)

[bed_screws]                    # names match the coordinates
screw1: 30.5, 37
screw1_name: front left
screw2: 30.5, 207
screw2_name: back left
screw3: 204.5, 207
screw3_name: back right
screw4: 204.5, 37
screw4_name: front right
```

Also needed: Mainsail's `PAUSE`/`RESUME` macros (the Resume button sends `RESUME_BASE`, which
works with a cold extruder) and `[respond]`. `[gcode_arcs]` is **not** needed; swivels are
written as `G1` moves unless `use_gcode_arcs` is on.
