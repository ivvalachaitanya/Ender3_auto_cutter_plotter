# Changelog: `fix/gcode-alignment-safety`

Changes since `main`, newest first. Details are in the linked pages.

### Bed levelling and mesh
- Cut G-code loads a configurable mesh profile (`bed_mesh_profile`, `--mesh-profile`, Settings);
  this machine uses `cutter`, a cold, bare-bed 7×7 mesh taken after levelling.
- `scripts/measure_bed_corners.py`: probe the bed screws and print how far to turn each knob.
- `printer.cfg`: no mesh fade, 7×7 probe grid over the measured plate area, median-of-5 probing,
  corrected bed-screw names. → [Bed levelling](bed-levelling.md)

### Calibration and checking
- **Camera Accuracy Check** button: the camera measures the three test dots against the
  alignment and shows annotated views with a verdict. → [Workflow](workflow.md)
- Calibration from the first live scan: `mm_per_pixel` 0.0301, learned `camera_rotation_deg`.
  → [Calibration](calibration.md)

### G-code output
- Swivels written as short `G1` moves by default (the printer has no `[gcode_arcs]`; `G2` would
  have aborted every cut at the first corner).

### Registration-mark search (rebuilt)
- Steps sized to the camera's field of view, partial-mark following, rotation from the first
  mark's corners, any mark as anchor, reach-aware assisted parking with "move the sheet" errors,
  learned camera twist and measured mm/px. Simulated success 4–10/30 → 30/30.
  → [Registration-mark search](registration-mark-search.md)

### Cut-path pipeline (PolyCut-style)
- Per-segment flattening with exact corners; inside-first, nearest-neighbour, blade-aware
  ordering; smooth start points; entry swivel and lead-in; overcut through corners; optional noise
  filter; multi-pass; estimated time. → [Cut-path pipeline](cut-path-pipeline.md)

### Safety fixes
- Mock runs never save an alignment; Save G-code refuses without a real one.
- One alignment rule everywhere: same file and under 1 hour.
- Travel and lift-and-turn heights relative to the cut depth.
- SVG compound paths no longer get a cut line between subpaths; `rect`/`circle`/… no longer dropped.
- G-code transformer rewritten (comments, `.5` numbers, modal moves, relative moves, I/J/R, G28/G92).
- Manual alignment fixed (wrong mark positions/order, no camera offset, no file/time record).
- Frame Bounds mapped through the alignment; `dry` parameter whitelisted.
  → [Alignment & safety rules](alignment-and-safety.md)
