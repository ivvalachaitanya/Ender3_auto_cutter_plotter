# Alignment & safety rules

Rules that keep a cut from landing in the wrong place or the blade from going too low.

## When a saved alignment is used

The scan saves `affine_matrix.json`: the matrix plus `file`, `timestamp`, `camera_offset`,
`design_points`, `measured_bed_points`, `gantry_points` and `source` (`auto` or `manual`).
`load_alignment_for()` in `web_gui.py` accepts it only if:

- it was solved **for the selected file**, and
- it is **less than 1 hour old** (`ALIGNMENT_MAX_AGE_S`).

| Action | No valid alignment |
|---|---|
| START VINYL CUT | re-scans the marks first, then cuts |
| Save G-code | refuses (the file would not line up with the sheet) |
| Touch-Off Test, Camera Accuracy Check | refuse with the reason |
| Frame Bounds | traces the design outline but labels it "UNALIGNED" |

Other rules:

- `--mock` runs never write `affine_matrix.json`, so a simulated alignment can't reach a real cut.
- Manual alignment uses the same design marks as the auto scan, adds the camera offset to the
  locked crosshair positions, and records the file and time.
- Changing the camera offset in Settings re-solves the matrix from the gantry positions saved
  with it, and keeps the scan's original timestamp. An offset change doesn't make an old scan
  count as fresh.

## Z heights

- **Travel height** = `max(z_safe_mm, z_cut_mm + 5)`. The blade always clears the cut plane by
  5 mm, even with `z_cut_mm` > 0 (this machine cuts at Z = 3, so it travels at Z = 8).
- **Lift-and-turn** (`z_lift_swivel_mm`) is a height *above* `z_cut_mm`, never an absolute Z.
- Approach (`z_cut + 0.5`) and the touch-off hover (`z_cut + 1`) are relative to the cut depth.
- The touch-off routine lifts to travel height before `PAUSE`, so the blade never drags while paused.
- Dry run moves everything to travel height.

## G-code transformer

`GCodeTransformer` (`src/gcode_transformer.py`) re-aligns PolyCut/JScut toolpaths:

- comments (`;` and `( )`) are never parsed or rewritten;
- numbers like `X.5` and `Y-.5` are understood;
- `G90`/`G91` and modal motion are tracked; lines with only coordinates use the last motion mode;
- relative moves and arc offsets `I`/`J` are rotated (no translation), `R` is scaled;
- `X` and `Y` are always written together, because rotation changes both;
- non-motion commands (`G28`, `G92`, `G53`, `G4`, `G10`, `M…`) pass through untouched.

## Web GUI safeguards

- `/api/cut` only ever forwards `--dry-run` from its `dry` parameter, never arbitrary flags.
- Frame Bounds and the bounding box use the same Y orientation as the cut and are mapped through
  the alignment.
- `[gcode_arcs]` is not required: swivels are `G1` moves by default, and `--gcode-arcs` falls back
  to `G1` if the printer lacks `[gcode_arcs]`.
