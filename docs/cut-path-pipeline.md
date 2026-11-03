# Cut-path pipeline (SVG → drag-knife G-code)

Code: `SVGParser`, `optimize_cut_order`, `DragKnifePlanner` and `GCodeGenerator` in
`src/drag_knife_planner.py`, driven by `vision_cutter.py`.

Several techniques follow [PolyCut](https://github.com/IridiumIO/PolyCut) (GPL-3). This is an
independent Python implementation; no PolyCut code is included.

## Steps

1. **Parse and flatten.** Every SVG shape type is read (`path`, `rect`, `circle`, `ellipse`,
   `line`, `polyline`, `polygon`), group transforms and units resolved. Each segment is flattened
   on its own: straight edges keep their exact end points (sharp corners), curves are subdivided
   adaptively until every chord is within **0.01 mm** of the curve (max 0.5 mm long). Each
   subpath is its own contour, so the blade lifts between an outline and its hole. Marks, test
   dots and the paper-origin marker are skipped by element ID.
2. **Transform** into bed coordinates with the alignment matrix from the scan.
3. **Noise filter** (`path_noise_tolerance_mm`, default 0 = off). Removes collinear points and
   tiny zig-zags that would trigger needless swivels, never moving the path more than the
   tolerance from any original point. Meant for traced bitmaps; on clean vector designs it costs
   accuracy, so it is off.
4. **Order** (`optimize_cut_order`):
   - **Inside-first:** every contour enclosed by a closed contour is cut before it, so pieces don't
     shift once freed.
   - Nearest ready contour next; closed contours may run either direction; open paths may be
     reversed.
   - **Start points:** closed contours start on a smooth vertex or mid-edge, never on a corner.
   - **Blade-aware:** among candidates, prefer starting in the direction the blade already points
     (a penalty of up to 60 mm of travel for starting against it).
5. **Plan each contour** (`DragKnifePlanner`). The holder axis (pivot) runs `blade_offset` ahead
   of the blade tip along the cut direction.
   - **Corners** turning more than 30°: the pivot swivels on a circle of radius `blade_offset`
     about the corner, so the tip stays on the corner.
   - **Entry swivel:** the planner tracks the blade's heading across contours (a lifted drag blade
     keeps it). The tip lands exactly on the start point and swivels in place to the new direction.
   - **Lead-in** (first cut of a job, heading unknown): plunge `blade_offset + lead_in_margin_mm`
     behind the start and drag into line.
   - **Overcut** (`overcut_mm`): closed contours re-cut their own start, following the shape
     through corners.
6. **Generate G-code** (`GCodeGenerator`):
   - header: `G21`, `G90`, `BED_MESH_PROFILE LOAD=<bed_mesh_profile>`, safe Z;
   - optional touch-off routine (3 dots, hover over paper origin, `PAUSE` at safe height);
   - per contour and pass: travel at safe Z, approach + slow plunge, cut, lift;
   - swivels as short `G1` chords within 0.005 mm of the arc (5 moves for a 90° swivel), or
     `G2`/`G3` with `use_gcode_arcs`;
   - `passes` / `pass_depth_step_mm`: each contour repeated, each pass deeper;
   - estimated time and cut length in the header;
   - footer: lift and present the sheet (X0 Y200).

## Results

A drag-knife simulator (blade tip trailing the pivot at the blade offset; blade keeps its
heading while lifted) compared the old and new pipelines on the sample SVGs:

| File | Cut off the line (> 0.1 mm) | Outline covered | Travel |
|---|---|---|---|
| Basic shapes | 1.85 → **0.00 mm** | 99.1 → **100%** | −34% |
| Cut_default | 0.47 → **0.39 mm** | 99.1 → **100%** | −44% |
| sample_test | 1.63 → **0.36 mm** | 99.0 → **99.6%** | −42% |
| bed_mesh_test | 18.3 → **0.32 mm** | 97.3 → **99.9%** | −9% |

A 20 × 15 mm rectangle used to become 143 points with corners shaved by up to 0.18 mm; it is now
4 exact corners. The remaining off-line amount is the very first plunge of each job.

## PolyCut / JScut G-code input

Existing toolpaths can be re-aligned instead of generated (`--svg file.gcode`).
`GCodeTransformer` (`src/gcode_transformer.py`) applies the alignment to every motion move. See
[Alignment & safety rules](alignment-and-safety.md#g-code-transformer) for what it handles. The
included `samples/ender3_gen1_polycut_profile.json` is a PolyCut profile for the Ender 3.
