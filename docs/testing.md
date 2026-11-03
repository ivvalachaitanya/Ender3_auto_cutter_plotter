# Testing

## Simulators

The algorithms were measured with simulators before running on the printer. They live outside
the repo; this page records how they work, so they can be rebuilt.

**Drag-knife simulator (cut paths).** Expands the planned pivot path (lines and swivel arcs) into
0.02 mm steps and drags a blade tip behind it at the blade offset:
`tip = pivot − L · unit(pivot − tip)` when the distance exceeds L. The blade keeps its heading
while lifted. The simulated tip path is compared against the true SVG geometry (parsed at
0.05 mm), measuring length cut more than 0.1 mm off the line, deviation, outline covered, and
travel. Results: [Cut-path pipeline](cut-path-pipeline.md#results).

**Printer + camera simulator (mark search).** Renders a paper sheet with ArUco IDs 0–2 (10 mm
images with quiet zone) and some dark artwork on a dark bed, at a random offset and rotation. A
640 × 480 camera view is produced through the same pixel→bed mapping as `pixel_to_bed_space`, with
blur, noise, an unknown camera twist (±0.6°) and a ±2% mm/px error. A fake Moonraker client
counts moves and frames and keeps a simulated clock (2 s settle per look plus travel). Assisted
mode places the sheet under the camera's first stop; other modes use placements within camera
reach. Results: [Registration-mark search](registration-mark-search.md#results).

## Live results (Nov 2026, Ender 3, Klipper 0.13)

| Test | Result |
|---|---|
| Auto scan | 3 marks in 35 s / 8 frames, scale 0.9998, third mark 0.2 mm from prediction |
| Camera Accuracy Check | max 0.10 mm, RMS 0.08 mm (Excellent) |
| Touch-Off Test | needle on all three dots |
| Dry run (Cut_default.svg) | completed, 85 s vs 87 s estimated, Z 8 throughout |
| Bed levelling | tilt 0.79 → 0.02 mm/100 mm; mesh range 2.24 → 0.29 mm |

## Hardware test checklist

1. **Scan:** assisted with the sheet square, then skewed ~3°; spiral/auto after moving the sheet
   10–15 mm. The log should show all three marks locked and the sheet rotation.
2. **Camera Accuracy Check:** Excellent or Good.
3. **Touch-Off Test:** needle on the centre of each dot, just touching.
4. **Dry run:** the head traces every shape at safe height; inner pieces before their outlines.
5. **First real cut** of a test sheet with a rectangle, a circle, the letters O, A, B and a star:
   - rectangle and circle are cut (they used to be dropped);
   - no slash across letters or holes; holes cut before outlines;
   - closed shapes weed cleanly at the start point (overcut);
   - no hooks at the starts of separate short lines (entry swivel);
   - even depth across the whole sheet (bed mesh).
6. **PolyCut G-code:** re-aligned toolpath lines up with the print; arcs keep their shape;
   comments unchanged.
7. **Multi-pass on cardstock:** 2 passes, 0.1 mm per pass; the second pass goes deeper.
