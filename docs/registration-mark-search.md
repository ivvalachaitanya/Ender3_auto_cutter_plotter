# Registration-mark search

Code: `FiducialSearcher` and `FiducialDetector` in `src/vision_detector.py`.

The printed sheet has three ArUco marks (`DICT_4X4_50`, IDs 0 = top-left, 1 = top-right,
2 = bottom-left), 170 mm apart across and 180 mm top-to-bottom in `samples/Cut_default.svg`.
The scan finds, centres and measures all three, then solves a rotation + uniform scale +
translation (`cv2.estimateAffinePartial2D`) from design to bed coordinates.

## The constraint: a tiny field of view

At Z = 20 the camera sees **18.8 × 14.1 mm** (640 × 480 px at 0.0301 mm/px). A mark only decodes
when its whole 7.9 mm black square is in frame, so its centre must be within about
**±5 mm in X and ±2.7 mm in Y** of the image centre.

The previous search ignored this: its spiral stepped 14 mm in both directions, so about 2/3 of the
area was never seen. It predicted marks #2/#3 assuming a square sheet, so 1.5° of skew put them
out of view. "Exhaustive" probed 14 fixed points.

## How the search works now

1. **Gap-free steps.** Search steps are computed from frame size, mm/px and marker size, visited
   nearest-first.
2. **Follow partially visible marks.** A dark, marker-sized blob cut off by the frame edge (or
   visible but not decodable) gives an estimate of the mark's centre; the camera jumps there.
   Artwork and the bed edge are rejected by size, fill and edge-contact checks.
3. **Any mark can be the anchor.** Whichever ID is found first is used to predict the other two.
4. **Sheet rotation from the first mark.** The mark's sub-pixel corners give its angle (≈0.05°),
   so the 170–180 mm jumps to the other marks include the rotation. After two marks, scale and
   rotation come from the marks themselves for the third.
5. **Reach awareness.** The camera cannot see bed X < 29.6 mm (camera offset). Assisted mode
   parks at a spot from which all three marks stay reachable for up to ±4° of skew. If a predicted
   mark is out of reach the scan stops with a direction to move the sheet.
6. **Self-calibration.** Each scan measures mm/px from the known marker size and the camera's
   mounting twist; the twist is stored (smoothed) in `config.json` for the next scan, and a mm/px
   mismatch over 3% is reported.
7. **Centring.** Each mark is servoed to the image centre (usually one correction step) before
   its bed position is recorded, which removes lens distortion from the measurement.

## Modes

| Mode | Behaviour |
|---|---|
| `assisted` / `manual` | Park at the safe spot and wait for any mark; moves onto a partly visible one |
| `fast` | One look at the last position (plus cue following) |
| `spiral` | Gap-free search within 25 mm of the last position |
| `exhaustive` | Nearest-first sweep of the whole bed with cue-sized steps |
| `auto` | `spiral`, then `exhaustive` |

The search starts from `nominal_fiducials[0]` (the last scan's position of mark #0), so after one
successful scan the next one usually locks immediately.

## Results

Simulated printer and camera (see [Testing](testing.md)), 30 trials, sheet skewed up to ±3°:

| Mode | Old | New | Median time old → new |
|---|---|---|---|
| Assisted | 10/30 | 30/30 | 34 → 23 s |
| Fast | 30/30 | 30/30 | 28 → 22 s |
| Spiral (sheet moved ≤ 15 mm) | 4/30 | 30/30 | 34 → 24 s |
| Auto (moved ≤ 15 mm) | 6/30 | 30/30 | 41 → 24 s |
| Auto (moved ≤ 40 mm) | 5/30 | 30/30 | 86 → 91 s |

On the printer: all three marks in **35 s, 8 frames** in auto mode; scale 0.9998; the third mark
landed 0.2 mm from its prediction; camera accuracy check max 0.10 mm.
