# Workflow (Cutter Studio web GUI)

Start the server and open `http://localhost:5000`:

```bash
python web_gui.py --port 5000      # or double-click start_web_gui.bat
```

Restart the server after updating the code; the page is served from `web_gui.py`.

## Before the first cut of the day

1. **Blade out, sheet off** if you are going to home or probe the bed (see
   [Bed levelling](bed-levelling.md)). Home the printer (G28) from Mainsail or the console.
2. **Blade back in.** Lay the printed sheet flat on the bed; tape the corners if it curls.

## Step 1: Select Cut File

- Pick an `.svg` (or PolyCut/JScut `.gcode`) from the dropdown, or **Workspace → upload** your own.
- **Load File Preview** shows the cut paths on a 1:1 bed mat.

## Step 2: Print / Prep Sheet (SVG only)

- **Add ArUco Fiducials to SVG** adds the three registration marks (IDs 0, 1, 2: top-left,
  top-right, bottom-left). Each mark is a 10 mm image with a 7.9 mm black square.
- **Add 3 Test Dots to SVG** adds the touch-off targets used to check accuracy.
- **Print Sheet (1:1)** prints at true scale. Do not use "fit to page".

## Step 3: Scan & Align

Choose a search mode, then **Run Autonomous Scan & Align**:

| Mode | Use when | Typical time |
|---|---|---|
| Assisted | You slide the sheet under the camera; the rest is automatic | ~25 s |
| Fast | The sheet is where it was last time | ~20 s |
| Spiral | The sheet moved up to ~25 mm | ~25–35 s |
| Auto | Unsure: searches near the last spot, then the whole bed | ~25 s – 3 min |
| Whole-Bed Sweep | The sheet is anywhere on the bed | ~1–3 min |

The log reports each mark as it locks, the sheet rotation, the measured mm/px and camera twist.
If a mark is outside the camera's reach the scan stops with an instruction such as
*"Move the sheet 10 mm to the right"*. See [Registration-mark search](registration-mark-search.md).

**Manual mode** (the Auto/Manual toggle): jog the camera crosshair over marks #0 (top-left),
#1 (top-right), #2 (bottom-left), press **Lock Mark** for each, then
**Solve & Save 2D Affine Matrix**.

The alignment is saved to `affine_matrix.json` for **this file** and is valid for **1 hour**.

## Step 4: Check accuracy, then cut

1. **Camera Accuracy Check (no blade contact)**: the camera visits the three test dots and
   measures how far each is from where the alignment expects it. A window shows each camera
   view (green cross = expected, red circle = found, 1 mm scale bar), the error per dot, and a
   verdict: Excellent ≤ 0.15 mm, Good ≤ 0.3 mm, Fair ≤ 0.6 mm, Poor (re-scan). It checks the
   scan only.
2. **Touch-Off Test (3 Dots & Park)**: the blade lowers to the cut depth on each dot, then
   hovers over the sheet's (0,0). This checks the camera-to-blade offset and the cut depth:
   the needle should land on the centre of each dot and just touch.
3. **Frame Bounds** (optional): traces the job's outline at safe height, mapped through the
   alignment, so you can see where the cut will land.
4. **Dry run** (tick *Dry Run (Air Cut)*): the whole job runs with the blade at safe height.
5. **START VINYL CUT**. With *Verify Dots & Pause* ticked the job first visits the dots and pauses;
   press **RESUME CUT** to continue (it uses `RESUME_BASE`, so a cold extruder doesn't block it).
   **CANCEL** stops the job.

**Save G-code** downloads the aligned G-code instead of sending it. It refuses if there is no
recent alignment for the file, because the result would not line up with the sheet.

## Settings (Hardware Settings button)

Every calibration and cutting option is in the Settings modal and saved to `config.json`.
See [Configuration](configuration.md) for what each one does.
