# Calibration

Values below are this machine's current calibration (stored in `config.json`).

```
            +----------------------------------+
            |          Gantry toolhead         |
            |  [BLTouch]   [Knife]   [Camera]  |
            +-----|-----------|---------|------+
                  |<- 43 mm ->|<-29.6 ->|
                (probe)   (blade tip) (camera axis)
```

## Camera offset: `camera_offset_x_mm` = 29.6, `camera_offset_y_mm` = −1.2

The camera centre relative to the blade tip. Measure it once:

1. Make a small pin-prick with the blade at a known spot, e.g. (100, 100).
2. Jog until the camera crosshair is centred on the prick and read the gantry position
   (X<sub>cam</sub>, Y<sub>cam</sub>).
3. `camera_offset_x = 100 − X_cam`, `camera_offset_y = 100 − Y_cam`. The camera then sees bed
   point `gantry + offset`.

**Check it** with the Touch-Off Test after a scan: the needle must land on the centre of all three
dots. A consistent miss in one direction means this offset is off by that amount.

Consequence: the camera cannot see bed X below ≈ 29.6 mm (it cannot move left of gantry X = 0),
so registration marks must sit at X ≳ 30 mm. The scan reports a mark that is out of reach.

## Camera scale: `mm_per_pixel` = 0.0301

Measured from the ArUco marks themselves: a 7.895 mm black square measures 262 px at the image
centre at Z = 20. Every scan logs the measured value and warns if config differs by more than 3%.
(The old value 0.0293 was 3% low; it only affected how many centring steps were needed.)

Manual method: place a ruler under the camera, measure pixels between 10 mm marks,
`mm_per_pixel = 10 / pixels`.

## Camera twist: `camera_rotation_deg` = −0.18

The camera's small rotation relative to the gantry axes. Each auto scan measures it (the angle of
the first mark's edge versus the true sheet rotation from two marks) and stores a smoothed value,
so the next scan predicts the other marks better. No manual step needed.

## Blade offset: `blade_offset_mm` = 0.35

Roland-style 45° blades: about 0.35 mm between the holder axis and the blade tip.

- Rounded corners → increase by 0.05 mm.
- Little loops or overshoot at corners → decrease by 0.05 mm.

## Cut depth: `z_cut_mm` = 3.0

The Z at which the blade cuts through the vinyl but not the backing. Z = 0 is set by the BLTouch
(`z_offset`) and is still the placeholder 2.0 in `printer.cfg`. That doesn't matter as long as
`z_cut_mm` is chosen by test. Re-check it with the Touch-Off Test after any re-homing on a changed
bed (the needle should just touch the dots), then with a test cut.

The holder is spring-loaded, so small depth errors become small pressure changes, not missed cuts.
Uneven depth across the sheet is a bed problem: see [Bed levelling](bed-levelling.md).

## Checking calibration

| Check | Button / tool | Checks | Pass |
|---|---|---|---|
| Camera Accuracy Check | Step 4 | Scan / alignment (camera only) | ≤ 0.15 mm Excellent, ≤ 0.3 mm Good |
| Touch-Off Test | Step 4 | Camera-to-blade offset and cut depth | needle on each dot centre |
| Bed corners | `python scripts/measure_bed_corners.py` | Bed tilt | all corners "OK" (≤ 0.1 mm) |

First live result (Nov 2026): accuracy check max 0.10 mm, RMS 0.08 mm; touch-off on all three dots.
