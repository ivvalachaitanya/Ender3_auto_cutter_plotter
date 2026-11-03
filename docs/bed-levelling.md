# Bed levelling

A drag knife cuts vinyl about 0.1 mm thick, so the bed height under the blade must be known to a
few hundredths of a millimetre. Klipper's bed mesh corrects the blade height as it moves; the mesh
must be **current**, **cover the sheet**, and be **applied in full** at the cut height.

## What was wrong (Nov 2026)

Symptom: some places missed, others cut too deep, even though a mesh existed.

| Problem | Effect |
|---|---|
| The saved `default` mesh was stale: the bed had tilted the other way since (springs moved) | Up to **1.4 mm** depth error |
| `fade_end: 10` with the cut at Z = 3 | Klipper applied only (10−3)/9 = **78%** of the correction |
| Mesh area Y 20–190, cuts up to Y ≈ 217 | Beyond Y 190 Klipper reused edge values |
| 5×5 points, 2 samples, 0.15 mm tolerance, noisy probe | Coarse and noisy readings |

## The procedure (blade out, sheet off, bed cold)

1. **Remove the blade** and take the sheet off. Paper that lifts even slightly makes the probe
   read inconsistently. A carrier mat, if you cut on one, stays.
2. **Level with the knobs:**

   ```bash
   python scripts/measure_bed_corners.py
   ```

   It homes if needed, clears the mesh, probes as close to each bed screw as the probe can reach
   (the right-hand screws are beyond its reach, so it fits a plane), and prints how far to lower
   each corner (M4 knob: 0.7 mm per turn; tightening lowers that corner). Repeat until all corners
   read **OK** (within 0.1 mm).
   Tips: the plate sits on four springs, so one knob moves the others. If three corners are high,
   **raise the low one** instead of lowering three. The tool reports against a fitted plane; a small
   diagonal twist (≈0.1 mm) remains and the mesh corrects it.
3. **Re-home** (G28): levelling changes the bed height, so Z = 0 must be re-established.
4. **Mesh** and save it as the cutting profile:

   ```
   BED_MESH_CALIBRATE PROFILE=cutter
   SAVE_CONFIG
   ```

5. `config.json` has `"bed_mesh_profile": "cutter"`, so every cut starts with
   `BED_MESH_PROFILE LOAD=cutter`.
6. **Blade back in**, sheet on, scan, then a Touch-Off Test to confirm the cut depth.

Result on this machine: tilt 0.79 → 0.02 mm/100 mm in three rounds; new `cutter` mesh range
**0.29 mm** (was 2.24 mm before levelling).

## `printer.cfg` changes

A backup of the original is on the printer as `printer-20261104-before-cutter-tune.cfg`.

| Setting | Was | Now |
|---|---|---|
| `[bed_mesh] fade_end` | 10 | 0 |
| `[bed_mesh] probe_count` | 5, 5 | 7, 7 |
| `[bed_mesh] mesh_min` / `mesh_max` | 20,20 / 190,190 | 5,10 / 190,228 |
| `[bltouch] samples` / `samples_tolerance` / retries | 2 / 0.15 / 3 | 5 (median) / 0.15 / 5 |
| `[bed_screws]` names | "front right"/"back left" swapped | match their coordinates |

## Reach and plate edges

The BLTouch is 43 mm left of the knife, so the probe can reach bed X ≤ 192 and Y ≤ 230. Measured
plate edges (capped-depth probes, so a miss is harmless):

| Edge | Result |
|---|---|
| Left | plate continues to X ≤ 2 |
| Front | plate ends near Y ≈ 3; consistent surface from Y = 10 (a clip/raised edge at Y ≈ 7) |
| Back | plate continues to Y ≥ 229 (probe limit) |
| Right | beyond X 192 (probe limit); Klipper extends the edge values, fine on a level bed |

The knife can still use the full 235 × 235 mm.

## Probe noise

This BLTouch spreads 0.06–0.075 mm over repeated probes at one spot (a good one is ≤ 0.02 mm).
Checks done: no false triggers with the pin down in mid-air (60/60), no BLTouch/MCU errors in
`klippy.log`, healthy serial link. The LED flashing once per touch is normal (the pin retracts
on trigger). Taking the median of 5 samples makes the mesh reliable despite the noise. To reduce
the noise itself:

- check the BLTouch mount on the printed bracket is tight and does not flex;
- check the pin slides freely and flashes **once** per touch (several flickers = a sticky pin);
- reseat the pin-27 adapter and plugs (Melzi board), and keep the probe cable away from the
  stepper and hotend cables.

## Options for the future

- Re-probe `cutter` over the full new area (X 5–190, Y 10–228); about 10 minutes. The current
  profile was taken over X 20–190, Y 20–207.
- Probe a fresh mesh over the job area before each cut (≈2 min per job): the mesh can then never
  go stale.
