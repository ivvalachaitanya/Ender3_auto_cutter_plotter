# Documentation

Detailed documentation for the Ender 3 Print & Cut pipeline. Start with the
[project README](../README.md) for installation and a quick start.

| Page | What it covers |
|---|---|
| [Workflow](workflow.md) | Step-by-step use of the Cutter Studio web GUI, from file to finished cut |
| [Configuration](configuration.md) | Every `config.json` key, every CLI flag, and the recommended Klipper `printer.cfg` |
| [Calibration](calibration.md) | Camera offset, mm/px, camera twist, blade offset, cut depth, and how to check them |
| [Bed levelling](bed-levelling.md) | Levelling with the BLTouch, the `cutter` mesh profile, probe noise, plate edges |
| [Registration-mark search](registration-mark-search.md) | How the camera finds the three ArUco marks quickly and reliably |
| [Cut-path pipeline](cut-path-pipeline.md) | How SVG shapes become drag-knife G-code (ordering, swivels, lead-in, overcut) |
| [Alignment & safety rules](alignment-and-safety.md) | When a saved alignment is reused, Z-height rules, G-code transformer, GUI safeguards |
| [Testing](testing.md) | The simulators used to measure the algorithms, live results, and a hardware test checklist |
| [Changelog](changelog.md) | Summary of the changes on the `fix/gcode-alignment-safety` branch |
