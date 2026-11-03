#!/usr/bin/env python3
"""
Measure bed tilt at the four bed screws with the BLTouch and tell you how much to lower each corner.

Run it, adjust the knobs, run it again until every corner reads OK.

    python scripts/measure_bed_corners.py            # uses moonraker_url from config.json
    python scripts/measure_bed_corners.py --url http://192.168.1.20:7125

How it works: the active bed mesh is cleared (raw heights), the probe is placed as close to each
[bed_screws] position as it can reach (the right-hand screws are beyond the probe's reach on an
Ender 3 with the probe 43 mm left of the tool), a plane is fitted through the four readings, and the
plane height at each screw is reported relative to the lowest corner.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import requests

M4_PITCH_MM = 0.7      # Ender 3 bed screws: M4, 0.7 mm per turn
OK_MM = 0.10           # a corner within this of the lowest one counts as level


def gcode(url, script, timeout=300):
    r = requests.post(f"{url}/printer/gcode/script", params={"script": script}, timeout=timeout)
    r.raise_for_status()


def query(url, objects):
    r = requests.get(f"{url}/printer/objects/query?" + "&".join(objects), timeout=10)
    r.raise_for_status()
    return r.json()["result"]["status"]


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = {}
    try:
        with open(os.path.join(root, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=cfg.get("moonraker_url", "http://localhost:7125"))
    ap.add_argument("--travel-z", type=float, default=10.0)
    args = ap.parse_args()
    url = args.url.rstrip("/")

    st = query(url, ["configfile=settings", "toolhead=homed_axes,axis_minimum,axis_maximum"])
    settings = st["configfile"]["settings"]
    probe_cfg = settings.get("bltouch") or settings.get("probe")
    screws_cfg = settings.get("bed_screws")
    if not probe_cfg or not screws_cfg:
        sys.exit("Needs a [bltouch]/[probe] section and a [bed_screws] section in printer.cfg.")
    off = np.array([float(probe_cfg.get("x_offset", 0.0)), float(probe_cfg.get("y_offset", 0.0))])
    lo = np.array(st["toolhead"]["axis_minimum"][:2], dtype=float)
    hi = np.array(st["toolhead"]["axis_maximum"][:2], dtype=float)
    screws = []
    for i in range(1, 10):
        if f"screw{i}" in screws_cfg:
            screws.append((screws_cfg.get(f"screw{i}_name", f"screw {i}"), np.array(screws_cfg[f"screw{i}"], dtype=float)))

    if st["toolhead"]["homed_axes"] != "xyz":
        print("Homing (G28)...")
        gcode(url, "G28")
    gcode(url, "BED_MESH_CLEAR")

    readings = []  # (probe_xy, z)
    for name, pos in screws:
        nozzle = np.clip(pos - off, lo, hi)          # get the probe as close to the screw as reachable
        probe_xy = nozzle + off
        gcode(url, f"G90\nG0 Z{args.travel_z:.1f} F600\nG0 X{nozzle[0]:.2f} Y{nozzle[1]:.2f} F4800")
        gcode(url, "PROBE")
        z = float(query(url, ["probe=last_z_result"])["probe"]["last_z_result"])
        readings.append((probe_xy, z))
        note = "" if np.allclose(probe_xy, pos, atol=0.5) else f" (nearest reachable point to the screw at {pos[0]:.0f},{pos[1]:.0f})"
        print(f"  probed {name:12s} at ({probe_xy[0]:6.1f}, {probe_xy[1]:6.1f}): z = {z:+.3f}{note}")
        gcode(url, f"G0 Z{args.travel_z:.1f} F600")

    xy = np.array([r[0] for r in readings])
    zs = np.array([r[1] for r in readings])
    A = np.c_[np.ones(len(zs)), xy]
    coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
    fit_err = float(np.max(np.abs(A @ coef - zs)))
    at_screw = {name: float(coef[0] + coef[1] * pos[0] + coef[2] * pos[1]) for name, pos in screws}
    ref = min(at_screw.values())

    print()
    print(f"Tilt: {coef[1] * 100:+.2f} mm per 100 mm left->right, {coef[2] * 100:+.2f} mm per 100 mm front->back"
          f"  (bed twist/warp not explained by a plane: {fit_err:.3f} mm)")
    print()
    print(f"{'Corner':14s} {'height':>8s}   action")
    level = True
    for name, pos in screws:
        dz = at_screw[name] - ref
        if dz <= OK_MM:
            action = "OK"
        else:
            level = False
            turns = dz / M4_PITCH_MM
            action = f"LOWER by {dz:.2f} mm  (~{turns:.1f} turn{'s' if turns >= 1.05 else ''}: tighten the knob)"
        print(f"{name:14s} {dz:+8.3f}   {action}")
    print()
    print("All corners within %.2f mm: bed is level." % OK_MM if level else
          "Adjust the corners marked LOWER, then run this again.")
    # Leave the head clear of the bed, at the centre.
    gcode(url, f"G0 Z{args.travel_z:.1f} F600\nG0 X{(lo[0] + hi[0]) / 2:.1f} Y{(lo[1] + hi[1]) / 2:.1f} F4800")


if __name__ == "__main__":
    main()
