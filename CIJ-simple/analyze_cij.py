#!/usr/bin/env python3
"""Compute Cij, Young's modulus and bulk modulus from Cij strain jobs.

Reads stress_{mode}_{sign}.txt files produced by run_cij.py and fits the
isotropic elastic constants from the +/-0.003 strain pairs.
"""

import csv
import statistics
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
STRAIN = 0.003
MODES = ["xx", "yy", "zz", "xy", "xz", "yz"]
SIGNS = [1.0, -1.0]


def read_stress(path):
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            q = line.split()
            if len(q) >= 7 and q[0].replace(".", "", 1).isdigit():
                vals = [float(x) for x in q[-6:]]
                return vals
    raise RuntimeError(f"no data in {path}")


def frame_cij(frame_dir):
    p = {}
    for mode in MODES:
        for sign in SIGNS:
            sign_txt = "+" if sign > 0 else "-"
            path = frame_dir / f"{mode}{sign_txt}" / f"stress_{mode}_{sign_txt}.txt"
            vals = read_stress(path)
            # LAMMPS pressure tensor order: xx yy zz yz xz xy
            p[(mode, sign_txt)] = {
                "pxx": vals[0],
                "pyy": vals[1],
                "pzz": vals[2],
                "pyz": vals[3],
                "pxz": vals[4],
                "pxy": vals[5],
            }

    c11 = []
    c12 = []
    for mode, axis in [("xx", "pxx"), ("yy", "pyy"), ("zz", "pzz")]:
        for sign in SIGNS:
            st = "+" if sign > 0 else "-"
            eps = sign * STRAIN
            stress_axis = -p[(mode, st)][axis]
            c11.append(stress_axis / eps)
            off = ["pxx", "pyy", "pzz"]
            off.remove(axis)
            for o in off:
                c12.append(-p[(mode, st)][o] / eps)

    c44 = []
    for mode, comp in [("xy", "pxy"), ("xz", "pxz"), ("yz", "pyz")]:
        for sign in SIGNS:
            st = "+" if sign > 0 else "-"
            gamma = sign * STRAIN
            c44.append(-p[(mode, st)][comp] / gamma)

    c11v = float(np.mean(c11))
    c12v = float(np.mean(c12))
    c44v = float(np.mean(c44))
    e = c44v * (3.0 * c11v - 4.0 * c44v) / (c11v - c44v)
    k = (c11v + 2.0 * c12v) / 3.0
    gpa = 1.01325e-4  # real units: pressure in atm -> GPa
    return {
        "C11_GPa": c11v * gpa,
        "C12_GPa": c12v * gpa,
        "C44_GPa": c44v * gpa,
        "Young_GPa": e * gpa,
        "Bulk_GPa": k * gpa,
        "Shear_GPa": c44v * gpa,
    }


def main():
    rows = []
    for frame_dir in sorted(
        (SCRIPT_DIR / "strain").iterdir(),
        key=lambda p: p.name,
    ):
        if not frame_dir.is_dir() or not frame_dir.name.startswith("frame"):
            continue
        rows.append({"frame": frame_dir.name, **frame_cij(frame_dir)})

    if not rows:
        raise SystemExit("no strain job outputs found")

    with open(SCRIPT_DIR / "cij_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'frame':<8} {'C11':>8} {'C12':>8} {'C44':>8} {'E':>8} {'K':>8} {'G':>8}")
    for r in rows:
        print(
            f"{r['frame']:<8} {r['C11_GPa']:8.2f} {r['C12_GPa']:8.2f} "
            f"{r['C44_GPa']:8.2f} {r['Young_GPa']:8.2f} "
            f"{r['Bulk_GPa']:8.2f} {r['Shear_GPa']:8.2f}"
        )

    keys = ["C11_GPa", "C12_GPa", "C44_GPa", "Young_GPa", "Bulk_GPa", "Shear_GPa"]
    print("\naverage over frames:")
    for k in keys:
        vals = [r[k] for r in rows]
        print(f"  {k}: {statistics.mean(vals):.2f} +/- {statistics.stdev(vals):.2f}")


if __name__ == "__main__":
    main()
