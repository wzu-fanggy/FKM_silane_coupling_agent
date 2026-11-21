"""Compute SiO2 radius percentiles and initial max-z from a LAMMPS data file.

Usage:
    python compute_sio2_radius.py [data_file]

Writes radius_params.txt next to the data file (or in the current directory).
"""

import math
import re
import sys
from pathlib import Path


SIO2_TYPES = {7, 8, 9}


def main():
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])
    else:
        data_path = Path(
            r"F:\Lammps-MS\FKM-SIO2-TEST\pull\NEW\ex\pull_equil_600Aadd.data"
        )

    box = {}
    atoms = []
    in_atoms = False

    with open(data_path, "r", encoding="ascii") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m = re.match(
                r"^([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+(xlo xhi|ylo yhi|zlo zhi)$", line
            )
            if m:
                key = m.group(3).split()[0]
                box[key] = (float(m.group(1)), float(m.group(2)))
                continue

            if line.startswith("Atoms"):
                in_atoms = True
                continue
            if line.startswith("Bonds"):
                in_atoms = False
                continue
            if not in_atoms:
                continue

            parts = line.split()
            if len(parts) != 10:
                continue
            try:
                fields = [float(x) for x in parts]
            except ValueError:
                continue
            atoms.append(fields)

    if not atoms:
        raise SystemExit("No atoms parsed. Check the data file format.")
    if len(box) != 3:
        raise SystemExit("Box bounds not parsed correctly.")

    lengths = {
        "x": box["xlo"][1] - box["xlo"][0],
        "y": box["ylo"][1] - box["ylo"][0],
        "z": box["zlo"][1] - box["zlo"][0],
    }

    z0 = max(a[6] + a[9] * lengths["z"] for a in atoms)
    sio2 = [a for a in atoms if int(a[2]) in SIO2_TYPES]
    com = []
    for idx, key in ((4, "x"), (5, "y"), (6, "z")):
        com.append(
            sum(a[idx] + a[idx + 3] * lengths[key] for a in sio2) / len(sio2)
        )
    dists = [
        math.sqrt(
            (a[4] + a[7] * lengths["x"] - com[0]) ** 2
            + (a[5] + a[8] * lengths["y"] - com[1]) ** 2
            + (a[6] + a[9] * lengths["z"] - com[2]) ** 2
        )
        for a in sio2
    ]
    dists.sort()
    idx95 = min(len(dists) - 1, int(math.ceil(0.95 * len(dists))) - 1)
    r95 = dists[idx95]
    rmean = sum(dists) / len(dists)
    rmax = dists[-1]

    text = (
        f"Z0 = {z0:.6f}\n"
        f"R95 = {r95:.6f}\n"
        f"Rmean = {rmean:.6f}\n"
        f"Rmax = {rmax:.6f}\n"
        f"D = {2 * r95:.6f}\n"
    )
    print(text, end="")

    out = data_path.parent / "radius_params.txt"
    out.write_text(text, encoding="ascii")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
