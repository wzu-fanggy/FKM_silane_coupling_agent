#!/usr/bin/env python3
"""Count initial contact FKM atoms around the SiO2/KH570 cluster."""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


BARE = {
    "cluster": {7, 8, 9},
    "fkm": {1, 2, 3, 4, 5, 6},
}
KH570 = {
    "cluster": {5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17},
    "fkm": {1, 2, 3, 4, 8, 10},
}
KH560 = {
    "cluster": {5, 7, 9, 10, 11, 12, 13, 14, 15},
    "fkm": {1, 2, 3, 4, 6, 8},
}
KH550 = {
    "cluster": {6, 8, 9, 10, 11, 12, 13, 14},
    "fkm": {1, 2, 3, 4, 5, 7},
}


def parse_frame(path):
    lines = Path(path).read_text(encoding="ascii", errors="ignore").splitlines()
    box = {}
    atoms = []
    sec = None
    for line in lines:
        s = line.strip()
        if s == "Atoms # full":
            sec = "atoms"
            continue
        if s == "Velocities":
            sec = None
        q = line.split()
        if len(q) == 4 and q[2] in ("xlo", "ylo", "zlo"):
            box[q[2]] = (float(q[0]), float(q[1]))
        if sec == "atoms" and len(q) >= 10 and q[0].isdigit():
            atoms.append(
                (
                    int(q[2]),
                    float(q[4]),
                    float(q[5]),
                    float(q[6]),
                    int(q[7]),
                    int(q[8]),
                    int(q[9]),
                )
            )
    lengths = (
        box["xlo"][1] - box["xlo"][0],
        box["ylo"][1] - box["ylo"][0],
        box["zlo"][1] - box["zlo"][0],
    )
    return atoms, lengths


def contact_count(path, types, cutoff):
    atoms, (lx, ly, lz) = parse_frame(path)
    cluster = np.array(
        [
            [a[1] + a[4] * lx, a[2] + a[5] * ly, a[3] + a[6] * lz]
            for a in atoms
            if a[0] in types["cluster"]
        ],
        dtype=float,
    )
    fkm = np.array(
        [
            [a[1] + a[4] * lx, a[2] + a[5] * ly, a[3] + a[6] * lz]
            for a in atoms
            if a[0] in types["fkm"]
        ],
        dtype=float,
    )

    images = []
    for dx in (-lx, 0.0, lx):
        for dy in (-ly, 0.0, ly):
            im = cluster.copy()
            im[:, 0] += dx
            im[:, 1] += dy
            images.append(im)
    tree = cKDTree(np.vstack(images))
    pairs = tree.query_ball_point(fkm, r=cutoff)
    return sum(1 for p in pairs if p)


if __name__ == "__main__":
    base_bare = Path(
        r"F:\Lammps-MS\FKM-SIO2-TEST\sci-pull\first_deal\10A-20A\10A"
    )
    base_kh570 = Path(
        r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\sci-pull\first_deal\10A-20A\10A"
    )
    base_kh560 = Path(
        r"F:\Lammps-MS\FKM-SIO2-TEST\KH560\sci-pull\first_deal\10A-20A\10A"
    )
    base_kh550 = Path(
        r"F:\Lammps-MS\FKM-SIO2-TEST\KH550\sci-pull\first_deal\10A-20A\10A"
    )
    for cutoff in (3.0, 3.5):
        print(f"cutoff = {cutoff} A")
        for i in (1, 2, 3):
            bare = contact_count(base_bare / str(i) / f"frame{i}.data", BARE, cutoff)
            kh = contact_count(base_kh570 / str(i) / f"frame{i}.data", KH570, cutoff)
            kh560 = contact_count(base_kh560 / str(i) / f"frame{i}.data", KH560, cutoff)
            kh550 = contact_count(base_kh550 / str(i) / f"frame{i}.data", KH550, cutoff)
            print(
                f"  frame{i}: bare={bare}, KH570={kh}, KH560={kh560}, KH550={kh550}, "
                f"KH570/bare={kh / bare:.3f}, KH560/bare={kh560 / bare:.3f}, "
                f"KH550/bare={kh550 / bare:.3f}"
            )
