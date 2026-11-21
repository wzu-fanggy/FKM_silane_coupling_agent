#!/usr/bin/env python3
"""Corrected radial mass density distribution for the bare SiO2/FKM system.

Mass is read from the data file, the SiO2 COM is mass-weighted, and distances
use the minimum-image convention.  This replaces the old script whose hardcoded
atom masses did not match the typed PCFF data.
"""

import csv
import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DUMP = SCRIPT_DIR.parent / "nvt_500ps_gpu_end.dump"
INPUT_DATA = SCRIPT_DIR.parent / "nvt_500ps_gpu.data"
OUT_PNG = SCRIPT_DIR / "radial_mass_density_bare.png"
OUT_TXT = SCRIPT_DIR / "radial_density_data_bare.txt"
OUT_CSV = SCRIPT_DIR / "radial_density_data_bare.csv"

R_SURF = 15.03
DELTA_PERT = 8.59
SIO2_TYPES = {7, 8, 9}
FKM_TYPES = {1, 2, 3, 4, 5, 6}

R_MAX = 55.0
DR = 0.5
SMOOTH_WINDOW = 5
DENSITY_CONVERSION = 1.66053906660


def find_startswith(lines, prefix):
    for i, line in enumerate(lines):
        if line.strip().startswith(prefix):
            return i
    raise RuntimeError(f"section not found: {prefix}")


def parse_masses(path):
    lines = path.read_text(encoding="ascii").splitlines()
    mi = next(i for i, l in enumerate(lines) if l.strip() == "Masses")
    pi = find_startswith(lines, "Pair Coeffs")
    mass = {}
    for line in lines[mi + 1 : pi]:
        q = line.split()
        if len(q) >= 2 and q[0].isdigit():
            mass[int(q[0])] = float(q[1])
    return mass


def read_last_frame(path):
    frame = None
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        line = f.readline()
        while line:
            if not line.startswith("ITEM: TIMESTEP"):
                line = f.readline()
                continue
            step = int(f.readline().strip())
            f.readline()
            natoms = int(f.readline().strip())
            f.readline()
            box = []
            for _ in range(3):
                lo, hi = f.readline().split()
                box.append((float(lo), float(hi)))
            f.readline()
            ids = np.empty(natoms, dtype=np.int64)
            types = np.empty(natoms, dtype=np.int64)
            pos = np.empty((natoms, 3))
            img = np.empty((natoms, 3), dtype=np.int64)
            for i in range(natoms):
                q = f.readline().split()
                ids[i] = int(q[0])
                types[i] = int(q[1])
                pos[i] = [float(q[2]), float(q[3]), float(q[4])]
                img[i] = [int(q[5]), int(q[6]), int(q[7])]
            frame = {"step": step, "box": box, "ids": ids, "types": types,
                     "pos": pos, "img": img}
            line = f.readline()
    if frame is None:
        raise RuntimeError(f"no frame found in {path}")
    return frame


def mass_weighted_com(frame, types, mass):
    sel = np.isin(frame["types"], list(types))
    span = np.array([b[1] - b[0] for b in frame["box"]])
    unwrapped = frame["pos"] + frame["img"] * span
    m = np.array([mass[int(t)] for t in frame["types"][sel]], dtype=float)
    return np.sum(unwrapped[sel] * m[:, None], axis=0) / m.sum(), span


def min_image_dist(pos, com, span):
    delta = pos - com
    delta -= np.round(delta / span) * span
    return np.linalg.norm(delta, axis=1)


def moving_average(data, window):
    out = np.empty_like(data)
    half = window // 2
    for i in range(len(data)):
        out[i] = data[max(0, i - half) : min(len(data), i + half + 1)].mean()
    return out


def main():
    mass = parse_masses(INPUT_DATA)
    frame = read_last_frame(INPUT_DUMP)
    com, span = mass_weighted_com(frame, SIO2_TYPES, mass)
    print(f"timestep {frame['step']}, atoms {len(frame['ids'])}")
    print(f"SiO2 COM = ({com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f}) A")

    r = min_image_dist(frame["pos"], com, span)
    masses = np.array([mass[int(t)] for t in frame["types"]], dtype=float)

    n_bins = int(R_MAX / DR)
    edges = np.linspace(0.0, R_MAX, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    density = np.zeros(n_bins)
    for i in range(len(r)):
        if r[i] < R_MAX:
            density[min(n_bins - 1, int(r[i] / DR))] += masses[i]
    for j in range(n_bins):
        vol = (4.0 / 3.0) * math.pi * (edges[j + 1] ** 3 - edges[j] ** 3)
        density[j] /= vol

    rho = density * DENSITY_CONVERSION
    rho_smooth = moving_average(rho, SMOOTH_WINDOW)

    layers = [
        ("Core (SiO2)", 0.0, R_SURF),
        ("Contact", R_SURF, R_SURF + 3.0),
        ("Interface", R_SURF + 3.0, R_SURF + DELTA_PERT),
        ("Bulk", R_SURF + DELTA_PERT, 1e6),
    ]
    print(f"{'layer':<18} {'r range (A)':<22} {'rho (g/cm3)':<14} {'atoms'}")
    stats = []
    for name, rmin, rmax in layers:
        mask = (r >= rmin) & (r < rmax)
        vol = (4.0 / 3.0) * math.pi * (min(rmax, r.max()) ** 3 - rmin**3)
        rho_avg = masses[mask].sum() / vol * DENSITY_CONVERSION if mask.any() else 0.0
        stats.append((name, rmin, min(rmax, r.max()), rho_avg, int(mask.sum())))
        print(f"{name:<18} {rmin:6.2f}-{min(rmax, r.max()):6.2f} "
              f"{rho_avg:14.4f} {int(mask.sum())}")

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(centers, rho_smooth, "b-", lw=2.2,
            label=f"mass density (smoothed, w={SMOOTH_WINDOW})")
    ax.plot(centers, rho, "gray", lw=0.7, alpha=0.4, label="raw")
    for name, rmin, rmax, _, _ in stats:
        ax.axvline(rmin, color="tab:red", ls="--", lw=1.2)
    ax.set_xlabel("Distance from SiO2 COM (A)")
    ax.set_ylabel("Mass density (g/cm3)")
    ax.set_title(f"Bare SiO2/FKM radial mass density (t={frame['step']} fs)")
    ax.set_xlim(0, R_MAX)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("# Bare SiO2/FKM RMDD\n")
        f.write(f"# timestep {frame['step']}\n")
        f.write(f"# SiO2 COM ({com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f}) A\n")
        for name, rmin, rmax, rho_avg, n in stats:
            f.write(f"# {name}: R=[{rmin:.2f},{rmax:.2f}] A, "
                    f"rho={rho_avg:.4f} g/cm3, atoms={n}\n")
        f.write("# r(A) rho_raw(g/cm3) rho_smooth(g/cm3)\n")
        for i in range(n_bins):
            f.write(f"{centers[i]:.4f} {rho[i]:.6f} {rho_smooth[i]:.6f}\n")

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["r(A)", "rho_raw(g/cm3)", "rho_smooth(g/cm3)"])
        for i in range(n_bins):
            writer.writerow([f"{centers[i]:.4f}", f"{rho[i]:.6f}",
                             f"{rho_smooth[i]:.6f}"])

    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_TXT}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
