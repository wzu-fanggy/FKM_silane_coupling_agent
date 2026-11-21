#!/usr/bin/env python3
"""Post-process pull-out raw force data with multiple denoising methods.

Reads pullout_raw.txt (step displacement force) produced by
pullout_noise_reduction.lammps and renders a comparison figure with:
  - raw instantaneous force
  - Savitzky-Golay smoothing
  - displacement-binned average
  - median filter
  - LAMMPS 1 ps time-average (pullout_smooth.txt, optional)
Also prints peak force, total work and plateau force.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from scipy.integrate import cumulative_trapezoid


def load_raw(path):
    rows = []
    with path.open(encoding="ascii", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 3:
                continue
            try:
                rows.append([float(p[0]), float(p[1]), float(p[2])])
            except ValueError:
                continue
    a = np.asarray(rows)
    # keep the last entry at any duplicated step (chunk boundaries)
    _, idx = np.unique(a[:, 0], return_index=True)
    return a[np.sort(idx)]


def load_smooth(path):
    if not path.exists():
        return None
    rows = []
    with path.open(encoding="ascii", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 2:
                continue
            try:
                rows.append([float(p[0]), float(p[1])])
            except ValueError:
                continue
    return np.asarray(rows)


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    raw_path = work / "pullout_raw.txt"
    smooth_path = work / "pullout_smooth.txt"
    out_png = work / "denoise_compare.png"

    raw = load_raw(raw_path)
    if raw.size == 0:
        raise SystemExit("no raw data")
    step, disp, force = raw[:, 0], raw[:, 1], raw[:, 2]

    # Savitzky-Golay (odd window, clamp to data size, cap to avoid
    # edge artifacts on long curves)
    w = min(31, len(force) // 2 * 2 - 1)
    if w < 5:
        w = 5
    sg = savgol_filter(force, w, 2)

    # displacement-bin average (0.5 A bins)
    bins = np.arange(0.0, np.ceil(disp.max() + 0.5), 0.5)
    idx = np.digitize(disp, bins)
    bcenter = np.array([(bins[i - 1] + bins[i]) / 2 for i in range(1, len(bins))])
    bforce = np.array(
        [force[idx == i].mean() for i in range(1, len(bins)) if (idx == i).any()]
    )
    bcenter = bcenter[: len(bforce)]

    # median filter
    wm = min(21, len(force) // 2 * 2 - 1)
    if wm < 3:
        wm = 3
    med = median_filter(force, size=wm)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(disp, force, "o", ms=2, alpha=0.25, label="raw")
    ax.plot(disp, sg, lw=2, label=f"Savitzky-Golay (w={w})")
    if bcenter.size:
        ax.plot(bcenter, bforce, "s-", ms=3, lw=1.2, label="0.5 A bin average")
    ax.plot(disp, med, lw=1.6, ls="--", label=f"median (w={wm})")

    smooth = load_smooth(smooth_path)
    if smooth is not None and smooth.size:
        # smooth file's first column is TimeStep, not displacement.
        # Map it back to displacement using the raw step->disp relation.
        s_step = smooth[:, 0]
        s_disp = np.interp(s_step, step, disp)
        ax.plot(s_disp, smooth[:, 1], lw=2.5, label="LAMMPS 1 ps average")

    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("displacement (A)")
    ax.set_ylabel("pull force (kcal/mol/A)")
    ax.set_title("Pull-out force denoising comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)

    # metrics on smoothed curve (early elastic peak lives in first 10 A)
    early = disp < 10.0
    imax = int(np.argmax(sg[early]))
    work_tot = cumulative_trapezoid(sg, disp, initial=0.0)[-1]
    late = disp > 20.0
    f_plateau = sg[late].mean() if late.any() else 0.0

    print(f"points = {len(force)}")
    print(f"raw peak = {force.max():.1f} at {disp[np.argmax(force)]:.1f} A")
    print(f"SG peak  = {sg[imax]:.1f} at {disp[imax]:.1f} A")
    print(f"plateau avg (>20 A) = {f_plateau:.1f}")
    print(f"total work = {work_tot:.0f} kcal/mol")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
