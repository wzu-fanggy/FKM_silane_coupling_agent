"""Accurate SiO2 surface geometry from the LAMMPS coordinate dump.

Run after in.geometry.lammps:
    python compute_geometry.py

Outputs:
    geometry_results.csv
    radial_density.png
    sio2_surface.png
"""

from pathlib import Path
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DUMP = ROOT / "geometry_atoms.dump"
DATA = ROOT / "pull_equil" / "pull_equil.data"
CSV_OUT = ROOT / "geometry_results.csv"


def read_dump(path):
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    box = {}
    atoms = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("ITEM: BOX BOUNDS"):
            i += 1
            for axis in ("x", "y", "z"):
                parts = lines[i].split()
                box[axis + "lo"] = float(parts[0])
                box[axis + "hi"] = float(parts[1])
                i += 1
            continue
        if line.startswith("ITEM: ATOMS"):
            cols = line.split()[2:]
            i += 1
            while i < len(lines) and not lines[i].startswith("ITEM:"):
                parts = lines[i].split()
                if len(parts) == len(cols):
                    atoms.append({k: (int(v) if k in ("id", "type", "ix", "iy", "iz") else float(v))
                                  for k, v in zip(cols, parts)})
                i += 1
            continue
        i += 1
    return box, atoms


def read_data(path):
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    box = {}
    atoms = []
    in_atoms = False
    for line in lines:
        s = line.strip()
        if s == "Atoms # full":
            in_atoms = True
            continue
        if in_atoms:
            if not s:
                continue
            if s.startswith("Bonds"):
                break
            p = s.split()
            if len(p) >= 10:
                atoms.append({
                    "id": int(p[0]), "type": int(p[2]),
                    "x": float(p[4]), "y": float(p[5]), "z": float(p[6]),
                    "ix": int(p[7]), "iy": int(p[8]), "iz": int(p[9]),
                })
        parts = s.split()
        if len(parts) == 4 and parts[2] == "xlo" and parts[3] == "xhi":
            box["xlo"], box["xhi"] = float(parts[0]), float(parts[1])
        elif len(parts) == 4 and parts[2] == "ylo" and parts[3] == "yhi":
            box["ylo"], box["yhi"] = float(parts[0]), float(parts[1])
        elif len(parts) == 4 and parts[2] == "zlo" and parts[3] == "zhi":
            box["zlo"], box["zhi"] = float(parts[0]), float(parts[1])
    return box, atoms


def atoms_to_arrays(atoms):
    ids = np.array([a["id"] for a in atoms], dtype=int)
    types = np.array([a["type"] for a in atoms], dtype=int)
    x = np.array([a["x"] for a in atoms], dtype=float)
    y = np.array([a["y"] for a in atoms], dtype=float)
    z = np.array([a["z"] for a in atoms], dtype=float)
    ix = np.array([a["ix"] for a in atoms], dtype=int)
    iy = np.array([a["iy"] for a in atoms], dtype=int)
    return ids, types, x, y, z, ix, iy


def radial_density(rs, rmax, dr=0.2):
    bins = np.arange(0.0, rmax + dr, dr)
    counts, _ = np.histogram(rs, bins=bins)
    centers = bins[:-1] + dr / 2
    vols = (4.0 / 3.0) * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    density = counts / np.maximum(vols, 1e-12)
    return centers, density


def main():
    if DUMP.exists() and DUMP.stat().st_size > 0:
        box, atoms = read_dump(DUMP)
        source = DUMP.name
    elif DATA.exists():
        box, atoms = read_data(DATA)
        source = str(DATA.relative_to(ROOT))
    else:
        sys.exit("No geometry_atoms.dump and no pull_equil/pull_equil.data found.")

    ids, types, x, y, z, ix, iy = atoms_to_arrays(atoms)
    lx = box["xhi"] - box["xlo"]
    ly = box["yhi"] - box["ylo"]
    xu = x + ix * lx
    yu = y + iy * ly

    sio2 = types >= 7
    fkm = types <= 6
    n_sio2 = int(sio2.sum())
    n_fkm = int(fkm.sum())

    com = np.array([xu[sio2].mean(), yu[sio2].mean(), z[sio2].mean()])

    def dists(mask_x, mask_y, mask_z):
        dx = mask_x - com[0]
        dy = mask_y - com[1]
        dx -= np.round(dx / lx) * lx
        dy -= np.round(dy / ly) * ly
        dz = mask_z - com[2]
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    rs_sio2 = dists(xu[sio2], yu[sio2], z[sio2])
    rs_fkm = dists(xu[fkm], yu[fkm], z[fkm])

    rmax = max(float(rs_sio2.max()) * 1.2, float(rs_fkm.max()))
    r_sio2, rho_sio2 = radial_density(rs_sio2, rmax)
    r_fkm, rho_fkm = radial_density(rs_fkm, rmax)

    # SiO2 atomic density inside the particle is roughly constant; the surface
    # is taken where the density falls below half of the interior plateau.
    # The center can be noisy/hollow, so the search starts past the core.
    rg = float(np.sqrt(np.mean(rs_sio2**2)))
    plateau_lo = max(0.5 * rg, 4.0)
    plateau_hi = 0.85 * rg
    plateau = np.median(rho_sio2[(r_sio2 > plateau_lo) & (r_sio2 < plateau_hi) & (rho_sio2 > 0)])
    if not np.isfinite(plateau) or plateau <= 0:
        plateau = rho_sio2.max()
    start_r = max(0.7 * rg, 9.0)
    half_idx = np.where((r_sio2 > start_r) & (rho_sio2 < 0.5 * plateau))[0]
    r_surf = float(r_sio2[half_idx[0]]) if len(half_idx) else float(rs_sio2.max())
    r95 = float(np.percentile(rs_sio2, 95))
    area = 4.0 * np.pi * r_surf**2

    print("source    :", source)
    print("n_sio2    :", n_sio2)
    print("n_fkm     :", n_fkm)
    print("COM       :", np.round(com, 4))
    print("Rg (python):", round(rg, 4))
    print("R95       :", round(r95, 4))
    print("R_surf    :", round(r_surf, 4))
    print("A0 (4*pi*R^2):", round(area, 4), "A^2")

    with CSV_OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows([
            ("source", source),
            ("n_sio2", n_sio2),
            ("n_fkm", n_fkm),
            ("com_x_A", round(float(com[0]), 6)),
            ("com_y_A", round(float(com[1]), 6)),
            ("com_z_A", round(float(com[2]), 6)),
            ("rg_A", round(rg, 6)),
            ("r95_A", round(r95, 6)),
            ("r_surf_A", round(r_surf, 6)),
            ("contact_area_A2", round(float(area), 6)),
            ("contact_area_nm2", round(float(area) * 0.01, 6)),
        ])

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(r_sio2, rho_sio2, lw=1.0, color="#1f77b4", label="SiO2 atomic density")
    ax.plot(r_fkm, rho_fkm, lw=1.0, color="#d62728", label="FKM atomic density")
    ax.axvline(r_surf, color="k", ls="--", lw=1.0, label=f"R_surf = {r_surf:.2f} A")
    ax.set_xlabel("Distance from SiO2 COM (A)")
    ax.set_ylabel("Number density (atoms/A^3)")
    ax.set_title("Radial density around SiO2")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "radial_density.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    sc = ax.scatter(xu[sio2], yu[sio2], c=rs_sio2, s=2, cmap="viridis")
    ax.set_aspect("equal")
    ax.set_xlabel("Unwrapped x (A)")
    ax.set_ylabel("Unwrapped y (A)")
    ax.set_title("SiO2 atoms colored by distance from COM")
    fig.colorbar(sc, ax=ax, label="r (A)")
    fig.tight_layout()
    fig.savefig(ROOT / "sio2_surface.png")
    plt.close(fig)

    print("wrote", CSV_OUT)
    print("wrote radial_density.png and sio2_surface.png")


if __name__ == "__main__":
    main()
