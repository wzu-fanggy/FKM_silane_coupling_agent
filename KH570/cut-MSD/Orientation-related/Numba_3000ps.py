#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orientation ACF for the KH570-grafted FKM/SiO2 system.

Single continuous 3000 ps NVT trajectory (no concatenation needed).
Only FKM bonds are used; layer assignment follows the KH570 radial
density / MSD convention: R_SURF=15.05 A, interface thickness=6.08 A.
"""

import os
import re
import time

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(SCRIPT_DIR, "nvt_3000ps.data")
TRAJ = os.path.join(SCRIPT_DIR, "nvt_3000ps_traj.dump")
OUT_DIR = SCRIPT_DIR

R_SURF = 15.05
DELTA_PERT = 6.08
LAYERS = {
    "contact":   (R_SURF,          R_SURF + 3.0),
    "interface": (R_SURF + 3.0,    R_SURF + DELTA_PERT),
    "bulk":      (R_SURF + DELTA_PERT, 1e6),
}
LAYER_NAME = ["contact", "interface", "bulk"]
SIO2_TYPES = {5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17}
FKM_TYPES = {1, 2, 3, 4, 8, 10}

VEC_FILE = os.path.join(OUT_DIR, "bond_vecs_3000ps_fkm.f32")
LAYER_FILE = os.path.join(OUT_DIR, "bond_layers_3000ps_fkm.npy")


def read_frames(path):
    frames = []
    frame_buf = {}
    reading_atoms = False
    atoms_to_read = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if reading_atoms:
                p = line.split()
                frame_buf["atoms"][int(p[0])] = (
                    int(p[1]), float(p[2]), float(p[3]), float(p[4]),
                    int(p[5]), int(p[6]), int(p[7]),
                )
                atoms_to_read -= 1
                if atoms_to_read == 0:
                    reading_atoms = False
                    frames.append(frame_buf)
                    frame_buf = {}
            elif line == "ITEM: TIMESTEP":
                if frame_buf:
                    frames.append(frame_buf)
                frame_buf = {"step": int(f.readline().strip())}
            elif line == "ITEM: NUMBER OF ATOMS":
                frame_buf["n"] = int(f.readline().strip())
            elif "BOX BOUNDS" in line:
                frame_buf["bounds"] = (
                    tuple(map(float, f.readline().split())),
                    tuple(map(float, f.readline().split())),
                    tuple(map(float, f.readline().split())),
                )
            elif line.startswith("ITEM: ATOMS"):
                frame_buf["atoms"] = {}
                reading_atoms = True
                atoms_to_read = frame_buf.get("n", 0)
    if frame_buf:
        frames.append(frame_buf)
    return frames


print("=" * 60)
print("Orientation ACF - KH570 FKM/SiO2 (3000 ps, single trajectory)")
print("=" * 60)
t0 = time.time()

print("\n[1/6] Parsing data file...", flush=True)
with open(DATA, "r", encoding="utf-8", errors="replace") as f:
    content = f.read()

atoms_match = content.find("Atoms # full")
bonds_match = content.find("Bonds")
angles_match = content.find("Angles")

atom_type = {}
for line in content[atoms_match:bonds_match].split("\n"):
    parts = line.split()
    if len(parts) >= 7 and parts[0].isdigit():
        atom_type[int(parts[0])] = int(parts[2])

m = re.search(r"^(\d+)\s+bonds\s*$", content, re.MULTILINE)
n_bonds_header = int(m.group(1))

bonds = []
bond_count = 0
for line in content[bonds_match:angles_match].split("\n"):
    parts = line.split()
    if len(parts) >= 4 and parts[0].isdigit():
        bonds.append((int(parts[2]), int(parts[3])))
        bond_count += 1
        if bond_count >= n_bonds_header:
            break

n_atoms = len(atom_type)
n_bonds = len(bonds)
print(f"  {n_atoms} atoms, {n_bonds} bonds", flush=True)

fkm_bond_mask = np.array([
    atom_type.get(b[0], -1) in FKM_TYPES and
    atom_type.get(b[1], -1) in FKM_TYPES
    for b in bonds
], dtype=bool)
fkm_idx = np.where(fkm_bond_mask)[0]
n_bonds_fkm = len(fkm_idx)
print(f"  FKM bonds only: {n_bonds_fkm} (excluded {n_bonds - n_bonds_fkm})", flush=True)

print("\n[2/6] Loading trajectory...", flush=True)
frames = read_frames(TRAJ)
n_frames = len(frames)
print(f"  {n_frames} frames loaded, elapsed {time.time() - t0:.1f}s", flush=True)

(bx0, bx1), (by0, by1), (bz0, bz1) = frames[0]["bounds"]
Lx, Ly, Lz = bx1 - bx0, by1 - by0, bz1 - bz0
print(f"  Box: {Lx:.2f} x {Ly:.2f} x {Lz:.2f} A", flush=True)

sio2_ids = [aid for aid, d in frames[0]["atoms"].items() if d[0] in SIO2_TYPES]
n_sio2 = len(sio2_ids)
print(f"  SiO2+KH570 cluster atoms: {n_sio2}", flush=True)

bond_a1 = np.array([bonds[i][0] for i in fkm_idx], dtype=np.int32)
bond_a2 = np.array([bonds[i][1] for i in fkm_idx], dtype=np.int32)
bond_layer_arr = np.zeros(n_bonds_fkm, dtype=np.int32)

sio2_com = np.zeros(3)
for aid in sio2_ids:
    _, x, y, z, ix, iy, iz = frames[0]["atoms"][aid]
    sio2_com += np.array([x + ix * Lx, y + iy * Ly, z + iz * Lz])
sio2_com /= n_sio2

for i, bi in enumerate(fkm_idx):
    a1, a2 = bonds[bi]
    _, x1, y1, z1, ix1, iy1, iz1 = frames[0]["atoms"][a1]
    _, x2, y2, z2, ix2, iy2, iz2 = frames[0]["atoms"][a2]
    mx = (x1 + ix1 * Lx + x2 + ix2 * Lx) / 2 - sio2_com[0]
    my = (y1 + iy1 * Ly + y2 + iy2 * Ly) / 2 - sio2_com[1]
    mz = (z1 + iz1 * Lz + z2 + iz2 * Lz) / 2 - sio2_com[2]
    r = np.sqrt(mx * mx + my * my + mz * mz)
    for li, (name, (lo, hi)) in enumerate(LAYERS.items()):
        if lo <= r < hi:
            bond_layer_arr[i] = li
            break

n_c = int(np.sum(bond_layer_arr == 0))
n_i = int(np.sum(bond_layer_arr == 1))
n_b = int(np.sum(bond_layer_arr == 2))
print(f"  Contact: {n_c}, Interface: {n_i}, Bulk: {n_b}", flush=True)

print("\n[3/6] Computing bond vectors...", flush=True)
bond_vecs_all = np.zeros((n_frames, n_bonds_fkm, 3), dtype=np.float32)
for fi, fr in enumerate(frames):
    atoms = fr["atoms"]
    cx = sum(atoms[a][1] + atoms[a][4] * Lx for a in sio2_ids) / n_sio2
    cy = sum(atoms[a][2] + atoms[a][5] * Ly for a in sio2_ids) / n_sio2
    cz = sum(atoms[a][3] + atoms[a][6] * Lz for a in sio2_ids) / n_sio2

    a1_coords = np.array(
        [[atoms[a][1] + atoms[a][4] * Lx,
          atoms[a][2] + atoms[a][5] * Ly,
          atoms[a][3] + atoms[a][6] * Lz] for a in bond_a1],
        dtype=np.float32,
    )
    a2_coords = np.array(
        [[atoms[a][1] + atoms[a][4] * Lx,
          atoms[a][2] + atoms[a][5] * Ly,
          atoms[a][3] + atoms[a][6] * Lz] for a in bond_a2],
        dtype=np.float32,
    )

    vx = a2_coords[:, 0] - a1_coords[:, 0] - Lx * np.round((a2_coords[:, 0] - a1_coords[:, 0]) / Lx)
    vy = a2_coords[:, 1] - a1_coords[:, 1] - Ly * np.round((a2_coords[:, 1] - a1_coords[:, 1]) / Ly)
    vz = a2_coords[:, 2] - a1_coords[:, 2] - Lz * np.round((a2_coords[:, 2] - a1_coords[:, 2]) / Lz)
    norm = np.sqrt(vx * vx + vy * vy + vz * vz)
    norm[norm == 0] = 1.0
    bond_vecs_all[fi, :, 0] = vx / norm
    bond_vecs_all[fi, :, 1] = vy / norm
    bond_vecs_all[fi, :, 2] = vz / norm

    if (fi + 1) % 300 == 0 or fi == n_frames - 1:
        print(f"    Frame {fi+1}/{n_frames}, elapsed {time.time() - t0:.1f}s", flush=True)

del frames

print("\n[4/6] Saving cache...", flush=True)
bond_vecs_all.tofile(VEC_FILE)
np.save(LAYER_FILE, bond_layer_arr)
print(f"  {VEC_FILE} ({os.path.getsize(VEC_FILE) // 1024 // 1024} MB)", flush=True)

print("\n[5/6] Computing C2 ACF...", flush=True)
max_dt = n_frames - 1
layer_masks = [bond_layer_arr == li for li in range(3)]
n_layer_bonds = [int(mask.sum()) for mask in layer_masks]
print(f"  Layer bond counts: {n_layer_bonds}", flush=True)

from numba import njit, prange

@njit(parallel=True, cache=True)
def compute_acf_numba(vecs, max_dt, n_bonds_layer):
    n_frames = vecs.shape[0]
    c2_vals = np.zeros(max_dt + 1, dtype=np.float64)
    c2_vals[0] = 1.0
    for dt in prange(1, max_dt + 1):
        n_pairs = n_frames - dt
        accum = 0.0
        for t in range(n_pairs):
            for b in range(n_bonds_layer):
                dot = (vecs[t, b, 0] * vecs[t + dt, b, 0] +
                       vecs[t, b, 1] * vecs[t + dt, b, 1] +
                       vecs[t, b, 2] * vecs[t + dt, b, 2])
                accum += 0.5 * (3.0 * dot * dot - 1.0)
        c2_vals[dt] = accum / (n_pairs * n_bonds_layer)
    return c2_vals

print("  Compiling Numba kernel...", flush=True)
_dummy = np.random.randn(10, 5, 3)
_ = compute_acf_numba(_dummy, 5, 5)

C2 = {}
for li, name in enumerate(LAYER_NAME):
    if n_layer_bonds[li] == 0:
        C2[name] = np.zeros(max_dt + 1)
        continue
    v = bond_vecs_all[:, layer_masks[li], :].astype(np.float64)
    print(f"  Computing {name} (bonds={n_layer_bonds[li]})...", flush=True)
    t_acf = time.time()
    C2[name] = compute_acf_numba(v, max_dt, n_layer_bonds[li])
    print(f"    Done in {time.time() - t_acf:.1f}s", flush=True)

print(f"  Done, elapsed {time.time() - t0:.1f}s", flush=True)

print("\n[6/6] Saving results...", flush=True)
csv_path = os.path.join(OUT_DIR, "orientation_acf_3000ps.csv")
with open(csv_path, "w") as f:
    f.write("dt_ps,contact_C2,interface_C2,bulk_C2\n")
    for dt in range(1, max_dt + 1):
        f.write(f"{dt},{C2['contact'][dt]:.6f},{C2['interface'][dt]:.6f},{C2['bulk'][dt]:.6f}\n")
print(f"  CSV: {csv_path}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
colors = {"contact": "#E63946", "interface": "#2A9D8F", "bulk": "#457B9D"}
dt_arr = np.arange(1, max_dt + 1)
for name in LAYER_NAME:
    ax.plot(dt_arr, C2[name][1:], color=colors[name], label=name, lw=2)
ax.axhline(y=1.0 / np.e, color="gray", linestyle="--", lw=1, label=r"$1/e$")
ax.set_xlabel(r"$\Delta t$ (ps)", fontsize=13)
ax.set_ylabel(r"$C_2(\Delta t)$", fontsize=13)
ax.set_title("Segmental Orientation ACF - KH570 FKM/SiO2 (3000 ps NVT)", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max_dt)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "orientation_acf_3000ps.png")
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  PNG: {fig_path}")

print()
print("=" * 60)
print("Results")
print("=" * 60)
for name in LAYER_NAME:
    c = C2[name]
    tau_vals = np.where(c[1:] < 1.0 / np.e)[0]
    tau_str = str(tau_vals[0] + 1) if len(tau_vals) > 0 else "NOT_REACHED"
    print(
        f"  {name:10s}: C2(1)={c[1]:.4f}, C2(100)={c[100]:.4f}, "
        f"C2(500)={c[500]:.4f}, C2(1000)={c[1000]:.4f}, "
        f"C2({max_dt})={c[max_dt]:.4f}, tau~{tau_str} ps"
    )
print(f"\nTotal time: {time.time() - t0:.1f}s")
print("Done!")
