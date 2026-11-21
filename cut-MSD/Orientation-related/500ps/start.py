# -*- coding: utf-8 -*-
# Orientation Autocorrelation Function (C2) analysis for FKM/SiO2
# Based on Polymers 2021
# Usage: python orientation_acf.py

import numpy as np
import os, sys, time

# =============================================================================
# Configuration
# =============================================================================
DATA_PATH = r"F:\Lammps-MS\FKM-SIO2-TEST\cut-MSD\nvt_500ps_gpu.data"
TRAJ_PATH = r"F:\Lammps-MS\FKM-SIO2-TEST\cut-MSD\nvt_500ps_gpu_traj.dump"
OUT_DIR   = r"F:\Lammps-MS\FKM-SIO2-TEST\cut-MSD\Orientation-related"

# Parameters
R_SURF = 15.1
LAYERS = {
    "contact":   (R_SURF,            R_SURF + 3.0),
    "interface": (R_SURF + 3.0,    R_SURF + 8.6),
    "bulk":      (R_SURF + 8.6,      1e6),
}
SIO2_TYPES = {7, 8, 9}

print("=" * 60)
print("Orientation Autocorrelation Function Analysis")
print("=" * 60)
print(f"Data: {DATA_PATH}")
print(f"Traj: {TRAJ_PATH}")
print(f"Out:  {OUT_DIR}")
start_time = time.time()

# =============================================================================
# Parse data file
# =============================================================================
print("\n[1/6] Reading data file...", flush=True)
with open(DATA_PATH, 'r') as f:
    content = f.read()

atoms_match = content.find("Atoms # full")
bonds_match = content.find("Bonds")

atom_mol = {}
atoms_section = content[atoms_match:bonds_match]
for line in atoms_section.split('\n'):
    parts = line.split()
    if len(parts) >= 7 and parts[0].isdigit():
        atom_mol[int(parts[0])] = int(parts[1])

bonds = []
for line in content[bonds_match:].split('\n'):
    parts = line.split()
    if len(parts) >= 4 and parts[0].isdigit():
        bonds.append((int(parts[2]), int(parts[3])))

print(f"  Atoms: {len(atom_mol)}, Bonds: {len(bonds)}")

# =============================================================================
# Parse trajectory
# =============================================================================
print("\n[2/6] Reading trajectory...", flush=True)
frames = []
frame_buf = {}
reading_atoms = False

with open(TRAJ_PATH, 'r') as f:
    for raw in f:
        line = raw.strip()
        if reading_atoms:
            p = line.split()
            frame_buf["atoms"][int(p[0])] = (int(p[1]), float(p[2]), float(p[3]), float(p[4]), int(p[5]), int(p[6]), int(p[7]))
            atoms_to_read -= 1
            if atoms_to_read == 0:
                reading_atoms = False
                frames.append(frame_buf)
                frame_buf = {}
        elif line == "ITEM: TIMESTEP":
            if frame_buf: frames.append(frame_buf)
            frame_buf = {"step": int(f.readline().strip())}
        elif line == "ITEM: NUMBER OF ATOMS":
            frame_buf["n"] = int(f.readline().strip())
        elif "BOX BOUNDS" in line:
            frame_buf["bounds"] = (tuple(map(float, f.readline().split())), tuple(map(float, f.readline().split())), tuple(map(float, f.readline().split())))
        elif line.startswith("ITEM: ATOMS"):
            frame_buf["atoms"] = {}
            reading_atoms = True
            atoms_to_read = frame_buf.get("n", 0)

if frame_buf: frames.append(frame_buf)
print(f"  Frames: {len(frames)}, Time: {time.time()-start_time:.1f}s", flush=True)

(bx0,bx1),(by0,by1),(bz0,bz1) = frames[0]["bounds"]
Lx, Ly, Lz = bx1-bx0, by1-by0, bz1-bz0

sio2_ids = [aid for aid, d in frames[0]["atoms"].items() if d[0] in SIO2_TYPES]
n_sio2 = len(sio2_ids)

# =============================================================================
# SiO2 COM
# =============================================================================
print("\n[3/6] Calculating SiO2 COM and assigning bonds to layers...", flush=True)
sio2_com = np.array([0.0, 0.0, 0.0])
for aid in sio2_ids:
    _, x, y, z, ix, iy, iz = frames[0]["atoms"][aid]
    sio2_com += np.array([x + ix*Lx, y + iy*Ly, z + iz*Lz])
sio2_com /= n_sio2

bond_layers = {}
for i, (a1, a2) in enumerate(bonds):
    if a1 not in frames[0]["atoms"] or a2 not in frames[0]["atoms"]:
        continue
    _, x1, y1, z1, ix1, iy1, iz1 = frames[0]["atoms"][a1]
    _, x2, y2, z2, ix2, iy2, iz2 = frames[0]["atoms"][a2]
    
    mx = (x1+ix1*Lx + x2+ix2*Lx)/2 - sio2_com[0]
    my = (y1+iy1*Ly + y2+iy2*Ly)/2 - sio2_com[1]
    mz = (z1+iz1*Lz + z2+iz2*Lz)/2 - sio2_com[2]
    r = np.sqrt(mx*mx + my*my + mz*mz)
    
    for name, (lo, hi) in LAYERS.items():
        if lo <= r < hi:
            bond_layers[i] = name
            break

n_contact = sum(1 for v in bond_layers.values() if v=='contact')
n_interface = sum(1 for v in bond_layers.values() if v=='interface')
n_bulk = sum(1 for v in bond_layers.values() if v=='bulk')
print(f"  Contact: {n_contact}, Interface: {n_interface}, Bulk: {n_bulk}")
print(f"  Time: {time.time()-start_time:.1f}s", flush=True)

# =============================================================================
# Calculate bond vectors
# =============================================================================
print("\n[4/6] Computing bond vectors...", flush=True)
bond_vectors_by_frame = []

for fi, fr in enumerate(frames):
    atoms = fr["atoms"]
    cx = sum(atoms[a][1]+atoms[a][4]*Lx for a in sio2_ids)/n_sio2
    cy = sum(atoms[a][2]+atoms[a][5]*Ly for a in sio2_ids)/n_sio2
    cz = sum(atoms[a][3]+atoms[a][6]*Lz for a in sio2_ids)/n_sio2
    
    frame_vectors = []
    for i, (a1, a2) in enumerate(bonds):
        if a1 not in atoms or a2 not in atoms:
            frame_vectors.append(None)
            continue
        _, x1, y1, z1, ix1, iy1, iz1 = atoms[a1]
        _, x2, y2, z2, ix2, iy2, iz2 = atoms[a2]
        
        vx = (x2+ix2*Lx) - (x1+ix1*Lx)
        vy = (y2+iy2*Ly) - (y1+iy1*Ly)
        vz = (z2+iz2*Lz) - (z1+iz1*Lz)
        
        if vx > Lx/2: vx -= Lx
        if vx < -Lx/2: vx += Lx
        if vy > Ly/2: vy -= Ly
        if vy < -Ly/2: vy += Ly
        if vz > Lz/2: vz -= Lz
        if vz < -Lz/2: vz += Lz
        
        norm = np.sqrt(vx*vx + vy*vy + vz*vz)
        if norm > 0:
            frame_vectors.append((vx/norm, vy/norm, vz/norm))
        else:
            frame_vectors.append(None)
    
    bond_vectors_by_frame.append(frame_vectors)
    
    if (fi+1) % 50 == 0:
        pct = (fi+1) * 100 // len(frames)
        print(f"  Progress: {fi+1}/{len(frames)} ({pct}%), Time: {time.time()-start_time:.1f}s", flush=True)

print(f"  Done! Time: {time.time()-start_time:.1f}s", flush=True)

# =============================================================================
# Calculate C2
# =============================================================================
print("\n[5/6] Computing C2(t)...", flush=True)
n_frames = len(frames)
max_dt = n_frames - 1

C2_accum = {name: np.zeros(max_dt+1) for name in LAYERS}
count = {name: np.zeros(max_dt+1, dtype=int) for name in LAYERS}

for t0 in range(n_frames):
    vec0 = bond_vectors_by_frame[t0]
    for dt in range(1, n_frames - t0):
        vec1 = bond_vectors_by_frame[t0 + dt]
        for i in bond_layers.keys():
            if vec0[i] is None or vec1[i] is None:
                continue
            layer = bond_layers[i]
            dot = vec0[i][0]*vec1[i][0] + vec0[i][1]*vec1[i][1] + vec0[i][2]*vec1[i][2]
            P2 = (3*dot*dot - 1) / 2
            C2_accum[layer][dt] += P2
            count[layer][dt] += 1
    
    if (t0+1) % 50 == 0:
        pct = (t0+1) * 100 // n_frames
        print(f"  Progress: t0={t0+1}/{n_frames} ({pct}%), Time: {time.time()-start_time:.1f}s", flush=True)

C2 = {}
for name in LAYERS:
    valid = count[name] > 0
    C2[name] = np.zeros(max_dt+1)
    C2[name][valid] = C2_accum[name][valid] / count[name][valid]

print(f"  Done! Time: {time.time()-start_time:.1f}s", flush=True)

# =============================================================================
# Save and plot
# =============================================================================
print("\n[6/6] Saving results...", flush=True)

# Save CSV
csv_path = os.path.join(OUT_DIR, "orientation_acf.csv")
with open(csv_path, 'w') as f:
    f.write("dt_ps")
    for name in LAYERS:
        f.write(f",{name}_C2")
    f.write("\n")
    for dt in range(1, max_dt+1):
        f.write(str(dt))
        for name in LAYERS:
            f.write(f",{C2[name][dt]:.6f}")
        f.write("\n")
print(f"  CSV: {csv_path}")

# Plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
colors = {"contact": "#E63946", "interface": "#2A9D8F", "bulk": "#457B9D"}
dt_arr = np.arange(1, max_dt+1)

for name in LAYERS:
    ax.plot(dt_arr, C2[name][1:], color=colors[name], label=name, lw=2)

ax.set_xlabel(r"$\Delta t$ (ps)", fontsize=12)
ax.set_ylabel(r"$C_2(\Delta t)$", fontsize=12)
ax.set_title("Segmental Orientation ACF (FKM/SiO2, 500 ps NVT)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max_dt)
ax.set_ylim(-0.1, 1.0)

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "orientation_acf.png")
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Figure: {fig_path}")

# Summary
print()
print("=" * 60)
print("Results Summary")
print("=" * 60)
for name in LAYERS:
    c = C2[name]
    print(f"  {name:10s}: C2(1)={c[1]:.4f}, C2(50)={c[50]:.4f}, C2(100)={c[100]:.4f}")

print()
print(f"Total time: {time.time()-start_time:.1f}s")
print("Done!")