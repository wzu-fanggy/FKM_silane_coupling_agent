# -*- coding: utf-8 -*-
# Orientation Autocorrelation Function (C2) - Vectorized & Corrected
# Based on Polymers 2021

import numpy as np
import os, sys, time

DATA_PATH = r"F:\Lammps-MS\FKM-SIO2-TEST\cut-MSD\nvt_500ps_gpu.data"
TRAJ_PATH = r"F:\Lammps-MS\FKM-SIO2-TEST\cut-MSD\nvt_500ps_gpu_traj.dump"
OUT_DIR   = r"F:\Lammps-MS\FKM-SIO2-TEST\cut-MSD\Orientation-related"

R_SURF = 15.1
LAYERS = {
    "contact":   (R_SURF,            R_SURF + 3.0),
    "interface": (R_SURF + 3.0,    R_SURF + 8.6),
    "bulk":      (R_SURF + 8.6,      1e6),
}
SIO2_TYPES = {7, 8, 9}

print("=" * 60)
print("Orientation ACF - Vectorized")
print("=" * 60)
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
for line in content[atoms_match:bonds_match].split('\n'):
    parts = line.split()
    if len(parts) >= 7 and parts[0].isdigit():
        atom_mol[int(parts[0])] = int(parts[1])

bonds = []
for line in content[bonds_match:].split('\n'):
    parts = line.split()
    if len(parts) >= 4 and parts[0].isdigit():
        bonds.append((int(parts[2]), int(parts[3])))

n_bonds = len(bonds)
print(f"  {len(atom_mol)} atoms, {n_bonds} bonds")

# =============================================================================
# Parse trajectory - simpler method
# =============================================================================
print("\n[2/6] Reading trajectory...", flush=True)

frames = []
frame_buf = {}
reading_atoms = False
atoms_to_read = 0

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
            if frame_buf:
                frames.append(frame_buf)
            frame_buf = {"step": int(f.readline().strip())}
        elif line == "ITEM: NUMBER OF ATOMS":
            frame_buf["n"] = int(f.readline().strip())
        elif "BOX BOUNDS" in line:
            frame_buf["bounds"] = (
                tuple(map(float, f.readline().split())),
                tuple(map(float, f.readline().split())),
                tuple(map(float, f.readline().split()))
            )
        elif line.startswith("ITEM: ATOMS"):
            frame_buf["atoms"] = {}
            reading_atoms = True
            atoms_to_read = frame_buf.get("n", 0)

if frame_buf:
    frames.append(frame_buf)

n_frames = len(frames)
print(f"  Loaded {n_frames} frames, Time: {time.time()-start_time:.1f}s", flush=True)

(bx0,bx1),(by0,by1),(bz0,bz1) = frames[0]["bounds"]
Lx, Ly, Lz = bx1-bx0, by1-by0, bz1-bz0

sio2_ids = [aid for aid, d in frames[0]["atoms"].items() if d[0] in SIO2_TYPES]
n_sio2 = len(sio2_ids)

# =============================================================================
# SiO2 COM and bond layers
# =============================================================================
print("\n[3/6] Calculating COM and assigning layers...", flush=True)

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

n_c = sum(1 for v in bond_layers.values() if v=='contact')
n_i = sum(1 for v in bond_layers.values() if v=='interface')
n_b = sum(1 for v in bond_layers.values() if v=='bulk')
print(f"  Contact: {n_c}, Interface: {n_i}, Bulk: {n_b}")
print(f"  Time: {time.time()-start_time:.1f}s", flush=True)

# =============================================================================
# Extract bond vectors - precompute arrays
# =============================================================================
print("\n[4/6] Extracting bond vectors (vectorized)...", flush=True)

# Build atom index map
atom_ids_list = list(frames[0]["atoms"].keys())
atom_id_to_idx = {aid: i for i, aid in enumerate(atom_ids_list)}

# Precompute bond indices
valid_bond_indices = [i for i in bond_layers.keys()]
bond_a1_idx = np.array([atom_id_to_idx[bonds[i][0]] for i in valid_bond_indices])
bond_a2_idx = np.array([atom_id_to_idx[bonds[i][1]] for i in valid_bond_indices])
bond_layer_arr = np.array([{"contact": 0, "interface": 1, "bulk": 2}[bond_layers[i]] for i in valid_bond_indices])

# Extract all coordinates at once
print("  Building coordinate arrays...", flush=True)
coords = np.zeros((n_frames, len(atom_ids_list), 3))
sio2_coms = np.zeros((n_frames, 3))

for fi, fr in enumerate(frames):
    atoms = fr["atoms"]
    # SiO2 COM
    cx = sum(atoms[a][1]+atoms[a][4]*Lx for a in sio2_ids)/n_sio2
    cy = sum(atoms[a][2]+atoms[a][5]*Ly for a in sio2_ids)/n_sio2
    cz = sum(atoms[a][3]+atoms[a][6]*Lz for a in sio2_ids)/n_sio2
    sio2_coms[fi] = [cx, cy, cz]
    
    # All unwrapped coords
    for i, aid in enumerate(atom_ids_list):
        _, x, y, z, ix, iy, iz = atoms[aid]
        coords[fi, i, 0] = x + ix*Lx - cx
        coords[fi, i, 1] = y + iy*Ly - cy
        coords[fi, i, 2] = z + iz*Lz - cz
    
    if (fi+1) % 100 == 0:
        print(f"    Frame {fi+1}/{n_frames} ({(fi+1)*100//n_frames}%), Time: {time.time()-start_time:.1f}s", flush=True)

print(f"  Time: {time.time()-start_time:.1f}s", flush=True)

# =============================================================================
# Calculate C2 - vectorized
# =============================================================================
print("\n[5/6] Computing C2...", flush=True)

max_dt = n_frames - 1
n_valid = len(valid_bond_indices)

# Precompute bond vectors for all frames
print("  Computing bond vectors...", flush=True)
bond_vecs = np.zeros((n_frames, n_valid, 3))
for fi in range(n_frames):
    vx = coords[fi, bond_a2_idx, 0] - coords[fi, bond_a1_idx, 0]
    vy = coords[fi, bond_a2_idx, 1] - coords[fi, bond_a1_idx, 1]
    vz = coords[fi, bond_a2_idx, 2] - coords[fi, bond_a1_idx, 2]
    
    # PBC
    vx = vx - Lx * np.round(vx / Lx)
    vy = vy - Ly * np.round(vy / Ly)
    vz = vz - Lz * np.round(vz / Lz)
    
    # Normalize
    norm = np.sqrt(vx*vx + vy*vy + vz*vz)
    norm[norm == 0] = 1
    bond_vecs[fi, :, 0] = vx / norm
    bond_vecs[fi, :, 1] = vy / norm
    bond_vecs[fi, :, 2] = vz / norm

print("  Computing autocorrelation...", flush=True)

# C2 accumulation - vectorized over bonds
C2_accum = np.zeros((max_dt + 1, 3))
count = np.zeros((max_dt + 1, 3), dtype=np.int64)

for dt in range(1, max_dt + 1):
    # Vectorized dot product for this dt
    for t0 in range(n_frames - dt):
        v0 = bond_vecs[t0]
        v1 = bond_vecs[t0 + dt]
        
        # Dot product: (n_bonds, 3) * (n_bonds, 3) -> (n_bonds,)
        dot = np.sum(v0 * v1, axis=1)
        
        # P2 = (3*dot^2 - 1) / 2
        P2 = (3 * dot * dot - 1) / 2
        
        # Accumulate by layer
        for layer_idx in range(3):
            mask = bond_layer_arr == layer_idx
            C2_accum[dt, layer_idx] += np.sum(P2[mask])
            count[dt, layer_idx] += np.sum(mask)
    
    if dt % 50 == 0 or dt == max_dt:
        print(f"    dt={dt}/{max_dt} ({(dt)*100//max_dt}%), Time: {time.time()-start_time:.1f}s", flush=True)

# Average
C2 = {}
for li, name in enumerate(["contact", "interface", "bulk"]):
    valid = count[:, li] > 0
    C2[name] = np.zeros(max_dt + 1)
    C2[name][valid] = C2_accum[valid, li] / count[valid, li]

print(f"  Time: {time.time()-start_time:.1f}s", flush=True)

# =============================================================================
# Save results
# =============================================================================
print("\n[6/6] Saving results...", flush=True)

csv_path = os.path.join(OUT_DIR, "orientation_acf.csv")
with open(csv_path, 'w') as f:
    f.write("dt_ps,contact_C2,interface_C2,bulk_C2\n")
    for dt in range(1, max_dt+1):
        f.write(f"{dt},{C2['contact'][dt]:.6f},{C2['interface'][dt]:.6f},{C2['bulk'][dt]:.6f}\n")
print(f"  CSV: {csv_path}")

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

print()
print("=" * 60)
print("Results")
print("=" * 60)
for name in LAYERS:
    c = C2[name]
    print(f"  {name:10s}: C2(1)={c[1]:.4f}, C2(50)={c[50]:.4f}, C2(100)={c[100]:.4f}")

print(f"\nTotal time: {time.time()-start_time:.1f}s")
print("Done!")