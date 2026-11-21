# -*- coding: utf-8 -*-
# Radius of Gyration (Rg) analysis - Vectorized version
# Optimized with NumPy broadcasting

import numpy as np
import os, sys, time

DATA_PATH = r"F:\Lammps-MS\FKM-SIO2-TEST\RG\nvt_500ps_gpu.data"
TRAJ_PATH = r"F:\Lammps-MS\FKM-SIO2-TEST\RG\nvt_500ps_gpu_traj.dump"
OUT_DIR   = r"F:\Lammps-MS\FKM-SIO2-TEST\RG"

R_SURF = 15.1
LAYERS = {
    "contact":   (R_SURF,            R_SURF + 3.0),
    "interface": (R_SURF + 3.0,    R_SURF + 8.6),
    "bulk":      (R_SURF + 8.6,      1e6),
}
SIO2_TYPES = {7, 8, 9}

print("=" * 60)
print("Rg Analysis - Vectorized Version")
print("=" * 60)
start_time = time.time()

# =============================================================================
# Parse data file
# =============================================================================
print("\n[1/5] Reading data file...", flush=True)
with open(DATA_PATH, 'r') as f:
    lines = f.readlines()

atoms_start = bonds_start = -1
for i, line in enumerate(lines):
    if 'Atoms # full' in line:
        atoms_start = i + 2
    if line.strip() == 'Bonds':
        bonds_start = i
        break

# Build atom info and molecule-atom mapping
atom_info = {}  # atom_id -> (mol_id, type)
mol_atoms = {}  # mol_id -> [atom_ids]

for line in lines[atoms_start:bonds_start]:
    parts = line.split()
    if len(parts) >= 3:
        try:
            atom_id = int(parts[0])
            mol_id = int(parts[1])
            atom_type = int(parts[2])
            atom_info[atom_id] = (mol_id, atom_type)
            if mol_id not in mol_atoms:
                mol_atoms[mol_id] = []
            mol_atoms[mol_id].append(atom_id)
        except:
            pass

molecules = list(mol_atoms.keys())
n_mols = len(molecules)
n_atoms_total = len(atom_info)
print(f"  {n_mols} molecules, {n_atoms_total} atoms")

# =============================================================================
# Parse trajectory - vectorized
# =============================================================================
print("\n[2/5] Reading trajectory (vectorized)...", flush=True)

# First pass: count frames and box size
with open(TRAJ_PATH, 'r') as f:
    content = f.read()

n_frames = content.count('ITEM: TIMESTEP')
print(f"  {n_frames} frames")

# Find box bounds
idx = content.find('ITEM: BOX BOUNDS')
bounds_section = content[idx:idx+500]
lines_bs = bounds_section.split('\n')[1:4]
bx = [float(x) for x in lines_bs[0].split()[:2]]
by = [float(y) for y in lines_bs[1].split()[:2]]
bz = [float(z) for z in lines_bs[2].split()[:2]]
Lx, Ly, Lz = bx[1]-bx[0], by[1]-by[0], bz[1]-bz[0]
print(f"  Box: {Lx:.2f} x {Ly:.2f} x {Lz:.2f}")

# Parse all atoms into arrays
print("  Parsing atom data...", flush=True)
atom_ids = np.array(list(atom_info.keys()), dtype=np.int32)
atom_types = np.array([atom_info[aid][1] for aid in atom_ids])

# Create atom_id -> index mapping
atom_id_to_idx = {aid: i for i, aid in enumerate(atom_ids)}
n_atoms = len(atom_ids)

# Pre-allocate coordinate arrays: (n_frames, n_atoms, 3)
coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float64)

# Parse trajectory frame by frame but vectorize atom extraction
print("  Loading coordinates...", flush=True)
frame_coords = []
frame_start = 0

for fi in range(n_frames):
    # Find this frame's section
    frame_marker = f"ITEM: TIMESTEP\n"
    next_frame_marker = None
    
    start_pos = content.find(frame_marker, frame_start)
    if fi < n_frames - 1:
        next_start = content.find(frame_marker, start_pos + 1)
        frame_text = content[start_pos:next_start]
    else:
        frame_text = content[start_pos:]
    
    # Parse atoms from this frame
    lines_frame = frame_text.split('\n')
    in_atoms = False
    for line in lines_frame:
        if line.startswith("ITEM: ATOMS"):
            in_atoms = True
            continue
        if in_atoms and line.strip():
            parts = line.split()
            if len(parts) >= 8:
                aid = int(parts[0])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                ix, iy, iz = int(parts[5]), int(parts[6]), int(parts[7])
                if aid in atom_id_to_idx:
                    idx_atom = atom_id_to_idx[aid]
                    coords[fi, idx_atom, 0] = x + ix * Lx
                    coords[fi, idx_atom, 1] = y + iy * Ly
                    coords[fi, idx_atom, 2] = z + iz * Lz
    
    frame_start = start_pos + 1
    if (fi+1) % 100 == 0:
        print(f"    {fi+1}/{n_frames} ({(fi+1)*100//n_frames}%)", flush=True)

print(f"  Loaded coordinates: {coords.shape}, Time: {time.time()-start_time:.1f}s", flush=True)

# =============================================================================
# SiO2 COM and molecule layers
# =============================================================================
print("\n[3/5] Calculating COM and layers...", flush=True)

sio2_mask = np.isin(atom_types, list(SIO2_TYPES))
sio2_coords = coords[:, sio2_mask, :]  # (n_frames, n_sio2, 3)
sio2_coms = sio2_coords.mean(axis=1)    # (n_frames, 3)

# Build molecule-atom index arrays for vectorized computation
mol_ids_arr = np.array(molecules)
n_mols = len(molecules)

# For each molecule, get atom indices and count
mol_atom_indices = []
mol_n_atoms = []
for mol_id in molecules:
    atom_ids_mol = mol_atoms[mol_id]
    indices = [atom_id_to_idx[aid] for aid in atom_ids_mol if aid in atom_id_to_idx]
    mol_atom_indices.extend(indices)
    mol_n_atoms.append(len(indices))

mol_atom_indices = np.array(mol_atom_indices, dtype=np.int32)
mol_n_atoms = np.array(mol_n_atoms, dtype=np.int32)

# Calculate molecule COM for frame 0
mol_coords_0 = coords[0, mol_atom_indices, :]  # (total_atoms_in_all_mols, 3)
mol_COM_0 = np.zeros((n_mols, 3))
start_idx = 0
for i, n in enumerate(mol_n_atoms):
    end_idx = start_idx + n
    mol_COM_0[i] = mol_coords_0[start_idx:end_idx].mean(axis=0)
    start_idx = end_idx

# Assign layers based on frame 0
mol_layers = np.full(n_mols, -1, dtype=np.int32)
r_mol = np.sqrt(np.sum((mol_COM_0 - sio2_coms[0])**2, axis=1))

for li, (name, (lo, hi)) in enumerate(LAYERS.items()):
    mask = (r_mol >= lo) & (r_mol < hi)
    mol_layers[mask] = li

n_per_layer = [np.sum(mol_layers == li) for li in range(3)]
print(f"  Contact: {n_per_layer[0]}, Interface: {n_per_layer[1]}, Bulk: {n_per_layer[2]}")
print(f"  Time: {time.time()-start_time:.1f}s", flush=True)

# =============================================================================
# Calculate Rg - fully vectorized
# =============================================================================
print("\n[4/5] Calculating Rg (vectorized)...", flush=True)

# For each frame, calculate Rg for all molecules at once
Rg_results = np.zeros((n_frames, n_mols))

# Create molecule atom index offsets
mol_start_idx = np.zeros(n_mols + 1, dtype=np.int32)
mol_start_idx[1:] = np.cumsum(mol_n_atoms)

for fi in range(n_frames):
    # All atom coords for this frame, relative to SiO2 COM
    rel_coords = coords[fi] - sio2_coms[fi]  # (n_atoms, 3)
    
    # For each molecule, calculate COM and Rg
    for mi in range(n_mols):
        start = mol_start_idx[mi]
        end = mol_start_idx[mi + 1]
        mol_atoms_idx = mol_atom_indices[start:end]
        
        mol_atoms_coords = rel_coords[mol_atoms_idx]  # (n_atoms_in_mol, 3)
        mol_com = mol_atoms_coords.mean(axis=0)       # (3,)
        
        # Rg^2 = mean(|r_i - COM|^2)
        dr = mol_atoms_coords - mol_com
        Rg_sq = np.mean(np.sum(dr**2, axis=1))
        Rg_results[fi, mi] = np.sqrt(Rg_sq)
    
    if (fi+1) % 100 == 0:
        print(f"    {fi+1}/{n_frames} ({(fi+1)*100//n_frames}%), Time: {time.time()-start_time:.1f}s", flush=True)

print(f"  Done, Time: {time.time()-start_time:.1f}s", flush=True)

# =============================================================================
# Save and plot
# =============================================================================
print("\n[5/5] Saving results...", flush=True)

# Calculate statistics by layer per frame
csv_path = os.path.join(OUT_DIR, "rg_by_frame.csv")
with open(csv_path, 'w') as f:
    f.write("frame,contact_Rg_mean,contact_Rg_std,interface_Rg_mean,interface_Rg_std,bulk_Rg_mean,bulk_Rg_std\n")
    
    for fi in range(n_frames):
        means = []
        stds = []
        for li in range(3):
            mask = mol_layers == li
            vals = Rg_results[fi, mask]
            if len(vals) > 0:
                means.append(f"{vals.mean():.4f}")
                stds.append(f"{vals.std():.4f}")
            else:
                means.append("0")
                stds.append("0")
        
        f.write(f"{fi},{means[0]},{stds[0]},{means[1]},{stds[1]},{means[2]},{stds[2]}\n")

print(f"  CSV: {csv_path}")

# Overall statistics
print("\n  Overall Rg statistics:")
Rg_all_by_layer = {li: [] for li in range(3)}
for fi in range(n_frames):
    for li in range(3):
        mask = mol_layers == li
        Rg_all_by_layer[li].extend(Rg_results[fi, mask])

layer_names = ["contact", "interface", "bulk"]
for li in range(3):
    vals = Rg_all_by_layer[li]
    if vals:
        print(f"    {layer_names[li]:10s}: Rg = {np.mean(vals):.4f} ± {np.std(vals):.4f} A (n={len(vals)})")

# Plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Rg vs time
ax = axes[0]
colors = ["#E63946", "#2A9D8F", "#457B9D"]
time_arr = np.arange(n_frames)

for li in range(3):
    mask = mol_layers == li
    Rg_mean = Rg_results[:, mask].mean(axis=1)
    Rg_std = Rg_results[:, mask].std(axis=1)
    ax.plot(time_arr, Rg_mean, label=layer_names[li], color=colors[li], lw=2)
    ax.fill_between(time_arr, Rg_mean - Rg_std, Rg_mean + Rg_std, alpha=0.2, color=colors[li])

ax.set_xlabel("Frame", fontsize=12)
ax.set_ylabel("Rg (Angstrom)", fontsize=12)
ax.set_title("Radius of Gyration vs Time", fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Rg distribution
ax2 = axes[1]
for li in range(3):
    ax2.hist(Rg_all_by_layer[li], bins=30, alpha=0.6, label=layer_names[li], color=colors[li])
ax2.set_xlabel("Rg (Angstrom)", fontsize=12)
ax2.set_ylabel("Count", fontsize=12)
ax2.set_title("Rg Distribution", fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "rg_analysis.png")
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Figure: {fig_path}")

print(f"\nTotal time: {time.time()-start_time:.1f}s")
print("Done!")