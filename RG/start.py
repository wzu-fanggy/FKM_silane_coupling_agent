# -*- coding: utf-8 -*-
# Radius of Gyration (Rg) analysis by layer
# Based on Polymers 2021 - chain center of mass distance from SiO2

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
FKM_TYPES  = {1, 2, 3, 4, 5, 6}

print("=" * 60)
print("Radius of Gyration (Rg) Analysis")
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

# Build atom info: atom_id -> (mol_id, type)
atom_info = {}
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

# Get molecule info
molecules = list(mol_atoms.keys())
n_mols = len(molecules)
print(f"  {n_mols} molecules, {len(atom_info)} atoms")

# =============================================================================
# Parse trajectory
# =============================================================================
print("\n[2/5] Reading trajectory...", flush=True)
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
print(f"  {n_frames} frames, Time: {time.time()-start_time:.1f}s", flush=True)

(bx0,bx1),(by0,by1),(bz0,bz1) = frames[0]["bounds"]
Lx, Ly, Lz = bx1-bx0, by1-by0, bz1-bz0

# Get SiO2 atom IDs
sio2_ids = [aid for aid, d in frames[0]["atoms"].items() if d[0] in SIO2_TYPES]
n_sio2 = len(sio2_ids)
print(f"  SiO2 atoms: {n_sio2}")

# =============================================================================
# Calculate SiO2 COM and molecule layers
# =============================================================================
print("\n[3/5] Calculating COM and assigning layers...", flush=True)

# SiO2 COM for each frame
sio2_coms = []
for fr in frames:
    atoms = fr["atoms"]
    cx = sum(atoms[a][1]+atoms[a][4]*Lx for a in sio2_ids)/n_sio2
    cy = sum(atoms[a][2]+atoms[a][5]*Ly for a in sio2_ids)/n_sio2
    cz = sum(atoms[a][3]+atoms[a][6]*Lz for a in sio2_ids)/n_sio2
    sio2_coms.append((cx, cy, cz))

# Assign molecules to layers based on initial frame COM
sio2_com = sio2_coms[0]
mol_layers = {}
for mol_id in molecules:
    atom_ids = mol_atoms[mol_id]
    # Calculate molecule COM
    sum_x = sum_y = sum_z = 0.0
    count = 0
    for aid in atom_ids:
        if aid in frames[0]["atoms"]:
            _, x, y, z, ix, iy, iz = frames[0]["atoms"][aid]
            sum_x += x + ix*Lx
            sum_y += y + iy*Ly
            sum_z += z + iz*Lz
            count += 1
    if count > 0:
        cm_x = sum_x/count - sio2_com[0]
        cm_y = sum_y/count - sio2_com[1]
        cm_z = sum_z/count - sio2_com[2]
        r = np.sqrt(cm_x**2 + cm_y**2 + cm_z**2)
        
        for name, (lo, hi) in LAYERS.items():
            if lo <= r < hi:
                mol_layers[mol_id] = name
                break

n_per_layer = {
    "contact": sum(1 for v in mol_layers.values() if v=="contact"),
    "interface": sum(1 for v in mol_layers.values() if v=="interface"),
    "bulk": sum(1 for v in mol_layers.values() if v=="bulk")
}
print(f"  Molecules: contact={n_per_layer['contact']}, interface={n_per_layer['interface']}, bulk={n_per_layer['bulk']}")
print(f"  Time: {time.time()-start_time:.1f}s", flush=True)

# =============================================================================
# Calculate Rg for each frame and layer
# =============================================================================
print("\n[4/5] Calculating Rg...", flush=True)

Rg_by_frame = {name: [] for name in LAYERS}
mol_Rg_all = []  # (frame, mol_id, layer, Rg)

for fi, fr in enumerate(frames):
    atoms = fr["atoms"]
    sio2_com = sio2_coms[fi]
    
    for mol_id in molecules:
        atom_ids = mol_atoms[mol_id]
        layer = mol_layers.get(mol_id)
        if layer is None:
            continue
        
        # Molecule COM
        sum_x = sum_y = sum_z = 0.0
        count = 0
        for aid in atom_ids:
            if aid in atoms:
                _, x, y, z, ix, iy, iz = atoms[aid]
                sum_x += x + ix*Lx - sio2_com[0]
                sum_y += y + iy*Ly - sio2_com[1]
                sum_z += z + iz*Lz - sio2_com[2]
                count += 1
        
        if count < 2:
            continue
        cm = np.array([sum_x/count, sum_y/count, sum_z/count])
        
        # Rg^2 = <|r_i - cm|^2>
        sum_r2 = 0.0
        for aid in atom_ids:
            if aid in atoms:
                _, x, y, z, ix, iy, iz = atoms[aid]
                rx = x + ix*Lx - sio2_com[0] - cm[0]
                ry = y + iy*Ly - sio2_com[1] - cm[1]
                rz = z + iz*Lz - sio2_com[2] - cm[2]
                sum_r2 += rx**2 + ry**2 + rz**2
        
        Rg = np.sqrt(sum_r2 / count)
        Rg_by_frame[layer].append(Rg)
        mol_Rg_all.append((fi, mol_id, layer, Rg))
    
    if (fi+1) % 100 == 0:
        print(f"  Frame {fi+1}/{n_frames} ({(fi+1)*100//n_frames}%), Time: {time.time()-start_time:.1f}s", flush=True)

print(f"  Time: {time.time()-start_time:.1f}s", flush=True)

# =============================================================================
# Save and plot
# =============================================================================
print("\n[5/5] Saving results...", flush=True)

# Calculate statistics per frame
csv_path = os.path.join(OUT_DIR, "rg_by_frame.csv")
with open(csv_path, 'w') as f:
    f.write("frame,contact_Rg_mean,contact_Rg_std,interface_Rg_mean,interface_Rg_std,bulk_Rg_mean,bulk_Rg_std\n")
    
    for fi in range(n_frames):
        vals = {name: [] for name in LAYERS}
        for frame, mol_id, layer, Rg in mol_Rg_all:
            if frame == fi:
                vals[layer].append(Rg)
        
        means = {name: np.mean(v) if v else 0 for name, v in vals.items()}
        stds = {name: np.std(v) if len(v) > 1 else 0 for name, v in vals.items()}
        
        f.write(f"{fi},{means['contact']:.4f},{stds['contact']:.4f},{means['interface']:.4f},{stds['interface']:.4f},{means['bulk']:.4f},{stds['bulk']:.4f}\n")

print(f"  CSV: {csv_path}")

# Overall statistics
print("\n  Overall Rg statistics:")
for name in LAYERS:
    vals = Rg_by_frame[name]
    if vals:
        print(f"    {name:10s}: Rg = {np.mean(vals):.4f} ± {np.std(vals):.4f} A (n={len(vals)})")

# Plot Rg vs time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Rg vs time
ax = axes[0]
time_arr = np.arange(n_frames)
for name in LAYERS:
    ax.plot(time_arr, [np.mean([r for f, m, l, r in mol_Rg_all if f==fi and l==name]) for fi in range(n_frames)], 
            label=name, lw=2)
ax.set_xlabel("Frame", fontsize=12)
ax.set_ylabel("Rg (Angstrom)", fontsize=12)
ax.set_title("Radius of Gyration vs Time", fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Rg distribution (histogram)
ax2 = axes[1]
colors = {"contact": "#E63946", "interface": "#2A9D8F", "bulk": "#457B9D"}
for name in LAYERS:
    ax2.hist(Rg_by_frame[name], bins=30, alpha=0.6, label=name, color=colors[name])
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