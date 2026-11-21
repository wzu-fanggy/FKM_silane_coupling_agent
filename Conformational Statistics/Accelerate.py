# -*- coding: utf-8 -*-
# Conformational Statistics - Vectorized (Fixed)
# For FKM/SiO2 system

import numpy as np
import os, sys, time

DATA_PATH = r"F:\Lammps-MS\FKM-SIO2-TEST\Conformational Statistics\nvt_500ps_gpu.data"
OUT_DIR   = r"F:\Lammps-MS\FKM-SIO2-TEST\Conformational Statistics"

R_SURF = 15.1
LAYERS = {
    "contact":   (R_SURF,            R_SURF + 3.0),
    "interface": (R_SURF + 3.0,    R_SURF + 8.6),
    "bulk":      (R_SURF + 8.6,      1e6),
}

SIO2_TYPES = {7, 8, 9}
FKM_TYPES  = {1, 2, 3, 4, 5, 6}

BOND_LABELS = {1:"c-c", 2:"c-c2", 3:"c-c3", 4:"c-f", 5:"c1-c2", 6:"c1-f", 7:"c1-hc", 8:"c2-hc", 9:"c3-hc", 10:"ho-o", 11:"o-si"}
ANGLE_LABELS = {1:"c-c-c", 2:"c-c-c2", 3:"c-c-f", 4:"c-c2-c", 5:"c-c2-c1", 6:"c-c2-hc", 7:"c-c3-hc", 8:"c1-c2-hc", 9:"c2-c-c2", 10:"c2-c-c3", 11:"c2-c-f", 12:"c2-c1-f", 13:"c2-c1-hc", 14:"c3-c-f", 15:"f-c-f", 16:"f-c1-f", 17:"f-c1-hc", 18:"hc-c2-hc", 19:"hc-c3-hc", 20:"ho-o-si", 21:"o-si-o", 22:"si-o-si"}

print("=" * 60)
print("Conformational Statistics - Vectorized (Fixed)")
print("=" * 60)
start_time = time.time()

# =============================================================================
# Parse data file
# =============================================================================
print("\n[1/6] Reading data file...", flush=True)

with open(DATA_PATH, 'r') as f:
    lines = f.readlines()

# Find section boundaries
atoms_start = bonds_start = angles_start = dihedrals_start = -1
for i, line in enumerate(lines):
    if 'Atoms # full' in line: atoms_start = i + 2
    if line.strip() == 'Bonds' and bonds_start < 0: bonds_start = i + 1
    if line.strip() == 'Angles' and angles_start < 0: angles_start = i + 1
    if line.strip() == 'Dihedrals':
        dihedrals_start = i + 1
        break

print(f"  Sections found")

# First pass: find max atom_id and count
max_atom_id = 0
atom_count = 0
for line in lines[atoms_start:bonds_start]:
    parts = line.split()
    if len(parts) >= 3:
        try:
            aid = int(parts[0])
            if aid > max_atom_id: max_atom_id = aid
            atom_count += 1
        except: pass

print(f"  {atom_count} atoms, max_id = {max_atom_id}")

# Allocate arrays (1-indexed, so size = max_atom_id + 1)
n_ids = max_atom_id + 1
atom_valid = np.zeros(n_ids, dtype=np.bool_)
atom_mol = np.zeros(n_ids, dtype=np.int32)
atom_type = np.zeros(n_ids, dtype=np.int32)
atom_x = np.zeros(n_ids, dtype=np.float64)
atom_y = np.zeros(n_ids, dtype=np.float64)
atom_z = np.zeros(n_ids, dtype=np.float64)
atom_ix = np.zeros(n_ids, dtype=np.int32)
atom_iy = np.zeros(n_ids, dtype=np.int32)
atom_iz = np.zeros(n_ids, dtype=np.int32)

print("  Parsing atoms...", flush=True)
mol_atoms = {}

for line in lines[atoms_start:bonds_start]:
    parts = line.split()
    if len(parts) >= 10:
        try:
            aid = int(parts[0])
            mid = int(parts[1])
            typ = int(parts[2])
            x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
            ix, iy, iz = int(parts[7]), int(parts[8]), int(parts[9])
            
            atom_valid[aid] = True
            atom_mol[aid] = mid
            atom_type[aid] = typ
            atom_x[aid] = x
            atom_y[aid] = y
            atom_z[aid] = z
            atom_ix[aid] = ix
            atom_iy[aid] = iy
            atom_iz[aid] = iz
            
            if mid not in mol_atoms: mol_atoms[mid] = []
            mol_atoms[mid].append(aid)
        except: pass

print(f"  Parsed {np.sum(atom_valid)} atoms, {len(mol_atoms)} molecules")

# Parse bonds
print("  Parsing bonds...", flush=True)
bond_types = []
bond_a1 = []
bond_a2 = []

for line in lines[bonds_start:angles_start]:
    parts = line.split()
    if len(parts) >= 4:
        try:
            bond_types.append(int(parts[1]))
            bond_a1.append(int(parts[2]))
            bond_a2.append(int(parts[3]))
        except: pass

bond_types = np.array(bond_types)
bond_a1 = np.array(bond_a1)
bond_a2 = np.array(bond_a2)
print(f"  {len(bond_types)} bonds")

# Parse angles
print("  Parsing angles...", flush=True)
angle_types = []
angle_a1 = []
angle_a2 = []
angle_a3 = []

for line in lines[angles_start:dihedrals_start]:
    parts = line.split()
    if len(parts) >= 5:
        try:
            angle_types.append(int(parts[1]))
            angle_a1.append(int(parts[2]))
            angle_a2.append(int(parts[3]))
            angle_a3.append(int(parts[4]))
        except: pass

angle_types = np.array(angle_types)
angle_a1 = np.array(angle_a1)
angle_a2 = np.array(angle_a2)
angle_a3 = np.array(angle_a3)
print(f"  {len(angle_types)} angles")

# =============================================================================
# SiO2 COM and layer assignment
# =============================================================================
print("\n[2/6] Calculating COM and layers...", flush=True)

Lx = Ly = Lz = 57.6

# SiO2 COM
sio2_mask = np.isin(atom_type, list(SIO2_TYPES)) & atom_valid
sio2_x = atom_x[sio2_mask] + atom_ix[sio2_mask] * Lx
sio2_y = atom_y[sio2_mask] + atom_iy[sio2_mask] * Ly
sio2_z = atom_z[sio2_mask] + atom_iz[sio2_mask] * Lz
sio2_com = np.array([sio2_x.mean(), sio2_y.mean(), sio2_z.mean()])
print(f"  SiO2 COM: {sio2_com}")
print(f"  SiO2 atoms: {np.sum(sio2_mask)}")

# Atom distances from SiO2 COM
atom_r = np.sqrt((atom_x - sio2_com[0])**2 + (atom_y - sio2_com[1])**2 + (atom_z - sio2_com[2])**2)

# Layer assignment
atom_layer = np.zeros(n_ids, dtype=np.int32) - 1  # -1 = not assigned
atom_layer[(atom_r >= R_SURF) & (atom_r < R_SURF + 3.0)] = 0  # contact
atom_layer[(atom_r >= R_SURF + 3.0) & (atom_r < R_SURF + 8.6)] = 1  # interface
atom_layer[atom_r >= R_SURF + 8.6] = 2  # bulk

n_per_layer = [np.sum(atom_layer == i) for i in range(3)]
print(f"  Contact: {n_per_layer[0]}, Interface: {n_per_layer[1]}, Bulk: {n_per_layer[2]}")

# =============================================================================
# Calculate bond lengths (vectorized)
# =============================================================================
print("\n[3/6] Calculating bond lengths...", flush=True)

x1 = atom_x[bond_a1] + atom_ix[bond_a1] * Lx
y1 = atom_y[bond_a1] + atom_iy[bond_a1] * Ly
z1 = atom_z[bond_a1] + atom_iz[bond_a1] * Lz

x2 = atom_x[bond_a2] + atom_ix[bond_a2] * Lx
y2 = atom_y[bond_a2] + atom_iy[bond_a2] * Ly
z2 = atom_z[bond_a2] + atom_iz[bond_a2] * Lz

dx = x2 - x1
dy = y2 - y1
dz = z2 - z1

# PBC
dx = dx - Lx * np.round(dx / Lx)
dy = dy - Ly * np.round(dy / Ly)
dz = dz - Lz * np.round(dz / Lz)

bond_lengths = np.sqrt(dx*dx + dy*dy + dz*dz)
bond_layer = atom_layer[bond_a2]

print(f"  {len(bond_lengths)} bond lengths calculated")

# =============================================================================
# Calculate bond angles (vectorized)
# =============================================================================
print("\n[4/6] Calculating bond angles...", flush=True)

x1 = atom_x[angle_a1] + atom_ix[angle_a1] * Lx
y1 = atom_y[angle_a1] + atom_iy[angle_a1] * Ly
z1 = atom_z[angle_a1] + atom_iz[angle_a1] * Lz

x2 = atom_x[angle_a2] + atom_ix[angle_a2] * Lx
y2 = atom_y[angle_a2] + atom_iy[angle_a2] * Ly
z2 = atom_z[angle_a2] + atom_iz[angle_a2] * Lz

x3 = atom_x[angle_a3] + atom_ix[angle_a3] * Lx
y3 = atom_y[angle_a3] + atom_iy[angle_a3] * Ly
z3 = atom_z[angle_a3] + atom_iz[angle_a3] * Lz

v1x = x1 - x2; v1y = y1 - y2; v1z = z1 - z2
v2x = x3 - x2; v2y = y3 - y2; v2z = z3 - z2

# PBC
v1x = v1x - Lx * np.round(v1x / Lx)
v1y = v1y - Ly * np.round(v1y / Ly)
v1z = v1z - Lz * np.round(v1z / Lz)
v2x = v2x - Lx * np.round(v2x / Lx)
v2y = v2y - Ly * np.round(v2y / Ly)
v2z = v2z - Lz * np.round(v2z / Lz)

norm1 = np.sqrt(v1x*v1x + v1y*v1y + v1z*v1z)
norm2 = np.sqrt(v2x*v2x + v2y*v2y + v2z*v2z)
cos_angle = (v1x*v2x + v1y*v2y + v1z*v2z) / (norm1 * norm2)
cos_angle = np.clip(cos_angle, -1, 1)
bond_angles = np.arccos(cos_angle) * 180 / np.pi

angle_layer = atom_layer[angle_a2]

print(f"  {len(bond_angles)} angles calculated")

# =============================================================================
# Statistics
# =============================================================================
print("\n[5/6] Computing statistics...", flush=True)

bond_stats = {}
for bt in np.unique(bond_types):
    for li in range(3):
        mask = (bond_types == bt) & (bond_layer == li)
        if np.sum(mask) > 0:
            vals = bond_lengths[mask]
            label = BOND_LABELS.get(bt, str(bt))
            layer_name = ["contact", "interface", "bulk"][li]
            key = f"{label}_{layer_name}"
            bond_stats[key] = (np.sum(mask), vals.mean(), vals.std(), vals.min(), vals.max())

angle_stats = {}
for at in np.unique(angle_types):
    for li in range(3):
        mask = (angle_types == at) & (angle_layer == li)
        if np.sum(mask) > 0:
            vals = bond_angles[mask]
            label = ANGLE_LABELS.get(at, str(at))
            layer_name = ["contact", "interface", "bulk"][li]
            key = f"{label}_{layer_name}"
            angle_stats[key] = (np.sum(mask), vals.mean(), vals.std(), vals.min(), vals.max())

# =============================================================================
# Save and plot
# =============================================================================
print("\n[6/6] Saving results...", flush=True)

csv_path = os.path.join(OUT_DIR, "conformational_stats.csv")
with open(csv_path, 'w') as f:
    f.write("type,layer,count,mean,std,min,max\n")
    for key, (count, mean, std, min_v, max_v) in sorted(bond_stats.items()):
        layer = key.split('_', 1)[1]
        f.write(f"bond_{key},{layer},{count},{mean:.4f},{std:.4f},{min_v:.4f},{max_v:.4f}\n")
    for key, (count, mean, std, min_v, max_v) in sorted(angle_stats.items()):
        layer = key.split('_', 1)[1]
        f.write(f"angle_{key},{layer},{count},{mean:.2f},{std:.2f},{min_v:.2f},{max_v:.2f}\n")

print(f"  CSV: {csv_path}")

# Print summary
print("\n  Bond length summary:")
for key, (count, mean, std, _, _) in sorted(bond_stats.items()):
    print(f"    {key}: {mean:.4f} ± {std:.4f} A (n={count})")

print("\n  Bond angle summary:")
for key, (count, mean, std, _, _) in sorted(angle_stats.items()):
    print(f"    {key}: {mean:.2f} ± {std:.2f} deg (n={count})")

# Plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
colors = ["#E63946", "#2A9D8F", "#457B9D"]
layer_names = ["contact", "interface", "bulk"]

# Bond lengths histogram
ax = axes[0, 0]
for bt in [1, 4]:
    label = BOND_LABELS.get(bt, str(bt))
    for li in range(3):
        mask = (bond_types == bt) & (bond_layer == li)
        if np.sum(mask) > 5:
            ax.hist(bond_lengths[mask], bins=30, alpha=0.5, label=f"{layer_names[li]}_{label}", color=colors[li])
ax.set_xlabel("Bond Length (A)")
ax.set_ylabel("Count")
ax.set_title("Bond Length Distribution")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Bond lengths comparison
ax2 = axes[0, 1]
plot_bonds = [1, 4, 7, 8]
x = np.arange(len(plot_bonds))
width = 0.25
for li in range(3):
    means = []
    for bt in plot_bonds:
        label = BOND_LABELS.get(bt, str(bt))
        key = f"{label}_{layer_names[li]}"
        if key in bond_stats:
            means.append(bond_stats[key][1])
        else:
            means.append(0)
    ax2.bar(x + li*width, means, width, label=layer_names[li], color=colors[li])
ax2.set_xticks(x + width)
ax2.set_xticklabels([BOND_LABELS.get(bt, str(bt)) for bt in plot_bonds])
ax2.set_ylabel("Mean Bond Length (A)")
ax2.set_title("Mean Bond Length by Layer")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Angles histogram
ax3 = axes[1, 0]
for at in [1, 15]:
    label = ANGLE_LABELS.get(at, str(at))
    for li in range(3):
        mask = (angle_types == at) & (angle_layer == li)
        if np.sum(mask) > 5:
            ax3.hist(bond_angles[mask], bins=30, alpha=0.5, label=f"{layer_names[li]}_{label}", color=colors[li])
ax3.set_xlabel("Bond Angle (deg)")
ax3.set_ylabel("Count")
ax3.set_title("Bond Angle Distribution")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Angles comparison
ax4 = axes[1, 1]
plot_angles = [1, 9, 15]
x = np.arange(len(plot_angles))
for li in range(3):
    means = []
    for at in plot_angles:
        label = ANGLE_LABELS.get(at, str(at))
        key = f"{label}_{layer_names[li]}"
        if key in angle_stats:
            means.append(angle_stats[key][1])
        else:
            means.append(0)
    ax4.bar(x + li*width, means, width, label=layer_names[li], color=colors[li])
ax4.set_xticks(x + width)
ax4.set_xticklabels([ANGLE_LABELS.get(at, str(at)) for at in plot_angles])
ax4.set_ylabel("Mean Bond Angle (deg)")
ax4.set_title("Mean Bond Angle by Layer")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "conformational_stats.png")
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Figure: {fig_path}")

print(f"\nTotal time: {time.time()-start_time:.1f}s")
print("Done!")