# -*- coding: utf-8 -*-
# Conformational Statistics - Bond lengths and angles by layer
# For FKM/SiO2 system

import numpy as np
import os, sys, time, re

DATA_PATH = r"F:\Lammps-MS\FKM-SIO2-TEST\Conformational Statistics\nvt_500ps_gpu.data"
OUT_DIR   = r"F:\Lammps-MS\FKM-SIO2-TEST\Conformational Statistics"

R_SURF = 15.1
LAYERS = {
    "contact":   (R_SURF,            R_SURF + 3.0),
    "interface": (R_SURF + 3.0,    R_SURF + 8.6),
    "bulk":      (R_SURF + 8.6,      1e6),
}

# Atom types
SIO2_TYPES = {7, 8, 9}
FKM_TYPES  = {1, 2, 3, 4, 5, 6}

# Bond type labels
BOND_LABELS = {
    1: "c-c", 2: "c-c2", 3: "c-c3", 4: "c-f",
    5: "c1-c2", 6: "c1-f", 7: "c1-hc", 8: "c2-hc",
    9: "c3-hc", 10: "ho-o", 11: "o-si"
}

# Angle type labels
ANGLE_LABELS = {
    1: "c-c-c", 2: "c-c-c2", 3: "c-c-f", 4: "c-c2-c",
    5: "c-c2-c1", 6: "c-c2-hc", 7: "c-c3-hc", 8: "c-c2-hc",
    9: "c2-c-c2", 10: "c2-c-c3", 11: "c2-c-f", 12: "c2-c1-f",
    13: "c2-c1-hc", 14: "c3-c-f", 15: "f-c-f", 16: "f-c1-f",
    17: "f-c1-hc", 18: "hc-c2-hc", 19: "hc-c3-hc", 20: "ho-o-si",
    21: "o-si-o", 22: "si-o-si"
}

print("=" * 60)
print("Conformational Statistics - Bond Lengths & Angles")
print("=" * 60)
start_time = time.time()

# =============================================================================
# Parse data file
# =============================================================================
print("\n[1/6] Reading data file...", flush=True)
with open(DATA_PATH, 'r') as f:
    content = f.read()
    lines = content.split('\n')

# Find sections
atoms_start = bonds_start = angles_start = dihedrals_start = -1
for i, line in enumerate(lines):
    if 'Atoms # full' in line:
        atoms_start = i + 2
    if line.strip() == 'Bonds':
        bonds_start = i + 1
    if line.strip() == 'Angles':
        angles_start = i + 1
    if line.strip() == 'Dihedrals':
        dihedrals_start = i + 1
        break

print(f"  Atoms: line {atoms_start}")
print(f"  Bonds: line {bonds_start}")
print(f"  Angles: line {angles_start}")

# Parse atoms: atom_id -> (mol_id, type, x, y, z)
print("  Parsing atoms...", flush=True)
atoms = {}  # atom_id -> (mol_id, type, x, y, z, ix, iy, iz)
mol_atoms = {}  # mol_id -> [atom_ids]

for line in lines[atoms_start:bonds_start]:
    parts = line.split()
    if len(parts) >= 9:
        try:
            atom_id = int(parts[0])
            mol_id = int(parts[1])
            atom_type = int(parts[2])
            x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
            ix, iy, iz = int(parts[7]), int(parts[8]), int(parts[9])
            atoms[atom_id] = (mol_id, atom_type, x, y, z, ix, iy, iz)
            if mol_id not in mol_atoms:
                mol_atoms[mol_id] = []
            mol_atoms[mol_id].append(atom_id)
        except:
            pass

print(f"  {len(atoms)} atoms, {len(mol_atoms)} molecules")

# Parse bonds: bond_id -> (type, atom1, atom2)
print("  Parsing bonds...", flush=True)
bonds = []
for line in lines[bonds_start:angles_start]:
    parts = line.split()
    if len(parts) >= 4:
        try:
            bond_type = int(parts[1])
            a1 = int(parts[2])
            a2 = int(parts[3])
            bonds.append((bond_type, a1, a2))
        except:
            pass

print(f"  {len(bonds)} bonds")

# Parse angles: angle_id -> (type, atom1, atom2, atom3)
print("  Parsing angles...", flush=True)
angles = []
for line in lines[angles_start:dihedrals_start]:
    parts = line.split()
    if len(parts) >= 5:
        try:
            angle_type = int(parts[1])
            a1 = int(parts[2])
            a2 = int(parts[3])
            a3 = int(parts[4])
            angles.append((angle_type, a1, a2, a3))
        except:
            pass

print(f"  {len(angles)} angles")

# Get box size (assume same as previous)
with open(DATA_PATH, 'r') as f:
    box_content = f.read(1000)

Lx = Ly = Lz = 57.6  # Default, will recalculate if needed

# =============================================================================
# Calculate SiO2 COM
# =============================================================================
print("\n[2/6] Calculating SiO2 COM...", flush=True)

sio2_ids = [aid for aid, info in atoms.items() if info[1] in SIO2_TYPES]
n_sio2 = len(sio2_ids)

sum_x = sum_y = sum_z = 0.0
for aid in sio2_ids:
    mol_id, typ, x, y, z, ix, iy, iz = atoms[aid]
    sum_x += x + ix * Lx
    sum_y += y + iy * Ly
    sum_z += z + iz * Lz

sio2_com = np.array([sum_x/n_sio2, sum_y/n_sio2, sum_z/n_sio2])
print(f"  SiO2 COM: {sio2_com}")

# =============================================================================
# Assign atoms to layers
# =============================================================================
print("\n[3/6] Assigning atoms to layers...", flush=True)

atom_layers = {}  # atom_id -> layer_name

for aid, info in atoms.items():
    if info[1] in SIO2_TYPES:
        atom_layers[aid] = "sio2"
        continue
    
    mol_id, typ, x, y, z, ix, iy, iz = info
    dx = x + ix*Lx - sio2_com[0]
    dy = y + iy*Ly - sio2_com[1]
    dz = z + iz*Lz - sio2_com[2]
    r = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    for name, (lo, hi) in LAYERS.items():
        if lo <= r < hi:
            atom_layers[aid] = name
            break

# Count atoms per layer
n_per_layer = {"contact": 0, "interface": 0, "bulk": 0}
for layer in atom_layers.values():
    if layer in n_per_layer:
        n_per_layer[layer] += 1

print(f"  Contact: {n_per_layer['contact']}, Interface: {n_per_layer['interface']}, Bulk: {n_per_layer['bulk']}")

# =============================================================================
# Calculate bond lengths
# =============================================================================
print("\n[4/6] Calculating bond lengths...", flush=True)

bond_lengths = {name: {} for name in LAYERS}  # layer -> {bond_type -> [lengths]}

for bond_type, a1, a2 in bonds:
    if a1 not in atoms or a2 not in atoms:
        continue
    
    # Get layer from central atom (a2 is usually the central C)
    layer = atom_layers.get(a2)
    if layer not in LAYERS:
        continue
    
    # Get positions
    _, _, x1, y1, z1, ix1, iy1, iz1 = atoms[a1]
    _, _, x2, y2, z2, ix2, iy2, iz2 = atoms[a2]
    
    dx = (x1 + ix1*Lx) - (x2 + ix2*Lx)
    dy = (y1 + iy1*Ly) - (y2 + iy2*Ly)
    dz = (z1 + iz1*Lz) - (z2 + iz2*Lz)
    
    length = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    if bond_type not in bond_lengths[layer]:
        bond_lengths[layer][bond_type] = []
    bond_lengths[layer][bond_type].append(length)

# Print bond length statistics
print("\n  Bond length statistics:")
for name in LAYERS:
    print(f"\n  {name}:")
    for bt, lengths in sorted(bond_lengths[name].items()):
        if lengths:
            label = BOND_LABELS.get(bt, str(bt))
            print(f"    {label:10s}: {np.mean(lengths):.4f} ± {np.std(lengths):.4f} A (n={len(lengths)})")

# =============================================================================
# Calculate bond angles
# =============================================================================
print("\n[5/6] Calculating bond angles...", flush=True)

bond_angles = {name: {} for name in LAYERS}

for angle_type, a1, a2, a3 in angles:
    if a1 not in atoms or a2 not in atoms or a3 not in atoms:
        continue
    
    layer = atom_layers.get(a2)
    if layer not in LAYERS:
        continue
    
    # Get positions
    _, _, x1, y1, z1, ix1, iy1, iz1 = atoms[a1]
    _, _, x2, y2, z2, ix2, iy2, iz2 = atoms[a2]
    _, _, x3, y3, z3, ix3, iy3, iz3 = atoms[a3]
    
    # Vectors from central atom
    v1 = np.array([x1+ix1*Lx - (x2+ix2*Lx), y1+iy1*Ly - (y2+iy2*Ly), z1+iz1*Lz - (z2+iz2*Lz)])
    v2 = np.array([x3+ix3*Lx - (x2+ix2*Lx), y3+iy3*Ly - (y2+iy2*Ly), z3+iz3*Lz - (z2+iz2*Lz)])
    
    # Angle
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle) * 180 / np.pi
    
    if angle_type not in bond_angles[layer]:
        bond_angles[layer][angle_type] = []
    bond_angles[layer][angle_type].append(angle)

# Print angle statistics
print("\n  Bond angle statistics:")
for name in LAYERS:
    print(f"\n  {name}:")
    for at, angs in sorted(bond_angles[name].items()):
        if angs:
            label = ANGLE_LABELS.get(at, str(at))
            print(f"    {label:10s}: {np.mean(angs):.2f} ± {np.std(angs):.2f} deg (n={len(angs)})")

# =============================================================================
# Save and plot
# =============================================================================
print("\n[6/6] Saving and plotting...", flush=True)

# Save CSV
csv_path = os.path.join(OUT_DIR, "conformational_stats.csv")
with open(csv_path, 'w') as f:
    f.write("type,layer,count,mean,std,min,max\n")
    
    # Bond lengths
    f.write("\n# Bond Lengths\n")
    for layer in LAYERS:
        for bt, lengths in sorted(bond_lengths[layer].items()):
            if lengths:
                label = BOND_LABELS.get(bt, str(bt))
                f.write(f"bond_{label},{layer},{len(lengths)},{np.mean(lengths):.4f},{np.std(lengths):.4f},{min(lengths):.4f},{max(lengths):.4f}\n")
    
    # Bond angles
    f.write("\n# Bond Angles\n")
    for layer in LAYERS:
        for at, angs in sorted(bond_angles[layer].items()):
            if angs:
                label = ANGLE_LABELS.get(at, str(at))
                f.write(f"angle_{label},{layer},{len(angs)},{np.mean(angs):.2f},{np.std(angs):.2f},{min(angs):.2f},{max(angs):.2f}\n")

print(f"  CSV: {csv_path}")

# Plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Bond length distribution
ax = axes[0, 0]
colors = {"contact": "#E63946", "interface": "#2A9D8F", "bulk": "#457B9D"}

# Get common bond types across layers
common_bonds = set()
for layer in LAYERS:
    common_bonds.update(bond_lengths[layer].keys())

# Plot histogram for selected bond types
plot_bonds = [1, 4, 7, 8]  # c-c, c-f, c1-hc, c2-hc
for bt in plot_bonds:
    label = BOND_LABELS.get(bt, str(bt))
    for layer in LAYERS:
        if bt in bond_lengths[layer] and bond_lengths[layer][bt]:
            data = bond_lengths[layer][bt]
            ax.hist(data, bins=30, alpha=0.5, label=f"{layer}_{label}", color=colors[layer])

ax.set_xlabel("Bond Length (A)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Bond Length Distribution", fontsize=13)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: Bond length comparison by layer
ax2 = axes[0, 1]
x = np.arange(len(plot_bonds))
width = 0.25
for i, layer in enumerate(LAYERS):
    means = []
    for bt in plot_bonds:
        if bt in bond_lengths[layer] and bond_lengths[layer][bt]:
            means.append(np.mean(bond_lengths[layer][bt]))
        else:
            means.append(0)
    ax2.bar(x + i*width, means, width, label=layer, color=colors[layer])

ax2.set_xticks(x + width)
ax2.set_xticklabels([BOND_LABELS.get(bt, str(bt)) for bt in plot_bonds])
ax2.set_ylabel("Mean Bond Length (A)", fontsize=12)
ax2.set_title("Mean Bond Length by Layer", fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Bond angle distribution
ax3 = axes[1, 0]
plot_angles = [1, 3, 9, 15]  # c-c-c, c-c-f, c2-c-c2, f-c-f
for at in plot_angles:
    label = ANGLE_LABELS.get(at, str(at))
    for layer in LAYERS:
        if at in bond_angles[layer] and bond_angles[layer][at]:
            data = bond_angles[layer][at]
            ax3.hist(data, bins=30, alpha=0.5, label=f"{layer}_{label}", color=colors[layer])

ax3.set_xlabel("Bond Angle (deg)", fontsize=12)
ax3.set_ylabel("Count", fontsize=12)
ax3.set_title("Bond Angle Distribution", fontsize=13)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4: Bond angle comparison by layer
ax4 = axes[1, 1]
x = np.arange(len(plot_angles))
for i, layer in enumerate(LAYERS):
    means = []
    for at in plot_angles:
        if at in bond_angles[layer] and bond_angles[layer][at]:
            means.append(np.mean(bond_angles[layer][at]))
        else:
            means.append(0)
    ax4.bar(x + i*width, means, width, label=layer, color=colors[layer])

ax4.set_xticks(x + width)
ax4.set_xticklabels([ANGLE_LABELS.get(at, str(at)) for at in plot_angles], rotation=15)
ax4.set_ylabel("Mean Bond Angle (deg)", fontsize=12)
ax4.set_title("Mean Bond Angle by Layer", fontsize=13)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "conformational_stats.png")
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Figure: {fig_path}")

print(f"\nTotal time: {time.time()-start_time:.1f}s")
print("Done!")