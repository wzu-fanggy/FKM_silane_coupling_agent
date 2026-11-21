# -*- coding: utf-8 -*-
"""
Dihedral Angle Statistics by Layer - NUMBA OPTIMIZED VERSION
KH550-grafted FKM/SiO2 System - Trans/Gauche Distribution
"""

import numpy as np
from collections import defaultdict
import time
from numba import njit, prange
import os

# ============================================================
# 设置 (KH550)
# ============================================================
R_SURF = 15.083
SIO2_TYPES = {6, 8, 9, 10, 11, 12, 13, 14}
DELTA_PERT = 8.292

DATA_FILE = r"F:\Lammps-MS\FKM-SIO2-TEST\KH550\Conformational Statistics\nvt_500ps.data"
TRAJ_FILE = r"F:\Lammps-MS\FKM-SIO2-TEST\KH550\Conformational Statistics\nvt_500ps_traj.dump"
OUT_FILE = r"F:\Lammps-MS\FKM-SIO2-TEST\KH550\Conformational Statistics\dihedral_stats_results_NUMBA.txt"
HISTOGRAM_FILE = r"F:\Lammps-MS\FKM-SIO2-TEST\KH550\Conformational Statistics\dihedral_histograms_NUMBA.txt"

# 二面角类型标签（根据data文件 Masses/连接 自动生成，KH550）
DIHEDRAL_LABELS = {
    1: "c-c-c-c", 2: "c-c-c-f", 3: "f-c-c-f", 4: "c-c-c-f", 5: "c-c-c-c",
    6: "f-c-c-h", 7: "c-c-c-h", 8: "c-c-c-h", 9: "c-c-c-h", 10: "f-c-c-h",
    11: "c-c-c-h", 12: "h-c-c-h", 13: "f-c-c-h", 14: "c-c-c-c", 15: "c-c-c-c",
    16: "c-c-c-c", 17: "c-c-c-f", 18: "c-c-c-f", 19: "c-c-c-f", 20: "c-c-c-h",
    21: "h-c-c-h", 22: "c-c-c-h", 23: "h-c-c-n", 24: "c-c-c-n", 25: "h-c-c-si",
    26: "c-c-c-si", 27: "h-c-c-h", 28: "c-c-n-h", 29: "h-c-n-h", 30: "c-c-o-si",
    31: "h-c-o-si", 32: "h-c-si-o", 33: "c-c-si-o", 34: "h-c-c-o", 35: "c-o-si-c",
    36: "c-o-si-o", 37: "h-o-si-o", 38: "si-o-si-o", 39: "si-o-si-c",
}

# ============================================================
# Numba 加速的二面角计算（逐个）
# ============================================================
@njit(cache=True, fastmath=True)
def compute_dihedral_numba(p1x, p1y, p1z, p2x, p2y, p2z,
                            p3x, p3y, p3z, p4x, p4y, p4z,
                            Lx, Ly, Lz):
    b1x = p2x - p1x
    b1y = p2y - p1y
    b1z = p2z - p1z

    b2x = p3x - p2x
    b2y = p3y - p2y
    b2z = p3z - p2z

    b3x = p4x - p3x
    b3y = p4y - p3y
    b3z = p4z - p3z

    b1x -= Lx * round(b1x / Lx)
    b1y -= Ly * round(b1y / Ly)
    b1z -= Lz * round(b1z / Lz)

    b2x -= Lx * round(b2x / Lx)
    b2y -= Ly * round(b2y / Ly)
    b2z -= Lz * round(b2z / Lz)

    b3x -= Lx * round(b3x / Lx)
    b3y -= Ly * round(b3y / Ly)
    b3z -= Lz * round(b3z / Lz)

    b2_norm = np.sqrt(b2x*b2x + b2y*b2y + b2z*b2z)
    if b2_norm < 1e-10:
        return 0.0
    b2x /= b2_norm
    b2y /= b2_norm
    b2z /= b2_norm

    n1x = b1y * b2z - b1z * b2y
    n1y = b1z * b2x - b1x * b2z
    n1z = b1x * b2y - b1y * b2x
    n1_norm = np.sqrt(n1x*n1x + n1y*n1y + n1z*n1z)
    if n1_norm < 1e-10:
        return 0.0
    n1x /= n1_norm; n1y /= n1_norm; n1z /= n1_norm

    n2x = b2y * b3z - b2z * b3y
    n2y = b2z * b3x - b2x * b3z
    n2z = b2x * b3y - b2y * b3x
    n2_norm = np.sqrt(n2x*n2x + n2y*n2y + n2z*n2z)
    if n2_norm < 1e-10:
        return 0.0
    n2x /= n2_norm; n2y /= n2_norm; n2z /= n2_norm

    m1x = n1y * b2z - n1z * b2y
    m1y = n1z * b2x - n1x * b2z
    m1z = n1x * b2y - n1y * b2x

    x = n1x*n2x + n1y*n2y + n1z*n2z
    y = m1x*n2x + m1y*n2y + m1z*n2z

    phi = np.degrees(np.arctan2(y, x))
    return phi


@njit(cache=True, fastmath=True, parallel=True)
def compute_all_dihedrals_numba(coords, dihedral_indices, Lx, Ly, Lz):
    n_dih = dihedral_indices.shape[0]
    results = np.empty(n_dih, dtype=np.float64)
    for i in prange(n_dih):
        a1 = dihedral_indices[i, 0]
        a2 = dihedral_indices[i, 1]
        a3 = dihedral_indices[i, 2]
        a4 = dihedral_indices[i, 3]
        results[i] = compute_dihedral_numba(
            coords[a1, 0], coords[a1, 1], coords[a1, 2],
            coords[a2, 0], coords[a2, 1], coords[a2, 2],
            coords[a3, 0], coords[a3, 1], coords[a3, 2],
            coords[a4, 0], coords[a4, 1], coords[a4, 2],
            Lx, Ly, Lz)
    return results


@njit(cache=True, fastmath=True)
def classify_dihedral_numba(phi):
    phi_abs = abs(phi)
    if phi_abs > 150.0:
        return 0  # trans
    elif phi_abs >= 30.0 and phi_abs <= 90.0:
        return 1  # gauche
    else:
        return 2  # cis


@njit(cache=True, fastmath=True)
def classify_all_dihedrals_numba(phis):
    n = phis.shape[0]
    results = np.empty(n, dtype=np.int32)
    for i in range(n):
        results[i] = classify_dihedral_numba(phis[i])
    return results


@njit(cache=True, fastmath=True)
def circular_mean_numba(angles_deg):
    n = angles_deg.shape[0]
    if n == 0:
        return 0.0
    sin_sum = 0.0
    cos_sum = 0.0
    for i in range(n):
        rad = np.radians(angles_deg[i])
        sin_sum += np.sin(rad)
        cos_sum += np.cos(rad)
    return np.degrees(np.arctan2(sin_sum / n, cos_sum / n))


@njit(cache=True, fastmath=True)
def circular_std_numba(angles_deg):
    n = angles_deg.shape[0]
    if n == 0:
        return 0.0
    sin_sum = 0.0
    cos_sum = 0.0
    for i in range(n):
        rad = np.radians(angles_deg[i])
        sin_sum += np.sin(rad)
        cos_sum += np.cos(rad)
    sin_mean = sin_sum / n
    cos_mean = cos_sum / n
    R = np.sqrt(sin_mean*sin_mean + cos_mean*cos_mean)
    if R < 1e-10:
        return 180.0
    return np.degrees(np.sqrt(-2.0 * np.log(R)))


@njit(cache=True, fastmath=True)
def build_histogram_numba(phis, bin_edges):
    n_bins = bin_edges.shape[0] - 1
    hist = np.zeros(n_bins, dtype=np.int64)
    n = phis.shape[0]
    for i in range(n):
        idx = int((phis[i] - bin_edges[0]) / (bin_edges[1] - bin_edges[0]))
        if 0 <= idx < n_bins:
            hist[idx] += 1
    return hist


def read_data_file(filename):
    """读取键、角、二面角的定义"""
    with open(filename) as f:
        lines = f.readlines()
    bonds = []
    angles = []
    dihedrals = []
    in_bonds = in_angles = in_dihedrals = False
    for line in lines:
        stripped = line.strip()
        if stripped == 'Bonds':
            in_bonds = True; in_angles = False; in_dihedrals = False
            continue
        if stripped == 'Angles':
            in_bonds = False; in_angles = True; in_dihedrals = False
            continue
        if 'Dihedrals' in stripped:
            in_bonds = False; in_angles = False; in_dihedrals = True
            continue
        if 'Impropers' in stripped:
            in_dihedrals = False
            break
        p = line.split()
        if len(p) >= 4:
            if in_bonds:
                bonds.append((int(p[0]), int(p[2]), int(p[3])))
            if in_angles and len(p) >= 5:
                angles.append((int(p[0]), int(p[2]), int(p[3]), int(p[4])))
            if in_dihedrals and len(p) >= 6:
                dihedrals.append((int(p[0]), int(p[2]), int(p[3]), int(p[4]), int(p[5])))
    return bonds, angles, dihedrals


def preload_trajectory(traj_file, atoms_per_frame=17830):
    """一次性读取整个轨迹文件到内存"""
    print(f"  Pre-loading trajectory...")
    t0 = time.time()
    with open(traj_file, 'r') as f:
        all_lines = f.readlines()
    lines_per_frame = atoms_per_frame + 9
    n_frames = len(all_lines) // lines_per_frame
    frames = []
    for fi in range(n_frames):
        start_line = fi * lines_per_frame
        try:
            x_line = all_lines[start_line + 5].split()
            y_line = all_lines[start_line + 6].split()
            z_line = all_lines[start_line + 7].split()
            Lx = float(x_line[1]) - float(x_line[0])
            Ly = float(y_line[1]) - float(y_line[0])
            Lz = float(z_line[1]) - float(z_line[0])
        except:
            Lx = Ly = Lz = 57.6
        coords = np.empty((atoms_per_frame, 3), dtype=np.float64)
        types = np.empty(atoms_per_frame, dtype=np.int32)
        ix = np.empty(atoms_per_frame, dtype=np.int32)
        iy = np.empty(atoms_per_frame, dtype=np.int32)
        iz = np.empty(atoms_per_frame, dtype=np.int32)
        atom_start = start_line + 9
        for j in range(atoms_per_frame):
            line_idx = atom_start + j
            if line_idx >= len(all_lines):
                break
            p = all_lines[line_idx].split()
            if len(p) >= 8:
                atom_id = int(p[0]) - 1
                if 0 <= atom_id < atoms_per_frame:
                    types[atom_id] = int(p[1])
                    coords[atom_id, 0] = float(p[2])
                    coords[atom_id, 1] = float(p[3])
                    coords[atom_id, 2] = float(p[4])
                    ix[atom_id] = int(p[5])
                    iy[atom_id] = int(p[6])
                    iz[atom_id] = int(p[7])
        frames.append((coords, types, ix, iy, iz, Lx, Ly, Lz))
    print(f"  Loaded {n_frames} frames in {time.time()-t0:.1f}s")
    return frames, n_frames


@njit(cache=True, fastmath=True)
def assign_layers_numba(coords, types, ix, iy, iz, Lx, Ly, Lz,
                        sio2_com_x, sio2_com_y, sio2_com_z,
                        r_surf, sio2_type_set):
    n_atoms = coords.shape[0]
    layers = np.empty(n_atoms, dtype=np.int32)
    for i in range(n_atoms):
        t = types[i]
        if t in sio2_type_set:
            layers[i] = -1
            continue
        x = coords[i, 0] + ix[i] * Lx
        y = coords[i, 1] + iy[i] * Ly
        z = coords[i, 2] + iz[i] * Lz
        dx = x - sio2_com_x
        dy = y - sio2_com_y
        dz = z - sio2_com_z
        r = np.sqrt(dx*dx + dy*dy + dz*dz)
        if r < r_surf:
            layers[i] = -1
        elif r < r_surf + 3.0:
            layers[i] = 0
        elif r < r_surf + DELTA_PERT:
            layers[i] = 1
        else:
            layers[i] = 2
    return layers


@njit(cache=True, fastmath=True)
def compute_sio2_com_numba(coords, types, ix, iy, iz, Lx, Ly, Lz, sio2_type_set):
    n_atoms = coords.shape[0]
    sum_x = sum_y = sum_z = 0.0
    count = 0
    for i in range(n_atoms):
        if types[i] in sio2_type_set:
            x = coords[i, 0] + ix[i] * Lx
            y = coords[i, 1] + iy[i] * Ly
            z = coords[i, 2] + iz[i] * Lz
            sum_x += x; sum_y += y; sum_z += z
            count += 1
    if count == 0:
        return 0.0, 0.0, 0.0
    return sum_x / count, sum_y / count, sum_z / count


print("=" * 60)
print("Dihedral Angle Statistics - NUMBA OPTIMIZED VERSION (KH550)")
print("=" * 60)

t0_total = time.time()

print("\n[1/6] Reading data file...")
t0 = time.time()
bonds_def, angles_def, dihedrals_def = read_data_file(DATA_FILE)
print(f"  Dihedrals: {len(dihedrals_def)}  (took {time.time()-t0:.2f}s)")

dihedral_types = np.array([d[0] for d in dihedrals_def], dtype=np.int32)
dihedral_indices = np.array([[d[1]-1, d[2]-1, d[3]-1, d[4]-1] for d in dihedrals_def], dtype=np.int32)

print("\n[2/6] Pre-loading trajectory...")
t0 = time.time()
frames, n_frames = preload_trajectory(TRAJ_FILE)
print(f"  Total frames: {n_frames}  (took {time.time()-t0:.2f}s)")

sio2_type_set = np.array(list(SIO2_TYPES), dtype=np.int32)
HIST_BINS = np.arange(-180, 181, 5, dtype=np.float64)

dihedral_values = defaultdict(list)
torsion_counts = defaultdict(lambda: defaultdict(int))
total_counts = defaultdict(lambda: defaultdict(int))
histograms = defaultdict(lambda: np.zeros(len(HIST_BINS) - 1, dtype=np.int64))

print("\n[3/6] Processing trajectory...")
start_time = time.time()

for fi in range(n_frames):
    if fi % 100 == 0:
        print(f"  Processing frame {fi}/{n_frames}...")
    coords, types, ix, iy, iz, Lx, Ly, Lz = frames[fi]
    sio2_com_x, sio2_com_y, sio2_com_z = compute_sio2_com_numba(
        coords, types, ix, iy, iz, Lx, Ly, Lz, sio2_type_set)
    layers = assign_layers_numba(
        coords, types, ix, iy, iz, Lx, Ly, Lz,
        sio2_com_x, sio2_com_y, sio2_com_z, R_SURF, sio2_type_set)
    phis = compute_all_dihedrals_numba(coords, dihedral_indices, Lx, Ly, Lz)
    categories = classify_all_dihedrals_numba(phis)
    n_dih = len(dihedrals_def)
    for i in range(n_dih):
        a2_idx = dihedral_indices[i, 1]
        layer = layers[a2_idx]
        if layer < 0:
            continue
        dt = dihedral_types[i]
        phi = phis[i]
        cat = categories[i]
        cat_name = ["trans", "gauche", "cis"][cat]
        dihedral_values[(dt, layer)].append(phi)
        torsion_counts[(dt, layer)][cat_name] += 1
        total_counts[layer][cat_name] += 1
        idx = int((phi - (-180)) / 5)
        if 0 <= idx < len(HIST_BINS) - 1:
            histograms[(dt, layer)][idx] += 1

print(f"\n  Processing complete in {time.time()-start_time:.1f}s")

print("\n[4/6] Computing circular statistics...")

layer_names = ["contact", "interface", "bulk"]
stats_by_type = {}
for key, vals in dihedral_values.items():
    arr = np.array(vals, dtype=np.float64)
    stats_by_type[key] = {
        'mean': circular_mean_numba(arr),
        'std': circular_std_numba(arr),
        'count': len(vals)}

print("\n[5/6] Saving statistics...")

with open(OUT_FILE, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("DIHEDRAL ANGLE STATISTICS BY LAYER (NUMBA OPTIMIZED)\n")
    f.write("=" * 70 + "\n")
    f.write(f"Frames analyzed: {n_frames}\n")
    f.write(f"R_SURF = {R_SURF}\n")
    f.write(f"Contact: {R_SURF:.1f}-{R_SURF + 3.0:.1f} A\n")
    f.write(f"Interface: {R_SURF + 3.0:.1f}-{R_SURF + DELTA_PERT:.2f} A\n")
    f.write(f"Bulk: >{R_SURF + DELTA_PERT:.2f} A\n\n")
    f.write("=" * 70 + "\n")
    f.write("OVERALL TRANS/GAUCHE RATIO\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'Layer':<15} {'Trans':<12} {'Gauche':<12} {'Cis':<12} {'Total':<12} {'Trans%':<10}\n")
    f.write("-" * 70 + "\n")
    for li, name in enumerate(layer_names):
        trans = total_counts[li]['trans']
        gauche = total_counts[li]['gauche']
        cis = total_counts[li]['cis']
        total = trans + gauche + cis
        trans_pct = trans / total * 100 if total > 0 else 0
        f.write(f"{name:<15} {trans:<12} {gauche:<12} {cis:<12} {total:<12} {trans_pct:<10.1f}%\n")
    f.write("\n")
    f.write("=" * 70 + "\n")
    f.write("DIHEDRAL TYPE STATISTICS (with Circular Mean/Std)\n")
    f.write("-" * 70 + "\n")
    for li, layer_name in enumerate(layer_names):
        f.write(f"\n--- {layer_name.upper()} ---\n")
        f.write(f"{'Type':<15} {'Count':<10} {'CircMean':<12} {'CircStd':<12} {'Trans%':<10}\n")
        f.write("-" * 70 + "\n")
        for dt in sorted(set(k[0] for k in dihedral_values.keys() if k[1] == li)):
            key = (dt, li)
            if key in stats_by_type:
                stats = stats_by_type[key]
                label = DIHEDRAL_LABELS.get(dt, str(dt))
                trans = torsion_counts[key]['trans']
                gauche = torsion_counts[key]['gauche']
                cis = torsion_counts[key]['cis']
                total = trans + gauche + cis
                trans_pct = trans / total * 100 if total > 0 else 0
                f.write(f"{label:<15} {stats['count']:<10} {stats['mean']:<12.2f} "
                        f"{stats['std']:<12.2f} {trans_pct:<10.1f}%\n")

print(f"  Statistics saved to: {OUT_FILE}")

print("\n[6/6] Saving histograms...")

with open(HISTOGRAM_FILE, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("DIHEDRAL ANGLE HISTOGRAMS (5-degree bins) - KH550\n")
    f.write("=" * 70 + "\n")
    f.write("Use this to verify trans/gauche/cis threshold settings\n\n")
    for li, layer_name in enumerate(layer_names):
        f.write(f"\n{'='*70}\n")
        f.write(f"LAYER: {layer_name.upper()}\n")
        f.write(f"{'='*70}\n")
        for dt in sorted(set(k[0] for k in histograms.keys() if k[1] == li)):
            key = (dt, li)
            label = DIHEDRAL_LABELS.get(dt, str(dt))
            hist = histograms[key]
            total = np.sum(hist)
            if total == 0:
                continue
            f.write(f"\n--- {label} (total={total}) ---\n")
            f.write(f"{'Angle(deg)':<15} {'Count':<10} {'Fraction%':<12}\n")
            f.write("-" * 40 + "\n")
            for i, count in enumerate(hist):
                if count > 0:
                    angle = HIST_BINS[i] + 2.5
                    frac = count / total * 100
                    f.write(f"{angle:<15.1f} {count:<10} {frac:<12.2f}\n")

print(f"  Histograms saved to: {HISTOGRAM_FILE}")

print("\n" + "=" * 60)
print("SUMMARY - Trans/Gauche Ratio by Layer (KH550)")
print("=" * 60)
for li, name in enumerate(layer_names):
    trans = total_counts[li]['trans']
    gauche = total_counts[li]['gauche']
    cis = total_counts[li]['cis']
    total = trans + gauche + cis
    if total > 0:
        print(f"  {name}: Trans={trans/total*100:.1f}%, Gauche={gauche/total*100:.1f}%, Cis={cis/total*100:.1f}%")
    else:
        print(f"  {name}: No data")

print(f"\nTotal time: {time.time()-t0_total:.1f}s")
print("Done!")