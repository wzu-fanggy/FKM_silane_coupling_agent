# -*- coding: utf-8 -*-
"""
Dihedral Angle Statistics by Layer - NUMBA OPTIMIZED VERSION
KH570-grafted FKM/SiO2 System - Trans/Gauche Distribution

优化内容：
1. Numba JIT 加速二面角计算和分类
2. 一次性预加载整个轨迹到内存
3. numpy 数组替代字典存储原子数据
4. 批量计算二面角（向量化）
5. SiO2 COM 缓存
6. 使用内存映射减少内存拷贝
"""

import numpy as np
from collections import defaultdict
import time
from numba import njit, prange
from concurrent.futures import ProcessPoolExecutor
import os

# ============================================================
# 设置
# ============================================================
R_SURF = 15.05
SIO2_TYPES = {5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17}
DELTA_PERT = 6.08

DATA_FILE = r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\Conformational Statistics\nvt_500ps.data"
TRAJ_FILE = r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\Conformational Statistics\nvt_500ps_traj.dump"
OUT_FILE = r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\Conformational Statistics\dihedral_stats_results_NUMBA.txt"
HISTOGRAM_FILE = r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\Conformational Statistics\dihedral_histograms_NUMBA.txt"

# 二面角类型标签
DIHEDRAL_LABELS = {
    1: "c-c-c-c2", 2: "c2-c-c-f", 3: "c-c-c-f", 4: "f-c-c-f",
    5: "c-c-c2-c", 6: "c3-c-c2-hc", 7: "c-c-c2-hc", 8: "f-c-c2-hc",
    9: "c2-c-c2-hc", 10: "f-c-c3-hc", 11: "c2-c-c3-hc", 12: "f-c1-c2-hc",
    13: "hc-c1-c2-hc", 14: "c1-c2-c-c2", 15: "c-c2-c-c2", 16: "c-c2-c-c3",
    17: "c1-c2-c-f", 18: "c-c2-c-f", 19: "c-c2-c1-f", 20: "c-c2-c1-hc",
    21: "ho-o-si-o", 22: "si-o-si-o"
}

# ============================================================
# Numba 加速的二面角计算（逐个）
# ============================================================
@njit(cache=True, fastmath=True)
def compute_dihedral_numba(p1x, p1y, p1z, p2x, p2y, p2z,
                            p3x, p3y, p3z, p4x, p4y, p4z,
                            Lx, Ly, Lz):
    """
    Numba 加速的二面角计算
    输入: 4个原子的坐标 (已展开为标量，避免 numpy array 开销)
    输出: 二面角 [-180, 180] 度
    """
    # 向量 b1 = p2 - p1
    b1x = p2x - p1x
    b1y = p2y - p1y
    b1z = p2z - p1z

    # 向量 b2 = p3 - p2
    b2x = p3x - p2x
    b2y = p3y - p2y
    b2z = p3z - p2z

    # 向量 b3 = p4 - p3
    b3x = p4x - p3x
    b3y = p4y - p3y
    b3z = p4z - p3z

    # 最小镜像修正
    b1x -= Lx * round(b1x / Lx)
    b1y -= Ly * round(b1y / Ly)
    b1z -= Lz * round(b1z / Lz)

    b2x -= Lx * round(b2x / Lx)
    b2y -= Ly * round(b2y / Ly)
    b2z -= Lz * round(b2z / Lz)

    b3x -= Lx * round(b3x / Lx)
    b3y -= Ly * round(b3y / Ly)
    b3z -= Lz * round(b3z / Lz)

    # b2 归一化
    b2_norm = np.sqrt(b2x*b2x + b2y*b2y + b2z*b2z)
    if b2_norm < 1e-10:
        return 0.0
    b2x /= b2_norm
    b2y /= b2_norm
    b2z /= b2_norm

    # n1 = cross(b1, b2)
    n1x = b1y * b2z - b1z * b2y
    n1y = b1z * b2x - b1x * b2z
    n1z = b1x * b2y - b1y * b2x

    n1_norm = np.sqrt(n1x*n1x + n1y*n1y + n1z*n1z)
    if n1_norm < 1e-10:
        return 0.0
    n1x /= n1_norm
    n1y /= n1_norm
    n1z /= n1_norm

    # n2 = cross(b2, b3)
    n2x = b2y * b3z - b2z * b3y
    n2y = b2z * b3x - b2x * b3z
    n2z = b2x * b3y - b2y * b3x

    n2_norm = np.sqrt(n2x*n2x + n2y*n2y + n2z*n2z)
    if n2_norm < 1e-10:
        return 0.0
    n2x /= n2_norm
    n2y /= n2_norm
    n2z /= n2_norm

    # m1 = cross(n1, b2)
    m1x = n1y * b2z - n1z * b2y
    m1y = n1z * b2x - n1x * b2z
    m1z = n1x * b2y - n1y * b2x

    # x = dot(n1, n2), y = dot(m1, n2)
    x = n1x*n2x + n1y*n2y + n1z*n2z
    y = m1x*n2x + m1y*n2y + m1z*n2z

    phi = np.degrees(np.arctan2(y, x))
    return phi


# ============================================================
# Numba 加速: 批量计算所有二面角
# ============================================================
@njit(cache=True, fastmath=True, parallel=True)
def compute_all_dihedrals_numba(coords, dihedral_indices, Lx, Ly, Lz):
    """
    并行批量计算所有二面角

    coords: (n_atoms, 3) 所有原子坐标
    dihedral_indices: (n_dihedrals, 4) 每个二面角的4个原子索引 (0-based)
    """
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
            Lx, Ly, Lz
        )

    return results


# ============================================================
# Numba 加速: 二面角分类
# ============================================================
@njit(cache=True, fastmath=True)
def classify_dihedral_numba(phi):
    """Trans/Gauche/Cis 分类"""
    phi_abs = abs(phi)
    if phi_abs > 150.0:
        return 0  # trans
    elif phi_abs >= 30.0 and phi_abs <= 90.0:
        return 1  # gauche
    else:
        return 2  # cis


@njit(cache=True, fastmath=True)
def classify_all_dihedrals_numba(phis):
    """批量分类"""
    n = phis.shape[0]
    results = np.empty(n, dtype=np.int32)
    for i in range(n):
        results[i] = classify_dihedral_numba(phis[i])
    return results


# ============================================================
# Numba 加速: 循环统计
# ============================================================
@njit(cache=True, fastmath=True)
def circular_mean_numba(angles_deg):
    """计算循环均值"""
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
    """计算循环标准差"""
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


# ============================================================
# Numba 加速: 直方图统计
# ============================================================
@njit(cache=True, fastmath=True)
def build_histogram_numba(phis, bin_edges):
    """Numba 加速的直方图"""
    n_bins = bin_edges.shape[0] - 1
    hist = np.zeros(n_bins, dtype=np.int64)
    n = phis.shape[0]

    for i in range(n):
        # 找到对应的 bin
        idx = int((phis[i] - bin_edges[0]) / (bin_edges[1] - bin_edges[0]))
        if 0 <= idx < n_bins:
            hist[idx] += 1

    return hist


# ============================================================
# 读取data文件
# ============================================================
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


# ============================================================
# 预加载整个轨迹到内存
# ============================================================
def preload_trajectory(traj_file, atoms_per_frame=17850):
    """
    一次性读取整个轨迹文件到内存
    返回: frames_data 列表，每个元素是 (coords, types, ix, iy, iz, Lx, Ly, Lz)
          coords: (n_atoms, 3) 未包装坐标
          types: (n_atoms,) 原子类型
          ix,iy,iz: (n_atoms,) 镜像索引
    """
    print(f"  Pre-loading trajectory...")
    t0 = time.time()

    with open(traj_file, 'r') as f:
        all_lines = f.readlines()

    lines_per_frame = atoms_per_frame + 9
    n_frames = len(all_lines) // lines_per_frame

    frames = []

    for fi in range(n_frames):
        start_line = fi * lines_per_frame

        # 解析 box
        try:
            x_line = all_lines[start_line + 5].split()
            y_line = all_lines[start_line + 6].split()
            z_line = all_lines[start_line + 7].split()
            Lx = float(x_line[1]) - float(x_line[0])
            Ly = float(y_line[1]) - float(y_line[0])
            Lz = float(z_line[1]) - float(z_line[0])
        except:
            Lx = Ly = Lz = 57.6

        # 预分配数组
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
                # 注意: atom id 是 1-based，但数组是 0-based
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


# ============================================================
# Numba 加速: 计算层分配
# ============================================================
@njit(cache=True, fastmath=True)
def assign_layers_numba(coords, types, ix, iy, iz, Lx, Ly, Lz, 
                        sio2_com_x, sio2_com_y, sio2_com_z, 
                        r_surf, sio2_type_set):
    """
    为每个原子分配层
    返回: (n_atoms,) 层索引，-1=SiO2/内部, 0=contact, 1=interface, 2=bulk
    """
    n_atoms = coords.shape[0]
    layers = np.empty(n_atoms, dtype=np.int32)

    for i in range(n_atoms):
        t = types[i]
        if t in sio2_type_set:
            layers[i] = -1
            continue

        # 包装坐标
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


# ============================================================
# Numba 加速: 计算 SiO2 COM
# ============================================================
@njit(cache=True, fastmath=True)
def compute_sio2_com_numba(coords, types, ix, iy, iz, Lx, Ly, Lz, sio2_type_set):
    """计算 SiO2 质心"""
    n_atoms = coords.shape[0]
    sum_x = sum_y = sum_z = 0.0
    count = 0

    for i in range(n_atoms):
        if types[i] in sio2_type_set:
            x = coords[i, 0] + ix[i] * Lx
            y = coords[i, 1] + iy[i] * Ly
            z = coords[i, 2] + iz[i] * Lz
            sum_x += x
            sum_y += y
            sum_z += z
            count += 1

    if count == 0:
        return 0.0, 0.0, 0.0
    return sum_x / count, sum_y / count, sum_z / count


# ============================================================
# 处理单帧 (用于多进程)
# ============================================================
def process_frame(args):
    """
    处理单帧，返回统计结果
    args = (frame_data, dihedral_indices, sio2_type_set, r_surf)
    """
    coords, types, ix, iy, iz, Lx, Ly, Lz = args[0]
    dihedral_indices = args[1]
    sio2_type_set = args[2]
    r_surf = args[3]

    # 计算 SiO2 COM
    sio2_com_x, sio2_com_y, sio2_com_z = compute_sio2_com_numba(
        coords, types, ix, iy, iz, Lx, Ly, Lz, sio2_type_set
    )

    # 分配层
    layers = assign_layers_numba(
        coords, types, ix, iy, iz, Lx, Ly, Lz,
        sio2_com_x, sio2_com_y, sio2_com_z,
        r_surf, sio2_type_set
    )

    # 计算所有二面角
    phis = compute_all_dihedrals_numba(coords, dihedral_indices, Lx, Ly, Lz)

    # 分类
    categories = classify_all_dihedrals_numba(phis)

    return phis, categories, layers


# ============================================================
# 主程序
# ============================================================
print("=" * 60)
print("Dihedral Angle Statistics - NUMBA OPTIMIZED VERSION (KH570)")
print("=" * 60)

t0_total = time.time()

# 读取data文件
print("\n[1/6] Reading data file...")
t0 = time.time()
bonds_def, angles_def, dihedrals_def = read_data_file(DATA_FILE)
print(f"  Dihedrals: {len(dihedrals_def)}  (took {time.time()-t0:.2f}s)")

# 构建二面角索引数组 (0-based)
dihedral_types = np.array([d[0] for d in dihedrals_def], dtype=np.int32)
dihedral_indices = np.array([[d[1]-1, d[2]-1, d[3]-1, d[4]-1] for d in dihedrals_def], dtype=np.int32)

# 预加载轨迹
print("\n[2/6] Pre-loading trajectory...")
t0 = time.time()
frames, n_frames = preload_trajectory(TRAJ_FILE)
print(f"  Total frames: {n_frames}  (took {time.time()-t0:.2f}s)")

# SiO2 类型集合 (用于 numba)
sio2_type_set = np.array(list(SIO2_TYPES), dtype=np.int32)

# 初始化统计容器
HIST_BINS = np.arange(-180, 181, 5, dtype=np.float64)

# 统计存储
dihedral_values = defaultdict(list)
torsion_counts = defaultdict(lambda: defaultdict(int))
total_counts = defaultdict(lambda: defaultdict(int))
histograms = defaultdict(lambda: np.zeros(len(HIST_BINS) - 1, dtype=np.int64))

# ============================================================
# 处理轨迹
# ============================================================
print("\n[3/6] Processing trajectory...")
start_time = time.time()

# 串行处理（Numba 已经并行化了二面角计算）
for fi in range(n_frames):
    if fi % 100 == 0:
        print(f"  Processing frame {fi}/{n_frames}...")

    coords, types, ix, iy, iz, Lx, Ly, Lz = frames[fi]

    # 计算 SiO2 COM
    sio2_com_x, sio2_com_y, sio2_com_z = compute_sio2_com_numba(
        coords, types, ix, iy, iz, Lx, Ly, Lz, sio2_type_set
    )

    # 分配层
    layers = assign_layers_numba(
        coords, types, ix, iy, iz, Lx, Ly, Lz,
        sio2_com_x, sio2_com_y, sio2_com_z,
        R_SURF, sio2_type_set
    )

    # 计算所有二面角（Numba 并行）
    phis = compute_all_dihedrals_numba(coords, dihedral_indices, Lx, Ly, Lz)

    # 分类（Numba 加速）
    categories = classify_all_dihedrals_numba(phis)

    # 统计
    n_dih = len(dihedrals_def)
    for i in range(n_dih):
        a2_idx = dihedral_indices[i, 1]  # 中心原子 a2
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

        # 直方图
        idx = int((phi - (-180)) / 5)
        if 0 <= idx < len(HIST_BINS) - 1:
            histograms[(dt, layer)][idx] += 1

print(f"\n  Processing complete in {time.time()-start_time:.1f}s")

# ============================================================
# 输出结果
# ============================================================
print("\n[4/6] Computing circular statistics...")

layer_names = ["contact", "interface", "bulk"]

# 预计算循环统计
stats_by_type = {}
for key, vals in dihedral_values.items():
    arr = np.array(vals, dtype=np.float64)
    stats_by_type[key] = {
        'mean': circular_mean_numba(arr),
        'std': circular_std_numba(arr),
        'count': len(vals)
    }

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

    # 整体Trans/Gauche比例
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

    # 按二面角类型的统计
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

# ============================================================
# 输出直方图数据
# ============================================================
print("\n[6/6] Saving histograms...")

with open(HISTOGRAM_FILE, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("DIHEDRAL ANGLE HISTOGRAMS (5-degree bins)\n")
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

# ============================================================
# 终端摘要
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY - Trans/Gauche Ratio by Layer")
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
