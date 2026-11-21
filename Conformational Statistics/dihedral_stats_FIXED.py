# -*- coding: utf-8 -*-
"""
Dihedral Angle Statistics by Layer - FIXED VERSION
FKM/SiO2 System - Trans/Gauche Distribution

修复内容：
1. PBC最小镜像修正（每个分量分别处理）
2. 二面角向量方向修正（标准化学定义）
3. 添加循环统计（处理二面角周期性）
4. 添加直方图输出（用于验证trans/gauche阈值）
"""

import numpy as np
from collections import defaultdict
import time

# ============================================================
# 设置
# ============================================================
R_SURF = 15.1
SIO2_TYPES = {7, 8, 9}

DATA_FILE = r"F:\Lammps-MS\FKM-SIO2-TEST\Conformational Statistics\nvt_500ps_gpu.data"
TRAJ_FILE = r"F:\Lammps-MS\FKM-SIO2-TEST\Conformational Statistics\nvt_500ps_gpu_traj.dump"
OUT_FILE = r"F:\Lammps-MS\FKM-SIO2-TEST\Conformational Statistics\dihedral_stats_results_FIXED.txt"
HISTOGRAM_FILE = r"F:\Lammps-MS\FKM-SIO2-TEST\Conformational Statistics\dihedral_histograms.txt"

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
        if line.strip() == 'Bonds':
            in_bonds = True; in_angles = False; in_dihedrals = False
            continue
        if line.strip() == 'Angles':
            in_bonds = False; in_angles = True; in_dihedrals = False
            continue
        if 'Dihedrals' in line:
            in_bonds = False; in_angles = False; in_dihedrals = True
            continue
        if 'Impropers' in line:
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
# 读取轨迹帧
# ============================================================
def read_frame(traj_file, frame_idx, atoms_per_frame=17550):
    """按帧索引读取原子数据"""
    with open(traj_file) as f:
        lines = f.readlines()

    LINES_PER_FRAME = atoms_per_frame + 9
    start_line = frame_idx * LINES_PER_FRAME

    if start_line >= len(lines):
        return None, 57.6, 57.6, 57.6

    # 解析box
    Lx = Ly = Lz = 57.6
    try:
        x_line = lines[start_line + 5].split()
        y_line = lines[start_line + 6].split()
        z_line = lines[start_line + 7].split()
        Lx = float(x_line[1]) - float(x_line[0])
        Ly = float(y_line[1]) - float(y_line[0])
        Lz = float(z_line[1]) - float(z_line[0])
    except:
        pass

    # 读取原子
    atoms = {}
    atom_start = start_line + 9

    for j in range(atoms_per_frame):
        line_idx = atom_start + j
        if line_idx >= len(lines):
            break
        p = lines[line_idx].split()
        if len(p) >= 8:
            try:
                atoms[int(p[0])] = {
                    't': int(p[1]),
                    'x': float(p[2]), 'y': float(p[3]), 'z': float(p[4]),
                    'ix': int(p[5]), 'iy': int(p[6]), 'iz': int(p[7])
                }
            except:
                pass

    return atoms, Lx, Ly, Lz

# ============================================================
# 修正后的二面角计算（标准化学定义）
# ============================================================
def compute_dihedral_fixed(a1, a2, a3, a4, atoms, Lx, Ly, Lz):
    """
    标准二面角计算（化学定义）

    对于链 a1-a2-a3-a4，计算围绕 a2-a3 键的二面角
    范围: [-180, 180] 度
    """
    def get_pos(a):
        return np.array([
            atoms[a]['x'] + atoms[a]['ix'] * Lx,
            atoms[a]['y'] + atoms[a]['iy'] * Ly,
            atoms[a]['z'] + atoms[a]['iz'] * Lz
        ])

    p1, p2, p3, p4 = get_pos(a1), get_pos(a2), get_pos(a3), get_pos(a4)

    # 沿链方向的向量（标准定义）
    b1 = p2 - p1  # a1 → a2
    b2 = p3 - p2  # a2 → a3
    b3 = p4 - p3  # a3 → a4

    # 最小镜像修正（每个分量分别用对应的box长度修正）
    for vec in [b1, b2, b3]:
        vec[0] -= Lx * round(vec[0] / Lx)
        vec[1] -= Ly * round(vec[1] / Ly)
        vec[2] -= Lz * round(vec[2] / Lz)

    # 归一化 b2
    b2_norm_val = np.linalg.norm(b2)
    if b2_norm_val < 1e-10:
        return 0.0
    b2_norm = b2 / b2_norm_val

    # 垂直于 b1-b2 平面的法向量
    n1 = np.cross(b1, b2)
    n1_norm = np.linalg.norm(n1)
    if n1_norm < 1e-10:
        return 0.0
    n1 /= n1_norm

    # 垂直于 b2-b3 平面的法向量
    n2 = np.cross(b2, b3)
    n2_norm = np.linalg.norm(n2)
    if n2_norm < 1e-10:
        return 0.0
    n2 /= n2_norm

    # 用 atan2 计算正确象限的二面角
    # m1 = n1 × b2_norm，在 n1-b2_norm 平面内垂直于 n1
    m1 = np.cross(n1, b2_norm)

    x = np.dot(n1, n2)   # cos(phi)
    y = np.dot(m1, n2)   # sin(phi)

    phi = np.degrees(np.arctan2(y, x))  # 范围 [-180, 180]

    return phi

# ============================================================
# 循环统计（处理二面角周期性）
# ============================================================
def circular_mean(angles_deg):
    """计算循环均值"""
    if len(angles_deg) == 0:
        return 0.0
    angles_rad = np.radians(angles_deg)
    return np.degrees(np.arctan2(np.mean(np.sin(angles_rad)),
                                  np.mean(np.cos(angles_rad))))

def circular_std(angles_deg):
    """计算循环标准差"""
    if len(angles_deg) == 0:
        return 0.0
    angles_rad = np.radians(angles_deg)
    sin_mean = np.mean(np.sin(angles_rad))
    cos_mean = np.mean(np.cos(angles_rad))
    R = np.sqrt(sin_mean**2 + cos_mean**2)
    if R < 1e-10:
        return 180.0
    return np.degrees(np.sqrt(-2 * np.log(R)))

# ============================================================
# 二面角分类
# ============================================================
def classify_dihedral(phi):
    """Trans/Gauche/Cis 分类（基于绝对值）"""
    phi_abs = abs(phi)
    if phi_abs > 150:
        return "trans"      # ~180度
    elif 30 <= phi_abs <= 90:
        return "gauche"     # ~60度
    else:
        return "cis"        # ~0度

# ============================================================
# 主程序
# ============================================================
print("=" * 60)
print("Dihedral Angle Statistics by Layer - FIXED VERSION")
print("=" * 60)

t0 = time.time()

# 读取data文件
print("\n[1/5] Reading data file...")
bonds_def, angles_def, dihedrals_def = read_data_file(DATA_FILE)
print(f"  Dihedrals: {len(dihedrals_def)}")

# 统计帧数
print("\n[2/5] Counting frames...")
with open(TRAJ_FILE) as f:
    n_frames = f.read().count('ITEM: TIMESTEP')
print(f"  Total frames: {n_frames}")

# 初始化统计
dihedral_values = defaultdict(list)  # {(type, layer): [values]}
torsion_counts = defaultdict(lambda: defaultdict(int))  # {(type, layer): {trans: x, ...}}
total_counts = defaultdict(lambda: defaultdict(int))

# 直方图数据（用于验证阈值）
HIST_BINS = np.arange(-180, 181, 5)  # 5度一个bin
histograms = defaultdict(lambda: np.zeros(len(HIST_BINS) - 1, dtype=int))

print("\n[3/5] Processing trajectory...")
start_time = time.time()

for fi in range(n_frames):
    atoms, Lx, Ly, Lz = read_frame(TRAJ_FILE, fi)
    if not atoms:
        print(f"  Frame {fi}: FAILED")
        continue

    if fi % 100 == 0:
        print(f"  Processing frame {fi}/{n_frames}...")

    # SiO2中心
    sio2_coords = [[atoms[a]['x'], atoms[a]['y'], atoms[a]['z']] 
                    for a in atoms if atoms[a]['t'] in SIO2_TYPES]
    if not sio2_coords:
        continue
    sio2_com = np.mean(sio2_coords, axis=0)

    # 原子层分配
    atom_layer = {}
    for a in atoms:
        if atoms[a]['t'] in SIO2_TYPES:
            atom_layer[a] = -1
            continue

        x = atoms[a]['x'] + atoms[a]['ix'] * Lx
        y = atoms[a]['y'] + atoms[a]['iy'] * Ly
        z = atoms[a]['z'] + atoms[a]['iz'] * Lz
        r = np.linalg.norm([x - sio2_com[0], y - sio2_com[1], z - sio2_com[2]])

        if r < R_SURF:
            atom_layer[a] = -1
        elif r < R_SURF + 3.0:
            atom_layer[a] = 0  # contact
        elif r < R_SURF + 8.6:
            atom_layer[a] = 1  # interface
        else:
            atom_layer[a] = 2  # bulk

    # 统计二面角
    for dt, a1, a2, a3, a4 in dihedrals_def:
        layer = atom_layer.get(a2, -1)
        if layer < 0:
            continue

        # 可选：严格检查所有4个原子是否在同一层
        # layer_a1 = atom_layer.get(a1, -1)
        # layer_a3 = atom_layer.get(a3, -1)
        # layer_a4 = atom_layer.get(a4, -1)
        # if layer_a1 != layer or layer_a3 != layer or layer_a4 != layer:
        #     continue

        phi = compute_dihedral_fixed(a1, a2, a3, a4, atoms, Lx, Ly, Lz)
        dihedral_values[(dt, layer)].append(phi)

        # 直方图
        hist_idx = np.digitize(phi, HIST_BINS) - 1
        if 0 <= hist_idx < len(HIST_BINS) - 1:
            histograms[(dt, layer)][hist_idx] += 1

        category = classify_dihedral(phi)
        torsion_counts[(dt, layer)][category] += 1
        total_counts[layer][category] += 1

print(f"\n  Processing complete in {time.time()-start_time:.1f}s")

# ============================================================
# 输出结果
# ============================================================
print("\n[4/5] Saving statistics...")

layer_names = ["contact", "interface", "bulk"]

with open(OUT_FILE, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("DIHEDRAL ANGLE STATISTICS BY LAYER (FIXED VERSION)\n")
    f.write("=" * 70 + "\n")
    f.write(f"Frames analyzed: {n_frames}\n")
    f.write(f"R_SURF = {R_SURF}\n")
    f.write(f"Contact: {R_SURF:.1f}-{R_SURF + 3.0:.1f} A\n")
    f.write(f"Interface: {R_SURF + 3.0:.1f}-{R_SURF + 8.6:.1f} A\n")
    f.write(f"Bulk: >{R_SURF + 8.6:.1f} A\n\n")

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
            if key in dihedral_values and len(dihedral_values[key]) > 0:
                vals = dihedral_values[key]
                label = DIHEDRAL_LABELS.get(dt, str(dt))

                trans = torsion_counts[key]['trans']
                gauche = torsion_counts[key]['gauche']
                cis = torsion_counts[key]['cis']
                total = trans + gauche + cis
                trans_pct = trans / total * 100 if total > 0 else 0

                c_mean = circular_mean(vals)
                c_std = circular_std(vals)

                f.write(f"{label:<15} {len(vals):<10} {c_mean:<12.2f} "
                       f"{c_std:<12.2f} {trans_pct:<10.1f}%\n")

print(f"  Statistics saved to: {OUT_FILE}")

# ============================================================
# 输出直方图数据（用于验证阈值）
# ============================================================
print("\n[5/5] Saving histograms...")

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
                    angle = HIST_BINS[i] + 2.5  # bin中心
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

print(f"\nTotal time: {time.time()-t0:.1f}s")
print("Done!")
