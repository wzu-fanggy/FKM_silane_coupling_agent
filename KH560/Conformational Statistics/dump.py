# -*- coding: utf-8 -*-
"""
Conformational Statistics - Bond Lengths & Angles by Layer (KH560)
Time Evolution - Fixed Version with Proper Layer Statistics
"""
import numpy as np
from collections import defaultdict
import time

R_SURF = 15.095
DELTA_PERT = 8.28
SIO2_TYPES = {5, 7, 9, 10, 11, 12, 13, 14, 15}
DATA_FILE = r"F:\Lammps-MS\FKM-SIO2-TEST\KH560\Conformational Statistics\nvt_500ps.data"
TRAJ_FILE = r"F:\Lammps-MS\FKM-SIO2-TEST\KH560\Conformational Statistics\nvt_500ps_traj.dump"
OUT_FILE = r"F:\Lammps-MS\FKM-SIO2-TEST\KH560\Conformational Statistics\conform_stats_summary_fixed.txt"
ATOMS_PER_FRAME = 17840

# 键类型标签（根据data文件 Masses/连接 自动生成）
BOND_LABELS = {
    1: "c-c", 2: "c-c", 3: "c-c", 4: "c-f", 5: "c-c", 6: "c-f",
    7: "c-h", 8: "c-c", 9: "c-c", 10: "c-h", 11: "c-o", 12: "c-si",
    13: "c-h", 14: "c-o", 15: "c-c", 16: "c-h", 17: "c-o", 18: "h-o",
    19: "o-si",
}

# 键角类型标签（根据data文件 Masses/连接 自动生成）
ANGLE_LABELS = {
    1: "c-c-c", 2: "c-c-c", 3: "c-c-f", 4: "c-c-c", 5: "c-c-c",
    6: "c-c-h", 7: "c-c-h", 8: "c-c-h", 9: "c-c-c", 10: "c-c-c",
    11: "c-c-f", 12: "c-c-f", 13: "c-c-h", 14: "c-c-c", 15: "c-c-h",
    16: "c-c-o", 17: "c-c-si", 18: "c-c-c", 19: "c-c-h", 20: "c-c-o",
    21: "c-o-c", 22: "c-si-o", 23: "c-c-f", 24: "c-o-si", 25: "c-c-h",
    26: "c-c-o", 27: "c-c-h", 28: "c-c-o", 29: "c-o-c", 30: "f-c-f",
    31: "f-c-f", 32: "f-c-h", 33: "h-o-si", 34: "h-c-h", 35: "h-c-o",
    36: "h-c-si", 37: "h-c-h", 38: "h-c-o", 39: "h-c-h", 40: "h-c-o",
    41: "o-si-o", 42: "si-o-si",
}

# ============================================================
def read_data_file(filename):
    """从data文件读取键和角的定义"""
    with open(filename) as f:
        lines = f.readlines()
    bonds = []
    angles = []
    in_bonds = in_angles = False
    for line in lines:
        if line.strip() == 'Bonds':
            in_bonds = True
            continue
        if line.strip() == 'Angles':
            in_bonds = False
            in_angles = True
            continue
        if 'Dihedrals' in line:
            break
        p = line.split()
        if in_bonds and len(p) >= 4:
            bonds.append((int(p[0]), int(p[2]), int(p[3])))
        if in_angles and len(p) >= 5:
            angles.append((int(p[0]), int(p[2]), int(p[3]), int(p[4])))
    return bonds, angles

# ============================================================
def read_frame_by_index(traj_file, frame_idx, atoms_per_frame=ATOMS_PER_FRAME, lines=None):
    """按帧索引读取原子数据（lines 为预加载的整文件行列表，避免每帧重复读盘）"""
    if lines is None:
        with open(traj_file) as f:
            lines = f.readlines()

    # 每帧的行数
    LINES_PER_FRAME = atoms_per_frame + 9  # header lines + atoms
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

    # 读取原子数据
    atoms = {}
    atom_start = start_line + 9  # 跳过9行header

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
def main():
    print("=" * 60)
    print("Conformational Statistics - Time Evolution (Fixed, KH560)")
    print("=" * 60)

    print("\n[1/4] Reading data file...")
    bonds_def, angles_def = read_data_file(DATA_FILE)
    print(f"  Bonds: {len(bonds_def)}, Angles: {len(angles_def)}")

    print("\n[2/4] Counting frames...")
    with open(TRAJ_FILE) as f:
        n_frames = f.read().count('ITEM: TIMESTEP')
    print(f"  Total frames: {n_frames}")

    print("\n[3/4] Analyzing trajectory...")
    start_time = time.time()

    # 统计字典
    bond_stats = defaultdict(list)
    angle_stats = defaultdict(list)

    # 调试：记录每层的原子数
    layer_atom_counts = {'contact': [], 'interface': [], 'bulk': []}

    with open(TRAJ_FILE) as f:
        all_lines = f.readlines()

    for fi in range(n_frames):
        atoms, Lx, Ly, Lz = read_frame_by_index(TRAJ_FILE, fi, ATOMS_PER_FRAME, all_lines)
        if not atoms:
            print(f"  Frame {fi}: FAILED to read")
            continue

        if fi % 50 == 0:
            print(f"  Processing frame {fi}/{n_frames}...")

        # 计算SiO2中心
        sio2_coords = [[atoms[a]['x'], atoms[a]['y'], atoms[a]['z']]
                        for a in atoms if atoms[a]['t'] in SIO2_TYPES]
        if not sio2_coords:
            continue
        sio2_com = np.mean(sio2_coords, axis=0)

        # 原子层分配（与Accelerate.py一致）
        atom_layer = {}
        layer_count = {0: 0, 1: 0, 2: 0, -1: 0}

        for a in atoms:
            if atoms[a]['t'] in SIO2_TYPES:
                atom_layer[a] = -1
                continue

            x = atoms[a]['x'] + atoms[a]['ix'] * Lx
            y = atoms[a]['y'] + atoms[a]['iy'] * Ly
            z = atoms[a]['z'] + atoms[a]['iz'] * Lz
            r = np.sqrt((x - sio2_com[0])**2 +
                         (y - sio2_com[1])**2 +
                         (z - sio2_com[2])**2)

            if r < R_SURF:
                atom_layer[a] = -1
                layer_count[-1] += 1
            elif r < R_SURF + 3.0:
                atom_layer[a] = 0  # contact
                layer_count[0] += 1
            elif r < R_SURF + DELTA_PERT:
                atom_layer[a] = 1  # interface
                layer_count[1] += 1
            else:
                atom_layer[a] = 2  # bulk
                layer_count[2] += 1

        layer_atom_counts['contact'].append(layer_count[0])
        layer_atom_counts['interface'].append(layer_count[1])
        layer_atom_counts['bulk'].append(layer_count[2])

        # 键长统计
        for bt, a1, a2 in bonds_def:
            layer = atom_layer.get(a2, -1)
            if layer < 0:
                continue

            x1 = atoms[a1]['x'] + atoms[a1]['ix'] * Lx
            y1 = atoms[a1]['y'] + atoms[a1]['iy'] * Ly
            z1 = atoms[a1]['z'] + atoms[a1]['iz'] * Lz
            x2 = atoms[a2]['x'] + atoms[a2]['ix'] * Lx
            y2 = atoms[a2]['y'] + atoms[a2]['iy'] * Ly
            z2 = atoms[a2]['z'] + atoms[a2]['iz'] * Lz

            dx = x2 - x1; dy = y2 - y1; dz = z2 - z1
            dx = dx - Lx * np.round(dx / Lx)
            dy = dy - Ly * np.round(dy / Ly)
            dz = dz - Lz * np.round(dz / Lz)
            length = np.sqrt(dx*dx + dy*dy + dz*dz)
            bond_stats[(bt, layer)].append(length)

        # 键角统计
        for at, a1, a2, a3 in angles_def:
            layer = atom_layer.get(a2, -1)
            if layer < 0:
                continue

            def vec(a):
                return np.array([
                    atoms[a]['x'] + atoms[a]['ix'] * Lx,
                    atoms[a]['y'] + atoms[a]['iy'] * Ly,
                    atoms[a]['z'] + atoms[a]['iz'] * Lz
                ])

            v1 = vec(a1) - vec(a2)
            v2 = vec(a3) - vec(a2)
            v1 = v1 - Lx * np.round(v1[0] / Lx)
            v1[1] = v1[1] - Ly * np.round(v1[1] / Ly)
            v1[2] = v1[2] - Lz * np.round(v1[2] / Lz)
            v2 = v2 - Lx * np.round(v2[0] / Lx)
            v2[1] = v2[1] - Ly * np.round(v2[1] / Ly)
            v2[2] = v2[2] - Lz * np.round(v2[2] / Lz)

            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_a = np.clip(cos_a, -1, 1)
            angle = np.degrees(np.arccos(cos_a))
            angle_stats[(at, layer)].append(angle)

    elapsed = time.time() - start_time
    print(f"\n  Processing complete in {elapsed:.1f}s")

    # 统计总结
    print(f"\n  Total bond entries: {sum(len(v) for v in bond_stats.values())}")
    print(f"  Total angle entries: {sum(len(v) for v in angle_stats.values())}")

    layer_names = ["contact", "interface", "bulk"]
    for li, name in enumerate(layer_names):
        n_bonds = sum(len(v) for k, v in bond_stats.items() if k[1] == li)
        n_angles = sum(len(v) for k, v in angle_stats.items() if k[1] == li)
        print(f"  {name}: {n_bonds} bonds, {n_angles} angles")

    # 保存结果
    print(f"\n[4/4] Saving results to {OUT_FILE}")

    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Conformational Statistics - Time Evolution Summary (Fixed, KH560)\n")
        f.write(f"Frames analyzed: {n_frames}\n")
        f.write(f"R_SURF = {R_SURF}\n")
        f.write(f"Contact: {R_SURF:.1f}-{R_SURF + 3.0:.1f} A\n")
        f.write(f"Interface: {R_SURF + 3.0:.1f}-{R_SURF + DELTA_PERT:.2f} A\n")
        f.write(f"Bulk: >{R_SURF + DELTA_PERT:.2f} A\n")
        f.write("=" * 70 + "\n\n")

        # 平均每层原子数
        f.write("Average Layer Atom Counts:\n")
        for name in layer_names:
            counts = layer_atom_counts[name]
            if counts:
                f.write(f"  {name}: {np.mean(counts):.0f} +/- {np.std(counts):.0f}\n")
        f.write("\n")

        # BOND STATS
        f.write("BOND LENGTH STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Bond Type':<15} {'Layer':<12} {'Count':<10} {'Mean(A)':<12} {'Std(A)':<12} {'Range(A)':<20}\n")
        f.write("-" * 70 + "\n")

        for li, layer_name in enumerate(layer_names):
            for bt in sorted(set(k[0] for k in bond_stats.keys() if k[1] == li)):
                key = (bt, li)
                if key in bond_stats and len(bond_stats[key]) > 0:
                    vals = bond_stats[key]
                    label = BOND_LABELS.get(bt, str(bt))
                    f.write(f"{label:<15} {layer_name:<12} {len(vals):<10} {np.mean(vals):<12.4f} {np.std(vals):<12.4f} {np.min(vals):.4f}-{np.max(vals):.4f}\n")

        # ANGLE STATS
        f.write("\nANGLE STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Angle Type':<15} {'Layer':<12} {'Count':<10} {'Mean(deg)':<12} {'Std(deg)':<12} {'Range(deg)':<20}\n")
        f.write("-" * 70 + "\n")

        for li, layer_name in enumerate(layer_names):
            for at in sorted(set(k[0] for k in angle_stats.keys() if k[1] == li)):
                key = (at, li)
                if key in angle_stats and len(angle_stats[key]) > 0:
                    vals = angle_stats[key]
                    label = ANGLE_LABELS.get(at, str(at))
                    f.write(f"{label:<15} {layer_name:<12} {len(vals):<10} {np.mean(vals):<12.2f} {np.std(vals):<12.2f} {np.min(vals):.2f}-{np.max(vals):.2f}\n")

    print(f"\nDone! Total time: {time.time()-start_time:.1f}s")

if __name__ == '__main__':
    main()