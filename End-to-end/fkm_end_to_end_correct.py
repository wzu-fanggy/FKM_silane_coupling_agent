#!/usr/bin/env python3
"""
FKM端到端距离计算脚本（修正版）
================================
计算原理：
1. FKM链有支化结构（含侧基），不能简单找"只连1个键"的原子作为端点
2. 主链由特定键类型构成：c-c(1), c-c2(2), c-c3(3), c1-c2(5)
3. 端到端距离 = 主链上最长路径两个端点的空间距离（考虑PBC）

用法:
    python fkm_end_to_end_correct.py <lammps_data_file> [options]

选项:
    --mol-start ID    FKM分子起始ID (默认: 2)
    --mol-end ID      FKM分子结束ID (默认: 121)
    --box L           模拟盒子边长 (默认: 57.6 Å)
    --output FILE     输出文件路径 (默认: fkm_ete_results.txt)
    --plot            绘制分布直方图
"""

import sys
import numpy as np
from collections import defaultdict, deque
import argparse

# 主链键类型定义
MAIN_CHAIN_BONDS = {1, 2, 3, 5}  # c-c, c-c2, c-c3, c1-c2


def parse_lammps_data(filename):
    """解析LAMMPS data文件"""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # 找Atoms部分
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Atoms # full'):
            start_idx = i + 1
            break

    end_idx = len(lines)
    for i in range(start_idx, len(lines)):
        if lines[i].strip().startswith('Bonds') or lines[i].strip().startswith('Velocities'):
            end_idx = i
            break

    atoms = []
    atom_lookup = {}
    for i in range(start_idx, end_idx):
        line = lines[i].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 7:
            a = {
                'atom_id': int(parts[0]),
                'mol_id': int(parts[1]),
                'atom_type': int(parts[2]),
                'q': float(parts[3]),
                'x': float(parts[4]),
                'y': float(parts[5]),
                'z': float(parts[6]),
            }
            atoms.append(a)
            atom_lookup[a['atom_id']] = a

    return atoms, atom_lookup


def parse_bonds(filename):
    """解析键信息"""
    with open(filename, 'r') as f:
        lines = f.readlines()

    start_idx = None
    end_idx = len(lines)
    for i, line in enumerate(lines):
        if line.strip().startswith('Bonds'):
            start_idx = i + 1
        elif start_idx is not None and line.strip().startswith('Angles'):
            end_idx = i
            break

    bonds = []
    for i in range(start_idx, end_idx):
        line = lines[i].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 4:
            bonds.append({
                'bond_id': int(parts[0]),
                'bond_type': int(parts[1]),
                'atom1': int(parts[2]),
                'atom2': int(parts[3])
            })
    return bonds


def bfs_farthest(adj, start):
    """BFS找到从start出发最远的节点和距离"""
    visited = {start: 0}
    queue = deque([start])
    farthest_node = start
    max_dist = 0

    while queue:
        node = queue.popleft()
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited[neighbor] = visited[node] + 1
                queue.append(neighbor)
                if visited[neighbor] > max_dist:
                    max_dist = visited[neighbor]
                    farthest_node = neighbor

    return farthest_node, max_dist


def min_image_vector(r1, r2, box):
    """计算考虑PBC的最小镜像向量"""
    dr = r2 - r1
    dr -= box * np.round(dr / box)
    return dr


def calculate_ete(data_file, mol_start=2, mol_end=121, box=57.6):
    """计算所有FKM链的端到端距离"""

    atoms, atom_lookup = parse_lammps_data(data_file)
    all_bonds = parse_bonds(data_file)

    # 按分子分组
    mol_atoms = defaultdict(list)
    for atom in atoms:
        mol_atoms[atom['mol_id']].append(atom)

    results = []

    for mol_id in range(mol_start, mol_end + 1):
        if mol_id not in mol_atoms:
            continue

        mol = mol_atoms[mol_id]
        atom_ids = [a['atom_id'] for a in mol]
        atom_id_set = set(atom_ids)

        # 构建主链邻接表（只包含主链键）
        main_adj = defaultdict(list)
        for bond in all_bonds:
            if bond['atom1'] in atom_id_set and bond['atom2'] in atom_id_set:
                if bond['bond_type'] in MAIN_CHAIN_BONDS:
                    main_adj[bond['atom1']].append(bond['atom2'])
                    main_adj[bond['atom2']].append(bond['atom1'])

        if not main_adj:
            continue

        # 两次BFS找到主链最长路径的两个端点
        start_node = list(main_adj.keys())[0]
        far_node1, _ = bfs_farthest(main_adj, start_node)
        far_node2, path_len = bfs_farthest(main_adj, far_node1)

        # 获取端点坐标
        a1 = atom_lookup[far_node1]
        a2 = atom_lookup[far_node2]
        r1 = np.array([a1['x'], a1['y'], a1['z']])
        r2 = np.array([a2['x'], a2['y'], a2['z']])

        # 考虑PBC计算距离
        dr = min_image_vector(r1, r2, box)
        ete = np.linalg.norm(dr)

        results.append({
            'mol_id': mol_id,
            'ete_distance': ete,
            'path_length': path_len,
            'n_main_chain': len(main_adj),
            'end1': far_node1,
            'end2': far_node2,
            'r1': r1,
            'r2': r2
        })

    return results


def print_results(results, box):
    """打印结果"""
    ete_distances = [r['ete_distance'] for r in results]
    ete_array = np.array(ete_distances)

    print("=" * 70)
    print("FKM端到端距离计算结果")
    print("=" * 70)
    print(f"
主链键类型: c-c(1), c-c2(2), c-c3(3), c1-c2(5)")
    print(f"盒子大小: {box} Å")
    print(f"
{'分子ID':>8} {'主链原子':>10} {'路径长度':>10} {'端到端距离(Å)':>16}")
    print("-" * 70)

    for r in results:
        print(f"{r['mol_id']:>8} {r['n_main_chain']:>10} {r['path_length']:>10} {r['ete_distance']:>16.4f}")

    print("
" + "=" * 70)
    print("统计摘要")
    print("=" * 70)
    print(f"  样本数:  {len(ete_array)}")
    print(f"  平均值:  {np.mean(ete_array):.4f} Å")
    print(f"  标准差:  {np.std(ete_array):.4f} Å")
    print(f"  最小值:  {np.min(ete_array):.4f} Å")
    print(f"  最大值:  {np.max(ete_array):.4f} Å")
    print(f"  中位数:  {np.median(ete_array):.4f} Å")
    print(f"  <R²>:    {np.mean(ete_array**2):.4f} Å²")
    print(f"  √<R²>:   {np.sqrt(np.mean(ete_array**2)):.4f} Å")


def save_results(results, output_file, box):
    """保存结果到文件"""
    ete_array = np.array([r['ete_distance'] for r in results])

    with open(output_file, 'w') as f:
        f.write("# FKM端到端距离计算结果\n")
        f.write("# 计算方法: 主链最长路径端点距离（考虑PBC）\n")
        f.write("# 主链键类型: c-c(1), c-c2(2), c-c3(3), c1-c2(5)\n")
        f.write(f"# 盒子大小: {box} Å\n")
        f.write("#\n")
        f.write(f"# 统计摘要:\n")
        f.write(f"#   平均值: {np.mean(ete_array):.4f} Å\n")
        f.write(f"#   标准差: {np.std(ete_array):.4f} Å\n")
        f.write(f"#   <R²>:   {np.mean(ete_array**2):.4f} Å²\n")
        f.write(f"#   √<R²>:  {np.sqrt(np.mean(ete_array**2)):.4f} Å\n")
        f.write("#\n")
        f.write("# mol_id  n_main  path_len  ete(Å)  end1_x  end1_y  end1_z  end2_x  end2_y  end2_z\n")

        for r in results:
            e1 = r['r1']
            e2 = r['r2']
            f.write(f"{r['mol_id']:8d} {r['n_main_chain']:8d} {r['path_length']:8d} "
                   f"{r['ete_distance']:10.4f} {e1[0]:10.4f} {e1[1]:10.4f} {e1[2]:10.4f} "
                   f"{e2[0]:10.4f} {e2[1]:10.4f} {e2[2]:10.4f}\n")

    print(f"\n结果已保存到: {output_file}")


def plot_results(results, output_file):
    """绘制分布图"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("警告: 未安装matplotlib，跳过绘图")
        return

    ete_distances = [r['ete_distance'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 直方图
    axes[0].hist(ete_distances, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(np.mean(ete_distances), color='red', linestyle='--', linewidth=2,
                    label=f'Mean = {np.mean(ete_distances):.2f} Å')
    axes[0].axvline(np.median(ete_distances), color='green', linestyle='--', linewidth=2,
                    label=f'Median = {np.median(ete_distances):.2f} Å')
    axes[0].set_xlabel('End-to-End Distance (Å)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('FKM End-to-End Distance Distribution', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # 链索引 vs 端到端距离
    mol_ids = [r['mol_id'] for r in results]
    axes[1].scatter(mol_ids, ete_distances, c='steelblue', alpha=0.7, s=50)
    axes[1].axhline(np.mean(ete_distances), color='red', linestyle='--', linewidth=2,
                    label=f'Mean = {np.mean(ete_distances):.2f} Å')
    axes[1].set_xlabel('Molecule ID', fontsize=12)
    axes[1].set_ylabel('End-to-End Distance (Å)', fontsize=12)
    axes[1].set_title('End-to-End Distance vs Chain Index', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plot_file = output_file.replace('.txt', '.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"分布图已保存到: {plot_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='计算FKM聚合物链的端到端距离')
    parser.add_argument('data_file', help='LAMMPS data文件路径')
    parser.add_argument('--mol-start', type=int, default=2, help='FKM分子起始ID (默认: 2)')
    parser.add_argument('--mol-end', type=int, default=121, help='FKM分子结束ID (默认: 121)')
    parser.add_argument('--box', type=float, default=57.6, help='模拟盒子边长 (默认: 57.6 Å)')
    parser.add_argument('--output', default='fkm_ete_results.txt', help='输出文件路径')
    parser.add_argument('--plot', action='store_true', help='绘制分布图')

    args = parser.parse_args()

    results = calculate_ete(args.data_file, mol_start=args.mol_start, mol_end=args.mol_end, box=args.box)
    print_results(results, args.box)
    save_results(results, args.output, args.box)

    if args.plot:
        plot_results(results, args.output)
