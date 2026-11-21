#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Radial Mass Density Distribution (RMDD) for FKM-SiO2 Composite
FKM/SiO2复合材料径向质量密度分布计算
================================================================================

功能:
    1. 读取LAMMPS dump文件
    2. 计算SiO2核心质心(COM)
    3. 计算所有原子到SiO2 COM的径向质量密度分布
    4. 移动平均平滑曲线
    5. 按层统计: Core / Contact / Interface / Bulk
    6. 使用Numba加速核心计算 (可选, 无Numba时自动回退到纯Python)

作者: Auto-generated
日期: 2026-08-20
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

# ============================================================================
# 尝试导入Numba, 失败则使用纯Python回退
# ============================================================================

try:
    from numba import njit, prange
    HAS_NUMBA = True
    print("[INFO] Numba已加载, 将使用JIT加速")
except ImportError:
    HAS_NUMBA = False
    print("[WARNING] Numba未安装, 将使用纯Python计算 (较慢)")
    print("[TIP] 安装Numba: conda install numba 或 pip install numba")

    # 创建装饰器占位符
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper

    def prange(*args):
        return range(*args)


# ============================================================================
# 用户配置参数 (请根据实际体系修改)
# ============================================================================

# 输入文件路径 (Windows路径格式)
INPUT_FILE = r"F:\Lammps-MS\FKM-SIO2-TEST\cut-MSD\nvt_500ps_gpu_end.dump"

# 输出文件路径
OUTPUT_PLOT = r"F:\Lammps-MS\FKM-SIO2-TEST\cut-MSD\radial_mass_density.png"
OUTPUT_DATA = r"F:\Lammps-MS\FKM-SIO2-TEST\cut-MSD\radial_density_data.txt"

# 原子质量 (g/mol) —— 请根据实际力场参数修改!
# 当前设置基于典型FKM (Viton) + SiO2体系:
#   Type 1,5: C (碳, FKM主链)
#   Type 2,4,6: F (氟, FKM侧链)
#   Type 3: Cl (氯, 可能为交联剂)
#   Type 7,8: Si (硅, SiO2核心)
#   Type 9: O (氧, SiO2核心)
ATOM_MASSES = {
    1: 12.011,   # C
    2: 18.998,   # F
    3: 35.453,   # Cl
    4: 18.998,   # F
    5: 12.011,   # C
    6: 18.998,   # F
    7: 28.086,   # Si
    8: 28.086,   # Si
    9: 15.999,   # O
}

# SiO2核心原子类型 (用于计算质心)
SIO2_TYPES = {7, 8, 9}

# 分层边界定义 (Å) —— 从SiO2表面开始
R_SURF = 15.0  # SiO2表面半径 (从COM算起)
LAYER_BOUNDARIES = {
    'SiO2 Core':    (0.0,        R_SURF),          # 0.0 - 15.0
    'Contact':      (R_SURF,     R_SURF + 3.0),     # 15.0 - 18.0
    'Interface':    (R_SURF+3.0, R_SURF + 8.6),     # 18.0 - 23.6
    'Bulk':         (R_SURF+8.6, 999.0),            # > 23.6
}

# 计算参数
R_MAX = 55.0            # 最大径向距离 (Å)
DR = 0.5                # 分箱宽度 (Å)
SMOOTH_WINDOW = 5         # 移动平均窗口 (个bin)

# 图像参数
FIG_SIZE = (11, 7)
DPI = 300

# 单位转换: amu/Å^3 -> g/cm^3
DENSITY_CONVERSION = 1.66053906660


# ============================================================================
# Numba加速计算函数
# ============================================================================

@njit(cache=True, fastmath=True)
def _compute_unfolded_coords(x, y, z, ix, iy, iz, Lx, Ly, Lz):
    """计算展开坐标 (考虑周期性边界)"""
    n = len(x)
    x_u = np.empty(n, dtype=np.float64)
    y_u = np.empty(n, dtype=np.float64)
    z_u = np.empty(n, dtype=np.float64)
    for i in prange(n):
        x_u[i] = x[i] + ix[i] * Lx
        y_u[i] = y[i] + iy[i] * Ly
        z_u[i] = z[i] + iz[i] * Lz
    return x_u, y_u, z_u


@njit(cache=True, fastmath=True)
def _compute_radial_distances(x, y, z, cx, cy, cz):
    """计算每个点到中心(cx, cy, cz)的距离"""
    n = len(x)
    r = np.empty(n, dtype=np.float64)
    for i in prange(n):
        dx = x[i] - cx
        dy = y[i] - cy
        dz = z[i] - cz
        r[i] = np.sqrt(dx*dx + dy*dy + dz*dz)
    return r


@njit(cache=True, fastmath=True)
def _compute_mass_density(r, masses, r_max, dr, n_bins):
    """
    计算径向质量密度分布 (核心算法, Numba加速)

    原理:
        将空间划分为同心球壳, 计算每个球壳内的质量密度
        ρ(r) = Σm_i / V_shell
        V_shell = 4/3 * π * (r2^3 - r1^3)

    Parameters:
    -----------
    r : ndarray (float64)
        每个原子到中心的距离 (Å)
    masses : ndarray (float64)
        每个原子的质量 (amu)
    r_max : float
        最大径向距离 (Å)
    dr : float
        球壳厚度 (Å)
    n_bins : int
        球壳数量

    Returns:
    --------
    density : ndarray (float64)
        每个bin的质量密度 (amu/Å^3)
    bin_edges : ndarray (float64)
        bin边界
    """
    density = np.zeros(n_bins, dtype=np.float64)
    bin_edges = np.linspace(0.0, r_max, n_bins + 1)
    n_atoms = len(r)

    # 第一步: 将原子质量累加到对应bin
    for i in prange(n_atoms):
        ri = r[i]
        if ri >= r_max:
            continue
        bin_idx = int(ri / dr)
        if bin_idx >= n_bins:
            bin_idx = n_bins - 1
        density[bin_idx] += masses[i]

    # 第二步: 除以球壳体积得到密度
    four_pi_over_three = 4.1887902047863905
    for j in prange(n_bins):
        r1 = bin_edges[j]
        r2 = bin_edges[j + 1]
        shell_volume = four_pi_over_three * (r2*r2*r2 - r1*r1*r1)
        if shell_volume > 0:
            density[j] /= shell_volume

    return density, bin_edges


@njit(cache=True, fastmath=True)
def _moving_average(data, window):
    """移动平均平滑 (Numba加速)"""
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    half = window // 2

    for i in prange(n):
        start = i - half
        if start < 0:
            start = 0
        end = i + half + 1
        if end > n:
            end = n

        s = 0.0
        count = 0
        for j in range(start, end):
            s += data[j]
            count += 1
        result[i] = s / count

    return result


@njit(cache=True, fastmath=True)
def _compute_layer_stats(r, masses, r_min, r_max):
    """计算某一层的平均质量密度"""
    total_mass = 0.0
    n_atoms = 0
    n = len(r)

    for i in prange(n):
        if r_min <= r[i] < r_max:
            total_mass += masses[i]
            n_atoms += 1

    if n_atoms == 0:
        return 0.0, 0

    volume = 4.1887902047863905 * (r_max**3 - r_min**3)
    avg_density = total_mass / volume
    return avg_density, n_atoms


# ============================================================================
# 纯Python回退函数 (无Numba时使用)
# ============================================================================

def _compute_unfolded_coords_py(x, y, z, ix, iy, iz, Lx, Ly, Lz):
    """纯Python版本: 展开坐标"""
    x_u = x + ix * Lx
    y_u = y + iy * Ly
    z_u = z + iz * Lz
    return x_u, y_u, z_u


def _compute_radial_distances_py(x, y, z, cx, cy, cz):
    """纯Python版本: 径向距离"""
    return np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)


def _compute_mass_density_py(r, masses, r_max, dr, n_bins):
    """纯Python版本: 质量密度"""
    density = np.zeros(n_bins, dtype=np.float64)
    bin_edges = np.linspace(0.0, r_max, n_bins + 1)

    # 使用numpy的histogram功能更高效
    mass_per_bin, _ = np.histogram(r, bins=bin_edges, weights=masses)
    density = mass_per_bin.astype(np.float64)

    # 除以体积
    for j in range(n_bins):
        r1, r2 = bin_edges[j], bin_edges[j+1]
        vol = (4.0/3.0) * np.pi * (r2**3 - r1**3)
        if vol > 0:
            density[j] /= vol

    return density, bin_edges


def _moving_average_py(data, window):
    """纯Python版本: 移动平均"""
    result = np.zeros_like(data)
    half = window // 2
    for i in range(len(data)):
        start, end = max(0, i-half), min(len(data), i+half+1)
        result[i] = np.mean(data[start:end])
    return result


def _compute_layer_stats_py(r, masses, r_min, r_max):
    """纯Python版本: 层统计"""
    mask = (r >= r_min) & (r < r_max)
    total_mass = np.sum(masses[mask])
    n_atoms = np.sum(mask)
    if n_atoms == 0:
        return 0.0, 0
    volume = (4.0/3.0) * np.pi * (r_max**3 - r_min**3)
    return total_mass / volume, n_atoms


# ============================================================================
# 统一接口: 根据是否有Numba选择实现
# ============================================================================

def compute_unfolded_coords(x, y, z, ix, iy, iz, Lx, Ly, Lz):
    if HAS_NUMBA:
        return _compute_unfolded_coords(x, y, z, ix, iy, iz, Lx, Ly, Lz)
    return _compute_unfolded_coords_py(x, y, z, ix, iy, iz, Lx, Ly, Lz)


def compute_radial_distances(x, y, z, cx, cy, cz):
    if HAS_NUMBA:
        return _compute_radial_distances(x, y, z, cx, cy, cz)
    return _compute_radial_distances_py(x, y, z, cx, cy, cz)


def compute_mass_density(r, masses, r_max, dr, n_bins):
    if HAS_NUMBA:
        return _compute_mass_density(r, masses, r_max, dr, n_bins)
    return _compute_mass_density_py(r, masses, r_max, dr, n_bins)


def moving_average(data, window):
    if HAS_NUMBA:
        return _moving_average(data, window)
    return _moving_average_py(data, window)


def compute_layer_stats(r, masses, r_min, r_max):
    if HAS_NUMBA:
        return _compute_layer_stats(r, masses, r_min, r_max)
    return _compute_layer_stats_py(r, masses, r_min, r_max)


# ============================================================================
# 文件解析
# ============================================================================

def parse_lammps_dump(filepath):
    """
    解析LAMMPS dump文件

    支持格式:
        ITEM: TIMESTEP
        500000
        ITEM: NUMBER OF ATOMS
        17550
        ITEM: BOX BOUNDS pp pp pp
        xlo xhi
        ylo yhi
        zlo zhi
        ITEM: ATOMS id type x y z ix iy iz
        ...
    """
    print(f"[INFO] 读取文件: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # 解析头部
    timestep = int(lines[1].strip())
    n_atoms = int(lines[3].strip())

    xlo, xhi = map(float, lines[5].split())
    ylo, yhi = map(float, lines[6].split())
    zlo, zhi = map(float, lines[7].split())

    Lx, Ly, Lz = xhi - xlo, yhi - ylo, zhi - zlo
    box_center = np.array([xlo + Lx/2.0, ylo + Ly/2.0, zlo + Lz/2.0])

    print(f"[INFO] Timestep: {timestep}")
    print(f"[INFO] 原子总数: {n_atoms}")
    print(f"[INFO] 盒子: [{xlo:.2f}, {xhi:.2f}] x [{ylo:.2f}, {yhi:.2f}] x [{zlo:.2f}, {zhi:.2f}] Å")
    print(f"[INFO] 盒子中心: ({box_center[0]:.4f}, {box_center[1]:.4f}, {box_center[2]:.4f}) Å")

    # 分配数组
    atom_id = np.empty(n_atoms, dtype=np.int32)
    atom_type = np.empty(n_atoms, dtype=np.int32)
    x = np.empty(n_atoms, dtype=np.float64)
    y = np.empty(n_atoms, dtype=np.float64)
    z = np.empty(n_atoms, dtype=np.float64)
    ix = np.empty(n_atoms, dtype=np.int32)
    iy = np.empty(n_atoms, dtype=np.int32)
    iz = np.empty(n_atoms, dtype=np.int32)

    # 解析原子数据
    for i, line in enumerate(lines[9:]):
        if i >= n_atoms:
            break
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        atom_id[i] = int(parts[0])
        atom_type[i] = int(parts[1])
        x[i] = float(parts[2])
        y[i] = float(parts[3])
        z[i] = float(parts[4])
        ix[i] = int(parts[5])
        iy[i] = int(parts[6])
        iz[i] = int(parts[7])

    return {
        'timestep': timestep,
        'n_atoms': n_atoms,
        'box': (xlo, xhi, ylo, yhi, zlo, zhi),
        'box_center': box_center,
        'atom_id': atom_id,
        'atom_type': atom_type,
        'x': x, 'y': y, 'z': z,
        'ix': ix, 'iy': iy, 'iz': iz,
        'Lx': Lx, 'Ly': Ly, 'Lz': Lz,
    }


def compute_sio2_com(data):
    """计算SiO2核心的质心"""
    sio2_mask = np.isin(data['atom_type'], list(SIO2_TYPES))
    n_sio2 = np.sum(sio2_mask)

    print(f"\n[INFO] SiO2核心原子统计:")
    for t in SIO2_TYPES:
        count = np.sum(data['atom_type'] == t)
        print(f"  Type {t}: {count} atoms")
    print(f"  总计: {n_sio2} atoms")

    # 展开坐标
    x_sio2 = data['x'][sio2_mask] + data['ix'][sio2_mask] * data['Lx']
    y_sio2 = data['y'][sio2_mask] + data['iy'][sio2_mask] * data['Ly']
    z_sio2 = data['z'][sio2_mask] + data['iz'][sio2_mask] * data['Lz']

    com = np.array([np.mean(x_sio2), np.mean(y_sio2), np.mean(z_sio2)])
    print(f"[INFO] SiO2质心 (COM): ({com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f}) Å")
    return com


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 72)
    print("  径向质量密度分布计算 (Radial Mass Density Distribution)")
    print("  FKM/SiO2 复合材料")
    print("=" * 72)

    t_start = time.time()

    # 1. 解析dump文件
    data = parse_lammps_dump(INPUT_FILE)

    # 2. 计算SiO2质心
    sio2_com = compute_sio2_com(data)

    # 3. 展开所有原子坐标
    print(f"\n[INFO] 计算展开坐标...")
    t0 = time.time()
    x_u, y_u, z_u = compute_unfolded_coords(
        data['x'], data['y'], data['z'],
        data['ix'], data['iy'], data['iz'],
        data['Lx'], data['Ly'], data['Lz']
    )
    print(f"[TIME] 展开坐标: {time.time()-t0:.3f} s")

    # 4. 计算到SiO2 COM的径向距离
    print("[INFO] 计算径向距离...")
    t0 = time.time()
    r = compute_radial_distances(x_u, y_u, z_u, sio2_com[0], sio2_com[1], sio2_com[2])
    print(f"[TIME] 径向距离: {time.time()-t0:.3f} s")
    print(f"[INFO] 径向距离范围: {r.min():.4f} - {r.max():.4f} Å")

    # 5. 获取原子质量
    masses = np.array([ATOM_MASSES.get(int(t), 12.0) for t in data['atom_type']], dtype=np.float64)

    # 6. 计算径向质量密度
    print(f"\n[INFO] 计算径向质量密度...")
    print(f"[INFO] 参数: R_max={R_MAX} Å, dr={DR} Å, bins={int(R_MAX/DR)}")
    t0 = time.time()
    n_bins = int(R_MAX / DR)
    density, bin_edges = compute_mass_density(r, masses, R_MAX, DR, n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    print(f"[TIME] 密度计算: {time.time()-t0:.3f} s")

    # 转换为 g/cm^3
    density_gcm3 = density * DENSITY_CONVERSION

    # 7. 移动平均平滑
    print(f"[INFO] 移动平均平滑 (window={SMOOTH_WINDOW})...")
    t0 = time.time()
    density_smooth = moving_average(density_gcm3, SMOOTH_WINDOW)
    print(f"[TIME] 平滑: {time.time()-t0:.3f} s")

    # 8. 各层统计
    print(f"\n[INFO] 各层平均密度统计:")
    print("-" * 72)
    print(f"{'Layer':<15} {'R range (Å)':<22} {'Avg Density (g/cm³)':<22} {'Atoms':<10}")
    print("-" * 72)

    layer_results = {}
    for name, (rmin, rmax) in LAYER_BOUNDARIES.items():
        avg_d, n_atoms = compute_layer_stats(r, masses, rmin, min(rmax, r.max()))
        avg_d_gcm3 = avg_d * DENSITY_CONVERSION
        layer_results[name] = {
            'r_min': rmin,
            'r_max': rmax,
            'avg_density': avg_d_gcm3,
            'n_atoms': n_atoms
        }
        r_range = f"{rmin:.1f} - {min(rmax, r.max()):.1f}"
        print(f"{name:<15} {r_range:<22} {avg_d_gcm3:<22.4f} {n_atoms:<10}")
    print("-" * 72)

    # 9. 绘图
    print(f"\n[INFO] 绘制图像...")
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # 平滑曲线
    ax.plot(bin_centers, density_smooth, 'b-', linewidth=2.2, 
            label=f'Mass Density (smoothed, w={SMOOTH_WINDOW})')
    # 原始数据 (半透明)
    ax.plot(bin_centers, density_gcm3, 'gray', linewidth=0.7, alpha=0.4, label='Raw data')

    # 分层边界线
    ax.axvline(x=R_SURF, color='#FF4444', linestyle='--', linewidth=1.5, alpha=0.9)
    ax.axvline(x=R_SURF+3.0, color='#44AA44', linestyle='--', linewidth=1.5, alpha=0.9)
    ax.axvline(x=R_SURF+8.6, color='#AA44AA', linestyle='--', linewidth=1.5, alpha=0.9)

    # 分层区域阴影
    ax.axvspan(0, R_SURF, alpha=0.08, color='red')
    ax.axvspan(R_SURF, R_SURF+3.0, alpha=0.08, color='green')
    ax.axvspan(R_SURF+3.0, R_SURF+8.6, alpha=0.08, color='purple')
    ax.axvspan(R_SURF+8.6, R_MAX, alpha=0.05, color='orange')

    # 层标签
    ylim = ax.get_ylim()
    y_text = ylim[1] * 0.92
    ax.text(7.5, y_text, 'SiO₂\nCore', ha='center', fontsize=11, 
            color='#CC0000', fontweight='bold')
    ax.text(16.5, y_text, 'Contact\n(3Å)', ha='center', fontsize=10,
            color='#228822', fontweight='bold')
    ax.text(21.0, y_text, 'Interface\n(5.6Å)', ha='center', fontsize=10,
            color='#6622AA', fontweight='bold')
    ax.text(38.0, y_text, 'Bulk', ha='center', fontsize=11,
            color='#CC6600', fontweight='bold')

    ax.set_xlabel('Distance from SiO₂ COM (Å)', fontsize=14)
    ax.set_ylabel('Mass Density (g/cm³)', fontsize=14)
    ax.set_title(f'Radial Mass Density Distribution — FKM/SiO₂ Composite\n'
                 f'(N={data["n_atoms"]} atoms, t={data["timestep"]} fs, '
                 f'SiO₂ COM=({sio2_com[0]:.2f}, {sio2_com[1]:.2f}, {sio2_com[2]:.2f}) Å)',
                 fontsize=12)
    ax.set_xlim(0, R_MAX)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()

    # 保存图像
    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=DPI, bbox_inches='tight')
    print(f"[INFO] 图像已保存: {OUTPUT_PLOT}")

    # 10. 保存数据 (CSV格式)
    os.makedirs(os.path.dirname(OUTPUT_DATA), exist_ok=True)
    
    # 使用与OUTPUT_DATA相同目录，但扩展名为.csv
    csv_file = os.path.splitext(OUTPUT_DATA)[0] + '.csv'
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        import csv
        writer = csv.writer(f)
        
        # 写入注释信息（以#开头，Origin/Excel会自动忽略）
        writer.writerow(['# Radial Mass Density Distribution Data'])
        writer.writerow(['# Input file:', INPUT_FILE])
        writer.writerow(['# Timestep:', data['timestep']])
        writer.writerow(['# Total atoms:', data['n_atoms']])
        writer.writerow(['# SiO2 COM:', f'({sio2_com[0]:.6f}, {sio2_com[1]:.6f}, {sio2_com[2]:.6f}) Å'])
        writer.writerow(['#'])
        writer.writerow(['# Layer Statistics:'])
        writer.writerow(['# Layer', 'R_min(Å)', 'R_max(Å)', 'Avg Density(g/cm³)', 'Atoms'])
        for name, info in layer_results.items():
            writer.writerow(['#', name, f'{info["r_min"]:.1f}', f'{min(info["r_max"], r.max()):.1f}', 
                            f'{info["avg_density"]:.4f}', info['n_atoms']])
        writer.writerow(['#'])
        
        # 数据表头
        writer.writerow(['r(Å)', 'rho_raw(g/cm³)', 'rho_smooth(g/cm³)'])
        
        # 写入数据
        for i in range(n_bins):
            writer.writerow([f'{bin_centers[i]:.4f}', 
                            f'{density_gcm3[i]:.6f}', 
                            f'{density_smooth[i]:.6f}'])
    
    print(f"[INFO] CSV数据已保存: {csv_file}")
    print(f"[INFO] 共 {n_bins} 行数据")

    t_total = time.time() - t_start
    print(f"\n" + "=" * 72)
    print(f"  计算完成! 总耗时: {t_total:.3f} s")
    print("=" * 72)

    plt.show()


if __name__ == '__main__':
    main()