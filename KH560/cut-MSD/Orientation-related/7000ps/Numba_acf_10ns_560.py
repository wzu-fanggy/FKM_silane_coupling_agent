#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orientation ACF for the KH560-grafted FKM/SiO2 system -- 10 ns (3000ps + 7000ps).

两段全拼：nvt_3000ps_traj.dump(3001帧) + nvt_7000ps_traj.dump(7001帧)，
丢 7000 段首帧(续跑衔接重复帧) → 10001 帧 = 10 ns 完整时间线，max_dt=10000。

参数完全沿用 Numba_3000ps_560.py：
  FKM_TYPES    = {1,2,3,4,6,8}
  SIO2_TYPES   = {5,7,9,10,11,12,13,14,15}
  R_SURF       = 15.095 (KH560 实测)
  DELTA_PERT   = 8.28 (bulk 界)
  contact 层 = (R_SURF, R_SURF+3.0)
  interface   = (R_SURF+3.0, R_SURF+8.28)

仅 FKM 键取向 C2(二阶勒让德)，不含接枝复合体键。
输出 orientation_acf_10ns_560.csv / .png

数据流优化(与 KH550 版一致)：逐帧流式拾取、每帧即释、
二进制缓存复用、Numba 内存友好分层核，避免全量 float64 副本。

运行：python Numba_acf_10ns_560.py
"""
import os
import re
import time

import numpy as np

# ============================================================
# Paths
# ============================================================
# 相对路径：脚本位于 <体系>/Orientation-related/7000ps/，
# BASE（Orientation-related）= 脚本目录的上一级
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(SCRIPT_DIR)

# (traj_path, data_path, drop_first_frame)
# drop_first=True：该段首帧与上段末帧物理重复(续跑从0开始)，拼接时丢弃
SEGS = [
    (os.path.join(BASE, "nvt_3000ps_traj.dump"),
     os.path.join(BASE, "nvt_3000ps.data"),   False),
    (os.path.join(BASE, "7000ps", "nvt_7000ps_traj.dump"),
     os.path.join(BASE, "7000ps", "nvt_7000ps.data"), True),
]
# 拼接 = 3001 + (7001-1) = 10001 帧，ACF 覆盖 1..10000 ps

OUT_DIR = os.path.join(BASE, "7000ps")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 分层基准：KH560 实测（与 3000ps 脚本一致）
# ============================================================
DELTA_PERT = 8.28
R_SURF = 15.095
R_LABEL = "rsurf_15_095"
LAYER_NAME = ["contact", "interface", "bulk"]

SIO2_TYPES = {5, 7, 9, 10, 11, 12, 13, 14, 15}
FKM_TYPES = {1, 2, 3, 4, 6, 8}

print("=" * 60)
print("Orientation ACF - KH560 FKM/SiO2 (3000ps + 7000ps = 10ns)")
print("=" * 60)
t0 = time.time()

# ============================================================
# Step 1: 解析 data 文件(取第一段即可，结构一致)
# ============================================================
print("\n[1/] Parsing data file:", SEGS[0][1], flush=True)
with open(SEGS[0][1], 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

atoms_ma = content.find("Atoms # full")
bonds_ma = content.find("Bonds")
angles_ma = content.find("Angles")

atom_type = {}
for line in content[atoms_ma:bonds_ma].split('\n'):
    p = line.split()
    if len(p) >= 7 and p[0].isdigit():
        atom_type[int(p[0])] = int(p[2])

m = re.search(r'^(\d+)\s+bonds\s*$', content, re.MULTILINE)
n_bonds_hdr = int(m.group(1)) if m else None

bonds = []
for line in content[bonds_ma:angles_ma if angles_ma != -1 else None].split('\n'):
    p = line.split()
    if len(p) >= 4 and p[0].isdigit() and (n_bonds_hdr is None or int(p[0]) <= n_bonds_hdr):
        bonds.append((int(p[2]), int(p[3])))

n_bonds = len(bonds)
n_atoms = len(atom_type)
print(f"  {n_atoms} atoms, {n_bonds} bonds", flush=True)

fkm_mask = np.array([
    atom_type.get(b[0], -1) in FKM_TYPES and atom_type.get(b[1], -1) in FKM_TYPES
    for b in bonds], dtype=bool)
fkm_idx_arr = np.where(fkm_mask)[0]
n_fkm = len(fkm_idx_arr)
print(f"  FKM bonds only: {n_fkm} (excluded {n_bonds - n_fkm})", flush=True)

a1_arr = np.array([bonds[i][0] for i in fkm_idx_arr], dtype=np.int32)
a2_arr = np.array([bonds[i][1] for i in fkm_idx_arr], dtype=np.int32)

max_id = max(atom_type)
id_pos  = np.full(max_id + 1, -1, dtype=np.int32)
id_type = np.zeros(max_id + 1, dtype=np.int32)
for aid, t in atom_type.items():
    id_pos[aid] = aid
    id_type[aid] = t

sio2_atoms = np.array([aid for aid, t in atom_type.items() if t in SIO2_TYPES], dtype=np.int32)
n_sio2 = len(sio2_atoms)
row_a1 = np.array([id_pos[a] for a in a1_arr], dtype=np.int32)
row_a2 = np.array([id_pos[a] for a in a2_arr], dtype=np.int32)
sio2_rows = np.array([id_pos[a] for a in sio2_atoms], dtype=np.int32)

# ============================================================
# Step 2: 流式轨迹生成器 —— 每帧只拾取目标原子坐标
# ============================================================
def stream_frame_rows(traj_path):
    """逐帧 yield (step, coords, box)。
    coords: (max_id+1, 3) float64，仅 row_a1/row_a2/sio2_rows 槽位有值，其余为 0。
    不产生任何 Python dict / list 累积。"""
    reading = 0
    n_this = 0
    box = None
    step = None
    coords = np.zeros((max_id + 1, 3), dtype=np.float64)
    with open(traj_path, 'r', encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.strip()
            if reading:
                p = line.split()
                if len(p) >= 8:
                    aid = int(p[0])
                    coords[aid, 0] = float(p[2]) + int(p[5]) * box_L[0]
                    coords[aid, 1] = float(p[3]) + int(p[6]) * box_L[1]
                    coords[aid, 2] = float(p[4]) + int(p[7]) * box_L[2]
                reading -= 1
                if reading == 0:
                    yield step, coords, box
                    coords = np.zeros((max_id + 1, 3), dtype=np.float64)
            elif line.startswith("ITEM: TIMESTEP"):
                step = int(f.readline().strip())
            elif line.startswith("ITEM: NUMBER OF ATOMS"):
                n_this = int(f.readline().strip())
            elif line.startswith("ITEM: BOX BOUNDS"):
                bx = tuple(map(float, f.readline().split()))
                by = tuple(map(float, f.readline().split()))
                bz = tuple(map(float, f.readline().split()))
                box = (bx, by, bz)
                box_L = (bx[1] - bx[0], by[1] - by[0], bz[1] - bz[0])
            elif line.startswith("ITEM: ATOMS"):
                reading = n_this

# 用第一段首帧拿盒尺寸、SiO2 质心与分层
print("\n[2/] Reading first frame for box & layer assignment...", flush=True)
gen = stream_frame_rows(SEGS[0][0])
step0, coords0, box0 = next(gen)
gen = None
(bx0, bx1), (by0, by1), (bz0, bz1) = box0
Lx, Ly, Lz = bx1 - bx0, by1 - by0, bz1 - bz0
print(f"  Box: {Lx:.2f} x {Ly:.2f} x {Lz:.2f} A", flush=True)

gen = stream_frame_rows(SEGS[0][0])
_, coords_f, _ = next(gen)
gen = None
sio2_com = coords_f[sio2_rows].mean(axis=0)
print(f"  SiO2+KH560 cluster centroid: {sio2_com[0]:.3f}, {sio2_com[1]:.3f}, {sio2_com[2]:.3f}", flush=True)

mid = 0.5 * (coords_f[row_a1] + coords_f[row_a2])
r_mid = np.sqrt(((mid - sio2_com) ** 2).sum(axis=1))
del coords_f, coords0

# 预估总帧数 = 3001 + 7000
est_total = 3001 + (7001 - 1)
print(f"  Expected total frames (after dropping dup): {est_total} (3001+7000)", flush=True)

# ============================================================
# Step 3: 载入缓存 或 流式构建键向量 (单 R_SURF)
# ============================================================
def layer_assign_roc(r_mid, r_surf):
    lo_c, hi_c = r_surf, r_surf + 3.0
    hi_i = r_surf + DELTA_PERT
    arr = np.zeros(n_fkm, dtype=np.int32)
    arr[(r_mid >= lo_c) & (r_mid < hi_c)] = 0
    arr[(r_mid >= hi_c) & (r_mid < hi_i)] = 1
    arr[r_mid >= hi_i] = 2
    return arr

def build_or_load(r_surf, r_lab, r_mid):
    """计算分层并(缓存命中时)复用键向量。返回 (bond_vecs_all, bond_layer_arr)。"""
    bond_layer_arr = layer_assign_roc(r_mid, r_surf)

    vec_file = os.path.join(OUT_DIR, f"bond_vecs_10ns_fkm_{r_lab}.f32")
    layer_file = os.path.join(OUT_DIR, f"bond_layers_10ns_fkm_{r_lab}.npy")

    bond_vecs_all = None
    if os.path.exists(vec_file) and os.path.exists(layer_file):
        n_frames = int(os.path.getsize(vec_file) // (n_fkm * 3 * 4))
        bond_vecs_all = np.fromfile(vec_file, dtype=np.float32).reshape(n_frames, n_fkm, 3)
        bond_layer_arr = np.load(layer_file).astype(np.int32)
        print(f"    [{r_lab}] loaded cache: {n_frames} frames, {vec_file}", flush=True)
        return bond_vecs_all, bond_layer_arr

    print(f"    [{r_lab}] streaming segs...", flush=True)
    bond_vecs_all = np.zeros((est_total, n_fkm, 3), dtype=np.float32)
    n_frames = 0
    for seg_i, (tpath, dpath, drop_first) in enumerate(SEGS):
        print(f"      segment {seg_i+1}/2 ({os.path.basename(tpath)})", flush=True)
        is_first = True
        gen = stream_frame_rows(tpath)
        for step, coords, box in gen:
            if drop_first and is_first:
                is_first = False
                continue
            d = coords[row_a2] - coords[row_a1]
            d[:, 0] -= Lx * np.round(d[:, 0] / Lx)
            d[:, 1] -= Ly * np.round(d[:, 1] / Ly)
            d[:, 2] -= Lz * np.round(d[:, 2] / Lz)
            nrm = np.sqrt((d * d).sum(axis=1))
            nrm[nrm == 0] = 1.0
            bond_vecs_all[n_frames] = (d / nrm[:, None])
            n_frames += 1
            if n_frames % 500 == 0:
                print(f"        frames={n_frames} elapsed={time.time()-t0:.0f}s", flush=True)
        gen = None

    if n_frames != est_total:
        print(f"      [!] frame mismatch: expected {est_total}, got {n_frames}; trimming", flush=True)
        bond_vecs_all = bond_vecs_all[:n_frames]
    print(f"      total frames: {n_frames}, memory~{bond_vecs_all.nbytes//1024**2} MB", flush=True)

    bond_vecs_all.tofile(vec_file)
    np.save(layer_file, bond_layer_arr)
    print(f"      saved cache {vec_file} ({os.path.getsize(vec_file)//1024**2} MB)", flush=True)
    return bond_vecs_all, bond_layer_arr

print("\n[3/] Building/loading bond vectors...", flush=True)

# 尝试用 Numba
HAS_NUMBA = True
try:
    from numba import njit, prange
except ImportError:
    HAS_NUMBA = False
    print("  Numba unavailable - falling back to numpy (slower)", flush=True)

def acf_numpy(v, max_dt):
    N = v.shape[0]
    out = np.zeros(max_dt + 1)
    out[0] = 1.0
    for dt in range(1, max_dt + 1):
        npairs = N - dt
        dot = np.einsum('fbd,fbd->fb', v[:npairs], v[dt:])
        out[dt] = (0.5 * (3.0 * dot ** 2 - 1.0)).mean()
    return out

if HAS_NUMBA:
    # 内存友好核：直接对 float32 全量数组 + 逐层键索引，避免 float64 大副本
    @njit(parallel=True, cache=True)
    def compute_acf_numba(vecs_f32, bond_idx, max_dt):
        n_frames = vecs_f32.shape[0]
        n_bonds = len(bond_idx)
        c2_vals = np.zeros(max_dt + 1, dtype=np.float64)
        c2_vals[0] = 1.0
        for dt in prange(1, max_dt + 1):
            n_pairs = n_frames - dt
            accum = 0.0
            for t in range(n_pairs):
                for b in range(n_bonds):
                    ib = bond_idx[b]
                    dot = (vecs_f32[t, ib, 0] * vecs_f32[t + dt, ib, 0] +
                           vecs_f32[t, ib, 1] * vecs_f32[t + dt, ib, 1] +
                           vecs_f32[t, ib, 2] * vecs_f32[t + dt, ib, 2])
                    accum += 0.5 * (3.0 * dot * dot - 1.0)
            c2_vals[dt] = accum / (n_pairs * n_bonds)
        return c2_vals

    print("  Compiling Numba kernel (warm-up)...", flush=True)
    _d = np.random.randn(10, 5, 3).astype(np.float32)
    _di = np.arange(5, dtype=np.int32)
    _ = compute_acf_numba(_d, _di, 4)

# ============================================================
# Step 4-6: ACF + CSV + PNG (单 R_SURF)
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

colors = {"contact": "#E63946", "interface": "#2A9D8F", "bulk": "#457B9D"}

print(f"\n========== {R_LABEL}  (R_SURF={R_SURF}) ==========", flush=True)
bond_vecs_all, bond_layer_arr = build_or_load(R_SURF, R_LABEL, r_mid)
n_frames = bond_vecs_all.shape[0]
max_dt = n_frames - 1

layer_bond_idx = [
    np.ascontiguousarray(np.where(bond_layer_arr == li)[0].astype(np.int32))
    for li in range(3)
]
n_layer_bonds = [int(len(idx)) for idx in layer_bond_idx]
print(f"  Layer bond counts: {n_layer_bonds}", flush=True)

C2 = {}
for li, name in enumerate(LAYER_NAME):
    if n_layer_bonds[li] == 0:
        C2[name] = np.zeros(max_dt + 1)
        continue
    print(f"  Computing {name} (bonds={n_layer_bonds[li]})...", flush=True)
    t_acf = time.time()
    C2[name] = (compute_acf_numba(bond_vecs_all, layer_bond_idx[li], max_dt)
                if HAS_NUMBA else acf_numpy(bond_vecs_all[:, bond_layer_arr == li, :], max_dt))
    print(f"    Done in {time.time()-t_acf:.1f}s", flush=True)

csv_path = os.path.join(OUT_DIR, f"orientation_acf_10ns_{R_LABEL}.csv")
with open(csv_path, 'w') as fp:
    fp.write("dt_ps,contact_C2,interface_C2,bulk_C2\n")
    for dt in range(1, max_dt + 1):
        fp.write(f"{dt},{C2['contact'][dt]:.6f},{C2['interface'][dt]:.6f},{C2['bulk'][dt]:.6f}\n")
print(f"  CSV: {csv_path}")

fig, ax = plt.subplots(figsize=(10, 6))
dt_arr = np.arange(1, max_dt + 1)
for name in LAYER_NAME:
    ax.plot(dt_arr, C2[name][1:], color=colors[name], label=name, lw=2)
ax.axhline(y=1.0 / np.e, color='gray', linestyle='--', lw=1, label=r"$1/e$")
ax.set_xlabel(r"$\Delta t$ (ps)", fontsize=13)
ax.set_ylabel(r"$C_2(\Delta t)$", fontsize=13)
ax.set_title(f"KH560 Orientation ACF -- 10ns (3000+7000 ps), R_SURF={R_LABEL}", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max_dt)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
png_path = os.path.join(OUT_DIR, f"orientation_acf_10ns_{R_LABEL}.png")
fig.savefig(png_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  PNG: {png_path}")

print(f"  [{R_LABEL}] results:")
for name in LAYER_NAME:
    c = C2[name]
    taus = np.where(c[1:] < 1.0 / np.e)[0]
    tau_str = str(taus[0] + 1) if len(taus) else "NOT_REACHED"
    def g(k): return f"{c[k]:.4f}" if max_dt >= k else "N/A"
    print(f"    {name:10s}: C2(1)={c[1]:.4f}, C2(100)={g(100)}, C2(1000)={g(1000)}, "
          f"C2(5000)={g(5000)}, C2({max_dt})={c[max_dt]:.4f}, tau~{tau_str} ps")

del bond_vecs_all, bond_layer_arr, layer_bond_idx

print(f"\nTotal time: {time.time()-t0:.1f}s")
print("Done!")