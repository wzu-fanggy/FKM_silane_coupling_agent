# -*- coding: utf-8 -*-
# Orientation ACF for 7000ps combined trajectory (1000ps + 2000ps + 7000ps)
# ============================================================================
# DATA-STRUCTURE-OPTIMIZED STREAMING VERSION
# ----------------------------------------------------------------------------
# 相比 3000ps 的原版(Numba.py) 核心改动：
#   1. 轨迹坐标「逐帧流式」处理，绝不把全部原子坐标驻留为 Python dict/list，
#      —— 每帧处理完立即释放，全程内存峰值 = 单帧临时数组(几百 KB)。
#   2. data 文件只解析一次，得到 FKM 键集合、SiO2 质心与层划分。
#   3. 每帧仅保留 FKM 键端点 + SiO2 原子的坐标(通过 id_pos 索引直取)，其余原子行丢弃。
#      键向量逐帧写入预分配 np.ndarray，不累积 Python 对象。
#   4. 若二进制缓存(bond_vecs_*.f32 + bond_layers_*.npy)已存在则直接载入，
#      彻底跳过轨迹重读(绝不反复解析 4GB+ 的 dump)。
#   5. ACF 用 Numba JIT 并行内核(与 3000ps 一致)。
# 预期峰值内存 ~ 2 GB(bond_vecs 全量 f32 + 单层 f64 切片)，远低于原爆内存量级。
#
# 三段轨迹均为独立 run(各段 step 从 0 开始)，物理上从上一段末尾续跑，
# 拼接时间序列时丢弃第 2、3 段首帧(衔接重复帧)。
# ============================================================================

import numpy as np
import os, sys, time, re

# ============================================================
# Paths —— 只此一处定义，无冗余
# ============================================================
BASE = r"E:\A_tools\LAMMPS_GUI\projects\Lammps-MS\FKM-SIO2-TEST\cut-MSD\Orientation-related"

# (traj_path, data_path, drop_first_frame)
# drop_first=True：该段第一条帧与上一段末帧物理重复(续跑从0开始)，拼接时丢弃
SEGS = [
    (os.path.join(BASE, "3000ps", "nvt_1000ps_gpu_traj.dump"),
     os.path.join(BASE, "3000ps", "nvt_1000ps_gpu.data"),  False),
    (os.path.join(BASE, "3000ps", "nvt_2000ps_gpu_traj.dump"),
     os.path.join(BASE, "3000ps", "nvt_2000ps_gpu.data"),  True),
    (os.path.join(BASE, "7000ps", "nvt_7000ps_traj.dump"),
     os.path.join(BASE, "7000ps", "nvt_7000ps.data"),      True),
]
# 三段全拼 = 1ns(1000ps) + 2ns(2000ps) + 7ns(7000ps) = 10ns 完整时间线。
# 拼接：1001 + (2001-1) + (7001-1) = 10001 帧，ACF 覆盖到 ~10000 ps。

OUT_DIR = os.path.join(BASE, "7000ps")

# ============================================================
# Layer & types —— 与 3000ps 完全一致
# ============================================================
R_SURF = 15.1
LAYERS = {
    "contact":   (R_SURF,          R_SURF + 3.0),
    "interface": (R_SURF + 3.0,  R_SURF + 8.6),
    "bulk":      (R_SURF + 8.6,  1e6),
}
LAYER_NAME = ["contact", "interface", "bulk"]
SIO2_TYPES = {7, 8, 9}
FKM_TYPES  = {1, 2, 3, 4, 5, 6}

VEC_FILE   = os.path.join(OUT_DIR, "bond_vecs_7000ps_fkm.f32")
LAYER_FILE = os.path.join(OUT_DIR, "bond_layers_7000ps_fkm.npy")

print("=" * 60)
print("Orientation ACF - 7000ps streaming (1000+2000+7000)")
print("=" * 60)
t0 = time.time()

# ============================================================
# Step 1: 解析 data 文件(取第一段即可，结构一致)
# ============================================================
print("\n[1/6] Parsing data file:", SEGS[0][1], flush=True)
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

# 需要每帧取坐标的原子索引(按 atom id 索引)
max_id = max(atom_type)
id_pos  = np.full(max_id + 1, -1, dtype=np.int32)
id_type = np.zeros(max_id + 1, dtype=np.int32)
for aid, t in atom_type.items():
    id_pos[aid] = aid   # id 连续且从1开始 → row == id，直接索引
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

# 用第一段首帧拿盒子尺寸，并计算 SiO2 质心 + 层划分
print("\n[2/6] Reading first frame for box & layer assignment...", flush=True)
gen = stream_frame_rows(SEGS[0][0])
step0, coords0, box0 = next(gen)
gen = None
(bx0, bx1), (by0, by1), (bz0, bz1) = box0
Lx, Ly, Lz = bx1 - bx0, by1 - by0, bz1 - bz0
print(f"  Box: {Lx:.2f} x {Ly:.2f} x {Lz:.2f} A", flush=True)

# 用同一生成器拿首帧余下坐标(注意 stream 每次从头开，重新取一帧计算质心)
gen = stream_frame_rows(SEGS[0][0])
_, coords_f, _ = next(gen)
gen = None
sio2_com = coords_f[sio2_rows].mean(axis=0)
print(f"  SiO2 centroid: {sio2_com[0]:.3f}, {sio2_com[1]:.3f}, {sio2_com[2]:.3f}", flush=True)

mid = 0.5 * (coords_f[row_a1] + coords_f[row_a2])
r_mid = np.sqrt(((mid - sio2_com) ** 2).sum(axis=1))
bond_layer_arr = np.zeros(n_fkm, dtype=np.int32)
for li, (name, (lo, hi)) in enumerate(LAYERS.items()):
    bond_layer_arr[(r_mid >= lo) & (r_mid < hi)] = li
n_c, n_i, n_b = [int((bond_layer_arr == k).sum()) for k in range(3)]
print(f"  Contact: {n_c}, Interface: {n_i}, Bulk: {n_b}", flush=True)
del coords_f, coords0

# 预估总帧数（1001 + 2000 + 7000）
est_total = (1001 if not SEGS[0][2] else 1000) + \
            (2001 if not SEGS[1][2] else 2000) + \
            (7001 if not SEGS[2][2] else 7000)
print(f"  Expected total frames (after dropping dup): {est_total} (1001+2000+7000)", flush=True)

# ============================================================
# Step 3: 载入缓存 或 流式构建键向量
# ============================================================
bond_vecs_all = None
if os.path.exists(VEC_FILE) and os.path.exists(LAYER_FILE):
    print("\n[3/6] Loading cached bond_vecs...", flush=True)
    n_frames = int(os.path.getsize(VEC_FILE) // (n_fkm * 3 * 4))
    bond_vecs_all = np.fromfile(VEC_FILE, dtype=np.float32).reshape(n_frames, n_fkm, 3)
    bond_layer_arr = np.load(LAYER_FILE).astype(np.int32)
    print(f"  Loaded {n_frames} frames, shape={bond_vecs_all.shape}", flush=True)
else:
    print("\n[3/6] Streaming trajectories to build bond vectors...", flush=True)
    n_frames = 0
    # 预分配:先按预估帧数开上限，装不下再扩容(极少发生)
    bond_vecs_all = np.zeros((est_total, n_fkm, 3), dtype=np.float32)

    for seg_i, (tpath, dpath, drop_first) in enumerate(SEGS):
        print(f"  -> segment {seg_i+1}/3 ({os.path.basename(tpath)})", flush=True)
        is_first = True
        gen = stream_frame_rows(tpath)
        for step, coords, box in gen:
            if drop_first and is_first:
                is_first = False   # 丢弃衔接重复帧(该段首帧 = 上段末帧)
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
                print(f"    frames={n_frames} elapsed={time.time()-t0:.0f}s", flush=True)
        gen = None

    if n_frames != est_total:
        print(f"  [!] frame count mismatch: expected {est_total}, got {n_frames}; trimming", flush=True)
        bond_vecs_all = bond_vecs_all[:n_frames]
    print(f"  Total frames: {n_frames}, shape={bond_vecs_all.shape}, "
          f"memory~{bond_vecs_all.nbytes//1024**2} MB", flush=True)

    print("\n[4/6] Saving binary cache...", flush=True)
    bond_vecs_all.tofile(VEC_FILE)
    np.save(LAYER_FILE, bond_layer_arr)
    print(f"  Saved {VEC_FILE} ({os.path.getsize(VEC_FILE)//1024**2} MB)", flush=True)

max_dt = n_frames - 1
print(f"  bond_vecs shape: {bond_vecs_all.shape}, max_dt={max_dt}", flush=True)

# ============================================================
# Step 4: ACF —— Numba JIT 并行内核
# ============================================================
print("\n[5/6] Computing C2 ACF...", flush=True)
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("  Numba unavailable - falling back to numpy (slower)", flush=True)

layer_masks = [bond_layer_arr == li for li in range(3)]
n_layer = [lm.sum() for lm in layer_masks]
print(f"  Layer bond counts: {n_layer}", flush=True)

def acf_numpy(v):
    N = v.shape[0]
    out = np.zeros(max_dt + 1)
    out[0] = 1.0
    for dt in range(1, max_dt + 1):
        npairs = N - dt
        dot = np.einsum('fbd,fbd->fb', v[:npairs], v[dt:])
        out[dt] = (0.5 * (3.0 * dot ** 2 - 1.0)).mean()
    return out

if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def compute_acf_numba(vecs, max_dt, n_b):
        n_frames = vecs.shape[0]
        c2 = np.zeros(max_dt + 1)
        c2[0] = 1.0
        for dt in prange(1, max_dt + 1):
            npairs = n_frames - dt
            acc = 0.0; cnt = 0
            for t in range(npairs):
                for b in range(n_b):
                    dot = (vecs[t, b, 0] * vecs[t + dt, b, 0] +
                           vecs[t, b, 1] * vecs[t + dt, b, 1] +
                           vecs[t, b, 2] * vecs[t + dt, b, 2])
                    acc += 0.5 * (3.0 * dot * dot - 1.0)
                    cnt += 1
            if cnt > 0:
                c2[dt] = acc / cnt
        return c2

    print("  Compiling Numba kernel (warm-up)...", flush=True)
    _d = np.random.randn(10, 5, 3).astype(np.float32)
    _ = compute_acf_numba(_d, 5, 5)

C2 = {}
for li, name in enumerate(LAYER_NAME):
    n_b = n_layer[li]
    if n_b == 0:
        C2[name] = np.zeros(max_dt + 1)
        continue
    v = bond_vecs_all[:, layer_masks[li], :].astype(np.float64)
    print(f"  Computing {name} (bonds={n_b}, f64 mem~{v.nbytes//1024**2} MB)...", flush=True)
    ts = time.time()
    C2[name] = compute_acf_numba(v, max_dt, n_b) if HAS_NUMBA else acf_numpy(v)
    print(f"    Done in {time.time()-ts:.1f}s", flush=True)

print(f"  ACF elapsed {time.time()-t0:.1f}s", flush=True)

# ============================================================
# Step 6: Save CSV + PNG
# ============================================================
print("\n[6/6] Saving results...", flush=True)
csv_path = os.path.join(OUT_DIR, "orientation_acf_7000ps.csv")
with open(csv_path, 'w') as fp:
    fp.write("dt_ps,contact_C2,interface_C2,bulk_C2\n")
    for dt in range(1, max_dt + 1):
        fp.write(f"{dt},{C2['contact'][dt]:.6f},{C2['interface'][dt]:.6f},{C2['bulk'][dt]:.6f}\n")
print(f"  CSV: {csv_path}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
colors = {"contact": "#E63946", "interface": "#2A9D8F", "bulk": "#457B9D"}
dt_arr = np.arange(1, max_dt + 1)
for name in LAYER_NAME:
    ax.plot(dt_arr, C2[name][1:], color=colors[name], label=name, lw=2)
ax.axhline(y=1.0/np.e, color='gray', linestyle='--', lw=1, label=r"$1/e$")
ax.set_xlabel(r"$\Delta t$ (ps)", fontsize=13)
ax.set_ylabel(r"$C_2(\Delta t)$", fontsize=13)
ax.set_title("Segmental Orientation ACF -- combined (1000+2000+7000 ps), no duplicate frames", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max_dt)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
png_path = os.path.join(OUT_DIR, "orientation_acf_7000ps.png")
fig.savefig(png_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  PNG: {png_path}")

print()
print("=" * 60)
print("Results")
print("=" * 60)
for name in LAYER_NAME:
    c = C2[name]
    taus = np.where(c[1:] < 1.0/np.e)[0]
    tau = str(taus[0] + 1) if len(taus) else "NOT_REACHED"
    def g(k): return f"{c[k]:.4f}" if max_dt >= k else "N/A"
    print(f"  {name:10s}: C2(1)={c[1]:.4f}, C2(100)={g(100)}, C2(1000)={g(1000)}, "
          f"C2(5000)={g(5000)}, C2({max_dt})={c[max_dt]:.4f}, tau~{tau} ps")
print(f"\nTotal time: {time.time()-t0:.1f}s")
print("Done!")