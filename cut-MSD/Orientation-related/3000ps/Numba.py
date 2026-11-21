# -*- coding: utf-8 -*-
# Orientation ACF for 3000ps combined trajectory (1000ps + 2000ps)
# NUMBA JIT ACCELERATED: ~50-100x speedup on ACF computation
# Requires: pip install numba

import numpy as np
import os, sys, time

# ============================================================
# Paths
# ============================================================
DATA1 = r"F:\Lammps-MS\FKM-SIO2-TEST\cut-MSD\Orientation-related\3000ps\nvt_1000ps_gpu.data"
TRAJ1 = r"F:\Lammps-MS\FKM-SIO2-TEST\cut-MSD\Orientation-related\3000ps\nvt_1000ps_gpu_traj.dump"
DATA2 = r"F:\Lammps-MS\FKM-SIO2-TEST\cut-MSD\Orientation-related\3000ps\nvt_2000ps_gpu.data"
TRAJ2 = r"F:\Lammps-MS\FKM-SIO2-TEST\cut-MSD\Orientation-related\3000ps\nvt_2000ps_gpu_traj.dump"
OUT_DIR = r"F:\Lammps-MS\FKM-SIO2-TEST\cut-MSD\Orientation-related\3000ps"

# ============================================================
# Layer definition
# ============================================================
R_SURF = 15.1
LAYERS = {
    "contact":   (R_SURF,          R_SURF + 3.0),
    "interface": (R_SURF + 3.0,  R_SURF + 8.6),
    "bulk":      (R_SURF + 8.6,  1e6),
}
LAYER_NAME = ["contact", "interface", "bulk"]
SIO2_TYPES = {7, 8, 9}
FKM_TYPES = {1, 2, 3, 4, 5, 6}

VEC_FILE   = os.path.join(OUT_DIR, "bond_vecs_3000ps_fkm.f32")
LAYER_FILE = os.path.join(OUT_DIR, "bond_layers_3000ps_fkm.npy")

print("=" * 60)
print("Orientation ACF - 3000ps NUMBA JIT ACCELERATED")
print("=" * 60)
t0 = time.time()

# ============================================================
# Step 1: Parse data file
# ============================================================
import re
print("\n[1/7] Parsing data file...", flush=True)
with open(DATA1, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

atoms_match = content.find("Atoms # full")
bonds_match = content.find("Bonds")
angles_match = content.find("Angles")
atom_mol = {}
atom_type = {}
for line in content[atoms_match:bonds_match].split('\n'):
    parts = line.split()
    if len(parts) >= 7 and parts[0].isdigit():
        atom_mol[int(parts[0])] = int(parts[1])
        atom_type[int(parts[0])] = int(parts[2])

n_bonds_header = None
m = re.search(r'^(\d+)\s+bonds\s*$', content, re.MULTILINE)
if m:
    n_bonds_header = int(m.group(1))
    print(f"  Data file header: {n_bonds_header} bonds", flush=True)

bonds = []
bond_count = 0
for line in content[bonds_match:angles_match].split('\n'):
    parts = line.split()
    if len(parts) >= 4 and parts[0].isdigit():
        bid = int(parts[0])
        if n_bonds_header is None or bid <= n_bonds_header:
            bonds.append((int(parts[2]), int(parts[3])))
            bond_count += 1
            if n_bonds_header and bond_count >= n_bonds_header:
                break

n_bonds = len(bonds)
n_atoms = len(atom_mol)
print(f"  {n_atoms} atoms, {n_bonds} bonds", flush=True)

fkm_bond_mask = np.array([
    atom_type.get(b[0], -1) in FKM_TYPES and
    atom_type.get(b[1], -1) in FKM_TYPES
    for b in bonds
], dtype=bool)
fkm_idx = np.where(fkm_bond_mask)[0]
n_bonds_fkm = len(fkm_idx)
print(f"  FKM bonds only: {n_bonds_fkm} (excluded {n_bonds - n_bonds_fkm} SiO2/graft bonds)", flush=True)

# ============================================================
# Step 2: Trajectory reader
# ============================================================
def read_frames(traj_path):
    frames = []
    frame_buf = {}
    reading_atoms = False
    atoms_to_read = 0

    with open(traj_path, 'r', encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.strip()
            if reading_atoms:
                p = line.split()
                frame_buf["atoms"][int(p[0])] = (
                    int(p[1]), float(p[2]), float(p[3]), float(p[4]),
                    int(p[5]), int(p[6]), int(p[7]))
                atoms_to_read -= 1
                if atoms_to_read == 0:
                    reading_atoms = False
                    frames.append(frame_buf)
                    frame_buf = {}
            elif line == "ITEM: TIMESTEP":
                if frame_buf:
                    frames.append(frame_buf)
                frame_buf = {"step": int(f.readline().strip())}
            elif line == "ITEM: NUMBER OF ATOMS":
                frame_buf["n"] = int(f.readline().strip())
            elif "BOX BOUNDS" in line:
                frame_buf["bounds"] = (
                    tuple(map(float, f.readline().split())),
                    tuple(map(float, f.readline().split())),
                    tuple(map(float, f.readline().split())),
                )
            elif line.startswith("ITEM: ATOMS"):
                frame_buf["atoms"] = {}
                reading_atoms = True
                atoms_to_read = frame_buf.get("n", 0)
    if frame_buf:
        frames.append(frame_buf)
    return frames

# ============================================================
# Step 3: Load or build bond_vecs binary
# ============================================================
if os.path.exists(VEC_FILE) and os.path.exists(LAYER_FILE):
    print("\n[2/7] Loading cached bond_vecs from binary...", flush=True)
    n_frames_total = int(os.path.getsize(VEC_FILE) // (n_bonds_fkm * 3 * 4))
    bond_vecs_all = np.fromfile(VEC_FILE, dtype=np.float32).reshape(n_frames_total, n_bonds_fkm, 3)
    bond_layer_arr = np.load(LAYER_FILE)
    n_frames = n_frames_total
    print(f"  Loaded {n_frames} frames, shape={bond_vecs_all.shape}", flush=True)
else:
    print("\n[2/7] Loading trajectory 1/2 (1000 ps)...", flush=True)
    frames1 = read_frames(TRAJ1)
    n_f1 = len(frames1)
    print(f"  {n_f1} frames loaded, elapsed {time.time()-t0:.1f}s", flush=True)

    print("\n[3/7] Loading trajectory 2/2 (2000 ps)...", flush=True)
    frames2 = read_frames(TRAJ2)
    n_f2 = len(frames2)
    print(f"  {n_f2} frames loaded, elapsed {time.time()-t0:.1f}s", flush=True)

    n_frames = n_f1 + n_f2
    print(f"\n  Total: {n_frames} frames ({n_f1}+{n_f2})", flush=True)

    # The second trajectory starts from the final frame of the first one
    # (its own TIMESTEP restarts at 0), so drop the duplicate first frame.
    frames = frames1 + frames2[1:]
    del frames1, frames2
    n_frames = len(frames)

    (bx0, bx1), (by0, by1), (bz0, bz1) = frames[0]["bounds"]
    Lx, Ly, Lz = bx1 - bx0, by1 - by0, bz1 - bz0
    print(f"  Box: {Lx:.2f} x {Ly:.2f} x {Lz:.2f} A", flush=True)

    sio2_ids = [aid for aid, d in frames[0]["atoms"].items() if d[0] in SIO2_TYPES]
    n_sio2 = len(sio2_ids)
    print(f"  SiO2 atoms: {n_sio2}", flush=True)

    bond_a1 = np.array([bonds[i][0] for i in fkm_idx], dtype=np.int32)
    bond_a2 = np.array([bonds[i][1] for i in fkm_idx], dtype=np.int32)
    bond_layer_arr = np.zeros(n_bonds_fkm, dtype=np.int32)

    # Assign layers
    sio2_com = np.zeros(3)
    for aid in sio2_ids:
        _, x, y, z, ix, iy, iz = frames[0]["atoms"][aid]
        sio2_com += np.array([x + ix*Lx, y + iy*Ly, z + iz*Lz])
    sio2_com /= n_sio2

    for i, bi in enumerate(fkm_idx):
        a1, a2 = bonds[bi]
        if a1 not in frames[0]["atoms"] or a2 not in frames[0]["atoms"]:
            continue
        _, x1, y1, z1, ix1, iy1, iz1 = frames[0]["atoms"][a1]
        _, x2, y2, z2, ix2, iy2, iz2 = frames[0]["atoms"][a2]
        mx = (x1+ix1*Lx + x2+ix2*Lx)/2 - sio2_com[0]
        my = (y1+iy1*Ly + y2+iy2*Ly)/2 - sio2_com[1]
        mz = (z1+iz1*Lz + z2+iz2*Lz)/2 - sio2_com[2]
        r = np.sqrt(mx*mx + my*my + mz*mz)
        for li, (name, (lo, hi)) in enumerate(LAYERS.items()):
            if lo <= r < hi:
                bond_layer_arr[i] = li
                break

    n_c = np.sum(bond_layer_arr == 0)
    n_i = np.sum(bond_layer_arr == 1)
    n_b = np.sum(bond_layer_arr == 2)
    print(f"  Contact: {n_c}, Interface: {n_i}, Bulk: {n_b}", flush=True)

    # Build bond vectors frame by frame (unchanged - already vectorized per frame)
    print("\n[4/7] Computing bond vectors frame by frame...", flush=True)
    bond_vecs_all = np.zeros((n_frames, n_bonds_fkm, 3), dtype=np.float32)

    for fi, fr in enumerate(frames):
        atoms = fr["atoms"]
        cx = sum(atoms[a][1]+atoms[a][4]*Lx for a in sio2_ids) / n_sio2
        cy = sum(atoms[a][2]+atoms[a][5]*Ly for a in sio2_ids) / n_sio2
        cz = sum(atoms[a][3]+atoms[a][6]*Lz for a in sio2_ids) / n_sio2

        a1_coords = np.array(
            [[atoms[a][1] + atoms[a][4]*Lx,
              atoms[a][2] + atoms[a][5]*Ly,
              atoms[a][3] + atoms[a][6]*Lz] for a in bond_a1],
            dtype=np.float32)
        a2_coords = np.array(
            [[atoms[a][1] + atoms[a][4]*Lx,
              atoms[a][2] + atoms[a][5]*Ly,
              atoms[a][3] + atoms[a][6]*Lz] for a in bond_a2],
            dtype=np.float32)

        vx = a2_coords[:,0] - a1_coords[:,0] - Lx * np.round((a2_coords[:,0]-a1_coords[:,0])/Lx)
        vy = a2_coords[:,1] - a1_coords[:,1] - Ly * np.round((a2_coords[:,1]-a1_coords[:,1])/Ly)
        vz = a2_coords[:,2] - a1_coords[:,2] - Lz * np.round((a2_coords[:,2]-a1_coords[:,2])/Lz)

        norm = np.sqrt(vx*vx + vy*vy + vz*vz)
        norm[norm == 0] = 1.0

        bond_vecs_all[fi, :, 0] = vx / norm
        bond_vecs_all[fi, :, 1] = vy / norm
        bond_vecs_all[fi, :, 2] = vz / norm

        if (fi+1) % 200 == 0 or fi == n_frames-1:
            print(f"    Frame {fi+1}/{n_frames} ({(fi+1)*100//n_frames}%), "
                  f"elapsed {time.time()-t0:.1f}s", flush=True)

    del frames, a1_coords, a2_coords

    print("\n[5/7] Saving bond_vecs to binary cache...", flush=True)
    bond_vecs_all.tofile(VEC_FILE)
    np.save(LAYER_FILE, bond_layer_arr)
    sz_mb = os.path.getsize(VEC_FILE) // 1024 // 1024
    print(f"  Saved {VEC_FILE} ({sz_mb} MB)", flush=True)

print(f"  bond_vecs shape: {bond_vecs_all.shape}, dtype: {bond_vecs_all.dtype}", flush=True)
print(f"  Memory: ~{bond_vecs_all.nbytes//1024//1024} MB", flush=True)

# ============================================================
# Step 4: Compute ACF - NUMBA JIT ACCELERATED
# ============================================================
print("\n[6/7] Computing C2 ACF (Numba JIT)...", flush=True)

try:
    from numba import njit, prange
    HAS_NUMBA = True
    print("  Numba available - using JIT compilation")
except ImportError:
    HAS_NUMBA = False
    print("  Numba not available - falling back to numpy (slower)")

max_dt = n_frames - 1

# Precompute layer masks
layer_masks = [bond_layer_arr == li for li in range(3)]
n_layer_bonds = [mask.sum() for mask in layer_masks]
print(f"  Layer bond counts: {n_layer_bonds}")

if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def compute_acf_numba(vecs, max_dt, n_bonds_layer):
        """
        vecs: (n_frames, n_bonds, 3) float64
        Returns: c2_vals (max_dt+1,) float64
        """
        n_frames = vecs.shape[0]
        c2_vals = np.zeros(max_dt + 1, dtype=np.float64)
        c2_vals[0] = 1.0

        for dt in prange(1, max_dt + 1):
            n_pairs = n_frames - dt
            accum = 0.0
            count = 0

            for t in range(n_pairs):
                for b in range(n_bonds_layer):
                    dot = (vecs[t, b, 0] * vecs[t+dt, b, 0] +
                           vecs[t, b, 1] * vecs[t+dt, b, 1] +
                           vecs[t, b, 2] * vecs[t+dt, b, 2])
                    P2 = 0.5 * (3.0 * dot * dot - 1.0)
                    accum += P2
                    count += 1

            if count > 0:
                c2_vals[dt] = accum / count

        return c2_vals

    # Warm-up compilation with small data
    print("  Compiling Numba kernel (first run may be slow)...", flush=True)
    _dummy = np.random.randn(10, 5, 3)
    _ = compute_acf_numba(_dummy, 5, 5)

    C2 = {}
    for li, name in enumerate(LAYER_NAME):
        mask = layer_masks[li]
        n_b = n_layer_bonds[li]
        if n_b == 0:
            C2[name] = np.zeros(max_dt + 1)
            continue

        v = bond_vecs_all[:, mask, :].astype(np.float64)
        print(f"  Computing {name} (bonds={n_b})...", flush=True)
        t_acf_start = time.time()
        c2_vals = compute_acf_numba(v, max_dt, n_b)
        print(f"    Done in {time.time()-t_acf_start:.1f}s", flush=True)
        C2[name] = c2_vals

else:
    # Fallback: numpy vectorized (original logic, optimized)
    C2 = {}
    for li, name in enumerate(LAYER_NAME):
        mask = layer_masks[li]
        n_b = n_layer_bonds[li]
        if n_b == 0:
            C2[name] = np.zeros(max_dt + 1)
            continue

        v = bond_vecs_all[:, mask, :].astype(np.float64)
        c2_vals = np.zeros(max_dt + 1, dtype=np.float64)
        c2_vals[0] = 1.0

        for dt in range(1, max_dt + 1):
            n_pairs = n_frames - dt
            dot = np.einsum('fbd,fbd->fb', v[:n_pairs], v[dt:])
            P2 = 0.5 * (3.0 * dot**2 - 1.0)
            c2_vals[dt] = P2.mean()

            if dt % 500 == 0:
                print(f"  {name}: dt={dt}/{max_dt}", flush=True)

        C2[name] = c2_vals

print(f"  Done, elapsed {time.time()-t0:.1f}s", flush=True)

# ============================================================
# Step 5: Save CSV + PNG
# ============================================================
print("\n[7/7] Saving results...", flush=True)

csv_path = os.path.join(OUT_DIR, "orientation_acf_3000ps.csv")
with open(csv_path, 'w') as f:
    f.write("dt_ps,contact_C2,interface_C2,bulk_C2\n")
    for dt in range(1, max_dt+1):
        f.write(f"{dt},{C2['contact'][dt]:.6f},{C2['interface'][dt]:.6f},{C2['bulk'][dt]:.6f}\n")
print(f"  CSV: {csv_path}")

fixed_csv_path = os.path.join(OUT_DIR, "orientation_acf_3000ps_fixed.csv")
with open(fixed_csv_path, 'w') as f:
    f.write("dt_ps,contact_C2,interface_C2,bulk_C2\n")
    for dt in range(1, max_dt+1):
        f.write(f"{dt},{C2['contact'][dt]:.6f},{C2['interface'][dt]:.6f},{C2['bulk'][dt]:.6f}\n")
print(f"  CSV: {fixed_csv_path}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
colors = {"contact": "#E63946", "interface": "#2A9D8F", "bulk": "#457B9D"}
dt_arr = np.arange(1, max_dt+1)

for name in LAYER_NAME:
    ax.plot(dt_arr, C2[name][1:], color=colors[name], label=name, lw=2)

ax.axhline(y=1.0/np.e, color='gray', linestyle='--', lw=1, label=r"$1/e$")
ax.set_xlabel(r"$\Delta t$ (ps)", fontsize=13)
ax.set_ylabel(r"$C_2(\Delta t)$", fontsize=13)
ax.set_title("Segmental Orientation ACF (fixed: FKM bonds only, no duplicate frame)", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max_dt)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "orientation_acf_3000ps.png")
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  PNG: {fig_path}")

fixed_fig_path = os.path.join(OUT_DIR, "orientation_acf_3000ps_fixed.png")
fig, ax = plt.subplots(figsize=(10, 6))
for name in LAYER_NAME:
    ax.plot(dt_arr, C2[name][1:], color=colors[name], label=name, lw=2)
ax.axhline(y=1.0/np.e, color='gray', linestyle='--', lw=1, label=r"$1/e$")
ax.set_xlabel(r"$\Delta t$ (ps)", fontsize=13)
ax.set_ylabel(r"$C_2(\Delta t)$", fontsize=13)
ax.set_title("Segmental Orientation ACF (fixed: FKM bonds only, no duplicate frame)", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max_dt)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
fig.savefig(fixed_fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  PNG: {fixed_fig_path}")

# ============================================================
# Results summary
# ============================================================
print()
print("=" * 60)
print("Results")
print("=" * 60)
for name in LAYER_NAME:
    c = C2[name]
    tau_vals = np.where(c[1:] < 1.0/np.e)[0]
    tau_str = str(tau_vals[0]+1) if len(tau_vals) > 0 else "NOT_REACHED"
    c100  = f"{c[100]:.4f}"  if max_dt >= 100  else "N/A"
    c500  = f"{c[500]:.4f}"  if max_dt >= 500  else "N/A"
    c1000 = f"{c[1000]:.4f}" if max_dt >= 1000 else "N/A"
    print(f"  {name:10s}: C2(1)={c[1]:.4f}, C2(100)={c100}, "
          f"C2(500)={c500}, C2(1000)={c1000}, "
          f"C2({max_dt})={c[max_dt]:.4f}, tau~{tau_str} ps")

print(f"\nTotal time: {time.time()-t0:.1f}s")
print("Done!")
