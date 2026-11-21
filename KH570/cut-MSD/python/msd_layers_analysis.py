# -*- coding: utf-8 -*-
# Layer-resolved MSD analysis for KH570-grafted FKM/SiO2 NVT trajectory
# Method based on:
#   - Polymers 2021: layer MSD with Rouse scaling check
#   - Materials 2025: D = slope(MSD)/6

import numpy as np
import os, sys

# =============================================================================
# Configuration
# =============================================================================
TRAJ   = r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\cut-MSD\nvt_500ps_traj.dump"
OUTDIR = r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\cut-MSD\python"

# Layer definitions (distance from SiO2 surface)
R_SURF     = 15.05    # Angstrom, SiO2 surface radius
DELTA_PERT = 6.08
SIO2_TYPES = {5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17}
FKM_TYPES  = {1, 2, 3, 4, 8, 10}
LAYERS = {
    "contact":   (R_SURF,            R_SURF + 3.0),
    "interface": (R_SURF + 3.0,    R_SURF + DELTA_PERT),
    "bulk":      (R_SURF + DELTA_PERT, 1e6),
}

print("=" * 60)
print("Layer-resolved MSD Analysis - KH570 FKM/SiO2 Interface")
print("=" * 60)
print(f"Trajectory: {TRAJ}")
print(f"Output:    {OUTDIR}")
print(f"\nLayer definitions (distance from SiO2 surface):")
for name, (lo, hi) in LAYERS.items():
    print(f"  {name:10s}: {lo:.1f} - {hi:.1f} A")
print()

# =============================================================================
# Parse LAMMPS dump trajectory
# =============================================================================
def read_lammps_dump(path):
    """Read LAMMPS dump file and return list of frames."""
    frames = []
    frame_buf = {}
    reading_atoms = False
    atoms_to_read = 0
    
    with open(path, 'r') as f:
        for raw in f:
            line = raw.strip()
            
            if reading_atoms:
                p = line.split()
                frame_buf["atoms"][int(p[0])] = (
                    int(p[1]),    # type
                    float(p[2]),  # x
                    float(p[3]),  # y
                    float(p[4]),  # z
                    int(p[5]),    # ix
                    int(p[6]),    # iy
                    int(p[7])     # iz
                )
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
                b1 = tuple(map(float, f.readline().split()))
                b2 = tuple(map(float, f.readline().split()))
                b3 = tuple(map(float, f.readline().split()))
                frame_buf["bounds"] = (b1, b2, b3)
            
            elif line.startswith("ITEM: ATOMS"):
                frame_buf["atoms"] = {}
                reading_atoms = True
                atoms_to_read = frame_buf.get("n", 0)
    
    if frame_buf:
        frames.append(frame_buf)
    
    return frames

print("Reading trajectory...")
sys.stdout.flush()
frames = read_lammps_dump(TRAJ)
print(f"  Loaded {len(frames)} frames")
sys.stdout.flush()

# =============================================================================
# Extract box dimensions and atom groups
# =============================================================================
(bx0, bx1), (by0, by1), (bz0, bz1) = frames[0]["bounds"]
Lx, Ly, Lz = bx1 - bx0, by1 - by0, bz1 - bz0
print(f"  Box: {Lx:.2f} x {Ly:.2f} x {Lz:.2f} A")

sio2_ids = sorted(aid for aid, d in frames[0]["atoms"].items() if d[0] in SIO2_TYPES)
fkm_ids  = sorted(aid for aid, d in frames[0]["atoms"].items() if d[0] in FKM_TYPES)
n_sio2 = len(sio2_ids)
n_fkm  = len(fkm_ids)
print(f"  SiO2 atoms: {n_sio2}, FKM atoms: {n_fkm}")
print()

# =============================================================================
# Compute SiO2 COM and unwrapped FKM coords for each frame
# =============================================================================
print("Extracting coordinates (SiO2 COM + FKM unwrapped)...")
sys.stdout.flush()

sio2_coms = []
fkm_unwrapped = []

for fi, fr in enumerate(frames):
    atoms = fr["atoms"]
    
    # SiO2 center of mass
    cx = sum(atoms[a][1] + atoms[a][4] * Lx for a in sio2_ids) / n_sio2
    cy = sum(atoms[a][2] + atoms[a][5] * Ly for a in sio2_ids) / n_sio2
    cz = sum(atoms[a][3] + atoms[a][6] * Lz for a in sio2_ids) / n_sio2
    sio2_coms.append((cx, cy, cz))
    
    # FKM unwrapped coords relative to SiO2 COM
    coords = np.empty((n_fkm, 3), dtype=np.float64)
    for j, aid in enumerate(fkm_ids):
        _, x, y, z, ix, iy, iz = atoms[aid]
        coords[j, 0] = x + ix * Lx - cx
        coords[j, 1] = y + iy * Ly - cy
        coords[j, 2] = z + iz * Lz - cz
    fkm_unwrapped.append(coords)
    
    if fi % 100 == 0 and fi > 0:
        print(f"    frame {fi}/{len(frames)} done")
        sys.stdout.flush()

print(f"  Done. SiO2 COM z range: {min(c[2] for c in sio2_coms):.2f} - {max(c[2] for c in sio2_coms):.2f} A")
print()

# =============================================================================
# Assign initial layer labels (based on frame 0)
# =============================================================================
print("Assigning layer labels (frame 0)...")
layer_atoms = {name: [] for name in LAYERS}
r0 = fkm_unwrapped[0]

for j in range(n_fkm):
    r = np.linalg.norm(r0[j])
    for name, (lo, hi) in LAYERS.items():
        if lo <= r < hi:
            layer_atoms[name].append(j)
            break

for name, idxs in layer_atoms.items():
    print(f"  {name:10s}: {len(idxs):5d} atoms")
print()

# =============================================================================
# Compute layer-resolved MSD (multi-time-origin averaging)
# =============================================================================
print("Computing layer-resolved MSD...")
sys.stdout.flush()

n_frames = len(frames)
max_dt = n_frames - 1  # ps

# Accumulation arrays
msd_accum = {name: np.zeros(max_dt + 1) for name in LAYERS}
msd_count = {name: np.zeros(max_dt + 1, dtype=int) for name in LAYERS}

for t0 in range(n_frames):
    c0 = fkm_unwrapped[t0]
    for dt in range(1, n_frames - t0):
        # Displacement vector (already unwrapped)
        dr2 = np.sum((fkm_unwrapped[t0 + dt] - c0) ** 2, axis=1)
        
        for name, idxs in layer_atoms.items():
            msd_accum[name][dt] += np.sum(dr2[idxs])
            msd_count[name][dt] += len(idxs)
    
    if t0 % 50 == 0:
        print(f"  t0 = {t0}/{n_frames} processed")
        sys.stdout.flush()

# Average MSD
msd = {}
for name in LAYERS:
    valid = msd_count[name] > 0
    msd[name] = np.zeros(max_dt + 1)
    msd[name][valid] = msd_accum[name][valid] / msd_count[name][valid]
print()

# =============================================================================
# Save CSV
# =============================================================================
csv_path = os.path.join(OUTDIR, "msd_layers.csv")
with open(csv_path, 'w') as fout:
    # Header
    fout.write("dt_ps")
    for name in LAYERS:
        fout.write(f",{name}_A2")
    fout.write(",dt_ps_scaled")
    for name in LAYERS:
        fout.write(f",{name}_A2_over_sqrt_dt")
    fout.write("\n")
    
    # Data
    for dt in range(1, max_dt + 1):
        fout.write(str(dt))
        for name in LAYERS:
            fout.write(f",{msd[name][dt]:.6f}")
        
        dt_sqrt = dt ** 0.5
        fout.write(f",{dt_sqrt:.6f}")
        for name in LAYERS:
            fout.write(f",{msd[name][dt] / dt_sqrt:.6f}")
        fout.write("\n")

print(f"CSV saved: {csv_path}")

# =============================================================================
# Plot
# =============================================================================
print("Generating plots...")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {"contact": "#E63946", "interface": "#2A9D8F", "bulk": "#457B9D"}
    dt_arr = np.arange(1, max_dt + 1, dtype=float)
    
    # Plot 1: MSD vs dt
    ax = axes[0]
    for name in LAYERS:
        ax.plot(dt_arr, msd[name][1:], color=colors[name], label=name, lw=2)
    ax.set_xlabel(r"$\Delta t$ (ps)", fontsize=12)
    ax.set_ylabel(r"MSD ($\AA^2$)", fontsize=12)
    ax.set_title("Layer-resolved MSD (KH570 FKM/SiO2, 500 ps NVT)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_dt)
    
    # Plot 2: MSD / sqrt(dt) vs dt (Rouse scaling check)
    ax2 = axes[1]
    for name in LAYERS:
        ax2.plot(dt_arr, msd[name][1:] / np.sqrt(dt_arr), 
                 color=colors[name], label=name, lw=2)
    ax2.set_xlabel(r"$\Delta t$ (ps)", fontsize=12)
    ax2.set_ylabel(r"MSD / $\sqrt{\Delta t}$ ($\AA^2 / \sqrt{ps}$)", fontsize=12)
    ax2.set_title("Rouse scaling check (plateau = Rouse behavior)", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_dt)
    
    plt.tight_layout()
    fig_path = os.path.join(OUTDIR, "msd_layers.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {fig_path}")
    
except Exception as e:
    print(f"Plot error: {e}")

# =============================================================================
# Estimate diffusion coefficients
# =============================================================================
print()
print("=" * 60)
print("Results Summary")
print("=" * 60)

D_values = {}
t_arr = np.arange(max_dt + 1)

for name in LAYERS:
    d = msd[name]
    
    # Fit D in Einstein regime (dt = 200-400 ps)
    mask = (t_arr >= 200) & (t_arr <= 400)
    valid = mask[1:]
    
    if valid.sum() > 3:
        slopes = d[1:][valid] / t_arr[1:][valid]
        D = slopes.mean() / 6.0
        D_values[name] = D
        print(f"  {name:10s}: D = {D:.4f} A^2/ps = {D*1e-4:.3e} cm^2/s  (Einstein, dt=200-400 ps)")
    
    print(f"  {name:10s}: MSD(100) = {d[100]:.3f} A^2, MSD(300) = {d[300]:.3f} A^2")

# Relative D
print()
print("Relative to bulk:")
D_bulk = D_values.get("bulk", 1e-10)
for name in LAYERS:
    ratio = D_values.get(name, 0) / D_bulk
    print(f"  {name:10s}: {ratio:.2f}")

# Save summary
summary_path = os.path.join(OUTDIR, "msd_layers_summary.txt")
with open(summary_path, 'w') as fout:
    fout.write("=" * 60 + "\n")
    fout.write("Layer-resolved MSD Summary\n")
    fout.write("=" * 60 + "\n")
    fout.write(f"Trajectory: {TRAJ}\n")
    fout.write(f"Frames: {len(frames)} @ 1 ps/frame = {max_dt} ps\n")
    fout.write(f"Box: {Lx:.2f} x {Ly:.2f} x {Lz:.2f} A\n")
    fout.write(f"R_surf = {R_SURF:.1f} A\n")
    fout.write("\n")
    fout.write("Layer definitions:\n")
    for name, (lo, hi) in LAYERS.items():
        fout.write(f"  {name}: {lo:.1f} - {hi:.1f} A\n")
    fout.write("\nAtom counts:\n")
    for name, idxs in layer_atoms.items():
        fout.write(f"  {name}: {len(idxs)} atoms\n")
    fout.write("\n")
    fout.write("Diffusion coefficients (Einstein, dt=200-400 ps):\n")
    for name in LAYERS:
        D = D_values.get(name, 0)
        ratio = D / D_bulk if D_bulk > 0 else 0
        fout.write(f"  {name}: D = {D*1e-4:.3e} cm^2/s (relative = {ratio:.2f})\n")
    fout.write("\n")
    fout.write("MSD at key time lags:\n")
    for name in LAYERS:
        fout.write(f"  {name}: MSD(100)={msd[name][100]:.3f} A^2, MSD(300)={msd[name][300]:.3f} A^2\n")

print()
print(f"Summary saved: {summary_path}")
print()
print("Done!")
