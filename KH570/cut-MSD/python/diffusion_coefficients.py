# -*- coding: utf-8 -*-
# Calculate diffusion coefficient D from layer-resolved MSD data
# D = mean(MSD/t) / 6 (Einstein relation)
# Based on Polymers 2021 and Materials 2025

import numpy as np
import os

# =============================================================================
# Configuration
# =============================================================================
CSV_PATH = r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\cut-MSD\python\msd_layers.csv"
OUT_DIR  = r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\cut-MSD\python"

# Fit range for Einstein regime (ps)
DT_MIN = 200
DT_MAX = 400

LAYERS = ["contact", "interface", "bulk"]

print("=" * 60)
print("Diffusion Coefficient Calculation")
print("=" * 60)
print(f"Input:  {CSV_PATH}")
print(f"Output: {OUT_DIR}")
print(f"Fit range: dt = {DT_MIN} - {DT_MAX} ps")
print()

# =============================================================================
# Load MSD data
# =============================================================================
msd_data = {name: [] for name in LAYERS}
dt_data = []

with open(CSV_PATH, "r") as f:
    header = f.readline()  # skip header
    for line in f:
        parts = line.strip().split(",")
        dt_data.append(int(parts[0]))
        msd_data["contact"].append(float(parts[1]))
        msd_data["interface"].append(float(parts[2]))
        msd_data["bulk"].append(float(parts[3]))

dt_arr = np.array(dt_data)
max_dt = dt_arr[-1]
print(f"Loaded {len(dt_arr)} data points, max dt = {max_dt} ps")
print()

# =============================================================================
# Calculate D for each layer
# D = mean(MSD/t) / 6
# =============================================================================
D_values = {}
results = []

print("=" * 60)
print("Diffusion Coefficients (D = mean(MSD/t) / 6)")
print("=" * 60)

for name in LAYERS:
    msd_arr = np.array(msd_data[name])
    
    # Select fit range (Einstein regime)
    mask = (dt_arr >= DT_MIN) & (dt_arr <= DT_MAX)
    t_fit = dt_arr[mask]
    msd_fit = msd_arr[mask]
    
    # Calculate slope = MSD/t for each point, then average
    slopes = msd_fit / t_fit
    D = slopes.mean() / 6.0
    D_cm2s = D * 1e-4  # A^2/ps -> cm^2/s
    
    D_values[name] = D_cm2s
    
    results.append({
        "layer": name,
        "D_A2_ps": D,
        "D_cm2_s": D_cm2s,
        "fit_points": len(t_fit),
        "fit_range_ps": f"{DT_MIN}-{DT_MAX}"
    })
    
    print(f"  {name:10s}: D = {D_cm2s:.3e} cm^2/s")

# =============================================================================
# Relative D (normalized to bulk)
# =============================================================================
print()
print("Relative to bulk:")
D_bulk = D_values.get("bulk", 1e-10)
for name in LAYERS:
    ratio = D_values.get(name, 0) / D_bulk if D_bulk > 0 else 0
    print(f"  {name:10s}: {ratio:.3f}")
    for r in results:
        if r["layer"] == name:
            r["relative_to_bulk"] = ratio

# =============================================================================
# Save CSV
# =============================================================================
csv_path = os.path.join(OUT_DIR, "diffusion_coefficients.csv")
with open(csv_path, "w") as f:
    f.write("layer,D_cm2_s,relative_to_bulk,fit_range_ps,fit_points\n")
    for r in results:
        f.write(f"{r['layer']},{r['D_cm2_s']:.6e},{r['relative_to_bulk']:.4f},{r['fit_range_ps']},{r['fit_points']}\n")

print(f"\nCSV saved: {csv_path}")

# =============================================================================
# Plot
# =============================================================================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    layers = [r["layer"] for r in results]
    D_vals = [r["D_cm2_s"] for r in results]
    rel_vals = [r["relative_to_bulk"] for r in results]
    colors = ["#E63946", "#2A9D8F", "#457B9D"]
    
    bars = ax.bar(layers, rel_vals, color=colors, edgecolor="black", linewidth=1.2)
    ax.set_ylabel("Relative D (normalized to bulk)", fontsize=12)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_title(
        "Diffusion Coefficient by Layer (KH570 FKM/SiO2, 500 ps NVT)\n"
        "D = mean(MSD/t) / 6, fit range: 200-400 ps",
        fontsize=12
    )
    ax.set_ylim(0, 1.2)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Bulk")
    
    # Value labels
    for bar, val in zip(bars, rel_vals):
        ax.text(
            bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.2f}", ha="center", va="bottom",
            fontsize=11, fontweight="bold"
        )
    
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    fig_path = os.path.join(OUT_DIR, "diffusion_coefficients.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {fig_path}")
    
except ImportError:
    print("matplotlib not available, skipping figure")

print()
print("Done!")
