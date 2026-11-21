from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


OUT = Path(r"C:\Users\Hxic\Documents\ChatGPT\lammps\interface_energy_errorbar.png")

labels = ["E_total", "E_vdw", "E_ele"]
means = [-1302.29, -969.46, -332.83]
stds = [15.40, 9.30, 14.15]
colors = ["tab:blue", "tab:orange", "tab:green"]

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(labels, means, yerr=stds, capsize=8, color=colors, alpha=0.85)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 15, f"{m:.1f} ± {s:.1f}", ha="center", fontsize=10)
ax.set_ylabel("能量 (kcal/mol)")
ax.set_title("FKM-SiO2 静态界面能分解 (最后50帧, mean ± std)")
ax.axhline(0, color="black", lw=0.8)
fig.tight_layout()
fig.savefig(OUT, dpi=160)
plt.close(fig)
print(f"Wrote {OUT}")
