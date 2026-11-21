"""Annotated figure: how R_surf is derived from the smoothed SiO2 profile."""

import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


CSV = Path(r"C:\Users\Hxic\Documents\ChatGPT\lammps\interface_layer\outputs\radial_density_profile.csv")
OUT = Path(r"C:\Users\Hxic\Documents\ChatGPT\lammps\interface_layer\sio2_rsurf_how.png")


def main():
    with open(CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    r = np.array([float(x["r_A"]) for x in rows])
    n = np.array([float(x["SiO2_n_atoms_per_A3"]) for x in rows])

    w = 5
    n_s = np.convolve(n, np.ones(w) / w, mode="same")

    core_mask = (r >= 5.0) & (r <= 13.0)
    rho_core = float(np.mean(n_s[core_mask]))
    half = 0.5 * rho_core

    above = n_s >= half
    idx = np.flatnonzero(above)
    last = idx[-1]
    r0, r1 = r[last], r[last + 1]
    v0, v1 = n_s[last], n_s[last + 1]
    r_surf = r0 + (half - v0) * (r1 - r0) / (v1 - v0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(r, n, lw=0.8, color="#B8C4D8", label="SiO2 原始数密度")
    ax.plot(r, n_s, lw=1.8, color="#4C78A8", label="SiO2 平滑数密度（5 bin 滑动平均）")
    ax.axvspan(5.0, 13.0, color="#F2C14E", alpha=0.25, label="核心平台区 5–13 Å（取平均）")
    ax.axhline(rho_core, color="#F2C14E", ls="--", lw=1.2, label=f"核心平台 ρ_core = {rho_core:.4f}")
    ax.axhline(half, color="#E45756", ls="--", lw=1.2, label=f"半高阈值 0.5×ρ_core = {half:.4f}")
    ax.axvline(r_surf, color="#2F3B52", ls="-.", lw=1.5, label=f"R_surf = {r_surf:.3f} Å")
    ax.annotate(
        "最后一次穿过半高阈值",
        xy=(r_surf, half),
        xytext=(10.5, 0.018),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="#2F3B52"),
    )
    ax.set_xlabel("到 SiO2 质心距离 r (Å)")
    ax.set_ylabel("SiO2 数密度 (atoms/Å³)")
    ax.set_title("R_surf 的计算过程：核心平台平均 → 半高阈值 → 最后一次穿越点")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 0.16)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    plt.close(fig)
    print(f"rho_core={rho_core:.4f}, half={half:.4f}, R_surf={r_surf:.3f} A")
    print(OUT)


if __name__ == "__main__":
    main()
