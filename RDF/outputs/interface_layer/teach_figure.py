"""Schematic figure for learning the interface-layer concept."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


OUT = r"C:\Users\Hxic\Documents\ChatGPT\lammps\interface_layer\interface_layer_concept.png"


def main():
    fig = plt.figure(figsize=(12, 5.2))

    # Left: cartoon of shells around the nanoparticle.
    ax = fig.add_subplot(121)
    ax.set_aspect("equal")
    ax.set_xlim(-18, 18)
    ax.set_ylim(-16, 16)
    ax.axis("off")

    r_surf = 7.0
    r_delta = 13.0
    for r, color in [(r_surf, "#4C78A8"), (r_delta, "#E45756")]:
        ax.add_patch(Circle((0, 0), r, fill=False, lw=1.6, ls="--", color=color))

    ax.add_patch(Circle((0, 0), r_surf - 0.7, facecolor="#B8C4D8", edgecolor="#2F3B52", lw=1.2, zorder=2))
    ax.text(0, 0, "SiO2", ha="center", va="center", fontsize=12, color="#1F2937", zorder=3)

    rng = np.random.default_rng(7)
    n_layer = 90
    rr = rng.uniform(r_surf + 0.4, r_delta - 0.5, n_layer)
    th = rng.uniform(0, 2 * np.pi, n_layer)
    ax.scatter(rr * np.cos(th), rr * np.sin(th), s=14, color="#E45756", alpha=0.75, zorder=4)

    n_bulk = 36
    rr2 = rng.uniform(r_delta + 0.5, 16.5, n_bulk)
    th2 = rng.uniform(0, 2 * np.pi, n_bulk)
    ax.scatter(rr2 * np.cos(th2), rr2 * np.sin(th2), s=10, color="#8C8C8C", alpha=0.55, zorder=4)

    ax.annotate(
        "SiO2 表面\nR_surf",
        xy=(r_surf, 0),
        xytext=(r_surf + 1.8, 8.2),
        fontsize=10,
        color="#2F3B52",
        arrowprops=dict(arrowstyle="->", color="#2F3B52"),
    )
    ax.annotate(
        "界面层\nΔ = r(体相) - R_surf",
        xy=(r_delta * np.cos(0.9), r_delta * np.sin(0.9)),
        xytext=(3.5, -11.5),
        fontsize=10,
        color="#B03A2E",
        arrowprops=dict(arrowstyle="->", color="#B03A2E"),
    )
    ax.text(0, 14.5, "红色：靠近表面的 FKM（排列更密）", ha="center", fontsize=10, color="#B03A2E")
    ax.text(0, 12.8, "灰色：远离表面的 FKM（体相，密度均匀）", ha="center", fontsize=10, color="#555555")
    ax.set_title("(a) 界面层是什么", fontsize=13)

    # Right: schematic density profile.
    ax2 = fig.add_subplot(122)
    r = np.linspace(0, 35, 800)
    r_surf = 15.0
    delta = 8.6
    r_peak = 17.3
    bulk = 0.087

    sio2 = np.where(r < r_surf - 1.2, 0.098, np.where(r < r_surf + 0.2, 0.098 * (r_surf + 0.2 - r) / 1.4, 0.0))

    fkm = np.zeros_like(r)
    rise = np.clip((r - r_surf) / 1.1, 0, 1)
    peak_env = 0.032 * np.exp(-0.5 * ((r - r_peak) / 0.9) ** 2)
    second_env = 0.018 * np.exp(-0.5 * ((r - 22.8) / 1.4) ** 2)
    plateau = 0.5 + 0.5 * np.tanh((r - (r_surf + delta)) / 1.0)
    fkm = rise * (bulk + peak_env + second_env) * plateau
    fkm[r <= r_surf] = 0.0

    ax2.plot(r, sio2, color="#4C78A8", lw=2, label="SiO2 数密度")
    ax2.plot(r, fkm, color="#E45756", lw=2, label="FKM 数密度")
    ax2.axvline(r_surf, color="#2F3B52", ls="--", lw=1.2)
    ax2.axvline(r_surf + delta, color="#B03A2E", ls="--", lw=1.2)
    ax2.axhline(bulk, color="gray", ls=":", lw=1.2)
    ax2.text(r_surf + 0.2, 0.115, "R_surf", fontsize=10, color="#2F3B52")
    ax2.text(r_surf + delta - 0.7, 0.115, "R_surf + Δ", fontsize=10, color="#B03A2E")
    ax2.annotate(
        "第一吸附层峰",
        xy=(r_peak, fkm[np.argmin(np.abs(r - r_peak))]),
        xytext=(19.5, 0.118),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="#B03A2E"),
    )
    ax2.annotate(
        "体相平台 ρ_bulk",
        xy=(27, bulk),
        xytext=(24.5, 0.050),
        fontsize=10,
        color="#555555",
        arrowprops=dict(arrowstyle="->", color="#555555"),
    )
    ax2.axvspan(r_surf, r_surf + delta, color="#E45756", alpha=0.08)
    ax2.text(19.0, 0.098, "界面层 Δ", fontsize=11, color="#B03A2E", ha="center")
    ax2.set_xlabel("到 SiO2 质心的距离 r (Å)")
    ax2.set_ylabel("数密度 (atoms/Å³)")
    ax2.set_xlim(0, 32)
    ax2.set_ylim(0, 0.14)
    ax2.set_title("(b) 径向密度剖面怎么读", fontsize=13)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
