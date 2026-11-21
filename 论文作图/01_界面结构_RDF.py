#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文配图 01 —— 界面结构：FKM 径向数密度/质量密度分布 + 界面关键指标六格图。

数据源（均为 E 盘真实路径，原脚本的 F: 路径已失效）：
  RDF 径向分布曲线:   {SYS}\RDF\outputs\interface_layer\radial_density_profile.csv
  RDF 界面指标汇总:   {SYS}\RDF\outputs\interface_layer\interface_layer_summary.txt

饱和颜色方案（全项目统一，改一处即可全局生效）：
  bare=浅绿 #90EE90,  KH550=蓝 #457B9D,  KH560=黄 #FFD700,  KH570=红 #E63946
"""
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 路径配置（只有这里需要改）------------------------------------------
ROOT = Path(r"E:\A_tools\LAMMPS_GUI\projects\Lammps-MS\FKM-SIO2-TEST")
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

ORDER = ["bare", "KH570", "KH560", "KH550"]
COLORS = {"bare": "#90EE90", "KH550": "#457B9D", "KH560": "#FFD700", "KH570": "#E63946"}

# 每个体系 RDF 输出所在目录（bare 直接在项目根的 RDF 下；接枝体系各自子目录）
RDF_DIR = {
    "bare":  ROOT / "RDF" / "outputs" / "interface_layer",
    "KH570": ROOT / "KH570" / "RDF" / "outputs" / "interface_layer",
    "KH560": ROOT / "KH560" / "RDF" / "outputs" / "interface_layer",
    "KH550": ROOT / "KH550" / "RDF" / "outputs" / "interface_layer",
}
# 各体系 radial_density_profile.csv 列号：(r 列, core 列, fkm 数密度列, fkm 质量密度列)
COLS = {
    "bare":  (0, 1, 2, 3),
    "KH570": (0, 2, 4, 5),
    "KH560": (0, 2, 4, 5),
    "KH550": (0, 2, 4, 5),
}
# ---------------------------------------------------------------------------


def load_profile(name):
    """读取一个体系的径向密度分布 csv。"""
    path = RDF_DIR[name] / "radial_density_profile.csv"
    if not path.exists():
        raise FileNotFoundError(f"未找到 {path}")
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.reader(f)):
            if i == 0:
                continue
            rows.append([float(x) for x in row if x.strip() != ""])
    r_i, core_i, n_i, m_i = COLS[name]
    return (
        [x[r_i] for x in rows],
        [x[core_i] for x in rows],
        [x[n_i] for x in rows],
        [x[m_i] for x in rows],
    )


def interface_summary(name):
    """从 interface_layer_summary.txt 正则抓取关键指标。"""
    path = RDF_DIR[name] / "interface_layer_summary.txt"
    if not path.exists():
        raise FileNotFoundError(f"未找到 {path}")
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")

    def num(key):
        m = re.search(re.escape(key) + r".*?[=:]\s*([-\d.]+)", txt)
        return float(m.group(1)) if m else float("nan")

    m = re.search(r"first FKM density peak r = [\d.]+ A, rho = ([\d.]+)", txt)
    return {
        "R_surf": num("R_surf"),
        "delta": num("interface thickness delta"),
        "delta_pert": num("delta_pert"),
        "bulk_mass": num("FKM bulk mass density"),
        "peak_r": num("first FKM density peak r"),
        "peak_rho": float(m.group(1)) if m else float("nan"),
    }


def main():
    data = {n: load_profile(n) for n in ORDER}
    met = {n: interface_summary(n) for n in ORDER}

    # ---- 图 1a：FKM 径向数密度（相当于 RDF）-------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for n in ORDER:
        r, _, fn, _ = data[n]
        ax.plot(r, fn, color=COLORS[n], lw=1.6, label=n)
    ax.set_xlabel(r"距填料中心距离 r ($\AA$)")
    ax.set_ylabel(r"FKM 数密度 (atoms/$\AA^3$)")
    ax.set_title("FKM 径向数密度剖面（四体系）")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT / "fig1a_fkm_radial_number_density.png", dpi=300)
    plt.close(fig)

    # ---- 图 1b：FKM 径向质量密度 ----------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for n in ORDER:
        r, _, _, fm = data[n]
        ax.plot(r, fm, color=COLORS[n], lw=1.6, label=n)
    ax.set_xlabel(r"距填料中心距离 r ($\AA$)")
    ax.set_ylabel(r"FKM 质量密度 (g/cm$^3$)")
    ax.set_title("FKM 径向质量密度剖面（四体系）")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT / "fig1b_fkm_radial_mass_density.png", dpi=300)
    plt.close(fig)

    # ---- 图 1c：SiO2 核径向分布（填料几何）-------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for n in ORDER:
        r, co, _, _ = data[n]
        ax.plot(r, co, color=COLORS[n], lw=1.5, label=n)
    ax.set_xlabel(r"距填料中心距离 r ($\AA$)")
    ax.set_ylabel(r"SiO$_2$/core 数密度 (atoms/$\AA^3$)")
    ax.set_title("SiO$_2$（团簇核）径向分布")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(0, 25)
    fig.tight_layout()
    fig.savefig(OUT / "fig1c_sio2_core_radial_profile.png", dpi=300)
    plt.close(fig)

    # ---- 图 1d：界面关键指标六格图 --------------------------------------
    xs = np.arange(len(ORDER))
    labels = list(met["bare"].keys())
    titles = {
        "R_surf":     (r"$R_{\rm surf}$（填料表面半径） ($\AA$)", ".3f"),
        "delta":      (r"$\delta$（界面厚度） ($\AA$)", ".3f"),
        "delta_pert": (r"$\delta_{\rm pert}$（扰动层范围） ($\AA$)", ".3f"),
        "bulk_mass":  ("FKM 块体质量密度 (g/cm$^3$)", ".4f"),
        "peak_r":     (r"$r_{\rm peak}$（密度首峰位置） ($\AA$)", ".3f"),
        "peak_rho":   (r"$\rho_{\rm peak}$（首峰高度） (atoms/$\AA^3$)", ".4f"),
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, key, L in zip(axes.flat, list(met["bare"].keys()), "abcdef"):
        ylab, fmt = titles[key]
        vals = [met[n][key] for n in ORDER]
        bars = ax.bar(xs, vals, color=[COLORS[n] for n in ORDER],
                      edgecolor="black", width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:{fmt}}",
                    ha="center", va="bottom", fontsize=8)
        ax.text(0.02, 0.95, f"({L})", transform=ax.transAxes,
                va="top", ha="left", fontsize=12, fontweight="bold")
        ax.set_xticks(xs); ax.set_xticklabels(ORDER)
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("界面关键指标对比（Interface metrics）— 4 体系", fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "fig1d_interface_metrics.png", dpi=300)
    plt.close(fig)

    print("已生成 01_界面结构_RDF 的 4 张图到:", OUT)
    for n in ORDER:
        print(f"  {n:>5}: " + "  ".join(f"{k}={met[n][k]:.3f}" for k in labels))


if __name__ == "__main__":
    main()