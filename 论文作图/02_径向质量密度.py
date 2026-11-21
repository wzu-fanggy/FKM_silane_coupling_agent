#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文配图 02 —— 径向质量密度分布（核-壳-基体）与分区平均密度。

数据源（E 盘真实路径）：
  {SYS}\cut-MSD\Density distribution\radial_density_data_{bare,kh570,kh560,kh550}.txt

txt 格式：以 # 开头的行给出分区范围 (Core/Contact/Interface/Bulk 的 R=[min,max] 与 rho)，
  常规行 3 列：r, 原始质量密度, 平滑后质量密度。
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

# ---- 路径配置 ----
ROOT = Path(r"E:\A_tools\LAMMPS_GUI\projects\Lammps-MS\FKM-SIO2-TEST")
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

ORDER = ["bare", "KH570", "KH560", "KH550"]
COLORS = {"bare": "#90EE90", "KH550": "#457B9D", "KH560": "#FFD700", "KH570": "#E63946"}

FILES = {
    "bare":  ROOT / "cut-MSD" / "Density distribution" / "radial_density_data_bare.txt",
    "KH570": ROOT / "KH570" / "cut-MSD" / "Density distribution" / "radial_density_data_kh570.txt",
    "KH560": ROOT / "KH560" / "cut-MSD" / "Density distribution" / "radial_density_data_kh560.txt",
    "KH550": ROOT / "KH550" / "cut-MSD" / "Density distribution" / "radial_density_data_kh550.txt",
}
ZONES = ["Core", "Contact", "Interface", "Bulk"]
ZONE_ZH = {"Core": "核心", "Contact": "接触层", "Interface": "界面层", "Bulk": "块体"}
RE_ZONE = re.compile(r"# (Core|Contact|Interface|Bulk).*?R=\[([\d.]+),([\d.]+)\].*?rho=([\d.]+)")
# ---------------------------------------------------------------------------


def parse(path):
    if not path.exists():
        raise FileNotFoundError(f"未找到 {path}")
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    zones, r, raw, sm = {}, [], [], []
    for line in txt.splitlines():
        s = line.strip()
        m = RE_ZONE.match(s)
        if m:
            zones[m.group(1)] = (float(m.group(2)), float(m.group(3)),
                                 float(m.group(4)))
            continue
        if s and not s.startswith("#"):
            p = s.split()
            if len(p) == 3:
                r.append(float(p[0])); raw.append(float(p[1])); sm.append(float(p[2]))
    return zones, r, raw, sm


def main():
    dat = {s: parse(FILES[s]) for s in ORDER}

    # ---- 图 2a：总径向质量密度平滑曲线叠加 -------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for s in ORDER:
        zones, r, raw, sm = dat[s]
        ax.plot(r, sm, color=COLORS[s], lw=1.7, label=s)
    ax.axvline(15.05, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"距填料/接枝簇质心距离 (Å)")
    ax.set_ylabel("总质量密度 (g/cm³)")
    ax.set_title("径向质量密度分布（总，核-壳-基体）")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT / "fig2a_total_radial_density.png", dpi=300)
    plt.close(fig)

    # ---- 图 2b：分区平均质量密度柱状图 -----------------------------------
    x = np.arange(len(ZONES)); width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for si, s in enumerate(ORDER):
        means = [dat[s][0][z][2] for z in ZONES]
        ax.bar(x + (si - 1.5) * width, means, width=width,
               color=COLORS[s], edgecolor="black", label=s)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{z}\n{ZONE_ZH[z]}" for z in ZONES])
    ax.set_ylabel("分区平均质量密度 (g/cm³)")
    ax.set_title("分区平均质量密度（Core/Contact/Interface/Bulk）")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig2b_zone_density.png", dpi=300)
    plt.close(fig)

    # ---- 汇总 CSV：分区平均密度 + 各分区半径范围 --------------------------
    with open(OUT / "fig2_density_summary.csv", "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["zone", "zone_range_A"] + ORDER)
        for z in ZONES:
            r0, r1 = dat["bare"][0][z][0], dat["bare"][0][z][1]
            w.writerow([z, f"[{r0:.2f},{r1:.2f}]"]
                       + [f"{dat[s][0][z][2]:.4f}" for s in ORDER])

    print("分区平均密度 (g/cm³) 与分区范围 [Å]:")
    for s in ORDER:
        rng = ", ".join(f"{z}=[{v[0]:.2f}-{v[1]:.2f}]" for z, v in dat[s][0].items())
        print(f"  {s:>6}: {rng}")
    print("已生成 02_径向质量密度 的 2 张图 + summary CSV 到:", OUT)


if __name__ == "__main__":
    main()