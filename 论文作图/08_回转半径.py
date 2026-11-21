#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文配图 08 —— FKM 链回转半径 Rg（按区域 + 随帧演化）。

数据源（E 盘真实路径）：
  {SYS}\RG\rg_by_frame_{bare?570/560/550}.csv
  列：frame, contact_Rg_mean, interface_Rg_mean, bulk_Rg_mean,
      contact_Rg_std, interface_Rg_std, bulk_Rg_std
"""
import csv
from pathlib import Path

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
REGIONS = ["contact", "interface", "bulk"]
REGION_ZH = {"contact": "接触层", "interface": "界面层", "bulk": "块体"}

FILES = {
    "bare":  ROOT / "RG" / "rg_by_frame.csv",
    "KH570": ROOT / "KH570" / "RG" / "rg_by_frame_570.csv",
    "KH560": ROOT / "KH560" / "RG" / "rg_by_frame_560.csv",
    "KH550": ROOT / "KH550" / "RG" / "rg_by_frame_550.csv",
}
# ---------------------------------------------------------------------------


def load(path):
    frames, means, stds = [], {}, {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            frames.append(int(row["frame"]))
            for r in REGIONS:
                means.setdefault(r, []).append(float(row[f"{r}_Rg_mean"]))
                stds.setdefault(r, []).append(float(row[f"{r}_Rg_std"]))
    return frames, means, stds


def main():
    dat = {s: load(FILES[s]) for s in ORDER}

    # ---- 图 8a：各区域 Rg 均值柱状图（误差棒=帧均值下链间 std）------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, region in zip(axes, REGIONS):
        means = [sum(dat[s][1][region]) / len(dat[s][1][region]) for s in ORDER]
        stds = [sum(dat[s][2][region]) / len(dat[s][2][region]) for s in ORDER]
        x = range(len(ORDER))
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=[COLORS[s] for s in ORDER], edgecolor="black", width=0.6)
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m + max(stds) * 0.04,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(list(x)); ax.set_xticklabels(ORDER)
        ax.set_title(f"{region.capitalize()} / {REGION_ZH[region]} 区 Rg", fontsize=11)
        ax.set_ylabel("回转半径 Rg (Å)")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("FKM 链回转半径对比（各区域，均值±链间标准差，200 帧平均）",
                 y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "fig8a_rg_region.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- 图 8b：各区域 Rg 随帧演化 ------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, region in zip(axes, REGIONS):
        for s in ORDER:
            frames = dat[s][0]
            ax.plot(frames, dat[s][1][region], lw=1.2, color=COLORS[s], label=s)
        ax.set_title(f"{region.capitalize()}/{REGION_ZH[region]} Rg（随时间）")
        ax.set_xlabel("frame"); ax.set_ylabel("Rg (Å)")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("FKM 链回转半径随帧演化", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "fig8b_rg_timeseries.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("已生成 08_回转半径 的 2 张图到:", OUT)


if __name__ == "__main__":
    main()