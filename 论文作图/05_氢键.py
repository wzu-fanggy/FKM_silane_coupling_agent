#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文配图 05 —— 界面氢键：分类柱状图 + 总界面氢键时序。

数据源（E 盘真实路径）：
  汇总: compare\H\hbond_compare_summary.csv   (类别 A/B1/B2/D/E/F/total)
  时序: {SYS}\H\hbond_last50.csv              (time_ps, total_interface)
裸体系（bare）无硅烷接枝，没有氢键类别，这里只对比 KH550/KH560/KH570。
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
HB_SUMMARY = Path(__file__).resolve().parent.parent / "compare" / "H" / "hbond_compare_summary.csv"
OUT.mkdir(parents=True, exist_ok=True)

SYS = ["KH550", "KH560", "KH570"]
COLORS = {"KH550": "#457B9D", "KH560": "#FFD700", "KH570": "#E63946"}

HB_LAST50 = {
    "KH550": ROOT / "KH550" / "H" / "hbond_last50.csv",
    "KH560": ROOT / "KH560" / "H" / "hbond_last50.csv",
    "KH570": ROOT / "KH570" / "H" / "hbond_last50.csv",
}
# 类别显示顺序与标签
CATS = [
    "A", "B1", "B2", "D", "E", "F", "total",
]
CAT_LABEL = {
    "A": "A: 硅羟基→FKM氟",
    "B1": "B1: FKM碳氢→SiO2氧",
    "B2": "B2: FKM碳氢→接枝受体",
    "D": "D: 接枝氢→FKM氟",
    "E": "E: 接枝氢→SiO2氧",
    "F": "F: 硅羟基→接枝受体",
    "total": "总界面氢键",
}
# ---------------------------------------------------------------------------


def load_summary():
    """读取 hbond_compare_summary.csv，返回 {(cat, sys): (mean, std)}。"""
    out = {}
    with open(HB_SUMMARY, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row["category"].split()[0].split("(")[0]  # 取类别首 token
            for s in SYS:
                key = f"{s}_std"
                out[(cat, s)] = (float(row[s]), float(row[key]))
    return out


def load_timeseries(path):
    t, tot = [], []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            t.append(float(row["time_ps"]))
            tot.append(float(row["total_interface"]))
    return t, tot


def main():
    d = load_summary()

    # ---- 图 5a：各类别柱状图（2x4，最后空一格）---------------------------
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    for idx, cat in enumerate(CATS):
        ax = axes.flat[idx]
        means = [d[(cat, s)][0] for s in SYS]
        stds = [d[(cat, s)][1] for s in SYS]
        bars = ax.bar(SYS, means, yerr=stds, capsize=5,
                      color=[COLORS[s] for s in SYS], edgecolor="black", width=0.6)
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m + max(stds) * 0.05,
                    f"{m:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(0.02, 0.95, f"({chr(97+idx)})", transform=ax.transAxes,
                va="top", ha="left", fontsize=12, fontweight="bold")
        ax.set_title(CAT_LABEL[cat], fontsize=9)
        ax.set_ylabel("平均每帧氢键数 (3.0 Å/120°)", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    axes.flat[7].axis("off")
    fig.suptitle("界面氢键分类对比（KH550 / KH560 / KH570）", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "fig5a_hbond_metrics.png", dpi=300)
    plt.close(fig)

    # ---- 图 5b：总界面氢键时序 ------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for s in SYS:
        t, tot = load_timeseries(HB_LAST50[s])
        ax.plot(t, tot, "-o", ms=3, lw=1.4, color=COLORS[s], label=s)
    ax.set_xlabel("时间 (ps)")
    ax.set_ylabel("总界面氢键数")
    ax.set_title("总界面氢键随时间变化（最后 50 帧, 3.0 Å/120°）")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig5b_hbond_total_timeseries.png", dpi=300)
    plt.close(fig)

    print("已生成 05_氢键 的 2 张图到:", OUT)


if __name__ == "__main__":
    main()