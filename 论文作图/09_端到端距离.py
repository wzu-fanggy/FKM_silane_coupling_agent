#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文配图 09 —— FKM 链端到端距离 ete（按区域 + 随帧演化）。

数据源（E 盘真实路径）：
  {SYS}\End-to-end\ete_multiframe_full.csv
  列: layer, ete_A, frame   （200 帧 × 约 120 条链）
"""
import csv
from collections import defaultdict
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
LAYERS = ["contact", "interface", "bulk"]
LAYER_ZH = {"contact": "接触层", "interface": "界面层", "bulk": "块体"}

FILES = {
    "bare":  ROOT / "End-to-end" / "ete_multiframe_full.csv",
    "KH570": ROOT / "KH570" / "End-to-end" / "ete_multiframe_full.csv",
    "KH560": ROOT / "KH560" / "End-to-end" / "ete_multiframe_full.csv",
    "KH550": ROOT / "KH550" / "End-to-end" / "ete_multiframe_full.csv",
}
# ---------------------------------------------------------------------------


def load(path):
    pooled = defaultdict(list)
    per_frame = defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            layer = row["layer"]
            ete = float(row["ete_A"])
            frame = int(row["frame"])
            pooled[layer].append(ete)
            per_frame[frame][layer].append(ete)
    frames = sorted(per_frame.keys())
    per_frame_mean = {fr: {l: sum(v) / len(v) for l, v in per_frame[fr].items()}
                      for fr in frames}
    return pooled, frames, per_frame_mean


def stats(vals):
    n = len(vals)
    m = sum(vals) / n
    s = (sum((x - m) ** 2 for x in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, s


def main():
    dat = {s: load(FILES[s]) for s in ORDER}

    # ---- 图 9a：各区域 ete 均值柱状图（误差棒=链间σ）----------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, region in zip(axes, LAYERS):
        means, errs = [], []
        for s in ORDER:
            m, sd = stats(dat[s][0][region])
            means.append(m); errs.append(sd)
        x = range(len(ORDER))
        bars = ax.bar(x, means, yerr=errs, capsize=5,
                      color=[COLORS[s] for s in ORDER], edgecolor="black", width=0.6)
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m + max(errs) * 0.03,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(list(x)); ax.set_xticklabels(ORDER)
        ax.set_title(f"{region} / {LAYER_ZH[region]}", fontsize=11)
        ax.set_ylabel("端到端距离 (Å)")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("FKM 链端到端距离对比（各区域，200 帧×120 链聚合，误差棒=链间σ）",
                 y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "fig9a_ete_region.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- 图 9b：各区域 ete 随帧演化 ------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, region in zip(axes, LAYERS):
        for s in ORDER:
            frames = dat[s][1]
            xs = [fr for fr in frames if region in dat[s][2][fr]]
            ys = [dat[s][2][fr][region] for fr in xs]
            ax.plot(xs, ys, lw=1.2, color=COLORS[s], label=s)
        ax.set_title(f"{region}/{LAYER_ZH[region]} ete（随帧）")
        ax.set_xlabel("frame"); ax.set_ylabel("端到端距离 (Å)")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("FKM 端到端距离随帧演化", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "fig9b_ete_timeseries.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("已生成 09_端到端距离 的 2 张图到:", OUT)


if __name__ == "__main__":
    main()