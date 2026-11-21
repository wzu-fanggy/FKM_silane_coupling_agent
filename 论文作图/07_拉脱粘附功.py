#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文配图 07 —— 拉脱粘附功（界面强度核心图）：峰值力 / 分离功 随拉速。

数据源（E 盘真实路径）：
  compare\pull\pull_comparison.csv    (原始 F_max / W_sep / residual)
  compare\pull\pull_comparison_normalized.csv  (每接触原子归一化)
两者均由 compare\pull\compare_pull_normalized.py 预先算好的汇总表，直接读即可。

核心结论锚点：界面强度排序 KH570 > KH560 ≈ KH550。
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 路径配置 ----
COMPARE_PULL = Path(__file__).resolve().parent.parent / "compare" / "pull"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

RAW_CSV = COMPARE_PULL / "pull_comparison.csv"
NORM_CSV = COMPARE_PULL / "pull_comparison_normalized.csv"

ORDER = ["bare", "KH550", "KH560", "KH570"]
COLORS = {"bare": "#90EE90", "KH550": "#457B9D", "KH560": "#FFD700", "KH570": "#E63946"}
EDGES = {"bare": "#5fbf5f", "KH550": "#355f7d", "KH560": "#c4a800", "KH570": "#c42b38"}

# 各体系在两类表中的列名前缀（bare 用 bare_，接枝的是 khxxx_）
RAW_PREFIX = {"bare": "bare_", "KH550": "kh550_", "KH560": "kh560_", "KH570": "kh570_"}
# 原始表：F_max 列 = 前缀 + "F_max"，W 列 = 前缀 + "W_sep"
# 归一化表：F 列 = 前缀 + "F_per_contact"，W 列 = 前缀 + "W_per_contact"
# ---------------------------------------------------------------------------


def load(path, col):
    """col: {system: 列名}。返回 {speed: {system: value}}。"""
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f)):
            speed = int(row["speed_Aps"])
            out[speed] = {}
            for s in ORDER:
                out[speed][s] = float(row[col[s]])
    return out


def grouped_bars(panels, ylabels, fname, title):
    """panels = [(speed->{system->val}, colmap-for-label), ...]，每面板一组。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    xs = np.arange(len(panels[0][0])); width = 0.2
    speeds = sorted(panels[0][0].keys())
    for ax, (data, _colmap), ylab in zip(axes, panels, ylabels):
        for j, s in enumerate(ORDER):
            means = [data[speed][s] for speed in speeds]
            ax.bar(xs + (j - 1.5) * width, means, width, color=COLORS[s],
                   edgecolor=EDGES[s], linewidth=0.8, label=s)
        ax.set_xticks(xs); ax.set_xticklabels([f"{s}" for s in speeds])
        ax.set_xlabel("拉脱速率 (Å/ps)")
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y", alpha=0.3); ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[n], ec=EDGES[n])
               for n in ORDER]
    for ax in axes:
        ax.legend(handles, ORDER, ncol=4, frameon=False, loc="upper left")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=300)
    plt.close(fig)


def fits(panels, ylabels, fname, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (data, _colmap), ylab in zip(axes, panels, ylabels):
        lg = []
        for s in ORDER:
            x = np.array(sorted(data.keys()), float)
            y = np.array([data[speed][s] for speed in sorted(data.keys())], float)
            k_, b_ = np.polyfit(x, y, 1)
            yhat = k_ * x + b_
            r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            xs = np.linspace(x.min(), x.max(), 80)
            ax.plot(xs, k_ * xs + b_, "-", color=COLORS[s], linewidth=1.6)
            lg.append(Line2D([0], [0], color=COLORS[s],
                             label=f"{s} (k={k_:.2f}, R$^2$={r2:.2f})"))
        ax.set_xlabel("拉脱速率 (Å/ps)")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(handles=lg, frameon=False, loc="upper left", fontsize=9)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=300)
    plt.close(fig)


def main():
    if not RAW_CSV.exists() or not NORM_CSV.exists():
        raise FileNotFoundError("缺少 pull_comparison.csv / pull_comparison_normalized.csv\n"
                                f"检查目录: {COMPARE_PULL}")

    # 原始（未归一化）：柱状图（两面板：峰值力 / 分离功）
    f_col = {s: RAW_PREFIX[s] + "F_max" for s in ORDER}
    w_col = {s: RAW_PREFIX[s] + "W_sep" for s in ORDER}
    data_f = load(RAW_CSV, f_col)
    data_w = load(RAW_CSV, w_col)

    grouped_bars([(data_f, f_col), (data_w, w_col)],
                 ["峰值力 (kcal/mol/Å)", "分离功 (kcal/mol)"],
                 "fig7a_pull_raw_bars.png",
                 "拉脱峰值力与分离功（原始，不同拉速）— 界面强度对比")

    # 归一化（每接触原子）：柱状图 + 线性拟合
    nf_col = {s: RAW_PREFIX[s] + "F_per_contact" for s in ORDER}
    nw_col = {s: RAW_PREFIX[s] + "W_per_contact" for s in ORDER}
    data_nf = load(NORM_CSV, nf_col)
    data_nw = load(NORM_CSV, nw_col)

    grouped_bars([(data_nf, nf_col), (data_nw, nw_col)],
                 ["峰值力/接触原子 (kcal/mol/Å/atom)", "分离功/接触原子 (kcal/mol/atom)"],
                 "fig7c_pull_norm_bars.png",
                 "归一化拉脱结果（每接触原子，不同拉速）")
    fits([(data_nf, nf_col), (data_nw, nw_col)],
         ["峰值力/接触原子 (kcal/mol/Å/atom)", "分离功/接触原子 (kcal/mol/atom)"],
         "fig7d_pull_norm_fit.png",
         "归一化拉脱结果线性拟合 (y=kx+b)")

    print("已生成 07_拉脱粘附功 的 3 张图到:", OUT)
    print("  fig7a 原始柱状图 | fig7c 归一化柱状图 | fig7d 归一化线性拟合")
    print("核心结论锚点：界面强度排序 KH570 > KH560 ≈ KH550")


if __name__ == "__main__":
    main()