#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文配图 03 —— 链段取向动力学：C2(t) 衰减曲线 + KWW 弛豫参数。

数据源（E 盘真实路径）：
  取向自相关 C2(t):  {SYS}\cut-MSD\Orientation-related\...\orientation_acf_3000ps.csv
   KWW 弛豫参数汇总: 论文作图\03 生成时读取 compare\cut-MSD\Orientation-related\orientation_compare_summary.csv

注意：讨论.docx 已要求取向数据加长到 7 ns 重算。当前 3000 ps 下 KH570/KH550 的
  tau_KWW 为 extrapolated（未充分松弛）。等 7 ns 数据到位后，把下面 ACF_FILES 的
  路径（和文件名里的 3000ps）换成新轨迹产出的文件即可，脚本逻辑不变。裸体系注意
  是 1000+2000+7000 三个文件一起读取，别漏。
"""
import csv
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
SUMMARY = Path(__file__).resolve().parent.parent / "compare" / "cut-MSD" / "Orientation-related"
OUT.mkdir(parents=True, exist_ok=True)

ORDER = ["bare", "KH570", "KH560", "KH550"]
COLORS = {"bare": "#90EE90", "KH550": "#457B9D", "KH560": "#FFD700", "KH570": "#E63946"}
LAYERS = ["contact", "interface", "bulk"]
LAYER_ZH = {"contact": "接触层", "interface": "界面层", "bulk": "块体"}
ACF_COL = {"contact": "contact_C2", "interface": "interface_C2", "bulk": "bulk_C2"}

# 各体系 C2(t) 曲线文件（bare 有一个额外的 3000ps 子目录）
ACF_FILES = {
    "bare":  ROOT / "cut-MSD" / "Orientation-related" / "3000ps" / "orientation_acf_3000ps.csv",
    "KH570": ROOT / "KH570" / "cut-MSD" / "Orientation-related" / "orientation_acf_3000ps.csv",
    "KH560": ROOT / "KH560" / "cut-MSD" / "Orientation-related" / "orientation_acf_3000ps.csv",
    "KH550": ROOT / "KH550" / "cut-MSD" / "Orientation-related" / "orientation_acf_3000ps.csv",
}
# ---------------------------------------------------------------------------


def read_acf(path):
    t, cols = [], {"contact": [], "interface": [], "bulk": []}
    with open(path, encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            t.append(float(row["dt_ps"]))
            for L in LAYERS:
                cols[L].append(float(row[ACF_COL[L]]))
    return np.asarray(t), {L: np.asarray(cols[L]) for L in LAYERS}


def read_kww_summary():
    """读取已汇总的 KWW 参数 (orientation_compare_summary.csv)。"""
    path = SUMMARY / "orientation_compare_summary.csv"
    out = {}
    if not path.exists():
        raise FileNotFoundError(f"未找到 {path}")
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            layer, sys = row["layer"], row["system"]
            out[(layer, sys)] = row
    return out


def main():
    acf = {s: read_acf(ACF_FILES[s]) for s in ORDER}
    kww = read_kww_summary()

    # ---- 图 3a：C2(t) 衰减，每个区域一栏（半对数 x）----------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, L in zip(axes, LAYERS):
        for s in ORDER:
            t, cols = acf[s]
            ax.plot(t, cols[L], color=COLORS[s], lw=1.5, label=s)
        ax.axhline(1.0 / np.e, color="k", ls=":", lw=0.8, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(-0.05, 0.95)
        ax.set_xlabel("时间差 (ps)")
        ax.set_ylabel(r"$C_2(t)=\langle\frac{1}{2}(3\cos^2\theta-1)\rangle$")
        ax.set_title(f"{LAYER_ZH[L]} 区")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("FKM 键段取向自相关函数 C2(t) 四体系对比（虚线: 1/e）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig3a_orient_acf.png", dpi=300)
    plt.close(fig)

    # ---- 图 3b：平均取向弛豫时间 <tau>（对数）+ 平台值 C -----------------
    x = np.arange(len(LAYERS)); width = 0.19
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for si, s in enumerate(ORDER):
        vals = [float(kww[(L, s)]["tau_mean_ps"]) for L in LAYERS]
        ax.bar(x + (si - 1.5) * width, vals, width=width, color=COLORS[s],
               edgecolor="black", label=s)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"{L}\n{LAYER_ZH[L]}" for L in LAYERS])
    ax.set_ylabel(r"平均取向弛豫时间 $\langle\tau\rangle$ (ps)")
    ax.set_title(r"KWW 平均取向弛豫时间 $\langle\tau\rangle$（对数坐标）")
    ax.legend(); ax.grid(True, axis="y", which="both", alpha=0.3)

    ax = axes[1]
    for si, s in enumerate(ORDER):
        vals = [float(kww[(L, s)]["C_plateau"]) for L in LAYERS]
        ax.bar(x + (si - 1.5) * width, vals, width=width, color=COLORS[s],
               edgecolor="black", label=s)
    ax.set_xticks(x); ax.set_xticklabels([f"{L}\n{LAYER_ZH[L]}" for L in LAYERS])
    ax.set_ylabel(r"$C_2(t\to\infty)$ 平台值")
    ax.set_title("长期残余取向（平台值 C）")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("FKM 键段取向弛豫 KWW 参数对比", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig3b_orient_kww.png", dpi=300)
    plt.close(fig)

    print("已生成 03_取向动力学 的 2 张图到:", OUT)
    print("提示：当前为 3000 ps 数据，KH570/KH550 的 tau 为 extrapolated，"
          "7 ns 数据到位后请更新 ACF_FILES 再重出。")


if __name__ == "__main__":
    main()