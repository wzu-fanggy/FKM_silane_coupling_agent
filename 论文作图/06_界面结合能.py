#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文配图 06 —— 界面相互作用能（每接触原子，含范德华/静电分量）。

数据源（E 盘真实路径）：
  汇总: {SYS}\static_interface_energy\outputs\interface_energy_normalized_summary.txt
  时序: {SYS}\static_interface_energy\outputs\interface_energy_normalized.csv

注意：KH570 系统层使用了不同截断 (R_cut=21.125 Å) 会高估每原子值；为跨体系可比，
  这里采用其公共截断 23.625 Å 的重算值（硬编码，见 KH570_COMMON）。
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

ORDER = ["bare", "KH550", "KH560", "KH570"]
COLORS = {"bare": "#90EE90", "KH550": "#457B9D", "KH560": "#FFD700", "KH570": "#E63946"}

SUMS = {
    "bare":  ROOT / "static_interface_energy" / "outputs" / "interface_energy_normalized_summary.txt",
    "KH550": ROOT / "KH550" / "static_interface_energy" / "outputs" / "interface_energy_normalized_summary.txt",
    "KH560": ROOT / "KH560" / "static_interface_energy" / "outputs" / "interface_energy_normalized_summary.txt",
    "KH570": ROOT / "KH570" / "static_interface_energy" / "outputs" / "interface_energy_normalized_summary.txt",
}
NORM = {
    "bare":  ROOT / "static_interface_energy" / "outputs" / "interface_energy_normalized.csv",
    "KH550": ROOT / "KH550" / "static_interface_energy" / "outputs" / "interface_energy_normalized.csv",
    "KH560": ROOT / "KH560" / "static_interface_energy" / "outputs" / "interface_energy_normalized.csv",
    "KH570": ROOT / "KH570" / "static_interface_energy" / "outputs" / "interface_energy_normalized.csv",
}
KEYS = ["E_total_per_contact_atom", "E_vdw_per_contact_atom", "E_ele_per_contact_atom"]
FULL = {"E_total_per_contact_atom": "总结合能", "E_vdw_per_contact_atom": "范德华",
        "E_ele_per_contact_atom": "静电"}
# KH570 公共截断 (23.625 Å) 重算值：(mean, std)
KH570_COMMON = {
    "E_total_per_contact_atom": (-0.409404, 0.0029633),
    "E_vdw_per_contact_atom": (-0.317942, 0.0058053),
    "E_ele_per_contact_atom": (-0.091461, 0.0053440),
}
# ---------------------------------------------------------------------------


def parse_summary(path):
    if not path.exists():
        raise FileNotFoundError(f"未找到 {path}")
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    pk = {}
    for k in KEYS:
        m = re.search(re.escape(k) + r"_mean\s*=\s*([\-\d.]+)", txt)
        ss = re.search(re.escape(k) + r"_std\s*=\s*([\-\d.]+)", txt)
        pk[k] = (float(m.group(1)), float(ss.group(1)))
    return pk


def main():
    vals = {s: parse_summary(SUMS[s]) for s in ORDER}
    vals["KH570"] = {k: KH570_COMMON[k] for k in KEYS}  # 用公共截断值作横跨对比

    x = np.arange(len(ORDER)); width = 0.25
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for i, k in enumerate(KEYS):
        means = [vals[s][k][0] for s in ORDER]
        errs = [vals[s][k][1] for s in ORDER]
        pos = [xx + (i - 1) * width for xx in x]
        ax.bar(pos, means, width=width, yerr=errs, capsize=4,
               color=[COLORS[s] for s in ORDER], edgecolor="black",
               label=f"{k.split('_')[1]} / {FULL[k]}")
    ax.set_xticks(x); ax.set_xticklabels(ORDER)
    ax.set_ylabel("每接触原子结合能 (kcal/mol·atom)")
    ax.set_title("界面结合能（per contact atom；KH570 采公共截断 23.625 Å 值）")
    ax.axhline(0, color="k", lw=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig6a_interface_energy.png", dpi=300)
    plt.close(fig)

    # ---- 图 6b：每接触原子总结合能时序 --------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for s in ORDER:
        tt, ee = [], []
        with open(NORM[s], encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                tt.append(float(row["time_ps"]))
                ee.append(float(row["E_total_per_contact_atom"]))
        ax.plot(tt, ee, "-o", ms=3, lw=1.2, color=COLORS[s], label=s)
    ax.set_xlabel("时间 (ps)")
    ax.set_ylabel("E_total per contact atom (kcal/mol·atom)")
    ax.set_title("界面结合能（每接触原子）随帧演化")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig6b_interface_energy_timeseries.png", dpi=300)
    plt.close(fig)

    # ---- 汇总 CSV ----------------------------------------------------
    with open(OUT / "fig6_interface_energy_summary.csv", "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity"] + ORDER + [s + "_std" for s in ORDER])
        for k in KEYS:
            w.writerow([k] + [f"{vals[s][k][0]:.5f}" for s in ORDER]
                       + [f"{vals[s][k][1]:.5f}" for s in ORDER])

    print("每接触原子界面能 (kcal/mol·atom) [KH570 公共截断]:")
    for k in KEYS:
        print(f"  {k:>28}: " + "  ".join(
            f"{s}={vals[s][k][0]:.4f}±{vals[s][k][1]:.4f}" for s in ORDER))
    print("已生成 06_界面结合能 的 2 张图 + summary CSV 到:", OUT)


if __name__ == "__main__":
    main()