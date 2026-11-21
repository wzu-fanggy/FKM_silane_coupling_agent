#!/usr/bin/env python3
"""Compare FKM chain radius of gyration (Rg) across bare / KH550 / KH560 / KH570,
separated by region (contact / interface / bulk). Data: rg_by_frame_*.csv (200 frames)."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent
ORDER = ["bare", "KH570", "KH560", "KH550"]
COLORS = {"bare": "#90EE90", "KH550": "#457B9D", "KH560": "#FFD700", "KH570": "#E63946"}
REGIONS = ["contact", "interface", "bulk"]
REGION_ZH = {"contact": "接触层", "interface": "界面层", "bulk": "块体"}

FILES = {
    "bare":  r"F:\Lammps-MS\FKM-SIO2-TEST\RG\rg_by_frame.csv",
    "KH570": r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\RG\rg_by_frame_570.csv",
    "KH560": r"F:\Lammps-MS\FKM-SIO2-TEST\KH560\RG\rg_by_frame_560.csv",
    "KH550": r"F:\Lammps-MS\FKM-SIO2-TEST\KH550\RG\rg_by_frame_550.csv",
}


def load_region(path, region):
    """Return (time-mean Rg, time-mean std, list of per-frame means, list of frames)."""
    frames, means, stds = [], [], []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            frames.append(int(row["frame"]))
            means.append(float(row[f"{region}_Rg_mean"]))
            stds.append(float(row[f"{region}_Rg_std"]))
    return sum(means) / len(means), sum(stds) / len(stds), frames, means


def main():
    data = {s: {r: load_region(FILES[s], r) for r in REGIONS} for s in ORDER}

    # ---- Fig 1: grouped bars per region, 4 systems, error = frame-mean chain std ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, region in zip(axes, REGIONS):
        means = [data[s][region][0] for s in ORDER]
        errs = [data[s][region][1] for s in ORDER]
        x = range(len(ORDER))
        bars = ax.bar(x, means, yerr=errs, capsize=5,
                      color=[COLORS[s] for s in ORDER], edgecolor="black", width=0.6)
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m + max(errs) * 0.04,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(ORDER)
        ax.set_title(f"{region.capitalize()} / {REGION_ZH[region]} 区 Rg"
                     , fontsize=11)
        ax.set_ylabel("回转半径 Rg (Å)")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("FKM 链回转半径对比（不同区域，均值±链间标准差，200 帧平均）", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "compare_rg_region.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 2: timeseries (1x3), per region, 4 curves ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, region in zip(axes, REGIONS):
        for s in ORDER:
            freq = data[s][region]
            ax.plot(freq[2], freq[3], lw=1.2, color=COLORS[s], label=s)
        ax.set_title(f"{region.capitalize()}/{REGION_ZH[region]} Rg（随时间）")
        ax.set_xlabel("frame")
        ax.set_ylabel("Rg (Å)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("FKM 链回转半径随帧演化（500 ps）", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "compare_rg_timeseries.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---- summary CSV ----
    with open(OUT / "rg_compare_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region"] + ORDER + [f"{s}_chain_std" for s in ORDER])
        for region in REGIONS:
            row = [region]
            row += [f"{data[s][region][0]:.4f}" for s in ORDER]
            row += [f"{data[s][region][1]:.4f}" for s in ORDER]
            w.writerow(row)

    print("Region mean-Rg (Å) [chain_std], averaged over 200 frames:")
    for region in REGIONS:
        line = "  ".join(f"{s}={data[s][region][0]:.3f}({data[s][region][1]:.3f})" for s in ORDER)
        print(f"  {region:>9}: {line}")
    print("Saved compare_rg_region.png, compare_rg_timeseries.png, rg_compare_summary.csv")


if __name__ == "__main__":
    main()