#!/usr/bin/env python3
"""Compare FKM end-to-end distance (ete) across bare / KH550 / KH560 / KH570,
separated by region (contact / interface / bulk). Data: ete_multiframe_full.csv
(200 frames x ~120 chains)."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent
ORDER = ["bare", "KH570", "KH560", "KH550"]
COLORS = {"bare": "#90EE90", "KH550": "#457B9D", "KH560": "#FFD700", "KH570": "#E63946"}
LAYERS = ["contact", "interface", "bulk"]
LAYER_ZH = {"contact": "接触层", "interface": "界面层", "bulk": "块体"}

BASE = Path(r"F:\Lammps-MS\FKM-SIO2-TEST")
FILES = {
    "bare":  BASE / "End-to-end" / "ete_multiframe_full.csv",
    "KH570": BASE / "KH570" / "End-to-end" / "ete_multiframe_full.csv",
    "KH560": BASE / "KH560" / "End-to-end" / "ete_multiframe_full.csv",
    "KH550": BASE / "KH550" / "End-to-end" / "ete_multiframe_full.csv",
}


def load(path):
    """Return per-layer pooled etes and per-frame per-layer means."""
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
    per_frame_mean = {f: {l: sum(v) / len(v) for l, v in per_frame[f].items()} for f in frames}
    return pooled, frames, per_frame_mean


def stats(vals):
    n = len(vals)
    m = sum(vals) / n
    s = (sum((x - m) ** 2 for x in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, s


def main():
    dat = {s: load(FILES[s]) for s in ORDER}

    # ---- Fig 1: region bars ----
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
        ax.set_xticks(x); ax.set_xticklabels(ORDER)
        ax.set_title(f"{region} / {LAYER_ZH[region]}", fontsize=11)
        ax.set_ylabel("端到端距离 (Å)")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("FKM 链端到端距离对比（按区域，200帧×120链聚合，误差棒=链间σ）",
                 y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "compare_ete_region.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 2: timeseries ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, region in zip(axes, LAYERS):
        for s in ORDER:
            frames = dat[s][1]
            xs = [f for f in frames if region in dat[s][2][f]]
            ys = [dat[s][2][f][region] for f in xs]
            ax.plot(xs, ys, lw=1.2, color=COLORS[s], label=s)
        ax.set_title(f"{region}/{LAYER_ZH[region]} ete（随帧）")
        ax.set_xlabel("frame"); ax.set_ylabel("端到端距离 (Å)")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("FKM 端到端距离随帧演化（500 ps）", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "compare_ete_timeseries.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---- CSV ----
    with open(OUT / "ete_compare_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region"] + ORDER + [s + "_std" for s in ORDER])
        for region in LAYERS:
            means, stds = zip(*(stats(dat[s][0][region]) for s in ORDER))
            w.writerow([region] + [f"{m:.4f}" for m in means] + [f"{sd:.4f}" for sd in stds])

    print("End-to-end region mean (Å) [chain std] pooled over 200 frames x 120 chains:")
    for region in LAYERS:
        line = "  ".join(f"{s}={stats(dat[s][0][region])[0]:.3f}({stats(dat[s][0][region])[1]:.3f})" for s in ORDER)
        print(f"  {region:>9}: {line}")
    print("Saved compare_ete_region.png, compare_ete_timeseries.png, ete_compare_summary.csv")


if __name__ == "__main__":
    main()