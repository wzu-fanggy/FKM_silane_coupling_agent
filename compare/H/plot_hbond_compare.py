#!/usr/bin/env python3
"""Compare interfacial H-bond statistics across KH550 / KH560 / KH570.

Aligned categories (3.0 A / 120 deg, last-50-frame mean):
  A  : SiO2 silanol O-H  -> FKM F               (common)
  B1 : FKM C-H           -> SiO2 O              (common)
  B2 : FKM C-H           -> graft acceptor      (acceptor: O for KH560/570, N for KH550)
  D  : graft H           -> FKM F               (KH550 = amine N-H + alkyl C-H)
  E  : graft H           -> SiO2 O              (KH550 = amine N-H + alkyl C-H)
  F  : SiO2 silanol O-H  -> graft acceptor      (acceptor: O for KH560/570, N for KH550)
  total: A + B1 + B2 + D
Bare (no silane graft) has no H-bond categories and is excluded.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent
SYS = ["KH550", "KH560", "KH570"]
COLORS = {"KH550": "#457B9D", "KH560": "#FFD700", "KH570": "#E63946"}

# ---- raw per-system values (mean, std) parsed from hbond_last50_summary.txt ----
RAW = {
    "KH550": {
        "A": (137.24, 6.91),
        "B1": (92.48, 7.15),
        "B2": (5.80, 1.56),
        "D1": (14.38, 2.82), "D2": (101.38, 9.68),
        "E1": (2.22, 1.37), "E2": (33.00, 4.60),
        "F": (0.78, 0.74),
        "total": (351.28, 15.23),
    },
    "KH560": {
        "A": (134.92, 5.94),
        "B1": (95.42, 5.77),
        "B2": (9.86, 2.07),
        "D": (115.98, 8.11),
        "E": (37.30, 5.19),
        "F": (9.16, 1.86),
        "total": (356.18, 12.42),
    },
    "KH570": {
        "A": (146.14, 7.10),
        "B1": (95.92, 6.57),
        "B2": (17.10, 2.75),
        "D": (120.14, 9.43),
        "E": (30.02, 3.93),
        "F": (12.16, 1.71),
        "total": (379.30, 13.86),
    },
}


def sumkv(system, *keys):
    m = sum(RAW[system][k][0] for k in keys)
    s = math.sqrt(sum(RAW[system][k][1] ** 2 for k in keys))
    return m, s


def aligned(cat):
    out = {}
    for s in SYS:
        if cat in RAW[s]:
            out[s] = RAW[s][cat]
        elif cat == "D":
            out[s] = sumkv(s, "D1", "D2")
        elif cat == "E":
            out[s] = sumkv(s, "E1", "E2")
    return out


CATS = [
    ("A",  "A: SiO2 O-H(silanol) → FKM F", "硅羟基 → FKM 氟", "Å"),
    ("B1", "B1: FKM C-H → SiO2 O", "FKM 碳氢 → SiO2 氧", ""),
    ("B2", "B2: FKM C-H → 接枝受体", "FKM 碳氢 → 接枝受体 (560/570:O, 550:N)", ""),
    ("D",  "D: 接枝 H → FKM F", "接枝基团氢 → FKM 氟 (550=NH+CH)", ""),
    ("E",  "E: 接枝 H → SiO2 O", "接枝基团氢 → SiO2 氧 (550=NH+CH)", ""),
    ("F",  "F: SiO2 O-H → 接枝受体", "硅羟基 → 接枝受体 (560/570:O, 550:N)", ""),
    ("total", "total interface H-bonds", "总界面氢键", ""),
]

YLAB = "平均每帧氢键数 (3.0 Å / 120°)"


def main():
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    letters = list("abcdefg")
    flat = axes.flat
    for idx, (cat, en, zh, unit) in enumerate(CATS):
        ax = next(flat)
        vals = aligned(cat)
        means = [vals[s][0] for s in SYS]
        stds = [vals[s][1] for s in SYS]
        bars = ax.bar(SYS, means, yerr=stds, capsize=5,
                      color=[COLORS[s] for s in SYS], edgecolor="black", width=0.6)
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m + max(stds) * 0.04,
                    f"{m:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(0.02, 0.95, f"({letters[idx]})", transform=ax.transAxes,
                va="top", ha="left", fontsize=12, fontweight="bold")
        ax.set_title(f"{en}\n{zh}", fontsize=9)
        ax.set_ylabel(YLAB, fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    # hide leftover 8th subplot
    next(flat).axis("off")
    fig.suptitle("界面氢键对比（Interfacial H-bond comparison）— KH550 / KH560 / KH570",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "compare_hbond_metrics.png", dpi=180)
    plt.close(fig)

    # ---- CSV summary ----
    with open(OUT / "hbond_compare_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["category"] + SYS + [f"{s}_std" for s in SYS])
        for cat, en, zh, unit in CATS:
            vals = aligned(cat)
            row = [f"{cat} ({en}) (3.0A/120deg)"]
            row += [f"{vals[s][0]:.3f}" for s in SYS]
            row += [f"{vals[s][1]:.3f}" for s in SYS]
            w.writerow(row)

    print("=== aligned mean (std) ===")
    for cat, en, zh, unit in CATS:
        vals = aligned(cat)
        print(f"{cat:>7}: " + "  ".join(f"{s}={vals[s][0]:.1f}±{vals[s][1]:.1f}" for s in SYS))
    print("Saved compare_hbond_metrics.png, hbond_compare_summary.csv")


if __name__ == "__main__":
    main()