#!/usr/bin/env python3
"""Time series of total interface H-bonds over the last 50 frames (151-200 ps)."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent
SYS = ["KH550", "KH560", "KH570"]
COLORS = {"KH550": "#457B9D", "KH560": "#FFD700", "KH570": "#E63946"}
FILES = {
    "KH550": r"F:\Lammps-MS\FKM-SIO2-TEST\KH550\H\hbond_last50.csv",
    "KH560": r"F:\Lammps-MS\FKM-SIO2-TEST\KH560\H\hbond_last50.csv",
    "KH570": r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\H\hbond_last50.csv",
}


def load(path):
    t, tot = [], []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            t.append(float(row["time_ps"]))
            tot.append(float(row["total_interface"]))
    return t, tot


def main():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for s in SYS:
        t, tot = load(FILES[s])
        ax.plot(t, tot, "-o", ms=3, lw=1.4, color=COLORS[s], label=s)
    ax.set_xlabel("时间 (ps)")
    ax.set_ylabel("总界面氢键数")
    ax.set_title("总界面氢键随时间变化（最后50帧, 3.0 Å / 120°）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "compare_hbond_total_timeseries.png", dpi=180)
    plt.close(fig)
    print("Saved compare_hbond_total_timeseries.png")


if __name__ == "__main__":
    main()