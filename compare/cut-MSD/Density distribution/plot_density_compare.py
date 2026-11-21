#!/usr/bin/env python3
"""Compare radial mass-density distribution (core/shell/matrix) across bare /
KH550 / KH560 / KH570. Data: radial_density_data_{*.txt} (isolated cluster in
vacuum; total mass density of SiO2 + graft + FKM). Zone mean densities parsed
from the txt headers."""

import csv, re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent
ORDER = ["bare", "KH570", "KH560", "KH550"]
COLORS = {"bare": "#90EE90", "KH550": "#457B9D", "KH560": "#FFD700", "KH570": "#E63946"}
ZONES = ["Core", "Contact", "Interface", "Bulk"]
ZONE_ZH = {"Core": "核心", "Contact": "接触层", "Interface": "界面层", "Bulk": "块体"}

BASE = Path(r"F:\Lammps-MS\FKM-SIO2-TEST")
FILES = {
    "bare":  BASE / "cut-MSD" / "Density distribution" / "radial_density_data_bare.txt",
    "KH570": BASE / "KH570" / "cut-MSD" / "Density distribution" / "radial_density_data_kh570.txt",
    "KH560": BASE / "KH560" / "cut-MSD" / "Density distribution" / "radial_density_data_kh560.txt",
    "KH550": BASE / "KH550" / "cut-MSD" / "Density distribution" / "radial_density_data_kh550.txt",
}


def parse(path):
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    zones = {}
    r, raw, sm = [], [], []
    for line in txt.splitlines():
        s = line.strip()
        m = re.match(r"# (Core|Contact|Interface|Bulk).*?R=\[([\d.]+),([\d.]+)\].*?rho=([\d.]+).*?atoms=(\d+)", s)
        if m:
            zones[m.group(1)] = (float(m.group(2)), float(m.group(3)),
                                 float(m.group(4)), int(m.group(5)))
            continue
        if s and not s.startswith("#"):
            p = s.split()
            if len(p) == 3:
                r.append(float(p[0])); raw.append(float(p[1])); sm.append(float(p[2]))
    return zones, r, raw, sm


def main():
    dat = {s: parse(FILES[s]) for s in ORDER}

    # Fig 1: total radial mass density (smoothed) overlay
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for s in ORDER:
        zones, r, raw, sm = dat[s]
        ax.plot(r, sm, color=COLORS[s], lw=1.7, label=s)
    ax.axvline(15.05, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"距离料/接枝簇质心 (Å)")
    ax.set_ylabel("总质量密度 (g/cm³)")
    ax.set_title("径向质量密度分布（总，核-壳-基体；整个孤立簇,真空环境）")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT / "compare_total_radial_density.png", dpi=180)
    plt.close(fig)

    # Fig 2: zone mean density bars
    fig, ax = plt.subplots(figsize=(10, 5.6))
    import numpy as np
    x = np.arange(len(ZONES)); width = 0.2
    for si, s in enumerate(ORDER):
        zones = dat[s][0]
        means = [zones[z][2] for z in ZONES]
        ax.bar(x + (si - 1.5) * width, means, width=width,
               color=COLORS[s], edgecolor="black", label=s)
    ax.set_xticks(x); ax.set_xticklabels([f"{z}\n{ZONE_ZH[z]}" for z in ZONES])
    ax.set_ylabel("分区平均密度 (g/cm³)")
    ax.set_title("分区平均质量密度（Core/Contact/Interface/Bulk；分区边界随体系不同,见文档）")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "compare_zone_density.png", dpi=180)
    plt.close(fig)

    # CSV
    with open(OUT / "density_compare_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["zone"] + ORDER)
        for z in ZONES:
            w.writerow([z] + [f"{dat[s][0][z][2]:.4f}" for s in ORDER])

    print("Zone mean total density (g/cm3) + [zone radial range A]:")
    for s in ORDER:
        print(f"  {s:>6}: " + "  ".join(f"{z}={dat[s][0][z][2]:.4f}[{dat[s][0][z][0]:.2f}-{dat[s][0][z][1]:.2f}]" for z in ZONES))
    print("Saved compare_total_radial_density.png, compare_zone_density.png, "
          "density_compare_summary.csv")


if __name__ == "__main__":
    main()