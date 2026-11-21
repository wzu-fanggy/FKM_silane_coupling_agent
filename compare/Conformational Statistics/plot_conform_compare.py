#!/usr/bin/env python3
"""Compare conformational statistics (bond length class, angle class, dihedral
T/G/C) across bare / KH550 / KH560 / KH570, by region (contact/interface/bulk).

Bond/angle types are remapped differently per system, so classes are formed by
the physical quantity (bond length / angle value) as in the per-system
analyze_conform_stats.py, making the comparison type-ID independent."""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent
ORDER = ["bare", "KH550", "KH560", "KH570"]
COLORS = {"bare": "#90EE90", "KH550": "#457B9D", "KH560": "#FFD700", "KH570": "#E63946"}
LAYERS = ["contact", "interface", "bulk"]
LAYER_ZH = {"contact": "接触层", "interface": "界面层", "bulk": "块体"}

BASE = Path(r"F:\Lammps-MS\FKM-SIO2-TEST")
SUMMARY = {
    "bare":  BASE / "Conformational Statistics" / "conform_stats_summary_fixed.txt",
    "KH550": BASE / "KH550" / "Conformational Statistics" / "conform_stats_summary_fixed.txt",
    "KH560": BASE / "KH560" / "Conformational Statistics" / "conform_stats_summary_fixed.txt",
    "KH570": BASE / "KH570" / "Conformational Statistics" / "conform_stats_summary_fixed.txt",
}
DIHEDRAL = {
    "bare":  BASE / "Conformational Statistics" / "dihedral_stats_results_NUMBA.txt",
    "KH550": BASE / "KH550" / "Conformational Statistics" / "dihedral_stats_results_NUMBA.txt",
    "KH560": BASE / "KH560" / "Conformational Statistics" / "dihedral_stats_results_NUMBA.txt",
    "KH570": BASE / "KH570" / "Conformational Statistics" / "dihedral_stats_results_NUMBA.txt",
}

ROW = re.compile(r"^\s*(\d+)\s+(contact|interface|bulk)\s+(\d+)\s+([\d.]+)\s+([\d.]+)")


def parse_summary(path):
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    layer_atoms = {}
    bonds, angles = [], []
    section = None
    for line in lines:
        s = line.strip()
        if s.startswith("Average Layer Atom Counts"):
            section = "atoms"; continue
        if s.startswith("BOND LENGTH STATISTICS"):
            section = "bonds"; continue
        if s.startswith("ANGLE STATISTICS"):
            section = "angles"; continue
        if section == "atoms":
            m = re.match(r"^\s*(\w+):\s+([\d.]+)\s+\+/-", s)
            if m:
                layer_atoms[m.group(1)] = float(m.group(2))
            continue
        if section in ("bonds", "angles"):
            m = ROW.match(line)
            if m:
                ident, layer, count, mean, std = m.groups()
                (bonds if section == "bonds" else angles).append(
                    (int(ident), layer, int(count), float(mean), float(std)))
    return layer_atoms, bonds, angles


def parse_dihedral(path):
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    out = {}
    for m in re.finditer(
            r"^\s*(contact|interface|bulk)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s*%",
            txt, re.M):
        layer, tr, ga, ci = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        total = tr + ga + ci
        out[layer] = (100.0 * tr / total, 100.0 * ga / total, 100.0 * ci / total)
    return out


def bond_class(mean):
    if mean < 1.20:
        return "C-H"
    if 1.35 <= mean <= 1.45:
        return "C-F / C=C"
    if mean > 1.45:
        return "C-C"
    return "other"


def angle_class(mean):
    if mean < 95:
        return "<95"
    if mean < 105:
        return "95-105"
    if mean < 120:
        return "105-120"
    return ">120"


def class_stats(recs, classifier):
    agg = {}
    for ident, layer, count, mean, std in recs:
        key = (classifier(mean), layer)
        e = agg.setdefault(key, [0, 0.0, 0.0])
        e[0] += count
        e[1] += mean * count
        e[2] += std * count
    return {k: (n, sm / n, ss / n) for k, (n, sm, ss) in agg.items()}


def grouped_bar(ax, systems, layers, fn, classes, ylabel, fmt=".3f", title=""):
    """fn(system, layer, cls) -> value (or None). bars: per layer group. x = classes."""
    n_cls = len(classes)
    x = np.arange(n_cls)
    width = 0.2
    colors = [COLORS[s] for s in systems]
    for li, layer in enumerate(layers):
        vals = [fn(system, layer, cls) for system in systems for cls in classes]
        return None  # placeholder, replaced below


def main():
    data = {s: parse_summary(SUMMARY[s]) for s in ORDER}
    layer_atoms = {s: data[s][0] for s in ORDER}
    bond_cls = {s: class_stats(data[s][1], bond_class) for s in ORDER}
    angle_cls = {s: class_stats(data[s][2], angle_class) for s in ORDER}
    dih = {s: parse_dihedral(DIHEDRAL[s]) for s in ORDER}

    BOND_CLASSES = ["C-H", "C-F / C=C", "C-C"]
    ANGLE_CLASSES = ["<95", "95-105", "105-120", ">120"]

    def bondval(s, layer, cls):
        return bond_cls[s].get((cls, layer))

    def angleval(s, layer, cls):
        return angle_cls[s].get((cls, layer))

    # ---- Fig 1: dihedral T/G/C stacked by layer, 4 systems ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, layer in zip(axes, LAYERS):
        tgc = np.array([dih[s][layer] for s in ORDER])
        x = np.arange(len(ORDER))
        b0 = ax.bar(x, tgc[:, 0], width=0.55, color=COLORS["KH570"], label="trans 反式")
        b1 = ax.bar(x, tgc[:, 1], width=0.55, bottom=tgc[:, 0], color=COLORS["KH560"], label="gauche 邻位交叉")
        b2 = ax.bar(x, tgc[:, 2], width=0.55, bottom=tgc[:, 0] + tgc[:, 1], color=COLORS["bare"], label="cis 顺式")
        for xx, t in zip(x, tgc[:, 0]):
            ax.text(xx, t + 0.6, f"{t:.1f}%", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(ORDER, fontsize=9)
        ax.set_ylabel("二面角占比 (%)")
        ax.set_title(f"{layer} / {LAYER_ZH[layer]}", fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", alpha=0.3)
        if layer == "bulk":
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("二面角 T/G/C 占比（按区域，4 体系）", y=1.03, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "compare_conform_dihedral.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 2: bond length class, per layer, 4 systems grouped ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, layer in zip(axes, LAYERS):
        x = np.arange(len(BOND_CLASSES)); width = 0.2
        maxerr = 0
        for si, s in enumerate(ORDER):
            means, errs = [], []
            for cls in BOND_CLASSES:
                v = bondval(s, layer, cls)
                means.append(v[1] if v else np.nan)
                errs.append(v[2] if v else 0)
                maxerr = max(maxerr, v[2] if v else 0)
            bars = ax.bar(x + (si - 1.5) * width, means, width=width,
                          yerr=errs, capsize=3, color=COLORS[s], edgecolor="black", label=s)
        ax.set_xticks(x); ax.set_xticklabels(BOND_CLASSES)
        ax.set_ylabel("键长 (Å)")
        ax.set_title(f"键长（{layer}/{LAYER_ZH[layer]}）")
        ax.grid(True, axis="y", alpha=0.3)
        if layer == "bulk":
            ax.legend(fontsize=8)
    fig.suptitle("键长分类均值（按区域，4 体系，误差棒=加权σ）", y=1.03, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "compare_conform_bond.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 3: bond angle class, per layer, 4 systems grouped ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, layer in zip(axes, LAYERS):
        x = np.arange(len(ANGLE_CLASSES)); width = 0.2
        for si, s in enumerate(ORDER):
            means, errs = [], []
            for cls in ANGLE_CLASSES:
                v = angleval(s, layer, cls)
                means.append(v[1] if v else np.nan)
                errs.append(v[2] if v else 0)
            ax.bar(x + (si - 1.5) * width, means, width=width,
                   yerr=errs, capsize=3, color=COLORS[s], edgecolor="black", label=s)
        ax.set_xticks(x); ax.set_xticklabels(ANGLE_CLASSES)
        ax.set_ylabel("键角 (°)")
        ax.set_title(f"键角（{layer}/{LAYER_ZH[layer]}）")
        ax.grid(True, axis="y", alpha=0.3)
        if layer == "bulk":
            ax.legend(fontsize=8)
    fig.suptitle("键角分类均值（按区域，4 体系，误差棒=加权σ）", y=1.03, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "compare_conform_angle.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---- summary CSV ----
    with open(OUT / "conform_compare_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "system", "layer", "class", "count", "mean", "std"])
        for s in ORDER:
            for layer in LAYERS:
                for cls in BOND_CLASSES:
                    v = bondval(s, layer, cls)
                    if v:
                        w.writerow(["bond_len", s, layer, cls, *v])
                for cls in ANGLE_CLASSES:
                    v = angleval(s, layer, cls)
                    if v:
                        w.writerow(["angle", s, layer, cls, *v])
                for tgc in ("trans", "gauche", "cis"):
                    vals = dih[s][layer]
                    w.writerow(["dihedral", s, layer, tgc,
                                "", f"{vals[{'trans':0,'gauche':1,'cis':2}[tgc]]:.3f}", ""])

    print("=== layer atom counts ===")
    for s in ORDER:
        print(f"  {s:>6}: " + "  ".join(f"{l}={layer_atoms[s].get(l,0):.0f}" for l in LAYERS))
    print("=== dihedral trans% by layer ===")
    for l in LAYERS:
        print(f"  {l:>9}: " + "  ".join(f"{s}={dih[s][l][0]:.1f}%" for s in ORDER))
    print("Saved: compare_conform_dihedral.png, compare_conform_bond.png, "
          "compare_conform_angle.png, conform_compare_summary.csv")


if __name__ == "__main__":
    main()