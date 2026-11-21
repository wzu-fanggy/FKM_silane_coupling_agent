#!/usr/bin/env python3
"""Bar-chart comparison of the interface-layer key metrics across the 4 systems,
parsed from each system's RDF interface_layer_summary.txt."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent

SUMMARIES = {
    "bare":  Path(r"F:\Lammps-MS\FKM-SIO2-TEST\RDF\outputs\interface_layer\interface_layer_summary.txt"),
    "KH570": Path(r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\RDF\outputs\interface_layer\interface_layer_summary.txt"),
    "KH560": Path(r"F:\Lammps-MS\FKM-SIO2-TEST\KH560\RDF\outputs\interface_layer\interface_layer_summary.txt"),
    "KH550": Path(r"F:\Lammps-MS\FKM-SIO2-TEST\KH550\RDF\outputs\interface_layer\interface_layer_summary.txt"),
}
ORDER = ["bare", "KH570", "KH560", "KH550"]
COLORS = {"bare": "#90EE90", "KH550": "#457B9D", "KH560": "#FFD700", "KH570": "#E63946"}


def parse(path):
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    def num(key):
        m = re.search(re.escape(key) + r".*?[=:]\s*([-\d.]+)", txt)
        return float(m.group(1)) if m else float("nan")
    return {
        "R_surf": num("R_surf"),
        "delta": num("interface thickness delta"),
        "delta_pert": num("delta_pert"),
        "bulk_mass": num("FKM bulk mass density"),
        "bulk_num": num("FKM bulk number density"),
        "peak_r": num("first FKM density peak r"),
        "peak_rho": num("first FKM density peak r"),
        "rho_core": num("rho_core_SiO2"),
    }


def main():
    M = {n: parse(SUMMARIES[n]) for n in ORDER}
    labels = ORDER
    x = np.arange(len(labels))

    # peak_rho needs the value after ", rho =" of the peak line
    def peak_rho(path):
        txt = Path(path).read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"first FKM density peak r = [\d.]+ A, rho = ([\d.]+)", txt)
        return float(m.group(1)) if m else float("nan")
    peak_rho_vals = {n: peak_rho(SUMMARIES[n]) for n in ORDER}

    def bar(ax, key, ylabel, title, fmt=".3f", rot=False):
        vals = [M[n][key] for n in ORDER]
        bars = ax.bar(x, vals, color=[COLORS[n] for n in ORDER], edgecolor="black", width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:{fmt}}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    metrics = {
        "R_surf":        (r"$R_{\rm surf}$（填料表面半径） ($\AA$)", "R_surf — 填料表面半径", ".3f"),
        "delta":         (r"$\delta$（界面厚度） ($\AA$)", "δ — 界面厚度", ".3f"),
        "delta_pert":    (r"$\delta_{\rm pert}$（扰动层范围） ($\AA$)", "δ_pert — 扰动层范围", ".3f"),
        "bulk_mass":     ("g/cm$^3$", "FKM bulk mass density — FKM 块体密度", ".4f"),
        "peak_r":        (r"$r_{\rm peak}$（首峰位置） ($\AA$)", "r_peak — FKM 密度首峰位置", ".3f"),
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    letters = list("abcdef")
    for (key, (ylab, title, fmt)), ax, L in zip(metrics.items(), axes.flat, letters):
        bar(ax, key, ylab, title, fmt)
        ax.text(0.02, 0.95, f"({L})", transform=ax.transAxes, va="top", ha="left",
                fontsize=12, fontweight="bold")
    # last subplot: first peak height rho
    axp = axes[1, 2]
    L = "f"
    axp.text(0.02, 0.95, f"({L})", transform=axp.transAxes, va="top", ha="left",
             fontsize=12, fontweight="bold")
    vals = [peak_rho_vals[n] for n in ORDER]
    bars = axp.bar(x, vals, color=[COLORS[n] for n in ORDER], edgecolor="black", width=0.6)
    for b, v in zip(bars, vals):
        axp.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    axp.set_xticks(x); axp.set_xticklabels(labels)
    axp.set_ylabel("atoms/$\\AA^3$")
    axp.set_title("$\\rho_{peak}$（首峰高度）— First FKM peak height")
    axp.grid(True, axis="y", alpha=0.3)

    fig.suptitle("界面关键参数对比（Interface-layer key metrics）— 4 systems, RDF analysis",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "compare_interface_metrics.png", dpi=180)
    plt.close(fig)

    for n in ORDER:
        print(f"{n:>5}: R_surf={M[n]['R_surf']:.3f}  delta={M[n]['delta']:.3f}  "
              f"delta_pert={M[n]['delta_pert']:.3f}  bulk_mass={M[n]['bulk_mass']:.4f}  "
              f"peak_r={M[n]['peak_r']:.3f}  peak_rho={peak_rho_vals[n]:.4f}")
    print("Saved compare_interface_metrics.png")


if __name__ == "__main__":
    main()