#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare segmental orientation ACF (C2(t)) and KWW relaxation across the
four FKM/SiO2 systems (bare / KH550 / KH560 / KH570).

C2(t) = <1.5*(u(t)*u(0))^2 - 0.5> for FKM bond unit vectors, layer-resolved
(contact / interface / bulk).  ACF data pre-computed by each system's
Numba_3000ps_*.py and stored in orientation_acf_3000ps.csv; KWW fits stored
in kww_fitting_results.csv.  This script only reads and compares.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent
BASE = Path(r"F:\Lammps-MS\FKM-SIO2-TEST")

ORDER = ["bare", "KH570", "KH560", "KH550"]
COLORS = {"bare": "#90EE90", "KH550": "#457B9D", "KH560": "#FFD700", "KH570": "#E63946"}

# layer-boundary parameters (R_SURF A, DELTA_PERT A) from each Numba script
PARAMS = {
    "bare":  (15.10, 8.60),
    "KH550": (15.083, 8.292),
    "KH560": (15.095, 8.28),
    "KH570": (15.05, 6.08),
}

ACF_FILES = {
    "bare":  BASE / "cut-MSD" / "Orientation-related" / "3000ps" / "orientation_acf_3000ps.csv",
    "KH550": BASE / "KH550" / "cut-MSD" / "Orientation-related" / "orientation_acf_3000ps.csv",
    "KH560": BASE / "KH560" / "cut-MSD" / "Orientation-related" / "orientation_acf_3000ps.csv",
    "KH570": BASE / "KH570" / "cut-MSD" / "Orientation-related" / "orientation_acf_3000ps.csv",
}
KWW_FILES = {
    "bare":  BASE / "cut-MSD" / "Orientation-related" / "3000ps" / "kww_fitting_results.csv",
    "KH550": BASE / "KH550" / "cut-MSD" / "Orientation-related" / "kww_fitting_results.csv",
    "KH560": BASE / "KH560" / "cut-MSD" / "Orientation-related" / "kww_fitting_results.csv",
    "KH570": BASE / "KH570" / "cut-MSD" / "Orientation-related" / "kww_fitting_results.csv",
}

LAYERS = ["contact", "interface", "bulk"]
LAYER_ZH = {"contact": "接触层", "interface": "界面层", "bulk": "块体"}
ACF_COL = {"contact": "contact_C2", "interface": "interface_C2", "bulk": "bulk_C2"}


def read_acf(path):
    t, cols = [], {"contact": [], "interface": [], "bulk": []}
    with open(path, encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            t.append(float(row["dt_ps"]))
            for L in LAYERS:
                cols[L].append(float(row[ACF_COL[L]]))
    return np.asarray(t), {L: np.asarray(cols[L]) for L in LAYERS}


def read_kww(path):
    k = {"contact": {}, "interface": {}, "bulk": {}}
    key_map = {"Contact Layer": "contact", "Interface Layer": "interface", "Bulk": "bulk"}
    with open(path, encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            L = key_map[row["Region"]]
            k[L] = {fld: row[fld] for fld in
                    ("A", "tau_KWW_ps", "beta", "C_plateau", "tau_mean_ps",
                     "tau_mean_ns", "tau_data_1e_ps", "R_squared", "Note")}
            for num in ("A", "tau_KWW_ps", "beta", "C_plateau", "tau_mean_ps",
                        "tau_mean_ns", "R_squared"):
                try:
                    k[L][num] = float(k[L][num])
                except ValueError:
                    k[L][num] = float("nan")
    return k


def main():
    acf = {s: read_acf(ACF_FILES[s]) for s in ORDER}
    kww = {s: read_kww(KWW_FILES[s]) for s in ORDER}

    # ---- Fig 1: ACF decay, one panel per layer (semilog x) ----
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
        ax.set_title(f"{LAYER_ZH[L]} (Contact/Interface/Bulk)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("FKM 键段取向自相关函数 C2(t) 四体系对比（3000 ps NVT；虚线:1/e）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "orient_acf_compare_3000ps.png", dpi=180)
    plt.close(fig)

    # ---- Fig 2: tau_mean + C_plateau grouped bars ----
    x = np.arange(len(LAYERS))
    width = 0.19
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for si, s in enumerate(ORDER):
        vals = [kww[s][L]["tau_mean_ps"] for L in LAYERS]
        ax.bar(x + (si - 1.5) * width, vals, width=width, color=COLORS[s],
               edgecolor="black", label=s)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{L}\n{LAYER_ZH[L]}" for L in LAYERS])
    ax.set_ylabel(r"平均取向弛豫时间 $\langle\tau\rangle$ (ps)")
    ax.set_title(r"KWW 平均取向弛豫时间 $\langle\tau\rangle$（对数坐标）")
    ax.legend(); ax.grid(True, axis="y", which="both", alpha=0.3)

    ax = axes[1]
    for si, s in enumerate(ORDER):
        vals = [kww[s][L]["C_plateau"] for L in LAYERS]
        ax.bar(x + (si - 1.5) * width, vals, width=width, color=COLORS[s],
               edgecolor="black", label=s)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{L}\n{LAYER_ZH[L]}" for L in LAYERS])
    ax.set_ylabel(r"$C_2(t\to\infty)$ 平台值")
    ax.set_title("长期残余取向（平台值 C）")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("FKM 键段取向弛豫 KWW 参数对比", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "orient_kww_compare_3000ps.png", dpi=180)
    plt.close(fig)

    # ---- CSV summary ----
    rows = []
    for L in LAYERS:
        for s in ORDER:
            rs, g = PARAMS[s], kww[s][L]
            rows.append({
                "layer": L, "system": s,
                "R_surf_contact_A": rs[0], "delta_pert_A": rs[1],
                "tau_KWW_ps": g["tau_KWW_ps"], "beta": g["beta"],
                "C_plateau": g["C_plateau"],
                "tau_mean_ps": g["tau_mean_ps"], "tau_mean_ns": g["tau_mean_ns"],
                "tau_data_1e_ps": g["tau_data_1e_ps"],
                "R_squared": g["R_squared"], "note": g["Note"],
            })
    with open(OUT / "orientation_compare_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- console table ----
    print("KWW mean orientation relaxation tau_mean (ps) [and C_plateau]:")
    for L in LAYERS:
        print(f"\n {L}:")
        for s in ORDER:
            g = kww[s][L]
            print(f"   {s:>6}: <tau>={g['tau_mean_ps']:>10.1f} ps   beta={g['beta']:.3f}  "
                  f"C={g['C_plateau']:.4f}  {g['Note']}")
    print("\nSaved orient_acf_compare_3000ps.png, orient_kww_compare_3000ps.png, "
          "orientation_compare_summary.csv")


if __name__ == "__main__":
    main()