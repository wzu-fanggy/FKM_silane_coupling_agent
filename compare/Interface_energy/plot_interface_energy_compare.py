#!/usr/bin/env python3
"""Compare interfacial (binding) energy per contact atom across the 4 systems.

Compatibility caveat:
  bare / KH550 / KH560 use their system R_cut ~23.375-23.625 A (delta_pert based).
  KH570 system-layer uses R_cut = 21.125 A, which inflates per-atom values
  (-0.711). A COMMON-cutoff rerun of KH570 at 23.625 A gives -0.409, which is
  directly comparable. We therefore plot the comparable common-cutoff KH570 value
  and flag the system-layer one."""

import csv, re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent
ORDER = ["bare", "KH550", "KH560", "KH570"]
COLORS = {"bare": "#90EE90", "KH550": "#457B9D", "KH560": "#FFD700", "KH570": "#E63946"}

# from each interface_energy_normalized_summary.txt
KEYS = ["E_total_per_contact_atom", "E_vdw_per_contact_atom", "E_ele_per_contact_atom"]
SUMS = {
    "bare":  r"F:\Lammps-MS\FKM-SIO2-TEST\static_interface_energy\outputs\interface_energy_normalized_summary.txt",
    "KH550": r"F:\Lammps-MS\FKM-SIO2-TEST\KH550\static_interface_energy\outputs\interface_energy_normalized_summary.txt",
    "KH560": r"F:\Lammps-MS\FKM-SIO2-TEST\KH560\static_interface_energy\outputs\interface_energy_normalized_summary.txt",
    "KH570": r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\static_interface_energy\outputs\interface_energy_normalized_summary.txt",
}
# KH570 common-cutoff rerun (R_cut 23.625) from KH570/static_interface_energy/compare
KH570_COMMON = {
    "E_total_per_contact_atom": (-0.409404, 0.0029633),
    "E_vdw_per_contact_atom": (-0.317942, 0.0058053),
    "E_ele_per_contact_atom": (-0.091461, 0.0053440),
}


def parse_summary(path):
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    pk = {}
    for k in KEYS:
        m = re.search(re.escape(k) + r"_mean\s*=\s*([\-\d.]+)", txt)
        ss = re.search(re.escape(k) + r"_std\s*=\s*([\-\d.]+)", txt)
        pk[k] = (float(m.group(1)), float(ss.group(1)))
    return pk


def main():
    vals = {s: parse_summary(SUMS[s]) for s in ORDER}
    vals["KH570"] = {k: KH570_COMMON[k] for k in KEYS}  # use common-cutoff for comparability

    labels = KEYS
    full = {"E_total_per_contact_atom": "total 总结合能",
            "E_vdw_per_contact_atom": "vdW 范德华",
            "E_ele_per_contact_atom": "electrostatic 静电"}
    short = {"E_total_per_contact_atom": "E_total",
             "E_vdw_per_contact_atom": "E_vdw",
             "E_ele_per_contact_atom": "E_ele"}
    x = range(len(ORDER)); width = 0.25

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for i, k in enumerate(labels):
        means = [vals[s][k][0] for s in ORDER]
        errs = [vals[s][k][1] for s in ORDER]
        pos = [xx + (i - 1) * width for xx in x]
        ax.bar(pos, means, width=width, yerr=errs, capsize=4,
               color=[COLORS[s] for s in ORDER], edgecolor="black",
               label=f"{short[k]} / {full[k]}")
    ax.set_xticks([xx + 0.0 for xx in x]); ax.set_xticklabels(ORDER)
    ax.set_ylabel("每接触原子结合能 (kcal/mol·atom)")
    ax.set_title("界面结合能（per contact atom；KH570 采公共截断 23.625 Å 值）")
    ax.axhline(0, color="k", lw=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "compare_interface_energy.png", dpi=180)
    plt.close(fig)

    # timeseries of E_total per atom (from normalized.csv per system)
    import numpy as np
    NORM = {
        "bare":  r"F:\Lammps-MS\FKM-SIO2-TEST\static_interface_energy\outputs\interface_energy_normalized.csv",
        "KH550": r"F:\Lammps-MS\FKM-SIO2-TEST\KH550\static_interface_energy\outputs\interface_energy_normalized.csv",
        "KH560": r"F:\Lammps-MS\FKM-SIO2-TEST\KH560\static_interface_energy\outputs\interface_energy_normalized.csv",
        "KH570": r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\static_interface_energy\outputs\interface_energy_normalized.csv",
    }
    fig2, ax2 = plt.subplots(figsize=(9.5, 5))
    for s in ORDER:
        tt, ee = [], []
        with open(NORM[s], encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                tt.append(float(row["time_ps"]))
                ee.append(float(row["E_total_per_contact_atom"]))
        ax2.plot(tt, ee, "-o", ms=3, lw=1.2, color=COLORS[s], label=s)
    ax2.set_xlabel("时间 (ps)")
    ax2.set_ylabel("E_total per contact atom (kcal/mol·atom)")
    ax2.set_title("界面结合能（每接触原子）随帧演化（最后 50 帧）")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT / "compare_interface_energy_timeseries.png", dpi=180)
    plt.close(fig2)

    # summary CSV
    with open(OUT / "interface_energy_compare_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity"] + ORDER + [s + "_std" for s in ORDER])
        for k in KEYS:
            w.writerow([k] + [f"{vals[s][k][0]:.5f}" for s in ORDER]
                       + [f"{vals[s][k][1]:.5f}" for s in ORDER])

    print("per-contact-atom interface energy (kcal/mol·atom) [common-cutoff KH570]:")
    for k in KEYS:
        print(f"  {k:>28}: " + "  ".join(f"{s}={vals[s][k][0]:.4f}±{vals[s][k][1]:.4f}" for s in ORDER))
    print("Saved compare_interface_energy.png, compare_interface_energy_timeseries.png, "
          "interface_energy_compare_summary.csv")


if __name__ == "__main__":
    main()