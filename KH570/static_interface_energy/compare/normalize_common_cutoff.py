#!/usr/bin/env python3
"""KH570 vs bare SiO2 interface energy at a common contact-atom cutoff.

The common cutoff is the bare-SiO2 perturbed-layer boundary (R_cut = 23.625 A),
so both systems count FKM atoms in the same spherical shell. This isolates the
"average binding strength per contact atom" comparison from the different
system-specific perturbed-layer thicknesses.

For KH570 the system-specific layer result is also kept in the CSV so the two
definitions can be compared side by side.
"""

import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


WORK = Path(__file__).resolve().parent
COMMON_R_CUT = 23.625

BARE_DUMP = Path(r"F:\Lammps-MS\FKM-SIO2-TEST\static_interface_energy\outputs\last50.dump")
BARE_ENERGY = Path(r"F:\Lammps-MS\FKM-SIO2-TEST\static_interface_energy\outputs\interface_energy_last50.csv")
KH_DUMP = Path(r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\static_interface_energy\outputs\last50.dump")
KH_ENERGY = Path(r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\static_interface_energy\outputs\interface_energy_last50.csv")

OUT_KH_CSV = WORK / "kh570_common_cutoff.csv"
OUT_BARE_CSV = WORK / "bare_common_cutoff.csv"
OUT_COMPARE_CSV = WORK / "compare_common_cutoff_summary.csv"
OUT_PNG = WORK / "compare_common_cutoff.png"

# Atom types.
BARE_FKM = {1, 2, 3, 4, 5, 6}
BARE_SIO2 = {7, 8, 9}
KH_FKM = {1, 2, 3, 4, 8, 10}
KH_SIO2 = {5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17}


def read_dump(path):
    frames = []
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(f.readline().strip())
            f.readline()  # ITEM: NUMBER OF ATOMS
            natoms = int(f.readline().strip())
            f.readline()  # ITEM: BOX BOUNDS
            lo = np.empty(3)
            hi = np.empty(3)
            for i in range(3):
                lo[i], hi[i] = map(float, f.readline().split())
            f.readline()  # ITEM: ATOMS
            types = np.empty(natoms, dtype=np.int64)
            pos = np.empty((natoms, 3))
            for i in range(natoms):
                parts = f.readline().split()
                types[i] = int(parts[1])
                pos[i] = [float(parts[2]), float(parts[3]), float(parts[4])]
            frames.append((step, lo, hi, types, pos))
    return frames


def minimum_image(delta, span):
    return delta - np.round(delta / span) * span


def contact_counts(frames, fkm_types, sio2_types, r_cut):
    counts = []
    for step, lo, hi, types, pos in frames:
        box = np.vstack((lo, hi))
        span = (box[1] - box[0])[:, None]
        is_fkm = np.isin(types, list(fkm_types))
        is_sio2 = np.isin(types, list(sio2_types))
        center = pos[is_sio2].mean(axis=0)
        dvec = minimum_image((pos[is_fkm] - center).T, span).T
        counts.append(int(np.count_nonzero(np.linalg.norm(dvec, axis=1) <= r_cut)))
    return counts


def read_energy(path):
    energy = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            energy[int(row["step"])] = {
                "E_total": float(row["E_total_kcal_mol"]),
                "E_vdw": float(row["E_vdw_kcal_mol"]),
                "E_ele": float(row["E_ele_kcal_mol"]),
            }
    return energy


def build_rows(frames, energy, fkm_types, sio2_types, r_cut):
    counts = contact_counts(frames, fkm_types, sio2_types, r_cut)
    rows = []
    for (step, _, _, _, _), n in zip(frames, counts):
        e = energy[step]
        rows.append(
            {
                "step": step,
                "time_ps": step * 0.001,
                "N_contact_atoms": n,
                "E_total_kcal_mol": e["E_total"],
                "E_vdw_kcal_mol": e["E_vdw"],
                "E_ele_kcal_mol": e["E_ele"],
                "E_total_per_contact_atom": e["E_total"] / n,
                "E_vdw_per_contact_atom": e["E_vdw"] / n,
                "E_ele_per_contact_atom": e["E_ele"] / n,
            }
        )
    return rows


def summarize(rows):
    n = [r["N_contact_atoms"] for r in rows]
    et = [r["E_total_kcal_mol"] for r in rows]
    etp = [r["E_total_per_contact_atom"] for r in rows]
    evp = [r["E_vdw_per_contact_atom"] for r in rows]
    eep = [r["E_ele_per_contact_atom"] for r in rows]
    return {
        "N_contact_mean": statistics.mean(n),
        "N_contact_std": statistics.stdev(n),
        "E_total_mean": statistics.mean(et),
        "E_total_std": statistics.stdev(et),
        "E_total_per_atom_mean": statistics.mean(etp),
        "E_total_per_atom_std": statistics.stdev(etp),
        "E_vdw_per_atom_mean": statistics.mean(evp),
        "E_vdw_per_atom_std": statistics.stdev(evp),
        "E_ele_per_atom_mean": statistics.mean(eep),
        "E_ele_per_atom_std": statistics.stdev(eep),
    }


def main():
    bare_frames = read_dump(BARE_DUMP)
    kh_frames = read_dump(KH_DUMP)
    bare_e = read_energy(BARE_ENERGY)
    kh_e = read_energy(KH_ENERGY)

    bare_rows = build_rows(bare_frames, bare_e, BARE_FKM, BARE_SIO2, COMMON_R_CUT)
    kh_rows = build_rows(kh_frames, kh_e, KH_FKM, KH_SIO2, COMMON_R_CUT)

    for path, rows in ((OUT_BARE_CSV, bare_rows), (OUT_KH_CSV, kh_rows)):
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    bare_s = summarize(bare_rows)
    kh_s = summarize(kh_rows)
    kh_sys = summarize(build_rows(kh_frames, kh_e, KH_FKM, KH_SIO2, 21.125))

    summary_rows = [
        {
            "system": "bare_SiO2",
            "r_cut_A": COMMON_R_CUT,
            "N_contact_mean": bare_s["N_contact_mean"],
            "N_contact_std": bare_s["N_contact_std"],
            "E_total_mean": bare_s["E_total_mean"],
            "E_total_std": bare_s["E_total_std"],
            "E_total_per_atom_mean": bare_s["E_total_per_atom_mean"],
            "E_total_per_atom_std": bare_s["E_total_per_atom_std"],
            "E_vdw_per_atom_mean": bare_s["E_vdw_per_atom_mean"],
            "E_ele_per_atom_mean": bare_s["E_ele_per_atom_mean"],
        },
        {
            "system": "KH570_common_cutoff",
            "r_cut_A": COMMON_R_CUT,
            "N_contact_mean": kh_s["N_contact_mean"],
            "N_contact_std": kh_s["N_contact_std"],
            "E_total_mean": kh_s["E_total_mean"],
            "E_total_std": kh_s["E_total_std"],
            "E_total_per_atom_mean": kh_s["E_total_per_atom_mean"],
            "E_total_per_atom_std": kh_s["E_total_per_atom_std"],
            "E_vdw_per_atom_mean": kh_s["E_vdw_per_atom_mean"],
            "E_ele_per_atom_mean": kh_s["E_ele_per_atom_mean"],
        },
        {
            "system": "KH570_system_layer_21.125",
            "r_cut_A": 21.125,
            "N_contact_mean": kh_sys["N_contact_mean"],
            "N_contact_std": kh_sys["N_contact_std"],
            "E_total_mean": kh_sys["E_total_mean"],
            "E_total_std": kh_sys["E_total_std"],
            "E_total_per_atom_mean": kh_sys["E_total_per_atom_mean"],
            "E_total_per_atom_std": kh_sys["E_total_per_atom_std"],
            "E_vdw_per_atom_mean": kh_sys["E_vdw_per_atom_mean"],
            "E_ele_per_atom_mean": kh_sys["E_ele_per_atom_mean"],
        },
    ]
    with open(OUT_COMPARE_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(WORK / "compare_common_cutoff.txt", "w", encoding="utf-8-sig") as f:
        f.write("Interface energy at common cutoff R_cut = 23.625 A\n")
        f.write("All values kcal/mol unless noted; errors are 1 SD over 50 frames.\n\n")
        for r in summary_rows:
            f.write(f"{r['system']}: R_cut={r['r_cut_A']:.3f} A, "
                    f"N_contact={r['N_contact_mean']:.1f}+/-{r['N_contact_std']:.1f}, "
                    f"E_total={r['E_total_mean']:.1f}+/-{r['E_total_std']:.1f}, "
                    f"E_total/atom={r['E_total_per_atom_mean']:.4f}+/-{r['E_total_per_atom_std']:.4f}, "
                    f"E_vdw/atom={r['E_vdw_per_atom_mean']:.4f}, "
                    f"E_ele/atom={r['E_ele_per_atom_mean']:.4f}\n")

    labels = ["Bare", "KH570\ncommon cutoff", "KH570\nown layer"]
    n_vals = [bare_s["N_contact_mean"], kh_s["N_contact_mean"], kh_sys["N_contact_mean"]]
    e_vals = [bare_s["E_total_per_atom_mean"], kh_s["E_total_per_atom_mean"], kh_sys["E_total_per_atom_mean"]]
    e_errs = [bare_s["E_total_per_atom_std"], kh_s["E_total_per_atom_std"], kh_sys["E_total_per_atom_std"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(labels, n_vals, color=["#4C72B0", "#DD8452", "#55A868"])
    axes[0].set_ylabel("N_contact atoms")
    axes[0].set_title("Contact atoms (R_cut = 23.625 A)")
    for i, v in enumerate(n_vals):
        axes[0].text(i, v + 30, f"{v:.0f}", ha="center")

    axes[1].bar(labels, e_vals, yerr=e_errs, capsize=5, color=["#4C72B0", "#DD8452", "#55A868"])
    axes[1].set_ylabel("E_total / N_contact (kcal/mol/atom)")
    axes[1].set_title("Mean binding energy per contact atom")
    for i, (v, e) in enumerate(zip(e_vals, e_errs)):
        axes[1].text(i, v + e + 0.008, f"{v:.3f}", ha="center")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)

    print(f"Wrote {OUT_KH_CSV}")
    print(f"Wrote {OUT_BARE_CSV}")
    print(f"Wrote {OUT_COMPARE_CSV}")
    print(f"Wrote {OUT_PNG}")
    print("Bare common cutoff: "
          f"N={bare_s['N_contact_mean']:.0f}, E/atom={bare_s['E_total_per_atom_mean']:.4f}")
    print("KH570 common cutoff: "
          f"N={kh_s['N_contact_mean']:.0f}, E/atom={kh_s['E_total_per_atom_mean']:.4f}")
    print("KH570 own layer: "
          f"N={kh_sys['N_contact_mean']:.0f}, E/atom={kh_sys['E_total_per_atom_mean']:.4f}")


if __name__ == "__main__":
    main()
