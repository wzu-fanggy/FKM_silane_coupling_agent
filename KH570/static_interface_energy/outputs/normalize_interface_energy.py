#!/usr/bin/env python3
"""Normalize KH570-grafted FKM/SiO2 interface energy by contact atoms.

Same protocol as the bare SiO2 system. The cutoff comes from the KH570
FKM radial density profile:
    R_surf     = 15.048 A  (SiO2 half-core-density surface radius)
    delta_pert =  6.077 A  (perturbed interface layer thickness)
Contact atoms are FKM atoms within R_surf + delta_pert of the per-frame
SiO2(+KH570) group geometric center, using the minimum-image distance.
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
DUMP = WORK / "last50.dump"
ENERGY_CSV = WORK / "interface_energy_last50.csv"
OUT_CSV = WORK / "interface_energy_normalized.csv"
OUT_PNG = WORK / "interface_energy_normalized.png"
OUT_TXT = WORK / "interface_energy_normalized_summary.txt"

# KH570 system interface layer parameters from KH570/RDF interface_layer analysis.
R_SURF = 15.048
DELTA_PERT = 6.077
R_CUT = R_SURF + DELTA_PERT

# KH570 system atom types (relabeled data, 17 types).
FKM_TYPES = {1, 2, 3, 4, 8, 10}
SIO2_TYPES = {5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17}


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
            ids = np.empty(natoms, dtype=np.int64)
            types = np.empty(natoms, dtype=np.int64)
            pos = np.empty((natoms, 3))
            for i in range(natoms):
                parts = f.readline().split()
                ids[i] = int(parts[0])
                types[i] = int(parts[1])
                pos[i] = [float(parts[2]), float(parts[3]), float(parts[4])]
            frames.append((step, natoms, lo, hi, ids, types, pos))
    return frames


def minimum_image_vector(delta, box):
    """Wrap a vector to the minimum-image representation."""
    d = delta.copy()
    span = (box[1] - box[0])[:, None]
    d -= np.round(d / span) * span
    return d


def main():
    frames = read_dump(DUMP)
    if len(frames) != 50:
        print(f"Warning: expected 50 frames, got {len(frames)}")

    energy = {}
    with open(ENERGY_CSV, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            energy[int(row["step"])] = {
                "E_total": float(row["E_total_kcal_mol"]),
                "E_vdw": float(row["E_vdw_kcal_mol"]),
                "E_ele": float(row["E_ele_kcal_mol"]),
            }

    rows = []
    for step, natoms, lo, hi, ids, types, pos in frames:
        box = np.vstack((lo, hi))
        is_sio2 = np.isin(types, list(SIO2_TYPES))
        is_fkm = np.isin(types, list(FKM_TYPES))
        sio2_pos = pos[is_sio2]
        fkm_pos = pos[is_fkm]

        # Per-frame SiO2(+KH570) group geometric center.
        center = sio2_pos.mean(axis=0)
        dvec = fkm_pos - center
        dvec = minimum_image_vector(dvec.T, box).T
        dist = np.linalg.norm(dvec, axis=1)
        n_contact = int(np.count_nonzero(dist <= R_CUT))

        e = energy[step]
        rows.append(
            {
                "step": step,
                "time_ps": step * 0.001,
                "N_contact_atoms": n_contact,
                "E_total_kcal_mol": e["E_total"],
                "E_vdw_kcal_mol": e["E_vdw"],
                "E_ele_kcal_mol": e["E_ele"],
                "E_total_per_contact_atom": e["E_total"] / n_contact,
                "E_vdw_per_contact_atom": e["E_vdw"] / n_contact,
                "E_ele_per_contact_atom": e["E_ele"] / n_contact,
            }
        )

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = [r["N_contact_atoms"] for r in rows]
    et = [r["E_total_kcal_mol"] for r in rows]
    ev = [r["E_vdw_kcal_mol"] for r in rows]
    ee = [r["E_ele_kcal_mol"] for r in rows]
    etp = [r["E_total_per_contact_atom"] for r in rows]
    evp = [r["E_vdw_per_contact_atom"] for r in rows]
    eep = [r["E_ele_per_contact_atom"] for r in rows]

    summary = {
        "N_contact_atoms_mean": statistics.mean(n),
        "N_contact_atoms_std": statistics.stdev(n),
        "E_total_mean": statistics.mean(et),
        "E_total_std": statistics.stdev(et),
        "E_vdw_mean": statistics.mean(ev),
        "E_vdw_std": statistics.stdev(ev),
        "E_ele_mean": statistics.mean(ee),
        "E_ele_std": statistics.stdev(ee),
        "E_total_per_contact_atom_mean": statistics.mean(etp),
        "E_total_per_contact_atom_std": statistics.stdev(etp),
        "E_vdw_per_contact_atom_mean": statistics.mean(evp),
        "E_vdw_per_contact_atom_std": statistics.stdev(evp),
        "E_ele_per_contact_atom_mean": statistics.mean(eep),
        "E_ele_per_contact_atom_std": statistics.stdev(eep),
        "R_surf_A": R_SURF,
        "delta_pert_A": DELTA_PERT,
        "R_cut_A": R_CUT,
    }

    with open(OUT_TXT, "w", encoding="utf-8-sig") as f:
        f.write("KH570 FKM/SiO2 interface energy normalized by contact atoms\n")
        f.write(f"R_surf = {R_SURF:.3f} A, delta_pert = {DELTA_PERT:.3f} A, "
                f"R_cut = {R_CUT:.3f} A\n")
        f.write(f"FKM group types: 1 2 3 4 8 10 | "
                f"SIO2 group types: 5 6 7 9 11 12 13 14 15 16 17\n\n")
        for key, value in summary.items():
            f.write(f"{key} = {value:.6g}\n")

    times = [r["time_ps"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    axes[0].plot(times, n, "-o", ms=3, color="tab:green")
    axes[0].axhline(statistics.mean(n), color="gray", ls=":", lw=1)
    axes[0].set_ylabel("N_contact atoms")
    axes[0].set_title(
        f"KH570-grafted FKM/SiO2 interface energy normalized by contact atoms "
        f"(R_cut = {R_CUT:.1f} A)"
    )

    axes[1].plot(times, et, "-o", ms=3, label="E_total")
    axes[1].plot(times, ev, "-o", ms=3, label="E_vdw")
    axes[1].plot(times, ee, "-o", ms=3, label="E_ele")
    axes[1].set_ylabel("kcal/mol")
    axes[1].legend()

    axes[2].plot(times, etp, "-o", ms=3, label="E_total / N_contact")
    axes[2].plot(times, evp, "-o", ms=3, label="E_vdw / N_contact")
    axes[2].plot(times, eep, "-o", ms=3, label="E_ele / N_contact")
    axes[2].set_ylabel("kcal/mol/atom")
    axes[2].set_xlabel("time (ps)")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_TXT}")
    print(f"Wrote {OUT_PNG}")
    print(f"N_contact = {summary['N_contact_atoms_mean']:.1f} +/- "
          f"{summary['N_contact_atoms_std']:.1f}")
    print(f"E_total/atom = {summary['E_total_per_contact_atom_mean']:.4f} +/- "
          f"{summary['E_total_per_contact_atom_std']:.4f} kcal/mol/atom")
    print(f"E_vdw/atom   = {summary['E_vdw_per_contact_atom_mean']:.4f} +/- "
          f"{summary['E_vdw_per_contact_atom_std']:.4f} kcal/mol/atom")
    print(f"E_ele/atom   = {summary['E_ele_per_contact_atom_mean']:.4f} +/- "
          f"{summary['E_ele_per_contact_atom_std']:.4f} kcal/mol/atom")


if __name__ == "__main__":
    main()
