#!/usr/bin/env python3
"""Compare FKM radial density profiles: bare SiO2 vs KH550-grafted SiO2."""

import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


WORK = Path(__file__).resolve().parent
BARE_CSV = Path(r"F:\Lammps-MS\FKM-SIO2-TEST\RDF\outputs\interface_layer\radial_density_profile.csv")
KH550_CSV = WORK / "radial_density_profile.csv"


def load(path, rsurf, fkm_col):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    r = np.array([float(x["r_A"]) for x in rows])
    fkm = np.array([float(x[fkm_col]) for x in rows])
    return r - rsurf, fkm


def main():
    r_bare, fkm_bare = load(BARE_CSV, 15.034, "FKM_n_atoms_per_A3")
    r_kh, fkm_kh = load(KH550_CSV, 15.05, "FKM_n_atoms_per_A3")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(r_bare, fkm_bare, lw=1.4, color="tab:blue", label="裸 SiO2（参考）")
    ax.plot(r_kh, fkm_kh, lw=1.4, color="tab:red", label="KH550 接枝 SiO2")
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.axhline(0.0911, color="red", ls=":", lw=1)
    ax.set_xlim(-2, 12)
    ax.set_xlabel("r - R_surf (Å)")
    ax.set_ylabel("FKM 数密度 (atoms/Å³)")
    ax.set_title("FKM 径向密度剖面对比：裸 SiO2 vs KH550 接枝")
    ax.legend()
    fig.tight_layout()
    fig.savefig(WORK / "fkm_profile_compare_bare_vs_kh550.png", dpi=160)
    print("saved", WORK / "fkm_profile_compare_bare_vs_kh550.png")


if __name__ == "__main__":
    main()
