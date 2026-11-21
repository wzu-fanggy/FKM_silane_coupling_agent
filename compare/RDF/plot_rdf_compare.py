#!/usr/bin/env python3
"""Compare the 4 systems' RDF / interface structure.
Key comparable observable (type-number independent, thus valid across the
different remapped systems): FKM radial number & mass density profile vs
distance r from the SiO2/cluster centre, plus the interface-layer metrics."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent

# established color scheme
COLORS = {"bare": "#90EE90", "KH550": "#457B9D", "KH560": "#FFD700", "KH570": "#E63946"}
ORDER = ["bare", "KH570", "KH560", "KH550"]

PROFILES = {
    "bare":  Path(r"F:\Lammps-MS\FKM-SIO2-TEST\RDF\outputs\interface_layer\radial_density_profile.csv"),
    "KH570": Path(r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\RDF\outputs\interface_layer\radial_density_profile.csv"),
    "KH560": Path(r"F:\Lammps-MS\FKM-SIO2-TEST\KH560\RDF\outputs\interface_layer\radial_density_profile.csv"),
    "KH550": Path(r"F:\Lammps-MS\FKM-SIO2-TEST\KH550\RDF\outputs\interface_layer\radial_density_profile.csv"),
}

KEYS = {  # column index by role, per system
    "r": 0,
    "fkm_n": {"bare": 2, "KH570": 4, "KH560": 4, "KH550": 4},
    "fkm_mass": {"bare": 3, "KH570": 5, "KH560": 5, "KH550": 5},
    "core": {"bare": 1, "KH570": 2, "KH560": 2, "KH550": 2},  # SiO2/core in each
}


def load_profile(name):
    path = PROFILES[name]
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.reader(f)):
            if i == 0:
                continue
            rows.append([float(x) for x in row if x.strip() != ""])
    r = [x[KEYS["r"]] for x in rows]
    fn = [x[KEYS["fkm_n"][name]] for x in rows]
    fm = [x[KEYS["fkm_mass"][name]] for x in rows]
    co = [x[KEYS["core"][name]] for x in rows]
    return r, fn, fm, co


def main():
    data = {n: load_profile(n) for n in ORDER}

    # ---- Fig 1: FKM number-density radial profile (the RDF) ----
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for n in ORDER:
        r, fn, *_ = data[n]
        ax.plot(r, fn, color=COLORS[n], lw=1.6, label=n)
    ax.set_xlabel(r"r from filler centre ($\AA$)")
    ax.set_ylabel(r"FKM number density (atoms/$\AA^3$)")
    ax.set_title("FKM radial density profile (RDF/interface) — 4 systems")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT / "compare_fkm_radial_number_density.png", dpi=180)
    plt.close(fig)

    # ---- Fig 2: FKM mass-density radial profile ----
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for n in ORDER:
        r, _, fm, _ = data[n]
        ax.plot(r, fm, color=COLORS[n], lw=1.6, label=n)
    ax.set_xlabel(r"r from filler centre ($\AA$)")
    ax.set_ylabel(r"FKM mass density (g/cm$^3$)")
    ax.set_title("FKM radial mass-density profile — 4 systems")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT / "compare_fkm_radial_mass_density.png", dpi=180)
    plt.close(fig)

    # ---- Fig 3: SiO2 / core radial profile (filler geometry) ----
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for n in ORDER:
        r, _, _, co = data[n]
        ax.plot(r, co, color=COLORS[n], lw=1.5, label=n)
    ax.set_xlabel(r"r from filler centre ($\AA$)")
    ax.set_ylabel(r"SiO$_2$/core number density (atoms/$\AA^3$)")
    ax.set_title("SiO$_2$ (cluster core) radial profile — 4 systems")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 25)
    fig.tight_layout()
    fig.savefig(OUT / "compare_sio2_core_radial_profile.png", dpi=180)
    plt.close(fig)

    print("Saved 3 profile figures.")


if __name__ == "__main__":
    main()