#!/usr/bin/env python3
"""Compare corrected bare and KH570-grafted radial mass density profiles."""

import csv
import re
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BARE_DIR = Path(__file__).resolve().parent
KH570_DIR = Path(r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\cut-MSD\Density distribution")


def load_rmdd(txt_path):
    layers = []
    data = []
    pat = re.compile(
        r"^#\s+(.+?):\s+R=\[([\d.]+),([\d.]+)\]\s+A,\s+rho=([\d.]+)\s+g/cm3,\s+atoms=(\d+)"
    )
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            m = pat.match(s)
            if m:
                layers.append(
                    {
                        "layer": m.group(1),
                        "rmin": float(m.group(2)),
                        "rmax": float(m.group(3)),
                        "rho": float(m.group(4)),
                        "atoms": int(m.group(5)),
                    }
                )
                continue
            q = s.split()
            if len(q) == 3 and q[0][0].isdigit():
                data.append((float(q[0]), float(q[1]), float(q[2])))
    arr = np.asarray(data)
    return layers, arr


def main():
    bare_layers, bare = load_rmdd(BARE_DIR / "radial_density_data_bare.txt")
    kh_layers, kh = load_rmdd(KH570_DIR / "radial_density_data_kh570.txt")

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(bare[:, 0], bare[:, 2], "b-", lw=2.0, label="bare SiO2")
    ax.plot(kh[:, 0], kh[:, 2], "r-", lw=2.0, label="KH570-grafted SiO2")
    for layer, color in zip(bare_layers, ["tab:blue", "tab:cyan", "tab:purple", "tab:gray"]):
        ax.axvline(layer["rmin"], color=color, ls="--", lw=1.0, alpha=0.5)
    for layer, color in zip(kh_layers, ["tab:red", "tab:orange", "tab:pink", "tab:gray"]):
        ax.axvline(layer["rmin"], color=color, ls=":", lw=1.0, alpha=0.5)
    ax.set_xlabel("Distance from cluster COM (A)")
    ax.set_ylabel("Mass density (g/cm3)")
    ax.set_title("Radial mass density: bare vs KH570-grafted SiO2/FKM")
    ax.set_xlim(0, 35)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(BARE_DIR / "rmdd_bare_vs_kh570.png", dpi=300)
    plt.close(fig)

    rows = []
    for b, k in zip(bare_layers, kh_layers):
        rows.append(
            {
                "layer": b["layer"].replace(" (SiO2)", "").replace(" (SiO2+KH570)", ""),
                "bare_rmin": b["rmin"],
                "bare_rmax": b["rmax"],
                "bare_rho": b["rho"],
                "bare_atoms": b["atoms"],
                "kh570_rmin": k["rmin"],
                "kh570_rmax": k["rmax"],
                "kh570_rho": k["rho"],
                "kh570_atoms": k["atoms"],
            }
        )

    with open(BARE_DIR / "rmdd_layer_comparison.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(BARE_DIR / "rmdd_layer_comparison.txt", "w", encoding="utf-8") as f:
        f.write("bare vs KH570 RMDD layer comparison\n")
        for r in rows:
            f.write(
                f"{r['layer']:<18} bare R=[{r['bare_rmin']:.2f},{r['bare_rmax']:.2f}] "
                f"rho={r['bare_rho']:.4f} atoms={r['bare_atoms']} | "
                f"KH570 R=[{r['kh570_rmin']:.2f},{r['kh570_rmax']:.2f}] "
                f"rho={r['kh570_rho']:.4f} atoms={r['kh570_atoms']}\n"
            )

    print(f"wrote {BARE_DIR / 'rmdd_bare_vs_kh570.png'}")
    print(f"wrote {BARE_DIR / 'rmdd_layer_comparison.csv'}")
    print(f"wrote {BARE_DIR / 'rmdd_layer_comparison.txt'}")
    for r in rows:
        print(
            f"{r['layer']:<18} bare {r['bare_rho']:.3f} g/cm3 ({r['bare_atoms']}) | "
            f"KH570 {r['kh570_rho']:.3f} g/cm3 ({r['kh570_atoms']})"
        )


if __name__ == "__main__":
    main()
