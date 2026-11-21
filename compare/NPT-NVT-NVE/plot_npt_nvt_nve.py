#!/usr/bin/env python3
"""Plot per-system NPT-NVT-NVE equilibration: x=time(ps), left y=Temperature(K,
blue), right y=Density(g/cm3, green). Faithful data, no fitting.
Input: each system's NPT-NVT-NVE log.lammps thermo output."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent

LOGS = {
    "bare":   Path(r"F:\Lammps-MS\FKM-SIO2-TEST\NPT-NVT-NVE\server-output\log.lammps"),
    "KH570":  Path(r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\NPT-NVT-NVE\output\log.lammps"),
    "KH560":  Path(r"F:\Lammps-MS\FKM-SIO2-TEST\KH560\NPT-NVT-NVE\log.lammps"),
    "KH550":  Path(r"F:\Lammps-MS\FKM-SIO2-TEST\KH550\NPT-NVT-NVE\output\log.lammps"),
}


def parse_log(path):
    step, temp, dens = [], [], []
    for line in path.read_text(encoding="ascii", errors="ignore").splitlines():
        q = line.split()
        if len(q) < 8 or not q[0].isdigit():
            continue
        try:
            _s = float(q[0]); _t = float(q[1]); _d = float(q[7])
        except ValueError:
            continue
        step.append(_s / 1000.0)  # dt=1.0 fs -> time(ps)
        temp.append(_t)
        dens.append(_d)
    return step, temp, dens


def main():
    made, missing = [], []
    for name, path in LOGS.items():
        if not path.exists():
            missing.append(name)
            print(f"{name}: MISSING log -> {path}")
            continue
        t, temp, dens = parse_log(path)
        fig, ax1 = plt.subplots(figsize=(9, 5.5))
        ax1.plot(t, temp, color="tab:blue", lw=1.1, label="Temperature")
        ax1.set_xlabel("Time (ps)")
        ax1.set_ylabel("Temperature (K)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.set_ylim(min(temp) - 15, max(temp) + 30)

        ax2 = ax1.twinx()
        ax2.plot(t, dens, color="tab:green", lw=1.1, alpha=0.85, label="Density")
        ax2.set_ylabel("Density (g/cm$^3$)", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")

        fig.suptitle(f"NPT-NVT-NVE equilibration — {name} (FKM/SiO$_2$) 300 K",
                     fontsize=13, y=0.97)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(OUT / f"npt_nvt_nve_{name}.png", dpi=180)
        plt.close(fig)
        made.append(name)
        print(f"{name}: n={len(t)} t{min(t):.1f}..{max(t):.1f} ps  "
              f"T{min(temp):.1f}..{max(temp):.1f} K  "
              f"rho {min(dens):.4f}..{max(dens):.4f} g/cm3")
    print(f"\nPlots made: {made}")
    print(f"Missing (no log): {missing}")


if __name__ == "__main__":
    main()