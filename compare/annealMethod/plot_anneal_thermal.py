#!/usr/bin/env python3
"""Plot per-system anneal curve: x=time(ps), left y=Temperature(K, blue),
right y=Density(g/cm3, green). Faithful data, no fitting.
Data parsed from each system's anneal log.lammps thermo output."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent

LOGS = {
    "bare":   Path(r"F:\Lammps-MS\FKM-SIO2-TEST\Anneal\log_anneal_20260816.lammps"),
    "KH570":  Path(r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\Anneal\output\log.lammps"),
    "KH560":  Path(r"F:\Lammps-MS\FKM-SIO2-TEST\KH560\Anneal\output\log.lammps"),
    "KH550":  Path(r"F:\Lammps-MS\FKM-SIO2-TEST\KH550\anneal\output\log.lammps"),
}
STEP_DT0_5 = 220000  # phases 1-2 run at dt=0.5 fs up to step 220000


def time_ps(step):
    fs = min(step, STEP_DT0_5) * 0.5 + max(0.0, step - STEP_DT0_5) * 1.0
    return fs / 1000.0


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
        step.append(_s); temp.append(_t); dens.append(_d)
    return ([time_ps(s) for s in step], temp, dens)


def main():
    for name, path in LOGS.items():
        t, temp, dens = parse_log(path)
        fig, ax1 = plt.subplots(figsize=(9, 5.5))
        ax1.plot(t, temp, color="tab:blue", lw=1.2, label="Temperature")
        ax1.set_xlabel("Time (ps)")
        ax1.set_ylabel("Temperature (K)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.set_ylim(min(temp) - 20, max(temp) + 40)

        ax2 = ax1.twinx()
        ax2.plot(t, dens, color="tab:green", lw=1.2, alpha=0.85, label="Density")
        ax2.set_ylabel("Density (g/cm$^3$)", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")

        fig.suptitle(f"Anneal thermal history — {name} (FKM/SiO$_2$)",
                     fontsize=13, y=0.97)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(OUT / f"anneal_{name}.png", dpi=180)
        plt.close(fig)
        # per-phase stats for the method doc
        print(f"{name}: n={len(temp)} t{min(t):.1f}..{max(t):.1f} ps  "
              f"T{min(temp):.1f}..{max(temp):.1f} K  "
              f"rho {min(dens):.3f}..{max(dens):.3f} g/cm3")


if __name__ == "__main__":
    main()