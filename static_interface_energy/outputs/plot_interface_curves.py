import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


CSV = Path(r"C:\Users\Hxic\Documents\ChatGPT\lammps\interface_energy_last50.csv")
OUT_DIR = Path(r"C:\Users\Hxic\Documents\ChatGPT\lammps")


def load_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "time_ps": float(row["time_ps"]),
                    "E_total": float(row["E_total_kcal_mol"]),
                    "E_vdw": float(row["E_vdw_kcal_mol"]),
                    "E_ele": float(row["E_ele_kcal_mol"]),
                }
            )
    return rows


def main():
    rows = load_csv(CSV)
    times = [r["time_ps"] for r in rows]
    series = {
        "E_total": [r["E_total"] for r in rows],
        "E_vdw": [r["E_vdw"] for r in rows],
        "E_ele": [r["E_ele"] for r in rows],
    }
    colors = {"E_total": "tab:blue", "E_vdw": "tab:orange", "E_ele": "tab:green"}
    titles = {
        "E_total": "FKM-SiO2 总非键界面能 (最后50帧)",
        "E_vdw": "FKM-SiO2 范德华界面能 (最后50帧)",
        "E_ele": "FKM-SiO2 静电界面能 (最后50帧)",
    }

    for name, values in series.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(times, values, "-o", color=colors[name], ms=3, lw=1.3, label=name)
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        ax.axhline(mean, color="red", ls="--", lw=1)
        ax.text(
            times[0],
            mean + 3,
            f"mean = {mean:.2f} ± {std:.2f} kcal/mol",
            color="red",
            fontsize=9,
        )
        ax.set_xlabel("时间 (ps)")
        ax.set_ylabel("能量 (kcal/mol)")
        ax.set_title(titles[name])
        ax.legend()
        fig.tight_layout()
        out = OUT_DIR / f"interface_{name.lower()}_last50_curve.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
