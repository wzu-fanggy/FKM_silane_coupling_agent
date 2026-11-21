#!/usr/bin/env python3
"""Parse the two rerun logs and output KH570 interface energy CSV/plots."""

import csv
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


WORK = Path(__file__).resolve().parent
LOG_TOTAL = WORK / "log.total50.txt"
LOG_VDW = WORK / "log.vdw50.txt"
OUT_CSV = WORK / "interface_energy_last50.csv"
OUT_PNG = WORK / "interface_energy_last50.png"


def parse_log(path):
    rows = []
    pattern = re.compile(r"^\s*(\d+)\s+(-?[\d.Ee+-]+)\s*$")
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            m = pattern.match(line)
            if m:
                rows.append((int(m.group(1)), float(m.group(2))))
    return rows


def main():
    total = parse_log(LOG_TOTAL)
    vdw = parse_log(LOG_VDW)
    if len(total) != len(vdw):
        raise SystemExit(f"frame mismatch: {len(total)} vs {len(vdw)}")

    rows = []
    for (step_t, e_total), (step_v, e_vdw) in zip(total, vdw):
        if step_t != step_v:
            raise SystemExit(f"step mismatch: {step_t} vs {step_v}")
        rows.append({
            "step": step_t,
            "time_ps": step_t * 0.001,
            "E_total_kcal_mol": e_total,
            "E_vdw_kcal_mol": e_vdw,
            "E_ele_kcal_mol": e_total - e_vdw,
        })

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    et = [r["E_total_kcal_mol"] for r in rows]
    ev = [r["E_vdw_kcal_mol"] for r in rows]
    ee = [r["E_ele_kcal_mol"] for r in rows]
    summary = {
        "E_total_mean": statistics.mean(et),
        "E_total_std": statistics.stdev(et),
        "E_vdw_mean": statistics.mean(ev),
        "E_vdw_std": statistics.stdev(ev),
        "E_ele_mean": statistics.mean(ee),
        "E_ele_std": statistics.stdev(ee),
    }
    print(summary)

    times = [r["time_ps"] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(times, et, "-o", ms=3, label="E_total")
    axes[0].plot(times, ev, "-o", ms=3, label="E_vdw")
    axes[0].plot(times, ee, "-o", ms=3, label="E_ele")
    axes[0].set_ylabel("界面能 (kcal/mol)")
    axes[0].legend()
    axes[0].set_title("KH570接枝体系 FKM-接枝SiO2 静态界面能分解（最后50帧）")

    labels = ["E_total", "E_vdw", "E_ele"]
    means = [summary["E_total_mean"], summary["E_vdw_mean"], summary["E_ele_mean"]]
    errs = [summary["E_total_std"], summary["E_vdw_std"], summary["E_ele_std"]]
    axes[1].bar(labels, means, yerr=errs, capsize=5)
    for i, (m, s) in enumerate(zip(means, errs)):
        axes[1].text(i, m + s + 5, f"{m:.1f}±{s:.1f}", ha="center")
    axes[1].set_ylabel("mean ± std (kcal/mol)")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
