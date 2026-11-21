#!/usr/bin/env python3
"""Normalize pull-out peak force and work by initial contact atoms (3.0 A)."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np

from count_contact import BARE, KH570, KH560, KH550, contact_count


OUT = Path(__file__).resolve().parent
CUTOFF = 3.0

BARE_FRAMES = Path(
    r"F:\Lammps-MS\FKM-SIO2-TEST\sci-pull\first_deal\10A-20A\10A"
)
KH570_FRAMES = Path(
    r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\sci-pull\first_deal\10A-20A\10A"
)
KH560_FRAMES = Path(
    r"F:\Lammps-MS\FKM-SIO2-TEST\KH560\sci-pull\first_deal\10A-20A\10A"
)
KH550_FRAMES = Path(
    r"F:\Lammps-MS\FKM-SIO2-TEST\KH550\sci-pull\first_deal\10A-20A\10A"
)
BARE_SUMMARY = Path(
    r"F:\Lammps-MS\FKM-SIO2-TEST\sci-pull\first_deal\10A-20A\analysis\pull_rate_summary.csv"
)
KH570_SUMMARY = Path(
    r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\sci-pull\first_deal\10A-20A\analysis\pull_rate_summary.csv"
)
KH560_SUMMARY = Path(
    r"F:\Lammps-MS\FKM-SIO2-TEST\KH560\sci-pull\first_deal\10A-20A\analysis\pull_rate_summary.csv"
)
KH550_SUMMARY = Path(
    r"F:\Lammps-MS\FKM-SIO2-TEST\KH550\sci-pull\first_deal\10A-20A\analysis\pull_rate_summary.csv"
)


def read_rows(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def contact_map(base, types):
    return {
        i: contact_count(base / str(i) / f"frame{i}.data", types, CUTOFF)
        for i in (1, 2, 3)
    }


def normalized_rows(rows, contacts):
    out = []
    for r in rows:
        speed = int(r["speed_Aps"])
        frame = int(r["frame"])
        f = float(r["F_max_kcal_mol_Ang"])
        w = float(r["W_sep_kcal_mol"])
        n = contacts[frame]
        out.append(
            {
                "speed_Aps": speed,
                "frame": frame,
                "N_contact": n,
                "F_norm": f / n,
                "W_norm": w / n,
            }
        )
    return out


def aggregate(rows):
    by_speed = {}
    for r in rows:
        by_speed.setdefault(r["speed_Aps"], []).append(r)
    agg = {}
    for speed, items in by_speed.items():
        f = [x["F_norm"] for x in items]
        w = [x["W_norm"] for x in items]
        agg[speed] = {
            "F_mean": sum(f) / len(f),
            "F_std": (sum((x - sum(f) / len(f)) ** 2 for x in f) / (len(f) - 1)) ** 0.5
            if len(f) > 1
            else 0.0,
            "W_mean": sum(w) / len(w),
            "W_std": (sum((x - sum(w) / len(w)) ** 2 for x in w) / (len(w) - 1)) ** 0.5
            if len(w) > 1
            else 0.0,
        }
    return agg


def aggregate_raw(rows):
    by_speed = {}
    for r in rows:
        by_speed.setdefault(int(r["speed_Aps"]), []).append(r)
    agg = {}
    for speed, items in by_speed.items():
        f = [float(x["F_max_kcal_mol_Ang"]) for x in items]
        w = [float(x["W_sep_kcal_mol"]) for x in items]
        r = [float(x["residual_fraction"]) for x in items]

        def mean_std(vals):
            m = sum(vals) / len(vals)
            s = (
                (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
                if len(vals) > 1
                else 0.0
            )
            return m, s

        f_m, f_s = mean_std(f)
        w_m, w_s = mean_std(w)
        r_m, r_s = mean_std(r)
        agg[speed] = {
            "F_mean": f_m,
            "F_std": f_s,
            "W_mean": w_m,
            "W_std": w_s,
            "R_mean": r_m,
            "R_std": r_s,
        }
    return agg


def main():
    bare_contacts = contact_map(BARE_FRAMES, BARE)
    kh570_contacts = contact_map(KH570_FRAMES, KH570)
    kh560_contacts = contact_map(KH560_FRAMES, KH560)
    kh550_contacts = contact_map(KH550_FRAMES, KH550)
    print("N_contact (3.0 A):")
    for i in (1, 2, 3):
        print(
            f"  frame{i}: bare={bare_contacts[i]}, "
            f"KH570={kh570_contacts[i]}, "
            f"KH560={kh560_contacts[i]}, "
            f"KH550={kh550_contacts[i]}"
        )

    bare = aggregate(normalized_rows(read_rows(BARE_SUMMARY), bare_contacts))
    kh570 = aggregate(normalized_rows(read_rows(KH570_SUMMARY), kh570_contacts))
    kh560 = aggregate(normalized_rows(read_rows(KH560_SUMMARY), kh560_contacts))
    kh550 = aggregate(normalized_rows(read_rows(KH550_SUMMARY), kh550_contacts))
    speeds = sorted(bare.keys() & kh570.keys() & kh560.keys() & kh550.keys())

    with open(OUT / "pull_comparison_normalized.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "speed_Aps",
                "bare_F_per_contact",
                "kh570_F_per_contact",
                "kh560_F_per_contact",
                "kh550_F_per_contact",
                "F_ratio_kh570_over_bare",
                "F_ratio_kh560_over_bare",
                "F_ratio_kh550_over_bare",
                "bare_W_per_contact",
                "kh570_W_per_contact",
                "kh560_W_per_contact",
                "kh550_W_per_contact",
                "W_ratio_kh570_over_bare",
                "W_ratio_kh560_over_bare",
                "W_ratio_kh550_over_bare",
            ]
        )
        for s in speeds:
            w.writerow(
                [
                    s,
                    round(bare[s]["F_mean"], 4),
                    round(kh570[s]["F_mean"], 4),
                    round(kh560[s]["F_mean"], 4),
                    round(kh550[s]["F_mean"], 4),
                    round(kh570[s]["F_mean"] / bare[s]["F_mean"], 3),
                    round(kh560[s]["F_mean"] / bare[s]["F_mean"], 3),
                    round(kh550[s]["F_mean"] / bare[s]["F_mean"], 3),
                    round(bare[s]["W_mean"], 3),
                    round(kh570[s]["W_mean"], 3),
                    round(kh560[s]["W_mean"], 3),
                    round(kh550[s]["W_mean"], 3),
                    round(kh570[s]["W_mean"] / bare[s]["W_mean"], 3),
                    round(kh560[s]["W_mean"] / bare[s]["W_mean"], 3),
                    round(kh550[s]["W_mean"] / bare[s]["W_mean"], 3),
                ]
            )

    # user colour mapping: bare=light green, KH550=blue, KH560=yellow, KH570=red
    order = ["bare", "KH550", "KH560", "KH570"]
    sets = {n: {"bare": bare, "KH550": kh550, "KH560": kh560, "KH570": kh570}[n] for n in (order)}
    colors = {"bare": "#90EE90", "KH550": "#457B9D", "KH560": "#FFD700", "KH570": "#E63946"}
    edges = {"bare": "#5fbf5f", "KH550": "#355f7d", "KH560": "#c4a800", "KH570": "#c42b38"}
    metrics = [
        ("F_mean", "F_std", "Peak force / contact atom\n(kcal/mol/A/atom)"),
        ("W_mean", "W_std", "Work of separation / contact atom\n(kcal/mol/atom)"),
    ]

    def bar_figure(fname):
        """Grouped bar chart with error bars per pull rate."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        xs = np.arange(len(speeds))
        width = 0.2
        for ax, (mm, ms, ylab) in zip(axes, metrics):
            for j, name in enumerate(order):
                means = [sets[name][s][mm] for s in speeds]
                stds = [sets[name][s][ms] for s in speeds]
                ax.bar(
                    xs + (j - 1.5) * width,
                    means,
                    width,
                    yerr=stds,
                    color=colors[name],
                    edgecolor=edges[name],
                    linewidth=0.8,
                    error_kw=dict(elinewidth=1.2, capsize=3, capthick=1.2, ecolor="black"),
                    label=name,
                )
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{s}" for s in speeds])
            ax.set_xlabel("Pull rate (A/ps)")
            ax.set_ylabel(ylab)
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_axisbelow(True)
        handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[n], ec=edges[n])
                   for n in order]
        for ax in axes:
            ax.legend(handles, order, ncol=4, frameon=False, loc="upper left")
        fig.suptitle("Normalized pull-out results (mean ± std, n=3)", y=1.02)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=180)
        plt.close(fig)

    def fit_figure(fname):
        """Linear least-squares fit (y = k*x + b); fit line only."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        for ax, (mm, ms, ylab) in zip(axes, metrics):
            legend_handles = []
            for name in order:
                x = np.array(speeds, dtype=float)
                y = np.array([sets[name][s][mm] for s in speeds], dtype=float)
                k_, b_ = np.polyfit(x, y, 1)  # y = k_*x + b_
                yhat = k_ * x + b_
                ss_res = float(((y - yhat) ** 2).sum())
                ss_tot = float(((y - y.mean()) ** 2).sum())
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                xs = np.linspace(x.min(), x.max(), 80)
                ax.plot(xs, k_ * xs + b_, "-", color=colors[name], linewidth=1.6)
                legend_handles.append(
                    Line2D([0], [0], color=colors[name],
                           label=f"{name} (k={k_:.2f}, R$^2$={r2:.2f})")
                )
            ax.set_xlabel("Pull rate (A/ps)")
            ax.set_ylabel(ylab)
            ax.grid(True, alpha=0.3)
            ax.legend(handles=legend_handles, frameon=False, loc="upper left", fontsize=9)
        fig.suptitle("Normalized pull-out results with linear fit "
                     "($y=kx+b$)", y=1.02)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=180)
        plt.close(fig)

    bar_figure("compare_bar_mean_std.png")
    fit_figure("compare_line_fit.png")

    raw_bare = aggregate_raw(read_rows(BARE_SUMMARY))
    raw_kh570 = aggregate_raw(read_rows(KH570_SUMMARY))
    raw_kh560 = aggregate_raw(read_rows(KH560_SUMMARY))
    raw_kh550 = aggregate_raw(read_rows(KH550_SUMMARY))
    raw_sets = {"bare": raw_bare, "KH570": raw_kh570, "KH560": raw_kh560, "KH550": raw_kh550}

    with open(OUT / "pull_comparison.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "speed_Aps",
                "bare_F_max",
                "kh570_F_max",
                "kh560_F_max",
                "kh550_F_max",
                "F_max_ratio_kh570_over_bare",
                "F_max_ratio_kh560_over_bare",
                "F_max_ratio_kh550_over_bare",
                "bare_W_sep",
                "kh570_W_sep",
                "kh560_W_sep",
                "kh550_W_sep",
                "W_sep_ratio_kh570_over_bare",
                "W_sep_ratio_kh560_over_bare",
                "W_sep_ratio_kh550_over_bare",
                "bare_residual",
                "kh570_residual",
                "kh560_residual",
                "kh550_residual",
                "residual_ratio_kh570_over_bare",
                "residual_ratio_kh560_over_bare",
                "residual_ratio_kh550_over_bare",
            ]
        )
        for s in speeds:
            w.writerow(
                [
                    s,
                    round(raw_bare[s]["F_mean"], 4),
                    round(raw_kh570[s]["F_mean"], 4),
                    round(raw_kh560[s]["F_mean"], 4),
                    round(raw_kh550[s]["F_mean"], 4),
                    round(raw_kh570[s]["F_mean"] / raw_bare[s]["F_mean"], 3),
                    round(raw_kh560[s]["F_mean"] / raw_bare[s]["F_mean"], 3),
                    round(raw_kh550[s]["F_mean"] / raw_bare[s]["F_mean"], 3),
                    round(raw_bare[s]["W_mean"], 4),
                    round(raw_kh570[s]["W_mean"], 4),
                    round(raw_kh560[s]["W_mean"], 4),
                    round(raw_kh550[s]["W_mean"], 4),
                    round(raw_kh570[s]["W_mean"] / raw_bare[s]["W_mean"], 3),
                    round(raw_kh560[s]["W_mean"] / raw_bare[s]["W_mean"], 3),
                    round(raw_kh550[s]["W_mean"] / raw_bare[s]["W_mean"], 3),
                    round(raw_bare[s]["R_mean"], 6),
                    round(raw_kh570[s]["R_mean"], 6),
                    round(raw_kh560[s]["R_mean"], 6),
                    round(raw_kh550[s]["R_mean"], 6),
                    round(raw_kh570[s]["R_mean"] / raw_bare[s]["R_mean"], 3),
                    round(raw_kh560[s]["R_mean"] / raw_bare[s]["R_mean"], 3),
                    round(raw_kh550[s]["R_mean"] / raw_bare[s]["R_mean"], 3),
                ]
            )

    # 两张目标图（柱状图 & 线性拟合折线图）已在归一化数据段生成

    print(f"\nwrote {OUT / 'pull_comparison_normalized.csv'}")
    print(f"wrote {OUT / 'pull_comparison.csv'}")
    print(f"\n{'speed':>5} {'bare F/n':>9} {'KH570 F/n':>10} {'KH560 F/n':>10} {'KH550 F/n':>10} "
          f"{'bare W/n':>9} {'KH570 W/n':>10} {'KH560 W/n':>10} {'KH550 W/n':>10}")
    for s in speeds:
        print(
            f"{s:>5} {bare[s]['F_mean']:9.3f} {kh570[s]['F_mean']:10.3f} "
            f"{kh560[s]['F_mean']:10.3f} {kh550[s]['F_mean']:10.3f} "
            f"{bare[s]['W_mean']:9.1f} {kh570[s]['W_mean']:10.1f} "
            f"{kh560[s]['W_mean']:10.1f} {kh550[s]['W_mean']:10.1f}"
        )


if __name__ == "__main__":
    main()
