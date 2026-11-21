#!/usr/bin/env python3
"""Analyze completed KH550 pull-out runs in the rate series.

Input : speed folders (2A/5A/10A/...) with 1/2/3/force_curve.txt
        columns: step disp force E_gg
Output: analysis/pull_rate_summary.csv
        analysis/speed_summary.csv
        analysis/{speed}_force_bin3.png / _bin5.png
        analysis/peak_force_vs_rate.png
        analysis/work_vs_rate.png
        analysis/residual_vs_rate.png
"""

import csv
import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)

BINS = (3.0, 5.0)


def read_force(path):
    rows = []
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            q = s.split()
            if len(q) >= 4:
                rows.append((float(q[0]), float(q[1]), float(q[2]), float(q[3])))
    return np.asarray(rows, dtype=float)


def trapz(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def frame_stats(speed_dir, frame_id):
    fc = speed_dir / str(frame_id) / "force_curve.txt"
    if not fc.exists():
        return None
    arr = read_force(fc)
    step, disp, force, egg = arr.T
    fmax = float(force.max())
    work = float(trapz(force, disp))
    e0 = float(egg[0])
    ef = float(egg[-1])
    resid = ef / e0 if e0 != 0 else math.nan
    return {
        "speed_Aps": speed_dir.name.replace("A", ""),
        "frame": frame_id,
        "F_max_kcal_mol_Ang": round(fmax, 4),
        "W_sep_kcal_mol": round(work, 4),
        "E_gg0_kcal_mol": round(e0, 4),
        "E_ggf_kcal_mol": round(ef, 4),
        "residual_fraction": round(resid, 6),
        "displacement_A": round(float(disp[-1]), 4),
    }


def bin_curve(speed_dir, frames, bw):
    groups = {}
    for i in frames:
        arr = read_force(speed_dir / str(i) / "force_curve.txt")
        disp = arr[:, 1]
        force = arr[:, 2]
        for d, f in zip(disp, force):
            key = math.floor(d / bw) * bw
            groups.setdefault(key, []).append(f)
    keys = sorted(groups)
    xs = [k + bw / 2 for k in keys]
    means = [float(np.mean(groups[k])) for k in keys]
    stds = [float(np.std(groups[k])) for k in keys]
    return np.asarray(xs), np.asarray(means), np.asarray(stds)


def render_speed_figures(speed_dir, frames):
    speed = speed_dir.name
    for bw in BINS:
        xs, means, stds = bin_curve(speed_dir, frames, bw)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, means, "-o", ms=3, color="tab:blue", label=f"{bw:.0f} A bin")
        ax.fill_between(
            xs,
            means - stds,
            means + stds,
            color="tab:blue",
            alpha=0.25,
            label="+/- 1 SD",
        )
        ax.set_xlabel("SiO2 displacement (A)")
        ax.set_ylabel("Pull force (kcal/mol/A)")
        ax.set_title(f"KH550 uniaxial pull {speed} ({bw:.0f} A bin)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"{speed}_force_bin{bw:.0f}.png", dpi=160)
        plt.close(fig)


def main():
    speed_dirs = []
    seen = set()
    for path in ROOT.rglob("*"):
        if not path.is_dir():
            continue
        if not path.name.endswith("A") or not path.name[:-1].isdigit():
            continue
        if path in seen:
            continue
        frames = sorted(
            [
                int(p.name)
                for p in path.iterdir()
                if p.is_dir() and p.name.isdigit() and (p / "force_curve.txt").exists()
            ]
        )
        if frames:
            speed_dirs.append((path, frames))
            seen.add(path)
    speed_dirs.sort(key=lambda item: int(item[0].name[:-1]))

    if not speed_dirs:
        raise SystemExit("no completed force curves found")

    per_frame = []
    speed_stats = []
    for speed_dir, frames in speed_dirs:
        speed = speed_dir.name
        render_speed_figures(speed_dir, frames)
        stats = [frame_stats(speed_dir, i) for i in frames]
        stats = [s for s in stats if s is not None]
        per_frame.extend(stats)

        fmax = np.mean([s["F_max_kcal_mol_Ang"] for s in stats])
        work = np.mean([s["W_sep_kcal_mol"] for s in stats])
        resid = np.mean([s["residual_fraction"] for s in stats])
        e0 = np.mean([s["E_gg0_kcal_mol"] for s in stats])
        ef = np.mean([s["E_ggf_kcal_mol"] for s in stats])
        speed_stats.append(
            {
                "speed_Aps": speed.replace("A", ""),
                "n_frames": len(stats),
                "F_max_mean": round(float(fmax), 4),
                "F_max_std": round(float(np.std([s["F_max_kcal_mol_Ang"] for s in stats])), 4),
                "W_sep_mean": round(float(work), 4),
                "W_sep_std": round(float(np.std([s["W_sep_kcal_mol"] for s in stats])), 4),
                "E_gg0_mean": round(float(e0), 4),
                "E_ggf_mean": round(float(ef), 4),
                "residual_mean": round(float(resid), 6),
                "residual_std": round(float(np.std([s["residual_fraction"] for s in stats])), 6),
            }
        )

    with open(OUT / "pull_rate_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_frame[0].keys()))
        writer.writeheader()
        writer.writerows(per_frame)

    with open(OUT / "speed_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(speed_stats[0].keys()))
        writer.writeheader()
        writer.writerows(speed_stats)

    rates = np.asarray([float(s["speed_Aps"]) for s in speed_stats])
    fmax = np.asarray([s["F_max_mean"] for s in speed_stats])
    work = np.asarray([s["W_sep_mean"] for s in speed_stats])
    resid = np.asarray([s["residual_mean"] for s in speed_stats])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(rates, fmax, yerr=[s["F_max_std"] for s in speed_stats], fmt="-o", capsize=4)
    ax.set_xlabel("Pull rate (A/ps)")
    ax.set_ylabel("Peak force (kcal/mol/A)")
    ax.set_title("KH550 peak force vs pull rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "peak_force_vs_rate.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(rates, work, yerr=[s["W_sep_std"] for s in speed_stats], fmt="-o", capsize=4)
    ax.set_xlabel("Pull rate (A/ps)")
    ax.set_ylabel("Work of separation (kcal/mol)")
    ax.set_title("KH550 work of separation vs pull rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "work_vs_rate.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(rates, resid, yerr=[s["residual_std"] for s in speed_stats], fmt="-o", capsize=4)
    ax.set_xlabel("Pull rate (A/ps)")
    ax.set_ylabel("E_gg residual fraction")
    ax.set_title("KH550 interface energy residual vs pull rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "residual_vs_rate.png", dpi=160)
    plt.close(fig)

    print(f"wrote {OUT}")
    print()
    for s in speed_stats:
        print(
            f"{s['speed_Aps']:>3} A/ps: F_max={s['F_max_mean']:.1f}+/-{s['F_max_std']:.1f} "
            f"| W={s['W_sep_mean']:.0f}+/-{s['W_sep_std']:.0f} "
            f"| residual={s['residual_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
