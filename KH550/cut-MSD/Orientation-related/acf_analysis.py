#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KWW / exponential fitting for the KH550 orientation ACF.

Reads orientation_acf_3000ps.csv and writes kww_fitting_results.csv plus
two figures.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import gamma
import warnings

warnings.filterwarnings("ignore")


def exp_decay(t, A, tau, C):
    return A * np.exp(-t / tau) + C


def kww(t, A, tau, beta, C):
    return A * np.exp(-(t / tau) ** beta) + C


def main():
    df = pd.read_csv("orientation_acf_3000ps.csv")
    t = df["dt_ps"].values
    t_max = t[-1]
    columns = ["contact_C2", "interface_C2", "bulk_C2"]
    labels = {
        "contact_C2": "Contact Layer",
        "interface_C2": "Interface Layer",
        "bulk_C2": "Bulk",
    }
    colors = {
        "contact_C2": "#E74C3C",
        "interface_C2": "#3498DB",
        "bulk_C2": "#2ECC71",
    }

    print("=" * 60)
    print("Orientation ACF KWW Analysis (KH550)")
    print(f"time range 1-{int(t_max)} ps, {len(df)} points")
    print("=" * 60)

    rows = []
    fit_curves = {}
    for col in columns:
        y = df[col].values
        y0 = y[0]
        C_guess = np.mean(y[-100:])
        A_guess = y0 - C_guess
        target = A_guess / np.e + C_guess
        idx = np.where(y <= target)[0]
        tau_guess = t[idx[0]] if len(idx) > 0 else 500.0

        cross_1e = np.where(y < 1.0 / np.e)[0]
        tau_1e = int(cross_1e[0] + 1) if len(cross_1e) else None

        mask = t <= min(2500, t_max)
        try:
            popt, pcov = curve_fit(
                kww,
                t[mask],
                y[mask],
                p0=[A_guess, tau_guess, 0.8, C_guess],
                bounds=([0, 1, 0.1, 0], [2, 10000, 1.0, 1.0]),
                maxfev=50000,
            )
            A_fit, tau_fit, beta_fit, C_fit = popt
            perr = np.sqrt(np.diag(pcov))
            tau_mean = tau_fit / beta_fit * gamma(1.0 / beta_fit)
            y_fit = kww(t[mask], *popt)
            ss_res = np.sum((y[mask] - y_fit) ** 2)
            ss_tot = np.sum((y[mask] - np.mean(y[mask])) ** 2)
            r2 = 1 - ss_res / ss_tot

            if tau_fit > 1.5 * t_max or tau_mean > 1.5 * t_max:
                note = "extrapolated: not fully relaxed in 3000 ps"
            else:
                note = ""
            rows.append(
                {
                    "Region": labels[col],
                    "Column": col,
                    "A": round(A_fit, 6),
                    "tau_KWW_ps": round(tau_fit, 2),
                    "beta": round(beta_fit, 4),
                    "C_plateau": round(C_fit, 6),
                    "tau_mean_ps": round(tau_mean, 2),
                    "tau_mean_ns": round(tau_mean / 1000.0, 4),
                    "tau_data_1e_ps": tau_1e if tau_1e else "NOT_REACHED",
                    "R_squared": round(r2, 6),
                    "Note": note,
                }
            )
            fit_curves[col] = {
                "t_fine": np.linspace(0, t_max, 3001),
                "y_fit": kww(np.linspace(0, t_max, 3001), *popt),
                "popt": popt,
                "tau_mean": tau_mean,
            }
            print(
                f"\n{labels[col]}: A={A_fit:.4f}, tau={tau_fit:.1f} ps, "
                f"beta={beta_fit:.3f}, C={C_fit:.4f}, <tau>={tau_mean:.1f} ps, "
                f"tau_1e={tau_1e}, R2={r2:.4f}"
            )
        except Exception as exc:
            print(f"\n{labels[col]}: fitting failed - {exc}")
            rows.append(
                {
                    "Region": labels[col],
                    "Column": col,
                    "A": np.nan,
                    "tau_KWW_ps": np.nan,
                    "beta": np.nan,
                    "C_plateau": np.nan,
                    "tau_mean_ps": np.nan,
                    "tau_mean_ns": np.nan,
                    "tau_data_1e_ps": tau_1e if tau_1e else "NOT_REACHED",
                    "R_squared": np.nan,
                    "Note": str(exc),
                }
            )

    res = pd.DataFrame(rows)
    res.to_csv("kww_fitting_results.csv", index=False, encoding="utf-8-sig")
    print("\nSaved: kww_fitting_results.csv")
    print(res.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]
    for col in columns:
        y = df[col].values
        ax1.plot(t, y, color=colors[col], lw=1.2, label=labels[col])
        mask = t <= 1000
        try:
            popt_e, _ = curve_fit(
                exp_decay, t[mask], y[mask], p0=[0.3, 300, 0.4], maxfev=10000
            )
            ax1.plot(
                t,
                exp_decay(t, *popt_e),
                "--",
                color=colors[col],
                alpha=0.6,
                label=f"{labels[col]} exp tau={popt_e[1]:.0f} ps",
            )
        except Exception:
            pass
    ax1.set_xlabel("Time Lag (ps)")
    ax1.set_ylabel("C2(t)")
    ax1.set_title("ACF Data vs Exponential Fit")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1500)

    ax2 = axes[1]
    for col in columns:
        y = df[col].values
        ax2.plot(t, y, color=colors[col], lw=1.2, label=labels[col])
    ax2.set_xlabel("Time Lag (ps)")
    ax2.set_ylabel("C2(t)")
    ax2.set_title("Long-time Decay (KH550)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, t_max)
    plt.tight_layout()
    plt.savefig("acf_exp_fit.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: acf_exp_fit.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col in zip(axes, columns):
        y = df[col].values
        ax.scatter(t[::10], y[::10], s=8, color=colors[col], alpha=0.5, label="Data")
        if col in fit_curves:
            fd = fit_curves[col]
            ax.plot(fd["t_fine"], fd["y_fit"], "k-", lw=2, label="KWW fit")
            A, tau, beta, C = fd["popt"]
            ax.text(
                0.05,
                0.05,
                f"A={A:.3f}\ntau={tau:.1f} ps\nbeta={beta:.3f}\n"
                f"C={C:.3f}\n<tau>={fd['tau_mean']:.1f} ps",
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )
        ax.set_xlabel("Time Lag (ps)")
        ax.set_ylabel("C2(t)")
        ax.set_title(labels[col])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, t_max)
    plt.tight_layout()
    plt.savefig("acf_kww_fit.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: acf_kww_fit.png")


if __name__ == "__main__":
    main()

