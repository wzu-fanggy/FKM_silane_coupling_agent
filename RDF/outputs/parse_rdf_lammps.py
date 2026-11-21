import csv
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


RDF_FILE = Path(r"F:\Lammps-MS\FKM-SIO2-TEST\RDF\outputs\rdf_avg.txt")
OUT_DIR = Path(r"C:\Users\Hxic\Documents\ChatGPT\lammps\rdf_lammps_out")

PAIRS = [
    (1, 7), (1, 8), (1, 9),
    (2, 7), (2, 8), (2, 9),
    (3, 7), (3, 8), (3, 9),
    (4, 7), (4, 8), (4, 9),
    (5, 7), (5, 8), (5, 9),
    (6, 7), (6, 8), (6, 9),
]
LABELS = [f"{i}-{j}" for i, j in PAIRS]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = RDF_FILE.read_text(encoding="ascii", errors="ignore").splitlines()
    block_starts = [i for i, l in enumerate(lines) if l and not l.startswith("#") and len(l.split()) == 2]
    start = block_starts[-1]
    rows = []
    for l in lines[start + 1 : start + 101]:
        p = [float(x) for x in l.split()]
        rows.append(p)

    r = np.array([row[1] for row in rows])
    vals = np.array([row[2:] for row in rows])
    g = vals[:, 0::2]
    n = vals[:, 1::2]

    with open(OUT_DIR / "rdf_g_r_last50.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["r_A"] + [f"g_{lab}" for lab in LABELS] + [f"n_{lab}" for lab in LABELS])
        for i in range(len(r)):
            writer.writerow([f"{r[i]:.4f}"] + [f"{v:.6f}" for v in g[i]] + [f"{v:.6f}" for v in n[i]])

    # Selected key pairs for plots
    selected = [(5, 7), (5, 8), (6, 8), (1, 8)]
    sel_idx = [LABELS.index(f"{i}-{j}") for i, j in selected]
    sel_labels = ["FKM F - SiO2 H", "FKM F - SiO2 O", "FKM H - SiO2 O", "FKM C1 - SiO2 O"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, lab, c in zip(sel_idx, sel_labels, colors):
        ax.plot(r, g[:, idx], lw=1.4, color=c, label=lab)
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r)")
    ax.set_title("FKM-SiO2 关键原子对 RDF（LAMMPS, 最后50帧平均）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rdf_selected_pairs.png", dpi=160)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for idx, lab, c in zip(sel_idx, sel_labels, colors):
        ax2.plot(r, n[:, idx], lw=1.4, color=c, label=lab)
    ax2.set_xlabel("r (Å)")
    ax2.set_ylabel("n(r)")
    ax2.set_title("配位数 n(r)（LAMMPS, 最后50帧平均）")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "rdf_selected_nr.png", dpi=160)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    for idx, lab in enumerate(LABELS):
        ax3.plot(r, g[:, idx], lw=0.9, label=lab)
    ax3.set_xlabel("r (Å)")
    ax3.set_ylabel("g(r)")
    ax3.set_title("FKM-SiO2 全部18对 RDF")
    ax3.legend(fontsize=6, ncol=3)
    fig3.tight_layout()
    fig3.savefig(OUT_DIR / "rdf_all_pairs.png", dpi=160)
    plt.close(fig3)

    print("selected pairs peak r / g(r):")
    mask = (r >= 1.0) & (r <= 4.0)
    for idx, lab in zip(sel_idx, sel_labels):
        sub = g[mask, idx]
        imax = int(np.argmax(sub))
        imax = int(np.nonzero(mask)[0][imax])
        print(f"{lab}: peak r={r[imax]:.3f} A, g={g[imax, idx]:.4f}")
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
