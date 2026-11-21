#!/usr/bin/env python3
"""Parse LAMMPS rdf_avg.txt (54 FKM-grafted SiO2 pairs) and make plots."""

import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


WORK = Path(__file__).resolve().parent
RDF_FILE = WORK / "rdf_avg.txt"

TYPE_NAMES = {
    1: "c", 2: "c1", 3: "c2", 4: "c3", 6: "f", 8: "hc",
    5: "c3h", 7: "h*", 9: "o", 10: "o3e", 11: "oc",
    12: "si", 13: "c2_KH560", 14: "c3_KH560", 15: "hc_KH560",
}
FKM_TYPES = [1, 2, 3, 4, 6, 8]
SIO2_TYPES = [5, 7, 9, 10, 11, 12, 13, 14, 15]
PAIRS = [(a, b) for a in FKM_TYPES for b in SIO2_TYPES]
LABELS = [f"{TYPE_NAMES[a]}-{TYPE_NAMES[b]}" for a, b in PAIRS]

SELECTED = [
    (6, 7, "FKM F - SiO2 h*", "tab:blue"),
    (6, 9, "FKM F - SiO2 o", "tab:orange"),
    (8, 9, "FKM hc - SiO2 o", "tab:green"),
    (8, 10, "FKM hc - KH560 o3e", "tab:red"),
    (6, 10, "FKM F - KH560 o3e", "tab:purple"),
    (1, 12, "FKM c - SiO2 si", "tab:brown"),
]


def load_blocks(path):
    lines = path.read_text(encoding="ascii", errors="ignore").splitlines()
    starts = []
    for i, line in enumerate(lines):
        s = line.strip()
        if s and not s.startswith("#") and len(s.split()) == 2:
            try:
                float(s.split()[0])
                float(s.split()[1])
            except ValueError:
                continue
            starts.append(i)
    if not starts:
        raise SystemExit("no rdf blocks found")
    return lines, starts[-1]


def main():
    if not RDF_FILE.exists():
        raise SystemExit(f"missing {RDF_FILE}")
    lines, start = load_blocks(RDF_FILE)

    rows = []
    for line in lines[start + 1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            continue
        rows.append(vals)
    arr = np.asarray(rows)
    if arr.shape[1] != 2 + 2 * len(PAIRS):
        raise SystemExit(
            f"column mismatch: {arr.shape[1]} vs expected {2 + 2 * len(PAIRS)}"
        )

    r = arr[:, 1]
    vals = arr[:, 2:]
    g = vals[:, 0::2]
    n = vals[:, 1::2]
    print(f"rows={len(r)} pairs={g.shape[1]}")

    csv_path = WORK / "rdf_g_r_last50.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["r_A"] + [f"g_{lab}" for lab in LABELS]
                        + [f"n_{lab}" for lab in LABELS])
        for i in range(len(r)):
            writer.writerow([f"{r[i]:.4f}"] + [f"{v:.6f}" for v in g[i]]
                            + [f"{v:.6f}" for v in n[i]])

    sel_idx = [PAIRS.index((a, b)) for a, b, _, _ in SELECTED]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for idx, (_, _, lab, color) in zip(sel_idx, SELECTED):
        ax.plot(r, g[:, idx], lw=1.4, color=color, label=lab)
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r)")
    ax.set_title("KH560体系 FKM-接枝SiO2 关键原子对 RDF（最后50帧平均）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(WORK / "rdf_selected_pairs.png", dpi=160)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8.5, 5.5))
    for idx, (_, _, lab, color) in zip(sel_idx, SELECTED):
        ax2.plot(r, n[:, idx], lw=1.4, color=color, label=lab)
    ax2.set_xlabel("r (Å)")
    ax2.set_ylabel("n(r)")
    ax2.set_title("配位数 n(r)（最后50帧平均）")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(WORK / "rdf_selected_nr.png", dpi=160)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for idx, lab in enumerate(LABELS):
        ax3.plot(r, g[:, idx], lw=0.8, label=lab)
    ax3.set_xlabel("r (Å)")
    ax3.set_ylabel("g(r)")
    ax3.set_title("FKM-接枝SiO2 全部54对 RDF")
    ax3.legend(fontsize=5, ncol=4)
    fig3.tight_layout()
    fig3.savefig(WORK / "rdf_all_pairs.png", dpi=160)
    plt.close(fig3)

    print("selected pairs peak r / g(r):")
    mask = (r >= 1.0) & (r <= 5.0)
    for idx, (_, _, lab, _) in zip(sel_idx, SELECTED):
        sub = g[mask, idx]
        im = int(np.nonzero(mask)[0][int(np.argmax(sub))])
        print(f"{lab}: peak r={r[im]:.3f} A, g={g[im, idx]:.4f}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
