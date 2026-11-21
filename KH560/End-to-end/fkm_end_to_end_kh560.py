#!/usr/bin/env python3
"""FKM end-to-end distance for the KH560-grafted system.

Same algorithm as the KH570 / bare-system analysis:
  - main chain bond types 1, 2, 3, 5
  - two BFS passes to find the chain diameter
  - PBC minimum-image distance between the two endpoints
  - FKM molecules are mol 2..121
"""

import argparse
import statistics
from collections import defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


MAIN_CHAIN_BONDS = {1, 2, 3, 5}


def parse_data(filename):
    lines = Path(filename).read_text(encoding="ascii", errors="ignore").splitlines()
    box = None
    for line in lines:
        if "xlo xhi" in line:
            p = line.split()
            box = np.array([float(p[1]) - float(p[0]), 0.0, 0.0])
        elif "ylo yhi" in line:
            p = line.split()
            box[1] = float(p[1]) - float(p[0])
        elif "zlo zhi" in line:
            p = line.split()
            box[2] = float(p[1]) - float(p[0])

    atoms_start = None
    bonds_start = None
    for i, line in enumerate(lines):
        if "Atoms # full" in line:
            atoms_start = i + 1
        if line.strip() == "Bonds":
            bonds_start = i + 1
            break

    atoms = {}
    mol_atoms = defaultdict(list)
    for line in lines[atoms_start:]:
        if not line.strip():
            continue
        if line.strip().startswith("Angles"):
            break
        p = line.split()
        if len(p) >= 7:
            atoms[int(p[0])] = {
                "mol": int(p[1]),
                "type": int(p[2]),
                "x": float(p[4]),
                "y": float(p[5]),
                "z": float(p[6]),
            }
            mol_atoms[int(p[1])].append(int(p[0]))

    bonds = []
    for line in lines[bonds_start:]:
        if not line.strip():
            continue
        if line.strip().startswith("Angles"):
            break
        p = line.split()
        if len(p) >= 4:
            bonds.append({"type": int(p[1]), "a1": int(p[2]), "a2": int(p[3])})
    return atoms, bonds, mol_atoms, box


def bfs_farthest(adj, start):
    visited = {start: 0}
    queue = deque([start])
    farthest, max_d = start, 0
    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if nb not in visited:
                visited[nb] = visited[node] + 1
                queue.append(nb)
                if visited[nb] > max_d:
                    max_d = visited[nb]
                    farthest = nb
    return farthest, max_d


def calc_ete(mol_ids, bonds, atoms, box):
    adj = defaultdict(list)
    for b in bonds:
        if b["a1"] in mol_ids and b["a2"] in mol_ids and b["type"] in MAIN_CHAIN_BONDS:
            adj[b["a1"]].append(b["a2"])
            adj[b["a2"]].append(b["a1"])
    if len(adj) < 2:
        return None
    start = list(adj.keys())[0]
    far1, _ = bfs_farthest(adj, start)
    far2, path = bfs_farthest(adj, far1)
    r1 = np.array([atoms[far1]["x"], atoms[far1]["y"], atoms[far1]["z"]])
    r2 = np.array([atoms[far2]["x"], atoms[far2]["y"], atoms[far2]["z"]])
    dr = r2 - r1 - box * np.round((r2 - r1) / box)
    return np.linalg.norm(dr), path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="?", default=r"F:\Lammps-MS\FKM-SIO2-TEST\KH560\End-to-end\nvt_500ps.data")
    parser.add_argument("--mol-start", type=int, default=2)
    parser.add_argument("--mol-end", type=int, default=121)
    parser.add_argument("--out", default=r"F:\Lammps-MS\FKM-SIO2-TEST\KH560\End-to-end\fkm_ete_results.txt")
    parser.add_argument("--png", default=r"F:\Lammps-MS\FKM-SIO2-TEST\KH560\End-to-end\fkm_ete_distribution.png")
    args = parser.parse_args()

    atoms, bonds, mol_atoms, box = parse_data(args.data)
    fkm_ids = sorted(m for m in mol_atoms if args.mol_start <= m <= args.mol_end)

    rows = []
    for mol_id in fkm_ids:
        mol_set = set(mol_atoms[mol_id])
        res = calc_ete(mol_set, bonds, atoms, box)
        if res is None:
            continue
        ete, path = res
        rows.append({"mol_id": mol_id, "path": path, "ete": ete})

    etes = [r["ete"] for r in rows]
    summary = {
        "mean": statistics.mean(etes),
        "std": statistics.stdev(etes),
        "min": min(etes),
        "max": max(etes),
        "median": statistics.median(etes),
        "rms": float(np.sqrt(np.mean(np.square(etes)))),
    }

    out = Path(args.out)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# KH560 FKM end-to-end distance results\n")
        f.write("# main chain bonds: 1,2,3,5 | FKM molecules: %d..%d\n" % (args.mol_start, args.mol_end))
        f.write("# mean=%.4f std=%.4f min=%.4f max=%.4f median=%.4f rms=%.4f\n" % (
            summary["mean"], summary["std"], summary["min"],
            summary["max"], summary["median"], summary["rms"]))
        f.write("mol_id,path_len,ete_A\n")
        for r in rows:
            f.write("%d,%d,%.4f\n" % (r["mol_id"], r["path"], r["ete"]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(etes, bins=25, color="#DD8452", edgecolor="white", alpha=0.9)
    ax.axvline(summary["mean"], color="#2F3B52", ls="--", lw=1.2,
               label="mean = %.2f A" % summary["mean"])
    ax.set_xlabel("End-to-end distance (Å)")
    ax.set_ylabel("Count")
    ax.set_title("KH560 system FKM end-to-end distance distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.png, dpi=160)
    plt.close(fig)

    print("FKM chains: %d" % len(rows))
    print("mean=%.4f std=%.4f min=%.4f max=%.4f median=%.4f rms=%.4f" % (
        summary["mean"], summary["std"], summary["min"],
        summary["max"], summary["median"], summary["rms"]))
    print("Wrote %s" % out)
    print("Wrote %s" % args.png)


if __name__ == "__main__":
    main()