"""Hydrogen-bond analysis for FKM/SiO2 from a LAMMPS dump.

Pairs:
  A: SiO2 O-H donor ... FKM F acceptor
  B: FKM C-H donor ... SiO2 O acceptor

Criteria:
  H...A distance <= 3.0 A, D-H...A angle >= 120 deg
  Sensitivity run with 3.5 A / 150 deg is also counted.
"""

import csv
import statistics
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


DUMP = Path(r"F:\Lammps-MS\FKM-SIO2-TEST\H\last50.dump")
OUT_DIR = Path(r"C:\Users\Hxic\Documents\ChatGPT\lammps\hbond_out")


def parse_dump(path):
    frames = []
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(f.readline().strip())
            f.readline()  # NUMBER OF ATOMS
            n_atoms = int(f.readline().strip())
            f.readline()  # BOX BOUNDS
            bounds = []
            for _ in range(3):
                bounds.append([float(x) for x in f.readline().split()])
            f.readline()  # ATOMS header
            block = [f.readline().strip() for _ in range(n_atoms)]
            arr = np.loadtxt(block).reshape(-1, 8)
            lengths = np.array([b[1] - b[0] for b in bounds])
            xyz = arr[:, 2:5] + arr[:, 5:8] * lengths
            frames.append(
                {
                    "step": step,
                    "time_ps": step * 0.001,
                    "types": arr[:, 1].astype(np.int64),
                    "xyz": xyz,
                    "L": lengths,
                }
            )
    return frames


def min_image_dx(dx, L):
    return dx - L * np.round(dx / L)


def nearest_donor_heavy(h_xyz, heavy_xyz, L):
    """Return donor heavy atom coordinates for each H (nearest within 1.3 A)."""
    out = np.full((len(h_xyz), 3), np.nan)
    for start in range(0, len(h_xyz), 512):
        hb = h_xyz[start : start + 512]
        dx = hb[:, None, 0] - heavy_xyz[None, :, 0]
        dy = hb[:, None, 1] - heavy_xyz[None, :, 1]
        dz = hb[:, None, 2] - heavy_xyz[None, :, 2]
        dx = min_image_dx(dx, L[0])
        dy = min_image_dx(dy, L[1])
        dz = min_image_dx(dz, L[2])
        d2 = dx * dx + dy * dy + dz * dz
        idx = np.argmin(d2, axis=1)
        best = np.min(d2, axis=1)
        valid = best < 1.3 * 1.3
        for j in range(len(hb)):
            if valid[j]:
                out[start + j] = heavy_xyz[idx[j]]
    return out


def count_hbonds(h_xyz, d_xyz, a_xyz, L, r_cut=3.0, ang_min=120.0):
    count = 0
    hb_list = []
    for start in range(0, len(h_xyz), 512):
        hb = h_xyz[start : start + 512]
        db = d_xyz[start : start + 512]
        dx = hb[:, None, 0] - a_xyz[None, :, 0]
        dy = hb[:, None, 1] - a_xyz[None, :, 1]
        dz = hb[:, None, 2] - a_xyz[None, :, 2]
        dx = min_image_dx(dx, L[0])
        dy = min_image_dx(dy, L[1])
        dz = min_image_dx(dz, L[2])
        d2 = dx * dx + dy * dy + dz * dz
        idx = np.argwhere(d2 < r_cut * r_cut)
        for i, j in idx:
            h = hb[i]
            d = db[i]
            a = a_xyz[j]
            v_ha = np.array(
                [min_image_dx(a[0] - h[0], L[0]), min_image_dx(a[1] - h[1], L[1]), min_image_dx(a[2] - h[2], L[2])]
            )
            v_hd = np.array(
                [min_image_dx(d[0] - h[0], L[0]), min_image_dx(d[1] - h[1], L[1]), min_image_dx(d[2] - h[2], L[2])]
            )
            n_ha = np.linalg.norm(v_ha)
            n_hd = np.linalg.norm(v_hd)
            if n_ha == 0 or n_hd == 0:
                continue
            cosang = np.dot(v_ha, v_hd) / (n_ha * n_hd)
            angle = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
            if angle >= ang_min:
                count += 1
                hb_list.append((n_ha, angle))
    return count, hb_list


def rdf_hist(h_xyz, a_xyz, L, rmax=6.0, dr=0.1):
    counts = np.zeros(int(rmax / dr))
    for start in range(0, len(h_xyz), 512):
        hb = h_xyz[start : start + 512]
        dx = hb[:, None, 0] - a_xyz[None, :, 0]
        dy = hb[:, None, 1] - a_xyz[None, :, 1]
        dz = hb[:, None, 2] - a_xyz[None, :, 2]
        dx = min_image_dx(dx, L[0])
        dy = min_image_dx(dy, L[1])
        dz = min_image_dx(dz, L[2])
        dist = np.sqrt(dx * dx + dy * dy + dz * dz).ravel()
        bins = (dist / dr).astype(int)
        mask = bins < len(counts)
        np.add.at(counts, bins[mask], 1)
    return counts


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = parse_dump(DUMP)
    print(f"frames: {len(frames)}")

    rows = []
    rdf_a = None
    rdf_b = None
    n_frames = 0
    for frame in frames:
        types = frame["types"]
        xyz = frame["xyz"]
        L = frame["L"]
        fkm_c = xyz[np.isin(types, [1, 2, 3, 4])]
        fkm_f = xyz[types == 5]
        fkm_h = xyz[types == 6]
        sio2_o = xyz[types == 8]
        sio2_h = xyz[types == 7]

        # Pair A: SiO2 O-H ... FKM F
        dA = nearest_donor_heavy(sio2_h, sio2_o, L)
        validA = ~np.isnan(dA[:, 0])
        nA, _ = count_hbonds(sio2_h[validA], dA[validA], fkm_f, L, 3.0, 120.0)
        nA_strict, _ = count_hbonds(sio2_h[validA], dA[validA], fkm_f, L, 3.0, 150.0)
        nA_loose, _ = count_hbonds(sio2_h[validA], dA[validA], fkm_f, L, 3.5, 120.0)

        # Pair B: FKM C-H ... SiO2 O
        dB = nearest_donor_heavy(fkm_h, fkm_c, L)
        validB = ~np.isnan(dB[:, 0])
        nB, _ = count_hbonds(fkm_h[validB], dB[validB], sio2_o, L, 3.0, 120.0)
        nB_strict, _ = count_hbonds(fkm_h[validB], dB[validB], sio2_o, L, 3.0, 150.0)
        nB_loose, _ = count_hbonds(fkm_h[validB], dB[validB], sio2_o, L, 3.5, 120.0)

        rows.append(
            {
                "step": frame["step"],
                "time_ps": frame["time_ps"],
                "A_SiOH_F": nA,
                "A_SiOH_F_150": nA_strict,
                "A_SiOH_F_3.5": nA_loose,
                "B_FKMH_SiO2O": nB,
                "B_FKMH_SiO2O_150": nB_strict,
                "B_FKMH_SiO2O_3.5": nB_loose,
                "total_120_3.0": nA + nB,
            }
        )

        ra = rdf_hist(sio2_h[validA], fkm_f, L)
        rb = rdf_hist(fkm_h[validB], sio2_o, L)
        rdf_a = ra if rdf_a is None else rdf_a + ra
        rdf_b = rb if rdf_b is None else rdf_b + rb
        n_frames += 1

    with open(OUT_DIR / "hbond_last50.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    col_a = [r["A_SiOH_F"] for r in rows]
    col_b = [r["B_FKMH_SiO2O"] for r in rows]
    summary = {
        "A_SiOH_F_mean": statistics.mean(col_a),
        "A_SiOH_F_std": statistics.stdev(col_a),
        "B_FKMH_SiO2O_mean": statistics.mean(col_b),
        "B_FKMH_SiO2O_std": statistics.stdev(col_b),
        "total_mean": statistics.mean([r["total_120_3.0"] for r in rows]),
        "total_std": statistics.stdev([r["total_120_3.0"] for r in rows]),
    }
    print(summary)
    with open(OUT_DIR / "hbond_last50_summary.txt", "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k} = {v}\n")

    times = [r["time_ps"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, col_a, "-o", ms=3, label="SiO2 O-H ... FKM F")
    ax.plot(times, col_b, "-o", ms=3, label="FKM C-H ... SiO2 O")
    ax.set_xlabel("时间 (ps)")
    ax.set_ylabel("氢键数目")
    ax.set_title("FKM-SiO2 界面氢键数目（最后50帧, 3.0 Å / 120°）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hbond_counts.png", dpi=160)
    plt.close(fig)

    dr = 0.1
    r = (np.arange(len(rdf_a)) + 0.5) * dr
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(r, rdf_a / n_frames, label="SiO2 O-H ... FKM F")
    ax2.plot(r, rdf_b / n_frames, label="FKM C-H ... SiO2 O")
    ax2.set_xlabel("H···A 距离 (Å)")
    ax2.set_ylabel("平均每帧原子对数")
    ax2.set_title("H···A 距离分布（最后50帧）")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "hbond_rdf_hist.png", dpi=160)
    plt.close(fig2)
    print(f"Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
