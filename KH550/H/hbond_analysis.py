#!/usr/bin/env python3
"""Redesigned hydrogen-bond analysis for the KH550-grafted system.

Donors
  SiO2 silanol : h* (6), heavy donor = SiO2 O (type 10)
  FKM          : hc (7), heavy donor = FKM C (1,2,3,4)
  KH550 N-H    : hn2 (8), heavy donor = na (9)
  KH550 C-H    : hc_KH550 (14), heavy donor = KH550 C (12,13)

Acceptors
  SiO2 O       : type 10
  KH550 N      : type 9 (na)
  FKM F        : type 5

Counted pairs (3.0 A / 120 deg main criterion):
  A   SiO2 O-H ... FKM F
  B1  FKM C-H ... SiO2 O
  B2  FKM C-H ... KH550 N
  D1  KH550 N-H ... FKM F
  D2  KH550 C-H ... FKM F
  E1  KH550 N-H ... SiO2 O        (intra-filler)
  E2  KH550 C-H ... SiO2 O        (intra-filler)
  F   SiO2 O-H ... KH550 N        (intra-filler)
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


WORK = Path(__file__).resolve().parent
DUMP = WORK / "last50.dump"

FKM_C = [1, 2, 3, 4]
FKM_F = [5]
FKM_H = [7]
SIO2_H = [6]
SIO2_O = [10]
KH550_NH = [8]
KH550_N = [9]
KH550_C = [12, 13]
KH550_H = [14]


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
            f.readline()
            n_atoms = int(f.readline().strip())
            f.readline()
            bounds = [[float(x) for x in f.readline().split()] for _ in range(3)]
            f.readline()
            block = [f.readline().strip() for _ in range(n_atoms)]
            arr = np.loadtxt(block).reshape(-1, 8)
            lengths = np.array([b[1] - b[0] for b in bounds])
            frames.append({
                "step": step,
                "time_ps": step * 0.001,
                "ids": arr[:, 0].astype(np.int64),
                "types": arr[:, 1].astype(np.int64),
                "xyz": arr[:, 2:5] + arr[:, 5:8] * lengths,
                "L": lengths,
            })
    return frames


def min_image_dx(dx, L):
    return dx - L * np.round(dx / L)


def nearest_donor_heavy(h_xyz, heavy_xyz, L):
    out = np.full((len(h_xyz), 3), np.nan)
    for start in range(0, len(h_xyz), 512):
        hb = h_xyz[start:start + 512]
        dx = min_image_dx(hb[:, None, 0] - heavy_xyz[None, :, 0], L[0])
        dy = min_image_dx(hb[:, None, 1] - heavy_xyz[None, :, 1], L[1])
        dz = min_image_dx(hb[:, None, 2] - heavy_xyz[None, :, 2], L[2])
        d2 = dx * dx + dy * dy + dz * dz
        idx = np.argmin(d2, axis=1)
        valid = np.min(d2, axis=1) < 1.3 * 1.3
        for j in range(len(hb)):
            if valid[j]:
                out[start + j] = heavy_xyz[idx[j]]
    return out


def count_hbonds(h_xyz, d_xyz, a_xyz, L, r_cut=3.0, ang_min=120.0):
    count = 0
    hb_list = []
    for start in range(0, len(h_xyz), 512):
        hb = h_xyz[start:start + 512]
        db = d_xyz[start:start + 512]
        dx = min_image_dx(hb[:, None, 0] - a_xyz[None, :, 0], L[0])
        dy = min_image_dx(hb[:, None, 1] - a_xyz[None, :, 1], L[1])
        dz = min_image_dx(hb[:, None, 2] - a_xyz[None, :, 2], L[2])
        d2 = dx * dx + dy * dy + dz * dz
        idx = np.argwhere(d2 < r_cut * r_cut)
        for i, j in idx:
            h = hb[i]
            d = db[i]
            a = a_xyz[j]
            v_ha = np.array([
                min_image_dx(a[0] - h[0], L[0]),
                min_image_dx(a[1] - h[1], L[1]),
                min_image_dx(a[2] - h[2], L[2]),
            ])
            v_hd = np.array([
                min_image_dx(d[0] - h[0], L[0]),
                min_image_dx(d[1] - h[1], L[1]),
                min_image_dx(d[2] - h[2], L[2]),
            ])
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
        hb = h_xyz[start:start + 512]
        dx = min_image_dx(hb[:, None, 0] - a_xyz[None, :, 0], L[0])
        dy = min_image_dx(hb[:, None, 1] - a_xyz[None, :, 1], L[1])
        dz = min_image_dx(hb[:, None, 2] - a_xyz[None, :, 2], L[2])
        dist = np.sqrt(dx * dx + dy * dy + dz * dz).ravel()
        bins = (dist / dr).astype(int)
        mask = bins < len(counts)
        np.add.at(counts, bins[mask], 1)
    return counts


def main():
    frames = parse_dump(DUMP)
    print(f"frames: {len(frames)}")

    rows = []
    rdf = {k: None for k in ("A", "B1", "B2", "D1", "D2")}
    n_frames = 0
    for frame in frames:
        types = frame["types"]
        xyz = frame["xyz"]
        L = frame["L"]

        fkm_f = xyz[np.isin(types, FKM_F)]
        fkm_h = xyz[np.isin(types, FKM_H)]
        sio2_h = xyz[np.isin(types, SIO2_H)]
        sio2_o = xyz[np.isin(types, SIO2_O)]
        kh550_nh = xyz[np.isin(types, KH550_NH)]
        kh550_n = xyz[np.isin(types, KH550_N)]
        kh550_ch = xyz[np.isin(types, KH550_H)]
        fkm_c = xyz[np.isin(types, FKM_C)]
        kh550_c = xyz[np.isin(types, KH550_C)]

        dA = nearest_donor_heavy(sio2_h, sio2_o, L)
        vA = ~np.isnan(dA[:, 0])
        nA, _ = count_hbonds(sio2_h[vA], dA[vA], fkm_f, L, 3.0, 120.0)
        nA_150, _ = count_hbonds(sio2_h[vA], dA[vA], fkm_f, L, 3.0, 150.0)
        nA_35, _ = count_hbonds(sio2_h[vA], dA[vA], fkm_f, L, 3.5, 120.0)

        dB = nearest_donor_heavy(fkm_h, fkm_c, L)
        vB = ~np.isnan(dB[:, 0])
        nB1, _ = count_hbonds(fkm_h[vB], dB[vB], sio2_o, L, 3.0, 120.0)
        nB1_150, _ = count_hbonds(fkm_h[vB], dB[vB], sio2_o, L, 3.0, 150.0)
        nB1_35, _ = count_hbonds(fkm_h[vB], dB[vB], sio2_o, L, 3.5, 120.0)
        nB2, _ = count_hbonds(fkm_h[vB], dB[vB], kh550_n, L, 3.0, 120.0)

        dD1 = nearest_donor_heavy(kh550_nh, kh550_n, L)
        vD1 = ~np.isnan(dD1[:, 0])
        nD1, _ = count_hbonds(kh550_nh[vD1], dD1[vD1], fkm_f, L, 3.0, 120.0)
        nE1, _ = count_hbonds(kh550_nh[vD1], dD1[vD1], sio2_o, L, 3.0, 120.0)

        dD2 = nearest_donor_heavy(kh550_ch, kh550_c, L)
        vD2 = ~np.isnan(dD2[:, 0])
        nD2, _ = count_hbonds(kh550_ch[vD2], dD2[vD2], fkm_f, L, 3.0, 120.0)
        nE2, _ = count_hbonds(kh550_ch[vD2], dD2[vD2], sio2_o, L, 3.0, 120.0)
        nF, _ = count_hbonds(sio2_h[vA], dA[vA], kh550_n, L, 3.0, 120.0)

        rows.append({
            "step": frame["step"],
            "time_ps": frame["time_ps"],
            "A_SiOH_F": nA,
            "A_SiOH_F_150": nA_150,
            "A_SiOH_F_3.5": nA_35,
            "B1_FKMH_SiO2O": nB1,
            "B1_FKMH_SiO2O_150": nB1_150,
            "B1_FKMH_SiO2O_3.5": nB1_35,
            "B2_FKMH_KH550N": nB2,
            "D1_KH550NH_F": nD1,
            "D2_KH550CH_F": nD2,
            "E1_KH550NH_SiO2O": nE1,
            "E2_KH550CH_SiO2O": nE2,
            "F_SiOH_KH550N": nF,
            "total_interface": nA + nB1 + nB2 + nD1 + nD2,
        })

        for key, (h, a) in {
            "A": (sio2_h[vA], fkm_f),
            "B1": (fkm_h[vB], sio2_o),
            "B2": (fkm_h[vB], kh550_n),
            "D1": (kh550_nh[vD1], fkm_f),
            "D2": (kh550_ch[vD2], fkm_f),
        }.items():
            hist = rdf_hist(h, a, L)
            rdf[key] = hist if rdf[key] is None else rdf[key] + hist
        n_frames += 1

    csv_path = WORK / "hbond_last50.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def stats(key):
        vals = [r[key] for r in rows]
        return statistics.mean(vals), statistics.stdev(vals)

    names = [
        "A_SiOH_F", "B1_FKMH_SiO2O", "B2_FKMH_KH550N",
        "D1_KH550NH_F", "D2_KH550CH_F",
        "E1_KH550NH_SiO2O", "E2_KH550CH_SiO2O", "F_SiOH_KH550N",
        "total_interface",
    ]
    summary = {}
    for name in names:
        mean, std = stats(name)
        summary[f"{name}_mean"] = mean
        summary[f"{name}_std"] = std
    print(summary)
    with open(WORK / "hbond_last50_summary.txt", "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k} = {v}\n")

    times = [r["time_ps"] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    curves = [
        ("A_SiOH_F", "SiO2 O-H ... FKM F", "tab:blue"),
        ("B1_FKMH_SiO2O", "FKM C-H ... SiO2 O", "tab:orange"),
        ("B2_FKMH_KH550N", "FKM C-H ... KH550 N", "tab:green"),
        ("D1_KH550NH_F", "KH550 N-H ... FKM F", "tab:red"),
        ("D2_KH550CH_F", "KH550 C-H ... FKM F", "tab:purple"),
        ("E1_KH550NH_SiO2O", "KH550 N-H ... SiO2 O(内部)", "tab:brown"),
        ("E2_KH550CH_SiO2O", "KH550 C-H ... SiO2 O(内部)", "tab:cyan"),
        ("F_SiOH_KH550N", "SiO2 O-H ... KH550 N(内部)", "tab:pink"),
    ]
    for key, name, color in curves:
        ax.plot(times, [r[key] for r in rows], "-o", ms=3, color=color, label=name)
    ax.set_xlabel("时间 (ps)")
    ax.set_ylabel("氢键数目")
    ax.set_title("KH550接枝体系 氢键数目（最后50帧, 3.0 Å / 120°）")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(WORK / "hbond_counts.png", dpi=160)
    plt.close(fig)

    dr = 0.1
    r = (np.arange(len(rdf["A"])) + 0.5) * dr
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for key, name in [
        ("A", "SiO2 O-H ... FKM F"),
        ("B1", "FKM C-H ... SiO2 O"),
        ("B2", "FKM C-H ... KH550 N"),
        ("D1", "KH550 N-H ... FKM F"),
        ("D2", "KH550 C-H ... FKM F"),
    ]:
        ax2.plot(r, rdf[key] / n_frames, label=name)
    ax2.set_xlabel("H···A 距离 (Å)")
    ax2.set_ylabel("平均每帧原子对数")
    ax2.set_title("H···A 距离分布（最后50帧）")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(WORK / "hbond_rdf_hist.png", dpi=160)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    labels = [name for _, name, _ in curves]
    means = [summary[f"{key}_mean"] for key, _, _ in curves]
    stds = [summary[f"{key}_std"] for key, _, _ in curves]
    colors = [c for _, _, c in curves]
    ax3.bar(range(len(labels)), means, yerr=stds, capsize=5, color=colors)
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax3.set_ylabel("平均氢键数")
    ax3.set_title("KH550接枝体系 氢键均值±标准差")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax3.text(i, m + s + 0.5, f"{m:.1f}", ha="center", fontsize=8)
    fig3.tight_layout()
    fig3.savefig(WORK / "hbond_means.png", dpi=160)
    plt.close(fig3)
    print(f"Outputs in {WORK}")


if __name__ == "__main__":
    main()
