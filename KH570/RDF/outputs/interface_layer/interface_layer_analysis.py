#!/usr/bin/env python3
"""Interface layer analysis for FKM around KH570-grafted SiO2.

The central object is the full grafted cluster (SiO2 core + KH570 graft).
Radial density profiles are centered on the per-frame geometric center of
the cluster.  Separate profiles are made for:
  - SiO2 core       (h*, o, si)            types 9, 11, 14
  - KH570 graft     (organic + graft O/Si) types 5, 6, 7, 12, 13, 15, 16, 17
  - FKM             types 1, 2, 3, 4, 8, 10

R_surf is defined from the SiO2 core half-core-density crossing.
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


WORK = Path(__file__).resolve().parent
DUMP = Path(r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\RDF\traj\last50.dump")

DR = 0.25
RMAX = 40.0

MASS = {
    1: 12.01115, 2: 12.01115, 3: 12.01115, 4: 12.01115,
    5: 12.01115, 6: 12.01115, 7: 12.01115,
    8: 18.9984,
    9: 1.00797, 10: 1.00797,
    11: 15.9994, 12: 15.9994, 13: 15.9994,
    14: 28.086,
    15: 12.01115, 16: 12.01115, 17: 1.00797,
}

CORE_TYPES = [9, 11, 14]
GRAFT_TYPES = [5, 6, 7, 12, 13, 15, 16, 17]
CLUSTER_TYPES = CORE_TYPES + GRAFT_TYPES
FKM_TYPES = [1, 2, 3, 4, 8, 10]


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
                "types": arr[:, 1].astype(np.int64),
                "xyz_raw": arr[:, 2:5],
                "L": lengths,
            })
    return frames


def min_image_dx(dx, L):
    return dx - L * np.round(dx / L)


def shell_density(dist, masses, dr=DR, rmax=RMAX):
    nbins = int(rmax / dr)
    counts = np.zeros(nbins)
    masssum = np.zeros(nbins)
    for d, m in zip(dist, masses):
        i = int(d / dr)
        if i < nbins:
            counts[i] += 1
            masssum[i] += m
    rmid = (np.arange(nbins) + 0.5) * dr
    rlo = rmid - 0.5 * dr
    rhi = rmid + 0.5 * dr
    vol = (4.0 / 3.0) * np.pi * (rhi**3 - rlo**3)
    return rmid, counts / vol, masssum / vol, counts


def smooth(a, w=5):
    return np.convolve(a, np.ones(w) / w, mode="same")


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    frames = parse_dump(DUMP)
    print(f"frames: {len(frames)}")

    nbins = int(RMAX / DR)
    core_count = np.zeros(nbins)
    graft_count = np.zeros(nbins)
    cluster_count = np.zeros(nbins)
    fkm_count = np.zeros(nbins)
    fkm_mass = np.zeros(nbins)
    half_lengths = []

    for frame in frames:
        types = frame["types"]
        xyz = frame["xyz_raw"]
        L = frame["L"]
        half_lengths.append(0.5 * L)

        core_mask = np.isin(types, CORE_TYPES)
        graft_mask = np.isin(types, GRAFT_TYPES)
        cluster_mask = np.isin(types, CLUSTER_TYPES)
        fkm_mask = np.isin(types, FKM_TYPES)

        center = xyz[cluster_mask].mean(axis=0)
        for mask, count_buf, mass_buf in (
            (core_mask, core_count, None),
            (graft_mask, graft_count, None),
            (cluster_mask, cluster_count, None),
            (fkm_mask, fkm_count, fkm_mass),
        ):
            d = np.linalg.norm(min_image_dx(xyz[mask] - center, L), axis=1)
            m = np.array([MASS[t] for t in types[mask]])
            _, _, _, c = shell_density(d, m)
            count_buf += c
            if mass_buf is not None:
                _, _, md, _ = shell_density(d, m)
                mass_buf += md

    n_frames = len(frames)
    rmid, _, _, _ = shell_density(np.array([0.0]), np.array([0.0]))
    rlo = rmid - 0.5 * DR
    rhi = rmid + 0.5 * DR
    vol = (4.0 / 3.0) * np.pi * (rhi**3 - rlo**3)

    core_n = core_count / (n_frames * vol)
    graft_n = graft_count / (n_frames * vol)
    cluster_n = cluster_count / (n_frames * vol)
    fkm_n = fkm_count / (n_frames * vol)
    fkm_rho_gcm3 = fkm_mass / n_frames * 1.66054

    core_s = smooth(core_n)
    graft_s = smooth(graft_n)
    cluster_s = smooth(cluster_n)
    fkm_s = smooth(fkm_n)

    core_mask = (rmid >= 5.0) & (rmid <= 13.0)
    rho_core = float(np.mean(core_s[core_mask]))
    above = core_s >= 0.5 * rho_core
    idx = np.flatnonzero(above)
    last = idx[-1]
    r_surf = float(rmid[last])
    if last + 1 < len(rmid):
        r0, r1 = rmid[last], rmid[last + 1]
        v0, v1 = core_s[last], core_s[last + 1]
        r_surf = r0 + (0.5 * rho_core - v0) * (r1 - r0) / (v1 - v0)

    lhalf = float(np.mean(half_lengths))
    rmax_use = min(lhalf - 2.0, 30.0)
    use = (rmid > r_surf + 6.0) & (rmid < rmax_use)
    use_idx = np.flatnonzero(use)
    if len(use_idx) >= 8:
        use_idx = use_idx[-8:]
    bulk_rho = float(np.mean(fkm_s[use_idx]))
    bulk_std = float(np.std(fkm_s[use_idx]))
    bulk_mass = float(np.mean(fkm_rho_gcm3[use_idx]))

    lo = bulk_rho * 0.10
    hi = bulk_rho * 0.90
    r_lo = None
    r_hi = None
    mask = (rmid > r_surf) & (rmid < rmax_use)
    rr = rmid[mask]
    rho = fkm_s[mask]
    for x, y in zip(rr, rho):
        if r_lo is None and y >= lo:
            r_lo = x
        if r_hi is None and y >= hi:
            r_hi = x
    if r_lo is None:
        r_lo = r_surf
    if r_hi is None:
        r_hi = rr[-1]
    delta = r_hi - r_lo

    perturb = (rmid > r_surf) & (rmid < rmax_use) & (np.abs(fkm_s - bulk_rho) > 0.10 * bulk_rho)
    perturb_idx = np.flatnonzero(perturb)
    r_pert = float(rmid[perturb_idx[-1]]) if len(perturb_idx) else r_surf
    delta_pert = r_pert - r_surf

    integ = (rmid > r_surf) & (rmid < rmax_use)
    excess = np.sum((fkm_s[integ] - bulk_rho) * 4.0 * np.pi * rmid[integ] ** 2 * DR)
    excess_thick = excess / (bulk_rho * 4.0 * np.pi * r_surf**2)

    lay = (rmid > r_surf) & (rmid < rmax_use)
    pk_idx = int(np.argmax(fkm_s[lay]))
    r_pk = float(rmid[lay][pk_idx])
    rho_pk = float(fkm_s[lay][pk_idx])

    with open(WORK / "interface_layer_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"frames = {n_frames}\n")
        f.write(f"R_surf (half core density) = {r_surf:.3f} A\n")
        f.write(f"rho_core_SiO2 = {rho_core:.4f} atoms/A^3\n")
        f.write(f"bulk_region = {rmid[use_idx[0]]:.2f} .. {rmid[use_idx[-1]]:.2f} A\n")
        f.write(f"FKM bulk number density = {bulk_rho:.5f} atoms/A^3\n")
        f.write(f"FKM bulk mass density = {bulk_mass:.4f} g/cm^3\n")
        f.write(f"r(10%) = {r_lo:.3f} A\n")
        f.write(f"r(90%) = {r_hi:.3f} A\n")
        f.write(f"interface thickness delta = {delta:.3f} A\n")
        f.write(f"perturbed layer extends to r = {r_pert:.3f} A, delta_pert = {delta_pert:.3f} A\n")
        f.write(f"integral excess layer thickness = {excess_thick:.3f} A\n")
        f.write(f"first FKM density peak r = {r_pk:.3f} A, rho = {rho_pk:.4f} atoms/A^3\n")

    with open(WORK / "radial_density_profile.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "r_A", "cluster_n_atoms_per_A3", "core_n_atoms_per_A3",
            "graft_n_atoms_per_A3", "FKM_n_atoms_per_A3", "FKM_mass_g_per_cm3",
        ])
        for i in range(len(rmid)):
            writer.writerow([
                f"{rmid[i]:.3f}", f"{cluster_n[i]:.6f}", f"{core_n[i]:.6f}",
                f"{graft_n[i]:.6f}", f"{fkm_n[i]:.6f}", f"{fkm_rho_gcm3[i]:.6f}",
            ])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rmid, core_s, color="tab:blue", lw=1.3, label="SiO2 核数密度")
    ax.plot(rmid, graft_s, color="tab:red", lw=1.3, label="KH570 接枝数密度")
    ax.plot(rmid, cluster_s, color="tab:gray", lw=1.0, ls="--", label="接枝复合体总数密度")
    ax.axvline(r_surf, color="tab:blue", ls="--", lw=1, label=f"R_surf={r_surf:.2f} Å")
    ax.set_xlabel("到接枝复合体质心距离 r (Å)")
    ax.set_ylabel("数密度 (atoms/Å³)")
    ax.set_title("KH570接枝体系：SiO2核 / KH570接枝 / 复合体径向密度")
    ax.legend()
    ax.set_xlim(0, 35)
    fig.tight_layout()
    fig.savefig(WORK / "cluster_radial_profile.png", dpi=160)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(rmid, fkm_s, color="tab:orange", lw=1.3, label="FKM 数密度(平滑)")
    ax2.axvline(r_surf, color="tab:blue", ls="--", lw=1, label=f"R_surf={r_surf:.2f} Å")
    ax2.axvline(r_lo, color="tab:green", ls="--", lw=1, label=f"r(10%)={r_lo:.2f} Å")
    ax2.axvline(r_hi, color="tab:red", ls="--", lw=1, label=f"r(90%)={r_hi:.2f} Å")
    ax2.axhline(bulk_rho, color="gray", ls=":", lw=1, label=f"bulk={bulk_rho:.4f}")
    ax2.set_xlabel("到接枝复合体质心距离 r (Å)")
    ax2.set_ylabel("FKM 数密度 (atoms/Å³)")
    ax2.set_title(f"KH570接枝体系 FKM 径向剖面，Δ={delta:.2f} Å")
    ax2.legend()
    ax2.set_xlim(0, 35)
    fig2.tight_layout()
    fig2.savefig(WORK / "fkm_radial_profile.png", dpi=160)
    plt.close(fig2)

    print(f"R_surf={r_surf:.3f} A, bulk_rho={bulk_rho:.5f}, delta={delta:.3f} A, "
          f"delta_pert={delta_pert:.3f} A, excess_thick={excess_thick:.3f} A, "
          f"first_peak={r_pk:.3f} A ({rho_pk:.4f})")
    print(f"Outputs in {WORK}")


if __name__ == "__main__":
    main()
