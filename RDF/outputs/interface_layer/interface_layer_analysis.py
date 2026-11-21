"""Interface layer thickness for FKM around SiO2 from radial density profiles.

Definitions:
  R_surf : SiO2 surface radius from its own radial number-density profile,
           taken at half the core plateau density.
  rho_FKM(r) : time-averaged FKM number/mass density in spherical shells
               centered on the instantaneous SiO2 center of mass.
  delta   : interface layer thickness, the radial width of the FKM density
            rise from 10% to 90% of the bulk plateau value.
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


DUMP = Path(r"F:\Lammps-MS\FKM-SIO2-TEST\RDF\traj\last50.dump")
OUT_DIR = Path(r"C:\Users\Hxic\Documents\ChatGPT\lammps\interface_layer\outputs")

DR = 0.25
RMAX = 40.0
MASS = {1: 12.01115, 2: 12.01115, 3: 12.01115, 4: 12.01115, 5: 18.9984, 6: 1.00797,
        7: 1.00797, 8: 15.9994, 9: 28.086}


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
            xyz_raw = arr[:, 2:5]
            xyz = xyz_raw + arr[:, 5:8] * lengths
            frames.append(
                {
                    "step": step,
                    "types": arr[:, 1].astype(np.int64),
                    "xyz_raw": xyz_raw,
                    "L": lengths,
                }
            )
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
    n_density = counts / vol
    mass_density = masssum / vol
    return rmid, n_density, mass_density, counts


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = parse_dump(DUMP)
    print(f"frames: {len(frames)}")

    nbins = int(RMAX / DR)
    sio2_count = np.zeros(nbins)
    fkm_count = np.zeros(nbins)
    fkm_mass = np.zeros(nbins)
    half_lengths = []
    coms = []

    for frame in frames:
        types = frame["types"]
        xyz = frame["xyz_raw"]
        L = frame["L"]
        half_lengths.append(0.5 * L)
        sio2_mask = np.isin(types, [7, 8, 9])
        fkm_mask = np.isin(types, [1, 2, 3, 4, 5, 6])
        com = xyz[sio2_mask].mean(axis=0)
        coms.append(com)
        d_sio2 = np.linalg.norm(min_image_dx(xyz[sio2_mask] - com, L), axis=1)
        d_fkm = np.linalg.norm(min_image_dx(xyz[fkm_mask] - com, L), axis=1)
        m_sio2 = np.array([MASS[t] for t in types[sio2_mask]])
        m_fkm = np.array([MASS[t] for t in types[fkm_mask]])
        _, _, _, c = shell_density(d_sio2, m_sio2)
        sio2_count += c
        _, _, _, c = shell_density(d_fkm, m_fkm)
        fkm_count += c
        _, _, mass_density, _ = shell_density(d_fkm, m_fkm)
        fkm_mass += mass_density

    rmid, _, _, _ = shell_density(np.array([0.0]), np.array([0.0]))
    n_frames = len(frames)
    rlo = rmid - 0.5 * DR
    rhi = rmid + 0.5 * DR
    vol = (4.0 / 3.0) * np.pi * (rhi**3 - rlo**3)
    sio2_n = sio2_count / (n_frames * vol)
    fkm_n = fkm_count / (n_frames * vol)
    fkm_rho_gcm3 = fkm_mass / n_frames * 1.66054

    # Smooth profiles (moving average over 5 bins = 1.25 A) to suppress
    # shell-volume noise near the particle center.
    def smooth(a, w=5):
        return np.convolve(a, np.ones(w) / w, mode="same")

    sio2_s = smooth(sio2_n)
    fkm_s = smooth(fkm_n)

    # SiO2 surface radius: last crossing of half the core plateau density.
    core_mask = (rmid >= 5.0) & (rmid <= 13.0)
    rho_core = float(np.mean(sio2_s[core_mask]))
    above = sio2_s >= 0.5 * rho_core
    idx = np.flatnonzero(above)
    if len(idx):
        last = idx[-1]
        r_surf = float(rmid[last])
        if last + 1 < len(rmid):
            # Linear interpolation between the last above-half and next bin.
            r0, r1 = rmid[last], rmid[last + 1]
            v0, v1 = sio2_s[last], sio2_s[last + 1]
            r_surf = r0 + (0.5 * rho_core - v0) * (r1 - r0) / (v1 - v0)
    else:
        r_surf = float("nan")

    # Usable range: beyond surface but inside half the box.
    lhalf = float(np.mean(half_lengths))
    rmax_use = min(lhalf - 2.0, 30.0)
    use = (rmid > r_surf + 6.0) & (rmid < rmax_use)
    use_idx = np.flatnonzero(use)
    if len(use_idx) >= 8:
        use_idx = use_idx[-8:]  # prefer the flattest far-box segment
    bulk_rho = float(np.mean(fkm_s[use_idx]))
    bulk_std = float(np.std(fkm_s[use_idx]))
    bulk_mass = float(np.mean(fkm_rho_gcm3[use_idx]))

    # 10% -> 90% rise width of the smoothed FKM density profile.
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

    # Perturbed layer: last radius (beyond surface) where FKM density is
    # still outside the bulk +/-10% band.
    perturb = (rmid > r_surf) & (rmid < rmax_use) & (np.abs(fkm_s - bulk_rho) > 0.10 * bulk_rho)
    perturb_idx = np.flatnonzero(perturb)
    r_pert = float(rmid[perturb_idx[-1]]) if len(perturb_idx) else r_surf
    delta_pert = r_pert - r_surf

    # Integral excess-layer thickness: excess FKM atoms per unit particle
    # surface area, normalized by bulk density.
    integ = (rmid > r_surf) & (rmid < rmax_use)
    excess = np.sum((fkm_s[integ] - bulk_rho) * 4.0 * np.pi * rmid[integ] ** 2 * DR)
    excess_thick = excess / (bulk_rho * 4.0 * np.pi * r_surf**2)

    # First density maximum of FKM beyond the surface (layering peak).
    lay = (rmid > r_surf) & (rmid < rmax_use)
    pk_idx = int(np.argmax(fkm_s[lay]))
    r_pk = float(rmid[lay][pk_idx])
    rho_pk = float(fkm_s[lay][pk_idx])

    with open(OUT_DIR / "interface_layer_summary.txt", "w", encoding="utf-8") as f:
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

    with open(OUT_DIR / "radial_density_profile.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["r_A", "SiO2_n_atoms_per_A3", "FKM_n_atoms_per_A3", "FKM_mass_g_per_cm3"])
        for i in range(len(rmid)):
            writer.writerow([f"{rmid[i]:.3f}", f"{sio2_n[i]:.6f}", f"{fkm_n[i]:.6f}", f"{fkm_rho_gcm3[i]:.6f}"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rmid, sio2_s, color="tab:blue", lw=1.3, label="SiO2 数密度(平滑)")
    ax.axvline(r_surf, color="tab:blue", ls="--", lw=1, label=f"R_surf={r_surf:.2f} Å")
    ax.set_xlabel("到 SiO2 质心距离 r (Å)")
    ax.set_ylabel("数密度 (atoms/Å³)")
    ax.set_title("SiO2 径向数密度剖面")
    ax.legend()
    ax.set_xlim(0, 35)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sio2_radial_profile.png", dpi=160)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(rmid, fkm_s, color="tab:orange", lw=1.3, label="FKM 数密度(平滑)")
    ax2.axvline(r_surf, color="tab:blue", ls="--", lw=1, label=f"R_surf={r_surf:.2f} Å")
    ax2.axvline(r_lo, color="tab:green", ls="--", lw=1, label=f"r(10%)={r_lo:.2f} Å")
    ax2.axvline(r_hi, color="tab:red", ls="--", lw=1, label=f"r(90%)={r_hi:.2f} Å")
    ax2.axhline(bulk_rho, color="gray", ls=":", lw=1, label=f"bulk={bulk_rho:.4f}")
    ax2.set_xlabel("到 SiO2 质心距离 r (Å)")
    ax2.set_ylabel("FKM 数密度 (atoms/Å³)")
    ax2.set_title(f"FKM 径向数密度剖面，界面层厚度 Δ={delta:.2f} Å")
    ax2.legend()
    ax2.set_xlim(0, 35)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "fkm_radial_profile.png", dpi=160)
    plt.close(fig2)

    print(
        f"R_surf={r_surf:.3f} A, bulk_rho={bulk_rho:.5f}, delta={delta:.3f} A, "
        f"delta_pert={delta_pert:.3f} A, excess_thick={excess_thick:.3f} A, "
        f"first_peak={r_pk:.3f} A ({rho_pk:.4f})"
    )
    print(f"Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
