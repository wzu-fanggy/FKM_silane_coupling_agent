"""Interface-limited hydrogen-bond analysis for FKM/SiO2.

Only FKM atoms inside the interface shell around the SiO2 center of mass
are counted.  The shell is [R_surf, R_surf + delta] with
R_surf = 15.03 A and delta = 8.59 A.

Pairs:
  A: SiO2 O-H donor ... FKM F acceptor
  B: FKM C-H donor ... SiO2 O acceptor

Criteria: H...A <= 3.0 A and D-H...A >= 120 deg.
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
OUT_DIR = Path(r"C:\Users\Hxic\Documents\ChatGPT\lammps\hbond_interface\outputs")

R_SURF = 15.03
DELTA = 8.59
R_LO = R_SURF - 0.5
R_HI = R_SURF + DELTA
N_OH = 228
SURFACE_AREA_A2 = 4.0 * np.pi * R_SURF**2
COMBOS = {
    "3.0A_120deg": (3.0, 120.0),
    "3.0A_150deg": (3.0, 150.0),
    "3.5A_120deg": (3.5, 120.0),
    "3.5A_150deg": (3.5, 150.0),
}


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
    return count


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = parse_dump(DUMP)
    print(f"frames: {len(frames)}, shell = {R_LO:.2f}..{R_HI:.2f} A")

    rows = []
    for frame in frames:
        types = frame["types"]
        xyz = frame["xyz"]
        L = frame["L"]
        sio2_mask = np.isin(types, [7, 8, 9])
        fkm_mask = np.isin(types, [1, 2, 3, 4, 5, 6])
        com = xyz[sio2_mask].mean(axis=0)

        fkm_xyz = xyz[fkm_mask]
        d_fkm = np.linalg.norm(min_image_dx(fkm_xyz - com, L), axis=1)
        shell = (d_fkm >= R_LO) & (d_fkm <= R_HI)
        shell_types = types[fkm_mask][shell]
        shell_xyz = fkm_xyz[shell]

        fkm_c = shell_xyz[np.isin(shell_types, [1, 2, 3, 4])]
        fkm_f = shell_xyz[shell_types == 5]
        fkm_h = shell_xyz[shell_types == 6]
        sio2_o = xyz[types == 8]
        sio2_h = xyz[types == 7]

        dA = nearest_donor_heavy(sio2_h, sio2_o, L)
        validA = ~np.isnan(dA[:, 0])
        dB = nearest_donor_heavy(fkm_h, fkm_c, L)
        validB = ~np.isnan(dB[:, 0])

        row = {
            "step": frame["step"],
            "time_ps": frame["time_ps"],
            "n_shell_FKM_atoms": int(shell.sum()),
        }
        for name, (rc, ang) in COMBOS.items():
            nA = count_hbonds(sio2_h[validA], dA[validA], fkm_f, L, rc, ang)
            nB = count_hbonds(fkm_h[validB], dB[validB], sio2_o, L, rc, ang)
            row[f"A_{name}"] = nA
            row[f"B_{name}"] = nB
            row[f"total_{name}"] = nA + nB
        rows.append(row)

    with open(OUT_DIR / "hbond_interface_last50.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"n_shell_FKM_atoms_mean": statistics.mean([r["n_shell_FKM_atoms"] for r in rows]),
               "shell_A": f"{R_LO:.2f}..{R_HI:.2f}",
               "N_OH": N_OH,
               "surface_area_A2": SURFACE_AREA_A2,
               "surface_area_nm2": SURFACE_AREA_A2 / 100.0}
    for name in COMBOS:
        col_a = [r[f"A_{name}"] for r in rows]
        col_b = [r[f"B_{name}"] for r in rows]
        col_t = [r[f"total_{name}"] for r in rows]
        summary[f"A_{name}_mean"] = statistics.mean(col_a)
        summary[f"A_{name}_std"] = statistics.stdev(col_a)
        summary[f"B_{name}_mean"] = statistics.mean(col_b)
        summary[f"B_{name}_std"] = statistics.stdev(col_b)
        summary[f"total_{name}_mean"] = statistics.mean(col_t)
        summary[f"total_{name}_std"] = statistics.stdev(col_t)
        a_mean = statistics.mean(col_a)
        b_mean = statistics.mean(col_b)
        t_mean = statistics.mean(col_t)
        summary[f"A_{name}_per_OH"] = a_mean / N_OH
        summary[f"B_{name}_per_OH"] = b_mean / N_OH
        summary[f"total_{name}_per_OH"] = t_mean / N_OH
        summary[f"A_{name}_per_nm2"] = a_mean / (SURFACE_AREA_A2 / 100.0)
        summary[f"B_{name}_per_nm2"] = b_mean / (SURFACE_AREA_A2 / 100.0)
        summary[f"total_{name}_per_nm2"] = t_mean / (SURFACE_AREA_A2 / 100.0)
    print(summary)
    with open(OUT_DIR / "hbond_interface_last50_summary.txt", "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k} = {v}\n")

    with open(OUT_DIR / "hbond_density_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["criteria", "A_mean", "A_per_OH", "A_per_nm2",
                         "B_mean", "B_per_OH", "B_per_nm2",
                         "total_mean", "total_per_OH", "total_per_nm2"])
        for name in COMBOS:
            writer.writerow([
                name,
                f"{summary[f'A_{name}_mean']:.2f}",
                f"{summary[f'A_{name}_per_OH']:.3f}",
                f"{summary[f'A_{name}_per_nm2']:.2f}",
                f"{summary[f'B_{name}_mean']:.2f}",
                f"{summary[f'B_{name}_per_OH']:.3f}",
                f"{summary[f'B_{name}_per_nm2']:.2f}",
                f"{summary[f'total_{name}_mean']:.2f}",
                f"{summary[f'total_{name}_per_OH']:.3f}",
                f"{summary[f'total_{name}_per_nm2']:.2f}",
            ])

    times = [r["time_ps"] for r in rows]
    for label, name in [("文献口径 3.5 Å / 150°", "3.5A_150deg"), ("默认口径 3.0 Å / 120°", "3.0A_120deg")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(times, [r[f"A_{name}"] for r in rows], "-o", ms=3, label="SiO2 O-H ... FKM F")
        ax.plot(times, [r[f"B_{name}"] for r in rows], "-o", ms=3, label="FKM C-H ... SiO2 O")
        ax.set_xlabel("时间 (ps)")
        ax.set_ylabel("氢键数目")
        ax.set_title(f"界面区域氢键（{R_LO:.1f}–{R_HI:.1f} Å, {label}）")
        ax.legend()
        fig.tight_layout()
        suffix = name.replace(".", "p")
        fig.savefig(OUT_DIR / f"hbond_interface_{suffix}.png", dpi=160)
        plt.close(fig)
    print(f"Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
