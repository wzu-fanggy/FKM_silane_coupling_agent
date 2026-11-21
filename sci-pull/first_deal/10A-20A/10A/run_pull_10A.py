#!/usr/bin/env python3
"""Generate uniaxial pull-out inputs for the 10 A/ps speed.

This script only prepares the working directories and LAMMPS inputs:
  1. read the last 3 frames of ../nvt_100ps_traj.dump;
  2. for every frame create 1/, 2/, 3/ and write frameN.data + in.pull.lammps;
  3. you run each in.pull.lammps in the LAMMPS GUI with that folder as the
     working directory;
  4. after the frames finish, run `python run_pull_10A.py --render` to draw
     force_bin5.png and force_bin10.png.
"""

import argparse
import math
from collections import deque
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent  # 10A-30A
INPUT_DUMP = ROOT / "nvt_100ps_traj.dump"
REF_DATA = SCRIPT_DIR.parent.parent / "nvt_2000ps_gpu_centered_vacuum250.data"

VELOCITY_APS = 10.0
DISPLACEMENT_A = 100.0
FORCE_EVERY = 200
DUMP_EVERY = 200
N_FRAMES = 3


def find_sections(lines):
    wanted = {
        "Atoms # full": None,
        "Velocities": None,
        "Bonds": None,
    }
    for i, line in enumerate(lines):
        s = line.strip()
        if s in wanted:
            wanted[s] = i
    return wanted


def parse_ref_atoms(lines, sec):
    atoms = {}
    for line in lines[sec["Atoms # full"] + 1 : sec["Velocities"]]:
        q = line.split()
        if len(q) >= 10 and q[0].isdigit():
            atoms[int(q[0])] = (int(q[1]), int(q[2]), float(q[3]))
    return atoms


def parse_box(lines):
    box = {}
    for line in lines:
        q = line.split()
        if len(q) == 4 and q[2] in ("xlo", "ylo", "zlo"):
            box[q[2]] = (float(q[0]), float(q[1]))
    return box


def read_last_frames(path, n=3):
    frames = deque(maxlen=n)
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        line = f.readline()
        while line:
            if not line.startswith("ITEM: TIMESTEP"):
                line = f.readline()
                continue
            step = int(f.readline().strip())
            f.readline()  # ITEM: NUMBER OF ATOMS
            natoms = int(f.readline().strip())
            f.readline()  # ITEM: BOX BOUNDS pp pp ff
            box = []
            for _ in range(3):
                lo, hi = f.readline().split()
                box.append((float(lo), float(hi)))
            f.readline()  # ITEM: ATOMS id type x y z vx vy vz ix iy iz
            atoms = []
            for _ in range(natoms):
                q = f.readline().split()
                atoms.append(
                    (
                        int(q[0]),  # id
                        int(q[1]),  # type
                        float(q[2]),  # x
                        float(q[3]),  # y
                        float(q[4]),  # z
                        float(q[5]),  # vx
                        float(q[6]),  # vy
                        float(q[7]),  # vz
                        int(q[8]),  # ix
                        int(q[9]),  # iy
                        int(q[10]),  # iz
                    )
                )
            frames.append({"step": step, "box": box, "atoms": atoms})
            line = f.readline()
    return list(frames)


def build_frame_data(ref_lines, ref_atoms, frame, out_path):
    sec = find_sections(ref_lines)
    atoms_by_id = {a[0]: a for a in frame["atoms"]}
    atom_text = {}
    for aid, atom in atoms_by_id.items():
        if aid not in ref_atoms:
            raise RuntimeError(f"atom {aid} from dump not found in reference data")
        mol, typ, charge = ref_atoms[aid]
        atom_text[aid] = (
            f"  {aid} {mol} {typ} {charge:.6f} {atom[2]:.6f} {atom[3]:.6f} "
            f"{atom[4]:.6f} {atom[8]} {atom[9]} {atom[10]}"
        )

    vel_text = {
        a[0]: f"  {a[0]} {a[5]:.6f} {a[6]:.6f} {a[7]:.6f}"
        for a in frame["atoms"]
    }

    out = []
    section = None
    for line in ref_lines:
        s = line.strip()
        if s == "Atoms # full":
            section = "atoms"
            out.append(line)
            continue
        if s == "Velocities":
            section = "velocities"
            out.append(line)
            continue
        if s == "Bonds":
            section = None
            out.append(line)
            continue

        q = line.split()
        if section == "atoms" and len(q) >= 10 and q[0].isdigit():
            out.append(atom_text[int(q[0])])
            continue
        if section == "velocities" and len(q) >= 4 and q[0].isdigit():
            out.append(vel_text[int(q[0])])
            continue
        if len(q) == 4 and q[2] in ("xlo", "ylo", "zlo"):
            axis = {"xlo": 0, "ylo": 1, "zlo": 2}[q[2]]
            lo, hi = frame["box"][axis]
            out.append(f"{lo:.15f} {hi:.15f} {q[2]} {q[3]}")
            continue
        out.append(line)

    out_path.write_text("\n".join(out) + "\n", encoding="ascii")


def write_input(frame_dir, frame_idx, vel_aps, displacement_a, box):
    vel_fs = vel_aps / 1000.0
    steps = int(round(displacement_a / vel_fs))
    zlo = box["zlo"][0]
    zhi = box["zlo"][1]
    script = f"""# ============================================================
# Uniaxial pull-out, frame {frame_idx}, {vel_aps:.0f} A/ps
# SiO2 pulled +Z for {displacement_a:.0f} A ({steps} steps at 1 fs)
# Force recorded every {FORCE_EVERY} steps
# Run from this folder; outputs stay here.
# ============================================================

units           real
atom_style      full
boundary        p p f

pair_style      lj/class2/coul/long 15.5
kspace_style    pppm 1e-4
kspace_modify   slab 3.0
bond_style      class2
angle_style     class2
dihedral_style  class2
improper_style  class2
special_bonds   lj/coul 0 0 1

read_data       frame{frame_idx}.data

neighbor        0.3 bin
neigh_modify    delay 0 every 1 check yes

# ---- groups ----
group           sio2   type 7 8 9
group           matrix type 1 2 3 4 5 6

region          clamp block INF INF INF INF INF 46.2 units box
group           clamp region clamp
group           clamp subtract clamp sio2
group           mobile subtract matrix clamp

# ---- constraints ----
fix             freeze clamp setforce 0 0 0
fix             pull  sio2 move linear 0 0 {vel_fs:.6f}
fix             temp  mobile nvt temp 300.0 300.0 100.0
fix             wallz all wall/reflect zlo {zlo:.15f} zhi {zhi:.15f}

# ---- force/displacement output ----
variable        z0     equal xcm(sio2,z)
variable        disp   equal xcm(sio2,z)-${{z0}}
compute         fsio   sio2 reduce sum fz
variable        force  equal -c_fsio
compute         gg     sio2 group/group matrix pair yes kspace yes
variable        egg    equal c_gg
fix             out all print {FORCE_EVERY} "$(step) $(v_disp) $(v_force) $(v_egg)" &
                file force_curve.txt screen no

# ---- trajectory ----
dump            traj all custom {DUMP_EVERY} traj.dump id type x y z vx vy vz ix iy iz
dump_modify     traj sort id
dump_modify     traj flush yes

thermo          1000
thermo_style    custom step temp pe press v_disp v_force

timestep        1.0
run             {steps}

write_data      pull_end.data
print           ">>> frame {frame_idx} pull-out completed."
"""
    (frame_dir / "in.pull.lammps").write_text(script, encoding="ascii")
    return steps


def load_force_curves(speed_dir, frame_dirs):
    rows = []
    for i in frame_dirs:
        path = speed_dir / str(i) / "force_curve.txt"
        if not path.exists():
            raise RuntimeError(f"missing force curve: {path}")
        with open(path, "r", encoding="ascii", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                q = s.split()
                if len(q) < 3:
                    continue
                step = float(q[0])
                disp = float(q[1])
                force = float(q[2])
                egg = float(q[3]) if len(q) > 3 else None
                resid = float(q[4]) if len(q) > 4 else None
                rows.append((i, step, disp, force, egg, resid))
    return rows


def analyze_residue(rows, speed_dir):
    """Write a per-frame interface-energy residue summary."""
    import csv

    frame_map = {}
    for row in rows:
        frame_map.setdefault(row[0], []).append(row)

    summaries = []
    for frame in sorted(frame_map):
        fr = sorted(frame_map[frame], key=lambda r: r[1])
        egg0 = next((r[4] for r in fr if r[4] is not None), None)
        eggf = next((r[4] for r in reversed(fr) if r[4] is not None), None)
        if egg0 is not None and eggf is not None and egg0 != 0:
            residual_frac = eggf / egg0
        else:
            residual_frac = None
        summaries.append(
            {
                "frame": frame,
                "E_gg_initial_kcal_mol": egg0,
                "E_gg_final_kcal_mol": eggf,
                "residual_fraction": residual_frac,
            }
        )

    out_csv = speed_dir / "residue_summary.csv"
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"  wrote {out_csv}", flush=True)
    for s in summaries:
        rf = s["residual_fraction"]
        rf_txt = f"{rf:.4f}" if rf is not None else "n/a"
        print(
            f"  frame {s['frame']}: E_gg {s['E_gg_initial_kcal_mol']:.2f} -> "
            f"{s['E_gg_final_kcal_mol']:.2f} kcal/mol, "
            f"residual = {rf_txt}",
            flush=True,
        )


def render_force_curves(rows, speed_dir, bins=(5.0, 10.0)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    for bw in bins:
        groups = {}
        for row in rows:
            disp = row[2]
            force = row[3]
            key = math.floor(disp / bw) * bw
            groups.setdefault(key, []).append(force)

        xs = []
        means = []
        stds = []
        ns = []
        for key in sorted(groups):
            arr = np.asarray(groups[key])
            xs.append(key + bw / 2)
            means.append(arr.mean())
            stds.append(arr.std())
            ns.append(len(arr))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, means, "-o", ms=3, color="tab:blue", label=f"{bw:.0f} A bin")
        ax.fill_between(
            xs,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color="tab:blue",
            alpha=0.25,
            label="+/- 1 SD",
        )
        ax.set_xlabel("SiO2 displacement (A)")
        ax.set_ylabel("Pull force (kcal/mol/A)")
        ax.set_title(f"Uniaxial pull {VELOCITY_APS:.0f} A/ps, {bw:.0f} A bin")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(speed_dir / f"force_bin{bw:.0f}.png", dpi=160)
        plt.close(fig)
        print(f"  wrote force_bin{bw:.0f}.png "
              f"({len(xs)} bins, mean n/point = {np.mean(ns):.0f})", flush=True)


def find_frame_dirs(speed_dir):
    dirs = []
    for path in speed_dir.iterdir():
        if (
            path.is_dir()
            and path.name.isdigit()
            and (path / "force_curve.txt").exists()
        ):
            dirs.append(int(path.name))
    return sorted(dirs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--velocity", type=float, default=VELOCITY_APS)
    parser.add_argument("--displacement", type=float, default=DISPLACEMENT_A)
    parser.add_argument("--frames", type=int, default=N_FRAMES)
    parser.add_argument("--force-every", type=int, default=FORCE_EVERY)
    parser.add_argument("--dump-every", type=int, default=DUMP_EVERY)
    parser.add_argument("--render", action="store_true",
                        help="render force figures from existing force_curve.txt files")
    return parser.parse_args()


def main():
    args = parse_args()
    global FORCE_EVERY, DUMP_EVERY
    FORCE_EVERY = args.force_every
    DUMP_EVERY = args.dump_every

    if args.render:
        frame_dirs = find_frame_dirs(SCRIPT_DIR)
        if not frame_dirs:
            raise SystemExit("no frame folders found")
        print(f"rendering from frames {frame_dirs}", flush=True)
        rows = load_force_curves(SCRIPT_DIR, frame_dirs)
        render_force_curves(rows, SCRIPT_DIR, bins=(5.0, 10.0))
        analyze_residue(rows, SCRIPT_DIR)
        print("done", flush=True)
        return

    if not INPUT_DUMP.exists():
        raise SystemExit(f"input dump not found: {INPUT_DUMP}")
    if not REF_DATA.exists():
        raise SystemExit(f"reference data not found: {REF_DATA}")

    print(f"reading last {args.frames} frames from {INPUT_DUMP}", flush=True)
    frames = read_last_frames(INPUT_DUMP, n=args.frames)
    print(f"frames: " + ", ".join(f"step {fr['step']}" for fr in frames), flush=True)

    ref_lines = REF_DATA.read_text(encoding="ascii").splitlines()
    sec = find_sections(ref_lines)
    ref_atoms = parse_ref_atoms(ref_lines, sec)
    ref_box = parse_box(ref_lines)

    for idx, frame in enumerate(frames, start=1):
        frame_dir = SCRIPT_DIR / str(idx)
        frame_dir.mkdir(parents=True, exist_ok=True)
        data_path = frame_dir / f"frame{idx}.data"
        build_frame_data(ref_lines, ref_atoms, frame, data_path)
        steps = write_input(frame_dir, idx, args.velocity, args.displacement, ref_box)
        print(
            f"frame {idx}: {frame_dir}  "
            f"(step {frame['step']}, run {steps} steps at {args.velocity:.0f} A/ps)",
            flush=True,
        )

    print()
    print("Generation complete. Run each in.pull.lammps in the LAMMPS GUI:")
    for idx in range(1, len(frames) + 1):
        print(f"  {SCRIPT_DIR / str(idx) / 'in.pull.lammps'}")
    print("Each run writes force_curve.txt, traj.dump, pull_end.data in that folder.")
    print("After all frames finish, run:  python run_pull_10A.py --render")


if __name__ == "__main__":
    main()
