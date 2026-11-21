#!/usr/bin/env python3
"""Fully automatic uniaxial pull-out runner for the 2.0 A/ps speed.

This script:
  1. reads the last 3 frames of nvt_100ps_traj.dump (10A-20A or first_deal);
  2. creates 1/, 2/, 3/ with frameN.data and in.pull.lammps;
  3. runs each frame with LAMMPS to completion using GPU acceleration and
     16 OpenMP threads;
  4. leaves force_curve.txt, traj.dump, pull_end.data in each frame folder.
"""

import argparse
import os
import subprocess
import time
from collections import deque
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent  # 10A-20A
CANDIDATE_DUMPS = [
    ROOT / "nvt_100ps_traj.dump",
    SCRIPT_DIR.parent.parent / "nvt_100ps_traj.dump",
]
REF_DATA = SCRIPT_DIR.parent.parent / "nvt_3000ps_vacuum250.data"

DEFAULT_LMP = r"F:\LAMMPS 64-bit 4Jul2026 with GUI\bin\lmp.exe"
VELOCITY_APS = 2.0
DISPLACEMENT_A = 100.0
FORCE_EVERY = 200
DUMP_EVERY = 200
N_FRAMES = 3

SIO2_TYPES = "5 7 9 10 11 12 13 14 15"
MATRIX_TYPES = "1 2 3 4 6 8"


def resolve_input_dump():
    for p in CANDIDATE_DUMPS:
        if p.exists() and p.stat().st_size > 0:
            return p
    raise SystemExit(
        "no non-empty nvt_100ps_traj.dump in 10A-20A or first_deal"
    )


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
            f.readline()
            natoms = int(f.readline().strip())
            f.readline()
            box = []
            for _ in range(3):
                lo, hi = f.readline().split()
                box.append((float(lo), float(hi)))
            f.readline()
            atoms = []
            for _ in range(natoms):
                q = f.readline().split()
                atoms.append(
                    (
                        int(q[0]),
                        int(q[1]),
                        float(q[2]),
                        float(q[3]),
                        float(q[4]),
                        float(q[5]),
                        float(q[6]),
                        float(q[7]),
                        int(q[8]),
                        int(q[9]),
                        int(q[10]),
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


def write_input(frame_dir, frame_idx, vel_aps, displacement_a, box, run_steps=None):
    vel_fs = vel_aps / 1000.0
    dt_fs = 0.5 if vel_aps >= 15 else 1.0
    if run_steps is None:
        steps = int(round(displacement_a / (vel_fs * dt_fs)))
    else:
        steps = int(run_steps)
    zlo = box["zlo"][0]
    zhi = box["zlo"][1]
    script = f"""# ============================================================
# Uniaxial pull-out, frame {frame_idx}, {vel_aps:.0f} A/ps
# SiO2 pulled +Z ({steps} steps at {dt_fs} fs)
# Force and SiO2-FKM interface energy every {FORCE_EVERY} steps
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

group           sio2   type {SIO2_TYPES}
group           matrix type {MATRIX_TYPES}

region          clamp block INF INF INF INF INF {zlo + 25.0:.6f} units box
group           clamp region clamp
group           clamp subtract clamp sio2
group           mobile subtract matrix clamp

fix             freeze clamp setforce 0 0 0
fix             pull  sio2 move linear 0 0 {vel_fs:.6f}
fix             temp  mobile nvt temp 300.0 300.0 100.0
fix             wallz all wall/reflect zlo {zlo:.15f} zhi {zhi:.15f}

variable        z0     equal xcm(sio2,z)
variable        disp   equal xcm(sio2,z)-${{z0}}
compute         fsio   sio2 reduce sum fz
variable        force  equal -c_fsio
compute         gg     sio2 group/group matrix pair yes kspace yes
variable        egg    equal c_gg
fix             out all print {FORCE_EVERY} "$(step) $(v_disp) $(v_force) $(v_egg)" &
                file force_curve.txt screen no

dump            traj all custom {DUMP_EVERY} traj.dump id type x y z vx vy vz ix iy iz
dump_modify     traj sort id
dump_modify     traj flush yes

thermo          1000
thermo_style    custom step temp pe press v_disp v_force

timestep        {dt_fs}
run             {steps}

write_data      pull_end.data
print           ">>> frame {frame_idx} pull-out completed."
"""
    (frame_dir / "in.pull.lammps").write_text(script, encoding="ascii")
    return steps


def run_lammps(lmp, frame_dir, log_name, omp, gpu):
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(omp)
    cmd = [
        lmp,
        "-in", "in.pull.lammps",
        "-log", log_name,
        "-screen", "none",
    ]
    if gpu > 0:
        cmd += ["-sf", "gpu", "-pk", "gpu", str(gpu)]
    proc = subprocess.run(
        cmd,
        cwd=str(frame_dir),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        log = frame_dir / log_name
        tail = ""
        if log.exists():
            tail = "\n".join(
                log.read_text(encoding="ascii", errors="ignore").splitlines()[-20:]
            )
        raise RuntimeError(
            f"LAMMPS failed in {frame_dir} with return code {proc.returncode}\n{tail}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmp", default=DEFAULT_LMP)
    parser.add_argument("--velocity", type=float, default=VELOCITY_APS)
    parser.add_argument("--displacement", type=float, default=DISPLACEMENT_A)
    parser.add_argument("--frames", type=int, default=N_FRAMES)
    parser.add_argument("--force-every", type=int, default=FORCE_EVERY)
    parser.add_argument("--dump-every", type=int, default=DUMP_EVERY)
    parser.add_argument("--steps", type=int, default=None,
                        help="override run steps (smoke tests)")
    parser.add_argument("--omp", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    global FORCE_EVERY, DUMP_EVERY
    FORCE_EVERY = args.force_every
    DUMP_EVERY = args.dump_every

    input_dump = resolve_input_dump()
    if not REF_DATA.exists():
        raise SystemExit(f"reference data not found: {REF_DATA}")

    print(f"reading last {args.frames} frames from {input_dump}", flush=True)
    frames = read_last_frames(input_dump, n=args.frames)
    print("frames: " + ", ".join(f"step {fr['step']}" for fr in frames), flush=True)

    ref_lines = REF_DATA.read_text(encoding="ascii").splitlines()
    sec = find_sections(ref_lines)
    ref_atoms = parse_ref_atoms(ref_lines, sec)
    ref_box = parse_box(ref_lines)

    for idx, frame in enumerate(frames, start=1):
        frame_dir = SCRIPT_DIR / str(idx)
        frame_dir.mkdir(parents=True, exist_ok=True)
        data_path = frame_dir / f"frame{idx}.data"
        build_frame_data(ref_lines, ref_atoms, frame, data_path)
        steps = write_input(
            frame_dir,
            idx,
            args.velocity,
            args.displacement,
            ref_box,
            args.steps,
        )
        print(
            f"[frame {idx}] generated data and input "
            f"({steps} steps at {args.velocity:.0f} A/ps)",
            flush=True,
        )

        t0 = time.time()
        run_lammps(
            args.lmp,
            frame_dir,
            f"log.frame{idx}.lammps",
            args.omp,
            args.gpu,
        )
        elapsed = time.time() - t0
        print(f"[frame {idx}] LAMMPS finished in {elapsed:.1f} s", flush=True)

    print("All frames completed.", flush=True)


if __name__ == "__main__":
    main()
