#!/usr/bin/env python3
"""Generate and optionally run Cij strain-deformation jobs for FKM/SiO2.

Reads the last 2 ns of nvt_2000ps_gpu_traj.dump, takes N independent frames,
builds one data file per frame, and writes 12 small LAMMPS inputs per frame
(6 strain modes x +/- 0.003).  Each job does 20 ps NVT equilibration followed
by 80 ps NVT stress sampling.

Usage:
    python run_cij.py                # generate all inputs only
    python run_cij.py --run          # generate and run all jobs
    python run_cij.py --frames 5     # use 5 frames instead of 20
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DATA = SCRIPT_DIR / "nvt_2000ps_gpu.data"
INPUT_DUMP = SCRIPT_DIR / "nvt_2000ps_gpu_traj.dump"
DEFAULT_LMP = r"F:\LAMMPS 64-bit 4Jul2026 with GUI\bin\lmp.exe"

STRAIN = 0.003
EQUIL_STEPS = 20000   # 20 ps at 1 fs
SAMPLE_STEPS = 80000  # 80 ps at 1 fs
MODES = ["xx", "yy", "zz", "xy", "xz", "yz"]
SIGNS = [1.0, -1.0]


def find_section(lines, name):
    for i, line in enumerate(lines):
        if line.strip() == name:
            return i
    raise RuntimeError(f"section not found: {name}")


def parse_ref_atoms(lines):
    si = find_section(lines, "Atoms # full")
    vi = find_section(lines, "Velocities")
    atoms = {}
    for line in lines[si + 1 : vi]:
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


def collect_steps(path):
    steps = []
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            if line.startswith("ITEM: TIMESTEP"):
                steps.append(int(f.readline().strip()))
    return steps


def read_frames(path, target_steps):
    wanted = set(target_steps)
    frames = {}
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        line = f.readline()
        while line:
            if not line.startswith("ITEM: TIMESTEP"):
                line = f.readline()
                continue
            step = int(f.readline().strip())
            f.readline()  # ITEM: NUMBER OF ATOMS
            natoms = int(f.readline().strip())
            f.readline()  # ITEM: BOX BOUNDS pp pp pp
            box = []
            for _ in range(3):
                lo, hi = f.readline().split()
                box.append((float(lo), float(hi)))
            f.readline()  # ITEM: ATOMS id type x y z ix iy iz
            ids = np.empty(natoms, dtype=np.int64)
            types = np.empty(natoms, dtype=np.int64)
            pos = np.empty((natoms, 3))
            img = np.empty((natoms, 3), dtype=np.int64)
            for i in range(natoms):
                q = f.readline().split()
                ids[i] = int(q[0])
                types[i] = int(q[1])
                pos[i] = [float(q[2]), float(q[3]), float(q[4])]
                img[i] = [int(q[5]), int(q[6]), int(q[7])]
            if step in wanted:
                frames[step] = {
                    "step": step,
                    "box": box,
                    "ids": ids,
                    "types": types,
                    "pos": pos,
                    "img": img,
                }
                wanted.discard(step)
            line = f.readline()
    if wanted:
        raise RuntimeError(f"missing frames: {sorted(wanted)}")
    return frames


def build_frame_data(ref_lines, ref_atoms, frame, out_path):
    si = find_section(ref_lines, "Atoms # full")
    vi = find_section(ref_lines, "Velocities")
    bi = find_section(ref_lines, "Bonds")
    atom_by_id = {
        int(a): (frame["types"][i], frame["pos"][i], frame["img"][i])
        for i, a in enumerate(frame["ids"])
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
            aid = int(q[0])
            mol, typ, charge = ref_atoms[aid]
            typ_dump, pos, img = atom_by_id[aid]
            out.append(
                f"  {aid} {mol} {typ} {charge:.6f} {pos[0]:.6f} {pos[1]:.6f} "
                f"{pos[2]:.6f} {int(img[0])} {int(img[1])} {int(img[2])}"
            )
            continue
        if section == "velocities" and len(q) >= 4 and q[0].isdigit():
            # keep reference velocities; NVT will re-equilibrate them
            out.append(line)
            continue
        if len(q) == 4 and q[2] in ("xlo", "ylo", "zlo"):
            axis = {"xlo": 0, "ylo": 1, "zlo": 2}[q[2]]
            lo, hi = frame["box"][axis]
            out.append(f"{lo:.15f} {hi:.15f} {q[2]} {q[3]}")
            continue
        out.append(line)
    out_path.write_text("\n".join(out) + "\n", encoding="ascii")


def write_strain_input(job_dir, data_name, mode, sign, strain, lengths):
    job_dir.mkdir(parents=True, exist_ok=True)
    lx, ly, lz = lengths
    n_every = 1000
    n_repeat = SAMPLE_STEPS // n_every
    newton_line = "newton         off\n" if mode in ("xy", "xz", "yz") else ""
    if mode == "xx":
        change = f"change_box all x scale {1.0 + sign * strain:.6f} remap"
    elif mode == "yy":
        change = f"change_box all y scale {1.0 + sign * strain:.6f} remap"
    elif mode == "zz":
        change = f"change_box all z scale {1.0 + sign * strain:.6f} remap"
    elif mode == "xy":
        change = (
            "change_box all triclinic\n"
            f"change_box all xy delta {sign * strain * ly:.6f} remap"
        )
    elif mode == "xz":
        change = (
            "change_box all triclinic\n"
            f"change_box all xz delta {sign * strain * lz:.6f} remap"
        )
    else:  # yz
        change = (
            "change_box all triclinic\n"
            f"change_box all yz delta {sign * strain * lz:.6f} remap"
        )

    sign_txt = "+" if sign > 0 else "-"
    script = f"""# Cij strain job: {mode}, strain {sign_txt}{strain}
units           real
atom_style      full
boundary        p p p

pair_style      lj/class2/coul/long 15.5
kspace_style    pppm 1e-4
bond_style      class2
angle_style     class2
dihedral_style  class2
improper_style  class2
special_bonds   lj/coul 0 0 1

{newton_line}read_data       {data_name}

neighbor        0.3 bin
neigh_modify    delay 0 every 1 check yes

timestep        1.0

velocity        all create 300.0 4928459 dist gaussian
velocity        all zero linear

{change}
kspace_style    pppm 1e-4

fix             nvt all nvt temp 300.0 300.0 100.0
thermo          1000
thermo_style    custom step temp pe ke etotal press vol density
run             {EQUIL_STEPS}

reset_timestep  0

compute         T all temp
compute         p all pressure T virial
fix             av all ave/time {n_every} {n_repeat} {SAMPLE_STEPS} c_p[1] c_p[2] c_p[3] c_p[4] c_p[5] c_p[6] &
                file stress_{mode}_{sign_txt}.txt mode scalar
run             {SAMPLE_STEPS}

write_data      end_{mode}_{sign_txt}.data
print           ">>> {mode} {sign_txt}{strain} done."
"""
    (job_dir / "in.lammps").write_text(script, encoding="ascii")


def run_lammps(lmp, job_dir, omp, gpu):
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(omp)
    cmd = [lmp, "-in", "in.lammps", "-log", "log.lammps", "-screen", "none"]
    if gpu > 0:
        cmd += ["-sf", "gpu", "-pk", "gpu", str(gpu)]
    proc = subprocess.run(
        cmd,
        cwd=str(job_dir),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        log = job_dir / "log.lammps"
        tail = ""
        if log.exists():
            tail = "\n".join(
                log.read_text(encoding="ascii", errors="ignore").splitlines()[-20:]
            )
        raise RuntimeError(f"LAMMPS failed in {job_dir}\n{tail}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmp", default=DEFAULT_LMP)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--strain", type=float, default=STRAIN)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--omp", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpu != 1:
        print("CIJ-simple fixed GPU mode: --gpu forced to 1", flush=True)
        args.gpu = 1
    if not INPUT_DATA.exists() or not INPUT_DUMP.exists():
        raise SystemExit("missing nvt_2000ps_gpu.data or nvt_2000ps_gpu_traj.dump")

    ref_lines = INPUT_DATA.read_text(encoding="ascii").splitlines()
    ref_atoms = parse_ref_atoms(ref_lines)
    box = parse_box(ref_lines)
    lengths = (
        box["xlo"][1] - box["xlo"][0],
        box["ylo"][1] - box["ylo"][0],
        box["zlo"][1] - box["zlo"][0],
    )

    print("collecting frame steps ...", flush=True)
    steps = collect_steps(INPUT_DUMP)
    last = steps[-1]
    target_steps = [int(round(last * k / args.frames / 1000) * 1000)
                    for k in range(1, args.frames + 1)]
    target_steps = sorted(set(s for s in target_steps if s in steps))
    if len(target_steps) < args.frames:
        print(f"warning: using {len(target_steps)} available target frames")

    print("reading frames ...", flush=True)
    frames = read_frames(INPUT_DUMP, target_steps)
    frame_list = [frames[s] for s in sorted(target_steps)]

    total_jobs = len(frame_list) * len(MODES) * len(SIGNS)
    print(f"{len(frame_list)} frames, {total_jobs} strain jobs", flush=True)

    for idx, frame in enumerate(frame_list, start=1):
        frame_dir = SCRIPT_DIR / "frames" / f"frame{idx:02d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        data_name = f"frame{idx:02d}.data"
        build_frame_data(ref_lines, ref_atoms, frame, frame_dir / data_name)
        for mode in MODES:
            for sign in SIGNS:
                sign_txt = "+" if sign > 0 else "-"
                job_dir = (
                    SCRIPT_DIR
                    / "strain"
                    / f"frame{idx:02d}"
                    / f"{mode}{sign_txt}"
                )
                write_strain_input(
                    job_dir,
                    f"../../../frames/frame{idx:02d}/{data_name}",
                    mode,
                    sign,
                    args.strain,
                    lengths,
                )
                if args.run:
                    run_lammps(args.lmp, job_dir, args.omp, args.gpu)
        print(f"frame {idx:02d} ready "
              f"({frame['step']} fs, 12 jobs)", flush=True)

    print(f"generated {total_jobs} inputs under {SCRIPT_DIR / 'strain'}")
    if args.run:
        print("all Cij jobs finished")
    else:
        print("use --run to execute, then run analyze_cij.py")


if __name__ == "__main__":
    main()
