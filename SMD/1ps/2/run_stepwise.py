"""Stepwise pull + NVT relaxation driver for FKM/SiO2 SMD.

Each cycle:
  1. Pull: move the Colvars harmonic center by step_size at pull_velocity.
  2. Relax: keep the harmonic center fixed and run NVT relaxation.

Default settings (see README):
  pull velocity : 1 A/ps
  step size     : 2 A
  pull time     : 2 ps (2000 steps)
  relax time    : 5 ps (5000 steps)
  total cycles  : 30 (60 A total)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LMP = r"F:\LAMMPS 64-bit 4Jul2026 with GUI\bin\lmp.exe"


def write_colvars(path: Path, c0: float, c1: float, n_pull: int, moving: bool) -> None:
    if moving:
        pull_block = f"""harmonic {{
  name pull_sio2
  colvars z_sio2
  forceConstant 100.0
  centers {c0:.6f}
  targetCenters {c1:.6f}
  targetNumSteps {n_pull}
  outputAccumulatedWork on
  outputCenters on
}}"""
    else:
        pull_block = f"""harmonic {{
  name pull_sio2
  colvars z_sio2
  forceConstant 100.0
  centers {c1:.6f}
}}"""

    text = f"""colvarsTrajFrequency 1000
indexFile FKM.ndx

colvar {{
  name z_sio2
  distanceZ {{
    main {{ atomNumbersRange 1-1470 }}
    ref {{ dummyAtom (0.0, 0.0, 0.0) }}
    axis (0.0, 0.0, 1.0)
  }}
  outputAppliedForce on
}}

colvar {{
  name x_sio2
  distanceZ {{
    main {{ atomNumbersRange 1-1470 }}
    ref {{ dummyAtom (51.594, 52.935, 0.0) }}
    axis (1.0, 0.0, 0.0)
  }}
}}

colvar {{
  name y_sio2
  distanceZ {{
    main {{ atomNumbersRange 1-1470 }}
    ref {{ dummyAtom (51.594, 52.935, 0.0) }}
    axis (0.0, 1.0, 0.0)
  }}
}}

colvar {{
  name z_fkm
  distanceZ {{
    main {{ indexGroup FKM }}
    ref {{ dummyAtom (0.0, 0.0, 0.0) }}
    axis (0.0, 0.0, 1.0)
  }}
}}

{pull_block}

harmonic {{
  name hold_x_sio2
  colvars x_sio2
  forceConstant 100.0
  centers 0.0
}}

harmonic {{
  name hold_y_sio2
  colvars y_sio2
  forceConstant 100.0
  centers 0.0
}}

harmonic {{
  name hold_fkm
  colvars z_fkm
  forceConstant 10.0
  centers 50.647
}}
"""
    path.write_text(text, encoding="ascii")


def make_input(
    phase: str,
    cycle: int,
    data_in: Path,
    data_out: Path,
    colvars_file: Path,
    prefix: str,
    monitor_file: Path,
    nsteps: int,
    start_step: int,
) -> Path:
    inp = ROOT / f"in.step_{cycle:03d}_{phase}.lammps"
    text = f"""units           real
atom_style      full
boundary        p p f

pair_style      lj/class2/coul/long 15.5
bond_style      class2
angle_style     class2
dihedral_style  class2
improper_style  class2
special_bonds   lj/coul 0 0 1
kspace_style    pppm 1e-4
kspace_modify   slab 3.0

read_data       {data_in.as_posix()}

reset_timestep  {start_step}

group           SIO2 type 7 8 9
group           FKM type 1 2 3 4 5 6
region          anchor_region block INF INF INF INF 0.0 30.0
group           anchor region anchor_region
group           anchor subtract anchor SIO2
group           mobile_FKM subtract FKM anchor

group2ndx       FKM.ndx FKM

velocity        anchor set 0.0 0.0 0.0

fix             nve_sio2 SIO2 nve
fix             nvt_matrix mobile_FKM nvt temp 300.0 300.0 100.0
fix             freeze_anchor anchor setforce 0.0 0.0 0.0
fix             Colvars all colvars {colvars_file.as_posix()} output {prefix}
fix_modify      Colvars energy yes

variable        Step equal step
variable        xs equal xcm(SIO2,x)
variable        ys equal xcm(SIO2,y)
variable        zs equal xcm(SIO2,z)
variable        zf equal xcm(FKM,z)

fix             mon all print {nsteps} "{cycle} {phase} ${{Step}} ${{xs}} ${{ys}} ${{zs}} ${{zf}}" file {monitor_file.as_posix()} screen no

dump            traj all custom 1000 seg_{cycle:03d}_{phase}.dump id type x y z ix iy iz
dump_modify     traj sort id flush yes

run             {nsteps}

write_data      {data_out.as_posix()}
"""
    inp.write_text(text, encoding="ascii")
    return inp


def run_lmp(inp: Path, log: Path, threads: int) -> None:
    cmd = [
        LMP,
        "-sf", "omp",
        "-pk", "omp", str(threads),
        "-in", str(inp),
        "-log", str(log),
        "-screen", "none",
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    subprocess.run(cmd, cwd=str(ROOT), check=True, env=env)


def safe_unlink(path: Path) -> None:
    for _ in range(6):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(0.5)
    path.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=30)
    ap.add_argument("--step-size", type=float, default=2.0)
    ap.add_argument("--pull-velocity", type=float, default=1.0)
    ap.add_argument("--n-pull", type=int, default=None)
    ap.add_argument("--n-relax", type=int, default=5000)
    ap.add_argument("--start-z", type=float, default=50.618)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    if args.n_pull is None:
        args.n_pull = max(1, int(round(args.step_size / args.pull_velocity * 1000.0)))

    data = ROOT / "pull_equil.data"
    if not data.exists():
        raise SystemExit(f"missing {data}")

    monitor = ROOT / "stepwise_monitor.txt"
    monitor.unlink(missing_ok=True)
    traj = ROOT / "stepwise_traj.dump"
    traj.unlink(missing_ok=True)

    state = ROOT / "state.data"
    shutil.copyfile(data, state)
    global_step = 0
    t0 = time.perf_counter()
    total_steps = args.cycles * (args.n_pull + args.n_relax)
    steps_done = 0

    print(f"[START] {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"cycles={args.cycles} pull={args.n_pull} relax={args.n_relax} "
          f"total_steps={total_steps}")

    for i in range(args.cycles):
        c0 = args.start_z + i * args.step_size
        c1 = c0 + args.step_size
        t_cycle = time.perf_counter()

        cv_pull = ROOT / f"colvars_pull_{i:03d}.inp"
        write_colvars(cv_pull, c0, c1, args.n_pull, moving=True)
        state_pull = ROOT / "state_pull.data"
        mon_pull = ROOT / f"monitor_pull_{i:03d}.txt"
        prefix_pull = f"pull_{i:03d}"
        inp = make_input(
            "pull", i, state, state_pull, cv_pull,
            prefix_pull, mon_pull, args.n_pull, global_step,
        )
        print(f"[{datetime.now():%H:%M:%S}] "
              f"[cycle {i+1:03d}/{args.cycles}] pull "
              f"z {c0:.2f} -> {c1:.2f} A")
        run_lmp(inp, ROOT / f"log_pull_{i:03d}.txt", args.threads)
        global_step += args.n_pull
        print(f"[{datetime.now():%H:%M:%S}]   pull done "
              f"({time.perf_counter()-t_cycle:.1f} s)")

        cv_relax = ROOT / f"colvars_relax_{i:03d}.inp"
        write_colvars(cv_relax, c0, c1, args.n_pull, moving=False)
        state_relax = ROOT / "state_relax.data"
        mon_relax = ROOT / f"monitor_relax_{i:03d}.txt"
        prefix_relax = f"relax_{i:03d}"
        inp = make_input(
            "relax", i, state_pull, state_relax, cv_relax,
            prefix_relax, mon_relax, args.n_relax, global_step,
        )
        print(f"[{datetime.now():%H:%M:%S}] "
              f"[cycle {i+1:03d}/{args.cycles}] relax at z = {c1:.2f} A")
        run_lmp(inp, ROOT / f"log_relax_{i:03d}.txt", args.threads)
        global_step += args.n_relax
        t_cycle = time.perf_counter() - t_cycle
        steps_done += args.n_pull + args.n_relax
        elapsed = time.perf_counter() - t0
        if steps_done > 0:
            eta = elapsed / steps_done * (total_steps - steps_done)
        else:
            eta = 0.0
        print(f"[{datetime.now():%H:%M:%S}]   cycle done "
              f"({t_cycle:.1f} s) | elapsed {elapsed/60:.1f} min | "
              f"ETA {eta/60:.1f} min")

        cycle_data = ROOT / f"cycle_{i:03d}.data"
        shutil.copyfile(state_relax, cycle_data)
        shutil.copyfile(state_relax, state)

        with monitor.open("a", encoding="ascii") as f:
            for p in (mon_pull, mon_relax):
                if p.exists():
                    f.write(p.read_text(encoding="ascii"))
        for p in (mon_pull, mon_relax):
            safe_unlink(p)

        for p in (ROOT / f"seg_{i:03d}_pull.dump", ROOT / f"seg_{i:03d}_relax.dump"):
            if p.exists():
                with traj.open("ab") as out, p.open("rb") as src:
                    out.write(src.read())
                safe_unlink(p)

        print(f"[cycle {i+1:03d}/{args.cycles}] z {c0:.2f} -> {c1:.2f} A")

    final = ROOT / "stepwise_final.data"
    shutil.copyfile(state, final)
    print(f"[END] {datetime.now():%Y-%m-%d %H:%M:%S} "
          f"elapsed {(time.perf_counter()-t0)/60:.1f} min")
    print("Stepwise pull + relaxation finished.")
    print("Monitor :", monitor)
    print("Trajectory :", traj)
    print("Final   :", final)


if __name__ == "__main__":
    main()
