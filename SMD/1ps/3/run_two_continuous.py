"""Run corridor and total pull-out SMDs with one Python driver.

Mode corridor:
   20 A cylindrical free corridor around SiO2, outer FKM anchored.
   Result approximates W_pull-out.

Mode total:
   Full embedded FKM, only bottom anchored.
   Result approximates W_total.

Both modes:
   1 A/ps, 200 ps (200000 steps), 200 A total displacement.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LMP = r"F:\LAMMPS 64-bit 4Jul2026 with GUI\bin\lmp.exe"


def write_colvars(path: Path, target_z: float, nrun: int) -> None:
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

harmonic {{
  name pull_sio2
  colvars z_sio2
  forceConstant 100.0
  centers 50.618
  targetCenters {target_z:.6f}
  targetNumSteps {nrun}
  outputAccumulatedWork on
  outputCenters on
}}

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


def make_input(mode: str, nrun: int) -> Path:
    if mode == "corridor":
        anchor_block = """
region          bottom block INF INF INF INF 0.0 30.0
region          corridor cylinder z 51.586 52.991 20.0 INF INF
region          outer_block block INF INF INF INF 30.0 INF

group           anchor region bottom
group           corr region corridor
group           outer region outer_block
group           outer subtract outer corr
group           anchor union anchor outer
group           anchor subtract anchor SIO2
"""
    else:
        anchor_block = """
region          bottom block INF INF INF INF 0.0 30.0
group           anchor region bottom
group           anchor subtract anchor SIO2
"""

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

read_data       pull_equil.data

group           SIO2 type 7 8 9
group           FKM type 1 2 3 4 5 6
{anchor_block}
group           mobile_FKM subtract FKM anchor

group2ndx       FKM.ndx FKM

velocity        anchor set 0.0 0.0 0.0

fix             nve_sio2 SIO2 nve
fix             nvt_matrix mobile_FKM nvt temp 300.0 300.0 100.0
fix             freeze_anchor anchor setforce 0.0 0.0 0.0
fix             Colvars all colvars colvars.inp output smd
fix_modify      Colvars energy yes

variable        Step equal step
variable        Tps  equal step*0.001
variable        xs equal xcm(SIO2,x)
variable        ys equal xcm(SIO2,y)
variable        zs equal xcm(SIO2,z)
variable        zf equal xcm(FKM,z)
variable        zfmx equal bound(FKM,zmax)
variable        zsmin equal bound(SIO2,zmin)
variable        zsmax equal bound(SIO2,zmax)

thermo          1000
thermo_style    custom step v_Tps temp pe ke etotal press v_xs v_ys v_zs v_zf v_zfmx v_zsmin v_zsmax

fix             mon all print 1000 "${{Step}} ${{Tps}} ${{xs}} ${{ys}} ${{zs}} ${{zf}} ${{zfmx}} ${{zsmin}} ${{zsmax}}" file smd_monitor.txt screen no

dump            traj all custom 1000 smd_traj.dump id type x y z ix iy iz
dump_modify     traj sort id flush yes

run             {nrun}

write_restart   smd_end.restart
write_data      smd_end.data
write_dump      all custom smd_end.dump id type x y z ix iy iz
"""
    inp = ROOT / f"in.{mode}.lammps"
    inp.write_text(text, encoding="ascii")
    return inp


def run_lmp(mode: str, threads: int, nrun: int) -> None:
    outdir = ROOT / mode
    outdir.mkdir(parents=True, exist_ok=True)

    src = ROOT / "pull_equil.data"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    shutil.copyfile(src, outdir / "pull_equil.data")

    target_z = 50.618 + (float(nrun) * 0.001)  # 1 A/ps
    write_colvars(outdir / "colvars.inp", target_z, nrun)
    inp = make_input(mode, nrun)
    shutil.copyfile(inp, outdir / inp.name)

    cmd = [
        LMP,
        "-sf", "omp",
        "-pk", "omp", str(threads),
        "-in", inp.name,
        "-log", f"log.{mode}.txt",
        "-screen", "none",
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)

    t0 = time.perf_counter()
    print(f"[{datetime.now():%H:%M:%S}] {mode}: LAMMPS launched, nrun={nrun}")
    subprocess.run(cmd, cwd=str(outdir), check=True, env=env)
    dt = time.perf_counter() - t0
    print(f"[{datetime.now():%H:%M:%S}] {mode}: finished in {dt/60:.1f} min")


def read_work(outdir: Path) -> float:
    state = outdir / "smd.colvars.state"
    if state.exists():
        m = re.search(r"accumulatedWork\s+([-\d.eE+]+)", state.read_text(encoding="ascii"))
        if m:
            return float(m.group(1))
    traj = outdir / "smd.colvars.traj"
    if traj.exists():
        rows = [ln.split() for ln in traj.read_text(encoding="ascii").splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")]
        if rows:
            return float(rows[-1][-1])
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrun", type=int, default=200000)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--corridor-radius", type=float, default=20.0)
    args = ap.parse_args()

    # Radius is used by make_input via the hard-coded region line. Keeping it
    # in the parser documents the parameter; changing it requires regenerating.
    if args.corridor_radius != 20.0:
        print("Warning: corridor radius is currently fixed at 20.0 A in make_input().")

    run_lmp("corridor", args.threads, args.nrun)
    run_lmp("total", args.threads, args.nrun)

    w_corridor = read_work(ROOT / "corridor")
    w_total = read_work(ROOT / "total")
    w_bulk = w_total - w_corridor

    summary = ROOT / "decomposition_summary.csv"
    with summary.open("w", encoding="ascii") as f:
        f.write("metric,value\n")
        f.write(f"w_pullout_corridor_kcal_mol,{w_corridor:.3f}\n")
        f.write(f"w_total_full_kcal_mol,{w_total:.3f}\n")
        f.write(f"w_bulk_kcal_mol,{w_bulk:.3f}\n")

    print()
    print("W_pull-out (corridor) :", round(w_corridor, 3), "kcal/mol")
    print("W_total (full)        :", round(w_total, 3), "kcal/mol")
    print("W_bulk                :", round(w_bulk, 3), "kcal/mol")
    print("summary               :", summary)


if __name__ == "__main__":
    main()
