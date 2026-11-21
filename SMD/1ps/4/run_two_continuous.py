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


def compute_coms(corridor_radius: float):
    """Return mobile-FKM z COM and outer-shell COM for corridor mode."""
    lines = (ROOT / "pull_equil.data").read_text(encoding="ascii").splitlines()
    lx = 78.79982443944567 - 21.200175560556755
    ly = lx
    cx, cy = 51.586, 52.991

    total_m = [[0.0, 0.0, 0.0], 0]
    corr_m = [[0.0, 0.0, 0.0], 0]
    outer_m = [[0.0, 0.0, 0.0], 0]

    in_atoms = False
    for raw in lines:
        s = raw.strip()
        if s == "Atoms # full":
            in_atoms = True
            continue
        if in_atoms:
            if not s:
                continue
            if s.startswith("Bonds"):
                break
            p = s.split()
            if len(p) < 10:
                continue
            typ = int(p[2])
            if typ > 6:
                continue
            x = float(p[4]) + int(p[7]) * lx
            y = float(p[5]) + int(p[8]) * ly
            z = float(p[6])
            if z <= 30.0:
                continue
            total_m[0][0] += x
            total_m[0][1] += y
            total_m[0][2] += z
            total_m[1] += 1

            dx = x - cx
            dy = y - cy
            dx -= round(dx / lx) * lx
            dy -= round(dy / ly) * ly
            inside = dx * dx + dy * dy <= corridor_radius * corridor_radius
            if inside:
                corr_m[0][0] += x
                corr_m[0][1] += y
                corr_m[0][2] += z
                corr_m[1] += 1
            else:
                outer_m[0][0] += x
                outer_m[0][1] += y
                outer_m[0][2] += z
                outer_m[1] += 1

    def avg(v, n):
        return tuple(round(v[i] / n, 4) for i in range(3)) if n else None

    return {
        "total_mobile": avg(total_m[0], total_m[1]),
        "corridor_mobile": avg(corr_m[0], corr_m[1]),
        "outer": avg(outer_m[0], outer_m[1]),
    }


def write_colvars(
    path: Path,
    target_z: float,
    nrun: int,
    fkm_group: str,
    fkm_center_z: float,
    outer_com: tuple | None = None,
) -> None:
    if outer_com is not None:
        idx_line = "indexFile groups.ndx"
        ox, oy, oz = outer_com
        outer_colvar_block = f"""
colvar {{
  name x_outer
  distanceZ {{
    main {{ indexGroup outer }}
    ref {{ dummyAtom ({ox:.4f}, {oy:.4f}, 0.0) }}
    axis (1.0, 0.0, 0.0)
  }}
}}

colvar {{
  name y_outer
  distanceZ {{
    main {{ indexGroup outer }}
    ref {{ dummyAtom ({ox:.4f}, {oy:.4f}, 0.0) }}
    axis (0.0, 1.0, 0.0)
  }}
}}

colvar {{
  name z_outer
  distanceZ {{
    main {{ indexGroup outer }}
    ref {{ dummyAtom (0.0, 0.0, 0.0) }}
    axis (0.0, 0.0, 1.0)
  }}
}}
"""
        outer_harmonic_block = f"""
harmonic {{
  name hold_x_outer
  colvars x_outer
  forceConstant 100.0
  centers 0.0
}}

harmonic {{
  name hold_y_outer
  colvars y_outer
  forceConstant 100.0
  centers 0.0
}}

harmonic {{
  name hold_z_outer
  colvars z_outer
  forceConstant 100.0
  centers {oz:.4f}
}}
"""
    else:
        idx_line = "indexFile FKM.ndx"
        outer_colvar_block = ""
        outer_harmonic_block = ""

    text = f"""colvarsTrajFrequency 1000
{idx_line}

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
    main {{ indexGroup {fkm_group} }}
    ref {{ dummyAtom (0.0, 0.0, 0.0) }}
    axis (0.0, 0.0, 1.0)
  }}
}}

{outer_colvar_block}
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

{outer_harmonic_block}
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
  centers {fkm_center_z:.4f}
}}
"""
    path.write_text(text, encoding="ascii")


def make_input(mode: str, nrun: int, corridor_radius: float = 25.0) -> Path:
    if mode == "corridor":
        anchor_block = f"""
region          bottom block INF INF INF INF 0.0 30.0
region          corridor cylinder z 51.586 52.991 {corridor_radius:.1f} INF INF
region          outer_block block INF INF INF INF 30.0 INF

group           anchor_bottom region bottom
group           corr region corridor
group           outer region outer_block
group           outer subtract outer corr
group           anchor_bottom subtract anchor_bottom SIO2
group           outer subtract outer SIO2
group           anchor union anchor_bottom outer
"""
    else:
        anchor_block = """
region          bottom block INF INF INF INF 0.0 30.0
group           anchor region bottom
group           anchor subtract anchor SIO2
"""

    if mode == "corridor":
        fix_block = """
velocity        anchor_bottom set 0.0 0.0 0.0

fix             rigid_SIO2 SIO2 rigid single torque * off off off
fix             nvt_matrix mobile_FKM nvt temp 300.0 300.0 100.0
fix             nve_outer outer nve
fix             freeze_bottom anchor_bottom setforce 0.0 0.0 0.0
"""
    else:
        fix_block = """
velocity        anchor set 0.0 0.0 0.0

fix             rigid_SIO2 SIO2 rigid single torque * off off off
fix             nvt_matrix mobile_FKM nvt temp 300.0 300.0 100.0
fix             freeze_anchor anchor setforce 0.0 0.0 0.0
"""

    ndx_line = (
        "group2ndx groups.ndx mobile_FKM outer"
        if mode == "corridor"
        else "group2ndx FKM.ndx mobile_FKM"
    )

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

{ndx_line}

{fix_block}
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


def run_lmp(mode: str, threads: int, nrun: int, corridor_radius: float) -> None:
    outdir = ROOT / mode
    outdir.mkdir(parents=True, exist_ok=True)

    src = ROOT / "pull_equil.data"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    shutil.copyfile(src, outdir / "pull_equil.data")

    target_z = 50.618 + (float(nrun) * 0.001)  # 1 A/ps
    coms = compute_coms(corridor_radius)
    if mode == "corridor":
        fkm_center_z = coms["corridor_mobile"][2]
        outer_com = coms["outer"]
    else:
        fkm_center_z = coms["total_mobile"][2]
        outer_com = None
    write_colvars(
        outdir / "colvars.inp",
        target_z,
        nrun,
        "mobile_FKM",
        fkm_center_z,
        outer_com,
    )
    inp = make_input(mode, nrun, corridor_radius)
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
    ap.add_argument("--corridor-radius", type=float, default=25.0)
    args = ap.parse_args()

    run_lmp("corridor", args.threads, args.nrun, args.corridor_radius)
    run_lmp("total", args.threads, args.nrun, args.corridor_radius)

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
