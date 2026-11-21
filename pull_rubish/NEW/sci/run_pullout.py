"""One-file entry point for the FKM-SiO2 kick gradient.

Computes geometry, launches LAMMPS per velocity, applies the 10%-of-E0
threshold, and writes CONVERGED/ABANDONED outputs.

Run:
    python run_pullout.py
"""

import argparse
import datetime
import math
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_WORKDIR = r"F:\Lammps-MS\FKM-SIO2-TEST\pull\NEW\sci"
DEFAULT_LMP = r"F:\LAMMPS 64-bit 4Jul2026 with GUI\bin\lmp.exe"
DEFAULT_DATA = r"F:/Lammps-MS/FKM-SIO2-TEST/pull/NEW/sci/pull_equil_600Aadd.data"
DEFAULT_SCRIPT = "in.pullout_single.lammps"


def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--lmp", default=DEFAULT_LMP)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--script", default=DEFAULT_SCRIPT)
    parser.add_argument("--velocities", default="50,45,40,35,30,25,20,15")
    parser.add_argument("--chunk", type=int, default=1000)
    parser.add_argument("--residual-frac", type=float, default=0.10)
    parser.add_argument("--confirm", type=int, default=1)
    parser.add_argument("--poll", type=float, default=0.5)
    parser.add_argument("--exit-timeout", type=float, default=300.0)
    parser.add_argument("--z0", type=float, default=None)
    parser.add_argument("--r", type=float, default=None)
    parser.add_argument("--d", type=float, default=None)
    parser.add_argument("--geometry", default=None)
    return parser.parse_args()


def parse_box(path):
    box = {}
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4 and parts[2] in ("xlo", "ylo", "zlo"):
                box[parts[2][0]] = (float(parts[0]), float(parts[1]))
    return box


def parse_atoms(path):
    atoms = []
    in_atoms = False
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.startswith("Atoms"):
                in_atoms = True
                continue
            if s.startswith("Bonds"):
                in_atoms = False
                continue
            if not in_atoms or not s:
                continue
            parts = s.split()
            if len(parts) != 10:
                continue
            try:
                atoms.append(
                    (
                        int(float(parts[2])),
                        float(parts[4]),
                        float(parts[5]),
                        float(parts[6]),
                    )
                )
            except ValueError:
                continue
    return atoms


def compute_geometry(data_path):
    box = parse_box(data_path)
    if len(box) != 3:
        raise SystemExit("Could not parse box bounds.")
    area = (box["x"][1] - box["x"][0]) * (box["y"][1] - box["y"][0])

    atoms = parse_atoms(data_path)
    fkm_z = [a[3] for a in atoms if 1 <= a[0] <= 6]
    sio2 = [(a[1], a[2], a[3]) for a in atoms if 7 <= a[0] <= 9]
    if not fkm_z or not sio2:
        raise SystemExit("No FKM/SiO2 atoms found.")

    bin_w = 1.0
    zmax = max(fkm_z) + 2.0
    nb = max(1, int(math.ceil(zmax / bin_w)))
    counts = [0] * nb
    for z in fkm_z:
        counts[min(nb - 1, int(z // bin_w))] += 1
    dens = [c / (area * bin_w) for c in counts]
    positive = sorted(d for d in dens if d > 0)
    bulk = positive[int(0.9 * (len(positive) - 1))]
    thr = 0.05 * bulk
    z_top = 0.0
    for i in range(nb - 1, -1, -1):
        if dens[i] >= thr:
            z_top = (i + 1) * bin_w
            break

    n = len(sio2)
    com_x = sum(a[0] for a in sio2) / n
    com_y = sum(a[1] for a in sio2) / n
    com_z = sum(a[2] for a in sio2) / n
    radii = [
        math.sqrt((a[0] - com_x) ** 2 + (a[1] - com_y) ** 2 + (a[2] - com_z) ** 2)
        for a in sio2
    ]
    rbin = 0.5
    rmax = max(radii) + 1.0
    nrb = max(1, int(math.ceil(rmax / rbin)))
    rcounts = [0] * nrb
    for r in radii:
        rcounts[min(nrb - 1, int(r // rbin))] += 1
    rdens = []
    for i, c in enumerate(rcounts):
        r0 = i * rbin
        r1 = r0 + rbin
        vol = (4.0 / 3.0) * math.pi * (r1**3 - r0**3)
        rdens.append(c / vol if vol > 0 else 0.0)
    peak = max(rdens)
    r_surf = 0.0
    for i in range(nrb - 1, -1, -1):
        if rdens[i] >= 0.05 * peak:
            r_surf = (i + 1) * rbin
            break

    return {"Z0": z_top, "R_SURF": r_surf, "D": 2.0 * r_surf}


class ThresholdDetector:
    def __init__(self, z0, r, d, residual_frac=0.10, confirm=1, e0=None):
        self.z0 = z0
        self.r = r
        self.d = d
        self.residual_frac = residual_frac
        self.confirm = confirm
        self.e0 = e0
        self.state = "BEFORE"
        self.er_ref = None
        self.zdist = None
        self.last_time = None
        self.below_count = 0

    def add(self, time_ps, z_sio2, e_abs):
        if self.e0 is None:
            self.e0 = e_abs
        self.last_time = time_ps
        self.zdist = z_sio2 - self.z0
        info = {
            "state": self.state,
            "time_ps": time_ps,
            "zdist": self.zdist,
            "E_ref_R": self.er_ref,
            "E0": self.e0,
            "threshold": None,
            "E_gg": e_abs,
        }

        if self.state == "BEFORE":
            if self.zdist > self.r:
                self.state = "ENTERED"
                self.er_ref = e_abs
                info["state"] = self.state
                info["E_ref_R"] = self.er_ref
            return info

        threshold = self.residual_frac * self.e0
        info["threshold"] = threshold
        if self.zdist > self.r + 4 * self.d:
            self.state = "ABANDONED"
            info["state"] = self.state
            info["reason"] = "4D reached before E_gg dropped below threshold"
            return info

        if e_abs <= threshold:
            self.below_count += 1
        else:
            self.below_count = 0
        if self.below_count >= self.confirm:
            self.state = "CONVERGED"
            info["state"] = self.state
            info["reason"] = "E_gg below residual threshold"
        return info


def read_monitor(path):
    frames = []
    if not path.exists():
        return frames
    try:
        with open(path, "r", encoding="ascii", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    t = float(parts[0])
                    z = float(parts[1])
                    e = abs(float(parts[2]))
                except ValueError:
                    continue
                frames.append((t, z, e))
    except OSError:
        return frames
    dedup = {}
    for t, z, e in frames:
        dedup[t] = (z, e)
    return [(t, z, e) for t, (z, e) in sorted(dedup.items())]


def write_control(workdir, script, target):
    with open(Path(workdir) / "ctrl.inc", "w", encoding="ascii") as f:
        value = "1" if target == "done" else "2"
        f.write(f"variable stop equal {value}\n")


def run_one(args, velocity, tag, z0, r, d):
    t_start = time.time()
    start_str = now_str()
    workdir = Path(args.workdir)
    monitor = workdir / f"{tag}_monitor.txt"
    status = workdir / f"{tag}_status.txt"
    ctrl = workdir / "ctrl.inc"

    for name in (
        "_monitor.txt",
        "_traj.dump",
        "_status.txt",
        "_end_converged.data",
        "_end_converged.restart",
        "_end_converged.dump",
        "_end_abandoned.data",
        "_end_abandoned.restart",
        "_end_abandoned.dump",
    ):
        p = workdir / (tag + name)
        if p.exists():
            p.unlink()
    with open(ctrl, "w", encoding="ascii") as f:
        f.write("# empty\n")

    vel_fs = f"{velocity / 1000.0:.6f}"
    cmd = [
        args.lmp,
        "-in", args.script,
        "-var", "velZ", vel_fs,
        "-var", "tag", tag,
        "-var", "chunk", str(args.chunk),
        "-log", f"log.{tag}.txt",
        "-screen", "none",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(workdir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(
        f"[{tag}] start {start_str} | LAMMPS launched {now_str()} | "
        f"velZ={vel_fs} A/fs",
        flush=True,
    )

    detector = ThresholdDetector(
        z0=z0,
        r=r,
        d=d,
        residual_frac=args.residual_frac,
        confirm=args.confirm,
    )
    last_time = -1.0
    outcome = None
    stop_time = None
    final_e = None
    error_line = None
    info = {}
    e0_printed = False

    while True:
        if proc.poll() is not None:
            break
        frames = read_monitor(monitor)
        for t, z, e in frames:
            if t <= last_time:
                continue
            last_time = t
            final_e = e
            info = detector.add(t, z, e)
            if not e0_printed and detector.e0 is not None:
                threshold = args.residual_frac * detector.e0
                print(
                    f"[{tag}] E0={detector.e0:.3f} kcal/mol | "
                    f"threshold={threshold:.3f} kcal/mol | {now_str()}",
                    flush=True,
                )
                e0_printed = True
            if detector.state == "CONVERGED":
                outcome = "CONVERGED"
                stop_time = t
                write_control(args.workdir, args.script, "done")
                print(
                    f"[{tag}] decision CONVERGED at t={t:.2f} ps ({now_str()})",
                    flush=True,
                )
                break
            if detector.state == "ABANDONED":
                outcome = "ABANDONED"
                stop_time = t
                write_control(args.workdir, args.script, "abandon")
                print(
                    f"[{tag}] decision ABANDONED at t={t:.2f} ps ({now_str()})",
                    flush=True,
                )
                break
        if outcome is not None:
            break
        time.sleep(args.poll)

    if outcome is None:
        proc.wait(timeout=args.exit_timeout)
        outcome = "ABANDONED"
        log_path = workdir / f"log.{tag}.txt"
        if log_path.exists():
            for line in reversed(
                log_path.read_text(encoding="ascii", errors="ignore").splitlines()
            ):
                if "ERROR" in line:
                    error_line = line.strip()
                    break
        if error_line:
            outcome = "CRASH"
            print(f"[{tag}] CRASH: {error_line}", flush=True)
        elif status.exists():
            txt = status.read_text(encoding="ascii", errors="ignore").upper()
            if "CONVERGED" in txt:
                outcome = "CONVERGED"
            elif "HARD_STOP" in txt:
                outcome = "HARD_STOP"
    else:
        try:
            proc.wait(timeout=args.exit_timeout)
        except subprocess.TimeoutExpired:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)
                print(f"[{tag}] WARNING: killed after control timeout", flush=True)
            else:
                print(f"[{tag}] LAMMPS exited after control timeout", flush=True)
    print(f"[{tag}] LAMMPS exited {now_str()}", flush=True)

    threshold = args.residual_frac * detector.e0 if detector.e0 is not None else None
    end_str = now_str()
    elapsed = time.time() - t_start
    summary = {
        "velocity_Aps": velocity,
        "outcome": outcome,
        "start_time": start_str,
        "end_time": end_str,
        "elapsed_s": round(elapsed, 1),
        "stop_time_ps": stop_time,
        "E0_kcal_mol": detector.e0,
        "threshold_kcal_mol": threshold,
        "E_ref_R_kcal_mol": detector.er_ref,
        "final_E_gg_kcal_mol": final_e,
        "state": detector.state,
        "error": error_line if error_line else "",
    }
    print(
        f"[{tag}] end {end_str} | elapsed={elapsed:.1f} s | outcome={outcome}",
        flush=True,
    )
    return summary


def main():
    overall_start = time.time()
    print(f"[START] {now_str()}", flush=True)
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    geometry_path = (
        Path(args.geometry)
        if args.geometry
        else workdir / "geometry_params.txt"
    )
    if geometry_path.exists():
        geom = {}
        for line in geometry_path.read_text(encoding="ascii").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                geom[key.strip()] = float(value.strip())
    else:
        geom = compute_geometry(args.data)
        lines = [
            f"Z0 = {geom['Z0']:.6f}",
            f"R_SURF = {geom['R_SURF']:.6f}",
            f"D = {geom['D']:.6f}",
        ]
        geometry_path.write_text("\n".join(lines) + "\n", encoding="ascii")

    z0 = args.z0 if args.z0 is not None else geom["Z0"]
    r = args.r if args.r is not None else geom["R_SURF"]
    d = args.d if args.d is not None else geom["D"]
    print(f"Geometry: Z0={z0:.3f} A, R_SURF={r:.3f} A, D={d:.3f} A", flush=True)

    velocities = [int(x) for x in args.velocities.split(",")]
    rows = []
    for velocity in velocities:
        tag = f"kick{velocity}Aps"
        rows.append(run_one(args, velocity, tag, z0, r, d))

    out_csv = workdir / "pullout_python_summary.csv"
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary written to {out_csv}", flush=True)
    print(
        f"[END] {now_str()} | total elapsed={time.time()-overall_start:.1f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
