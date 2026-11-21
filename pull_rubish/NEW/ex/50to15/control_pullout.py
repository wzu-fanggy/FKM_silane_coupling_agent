"""Python controller for the FKM-SiO2 kick gradient.

It launches LAMMPS for one velocity at a time, reads the monitor file,
applies the bulk-separation criteria in Python, and tells LAMMPS to
finish as CONVERGED or ABANDONED by rewriting ctrl.inc.
"""

import argparse
import os
import statistics
import subprocess
import time
from pathlib import Path


DEFAULT_WORKDIR = r"F:\Lammps-MS\FKM-SIO2-TEST\pull\NEW\ex\50to15"
DEFAULT_LMP = r"F:\LAMMPS 64-bit 4Jul2026 with GUI\bin\lmp.exe"
DEFAULT_DATA = r"F:/Lammps-MS/FKM-SIO2-TEST/pull/NEW/ex/pull_equil_600Aadd.data"
DEFAULT_SCRIPT = "in.pullout_single.lammps"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--lmp", default=DEFAULT_LMP)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--script", default=DEFAULT_SCRIPT)
    parser.add_argument("--velocities", default="50,45,40,35,30,25,20,15")
    parser.add_argument("--chunk", type=int, default=1000)
    parser.add_argument("--z0", type=float, default=84.728902)
    parser.add_argument("--r", type=float, default=14.956041)
    parser.add_argument("--d", type=float, default=29.912082)
    parser.add_argument("--e0", default="1260.3837")
    parser.add_argument("--poll", type=float, default=1.0)
    parser.add_argument("--exit-timeout", type=float, default=300.0)
    return parser.parse_args()


def read_monitor(path):
    """Return list of (time_ps, z_sio2, e_abs) parsed from the monitor file."""
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

    # Monitor may contain duplicate lines at run-chunk boundaries; keep last.
    dedup = {}
    for t, z, e in frames:
        dedup[t] = (z, e)
    return [(t, z, e) for t, (z, e) in sorted(dedup.items())]


def write_control(workdir, script, target):
    with open(Path(workdir) / "ctrl.inc", "w", encoding="ascii") as f:
        f.write(f"jump {script} {target}\n")


def run_one(args, velocity, tag, e0):
    workdir = Path(args.workdir)
    monitor = workdir / f"{tag}_monitor.txt"
    status = workdir / f"{tag}_status.txt"
    ctrl = workdir / "ctrl.inc"

    for name in ("_monitor.txt", "_traj.dump", "_status.txt", "_end_converged.data",
                 "_end_converged.restart", "_end_converged.dump",
                 "_end_abandoned.data", "_end_abandoned.restart", "_end_abandoned.dump"):
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
    print(f"[{tag}] launched, velZ={vel_fs} A/fs", flush=True)

    phase = 0
    er_ref = None
    window = []
    last_time = -1.0
    outcome = None
    stop_time = None
    final_e = None
    error_line = None

    while True:
        if proc.poll() is not None:
            break

        frames = read_monitor(monitor)
        for t, z, e in frames:
            if t <= last_time:
                continue
            last_time = t
            zdist = z - args.z0
            final_e = e

            if phase == 0 and zdist > args.r:
                phase = 1
                er_ref = e
            elif phase == 1 and zdist > args.r + args.d:
                phase = 2
            elif phase == 2 and e < 0.1 * e0:
                phase = 3
                window = [e]
            elif phase == 3:
                window.append(e)
                window = window[-5:]

            if phase == 3 and len(window) >= 5 and statistics.mean(window) <= 1.01 * er_ref:
                outcome = "CONVERGED"
                stop_time = t
                write_control(args.workdir, args.script, "done")
                break
            if zdist > args.r + 4 * args.d:
                outcome = "ABANDONED"
                stop_time = t
                write_control(args.workdir, args.script, "abandon")
                break

        if outcome is not None:
            break
        time.sleep(args.poll)

    if outcome is None:
        proc.wait(timeout=args.exit_timeout)
        outcome = "ABANDONED"
        log_path = workdir / f"log.{tag}.txt"
        if log_path.exists():
            for line in reversed(log_path.read_text(encoding="ascii", errors="ignore").splitlines()):
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

    summary = {
        "velocity_Aps": velocity,
        "outcome": outcome,
        "stop_time_ps": stop_time,
        "final_E_gg_kcal_mol": final_e,
        "E_ref_kcal_mol": er_ref,
        "window_mean": statistics.mean(window) if window else None,
        "phase": phase,
        "error": error_line if error_line else "",
    }
    print(f"[{tag}] {summary}", flush=True)
    return summary


def main():
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    velocities = [int(x) for x in args.velocities.split(",")]
    e0s = [float(x) for x in args.e0.split(",")]
    if len(e0s) == 1:
        e0s = e0s * len(velocities)
    if len(e0s) != len(velocities):
        raise SystemExit("--e0 must have 1 value or one per velocity")

    rows = []
    for velocity, e0 in zip(velocities, e0s):
        tag = f"kick{velocity}Aps"
        rows.append(run_one(args, velocity, tag, e0))

    out_csv = workdir / "pullout_python_summary.csv"
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary written to {out_csv}", flush=True)


if __name__ == "__main__":
    main()
