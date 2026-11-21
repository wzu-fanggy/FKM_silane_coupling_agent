Kick gradient scan (single script, MS-style, dt = 0.5 fs)
==========================================================

Input:
  pull_equil.data (same directory, 400 A slab from pull/NEW)

Velocities:
  15, 17, 19, 21, 23, 25, 27, 29 A/ps

One LAMMPS input (runs all velocities by itself):
  in.kick_gradient.lammps

No shell wrapper is needed. Run:
  lmp -in in.kick_gradient.lammps

Each velocity (tag = kick_15Aps, kick_17Aps, ...) writes:
  kick_15Aps_monitor.txt
  kick_15Aps_traj.dump
  kick_15Aps_end.data / .restart / .dump
  log.kick_15Aps.txt

Summarize:
  python collect_kick_gradient.py

Defaults:
  dt     = 0.5 fs
  Nsteps = 16000 (8 ps)
  Ensemble = NVE, no thermostat, no SMD spring
