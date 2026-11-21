FKM-SiO2 pullout: external Python controller
=============================================

Files
-----
control_pullout.py          Python controller (launches LAMMPS per velocity)
in.pullout_single.lammps    single-velocity LAMMPS input (Python-controlled)
ctrl.inc                    control file, rewritten by Python to stop LAMMPS

Why external Python?
--------------------
This LAMMPS build has no PYTHON package, so LAMMPS input cannot call Python.
Instead, Python launches one LAMMPS process per velocity, reads the monitor
file, computes the bulk-separation criteria, and rewrites ctrl.inc.  LAMMPS
includes ctrl.inc after every run chunk, so it finishes gracefully as
CONVERGED or ABANDONED without being killed.

Normal run (defaults: 50,45,40,35,30,25,20,15 A/ps)
---------------------------------------------------
python control_pullout.py

Criteria (same as before)
-------------------------
Z0   = 84.728902 A
R95  = 14.956041 A
D    = 2*R95 = 29.912082 A
E0   = |E_gg| at t=0
Phase 0 -> 1 : zdist > R, record E_ref = |E_gg|
Phase 1 -> 2 : zdist > R + D
Phase 2 -> 3 : |E_gg| < 10% * E0
CONVERGED    : 5 frames after phase 3, 5-frame sliding mean <= 1.01*E_ref
ABANDONED    : zdist > R + 4*D without convergence

Outputs (in the same folder)
----------------------------
kickXXAps_monitor.txt        time, z(SiO2), |E_gg|
kickXXAps_traj.dump          trajectory every 0.5 ps
kickXXAps_end_converged.*    final files when converged
kickXXAps_end_abandoned.*    final files when abandoned
kickXXAps_status.txt         CONVERGED / ABANDONED / HARD_STOP
pullout_python_summary.csv   summary of all velocities

Optional fast-test flags
------------------------
python control_pullout.py --velocities 30,40 --chunk 100 --r 1 --d 2 --e0 100,100000
