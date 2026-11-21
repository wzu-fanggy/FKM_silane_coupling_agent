FKM-SiO2 pullout: scientific Python controller
================================================

Single entry point
------------------
run_pullout.py:
  - computes geometry (Z0, R_SURF, D) if geometry_params.txt is missing
  - launches in.pullout_single.lammps for each velocity (50 -> 15 A/ps)
  - reads the monitor every 0.5 s
  - rewrites ctrl.inc to stop LAMMPS as CONVERGED or ABANDONED

Threshold
---------
E0 is read from LAMMPS monitor frame 0 (time 0), so it is computed by
LAMMPS for the current SiO2 system.  Pass threshold:

    |E_gg| <= 0.10 * E0

Abandon threshold:

    zdist = z(SiO2 COM) - Z0 > R_SURF + 4*D

No convergence statistics are used; the threshold is the decision rule.

Box
---
The data file used by the scripts is:

    F:\Lammps-MS\FKM-SIO2-TEST\pull\NEW\sci\pull_equil_600Aadd.data

Its Z box is extended to 1000 A so the 15/20 A/ps cases do not hit the
top boundary before the 4D abandon line.

Run
---
cd F:\Lammps-MS\FKM-SIO2-TEST\pull\NEW\sci
python run_pullout.py

Fast test example
-----------------
python run_pullout.py --velocities 30,40 --chunk 100 --confirm 1

Outputs
-------
kickXXAps_monitor.txt        time, z(SiO2), |E_gg|
kickXXAps_traj.dump          trajectory
kickXXAps_end_converged.*    final files when converged
kickXXAps_end_abandoned.*    final files when abandoned
kickXXAps_status.txt         status written by LAMMPS
log.kickXXAps.txt            LAMMPS log
pullout_python_summary.csv   summary of all velocities
