FKM/SiO2 continuous SMD via Colvars
====================================

This replaces the deprecated fix smd with fix colvars.

Input:
  pull_equil.data                 (already in F:\Lammps-MS\FKM-SIO2-TEST\SMD)

Files:
  in.pull_smd_colvars.lammps      LAMMPS input
  colvars.inp                     Colvars configuration
  FKM.ndx                         generated automatically by group2ndx

Run in LAMMPS GUI:
  - put in.pull_smd_colvars.lammps and colvars.inp in the SMD folder
  - read_data pull_equil.data is relative to the same folder
  - run the input script as usual

Parameters:
  speed    1 A/ps     = 1e-3 A/fs   (10x faster than the first run)
  distance 150 A      (SiO2 COM z: 50.6 -> 200.6)
  Nrun     150000     (150 ps)
  Kpull    100 kcal/mol/A^2   moving harmonic restraint on SiO2 COM
  Kfkm     10 kcal/mol/A^2    static harmonic restraint on FKM COM z
  Kxy      100 kcal/mol/A^2    static harmonic restraints on SiO2 COM x/y
  anchor   FKM with z <= 30 A is frozen
  mobile   remaining FKM is NVT at 300 K
  SiO2     NVE, pulled by the Colvars spring

Important:
  The first finished run showed a lateral SiO2 COM drift in x/y
  (x COM moved about 27 A and crossed the periodic boundary).
  The current colvars.inp therefore also restrains x_sio2 and y_sio2.
  Re-run from pull_equil.data with the updated files.

Outputs:
  smd.colvars.traj   z_sio2, applied force fa_z_sio2, z_fkm,
                     center x0_z_sio2, accumulated work W_pull_sio2
  smd_monitor.txt    step, ps, SiO2 COM x/y/z, FKM COM z, FKM max z,
                     SiO2 min/max z
  smd_traj.dump      full trajectory every 10000 steps
  smd_end.data/restart/dump
