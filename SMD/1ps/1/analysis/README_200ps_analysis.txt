1 A/ps, 200 ps pull-out analysis
=================================

Data source:
  ALL-output.txt contains the full 0-200 ps thermo log.
  smd.colvars.traj / smd_monitor.txt / smd_traj.dump were still 0 bytes,
  so the force curve is reconstructed from the thermo log:
    F = Kpull * (center_z - z_sio2), Kpull = 100 kcal/mol/A^2
  The trapezoid-integrated work matches the Colvars final work within 0.15%.

Group/group interaction energy (pair + kspace, LAMMPS compute group/group):
  initial  -1260.85 kcal/mol
  final     -573.78 kcal/mol

Final frame contact:
  min SIO2-FKM distance = 1.59 A
  954 FKM atoms within 3.5 A
  5747 FKM atoms within 5.0 A

Per-frame interaction energy vs distance still requires a trajectory
with atom positions; smd_traj.dump was not written by the GUI run.
