1 A/ps, 150 ps pull-out analysis
=================================

Files:
  smd_1ps_zmonitor.png
  smd_1ps_xy_stability.png
  smd_1ps_force_displacement.png
  smd_1ps_work_displacement.png
  smd_1ps_analysis_summary.csv

Notes:
  smd.colvars.traj and smd_monitor.txt were 0 bytes after the GUI run,
  so the force curve is reconstructed from the thermo log:
    F = Kpull * (center_z - z_sio2), Kpull = 100 kcal/mol/A^2
  This is the same expression Colvars uses, so it should match closely.

  The final work is read directly from smd.colvars.state:
    6830.81 kcal/mol

  Final frame group/group energy (pair + kspace):
    initial  -1260.85 kcal/mol
    final     -731.46 kcal/mol
  The final frame still has FKM atoms in direct contact with SiO2
  (1214 within 3.5 A), so this is not a clean pull-out.
