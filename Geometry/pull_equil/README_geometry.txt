Workflow:
1. Run LAMMPS:
   F:\LAMMPS 64-bit 4Jul2026 with GUI\bin\lmp.exe -in in.geometry.lammps
   This reads pull_equil/pull_equil.data and writes:
     geometry_atoms.dump
     geometry_lammps_summary.txt

2. Run Python:
   python compute_geometry.py
   Outputs:
     geometry_results.csv
     radial_density.png
     sio2_surface.png

R_surf is taken where the SiO2 atomic radial density drops to half of the
interior plateau. Contact area uses A0 = 4*pi*R_surf^2.
