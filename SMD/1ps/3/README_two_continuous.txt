Two continuous SMD runs, one driver
===================================

Mode corridor:
  20 A free corridor around SiO2, outer FKM anchored.
  Approximates W_pull-out.

Mode total:
  Full embedded FKM, only bottom anchored.
  Approximates W_total.

Default run:
  1 A/ps, 200 ps, 200000 steps, 200 A total displacement.

Run:
  python run_two_continuous.py

Smoke test:
  python run_two_continuous.py --nrun 100

Outputs:
  corridor/    corridor pull-out files
  total/       full-embedded pull-out files
  decomposition_summary.csv

The script calls LAMMPS with:
  -sf omp -pk omp 16
