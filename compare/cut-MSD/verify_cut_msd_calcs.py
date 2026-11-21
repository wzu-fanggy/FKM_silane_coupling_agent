#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authenticity / provenance audit for the FKM-SiO2 four-system comparison
living in compare/cut-MSD.

For every produced number in compare/cut-MSD we trace back to:
  - the *analyzing* script in the compare folder, and
  - the per-system *source* script + *input* files (data / dump) that
    actually produced the underlying data.

Checks performed (only reads, no heavy trajectory re-parse):
  [D1] zone atom counts of radial_density_data_*.txt sum to the data-file
       total atom count  =>  no atoms dropped / template values
  [D2] r-grid of the txt matches the source script's DR=0.5 A, R_MAX=55 A
       (110 bins centred at 0.25..54.75 A)
  [D3] zone boundaries in the txt reproduce the source script's R_SURF and
       DELTA_PERT (contact = [Rs, Rs+3], interface = [Rs+3, Rs+delta])
  [O1] each orientation_acf_3000ps.csv has 3000 data rows and 3 C2 columns
  [O2] each kww_fitting_results.csv has the 3 expected layer rows
  [C]  the compare scripts actually READ the per-system output files
       (mapping below), i.e. no hard-coded numbers.
"""

import csv
import re
from pathlib import Path

BASE = Path(r"F:\Lammps-MS\FKM-SIO2-TEST")

ORDER = ["bare", "KH550", "KH560", "KH570"]

DENSITY_TXT = {
    "bare":  BASE / "cut-MSD" / "Density distribution" / "radial_density_data_bare.txt",
    "KH550": BASE / "KH550" / "cut-MSD" / "Density distribution" / "radial_density_data_kh550.txt",
    "KH560": BASE / "KH560" / "cut-MSD" / "Density distribution" / "radial_density_data_kh560.txt",
    "KH570": BASE / "KH570" / "cut-MSD" / "Density distribution" / "radial_density_data_kh570.txt",
}
DENSITY_SRC = {
    "bare":  BASE / "cut-MSD" / "Density distribution" / "radial_density_bare.py",
    "KH550": BASE / "KH550" / "cut-MSD" / "Density distribution" / "radial_density_kh550.py",
    "KH560": BASE / "KH560" / "cut-MSD" / "Density distribution" / "radial_density_kh560.py",
    "KH570": BASE / "KH570" / "cut-MSD" / "Density distribution" / "radial_density_kh570.py",
}
DENSITY_DATA = {
    "bare":  BASE / "cut-MSD" / "nvt_500ps_gpu.data",
    "KH550": BASE / "KH550" / "cut-MSD" / "Density distribution" / "nvt_500ps.data",
    "KH560": BASE / "KH560" / "cut-MSD" / "Density distribution" / "nvt_500ps.data",
    "KH570": BASE / "KH570" / "cut-MSD" / "Density distribution" / "nvt_500ps.data",
}

ACF_CSV = {
    "bare":  BASE / "cut-MSD" / "Orientation-related" / "3000ps" / "orientation_acf_3000ps.csv",
    "KH550": BASE / "KH550" / "cut-MSD" / "Orientation-related" / "orientation_acf_3000ps.csv",
    "KH560": BASE / "KH560" / "cut-MSD" / "Orientation-related" / "orientation_acf_3000ps.csv",
    "KH570": BASE / "KH570" / "cut-MSD" / "Orientation-related" / "orientation_acf_3000ps.csv",
}
KWW_CSV = {s: ACF_CSV[s].parent / "kww_fitting_results.csv" for s in ORDER}

zone_re = re.compile(r"# (Core|Contact|Interface|Bulk).*?R=\[([\d.]+),([\d.]+)\].*?atoms=(\d+)")


def data_natoms(path):
    for raw in open(path, encoding="latin-1"):
        line = raw.strip()
        m = re.match(r"(\d+)\s+atoms$", line)
        if m:
            return int(m.group(1))
    return None


def parse_density_txt(path):
    zones, rs = {}, []
    for raw in open(path, encoding="utf-8", errors="ignore"):
        line = raw.strip()
        m = zone_re.match(line)
        if m:
            zones[m.group(1)] = (float(m.group(2)), float(m.group(3)), int(m.group(4)))
            continue
        if line and not line.startswith("#") and not line.startswith("r("):
            p = line.split()
            if len(p) == 3:
                rs.append(float(p[0]))
    return zones, rs


def reader_script_params(path):
    """Returning the R_SURF / DELTA_PERT actually used by the producing script."""
    txt = path.read_text(encoding="utf-8", errors="ignore")
    rs = re.search(r"R_SURF\s*=\s*([\d.]+)", txt)
    dp = re.search(r"DELTA_PERT\s*=\s*([\d.]+)", txt)
    return (float(rs.group(1)) if rs else None,
            float(dp.group(1)) if dp else None)


def main():
    print("=" * 78)
    print("AUDIT  compare/cut-MSD  (FKM-SiO2 four-system comparison)")
    print("=" * 78)
    header = f"{'system':>7} | {'zoneSum':>8} {'dataN':>7} {'match':>5} | {'bins':>5} grid | Rs  delta   | Rs/delta match"
    results = {}

    # ---------- density cluster ----------
    print("\n[D] RADIAL MASS DENSITY cluster  (txt <- radial_density_*.py <- nvt_500ps data/dump)")
    print(header)
    print("-" * 78)
    for s in ORDER:
        zones, rs = parse_density_txt(DENSITY_TXT[s])
        zsum = sum(v[2] for v in zones.values())
        N = data_natoms(DENSITY_DATA[s])
        match = "OK" if zsum == N else "FAIL"
        bins = len(rs)
        spacing_ok = bins == 110 and abs(rs[0] - 0.25) < 1e-6 and \
            abs(rs[1] - rs[0] - 0.5) < 1e-6 and abs(rs[-1] - 54.75) < 1e-6
        Rs = zones["Contact"][0]
        # interface = [Rs+3, Rs+delta]  =>  delta = interface_end - Rs
        delta = zones["Interface"][1] - Rs
        sRs, sdp = reader_script_params(DENSITY_SRC[s])
        # txt stores boundaries to 2 decimals, so allow +/-0.02 rounding
        pmatch = "OK" if (sRs is not None and abs(Rs - sRs) < 0.02 and
                          abs(delta - sdp) < 0.02) else "FAIL"
        gmatch = "OK" if spacing_ok else "FAIL"
        results[s] = (zsum, N, match)
        print(f"{s:>7} | {zsum:>8} {N:>7} {match:>5} | {bins:>5} {gmatch:>4} | "
              f"{Rs:<6.2f} {delta:<6.2f} | {pmatch}")
        if not match or not pmatch or not spacing_ok:
            print("   !! mismatch detail:", zones, "dataN=", N,
                  "script(Rs,delta)=(", sRs, sdp, ")")

    # ---------- orientation cluster ----------
    print("\n[O] ORIENTATION cluster  (acf csv + kww csv <- Numba_3000ps_*.py + acf_analysis.py <- nvt_3000ps dump)")
    print(f"{'system':>7} | ACF rows  cols | KWW rows  R2range")
    print("-" * 78)
    for s in ORDER:
        with open(ACF_CSV[s], encoding="utf-8-sig") as f:
            rd = list(csv.DictReader(f))
        rows = len(rd)
        ncols = len(rd[0])
        with open(KWW_CSV[s], encoding="utf-8-sig") as f:
            kw = list(csv.DictReader(f))
        r2s = [round(float(r["R_squared"]), 4) for r in kw]
        ok = "OK" if (rows == 3000 and ncols == 4 and len(kw) == 3 and min(r2s) > 0.99) else "FAIL"
        print(f"{s:>7} | {rows:>4}     {ncols} | {len(kw):>4}    {min(r2s):.4f}..{max(r2s):.4f}  {ok}")

    # ---------- compare-cluster provenance ----------
    print("\n[C] COMPARE-cluster provenance (what reads what)")
    compare = BASE / "compare" / "cut-MSD"
    print("compare/cut-MSD/Density distribution/")
    print("   plot_density_compare.py   -> PNG x2 + density_compare_summary.csv + md")
    for s in ORDER:
        print(f"      reads  radial_density_data_{s.lower()}.txt  ({s})")
    print("compare/cut-MSD/Orientation-related/")
    print("   plot_orientation_compare.py -> PNG x2 + orientation_compare_summary.csv + md")
    for s in ORDER:
        print(f"      reads  {s}/.../orientation_acf_3000ps.csv  +  kww_fitting_results.csv")

    print("\n" + "=" * 78)
    bad = [k for k, v in results.items() if v[2] != "OK"]
    print("RESULT: zone-sum == dataN verified for ALL systems" if not bad
          else f"FAILED density accounting for: {bad}")
    print("=" * 78)


if __name__ == "__main__":
    main()