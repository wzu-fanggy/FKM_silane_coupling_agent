#!/usr/bin/env python3
"""Compare bare vs KH560 conformational statistics by bond/angle class."""

import re
from pathlib import Path


BARE = Path(r"F:\Lammps-MS\FKM-SIO2-TEST\Conformational Statistics\conform_stats_summary_fixed.txt")
KH = Path(r"F:\Lammps-MS\FKM-SIO2-TEST\KH560\Conformational Statistics\conform_stats_summary_fixed.txt")
KH_DIH = Path(r"F:\Lammps-MS\FKM-SIO2-TEST\KH560\Conformational Statistics\dihedral_stats_results_NUMBA.txt")
BARE_DIH = Path(r"F:\Lammps-MS\FKM-SIO2-TEST\Conformational Statistics\dihedral_stats_results_NUMBA.txt")

ROW = re.compile(r"^\s*(\d+)\s+(contact|interface|bulk)\s+(\d+)\s+([\d.]+)\s+([\d.]+)")
LAYER_ORDER = ["contact", "interface", "bulk"]


def parse_summary(path):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    layer_atoms = {}
    bonds = []
    angles = []
    section = None
    for line in lines:
        s = line.strip()
        if s.startswith("Average Layer Atom Counts"):
            section = "atoms"
            continue
        if s.startswith("BOND LENGTH STATISTICS"):
            section = "bonds"
            continue
        if s.startswith("ANGLE STATISTICS"):
            section = "angles"
            continue
        if section == "atoms":
            m = re.match(r"^\s*(\w+):\s+([\d.]+)\s+\+/-", s)
            if m:
                layer_atoms[m.group(1)] = float(m.group(2))
            continue
        m = ROW.match(line)
        if not m:
            continue
        ident, layer, count, mean, std = m.groups()
        rec = (int(ident), layer, int(count), float(mean), float(std))
        if section == "bonds":
            bonds.append(rec)
        elif section == "angles":
            angles.append(rec)
    return layer_atoms, bonds, angles


def bond_class(mean):
    if mean < 1.20:
        return "C-H"
    if 1.35 <= mean <= 1.45:
        return "C-F / C=C"
    if mean > 1.45:
        return "C-C"
    return "other"


def angle_class(mean):
    if mean < 95:
        return "<95 deg"
    if 95 <= mean < 105:
        return "95-105"
    if 105 <= mean < 120:
        return "105-120"
    return ">120"


def class_stats(recs, classifier):
    out = {}
    for ident, layer, count, mean, std in recs:
        key = (classifier(mean), layer)
        if key not in out:
            out[key] = [0, 0.0, 0.0]
        out[key][0] += count
        out[key][1] += mean * count
        out[key][2] += std * count
    res = {}
    for (cls, layer), (n, sm, ss) in out.items():
        res[(cls, layer)] = (n, sm / n, ss / n)
    return res


def parse_dihedral_overall(path):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out = {}
    for line in lines:
        m = re.match(r"^\s*(contact|interface|bulk)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s*%", line)
        if m:
            layer, tr, ga, ci, total, pct = m.groups()
            out[layer] = (int(tr), int(ga), int(ci), float(pct))
    return out


def main():
    bare_atoms, bare_bonds, bare_angles = parse_summary(BARE)
    kh_atoms, kh_bonds, kh_angles = parse_summary(KH)

    print("=" * 60)
    print("Layer atom counts (mean over 501 frames)")
    print("-" * 60)
    print(f"{'layer':<12} {'bare':>10} {'KH560':>10}")
    for layer in LAYER_ORDER:
        print(f"{layer:<12} {bare_atoms.get(layer, 0):>10.0f} {kh_atoms.get(layer, 0):>10.0f}")

    print()
    print("Bond length by class (count-weighted mean +/- std, A)")
    print("-" * 60)
    b_bare = class_stats(bare_bonds, bond_class)
    b_kh = class_stats(kh_bonds, bond_class)
    classes = ["C-H", "C-F / C=C", "C-C"]
    for cls in classes:
        for layer in LAYER_ORDER:
            bb = b_bare.get((cls, layer))
            bk = b_kh.get((cls, layer))
            if bb and bk:
                print(f"{cls:<12} {layer:<10} bare {bb[0]:>6} {bb[1]:.4f}+/-{bb[2]:.4f}   "
                      f"KH560 {bk[0]:>6} {bk[1]:.4f}+/-{bk[2]:.4f}")

    print()
    print("Angle by class (count-weighted mean +/- std, deg)")
    print("-" * 60)
    a_bare = class_stats(bare_angles, angle_class)
    a_kh = class_stats(kh_angles, angle_class)
    for cls in ["<95 deg", "95-105", "105-120", ">120"]:
        for layer in LAYER_ORDER:
            ab = a_bare.get((cls, layer))
            ak = a_kh.get((cls, layer))
            if ab and ak:
                print(f"{cls:<10} {layer:<10} bare {ab[0]:>6} {ab[1]:.2f}+/-{ab[2]:.2f}   "
                      f"KH560 {ak[0]:>6} {ak[1]:.2f}+/-{ak[2]:.2f}")

    print()
    print("Dihedral trans/gauche/cis by layer")
    print("-" * 60)
    db = parse_dihedral_overall(BARE_DIH)
    dk = parse_dihedral_overall(KH_DIH)
    for layer in LAYER_ORDER:
        b = db.get(layer)
        k = dk.get(layer)
        if b and k:
            def pct(t):
                total = t[0] + t[1] + t[2]
                return 100.0 * t[0] / total, 100.0 * t[1] / total, 100.0 * t[2] / total
            bp = pct(b)
            kp = pct(k)
            print(f"{layer:<10} bare {bp[0]:>6.1f} {bp[1]:>6.1f} {bp[2]:>6.1f}  "
                  f"KH560 {kp[0]:>6.1f} {kp[1]:>6.1f} {kp[2]:>6.1f}  (T/G/C %)")


if __name__ == "__main__":
    main()