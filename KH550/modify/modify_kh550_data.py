#!/usr/bin/env python3
"""Regenerate relabeled + centered LAMMPS data for the KH550 workflow.

Pipeline:
  1. Parse atoms/bonds, build connected components, and identify the FKM
     chains and the KH550-grafted SiO2 cluster.
  2. Relabel atom types shared between FKM and the grafted cluster into new
     independent types, duplicating their Masses and Pair Coeffs entries.
  3. Translate all atoms (periodic wrapping) so the grafted cluster COM is
     exactly at the box center.

Run from this directory:
    python modify_kh550_data.py
"""

import math
import re
from collections import Counter, defaultdict
from pathlib import Path


WORK = Path(__file__).resolve().parent
SRC = WORK / "anneal_final.data"
REL = WORK / "anneal_final_relabeled.data"
CENTERED = WORK / "anneal_final_relabeled_centered.data"

NEXT_SECTION = ("Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs",
                "Improper Coeffs", "Velocities", "Bonds", "Angles",
                "Dihedrals", "Impropers")


def find(lines, exact=None, startswith=None):
    for i, line in enumerate(lines):
        s = line.strip()
        if exact is not None and s == exact:
            return i
        if startswith is not None and s.startswith(startswith):
            return i
    raise RuntimeError(f"section not found: {exact or startswith}")


def section_entries(lines, start):
    """Return lines of a coefficient block, stopping at blank or next header."""
    out = []
    i = start + 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    while i < len(lines) and lines[i].strip() and not lines[i].strip()[0].isalpha():
        out.append(lines[i].strip())
        i += 1
    return out


def parse_atoms(lines, si, vi):
    atoms = {}
    for line in lines[si + 1:vi]:
        q = line.split()
        if len(q) >= 10 and q[0].isdigit():
            atoms[int(q[0])] = {
                "mol": int(q[1]),
                "type": int(q[2]),
                "charge": float(q[3]),
                "x": float(q[4]),
                "y": float(q[5]),
                "z": float(q[6]),
                "nx": int(q[7]),
                "ny": int(q[8]),
                "nz": int(q[9]),
            }
    return atoms


def parse_bonds(lines, bi, ai):
    bonds = []
    for line in lines[bi + 1:ai]:
        q = line.split()
        if len(q) >= 4 and q[0].isdigit():
            bonds.append((int(q[2]), int(q[3])))
    return bonds


def components(bonds):
    adj = defaultdict(list)
    for a, b in bonds:
        adj[a].append(b)
        adj[b].append(a)
    seen = set()
    comps = []
    for a in adj:
        if a in seen:
            continue
        stack, comp = [a], []
        seen.add(a)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(comp)
    return comps


def parse_box(lines):
    box = {}
    for line in lines:
        s = line.strip()
        for axis in ("xlo xhi", "ylo yhi", "zlo zhi"):
            if s.endswith(axis):
                vals = s.split()[:2]
                box[axis] = (float(vals[0]), float(vals[1]))
    return box


def parse_masses(lines, mi):
    masses = {}
    for entry in section_entries(lines, mi):
        q = entry.split()
        masses[int(q[0])] = float(q[1])
    return masses


def parse_pair_coeffs(lines, pi):
    pairs = {}
    for entry in section_entries(lines, pi):
        q = entry.split()
        pairs[int(q[0])] = " ".join(q[1:])
    return pairs


def unwrapped(atom, box):
    lx = box["xlo xhi"][1] - box["xlo xhi"][0]
    ly = box["ylo yhi"][1] - box["ylo yhi"][0]
    lz = box["zlo zhi"][1] - box["zlo zhi"][0]
    return (atom["x"] + atom["nx"] * lx,
            atom["y"] + atom["ny"] * ly,
            atom["z"] + atom["nz"] * lz)


def com_of(atoms, ids, masses, box):
    com = [0.0, 0.0, 0.0]
    total_mass = 0.0
    for aid in ids:
        a = atoms[aid]
        m = masses.get(a["type"], 12.0)
        u = unwrapped(a, box)
        com[0] += m * u[0]
        com[1] += m * u[1]
        com[2] += m * u[2]
        total_mass += m
    return [c / total_mass for c in com]


def wrap(value, lo, length):
    n = math.floor((value - lo) / length)
    return value - n * length, n


def rewrite(lines, type_map, id_to_new, new_masses, new_pairs, new_type_count,
            label_lines=None):
    """Apply relabeling to a data file line list."""
    out = []
    section = None
    for line in lines:
        s = line.strip()
        if s == "Masses":
            section = "masses"
            out.append(line)
            continue
        if s.startswith("Pair Coeffs"):
            section = "pair"
            out.append(line)
            continue
        if s == "Atoms # full":
            section = "atoms"
            out.append(line)
            continue
        if s.startswith(NEXT_SECTION):
            section = None
            out.append(line)
            continue
        if section == "atoms" and s and s[0].isdigit():
            q = line.split()
            if len(q) >= 10:
                aid = int(q[0])
                if aid in id_to_new:
                    q[2] = str(id_to_new[aid])
                    line = "  " + " ".join(q)
            out.append(line)
            continue
        if re.match(r"^\s*\d+\s+atom types\s*$", s):
            out.append(f"{new_type_count} atom types")
            continue
        if label_lines is not None and s == "Atom Type Labels":
            out.append(line)
            continue
        out.append(line)

    def insert_before_blank(header, extra, startswith=False):
        if startswith:
            hi = next((i for i, l in enumerate(out) if l.strip().startswith(header)),
                      None)
        else:
            hi = next((i for i, l in enumerate(out) if l.strip() == header), None)
        if hi is None:
            return
        j = hi + 1
        while j < len(out) and not out[j].strip():
            j += 1
        while j < len(out) and out[j].strip() and out[j].strip()[0].isdigit():
            j += 1
        out[j:j] = extra

    mass_extra = [f"{new} {new_masses[old]}" for old, new in type_map.items()]
    pair_extra = [f"{new} {new_pairs[old]}" for old, new in type_map.items()]
    insert_before_blank("Masses", mass_extra)
    insert_before_blank("Pair Coeffs", pair_extra, startswith=True)
    return out


def verify(path, big_ids, masses):
    lines = path.read_text(encoding="ascii").splitlines()
    box = parse_box(lines)
    si = find(lines, "Atoms # full")
    vi = find(lines, "Velocities")
    bi = find(lines, "Bonds")
    ai = find(lines, "Angles")
    atoms = parse_atoms(lines, si, vi)
    bonds = parse_bonds(lines, bi, ai)
    outside = 0
    lx = box["xlo xhi"][1] - box["xlo xhi"][0]
    ly = box["ylo yhi"][1] - box["ylo yhi"][0]
    lz = box["zlo zhi"][1] - box["zlo zhi"][0]
    for aid, a in atoms.items():
        if not (box["xlo xhi"][0] <= a["x"] <= box["xlo xhi"][1] and
                box["ylo yhi"][0] <= a["y"] <= box["ylo yhi"][1] and
                box["zlo zhi"][0] <= a["z"] <= box["zlo zhi"][1]):
            outside += 1
    max_comp = 0.0
    bad = 0
    for a, b in bonds:
        ua = unwrapped(atoms[a], box)
        ub = unwrapped(atoms[b], box)
        dx = abs(ua[0] - ub[0])
        dy = abs(ua[1] - ub[1])
        dz = abs(ua[2] - ub[2])
        max_comp = max(max_comp, dx, dy, dz)
        if dx > lx / 2 or dy > ly / 2 or dz > lz:
            bad += 1
    com = com_of(atoms, big_ids, masses, box)
    center = [(box["xlo xhi"][0] + box["xlo xhi"][1]) / 2,
              (box["ylo yhi"][0] + box["ylo yhi"][1]) / 2,
              (box["zlo zhi"][0] + box["zlo zhi"][1]) / 2]
    return {
        "atoms": len(atoms),
        "outside": outside,
        "max_bond_comp": max_comp,
        "bad_bonds": bad,
        "com": com,
        "center": center,
        "offset": [com[i] - center[i] for i in range(3)],
    }


def main():
    lines = SRC.read_text(encoding="ascii").splitlines()
    box = parse_box(lines)
    mi = find(lines, "Masses")
    pi = find(lines, startswith="Pair Coeffs")
    si = find(lines, "Atoms # full")
    vi = find(lines, "Velocities")
    bi = find(lines, "Bonds")
    ai = find(lines, "Angles")

    masses = parse_masses(lines, mi)
    pair_coeffs = parse_pair_coeffs(lines, pi)
    atoms = parse_atoms(lines, si, vi)
    bonds = parse_bonds(lines, bi, ai)
    comps = components(bonds)

    comp_sizes = Counter(len(c) for c in comps)
    print("component size distribution:", dict(sorted(comp_sizes.items())))
    big_comp = max(comps, key=len)
    big_ids = set(big_comp)
    nonbig_sizes = [len(c) for c in comps if c is not big_comp]
    fkm_size = Counter(nonbig_sizes).most_common(1)[0][0]
    fkm_ids = set()
    for c in comps:
        if c is not big_comp and len(c) == fkm_size:
            fkm_ids.update(c)
    print(f"grafted SiO2+KH550 component: {len(big_ids)} atoms")
    print(f"FKM chains: {len(nonbig_sizes) // 1} x {fkm_size} atoms")

    big_types = {atoms[i]["type"] for i in big_ids}
    fkm_types = {atoms[i]["type"] for i in fkm_ids}
    shared = sorted(big_types & fkm_types)
    print("shared types (FKM and grafted cluster):", shared)

    max_type = max(atoms[i]["type"] for i in atoms)
    type_map = {old: max_type + k + 1 for k, old in enumerate(shared)}
    print("type relabel:", {f"{k} -> {v}" for k, v in type_map.items()})

    id_to_new = {}
    for aid in big_ids:
        old = atoms[aid]["type"]
        if old in type_map:
            id_to_new[aid] = type_map[old]

    full_masses = dict(masses)
    for old, new in type_map.items():
        full_masses[new] = masses[old]

    relabeled = rewrite(lines, type_map, id_to_new, masses, pair_coeffs,
                        max_type + len(shared))
    REL.write_text("\n".join(relabeled) + "\n", encoding="ascii")
    print("wrote", REL.name)

    # Centering pass: shift all atoms so the grafted cluster COM == box center.
    centered_lines = []
    section = None
    com = com_of(atoms, big_ids, full_masses, box)
    center = [(box["xlo xhi"][0] + box["xlo xhi"][1]) / 2,
              (box["ylo yhi"][0] + box["ylo yhi"][1]) / 2,
              (box["zlo zhi"][0] + box["zlo zhi"][1]) / 2]
    shift = [center[i] - com[i] for i in range(3)]
    lx = box["xlo xhi"][1] - box["xlo xhi"][0]
    ly = box["ylo yhi"][1] - box["ylo yhi"][0]
    lz = box["zlo zhi"][1] - box["zlo zhi"][0]
    print("COM shift (atoms only):", [round(v, 4) for v in shift])

    for line in relabeled:
        s = line.strip()
        if s == "Atoms # full":
            section = "atoms"
            centered_lines.append(line)
            continue
        if s == "Velocities":
            section = None
            centered_lines.append(line)
            continue
        if section == "atoms" and s and s[0].isdigit():
            q = line.split()
            if len(q) >= 10:
                aid = int(q[0])
                a = atoms[aid]
                u = unwrapped(a, box)
                x, nx = wrap(u[0] + shift[0], box["xlo xhi"][0], lx)
                y, ny = wrap(u[1] + shift[1], box["ylo yhi"][0], ly)
                z, nz = wrap(u[2] + shift[2], box["zlo zhi"][0], lz)
                q[4] = f"{x:.6f}"
                q[5] = f"{y:.6f}"
                q[6] = f"{z:.6f}"
                q[7] = str(nx)
                q[8] = str(ny)
                q[9] = str(nz)
                centered_lines.append("  " + " ".join(q))
                continue
        centered_lines.append(line)
    CENTERED.write_text("\n".join(centered_lines) + "\n", encoding="ascii")
    print("wrote", CENTERED.name)

    print("\nVerification (centered file):")
    rep = verify(CENTERED, big_ids, full_masses)
    for k, v in rep.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
