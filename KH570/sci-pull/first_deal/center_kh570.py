#!/usr/bin/env python3
"""Translate all atoms so the grafted SiO2(+KH570) COM sits at box center.

Input : nvt_3000ps.data
Output: nvt_3000ps_centered.data

The box is fully periodic, so the whole-system translation is a pure origin
shift: every atom is moved by the same unwrapped vector, wrapped back into the
box with its image flags updated, and no bond is broken.
"""

import math
from pathlib import Path


WORK = Path(__file__).resolve().parent
SRC = WORK / "nvt_3000ps.data"
DST = WORK / "nvt_3000ps_centered.data"

SIO2_TYPES = {5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17}
BOND_TOL = 2.5


def find(lines, exact=None, startswith=None):
    for i, line in enumerate(lines):
        s = line.strip()
        if exact is not None and s == exact:
            return i
        if startswith is not None and s.startswith(startswith):
            return i
    raise RuntimeError(f"section not found: {exact or startswith}")


def parse_masses(lines, mi, pi):
    mass = {}
    for line in lines[mi + 1 : pi]:
        q = line.split()
        if len(q) >= 2 and q[0].isdigit():
            mass[int(q[0])] = float(q[1])
    return mass


def parse_atoms(lines, si, vi):
    atoms = {}
    for line in lines[si + 1 : vi]:
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


def parse_box(lines):
    box = {}
    for line in lines:
        s = line.strip()
        for axis in ("xlo xhi", "ylo yhi", "zlo zhi"):
            if s.endswith(axis):
                vals = s.split()[:2]
                box[axis] = (float(vals[0]), float(vals[1]))
    return box


def parse_bonds(lines, bi, ai):
    bonds = []
    for line in lines[bi + 1 : ai]:
        q = line.split()
        if len(q) >= 4 and q[0].isdigit():
            bonds.append((int(q[2]), int(q[3])))
    return bonds


def unwrapped(a, lengths):
    return (
        a["x"] + a["nx"] * lengths[0],
        a["y"] + a["ny"] * lengths[1],
        a["z"] + a["nz"] * lengths[2],
    )


def wrap(value, lo, length):
    n = math.floor((value - lo) / length)
    return value - n * length, n


def main():
    lines = SRC.read_text(encoding="ascii").splitlines()
    mi = find(lines, exact="Masses")
    pi = find(lines, startswith="Pair Coeffs")
    si = find(lines, exact="Atoms # full")
    vi = find(lines, exact="Velocities")
    bi = find(lines, exact="Bonds")
    ai = find(lines, exact="Angles")

    mass = parse_masses(lines, mi, pi)
    atoms = parse_atoms(lines, si, vi)
    box = parse_box(lines)
    bonds = parse_bonds(lines, bi, ai)

    lengths = tuple(
        box[axis][1] - box[axis][0] for axis in ("xlo xhi", "ylo yhi", "zlo zhi")
    )
    center = tuple(
        (box[axis][0] + box[axis][1]) / 2
        for axis in ("xlo xhi", "ylo yhi", "zlo zhi")
    )

    sio2 = {aid: a for aid, a in atoms.items() if a["type"] in SIO2_TYPES}
    com = [0.0, 0.0, 0.0]
    total_mass = 0.0
    for aid, a in sio2.items():
        m = mass[a["type"]]
        u = unwrapped(a, lengths)
        total_mass += m
        com[0] += m * u[0]
        com[1] += m * u[1]
        com[2] += m * u[2]
    com = [c / total_mass for c in com]
    shift = [center[i] - com[i] for i in range(3)]

    print(f"box center      : {[round(c, 4) for c in center]}")
    print(f"cluster COM before: {[round(c, 4) for c in com]}")
    print(f"shift           : {[round(s, 4) for s in shift]}")

    out = []
    section = None
    for line in lines:
        s = line.strip()
        if s == "Atoms # full":
            section = "atoms"
            out.append(line)
            continue
        if s == "Velocities":
            section = None
            out.append(line)
            continue
        if section == "atoms" and s and s[0].isdigit():
            q = line.split()
            if len(q) >= 10:
                aid = int(q[0])
                a = atoms[aid]
                u = unwrapped(a, lengths)
                x, nx = wrap(u[0] + shift[0], box["xlo xhi"][0], lengths[0])
                y, ny = wrap(u[1] + shift[1], box["ylo yhi"][0], lengths[1])
                z, nz = wrap(u[2] + shift[2], box["zlo zhi"][0], lengths[2])
                q[4] = f"{x:.6f}"
                q[5] = f"{y:.6f}"
                q[6] = f"{z:.6f}"
                q[7] = str(nx)
                q[8] = str(ny)
                q[9] = str(nz)
                out.append("  " + " ".join(q))
                continue
        out.append(line)

    DST.write_text("\n".join(out) + "\n", encoding="ascii")

    # Verification on the written file.
    vlines = DST.read_text(encoding="ascii").splitlines()
    vbox = parse_box(vlines)
    vatoms = parse_atoms(
        vlines,
        find(vlines, exact="Atoms # full"),
        find(vlines, exact="Velocities"),
    )
    vbonds = parse_bonds(
        vlines,
        find(vlines, exact="Bonds"),
        find(vlines, exact="Angles"),
    )
    vlengths = tuple(
        vbox[axis][1] - vbox[axis][0] for axis in ("xlo xhi", "ylo yhi", "zlo zhi")
    )

    outside = 0
    for a in vatoms.values():
        if not (
            vbox["xlo xhi"][0] <= a["x"] <= vbox["xlo xhi"][1]
            and vbox["ylo yhi"][0] <= a["y"] <= vbox["ylo yhi"][1]
            and vbox["zlo zhi"][0] <= a["z"] <= vbox["zlo zhi"][1]
        ):
            outside += 1

    max_bond = 0.0
    bad_bonds = 0
    for i, j in vbonds:
        ua = unwrapped(vatoms[i], vlengths)
        ub = unwrapped(vatoms[j], vlengths)
        d = max(abs(ua[k] - ub[k]) for k in range(3))
        max_bond = max(max_bond, d)
        if d > BOND_TOL:
            bad_bonds += 1

    com_after = [0.0, 0.0, 0.0]
    total_mass_after = 0.0
    for aid in sio2:
        a = vatoms[aid]
        m = mass[a["type"]]
        u = unwrapped(a, vlengths)
        total_mass_after += m
        com_after[0] += m * u[0]
        com_after[1] += m * u[1]
        com_after[2] += m * u[2]
    com_after = [c / total_mass_after for c in com_after]

    print(f"cluster COM after: {[round(c, 6) for c in com_after]}")
    print(f"atoms outside   : {outside}")
    print(f"max bond comp   : {max_bond:.4f} A")
    print(f"bad bonds       : {bad_bonds}")
    print(f"wrote           : {DST}")


if __name__ == "__main__":
    main()
