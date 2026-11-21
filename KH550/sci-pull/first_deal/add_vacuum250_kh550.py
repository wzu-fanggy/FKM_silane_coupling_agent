#!/usr/bin/env python3
"""Add a 250 A vacuum layer above the KH550 FKM/SiO2 slab.

Input : nvt_3000ps.data
Output: nvt_3000ps_vacuum250.data

The KH550 box is fully periodic and some bonds cross the z boundary.  We
recover continuous z coordinates from the nz image flags, shift the whole slab
so its lowest atom sits on the original zlo, then extend zhi by 250 A.  The
vacuum therefore exists only on the +z side, and the bottom clamp coordinate
used by the pull-out scripts stays valid.  X/Y coordinates and image flags are
left untouched because those directions remain periodic.

Each connected component (one grafted-SiO2 cluster + 120 FKM chains) is
re-imaged as a whole so every chain sits next to the cluster instead of being
left at the far top/bottom copy of the original periodic box.  This mirrors the
vacuum_periodic_image_fix.md fix used for the KH560 system.
"""

from collections import Counter, defaultdict
from pathlib import Path


WORK = Path(__file__).resolve().parent
SRC = WORK / "nvt_3000ps.data"
DST = WORK / "nvt_3000ps_vacuum250.data"

VACUUM = 250.0
BOND_TOL = 2.5


def find(lines, exact=None, startswith=None):
    for i, line in enumerate(lines):
        s = line.strip()
        if exact is not None and s == exact:
            return i
        if startswith is not None and s.startswith(startswith):
            return i
    raise RuntimeError(f"section not found: {exact or startswith}")


def parse_box(lines):
    box = {}
    for line in lines:
        s = line.strip()
        for axis in ("xlo xhi", "ylo yhi", "zlo zhi"):
            if s.endswith(axis):
                vals = s.split()[:2]
                box[axis] = (float(vals[0]), float(vals[1]))
    return box


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


def parse_bonds(lines, bi, ai):
    bonds = []
    for line in lines[bi + 1 : ai]:
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


def main():
    lines = SRC.read_text(encoding="ascii").splitlines()
    box = parse_box(lines)
    si = find(lines, exact="Atoms # full")
    vi = find(lines, exact="Velocities")
    bi = find(lines, exact="Bonds")
    ai = find(lines, exact="Angles")
    atoms = parse_atoms(lines, si, vi)
    bonds = parse_bonds(lines, bi, ai)
    comps = components(bonds)
    cluster = max(comps, key=len)
    fkm_chains = [c for c in comps if c is not cluster]

    xlo, xhi = box["xlo xhi"]
    ylo, yhi = box["ylo yhi"]
    zlo, zhi = box["zlo zhi"]
    hx = xhi - xlo
    hy = yhi - ylo
    hz = zhi - zlo

    z_u = {aid: a["z"] + a["nz"] * hz for aid, a in atoms.items()}
    cluster_zc = sum(z_u[a] for a in cluster) / len(cluster)

    # Choose one periodic image per connected molecule so every FKM chain is
    # placed next to the grafted SiO2 cluster instead of being left at the far
    # top/bottom copy of the original periodic box.
    shifted_u = {}
    n_changed = 0
    for comp in comps:
        comp_zc = sum(z_u[a] for a in comp) / len(comp)
        k = int(round((cluster_zc - comp_zc) / hz))
        if k != 0:
            n_changed += 1
        for aid in comp:
            shifted_u[aid] = z_u[aid] + k * hz

    # Lift the slab 5 A above zlo so the lowest atom leaves a visible gap to the
    # bottom boundary (avoids atoms sitting on / poking through the box floor).
    shift = zlo - min(shifted_u.values()) + 5.0
    new_z = {aid: zu + shift for aid, zu in shifted_u.items()}
    new_zhi = max(new_z.values()) + VACUUM

    cl_zlo = min(new_z[a] for a in cluster)
    cl_zhi = max(new_z[a] for a in cluster)
    separated = []
    for comp in fkm_chains:
        mn = min(new_z[a] for a in comp)
        mx = max(new_z[a] for a in comp)
        if mx < cl_zlo - 2.0 or mn > cl_zhi + 2.0:
            separated.append((mn, mx))

    print(f"components        : {len(comps)} (cluster {len(cluster)} + FKM {len(fkm_chains)})")
    print(f"periodic images adjusted: {n_changed}")
    print(f"original z box   : {zlo:.4f}..{zhi:.4f}")
    print(f"unwrapped z range: {min(z_u.values()):.4f}..{max(z_u.values()):.4f}")
    print(f"z shift          : {shift:.4f}")
    print(f"new z box        : {zlo:.4f}..{new_zhi:.4f}")
    print(f"material z range : {zlo:.4f}..{max(new_z.values()):.4f}")
    print(f"cluster z range  : {cl_zlo:.4f}..{cl_zhi:.4f}")
    print(f"FKM chains separated from cluster: {len(separated)}")
    print(f"top vacuum       : {VACUUM:.0f} A")

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
                q[6] = f"{max(zlo, new_z[aid]):.6f}"
                q[9] = "0"
                out.append("  " + " ".join(q))
                continue
        if s.endswith("zlo zhi"):
            out.append(f"{zlo:.15f} {new_zhi:.15f} zlo zhi")
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
    vxlo, vxhi = vbox["xlo xhi"]
    vylo, vyhi = vbox["ylo yhi"]
    vzlo, vzhi = vbox["zlo zhi"]
    vhx = vxhi - vxlo
    vhy = vyhi - vylo
    vhz = vzhi - vzlo

    outside = 0
    for a in vatoms.values():
        if not (
            vxlo <= a["x"] <= vxhi
            and vylo <= a["y"] <= vyhi
            and vzlo <= a["z"] <= vzhi
        ):
            outside += 1

    bad = 0
    max_comp = 0.0
    for i, j in vbonds:
        ai = vatoms[i]
        aj = vatoms[j]
        dx = abs((ai["x"] + ai["nx"] * vhx) - (aj["x"] + aj["nx"] * vhx))
        dy = abs((ai["y"] + ai["ny"] * vhy) - (aj["y"] + aj["ny"] * vhy))
        dz = abs(ai["z"] - aj["z"])
        m = max(dx, dy, dz)
        max_comp = max(max_comp, m)
        if m > BOND_TOL:
            bad += 1

    nz_nonzero = sum(1 for a in vatoms.values() if a["nz"] != 0)

    print(f"atoms outside box: {outside}")
    print(f"max bond comp    : {max_comp:.4f} A")
    print(f"bad bonds        : {bad}")
    print(f"atoms with nz!=0 : {nz_nonzero}")
    print(f"wrote            : {DST}")


if __name__ == "__main__":
    main()