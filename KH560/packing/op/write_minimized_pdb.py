#!/usr/bin/env python3
"""Write the final minimized PDB preserving the original 560use.pdb order.

The final PDB keeps the atom order and CONECT records of the original MS
PDB, but replaces the atom-name column with mapped PCFF labels (si/o/h*,
not s_m/osh/osi/hos) and coordinates with the minimized LAMMPS data.
"""

from pathlib import Path


WORK = Path(__file__).resolve().parent
ORIGINAL = WORK / "560use.pdb"
DATA = WORK / "minimized_560use.data"
DST = WORK / "minimized_560use.pdb"


def find(lines, exact):
    for i, line in enumerate(lines):
        if line.strip() == exact:
            return i
    raise RuntimeError(f"section not found: {exact}")


def parse_pdb(path):
    lines = path.read_text(encoding="ascii", errors="ignore").splitlines()
    atoms = []
    conect = []
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atoms.append(
                {
                    "serial": int(line[6:11]),
                    "element": line[76:78].strip(),
                }
            )
        elif line.startswith("CONECT"):
            conect.append(line)
    return atoms, conect


def parse_data(path):
    lines = path.read_text(encoding="ascii", errors="ignore").splitlines()
    ti = find(lines, "Atom Type Labels")
    nti = find(lines, "Bond Type Labels")
    si = find(lines, "Atoms # full")
    vi = find(lines, "Velocities")

    labels = {}
    for line in lines[ti + 1 : nti]:
        q = line.split()
        if len(q) >= 2 and q[0].isdigit():
            labels[int(q[0])] = q[1]

    coords = {}
    for line in lines[si + 1 : vi]:
        q = line.split()
        if len(q) >= 10 and q[0].isdigit():
            coords[int(q[0])] = (
                int(q[2]),
                float(q[4]),
                float(q[5]),
                float(q[6]),
            )
    return labels, coords


def main():
    orig_atoms, conect = parse_pdb(ORIGINAL)
    labels, coords = parse_data(DATA)

    out = []
    out.append("TITLE    KH560-grafted SiO2, PCFF atom types (mapped)")
    out.append("REMARK   Atom order and CONECT from original 560use.pdb")
    out.append("REMARK   Atom Name columns: 13-16, mapped PCFF atomType labels")
    out.append("REMARK   1         2         3         4         5         6         7         8")
    out.append("REMARK78901234567890123456789012345678901234567890123456789012345678901234567890")
    for a in orig_atoms:
        aid = a["serial"]
        typ, x, y, z = coords[aid]
        label = labels[typ]
        element = a["element"] or label[:1].upper()
        out.append(
            f"HETATM{aid:5d} {label:<4s} MOL A   1"
            f"    {x:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00  0.00          {element:>2s}"
        )
    out.extend(conect)
    out.append("END")

    DST.write_text("\n".join(out) + "\n", encoding="ascii")
    print(f"wrote {DST} ({len(orig_atoms)} atoms, {len(conect)} CONECT lines)")


if __name__ == "__main__":
    main()
