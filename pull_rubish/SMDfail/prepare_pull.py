"""Build a slab data file for pull-out from the NVE-verified data file.

The NVE data file is fully periodic.  For a z pull-out we need:
  1. unwrapped z coordinates (using image flags),
  2. a fixed slab with vacuum on both sides,
  3. z image flags reset to 0.

Usage:
    python prepare_pull.py <input.data> [output.data]
"""

from pathlib import Path
import sys


ZLO_NEW = 0.0
ZHI_NEW = 200.0
BOTTOM_MARGIN = 10.0


def parse_dump_z(path, lz):
    """Return {atom_id: unwrapped_z} using the dump image flags."""
    mapping = {}
    in_atoms = False
    for raw in path.read_text(encoding="ascii", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("ITEM: ATOMS"):
            in_atoms = True
            continue
        if not in_atoms:
            continue
        parts = line.split()
        if len(parts) >= 8:
            try:
                aid = int(parts[0])
                z = float(parts[4])
                iz = int(parts[7])
            except ValueError:
                continue
            mapping[aid] = z + iz * lz
    return mapping


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else r"F:\Lammps-MS\FKM-SIO2-TEST\NPT-NVT-NVE\server-output\nve_verify.data")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name("pull_prep.data")
    dump_path = src.with_name("nve_verify.dump")
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = src.read_text(encoding="ascii").splitlines()
    zlo = zhi = None
    atom_count = 0
    atoms_raw = []
    in_atoms = False
    saw_blank_after_atoms = False

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "Atoms # full":
            in_atoms = True
            continue
        if in_atoms:
            if not stripped:
                if atom_count > 0:
                    saw_blank_after_atoms = True
                    in_atoms = False
                continue
            parts = stripped.split()
            if len(parts) >= 10:
                try:
                    atoms_raw.append((idx, parts))
                    atom_count += 1
                except ValueError:
                    pass
        parts = stripped.split()
        if len(parts) == 4 and parts[2] == "zlo" and parts[3] == "zhi":
            zlo, zhi = float(parts[0]), float(parts[1])

    if zlo is None or zhi is None:
        raise SystemExit("z box bounds not found")
    if atom_count == 0:
        raise SystemExit("no atoms found")

    lz = zhi - zlo
    z_from_dump = parse_dump_z(dump_path, lz) if dump_path.exists() else {}
    shifted = []
    mapped = 0
    for idx, parts in atoms_raw:
        aid, mol, typ, q = parts[0], parts[1], parts[2], parts[3]
        x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
        ix, iy, iz = int(parts[7]), int(parts[8]), int(parts[9])
        z_unwrapped = z_from_dump.get(int(aid), z + iz * lz)
        if int(aid) in z_from_dump:
            mapped += 1
        shifted.append((idx, aid, mol, typ, q, x, y, z_unwrapped, ix, iy))

    zmin = min(row[7] for row in shifted)
    zmax = max(row[7] for row in shifted)
    shift = BOTTOM_MARGIN - zmin

    for idx, aid, mol, typ, q, x, y, z_u, ix, iy in shifted:
        z_new = z_u + shift
        lines[idx] = f"{aid} {mol} {typ} {q} {x:.10f} {y:.10f} {z_new:.10f} {ix} {iy} 0"

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) == 4 and parts[2] == "zlo" and parts[3] == "zhi":
            lines[idx] = f"{ZLO_NEW:.1f} {ZHI_NEW:.1f} zlo zhi"

    out.write_text("\n".join(lines) + "\n", encoding="ascii")

    print(f"read    : {src}")
    print(f"atoms   : {atom_count}")
    print(f"z from dump mapped: {mapped}/{atom_count}")
    print(f"old z box: {zlo:.6f} {zhi:.6f}")
    print(f"new z box: {ZLO_NEW:.1f} {ZHI_NEW:.1f}")
    print(f"old z range (unwrapped): {zmin:.2f} to {zmax:.2f} A")
    print(f"new z range: {zmin + shift:.2f} to {zmax + shift:.2f} A")
    print(f"shift   : {shift:.4f} A")
    print(f"written : {out}")


if __name__ == "__main__":
    main()
