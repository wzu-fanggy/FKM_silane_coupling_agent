#!/usr/bin/env python3
"""Create a copy of a LAMMPS data file with all atom charges set to zero."""

import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: make_charge_zero.py input.data output.data")
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    out_lines = []
    in_atoms = False
    with open(src, "r", encoding="ascii", errors="ignore", newline="") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("Atoms"):
                in_atoms = True
                out_lines.append(line)
                continue
            if stripped.startswith("Bonds"):
                in_atoms = False
                out_lines.append(line)
                continue
            if in_atoms and stripped:
                parts = stripped.split()
                if len(parts) == 10:
                    # id mol type q x y z nx ny nz
                    parts[3] = "0.0"
                    out_lines.append(" ".join(parts) + "\n")
                    continue
            out_lines.append(line)

    with open(dst, "w", encoding="ascii", newline="") as f:
        f.writelines(out_lines)
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()