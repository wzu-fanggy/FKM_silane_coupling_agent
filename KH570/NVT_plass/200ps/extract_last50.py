#!/usr/bin/env python3
"""Extract the last 50 frames of nvt_200ps_traj.dump into last50.dump."""

import sys
from pathlib import Path


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    src = work / "nvt_200ps_traj.dump"
    dst = work / "last50.dump"
    n_last = 50

    lines = src.read_text(encoding="ascii", errors="ignore").splitlines(keepends=True)
    idx = [i for i, line in enumerate(lines) if line.startswith("ITEM: TIMESTEP")]
    if len(idx) < n_last:
        raise SystemExit(f"only {len(idx)} frames, cannot take {n_last}")
    start = idx[-n_last]
    with open(dst, "w", encoding="ascii", newline="") as f:
        f.writelines(lines[start:])
    print(f"Wrote {dst}: {n_last} frames")


if __name__ == "__main__":
    main()
