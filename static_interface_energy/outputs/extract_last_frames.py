import sys
from pathlib import Path


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    n_last = int(sys.argv[3])

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
