"""Summarize kick gradient monitors into kick_gradient_summary.csv."""

from pathlib import Path

import numpy as np


def read_monitor(path: Path) -> np.ndarray:
    rows = []
    with path.open(encoding="ascii", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    return np.asarray(rows)


def main() -> None:
    records = []
    for path in sorted(Path(".").glob("kick_*Aps_monitor.txt")):
        a = read_monitor(path)
        if a.size == 0:
            continue
        vel = int(path.name.split("_")[1].replace("Aps_monitor.txt", ""))
        records.append(
            {
                "velocity_Aps": vel,
                "zsio2_start": a[0, 5],
                "zsio2_end": a[-1, 5],
                "zfkm_end": a[-1, 6],
                "zsep_end": a[-1, 7],
                "temp_max": a[:, 1].max(),
                "temp_end": a[-1, 1],
            }
        )

    if not records:
        raise SystemExit("no kick monitors found")

    import csv

    with open("kick_gradient_summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print("kick_gradient_summary.csv written")
    for r in records:
        print(
            f"{r['velocity_Aps']:>3} A/ps: "
            f"z_sio2 {r['zsio2_start']:.2f} -> {r['zsio2_end']:.2f} A, "
            f"zsep_end {r['zsep_end']:.2f} A, Tmax {r['temp_max']:.0f} K"
        )


if __name__ == "__main__":
    main()
