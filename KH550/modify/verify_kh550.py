#!/usr/bin/env python3
"""Thorough verification of KH550 relabeled / centered data."""

from collections import Counter, defaultdict
from pathlib import Path


WORK = Path(__file__).resolve().parent
ORIG = WORK / "anneal_final.data"
REL = WORK / "anneal_final_relabeled.data"
CENTERED = WORK / "anneal_final_relabeled_centered.data"

FAIL = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAIL.append(name)
    print(f"[{status}] {name} {detail}")


def read_sections(path):
    lines = path.read_text(encoding="ascii", errors="ignore").splitlines()
    headers = {}
    for line in lines:
        s = line.strip()
        for key, unit in [
            ("atoms", "atoms"),
            ("atom_types", "atom types"),
            ("bonds", "bonds"),
            ("bond_types", "bond types"),
            ("angles", "angles"),
            ("angle_types", "angle types"),
            ("dihedrals", "dihedrals"),
            ("dihedral_types", "dihedral types"),
            ("impropers", "impropers"),
            ("improper_types", "improper types"),
        ]:
            if s.endswith(" " + unit):
                try:
                    headers[key] = int(s.split()[0])
                except ValueError:
                    pass
    sec = None
    atoms = {}
    bonds = []
    masses = {}
    pairs = {}
    for line in lines:
        s = line.strip()
        if s == "Masses":
            sec = "mass"
            continue
        if s.startswith("Pair Coeffs"):
            sec = "pair"
            continue
        if s.startswith("Bond Coeffs"):
            if sec in ("mass", "pair"):
                sec = None
        if s == "Atoms # full":
            sec = "atoms"
            continue
        if s == "Velocities":
            sec = "vel"
            continue
        if s == "Bonds":
            sec = "bonds"
            continue
        if s == "Angles":
            sec = None
        q = line.split()
        if sec == "mass" and len(q) >= 2 and q[0].isdigit():
            masses[int(q[0])] = float(q[1])
        elif sec == "pair" and len(q) >= 2 and q[0].isdigit():
            pairs[int(q[0])] = " ".join(q[1:])
        elif sec == "atoms" and len(q) >= 10 and q[0].isdigit():
            atoms[int(q[0])] = {
                "mol": int(q[1]),
                "type": int(q[2]),
                "x": float(q[4]),
                "y": float(q[5]),
                "z": float(q[6]),
                "ix": int(q[7]),
                "iy": int(q[8]),
                "iz": int(q[9]),
            }
        elif sec == "bonds" and len(q) >= 4 and q[0].isdigit():
            bonds.append((int(q[2]), int(q[3])))
    return headers, atoms, bonds, masses, pairs


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
    print("=" * 70)
    print("KH550 relabeled / centered data verification")
    print("=" * 70)

    h0, a0, b0, m0, p0 = read_sections(ORIG)
    h1, a1, b1, m1, p1 = read_sections(REL)
    h2, a2, b2, m2, p2 = read_sections(CENTERED)

    print("\n-- headers --")
    check("atom_types 11 -> 14", h0.get("atom_types") == 11 and h1.get("atom_types") == h2.get("atom_types") == 14)
    for key in [
        "atoms",
        "bonds",
        "bond_types",
        "angles",
        "angle_types",
        "dihedrals",
        "dihedral_types",
        "impropers",
        "improper_types",
    ]:
        check(
            f"{key} preserved",
            h0.get(key) == h1.get(key) == h2.get(key),
            f"{h0.get(key)} / {h1.get(key)} / {h2.get(key)}",
        )

    print("\n-- atom count / id coverage --")
    check("atom ids 1..N complete", set(a1) == set(range(1, len(a1) + 1)))
    check("relabeled atom count", len(a1) == h0["atoms"])
    check("centered atom count", len(a2) == h0["atoms"])

    print("\n-- masses / pair coeffs --")
    check("masses count", len(m1) == h1["atom_types"], str(len(m1)))
    check("pair coeffs count", len(p1) == h1["atom_types"], str(len(p1)))
    for old, new in ((3, 12), (4, 13), (7, 14)):
        check(
            f"mass {new} == mass {old}",
            m1.get(new) == m1.get(old),
            f"{m1.get(old)} / {m1.get(new)}",
        )
        check(
            f"pair {new} == pair {old}",
            p1.get(new) == p1.get(old),
            f"{p1.get(old)} / {p1.get(new)}",
        )

    print("\n-- component analysis --")
    comps = components(b1)
    sizes = Counter(len(c) for c in comps)
    check("121 components", len(comps) == 121, str(len(comps)))
    check("120 x 134 FKM chains", sizes.get(134) == 120, str(dict(sizes)))
    check("1 x 1750 cluster", sizes.get(1750) == 1, str(dict(sizes)))

    big = max(comps, key=len)
    big_set = set(big)
    fkm_set = set()
    for c in comps:
        if len(c) == 134:
            fkm_set.update(c)
    big_types = {a1[i]["type"] for i in big_set}
    fkm_types = {a1[i]["type"] for i in fkm_set}
    check(
        "cluster types exact",
        big_types == {6, 8, 9, 10, 11, 12, 13, 14},
        str(sorted(big_types)),
    )
    check(
        "FKM types exact",
        fkm_types == {1, 2, 3, 4, 5, 7},
        str(sorted(fkm_types)),
    )
    check("type overlap empty", not (big_types & fkm_types))

    print("\n-- per-atom mapping --")
    map_ok = True
    for aid in range(1, len(a0) + 1):
        t0 = a0[aid]["type"]
        t1 = a1[aid]["type"]
        expected = {3: 12, 4: 13, 7: 14}.get(t0, t0)
        if aid not in big_set:
            expected = t0
        if t1 != expected:
            map_ok = False
            print("  mismatch", aid, t0, t1, expected)
            break
    check("per-atom type mapping", map_ok)
    check(
        "centered types == relabeled types",
        {aid: a["type"] for aid, a in a1.items()}
        == {aid: a["type"] for aid, a in a2.items()},
    )

    print("\n-- coordinates --")
    coord_ok = True
    for aid in range(1, len(a0) + 1):
        if (
            a0[aid]["x"],
            a0[aid]["y"],
            a0[aid]["z"],
            a0[aid]["ix"],
            a0[aid]["iy"],
            a0[aid]["iz"],
        ) != (
            a1[aid]["x"],
            a1[aid]["y"],
            a1[aid]["z"],
            a1[aid]["ix"],
            a1[aid]["iy"],
            a1[aid]["iz"],
        ):
            coord_ok = False
            print("  coord mismatch", aid)
            break
    check("relabeled coords unchanged", coord_ok)

    print("\n-- centered geometry --")
    box = {k: (0.0, 0.0) for k in ("xlo", "ylo", "zlo")}
    for line in CENTERED.read_text(encoding="ascii").splitlines():
        q = line.split()
        if len(q) == 4 and q[2] in ("xlo", "ylo", "zlo"):
            box[q[2]] = (float(q[0]), float(q[1]))
    L = {
        k: box[k][1] - box[k][0]
        for k in ("xlo", "ylo", "zlo")
    }
    center = {k: (box[k][0] + box[k][1]) / 2 for k in box}
    mass = m2
    com = [0.0, 0.0, 0.0]
    tm = 0.0
    for aid in big_set:
        a = a2[aid]
        m = mass[a["type"]]
        com[0] += m * (a["x"] + a["ix"] * L["xlo"])
        com[1] += m * (a["y"] + a["iy"] * L["ylo"])
        com[2] += m * (a["z"] + a["iz"] * L["zlo"])
        tm += m
    com = [c / tm for c in com]
    offset = [com[i] - [center[k] for k in ("xlo", "ylo", "zlo")][i] for i in range(3)]
    check("cluster COM at box center", max(abs(o) for o in offset) < 1e-6, str(offset))

    outside = 0
    for aid, a in a2.items():
        if not (
            box["xlo"][0] <= a["x"] <= box["xlo"][1]
            and box["ylo"][0] <= a["y"] <= box["ylo"][1]
            and box["zlo"][0] <= a["z"] <= box["zlo"][1]
        ):
            outside += 1
    check("all atoms inside centered box", outside == 0, f"outside={outside}")

    print("\n-- bonds --")
    cross = 0
    for i, j in b2:
        if (i in big_set) != (j in big_set):
            cross += 1
    check("no cluster-FKM covalent bonds", cross == 0, f"cross={cross}")

    print("\n" + "=" * 70)
    if FAIL:
        print(f"FAILED checks: {FAIL}")
    else:
        print("ALL CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
