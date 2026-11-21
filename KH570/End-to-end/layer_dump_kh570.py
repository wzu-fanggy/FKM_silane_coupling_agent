# -*- coding: utf-8 -*-
"""
FKM End-to-End Distance - Multi-Frame Trajectory Analysis (KH570 system)
分层 + 质心距离 + 端到端距离

KH570 参数:
  R_surf      = 15.05 A  (SiO2 半高密度表面半径)
  delta_pert  =  6.08 A  (KH570 界面扰动层厚度)
  FKM 分子    = mol 2..121
  FKM 原子类型 = 1,2,3,4,8,10
  SIO2 复合体 = 5,6,7,9,11,12,13,14,15,16,17
  主链键类型  = 1,2,3,5 (与裸体系一致)
"""
import numpy as np
from collections import defaultdict

R_SURF = 15.05
DELTA_PERT = 6.08
LAYERS = {
    'contact':   (R_SURF, R_SURF + 3.0),
    'interface': (R_SURF + 3.0, R_SURF + DELTA_PERT),
    'bulk':     (R_SURF + DELTA_PERT, 1e6),
}
SIO2_TYPES = {5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17}
MAIN_CHAIN_BOND_TYPES = {1, 2, 3, 5}


def read_lammps_dump(filename):
    frames = []
    with open(filename) as f:
        lines = f.readlines()
    n = len(lines)
    i = 0
    na = 0
    box = np.array([57.6, 57.6, 57.6])
    while i < n:
        line = lines[i].strip()
        if line == 'ITEM: TIMESTEP':
            i += 2
        elif line == 'ITEM: NUMBER OF ATOMS':
            na = int(lines[i+1].strip())
            i += 2
        elif 'ITEM: BOX BOUNDS' in line:
            x = lines[i+1].split()[:2]
            y = lines[i+2].split()[:2]
            z = lines[i+3].split()[:2]
            box = np.array([float(x[1])-float(x[0]), float(y[1])-float(y[0]), float(z[1])-float(z[0])])
            i += 4
        elif 'ITEM: ATOMS' in line:
            if na > 0:
                atoms = {}
                for j in range(1, na + 1):
                    p = lines[i+j].split()
                    if len(p) >= 8:
                        atoms[int(p[0])] = {
                            't': int(p[1]),
                            'x': float(p[2]),
                            'y': float(p[3]),
                            'z': float(p[4]),
                            'ix': int(p[5]),
                            'iy': int(p[6]),
                            'iz': int(p[7]),
                        }
                frames.append({'box': box, 'atoms': atoms})
            i += na + 1
        else:
            i += 1
    return frames


def read_data_file(filename):
    with open(filename) as f:
        lines = f.readlines()

    mol_atoms = defaultdict(list)
    bonds = []
    in_atoms = in_bonds = False

    for line in lines:
        if 'Atoms # full' in line:
            in_atoms = True
            continue
        if 'Bonds' == line.strip():
            in_atoms = False
            in_bonds = True
            continue
        if 'Angles' in line:
            break

        if in_atoms and line.strip():
            p = line.split()
            if len(p) >= 7:
                mol_atoms[int(p[1])].append(int(p[0]))

        if in_bonds and line.strip():
            p = line.split()
            if len(p) >= 4:
                bonds.append({'type': int(p[1]), 'a1': int(p[2]), 'a2': int(p[3])})

    return mol_atoms, bonds


def bfs_farthest(adj, start):
    visited = {start: 0}
    queue = [(start, 0)]
    farthest, max_d = start, 0
    while queue:
        node, d = queue.pop(0)
        for nb in adj[node]:
            if nb not in visited:
                visited[nb] = d + 1
                queue.append((nb, d + 1))
                if d + 1 > max_d:
                    max_d = d + 1
                    farthest = nb
    return farthest, max_d


def calc_ete(mol_ids_set, atoms, bonds, box):
    adj = defaultdict(list)
    for b in bonds:
        if b['a1'] in mol_ids_set and b['a2'] in mol_ids_set and b['type'] in MAIN_CHAIN_BOND_TYPES:
            adj[b['a1']].append(b['a2'])
            adj[b['a2']].append(b['a1'])
    if len(adj) < 2:
        return None, 0
    start = list(adj.keys())[0]
    far1, _ = bfs_farthest(adj, start)
    far2, path = bfs_farthest(adj, far1)
    r1 = np.array([atoms[far1]['x'], atoms[far1]['y'], atoms[far1]['z']])
    r2 = np.array([atoms[far2]['x'], atoms[far2]['y'], atoms[far2]['z']])
    dr = r2 - r1 - box * np.round((r2 - r1) / box)
    return np.linalg.norm(dr), path


def main():
    base = r"F:\Lammps-MS\FKM-SIO2-TEST\KH570\End-to-end"
    traj_file = base + r"\nvt_500ps_traj.dump"
    data_file = base + r"\nvt_500ps.data"
    csv_path = base + r"\ete_multiframe_full.csv"
    out_txt = base + r"\fkm_ete_layer_results.txt"

    print("Reading trajectory...")
    frames = read_lammps_dump(traj_file)
    print("Loaded " + str(len(frames)) + " frames")

    print("Reading molecule structure...")
    mol_atoms, bonds = read_data_file(data_file)
    fkm_ids = sorted([m for m in mol_atoms.keys() if m >= 2])
    print("FKM chains: " + str(len(fkm_ids)))
    print("Bonds: " + str(len(bonds)))

    results = {l: [] for l in LAYERS}
    results['outside'] = []

    for fi, frame in enumerate(frames):
        atoms = frame['atoms']
        box = frame['box']

        coords = [[atoms[a]['x'], atoms[a]['y'], atoms[a]['z']]
                  for a in atoms if atoms[a]['t'] in SIO2_TYPES]
        sio2_com = np.mean(coords, axis=0)

        for mol_id in fkm_ids:
            mol_ids_set = set(mol_atoms[mol_id])
            ete, path = calc_ete(mol_ids_set, atoms, bonds, box)
            if ete is None:
                continue

            chain_coords = [
                [atoms[a]['x'] + atoms[a]['ix'] * box[0],
                 atoms[a]['y'] + atoms[a]['iy'] * box[1],
                 atoms[a]['z'] + atoms[a]['iz'] * box[2]]
                for a in mol_atoms[mol_id]
            ]
            center = np.mean(chain_coords, axis=0) - sio2_com
            dist = np.linalg.norm(center)

            layer = 'outside'
            for name, (lo, hi) in LAYERS.items():
                if lo <= dist < hi:
                    layer = name
                    break

            results[layer].append({'frame': fi, 'mol_id': mol_id, 'dist': dist, 'ete': ete, 'path': path})

        if (fi + 1) % 50 == 0:
            print("Processed " + str(fi + 1) + " frames")

    print("\n" + "=" * 70)
    print("Layer-Resolved End-to-End Distance Analysis (KH570)")
    print("=" * 70)
    print("Layer boundaries: contact %.2f-%.2f, interface %.2f-%.2f, bulk > %.2f" % (
        LAYERS['contact'][0], LAYERS['contact'][1],
        LAYERS['interface'][0], LAYERS['interface'][1],
        LAYERS['bulk'][0]))

    lines = []
    for layer in ['contact', 'interface', 'bulk', 'outside']:
        data = results[layer]
        if data:
            dists = [d['dist'] for d in data]
            etes = [d['ete'] for d in data]
            lines.append("[%s] Count=%d <dist>=%.2f A <R_ee>=%.2f A std=%.2f range=%.2f-%.2f" % (
                layer.upper(), len(data), np.mean(dists), np.mean(etes),
                np.std(etes), np.min(etes), np.max(etes)))
            print("  [" + layer.upper() + "]")
            print("    Count: " + str(len(data)))
            print("    <dist to SiO2>: " + str(round(np.mean(dists), 2)) + " A")
            print("    <R_ee>: " + str(round(np.mean(etes), 2)) + " A")
            print("    std(R_ee): " + str(round(np.std(etes), 2)) + " A")
            print("    range(R_ee): " + str(round(np.min(etes), 2)) + " - " + str(round(np.max(etes), 2)) + " A")
        else:
            lines.append("[%s] Count=0" % layer.upper())

    with open(csv_path, 'w') as f:
        f.write("frame,mol_id,layer,dist_to_sio2_A,ete_A,path_length\n")
        for layer in ['contact', 'interface', 'bulk', 'outside']:
            for r in results[layer]:
                f.write(str(r['frame']) + "," + str(r['mol_id']) + "," + layer + "," +
                        str(round(r['dist'], 4)) + "," + str(round(r['ete'], 4)) + "," +
                        str(r['path']) + "\n")

    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write("KH570 FKM layer-resolved end-to-end distance\n")
        f.write("R_surf=%.2f A, delta_pert=%.2f A\n" % (R_SURF, DELTA_PERT))
        f.write("\n".join(lines) + "\n")

    print("\nCSV saved: " + csv_path)
    print("Summary saved: " + out_txt)


if __name__ == '__main__':
    main()
