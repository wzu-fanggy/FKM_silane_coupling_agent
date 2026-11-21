# -*- coding: utf-8 -*-
"""
FKM End-to-End Distance - Layer-Resolved Analysis (Single Frame)
"""
import numpy as np
from collections import defaultdict, deque

R_SURF = 15.1
LAYERS = {
    'contact':   (R_SURF, R_SURF + 3.0),
    'interface': (R_SURF + 3.0, R_SURF + 8.6),
    'bulk':     (R_SURF + 8.6, 1e6),
}
SIO2_TYPES = {7, 8, 9}
MAIN_CHAIN_BOND_TYPES = {1, 2, 3, 5}

def parse_data(filename):
    with open(filename) as f:
        lines = f.readlines()
    box = None
    for line in lines:
        if 'xlo xhi' in line:
            p = line.split()
            box = np.array([float(p[1])-float(p[0]), 0, 0])
        if 'ylo yhi' in line:
            p = line.split()
            box[1] = float(p[1])-float(p[0])
        if 'zlo zhi' in line:
            p = line.split()
            box[2] = float(p[1])-float(p[0])
    atoms_start = bonds_start = None
    for i, line in enumerate(lines):
        if 'Atoms # full' in line:
            atoms_start = i + 1
        if line.strip() == 'Bonds':
            bonds_start = i + 1
            break
    angles_start = None
    for i, line in enumerate(lines):
        if line.strip() == 'Angles':
            angles_start = i
            break
    atoms = {}
    mol_atoms = defaultdict(list)
    for line in lines[atoms_start:bonds_start-1]:
        if not line.strip():
            continue
        p = line.split()
        if len(p) >= 9:
            atoms[int(p[0])] = {
                'mol': int(p[1]),
                'type': int(p[2]),
                'x': float(p[4]),
                'y': float(p[5]),
                'z': float(p[6])
            }
            mol_atoms[int(p[1])].append(int(p[0]))
    bonds = []
    for line in lines[bonds_start:angles_start]:
        if not line.strip():
            continue
        p = line.split()
        if len(p) >= 4:
            bonds.append({'type': int(p[1]), 'a1': int(p[2]), 'a2': int(p[3])})
    return atoms, bonds, mol_atoms, box

def verify_connectivity(mol_ids, bonds):
    adj = defaultdict(list)
    for b in bonds:
        if b['a1'] in mol_ids and b['a2'] in mol_ids and b['type'] in MAIN_CHAIN_BOND_TYPES:
            adj[b['a1']].append(b['a2'])
            adj[b['a2']].append(b['a1'])
    if len(adj) < 2:
        return False
    visited = {list(adj.keys())[0]}
    queue = list(visited)
    while queue:
        node = queue.pop(0)
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return visited == set(adj.keys())

def sio2_center(atoms):
    coords = [[atoms[a]['x'], atoms[a]['y'], atoms[a]['z']] 
              for a in atoms if atoms[a]['type'] in SIO2_TYPES]
    return np.mean(coords, axis=0)

def chain_center(mol_ids, atoms):
    coords = [[atoms[a]['x'], atoms[a]['y'], atoms[a]['z']] for a in mol_ids]
    return np.mean(coords, axis=0)

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

def calc_ete(mol_ids, bonds, atoms, box):
    adj = defaultdict(list)
    for b in bonds:
        if b['a1'] in mol_ids and b['a2'] in mol_ids and b['type'] in MAIN_CHAIN_BOND_TYPES:
            adj[b['a1']].append(b['a2'])
            adj[b['a2']].append(b['a1'])
    if len(adj) < 2:
        return None
    start = list(adj.keys())[0]
    far1, _ = bfs_farthest(adj, start)
    far2, path = bfs_farthest(adj, far1)
    r1 = np.array([atoms[far1]['x'], atoms[far1]['y'], atoms[far1]['z']])
    r2 = np.array([atoms[far2]['x'], atoms[far2]['y'], atoms[far2]['z']])
    dr = r2 - r1 - box * np.round((r2 - r1) / box)
    return np.linalg.norm(dr)

def main():
    data_file = r"F:\Lammps-MS\FKM-SIO2-TEST\End-to-end\nvt_500ps_gpu.data"
    
    with open(data_file) as f:
        atoms, bonds, mol_atoms, box = parse_data(data_file)
    
    sio2 = sio2_center(atoms)
    fkm_ids = sorted([m for m in mol_atoms.keys() if m >= 2])
    
    print("=" * 70)
    print("FKM End-to-End Distance - Layer-Resolved Analysis")
    print("=" * 70)
    print("SiO2 COM: ({:.2f}, {:.2f}, {:.2f}) A".format(sio2[0], sio2[1], sio2[2]))
    print("FKM chains: {}".format(len(fkm_ids)))
    print()
    
    results = {l: [] for l in LAYERS}
    results['outside'] = []
    
    for mol_id in fkm_ids:
        if not verify_connectivity(set(mol_atoms[mol_id]), bonds):
            continue
        ete = calc_ete(set(mol_atoms[mol_id]), bonds, atoms, box)
        if ete is None:
            continue
        cc = chain_center(mol_atoms[mol_id], atoms)
        dist = np.linalg.norm(cc - sio2)
        layer = 'outside'
        for name, (lo, hi) in LAYERS.items():
            if lo <= dist < hi:
                layer = name
                break
        results[layer].append((mol_id, ete, dist))
    
    print("{:<12} {:>6} {:>10} {:>10} {:>10} {:>10}".format(
        "Layer", "Count", "<R_ee>", "Std", "Min", "Max"))
    print("-" * 70)
    
    all_ete = []
    for layer in ['contact', 'interface', 'bulk', 'outside']:
        data = [r[1] for r in results[layer]]
        all_ete.extend(data)
        if data:
            print("{:<12} {:>6} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                layer, len(data), np.mean(data), np.std(data), np.min(data), np.max(data)))
        else:
            print("{:<12} {:>6}".format(layer, 0))
    
    print("-" * 70)
    print("{:<12} {:>6} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
        "Total", len(all_ete), np.mean(all_ete), np.std(all_ete), 
        np.min(all_ete), np.max(all_ete)))

if __name__ == '__main__':
    main()