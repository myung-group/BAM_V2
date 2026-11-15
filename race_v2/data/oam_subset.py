"""Random subsets of the OAM training data (OMat24, MPtrj, sAlex) as extended XYZ for RACEv2 screening runs.

Reads the packed super-batch shards of BAM-GAMP-jax (``bam_jax.data.schema``: one pickle per shard, a list of
super-batches with positions, species index, cell, raw DFT energy, forces, stress) and writes
``<out>/<name>_{train,valid,test}.xyz`` plus ``<out>/<name>_meta.json``.

    python -m bam_omat24.oam_subset --name omat24 --train-root /scratch/t510a13/data/oam_data/omat24_train \
        --eval-root /scratch/t510a13/data/oam_data/omat24_val --atom-ref /scratch/t510a13/data/oam_data/refs/omat24_train_ls.npz \
        --n-train 10000 --n-valid 1000 --n-test 1000 --out /scratch/t510a13/data/datasets_ext/oam_subsets

Conventions of the shards (scripts/convert_aselmdb_to_shards.py): energy = raw total DFT energy (eV), forces eV/A,
stress eV/A^3 with ASE sign (positive = tensile) and ASE Voigt order (xx, yy, zz, yz, xz, xy).  Species are encoded
indices; ``species_encoding`` of the atom-reference npz maps Z -> index and is inverted here.

Sampling: shard files are shuffled (seed) over all shape buckets, and at most ``--per-shard`` graphs are drawn at
random from each opened shard, so the subset spans compositions, sizes and (for OMat24) the sub-datasets without
reading the whole set.  Train comes from the training split only; valid and test are disjoint draws from the
evaluation split (``--eval-root``).  The meta file records the element list, the per-element least-squares
reference energies of the full training split (``atom_ref``; used as E0) and the mean neighbour count at the
RACE cutoff.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np


def list_shards(root: Path) -> list[Path]:
    out = []
    for b in sorted(os.scandir(root), key=lambda e: e.name):
        if b.is_dir() and not b.name.startswith(('.', '_')):
            out += [Path(f.path) for f in os.scandir(b.path) if f.name.startswith('shard_') and f.name.endswith('.pkl')]
    return sorted(out)


def graphs_from_shard(path: Path, idx_to_z: np.ndarray):
    """Yield (numbers, positions, cell, energy, forces, stress) for every real graph of a shard."""
    with open(path, 'rb') as fh:
        d = pickle.load(fh)
    for sb in d['batches']:
        nm, gm = np.asarray(sb['node_mask']), np.asarray(sb['graph_mask'])
        batch = np.asarray(sb['batch'])
        for g in np.nonzero(gm)[0]:
            sel = nm & (batch == g)
            yield (idx_to_z[np.asarray(sb['species'])[sel]], np.asarray(sb['positions'], np.float64)[sel],
                   np.asarray(sb['cell'], np.float64)[g], float(sb['energy'][g]),
                   np.asarray(sb['forces'], np.float64)[sel], np.asarray(sb['stress'], np.float64)[g])


def draw(root: Path, n: int, per_shard: int, rng: np.random.Generator, idx_to_z: np.ndarray, skip=None):
    files = list_shards(root)
    rng.shuffle(files)
    out, used = [], []
    for f in files:
        if len(out) >= n:
            break
        gs = list(graphs_from_shard(f, idx_to_z))
        keep = rng.permutation(len(gs))
        k = 0
        for i in keep:
            if k >= per_shard or len(out) >= n:
                break
            key = (f.name, int(i))
            if skip is not None and key in skip:
                continue
            out.append(gs[i]); used.append(key); k += 1
    if len(out) < n:
        raise RuntimeError(f"{root}: only {len(out)} graphs drawn, {n} requested")
    return out, set(used)


def to_atoms(g):
    import ase
    from ase.calculators.singlepoint import SinglePointCalculator
    numbers, pos, cell, e, f, s = g
    a = ase.Atoms(numbers=numbers, positions=pos, cell=cell, pbc=True)
    a.calc = SinglePointCalculator(a, energy=e, forces=f, stress=s)
    return a


def mean_neighbors(atoms_list, cutoff: float, n_max: int = 500) -> float:
    from ase.neighborlist import neighbor_list
    counts = [len(neighbor_list('i', a, cutoff)) / len(a) for a in atoms_list[:n_max]]
    return float(np.mean(counts))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--name', required=True)
    ap.add_argument('--train-root', required=True, type=Path)
    ap.add_argument('--eval-root', required=True, type=Path)
    ap.add_argument('--atom-ref', required=True, type=Path)
    ap.add_argument('--n-train', type=int, default=10000)
    ap.add_argument('--n-valid', type=int, default=1000)
    ap.add_argument('--n-test', type=int, default=1000)
    ap.add_argument('--per-shard', type=int, default=50)
    ap.add_argument('--cutoff', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()

    import ase.io
    ref = np.load(args.atom_ref)
    enc = np.asarray(ref['species_encoding'])                 # Z -> index, -1 = not encoded
    zs = np.nonzero(enc >= 0)[0]
    idx_to_z = np.zeros(int(enc.max()) + 1, dtype=np.int64)
    idx_to_z[enc[zs]] = zs
    assert np.all(np.diff(enc[zs]) == 1) and enc[zs][0] == 0, "encoding is not Z-sorted"   # index order = sorted Z
    atom_ref = np.asarray(ref['atom_ref'], dtype=np.float64)

    rng = np.random.default_rng(args.seed)
    train, _ = draw(args.train_root, args.n_train, args.per_shard, rng, idx_to_z)
    ev, _ = draw(args.eval_root, args.n_valid + args.n_test, args.per_shard, rng, idx_to_z)
    ev = [ev[i] for i in rng.permutation(len(ev))]
    parts = {'train': train, 'valid': ev[:args.n_valid], 'test': ev[args.n_valid:]}

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {'name': args.name, 'train_root': str(args.train_root), 'eval_root': str(args.eval_root),
            'atom_ref_path': str(args.atom_ref), 'seed': args.seed, 'per_shard': args.per_shard,
            'atomic_numbers': [int(z) for z in zs], 'atom_energies': [float(x) for x in atom_ref],
            'units': 'energy eV (raw DFT), forces eV/A, stress eV/A^3 (ASE sign, Voigt xx yy zz yz xz xy)'}
    for k, gs in parts.items():
        atoms = [to_atoms(g) for g in gs]
        p = args.out / f'{args.name}_{k}.xyz'
        ase.io.write(p, atoms, format='extxyz')
        nat = np.array([len(a) for a in atoms])
        epa = np.array([a.get_potential_energy() / len(a) for a in atoms])
        meta[k] = {'path': str(p), 'n': len(atoms), 'n_atoms_mean': float(nat.mean()), 'n_atoms_max': int(nat.max()),
                   'E_per_atom_mean': float(epa.mean()),
                   'elements_present': sorted({int(z) for a in atoms for z in a.numbers})}
        if k == 'train':
            meta['avg_n_neighbors'] = mean_neighbors(atoms, args.cutoff)
        print(k, meta[k]['n'], 'atoms/graph', round(meta[k]['n_atoms_mean'], 1), 'E/atom', round(meta[k]['E_per_atom_mean'], 3),
              'elements', len(meta[k]['elements_present']), flush=True)
    print('avg_n_neighbors', round(meta['avg_n_neighbors'], 2))
    with open(args.out / f'{args.name}_meta.json', 'w') as fh:
        json.dump(meta, fh, indent=1)


if __name__ == '__main__':
    main()
