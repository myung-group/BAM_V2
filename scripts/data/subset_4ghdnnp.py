"""Balanced training subsets of the 4G-HDNNP splits (convert_4ghdnnp.py output) for local tests.

usage: python -I data_prep/subset_4ghdnnp.py <out_dir of convert_4ghdnnp.py>

Per system: 500 structures of each subset label (charge state / structure family) from train.xyz, drawn with
numpy RandomState(0).choice(indices of that label, 500, replace=False), labels in sorted order, one RandomState for
the system -> train_sub1000.xyz (archive order kept) and subset_sub1000.json.  Au2/MgO also gets train_sub100.xyz:
50 per label drawn the same way (new RandomState(0)) from train_sub1000.xyz.  The json files hold the indices,
source_index and avg_n_neighbors (cutoff 5 A, counted with the trainer's own graph code) for the configs.
The test split is not subsampled.
"""
import json
import sys
from collections import defaultdict

import numpy as np
import ase.io

sys.path.insert(0, '.')
from race_v2.data.xyz_graphs import preprocess_graph, atomic_numbers_to_indices   # noqa: E402

CUTOFF = 5.0


def balanced(frames, n_per):
    by = defaultdict(list)
    for i, a in enumerate(frames):
        by[a.info['subset']].append(i)
    rng = np.random.RandomState(0)
    return np.sort(np.concatenate([rng.choice(by[k], n_per, replace=False) for k in sorted(by)]))


def avg_neighbors(frames, Z):
    ai = atomic_numbers_to_indices(Z)
    return sum(int(preprocess_graph(a, ai, [0.0] * len(Z), CUTOFF).n_edge.sum()) for a in frames) / \
        sum(len(a) for a in frames)


def write(d, name, frames, idx, key, rule, Z):
    sub = [frames[i] for i in idx]
    ase.io.write(f'{d}/train_{name}.xyz', sub, format='extxyz')
    info = dict(rule=rule, **{key: [int(i) for i in idx]},
                source_index=[int(a.info['source_index']) for a in sub], avg_n_neighbors=avg_neighbors(sub, Z))
    json.dump(info, open(f'{d}/subset_{name}.json', 'w'))
    print(d, name, len(sub), f"avg_n_neighbors={info['avg_n_neighbors']:.3f}")
    return sub


def main():
    root = sys.argv[1]
    for s in ['carbon_chain', 'ag_cluster', 'nacl', 'aumgo']:
        d = f'{root}/{s}'
        fr = ase.io.read(f'{d}/train.xyz', ':')
        Z = sorted(set(int(z) for a in fr for z in a.numbers))
        sub = write(d, 'sub1000', fr, balanced(fr, 500), 'train_index',
                    '500 per subset from train.xyz, RandomState(0).choice per subset in sorted subset order', Z)
        if s == 'aumgo':
            write(d, 'sub100', sub, balanced(sub, 50), 'index_in_sub1000',
                  '50 per subset drawn from train_sub1000.xyz, RandomState(0).choice per subset in sorted order', Z)


if __name__ == '__main__':
    main()
