"""OMat24 streaming config from an in-memory OAM config (2026-11-11): train_path / valid_path / test_paths are
replaced by stream sources (raw aselmdb trees) with pre-made size tables.

usage: python -I runs/diag/omat_stream/make_config.py <base.json> <out.json> --name <run name> \
           --train aselmdb:<dir> --train-sizes <dir> --valid aselmdb:<dir> --valid-sizes <dir> \
           [--steps-per-epoch 250] [--epochs 2] [--workers 8] [--valid-max 2000] [--set key=<json> ...]
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('base'); ap.add_argument('out')
ap.add_argument('--name', required=True)
ap.add_argument('--train', required=True); ap.add_argument('--train-sizes', required=True)
ap.add_argument('--valid', required=True); ap.add_argument('--valid-sizes', required=True)
ap.add_argument('--steps-per-epoch', type=int, default=250); ap.add_argument('--epochs', type=int, default=2)
ap.add_argument('--workers', type=int, default=8); ap.add_argument('--valid-max', type=int, default=2000)
ap.add_argument('--batch', type=int, help='graphs per device and per pack (default: batch_size of the base)')
ap.add_argument('--set', action='append', default=[], help='key=<json value>, applied last')
a = ap.parse_args()
c = json.load(open(a.base))
for k in ('train_path', 'valid_path', 'test_paths', 'valid_frac', 'boltzmann_tilt', 'node_weight', 'taylor_aug',
          'avg_n_neighbors'):                       # avg_n_neighbors comes from the full size table in stream mode
    c.pop(k, None)
assert c['atomic_numbers'] == sorted(c['atomic_numbers']), 'reorder atom_energies too'
assert c.get('e0_mode', 'isolated') == 'isolated' and 'atom_energies' in c
if a.batch:
    c['batch_size'] = a.batch
run = Path('runs') / f'{a.name}_cue'
c.update(data_format='stream', n_epochs=a.epochs, restart=False, gauge_refit=False, eval_test_every=10 ** 9,
         log_file=str(run / f'loss_{a.name}_cue.out'), ckpt_name=str(run / f'ckpt_{a.name}_cue.pkl'),
         ckpt_last_every=1,
         note=f"OMat24 streaming run from {a.base}: {a.epochs} virtual epochs x {a.steps_per_epoch} steps")
c['stream'] = {'train': {'sources': [a.train], 'sizes': a.train_sizes},
               'valid': {'sources': [a.valid], 'sizes': a.valid_sizes, 'max_structures': a.valid_max},
               'workers': a.workers, 'eval_workers': max(1, a.workers // 2),
               'graphs_per_pack': int(c.get('batch_size', 16)), 'steps_per_epoch': a.steps_per_epoch}
for kv in a.set:
    k, v = kv.split('=', 1)
    c[k] = json.loads(v)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
run.mkdir(parents=True, exist_ok=True)              # the trainer opens the log before making directories
json.dump(c, open(a.out, 'w'), indent=1)
print(a.out)
