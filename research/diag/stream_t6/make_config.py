"""T6: a data_format "stream" copy of an in-memory config (train_path / valid_path / test_paths -> stream sources),
short run: n_epochs virtual epochs of steps_per_epoch steps (2026-11-09).

usage: python -I runs/diag/stream_t6/make_config.py <base.json> <out.json> --run-dir runs/stream_t6/<name> \
           --sizes <train size table dir> [--steps-per-epoch 250] [--epochs 2] [--workers 8] [--valid-max 1000]
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('base'); ap.add_argument('out')
ap.add_argument('--run-dir', required=True); ap.add_argument('--sizes', required=True)
ap.add_argument('--steps-per-epoch', type=int, default=250); ap.add_argument('--epochs', type=int, default=2)
ap.add_argument('--workers', type=int, default=8); ap.add_argument('--valid-max', type=int, default=1000)
a = ap.parse_args()
c = json.load(open(a.base))
tr, va, te = c.pop('train_path'), c.pop('valid_path'), c.pop('test_paths', [])
for k in ('valid_frac', 'boltzmann_tilt', 'node_weight', 'taylor_aug'):
    c.pop(k, None)
c['atomic_numbers'] = sorted(int(z) for z in c['atomic_numbers'])
assert c['atomic_numbers'] == [int(z) for z in json.load(open(a.base))['atomic_numbers']], 'reorder atom_energies too'
assert c.get('e0_mode', 'isolated') == 'isolated' and 'atom_energies' in c
run = Path(a.run_dir)
c.update(data_format='stream', n_epochs=a.epochs, restart=False, gauge_refit=False, eval_test_every=a.epochs,
         lr_warmup_epochs=0, log_file=str(run / 'loss.out'), ckpt_name=str(run / 'ckpt.pkl'),
         ckpt_last_every=a.epochs,
         note=f"T6 throughput smoke from {a.base}: {a.epochs} x {a.steps_per_epoch} steps, data_format stream")
c['stream'] = {'train': {'sources': [f'extxyz:{tr}'], 'sizes': a.sizes},
               'valid': {'sources': [f'extxyz:{va}'], 'max_structures': a.valid_max},
               'test': {'test': {'sources': [f'extxyz:{p}'], 'max_structures': a.valid_max} for p in te[:1]},
               'workers': a.workers, 'eval_workers': max(1, a.workers // 2),
               'graphs_per_pack': int(c.get('batch_size', 16)), 'steps_per_epoch': a.steps_per_epoch}
if not te:
    c['stream'].pop('test')
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
run.mkdir(parents=True, exist_ok=True)              # the trainer opens loss.out before making directories
json.dump(c, open(a.out, 'w'), indent=1)
print(a.out)
