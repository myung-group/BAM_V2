"""Transfer test OMol25 -> OMat24: one table from the common evaluations (race_v2.eval.eval_sets, eval_<tag>.json next
to each checkpoint) and the parameter checks of the second-stage checkpoints against the pretraining checkpoint.

    PYTHONPATH=. JAX_PLATFORMS=cpu python research/diag/transfer/collect.py --pre runs/omol/<pre run>/ckpt.pkl
        --arms tr2_scratch tr2_cont tr2_replay tr2_frozen --omat-meta data_local/omat24_val/r300_10k/meta.json
        [--out research/diag/transfer/results.json]

Checks per arm that started from the pretraining checkpoint (EMA parameters):
  task rows    : which rows of the task table differ from zero (expected: 1 for OMat24 only, 0 and 1 with replay)
  absent rows  : entries of the species-indexed tables for the elements in neither dataset, unchanged (the raw
                 parameters exactly; the float32 EMA average of a constant drifts by rounding, about 1e-6 relative)
  frozen leaves: leaves outside the config's train_only, bit-identical to the pretraining values
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def leaves(tree):
    import jax
    return {jax.tree_util.keystr(p): np.asarray(v) for p, v in jax.tree_util.tree_leaves_with_path(tree)}


def main():
    from race_v2.training.utils import load_checkpoint
    ap = argparse.ArgumentParser()
    ap.add_argument('--pre', required=True)
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--omat-meta', required=True)
    ap.add_argument('--omol-z', type=int, nargs='+', default=[1, 6, 7, 8, 9, 16, 17])
    ap.add_argument('--z-max', type=int, default=94)
    ap.add_argument('--out', default='research/diag/transfer/results.json')
    args = ap.parse_args()
    seen = set(args.omol_z) | {int(z) for z in json.load(open(args.omat_meta))['atomic_numbers']}
    absent = np.array([z - 1 for z in range(1, args.z_max + 1) if z not in seen])
    pre = leaves(load_checkpoint(args.pre)['ema_params'])
    res = {'absent_elements': (absent + 1).tolist(), 'arms': {}}
    rows = [('pretrained (stage 1)', Path(args.pre).parent / 'eval_pre.json', None)]
    rows += [(a, Path('runs/transfer') / a / f'eval_{a}.json', a) for a in args.arms]
    print(f"{'arm':22s} | {'OMat24 E':>9s} {'F':>8s} | {'OMol25 E':>9s} {'F':>8s}   (MAE: meV/atom, meV/A)")
    for name, ev, arm in rows:
        r = json.load(open(ev))['sets']
        om, ol = r['omat_test.xyz'], r['omol_test.xyz']
        out = dict(epoch=json.load(open(ev))['epoch'], omat=om, omol=ol)
        print(f"{name:22s} | {om['energy_mae_meV_per_atom']:9.1f} {om['force_mae_meVA']:8.1f} | "
              f"{ol['energy_mae_meV_per_atom']:9.1f} {ol['force_mae_meVA']:8.1f}")
        if arm:
            cfg = json.load(open(f'research/input/transfer/{arm}.json'))
            full = load_checkpoint(f'runs/transfer/{arm}/ckpt.pkl')
            ck, raw = leaves(full['ema_params']), leaves(full['params'])
            tt = next(v for k, v in ck.items() if 'task_table' in k)
            out['task_rows_nonzero'] = [int(i) for i in np.nonzero(np.abs(tt).max(1) > 0)[0]]
            if cfg.get('init_params_from'):
                for tag, tree in (('', ck), ('_raw', raw)):   # EMA average, then the raw (optimised) parameters
                    worst = 0.0
                    for k, v in tree.items():
                        ax = [i for i, n in enumerate(v.shape) if n == args.z_max]
                        if v.ndim >= 2 and ax:
                            worst = max(worst, float(np.abs(np.take(v, absent, ax[0]) - np.take(pre[k], absent, ax[0])).max()))
                    out['absent_rows_max_change' + tag] = worst
                pats = cfg.get('train_only')
                if pats:
                    fz = [k for k in ck if not any(re.search(p, k) for p in pats)]
                    out['frozen_leaves'] = len(fz)
                    out['frozen_identical'] = bool(all(np.array_equal(ck[k], pre[k]) for k in fz))
                    out['trainable_changed'] = int(sum(not np.array_equal(ck[k], pre[k]) for k in ck if k not in fz))
            print('    ', {k: v for k, v in out.items() if k not in ('omat', 'omol')})
        res['arms'][name] = out
    json.dump(res, open(args.out, 'w'), indent=1)
    print('written', args.out)


if __name__ == '__main__':
    main()
