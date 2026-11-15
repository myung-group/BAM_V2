"""Universal element list (make_omol_configs.py --elements universal): are the entries of absent elements and the
task rows 1.. untouched by training?  Compares two checkpoints of the same config and seed with different numbers of
epochs (same initial parameters): the entries of absent elements must be bit-identical, task rows 1.. exactly 0.

    python runs/diag/omol_universal/check_untouched.py <ckpt_a.pkl> <ckpt_b.pkl> --present 1 6 7 8 9 16 17
Result 2026-11-10 (OpenFF 10k subset, L3x64, 1 vs 2 epochs): AMSGrad f32 and SOAP f64 with soap_species_adam
"any_axis" pass; SOAP with soap_species_adam true moved x_hill (species on the last axis) of absent elements by 0.11.
"""
import argparse
import pickle
import numpy as np
import jax


def leaves(tree):
    return {jax.tree_util.keystr(k): np.asarray(v) for k, v in jax.tree_util.tree_leaves_with_path(tree)}


ap = argparse.ArgumentParser()
ap.add_argument('ckpt_a'); ap.add_argument('ckpt_b')
ap.add_argument('--present', type=int, nargs='+', required=True, help='atomic numbers present in the training data')
ap.add_argument('--z-max', type=int, default=94)
args = ap.parse_args()
PRESENT, NZ = args.present, args.z_max
a, b = pickle.load(open(args.ckpt_a, 'rb')), pickle.load(open(args.ckpt_b, 'rb'))
print('epochs', a.get('epoch'), b.get('epoch'))
bad = 0
if True:
    for key in ('params', 'ema_params'):
        la, lb = leaves(a[key]), leaves(b[key])
        absent = np.array([z - 1 for z in range(1, NZ + 1) if z not in PRESENT])
        present = np.array([z - 1 for z in PRESENT])
        n_tab = 0
        for k in lb:
            x, y = la[k], lb[k]
            if 'task_table' in k:
                bad += int(np.abs(y[1:]).max() != 0.0)
                print(f'  {key} {k} shape {y.shape}: |row 0| max {np.abs(y[0]).max():.3e}, |rows 1..| max '
                      f'{np.abs(y[1:]).max():.1e}')
            for ax in range(y.ndim):
                if y.shape[ax] == NZ:
                    n_tab += 1
                    d_abs = np.abs(np.take(x, absent, ax) - np.take(y, absent, ax)).max()
                    bad += int(d_abs != 0.0)
                    d_pre = np.abs(np.take(x, present, ax) - np.take(y, present, ax)).max()
                    print(f'  {key} {k} shape {y.shape} axis {ax}: absent rows max |diff| {d_abs:.1e}, present rows '
                          f'{d_pre:.2e}')
        print(f'  {key}: {n_tab} species-indexed axes, {sum(v.size for v in lb.values())} parameters')
print('PASS: absent elements and unused task rows untouched' if not bad else f'FAIL: {bad} tables changed')
