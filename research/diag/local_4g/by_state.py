"""Errors per charge state / subset of the local 4G-HDNNP runs, raw and after a per-composition gauge refit.

usage: python -I research/diag/local_4g/by_state.py   (from the repo root, after queue.sh and dump_frame_errors on the
       training subset and test.xyz of each run, see docs/local_test_report.md section 7)

Gauge refit: least squares of the TRAINING residuals (e_pred - e_ref) on the element counts (minimum norm), applied
to the test frames. It is the per-composition version of the trainer's single per-atom gauge_refit and uses no test
labels. For Ag3 (one element) it is a single shift, so it cannot remove a charge-state error.
"""
import numpy as np
import ase.io

SYSTEMS = [('ag_cluster', 1000), ('carbon_chain', 1000), ('nacl', 1000), ('aumgo', 100)]
for s, sub in SYSTEMS:
    lab = np.array([a.info['subset'] for a in ase.io.read(f'data_local/4ghdnnp/{s}/test.xyz', ':')])
    print(f'== {s}')
    for t in ['racelr', 'race']:
        tr = np.load(f'runs/local_4g/{s}_{t}/frames_train_sub{sub}.npz')
        te = np.load(f'runs/local_4g/{s}_{t}/frames_test.npz')
        c, *_ = np.linalg.lstsq(tr['counts'].astype(float), tr['e_pred'] - tr['e_ref'], rcond=None)
        e0 = (te['e_pred'] - te['e_ref']) * 1e3
        e1 = e0 - te['counts'] @ c * 1e3
        f = te['f_rmse'] * 1e3
        print(f'  {t:7s} all: E RMSE raw {np.sqrt((e0**2).mean()):7.1f} -> refit {np.sqrt((e1**2).mean()):7.1f} meV, '
              f'MAE refit {np.abs(e1).mean():6.1f} meV, F RMSE {np.sqrt((f**2).mean()):7.1f} meV/A')
        for k in sorted(set(lab)):
            m = lab == k
            print(f'          {k:22s} E RMSE raw {np.sqrt((e0[m]**2).mean()):7.1f} refit {np.sqrt((e1[m]**2).mean()):7.1f} '
                  f'(bias {e1[m].mean():+7.1f}) meV, F RMSE {np.sqrt((f[m]**2).mean()):7.1f} meV/A, '
                  f'max frame F RMSE {f[m].max():8.1f}')
