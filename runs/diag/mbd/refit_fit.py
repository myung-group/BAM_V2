"""Per-element energy refit from model energies of training frames (JAX-free).  Input: the extxyz of
refit_subset.py (reference energies) and the JSONL written by bam_omat24/relax_wbm.py run on it with a huge fmax
(no relaxation: initial_energy = model energy of the training geometry).  Least squares of (E_model - E_ref) on the
element counts, minimum norm; writes <ckpt>.species_refit.json in the format of bam_omat24/predict.py
(delta_eV is ADDED per atom of each element to the predicted total energy).

    python -I refit_fit.py --xyz <subset.extxyz> --relaxed <dir with relaxed_*.jsonl> --ckpt <ckpt.pkl> --atomic-numbers-from <config.json>"""
import argparse, json
from pathlib import Path
import numpy as np
import ase.io

ap = argparse.ArgumentParser()
ap.add_argument('--xyz', required=True); ap.add_argument('--relaxed', required=True); ap.add_argument('--ckpt', required=True)
ap.add_argument('--atomic-numbers-from', required=True)
a = ap.parse_args()
zs = [int(z) for z in json.load(open(a.atomic_numbers_from))['atomic_numbers']]
ref = {str(x.info['material_id']): float(x.info['ref_energy']) for x in ase.io.iread(a.xyz, format='extxyz')}
rows = {}
for p in sorted(Path(a.relaxed).glob('relaxed_*.jsonl')):
    for line in open(p):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[str(r['material_id'])] = r
ids = sorted(set(rows) & set(ref))
X = np.array([[sum(1 for z in rows[i]['numbers'] if z == zz) for zz in zs] for i in ids], float)
n = X.sum(1)
res = np.array([rows[i]['initial_energy'] - ref[i] for i in ids])
steps = np.array([rows[i]['n_steps'] for i in ids])
coef, _, rank, _ = np.linalg.lstsq(X, res, rcond=None)
after = res - X @ coef
pa = lambda v: v / n * 1e3                      # meV/atom
out = dict(atomic_numbers=zs, delta_eV=[float(-c) for c in coef], rank=int(rank), n_frames=len(ids),
           mean_residual_meV_per_atom=float(pa(res).mean()), mae_before_meV_per_atom=float(np.abs(pa(res)).mean()),
           mae_after_meV_per_atom=float(np.abs(pa(after)).mean()),
           rmse_before_eV=float(np.sqrt(np.mean(res ** 2))), rmse_after_eV=float(np.sqrt(np.mean(after ** 2))),
           note='training-split refit (random MPtrj training frames, model energy at the training geometry); '
                'delta is added per atom of each element to the predicted total energy')
dst = Path(a.ckpt).with_suffix('.species_refit.json')
json.dump(out, open(dst, 'w'), indent=1)
d = np.array(out['delta_eV']) * 1e3; present = X.sum(0) > 0
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if k not in ('atomic_numbers', 'delta_eV', 'note')})
print(f'frames {len(ids)} of {len(ref)} (max n_steps {steps.max()}), elements present {int(present.sum())} of {len(zs)}')
print('delta meV/atom: mean |delta| %.2f, max |delta| %.2f at Z=%d; largest:' % (np.abs(d[present]).mean(), np.abs(d).max(), zs[int(np.abs(d).argmax())]),
      ', '.join(f'Z{zs[i]} {d[i]:+.1f}' for i in np.argsort(-np.abs(d))[:10]))
print('->', dst)
