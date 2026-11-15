#!/bin/bash
# score_sample.sh <TAG> [<ckpt>.species_refit.json]: discovery CSV + sample metrics for runs/mbd/<TAG>/relaxed
# (login-node light: ~3000 structures).  With a refit file the per-element energy correction is applied and the
# outputs get the suffix _refit (the uncorrected ones are kept).
cd /scratch/t510a13/race_v2/work
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
T=$1; RF=${2:-}; SFX=${RF:+_refit}; W=/scratch/t510a13/probabilistic_symmetrization/bundle/scratch/t589a02/mp_data/wbm; PJ=/scratch/t510a13/envs/bam_jax/bin/python
$PJ scripts/eval/mbd/make_csv.py --relaxed runs/mbd/$T/relaxed --output runs/mbd/$T/discovery$SFX.csv.gz --model-name racev2 --refs $W/mp_elemental_ref_per_atom.npz --mp2020-cse $W/2022-10-19-wbm-computed-structure-entries.jsonl.gz --details runs/mbd/$T/details$SFX.csv.gz ${RF:+--species-refit $RF} 2>&1 | grep -E '^\[csv\] (kept|species)'
$PJ -I - <<PY
import pandas as pd, numpy as np, json
s = pd.read_csv('$W/2023-12-13-wbm-summary.csv.gz').set_index('material_id')
d = s.join(pd.read_csv('runs/mbd/$T/discovery$SFX.csv.gz').set_index('material_id'), how='inner')
det = pd.read_csv('runs/mbd/$T/details$SFX.csv.gz').set_index('material_id')
hull = d['e_above_hull_mp2020_corrected_ppd_mp']; err = d['e_form_per_atom_racev2'] - d['e_form_per_atom_mp2020_corrected']; each = hull + err
out = {}
for name, m in (('all', np.ones(len(d), bool)), ('unique_prototypes', d['unique_prototype'].values.astype(bool))):
    e, h, ep = err[m], hull[m], each[m]
    tp = ((h <= 0) & (ep <= 0)).sum(); fp = ((h > 0) & (ep <= 0)).sum(); fn = ((h <= 0) & (ep > 0)).sum()
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    out[name] = dict(n=int(m.sum()), MAE=round(float(e.abs().mean()), 4), RMSE=round(float(np.sqrt((e**2).mean())), 4), mean_error=round(float(e.mean()), 4),
                     R2=round(float(1 - (e**2).sum() / ((h - h.mean())**2).sum()), 3),   # E above hull, as in Matbench Discovery
                     F1=round(float(2*pr*rc/max(pr+rc, 1e-9)), 3), precision=round(float(pr), 3), recall=round(float(rc), 3), DAF=round(float(pr/max((h<=0).mean(), 1e-9)), 2))
    print(name, out[name])
step = d.index.str.split('-').str[1]
print('MAE by WBM step:', {k: round(float(err[step == k].abs().mean()), 4) for k in sorted(set(step))}, ' converged', int(det['converged'].sum()), 'of', len(det), ' mean steps', round(float(det['n_steps'].mean()), 1))
json.dump(out, open('runs/mbd/$T/sample_metrics$SFX.json', 'w'), indent=1)
PY
