"""Geometry-optimisation RMSD of relaxed WBM structures against the DFT-relaxed ones, as Matbench Discovery does it
(matbench_discovery/structure/symmetry.py and metrics/geo_opt.py, read 2026-11-13):
    StructureMatcher(stol=1.0, scale=False).get_rms_dist(pred, ref)[0]   (pymatgen: normalised by (V/N)^(1/3)),
structures that do not match count as 1.0, plain mean over all structures.

    python rmsd_sample.py --relaxed <dir with relaxed_*.jsonl> --mp2020-cse <wbm-computed-structure-entries.jsonl.gz> \
        --summary <wbm-summary.csv.gz> [--limit N] [--output rmsd.csv.gz]
Needs pymatgen (e.g. ~/.conda/envs/bam_jax); single thread, about 3k structures on a login node."""
import argparse, gzip, json, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--relaxed', required=True); ap.add_argument('--mp2020-cse', required=True)
ap.add_argument('--summary', required=True); ap.add_argument('--limit', type=int); ap.add_argument('--output')
ap.add_argument('--scripts', default='/home01/t589a02/prog/BAM-GAMP-jax/scripts')
a = ap.parse_args()
sys.path.insert(0, a.scripts)
import make_submission_csv as M  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pymatgen.analysis.structure_matcher import StructureMatcher  # noqa: E402
from pymatgen.core import Lattice, Structure  # noqa: E402

res = {}
for p in sorted(Path(a.relaxed).glob('relaxed_*.jsonl')):
    for line in open(p):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        res[str(r['material_id'])] = r
ids = sorted(res)[:a.limit] if a.limit else sorted(res)
cses = M._load_wbm_cses(a.mp2020_cse, set(ids))
sm = StructureMatcher(stol=1.0, scale=False)
t0 = time.time(); rows = []
for mid in ids:
    r = res[mid]; e = cses[mid]
    ref = e.structure if hasattr(e, 'structure') else Structure.from_dict(e['structure'])
    pred = Structure(Lattice(r['final_cell']), r['numbers'], r['final_positions'], coords_are_cartesian=True)
    try:
        out = sm.get_rms_dist(pred, ref)
    except Exception as exc:                                   # counted as unmatched
        print(f'  {mid}: {type(exc).__name__}: {exc}'); out = None
    rows.append((mid, float(out[0]) if out else np.nan, float(out[1]) if out else np.nan))
d = pd.DataFrame(rows, columns=['material_id', 'rmsd', 'max_dist']).set_index('material_id')
u = pd.read_csv(a.summary, usecols=['material_id', 'unique_prototype']).set_index('material_id')['unique_prototype']
d['unique_prototype'] = u.reindex(d.index).astype(bool)
for name, x in (('all', d), ('unique_prototypes', d[d.unique_prototype])):
    print(name, {'n': len(x), 'unmatched': int(x.rmsd.isna().sum()), 'RMSD': round(float(x.rmsd.fillna(1.0).mean()), 4),
                 'RMSD_of_matched': round(float(x.rmsd.mean()), 4), 'median': round(float(x.rmsd.median()), 4)})
print(f'{len(d)} structures, {time.time() - t0:.0f} s')
if a.output:
    d.to_csv(a.output)
