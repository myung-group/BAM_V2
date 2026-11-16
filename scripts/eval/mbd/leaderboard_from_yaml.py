"""Leaderboard values of every model from the model YAML files of a matbench-discovery clone, with the combined
scores recomputed by the site formulas (CPS; CMDS and CDS as in md_diatomics.py).

    python -I leaderboard_from_yaml.py <matbench-discovery clone> <out.json>

Per model: discovery metrics on the unique prototypes (F1, DAF, Precision, Accuracy, MAE, R2), RMSD (symprec 1e-5),
kappa SRME, CPS = 0.5 F1 + 0.4 (1 - kSRME / 2) + 0.1 (1 - RMSD / 0.15), the MD metrics with CMDS and the diatomics
metrics with the four pillars and CDS.  A value that the YAML file does not hold is null.  Light (74 small files).
"""
import glob
import json
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import md_diatomics as X  # noqa: E402


def num(v):
    return float(v) if isinstance(v, (int, float)) and v == v else None


def get(d, *keys):
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def main():
    repo, out = sys.argv[1], sys.argv[2]
    rows = []
    for f in sorted(glob.glob(f'{repo}/models/*/*.yml')):
        y = yaml.safe_load(open(f))
        if not isinstance(y, dict) or not isinstance(y.get('metrics'), dict):
            continue
        m = y['metrics']
        r = {'name': y.get('model_name'), 'key': y.get('model_key'), 'params': num(y.get('model_params')),
             'targets': y.get('targets'), 'file': str(Path(f).relative_to(repo))}
        for k, yk in (('F1', 'F1'), ('DAF', 'DAF'), ('Prec', 'Precision'), ('Acc', 'Accuracy'), ('MAE', 'MAE'), ('R2', 'R2')):
            r[k] = num(get(m, 'discovery', 'unique_prototypes', yk))
        r['RMSD'] = num(get(m, 'geo_opt', 'symprec=1e-5', 'rmsd'))
        r['kSRME'] = num(get(m, 'phonons', 'kappa_103', 'κ_SRME'))
        r['CPS'] = None if None in (r['F1'], r['RMSD'], r['kSRME']) else \
            0.5 * r['F1'] + 0.4 * (1 - r['kSRME'] / 2) + 0.1 * (1 - min(r['RMSD'], 0.15) / 0.15)
        md = m.get('md') if isinstance(m.get('md'), dict) else {}
        v = {k: num(md.get(k)) for k in ('vdos_error', 'adf_error', 'pressure_error', 'rdf_error')}
        v['run_time_sec'] = num(get(md, 'cost', 'run_time_sec'))
        r['md'] = {**v, 'n_systems': md.get('n_systems'), 'hardware': get(md, 'cost', 'hardware')}
        r['CMDS'] = None if any(v[k] is None for k in X.CMDS_WEIGHTS) else X.cmds(v)
        r['md_speed'] = None if v['run_time_sec'] is None else X.subscore(v['run_time_sec'], **X.CMDS_SPEED)
        di = m.get('diatomics') if isinstance(m.get('diatomics'), dict) else {}
        w = {k: num(di.get(k)) for k in ('pbe_energy_mae', 'pbe_force_mae', 'pbe_wall_dist_mae', 'pbe_bond_length_error',
                                         'pbe_well_depth_error', 'energy_jump', 'force_flips', 'tortuosity')}
        w['run_time_sec'] = num(get(di, 'cost', 'run_time_sec'))
        r['diatomics'] = w
        r['CDS'], r['pillars'] = (None, {}) if any(x is None for x in w.values()) else X.cds(w, 0)
        rows.append(r)
    commit = subprocess.run(['git', '-C', repo, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    json.dump({'source': 'janosh/matbench-discovery models/*/*.yml', 'commit': commit, 'models': rows}, open(out, 'w'), indent=1)
    print(f'{len(rows)} models (commit {commit}) -> {out}')


if __name__ == '__main__':
    main()
