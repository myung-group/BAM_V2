"""Matbench Discovery molecular dynamics (DynaMat v1.0 -> CMDS) and diatomics (-> CDS) with a RACE v2 checkpoint.

The rollouts, the diatomic sweep and all metrics are the benchmark's own code (janosh/matbench-discovery, read at
commit 6d587c1 of 2026-11): matbench_discovery.md.run_md_benchmark, predict_curves / write_results of
models/run_diatomics.py and matbench_discovery.metrics.{md,diatomics}.  This file adds only
  * an ASE calculator around relax_batched.RaceEvaluator (energy, forces and stress; the calculator of
    race_v2.inference.predict has no stress, and the MD pressure metric needs it), and
  * the two combined scores, which the benchmark computes on its web site and not in the package: the formulas and
    constants below are copied from site/src/lib/combined-scores.svelte.ts and site/src/lib/labels.ts.
The model is not registered in the benchmark, so its curated per-model diatomic exclusions do not exist: curves the
model cannot give (unknown element, non-finite value in the scored range) are counted as excluded, as the benchmark
does for run-discovered exclusions.

Environment: Python >= 3.14 (the benchmark code needs it) with the model stack and the benchmark's dependencies;
MBD_REPO = clone of janosh/matbench-discovery; MBD_CACHE_DIR = folder with md/2026-06-29-dynamat-v1.0-reference-
trajectories.h5 and diatomics/diatomics-dft.json.gz; BAM_PREDICT_X64=1 (float64 inference, the benchmark default).

    python md_diatomics.py md        --ckpt <pkl> --out-dir <dir> [--systems A B ...] [--dry-run]
    python md_diatomics.py score-md  --out-dir <dir>               # all per-system CSVs of <dir> -> metrics, CMDS
    python md_diatomics.py diatomics --ckpt <pkl> --out-dir <dir> [--dry-run]
    python md_diatomics.py systems                                 # names of the reference systems
"""
import argparse
import glob
import gzip
import importlib.util
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
MBD_REPO = os.environ.get('MBD_REPO')
KEY = 'race_v2_oam'

# site/src/lib/labels.ts and combined-scores.svelte.ts (commit 6d587c1)
CMDS_WEIGHTS = {'vdos_error': 0.3, 'adf_error': 0.2, 'run_time_sec': 0.2, 'pressure_error': 0.3}
CMDS_SPEED = dict(baseline=300_000, floor=9000, log=True)
CDS_WEIGHTS = {'accuracy': 4 / 9, 'geometry': 2 / 9, 'speed': 1 / 9, 'physicality': 2 / 9}
CDS_COMPONENTS = {
    'accuracy': [('pbe_energy_mae', 5 / 9, 4, 0, False), ('pbe_force_mae', 4 / 9, 5, 0, False)],
    'geometry': [('pbe_wall_dist_mae', 1 / 2, 0.5, 0, False), ('pbe_bond_length_error', 1 / 3, 0.8, 0, False),
                 ('pbe_well_depth_error', 1 / 6, 3.5, 0, False)],
    'speed': [('run_time_sec', 1, 5000, 50, True)],
    'physicality': [('energy_jump', 2 / 5, 3, 0, False), ('force_flips', 2 / 5, 3.5, 1, False),
                    ('tortuosity', 1 / 5, 2, 1, False)]}
N_SCORED_DIATOMIC_ELEMENTS = 87


def subscore(value, baseline, floor, log):
    v, lo, hi = (math.log10(value), math.log10(floor), math.log10(baseline)) if log else (value, floor, baseline)
    return min(1.0, max(0.0, 1 - (v - lo) / (hi - lo)))


def cmds(values):
    """calculate_cmds: weighted mean of 1 - error/100 (vDOS, ADF, pressure) and the log-clamped run time."""
    tot = 0.0
    for k, w in CMDS_WEIGHTS.items():
        v = values.get(k)
        if v is None or not math.isfinite(v) or v < 0 or (k == 'run_time_sec' and v <= 0):
            return None
        tot += w * (subscore(v, **CMDS_SPEED) if k == 'run_time_sec' else max(0.0, 1 - v / 100))
    return tot / sum(CMDS_WEIGHTS.values())


def cds(values, n_excluded):
    """calculate_cds: weighted mean of the pillar scores times the fraction of the scored elements completed."""
    tot, pillars = 0.0, {}
    for pillar, w in CDS_WEIGHTS.items():
        s = 0.0
        for key, cw, baseline, floor, log in CDS_COMPONENTS[pillar]:
            v = values.get(key)
            if v is None or not math.isfinite(v) or v < 0 or (log and v <= 0):
                return None, pillars
            s += cw * subscore(v, baseline, floor, log)
        pillars[pillar] = s
        tot += w * s
    return tot / sum(CDS_WEIGHTS.values()) * max(0.0, 1 - n_excluded / N_SCORED_DIATOMIC_ELEMENTS), pillars


def _paths():
    """The benchmark clone, the folder that holds race_v2 and the top-level modules of race_v2 (as kappa.py)."""
    if not MBD_REPO or not (Path(MBD_REPO) / 'matbench_discovery').is_dir():
        raise SystemExit('set MBD_REPO to a clone of janosh/matbench-discovery')
    pkg = None
    for base in (Path.cwd(), *HERE.parents):
        if (base / 'race_v2' / 'inference' / 'relax_batched.py').exists():
            pkg = base / 'race_v2'
            break
    if pkg is None:
        raise SystemExit('race_v2 not found: run from the work folder')
    for d in (MBD_REPO, str(pkg.parent), str(pkg / 'inference'), str(pkg / 'data')):
        if d not in sys.path:
            sys.path.insert(0, d)


def make_calculator(ckpt):
    from ase.calculators.calculator import Calculator, all_changes
    import relax_batched as RB

    class RaceV2Calculator(Calculator):
        """energy / forces / stress of one structure from relax_batched.RaceEvaluator (total energy = model + E0)."""
        implemented_properties = ['energy', 'free_energy', 'forces', 'stress']

        def __init__(self, ckpt_path, **kw):
            super().__init__(**kw)
            self.ev = RB.RaceEvaluator(ckpt_path, max_graphs=2)
            self.prepare = self.ev.prepare_fn()
            self.n_calls = 0

        def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            e, f, s = self.ev.evaluate([self.prepare(self.atoms)])[0]
            self.n_calls += 1
            self.results = {'energy': e, 'free_energy': e, 'forces': f, 'stress': s}

    t0 = time.perf_counter()
    calc = RaceV2Calculator(ckpt)
    if calc.ev.dtype != np.float64 and os.environ.get('ALLOW_FLOAT32') != '1':
        raise SystemExit('the evaluator is not float64: set BAM_PREDICT_X64=1 (or ALLOW_FLOAT32=1 to go on)')
    from matbench_discovery.hpc import detect_runtime, package_versions
    runtime = {'cost_source': 'runner', **detect_runtime(calc, requested_device='cuda'),
               'dtype': 'float64' if calc.ev.dtype == np.float64 else 'float32', 'acceleration': [],
               'load_time_sec': round(time.perf_counter() - t0, 2), 'n_processes': 1, 'hostname': socket.gethostname()}
    for k in ('SLURM_JOB_ID', 'SLURM_ARRAY_TASK_ID'):
        if os.getenv(k):
            runtime[k.casefold()] = os.environ[k]
    runtime['versions'] = package_versions(('ase', 'numpy', 'pymatgen', 'jax', 'scipy', 'flax', 'e3nn-jax',
                                            'cuequivariance-jax', 'cuequivariance-ops-jax-cu12', 'vesin'))
    print(f"calculator: {ckpt}  cutoff {calc.ev.cutoff}  dtype {runtime['dtype']}  {runtime['hardware']} "
          f"({runtime['device']})  neighbours {os.environ.get('BAM_NEIGHBORS', 'ase')}  load {runtime['load_time_sec']} s",
          flush=True)
    return calc, runtime


def cmd_md(a):
    from matbench_discovery.md import default_md_reference_path, run_md_benchmark
    calc, runtime = make_calculator(a.ckpt)
    t0 = time.perf_counter()
    df = run_md_benchmark(calculator=calc, model_key=a.key, out_dir=a.out_dir, ref_file=a.ref_file,
                          systems=a.systems, n_steps=a.n_steps, time_step_fs=0.25, record_interval=10, seed=0,
                          dry_run=a.dry_run, runtime=runtime)
    dt = time.perf_counter() - t0
    print(df.T.to_string())
    print(f"model calls {calc.n_calls}, wall {dt:.1f} s ({1e3 * dt / max(calc.n_calls, 1):.1f} ms per call including "
          f"compilation and metrics), padded shapes {sorted(calc.ev.shapes)}  reference {a.ref_file or default_md_reference_path()}")


def cmd_score_md(a):
    import pandas as pd
    from matbench_discovery.md import default_md_reference_path, list_reference_systems
    from matbench_discovery.metrics import md as M
    files = sorted(glob.glob(f'{a.out_dir}/*-{a.key}-md-metrics*.csv.gz'))
    df = pd.concat([pd.read_csv(f).set_index('system') for f in files])
    df = df[~df.index.duplicated(keep='last')]
    ref = set(list_reference_systems(a.ref_file or default_md_reference_path()))
    problems = M.coverage_problems(df.index, ref)
    m = M.calc_md_metrics(df)
    cost = m.get('cost') if isinstance(m.get('cost'), dict) else m
    vals = {k: m.get(k) for k in ('vdos_error', 'adf_error', 'pressure_error')}
    vals['run_time_sec'] = cost.get('run_time_sec')
    out = {'model': a.key, 'files': files, 'n_systems': int(len(df)), 'coverage_problems': problems, 'metrics': m,
           'cmds_components': vals, 'CMDS': None if problems else cmds(vals),
           'CMDS_note': 'site formula of commit 6d587c1; speed is the summed rollout time on this hardware'}
    print(df.to_string())
    print(json.dumps(out, indent=1, default=str))
    json.dump(out, open(f'{a.out_dir}/md_summary_{a.key}.json', 'w'), indent=1, default=str)


def cmd_diatomics(a):
    from datetime import UTC, datetime
    from ase.data import chemical_symbols
    from matbench_discovery.diatomics import generate_diatomics
    from matbench_discovery.hpc import merge_audit_metadata
    from matbench_discovery.metrics import diatomics as D
    from matbench_discovery.metrics.diatomics.exclusions import is_non_mp_formula
    spec = importlib.util.spec_from_file_location('mbd_run_diatomics', f'{MBD_REPO}/models/run_diatomics.py')
    R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
    calc, runtime = make_calculator(a.ckpt)
    max_z, n_points = (3, 10) if a.dry_run else (92, 119)
    z_values = list(range(1, max_z + 1))
    distances = np.geomspace(0.1, 6.0, n_points)
    calc.get_potential_energy(generate_diatomics('H', 'H', distances[-1:])[0])     # compilation outside the timing
    t0 = time.perf_counter()
    curves, invalid = R.predict_curves(calc, a.key, z_values, distances)
    run_time = round(time.perf_counter() - t0, 2)
    expected = {f'{chemical_symbols[z]}-{chemical_symbols[z]}' for z in z_values}
    missing = sorted(f for f in expected - set(curves) if not is_non_mp_formula(f))
    reasons = {f: 'invalid or unsupported curve' for f in sorted(set(missing) | set(invalid)) if not is_non_mp_formula(f)}
    meta = merge_audit_metadata([{**runtime, 'run_time_sec': run_time, 'n_shards': 1,
                                  'completed_at': datetime.now(UTC).isoformat()}], strict=True)
    meta['excluded_formula_reasons'] = reasons
    os.makedirs(a.out_dir, exist_ok=True)
    tag = '-dry-run' if a.dry_run else ''
    json_path = f'{a.out_dir}/diatomics{tag}-{a.key}.json.gz'
    results = R.write_results(json_path, curves, distances, meta)
    print(f'{len(curves)}/{len(z_values)} curves in {run_time} s -> {json_path}; excluded {reasons}', flush=True)
    ref = D.load_dft_reference_curves('PBE')
    pred = D.DiatomicCurves.from_dict(results)
    per_elem = D.calc_diatomic_metrics(ref, pred, interpolate=200)
    per_elem = {k: v for k, v in per_elem.items() if k not in reasons and f'{k}-{k}' not in reasons}
    means = D.diatomic_metric_means(per_elem)
    vals = {**means, 'run_time_sec': run_time}
    score, pillars = cds(vals, len(reasons))
    out = {'model': a.key, 'n_curves': len(curves), 'n_scored_elements': len(per_elem), 'excluded': reasons,
           'mean_metrics': means, 'run_time_sec': run_time, 'pillars': pillars, 'CDS': score,
           'CDS_note': 'site formula of commit 6d587c1; meaningless for --dry-run' if a.dry_run else
                       'site formula of commit 6d587c1; speed on this hardware'}
    print(json.dumps(out, indent=1, default=str))
    json.dump({**out, 'per_element': per_elem}, open(f'{a.out_dir}/diatomics_summary{tag}_{a.key}.json', 'w'),
              indent=1, default=str)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('md', 'score-md', 'diatomics', 'systems'):
        p = sub.add_parser(name)
        p.add_argument('--key', default=KEY, help='model key in the output file names')
        p.add_argument('--out-dir'); p.add_argument('--ckpt'); p.add_argument('--ref-file')
        p.add_argument('--dry-run', action='store_true')
        if name == 'md':
            p.add_argument('--systems', nargs='*'); p.add_argument('--n-steps', type=int, default=80_000)
    a = ap.parse_args()
    _paths()
    if a.cmd == 'systems':
        from matbench_discovery.md import default_md_reference_path, list_reference_systems
        print('\n'.join(list_reference_systems(a.ref_file or default_md_reference_path())))
        return
    if not a.out_dir or (a.cmd != 'score-md' and not a.ckpt):
        ap.error('--out-dir (and --ckpt for md / diatomics) are needed')
    {'md': cmd_md, 'score-md': cmd_score_md, 'diatomics': cmd_diatomics}[a.cmd](a)


if __name__ == '__main__':
    main()
