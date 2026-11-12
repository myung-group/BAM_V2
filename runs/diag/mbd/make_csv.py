"""Matbench Discovery discovery CSV from the JSONL results of bam_omat24/relax_wbm.py (2026-11-11).

Same computation as BAM-GAMP-jax scripts/make_submission_csv.py, whose functions are imported: the model's final
energy (uncorrected DFT scale) replaces the energy of WBM's DFT ComputedStructureEntry, the relaxed structure
replaces its structure, MaterialsProject2020Compatibility is applied, and the formation energy per atom follows from
the MP elemental references.  Only the input differs: JSONL files instead of one JSON file per structure.

    python make_csv.py --relaxed <dir with relaxed_*.jsonl> --output <name>.csv.gz --model-name <column prefix> \
        --refs <mp_elemental_ref_per_atom.npz> --mp2020-cse <wbm-computed-structure-entries.jsonl.gz> \
        [--details <details.csv.gz>] [--write-relaxed-json <relaxed.json>] [--scripts <BAM-GAMP-jax/scripts>]
Run with an environment that has pymatgen (e.g. /scratch/t510a13/envs/bam_jax), on a compute or data-mover node.
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--relaxed', required=True); ap.add_argument('--output', required=True)
ap.add_argument('--model-name', default='racev2'); ap.add_argument('--refs', required=True)
ap.add_argument('--mp2020-cse', required=True); ap.add_argument('--details'); ap.add_argument('--write-relaxed-json')
ap.add_argument('--scripts', default='/scratch/t510a13/probabilistic_symmetrization/BAM-GAMP-jax/scripts')
a = ap.parse_args()
sys.path.insert(0, a.scripts)
import make_submission_csv as M  # noqa: E402
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility  # noqa: E402

res = {}
for p in sorted(Path(a.relaxed).glob('relaxed_*.jsonl')):
    with open(p) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                           # a line cut by a kill
            res[str(r['material_id'])] = r         # a structure relaxed twice (resume): the last result
print(f'[csv] {len(res)} relaxed structures from {a.relaxed}')
e_form_fn = M._build_e_form_from_npz(a.refs)
t0 = time.time()
cses = M._load_wbm_cses(a.mp2020_cse, set(res))
compat = MaterialsProject2020Compatibility()
print(f'[csv] {len(cses)} WBM entries loaded, {time.time() - t0:.0f} s')
rows, details, relaxed, n_rej, n_fail = [], [], {}, 0, 0
for mid in sorted(res):
    r = res[mid]
    try:
        if r.get('energy_convention') != 'dft_total_uncorrected':
            raise ValueError(f"energy_convention {r.get('energy_convention')!r}")
        comp = M._composition_from_numbers(r['numbers'])
        e = float(r['final_energy'])
        e_corr = M._mp2020_corrected_energy(cses[mid], e, r, compat)
        if e_corr is None:
            n_rej += 1
            continue
        ef, ef_un = e_form_fn(e_corr, comp), e_form_fn(e, comp)
    except Exception as exc:
        n_fail += 1
        print(f'  {mid}: {type(exc).__name__}: {exc}')
        continue
    rows.append((mid, ef))
    details.append((mid, e, e_corr, ef_un, ef, bool(r.get('converged', False)), int(r.get('n_steps', -1))))
    if a.write_relaxed_json:
        relaxed[mid] = {'numbers': r['numbers'], 'positions': r['final_positions'], 'cell': r['final_cell'],
                        'pbc': [True, True, True]}
out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
with (gzip.open if str(out).endswith('.gz') else open)(out, 'wt') as f:
    f.write(f'material_id,e_form_per_atom_{a.model_name}\n')
    f.writelines(f'{m},{v:.6f}\n' for m, v in rows)
if a.details:
    with (gzip.open if a.details.endswith('.gz') else open)(a.details, 'wt') as f:
        f.write('material_id,energy,energy_mp2020,e_form_uncorrected,e_form,converged,n_steps\n')
        f.writelines(f'{r[0]},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},{r[5]},{r[6]}\n' for r in details)
if a.write_relaxed_json:
    json.dump(relaxed, open(a.write_relaxed_json, 'w'))
print(f'[csv] kept {len(rows)}, MP2020 rejected {n_rej}, failed {n_fail}, converged '
      f'{sum(d[5] for d in details)}, {time.time() - t0:.0f} s -> {out}')
