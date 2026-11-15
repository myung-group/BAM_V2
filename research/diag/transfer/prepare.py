"""Small-scale transfer test OMol25 -> OMat24 (2026-11-10): data and configs of the second stage.

    python -I research/diag/transfer/prepare.py --omol data_local/omol25_openff/sub10k --omat data_local/omat24_val/r300_10k
           --pre research/input/omol/tr_u94_race_cs_amsgrad_f32_s42.json [--out data_local/transfer_omol_omat] [--epochs 40]

Data (extxyz, read as data only; subsets made by scripts/data/convert_omol25.py):
  omol_{train,valid,test}.xyz : the OMol25 molecules in a cubic vacuum box (edge = extent + 2 cutoff + 2 A, pbc true),
                                so that one periodic model reads both datasets; info task = 0
  omat_{train,valid,test}.xyz : the OMat24 structures, info task = 1 (charge 0, spin 1)
  mix_{train,valid}.xyz       : both, shuffled (replay arm)
Energies are written referenced, E - sum_i E0(Z_i) with the E0 of the structure's own dataset (train-only least
squares of each subset, meta.json), so the configs use atom_energies = 0 and a single file can hold both tasks.

Configs research/input/transfer/<arm>.json, derived from the pretraining config --pre (same model configuration; only data,
periodic true, task "task" (info key), avg_n_neighbors and the start differ):
  tr2_scratch : OMat24 only, random start, avg_n_neighbors of OMat24
  tr2_cont    : OMat24 only, init_params_from the pretraining checkpoint (avg_n_neighbors pinned to the pretraining value)
  tr2_replay  : mix (OMol25 task 0 + OMat24 task 1), init_params_from
  tr2_frozen  : OMat24 only, init_params_from, train_only = species tables, task table, shift and scale
Every arm reports both test sets (omat_test, omol_test).
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CUTOFF = 5.0


def referenced(a, e0_by_z, task, box):
    import ase
    from ase.calculators.singlepoint import SinglePointCalculator
    res = a.calc.results
    e = float(res['energy']) - float(sum(e0_by_z[int(z)] for z in a.numbers))
    if box:
        pos = a.positions - a.positions.min(0)
        L = float(pos.max()) + 2 * CUTOFF + 2.0
        b = ase.Atoms(numbers=a.numbers, positions=pos + 0.5 * (L - pos.max(0)), cell=np.eye(3) * L, pbc=True)
        kw = {}
    else:
        b = ase.Atoms(numbers=a.numbers, positions=a.positions, cell=a.cell, pbc=a.pbc)
        kw = {'stress': res['stress']} if 'stress' in res else {}
    b.calc = SinglePointCalculator(b, energy=e, forces=np.asarray(res['forces']), **kw)
    b.info.update(task=int(task), charge=int(a.info.get('charge', 0)), spin=int(a.info.get('spin', 1)))
    return b


def main():
    import ase.io
    ap = argparse.ArgumentParser()
    ap.add_argument('--omol', type=Path, required=True)
    ap.add_argument('--omat', type=Path, required=True)
    ap.add_argument('--pre', type=Path, required=True, help='pretraining config (universal element list)')
    ap.add_argument('--out', type=Path, default=Path('data_local/transfer_omol_omat'))
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    sets, metas = {}, {}
    for name, d, task, box in (('omol', args.omol, 0, True), ('omat', args.omat, 1, False)):
        m = metas[name] = json.load(open(d / 'meta.json'))
        e0 = dict(zip([int(z) for z in m['atomic_numbers']], m['atom_energies']))
        for split in ('train', 'valid', 'test'):
            fr = [referenced(a, e0, task, box) for a in ase.io.read(str(d / f'{split}.xyz'), ':', format='extxyz')]
            sets[name, split] = fr
            ase.io.write(str(args.out / f'{name}_{split}.xyz'), fr, format='extxyz')
            per_atom = np.array([x.get_potential_energy() / len(x) for x in fr])
            print(f'{name}_{split}: {len(fr)} structures, referenced energy per atom mean {per_atom.mean():+.4f} '
                  f'std {per_atom.std():.4f} eV, |F| rms '
                  f'{np.sqrt(np.mean(np.concatenate([x.get_forces().ravel() for x in fr]) ** 2)):.3f} eV/A')
    for split in ('train', 'valid'):
        fr = sets['omol', split] + sets['omat', split]
        fr = [fr[i] for i in rng.permutation(len(fr))]
        ase.io.write(str(args.out / f'mix_{split}.xyz'), fr, format='extxyz')

    pre = json.load(open(args.pre))
    zs = pre['atomic_numbers']
    out = str(args.out)
    base = {k: v for k, v in pre.items() if k not in ('valid_frac', 'note')}
    base.update(atomic_numbers=zs, atom_energies=[0.0] * len(zs), periodic=True, stress_loss=False, task='task',
                gauge_refit=False, n_epochs=args.epochs, eval_test_every=5,
                test_paths=[f'{out}/omat_test.xyz', f'{out}/omol_test.xyz'])
    arms = {
        'tr2_scratch': dict(train_path=f'{out}/omat_train.xyz', valid_path=f'{out}/omat_valid.xyz',
                            avg_n_neighbors=round(metas['omat']['avg_n_neighbors'], 3)),
        'tr2_cont': dict(train_path=f'{out}/omat_train.xyz', valid_path=f'{out}/omat_valid.xyz',
                         init_params_from=pre['ckpt_name']),
        'tr2_replay': dict(train_path=f'{out}/mix_train.xyz', valid_path=f'{out}/mix_valid.xyz',
                           init_params_from=pre['ckpt_name']),
        'tr2_frozen': dict(train_path=f'{out}/omat_train.xyz', valid_path=f'{out}/omat_valid.xyz',
                           init_params_from=pre['ckpt_name'],
                           train_only=['species', 'x_hill', 'readout', 'skip', 'x_linear', 'task_table',
                                       r"\['shift'\]", r"\['scale'\]"]),
    }
    (ROOT / 'input' / 'transfer').mkdir(exist_ok=True)
    for name, kw in arms.items():
        c = dict(base, **kw, log_file=f'runs/transfer/{name}/loss.out', ckpt_name=f'runs/transfer/{name}/ckpt.pkl',
                 note='transfer test OMol25 -> OMat24 (research/diag/transfer/prepare.py): ' + name)
        json.dump(c, open(ROOT / 'input' / 'transfer' / f'{name}.json', 'w'), indent=1)
        print('written', f'research/input/transfer/{name}.json', {k: (v if k != 'init_params_from' else '...') for k, v in kw.items()
                                                         if k not in ('train_path', 'valid_path')})


if __name__ == '__main__':
    main()
