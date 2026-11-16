"""Configs of the OMol25 pretraining pilot (2026-11-06): which charge handling and which optimizer / precision to
use for a RACEv2 pretraining on OMol25, compared at the same budget on a subset made by scripts/data/convert_omol25.py.

    python -I scripts/train/make_omol_configs.py <subset_dir> [--tag omol100k] [--epochs 30] [--layers 5 --hidden 128]
        [--models race race_cs racelr racelr_q] [--opts soap_f64 amsgrad_f32] [--seed 42] [--batch 16]
        [--elements universal|subset] [--n-tasks 8]
-> configs/omol/<tag>[_u94]_<model>_<opt>_s<seed>.json, logs and checkpoints under runs/omol/<same name>/

--elements universal (default since 2026-11-10; name gets "_u94"): atomic_numbers = Z 1..94, so that a model trained
        on OMol25 can later be continued on, or get heads for, OMat24 / MPtrj / Alexandria / OC20 without changing the
        parameter tree (init_params_from needs an identical model configuration).  E0 of the elements that are absent
        from the subset is 0.0 and their rows of the species tables keep their initial values (zero gradient; the
        SOAP arms get soap_species_adam "any_axis" for this, keep weight_decay 0).
        subset: the elements of the subset only (the pilot configs of 2026-11-06).
--n-tasks N (default 8; RACE_LR arms only): reserves the zero-initialised task embedding table (models/race_lr.py,
        long_range "n_tasks").  OMol25 is task 0 (no 'task' key in the data); 0 switches the slot off.  The plain
        "race" arm has no slot.

models  race      RACEv2 (no charge / spin input; the reference)
        race_cs   + total charge and spin embedding (RACE_LR with qeq false)
        racelr    + embedding + Qeq long-range stratum, chi0 from Pauling electronegativities
        racelr_q  racelr + charge loss on the reference partial charges (only if the subset has them; weight 0.1)
opts    soap_f64     the RACEv2 recipe: SOAP, species tables by Adam, float64
        amsgrad_f32  AMSGrad, float32 (HANDOFF optimizer screening: never completed on OAM, check for divergence)
Every arm reads total_charge_key / total_spin_key, so all arms see identical graphs.
Periodic subsets (meta.json "periodic", e.g. convert_omol25.py --allow-periodic on an OMat24-type aselmdb) get
periodic true, the stress loss when the subset has stress (weight --stress-weight, the OAM value 10), and the Ewald
settings --ewald ALPHA N_REAL N_RECIP for the Qeq arms (choose them with scripts/data/check_ewald.py).
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'input' / 'input_3bpa_omat24_abl3_full_s42_cue.json'
Z_MAX = 94     # H .. Pu: OMol25 (Z <= 83) and OMat24 / MPtrj / Alexandria (89 elements up to Pu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('subset_dir', type=Path)
    ap.add_argument('--tag', default='omol100k')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--layers', type=int, default=5)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--lmax', type=int, default=None)
    ap.add_argument('--models', nargs='+', default=['race', 'race_cs', 'racelr', 'racelr_q'])
    ap.add_argument('--opts', nargs='+', default=['soap_f64', 'amsgrad_f32'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--charge-weight', type=float, default=0.1)
    ap.add_argument('--tp-backend', default='cue')
    ap.add_argument('--stress-weight', type=float, default=10.0)
    ap.add_argument('--ewald', type=float, nargs=3, default=None, metavar=('ALPHA', 'N_REAL', 'N_RECIP'),
                    help='Ewald settings of the Qeq arms on periodic data (required there)')
    ap.add_argument('--elements', choices=['universal', 'subset'], default='universal')
    ap.add_argument('--n-tasks', type=int, default=8)
    args = ap.parse_args()
    meta = json.load(open(args.subset_dir / 'meta.json'))
    base = json.load(open(BASE))
    has_q = bool(meta.get('ref_charges_key'))
    periodic = bool(meta.get('periodic', False))
    if periodic and args.ewald is None and any(m.startswith('racelr') for m in args.models):
        raise SystemExit('periodic subset: give --ewald ALPHA N_REAL N_RECIP (scripts/data/check_ewald.py)')
    numbers, e0 = [int(z) for z in meta['atomic_numbers']], [float(e) for e in meta['atom_energies']]
    if args.elements == 'universal':
        if max(numbers) > Z_MAX:
            raise SystemExit(f'subset has Z = {max(numbers)} > {Z_MAX}')
        by_z = dict(zip(numbers, e0))
        numbers = list(range(1, Z_MAX + 1))
        e0 = [by_z.get(z, 0.0) for z in numbers]
    tag = args.tag + ('_u94' if args.elements == 'universal' else '')
    (ROOT / 'input' / 'omol').mkdir(exist_ok=True)
    for model in args.models:
        if model == 'racelr_q' and not has_q:
            print('skip racelr_q: the subset has no reference partial charges')
            continue
        for opt in args.opts:
            name = f'{tag}_{model}_{opt}_s{args.seed}'
            c = {k: v for k, v in base.items() if not k.startswith('_') and k not in ('valid_frac', 'note')}
            c.update({
                'train_path': meta['train']['path'], 'valid_path': meta['valid']['path'],
                'test_paths': [meta['test']['path']],
                'atomic_numbers': numbers, 'atom_energies': e0,
                'e0_mode': 'isolated', 'avg_n_neighbors': round(meta['avg_n_neighbors'], 3), 'cutoff': 5.0,
                'periodic': periodic, 'stress_loss': periodic and bool(meta.get('stress', False)),
                'pad_mode': 'fixed', 'pack_max_graphs': 64,
                'batch_size': args.batch, 'lr_warmup_epochs': 2, 'n_epochs': args.epochs,
                'lr_schedule_patience': 10, 'eval_test_every': 10, 'ckpt_last_every': 5, 'restart': True,
                'seed': args.seed, 'n_layers': args.layers, 'hidden_size': args.hidden, 'tp_backend': args.tp_backend,
                'total_charge_key': 'charge', 'total_spin_key': 'spin',
                'log_file': f'runs/omol/{name}/loss.out', 'ckpt_name': f'runs/omol/{name}/ckpt.pkl',
            })
            if args.lmax is not None:
                c['lmax'] = args.lmax
            if c['stress_loss']:
                c['stress_weight'] = args.stress_weight
            if model == 'race':
                c['model'] = 'RACE'
            else:
                c['model'] = 'RACE_LR'
                lr = {'charge_spin_embedding': True}
                if args.n_tasks:
                    lr['n_tasks'] = args.n_tasks
                if model == 'race_cs':
                    lr['qeq'] = False
                else:
                    lr.update(chi_init='pauling', chi_init_missing='mean')
                    if periodic:
                        lr.update(ewald_alpha=args.ewald[0], ewald_n_real=int(args.ewald[1]),
                                  ewald_n_recip=int(args.ewald[2]))
                c['long_range'] = lr
                if model == 'racelr_q':
                    c['charge_label_key'] = 'ref_charges'
                    c['charge_weight'] = args.charge_weight
            if opt == 'soap_f64':
                # universal list: "any_axis", so that SOAP does not rotate the entries of absent elements (x_hill)
                c.update(x64=True, optimizer='soap',
                         soap_species_adam=('any_axis' if args.elements == 'universal' else True))
            elif opt == 'amsgrad_f32':
                c.update(x64=False, optimizer='amsgrad')
                c.pop('soap_species_adam', None)
            else:
                raise ValueError(opt)
            c['note'] = (f'OMol25 pretraining pilot 2026-11-06: {meta["train"]["n"]} train structures (official train '
                         f'split), test = official val split sample; model {model}, {opt}; E0 = train-only least '
                         'squares per element'
                         + (f'; universal element list Z 1..{Z_MAX} (E0 0 for absent elements)'
                            if args.elements == 'universal' else ''))
            c = {k: v for k, v in c.items() if v is not None}
            out = ROOT / 'input' / 'omol' / f'{name}.json'
            json.dump(c, open(out, 'w'), indent=1)
            print(out.relative_to(ROOT), c['model'], c.get('long_range'), c['optimizer'], 'x64', c['x64'])


if __name__ == '__main__':
    main()
