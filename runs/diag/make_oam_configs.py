"""Configs for the OAM subset screening (2026-11-05): RACEv2 final recipe ("v2") vs all new parts removed ("v1"),
one per dataset (omat24, mptrj, salex), from the 3BPA ablation baseline and the subset meta files.

    python -I runs/diag/make_oam_configs.py <subset_dir> [--epochs N] [--tag oam] [--seeds 42]
-> input/input_3bpa_omat24_<tag>_<dataset>_<variant>_s<seed>_cue.json  (name pattern of runs/run_cue.sh)
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'input' / 'input_3bpa_omat24_abl3_full_s42_cue.json'

# everything RACEv2 added on top of RACE V1 (the ablation flags of docs/ablation_protocol_2026-11-05.md, all at once)
V1_FLIPS = {'x_mode': 'species', 'x_norm': None, 'message_norm': 'layer', 'residual': 'none', 'layer_scale': None,
            'update': 'gate', 'readout_mode': 'linear', 'radial_prior': None, 'radial_shells': False,
            'pair_repulsion': None, 'pair_repulsion_switch': None, 'pair_repulsion_switch_width': None,
            'optimizer': 'amsgrad'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('subset_dir', type=Path)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--tag', default='oam')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--datasets', nargs='+', default=['omat24', 'mptrj', 'salex'])
    ap.add_argument('--variants', nargs='+', default=['v2', 'v1'])
    ap.add_argument('--x64', action='store_true', help='float64 model + optimizer (the benchmark setting)')
    ap.add_argument('--stress-weight', type=float, default=10.0)
    ap.add_argument('--layers', type=int, default=None, help='n_layers override (size screening)')
    ap.add_argument('--hidden', type=int, default=None, help='hidden_size override (size screening)')
    ap.add_argument('--species-adam', action='store_true', help='soap_species_adam (species tables by Adam)')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--warmup', type=int, default=None, help='lr_warmup_epochs (default of train_3bpa: 10)')
    ap.add_argument('--optimizer', default=None, help='optimizer override (e.g. amsgrad; default: the recipe SOAP)')
    ap.add_argument('--model', default='race', choices=['race', 'racelr'],
                    help='racelr: RACE_LR with the Qeq stratum (Ewald) and chi0 from Pauling; the OAM data are '
                         'neutral, so no charge / spin input')
    ap.add_argument('--ewald', type=float, nargs=3, default=None, metavar=('ALPHA', 'N_REAL', 'N_RECIP'),
                    help='Ewald settings for --model racelr (required; choose with runs/diag/check_ewald.py)')
    ap.add_argument('--cutoff', type=float, default=5.0,
                    help='!= 5: avg_n_neighbors recomputed on 500 training structures at this cutoff')
    args = ap.parse_args()
    if args.model == 'racelr' and args.ewald is None:
        raise SystemExit('--model racelr needs --ewald ALPHA N_REAL N_RECIP (runs/diag/check_ewald.py)')
    base = json.load(open(BASE))
    for ds in args.datasets:
        meta = json.load(open(args.subset_dir / f'{ds}_meta.json'))
        avg_n = meta['avg_n_neighbors']
        if args.cutoff != 5.0:
            import ase.io
            from ase.neighborlist import neighbor_list
            frames = ase.io.read(meta['train']['path'], index=':500')
            avg_n = sum(len(neighbor_list('i', a, args.cutoff)) / len(a) for a in frames) / len(frames)
        for var in args.variants:
            for seed in args.seeds:
                name = f'{args.tag}_{ds}_{var}{"_lr" if args.model == "racelr" else ""}_s{seed}'
                c = {k: v for k, v in base.items() if not k.startswith('_') and k not in ('valid_frac', 'note')}
                c.update({
                    'train_path': meta['train']['path'], 'valid_path': meta['valid']['path'],
                    'test_paths': [meta['test']['path']],
                    'atomic_numbers': meta['atomic_numbers'], 'atom_energies': meta['atom_energies'],
                    'e0_mode': 'isolated', 'avg_n_neighbors': round(avg_n, 3), 'cutoff': args.cutoff,
                    'periodic': True, 'stress_loss': True, 'stress_weight': args.stress_weight,
                    'x64': False, 'pad_mode': 'fixed', 'pack_max_graphs': 64,
                    'n_epochs': args.epochs, 'lr_schedule_patience': 10, 'eval_test_every': 10,
                    'ckpt_last_every': 5, 'restart': True, 'seed': seed,
                    'log_file': f'runs/{name}_cue/loss_{name}_cue.out',
                    'ckpt_name': f'runs/{name}_cue/ckpt_{name}_cue.pkl',
                    'note': f'OAM subset screening 2026-11-05: {ds} ({meta["train"]["n"]} train), '
                            f'{"RACEv2 final recipe" if var == "v2" else "all RACEv2 additions removed (V1 recipe)"}, float32, '
                            f'E0 = per-element least squares of the full training split, stress eV/A^3 (weight {args.stress_weight:g}), '
                            f'cutoff {args.cutoff:g} A',
                })
                if var == 'v1':
                    c.update(V1_FLIPS)
                if args.layers is not None:
                    c['n_layers'] = args.layers
                if args.hidden is not None:
                    c['hidden_size'] = args.hidden
                    c.pop('hidden_irreps', None)
                if args.species_adam:
                    c['soap_species_adam'] = True
                c['batch_size'] = args.batch
                if args.warmup is not None:
                    c['lr_warmup_epochs'] = args.warmup
                if args.optimizer is not None:
                    c['optimizer'] = args.optimizer
                    c.pop('soap_species_adam', None)
                if args.model == 'racelr':
                    c['model'] = 'RACE_LR'
                    c['long_range'] = {'chi_init': 'pauling', 'chi_init_missing': 'mean',
                                       'ewald_alpha': args.ewald[0], 'ewald_n_real': int(args.ewald[1]),
                                       'ewald_n_recip': int(args.ewald[2])}
                    c['note'] += '; RACE_LR (Qeq + Ewald, chi0 Pauling)'
                if args.x64:
                    c['x64'] = True
                    c['note'] = c['note'].replace('float32', 'float64')
                c = {k: v for k, v in c.items() if v is not None}
                out = ROOT / 'input' / f'input_3bpa_omat24_{name}_cue.json'
                json.dump(c, open(out, 'w'), indent=4)
                print(out.name, 'epochs', args.epochs, 'avg_n', c['avg_n_neighbors'], 'species', len(c['atomic_numbers']))


if __name__ == '__main__':
    main()
