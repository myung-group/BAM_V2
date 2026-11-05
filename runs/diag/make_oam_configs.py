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
    args = ap.parse_args()
    base = json.load(open(BASE))
    for ds in args.datasets:
        meta = json.load(open(args.subset_dir / f'{ds}_meta.json'))
        for var in args.variants:
            for seed in args.seeds:
                name = f'{args.tag}_{ds}_{var}_s{seed}'
                c = {k: v for k, v in base.items() if not k.startswith('_') and k not in ('valid_frac', 'note')}
                c.update({
                    'train_path': meta['train']['path'], 'valid_path': meta['valid']['path'],
                    'test_paths': [meta['test']['path']],
                    'atomic_numbers': meta['atomic_numbers'], 'atom_energies': meta['atom_energies'],
                    'e0_mode': 'isolated', 'avg_n_neighbors': round(meta['avg_n_neighbors'], 3),
                    'periodic': True, 'stress_loss': True, 'stress_weight': 10.0,
                    'x64': False, 'batch_size': 16, 'pad_mode': 'fixed', 'pack_max_graphs': 64,
                    'n_epochs': args.epochs, 'lr_schedule_patience': 10, 'eval_test_every': 10,
                    'ckpt_last_every': 5, 'restart': True, 'seed': seed,
                    'log_file': f'runs/{name}_cue/loss_{name}_cue.out',
                    'ckpt_name': f'runs/{name}_cue/ckpt_{name}_cue.pkl',
                    'note': f'OAM subset screening 2026-11-05: {ds} ({meta["train"]["n"]} train), '
                            f'{"RACEv2 final recipe" if var == "v2" else "all RACEv2 additions removed (V1 recipe)"}, float32, '
                            'E0 = per-element least squares of the full training split, stress eV/A^3 (weight 10)',
                })
                if var == 'v1':
                    c.update(V1_FLIPS)
                if args.x64:
                    c['x64'] = True
                    c['note'] = c['note'].replace('float32', 'float64')
                c = {k: v for k, v in c.items() if v is not None}
                out = ROOT / 'input' / f'input_3bpa_omat24_{name}_cue.json'
                json.dump(c, open(out, 'w'), indent=4)
                print(out.name, 'epochs', args.epochs, 'avg_n', c['avg_n_neighbors'], 'species', len(c['atomic_numbers']))


if __name__ == '__main__':
    main()
