"""Configs for the OAM subset screening (2026-11-05): RACEv2 final recipe ("v2") vs all new parts removed ("v1"),
one per dataset (omat24, mptrj, salex), from the 3BPA ablation baseline and the subset meta files.

    python -I scripts/train/make_oam_configs.py <subset_dir> [--epochs N] [--tag oam] [--seeds 42]
-> research/input/input_3bpa_omat24_<tag>_<dataset>_<variant>_s<seed>_cue.json  (name pattern of runs/run_cue.sh)

avg_n_neighbors (2026-11-06): the atom-weighted mean neighbour count of the FULL training split at the model cutoff
(sum of edges / sum of atoms, the same statistic the trainer prints as "avg_n_neighbors" in the training-set
statistics).  It is computed once with the JAX-free neighbour search (race_v2/data/neighbors.py, --workers processes)
and cached next to the xyz as <stem>.avg_n_c<cutoff>.json.  Earlier configs used the per-structure mean of the first
500 structures from the subset meta file (OMat24 10k: 26.965 vs the full-split value 27.909; 100k: 26.965 vs
26.966).  Run inside a job for large splits (no heavy work on login nodes).
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _count(args):
    import sys
    sys.path.insert(0, str(ROOT / 'race_v2' / 'data'))
    from neighbors import neighbor_pairs                     # JAX-free (race_v2/__init__ would import JAX)
    frames, cutoff = args
    return [(len(a), len(neighbor_pairs(a, cutoff)[0])) for a in frames]


def full_avg_neighbors(train_path, cutoff, workers=0):
    """Atom-weighted mean neighbour count (sum edges / sum atoms) of the whole training split; cached."""
    import os
    p = Path(train_path)
    cache = p.with_name(f'{p.stem}.avg_n_c{cutoff:g}.json')
    if cache.exists():
        d = json.load(open(cache))
        if d.get('source_size') == p.stat().st_size:
            return d['avg_n_neighbors']
    import ase.io
    import concurrent.futures as cf
    import multiprocessing as mp
    frames = ase.io.read(str(p), index=':')
    workers = workers or min(64, int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or len(os.sched_getaffinity(0)))
    chunks = [(frames[k:k + 512], cutoff) for k in range(0, len(frames), 512)]
    res = []
    if workers <= 1:
        for c in chunks:
            res += _count(c)
    else:
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('forkserver')) as ex:
            for part in ex.map(_count, chunks):
                res += part
    na = sum(r[0] for r in res); ne = sum(r[1] for r in res)
    val = ne / na
    json.dump({'avg_n_neighbors': val, 'n_structures': len(res), 'sum_atoms': na, 'sum_edges': ne, 'cutoff': cutoff,
               'source': str(p), 'source_size': p.stat().st_size}, open(cache, 'w'), indent=1)
    print(f'{p.name}: avg_n_neighbors {val:.3f} over {len(res)} structures (cutoff {cutoff:g}), cached in {cache.name}')
    return val
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
    ap.add_argument('--workers', type=int, default=0, help='processes for the avg_n_neighbors pass (0 = task CPUs)')
    ap.add_argument('--warmup', type=int, default=None, help='lr_warmup_epochs (default of train_3bpa: 10)')
    ap.add_argument('--optimizer', default=None, help='optimizer override (e.g. amsgrad; default: the recipe SOAP)')
    ap.add_argument('--model', default='race', choices=['race', 'racelr'],
                    help='racelr: RACE_LR with the Qeq stratum (Ewald) and chi0 from Pauling; the OAM data are '
                         'neutral, so no charge / spin input')
    ap.add_argument('--ewald', type=float, nargs=3, default=None, metavar=('ALPHA', 'N_REAL', 'N_RECIP'),
                    help='Ewald settings for --model racelr (required; choose with scripts/data/check_ewald.py)')
    ap.add_argument('--cutoff', type=float, default=5.0,
                    help='!= 5: avg_n_neighbors recomputed on 500 training structures at this cutoff')
    args = ap.parse_args()
    if args.model == 'racelr' and args.ewald is None:
        raise SystemExit('--model racelr needs --ewald ALPHA N_REAL N_RECIP (scripts/data/check_ewald.py)')
    base = json.load(open(BASE))
    for ds in args.datasets:
        meta = json.load(open(args.subset_dir / f'{ds}_meta.json'))
        avg_n = full_avg_neighbors(meta['train']['path'], args.cutoff, args.workers)
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
