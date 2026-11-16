"""Configs for multi-dataset training on the aselmdb corpora of ~/Datasets (OC25, MPtrj, sAlex, OMol25, OPoly26,
OMC25), streamed (data_format "stream", stream option "mixed"), 2026-11-11.  The recipe is the RACEv2 final recipe
(configs/base/race_v2_base_recipe.json); tasks and E0 tables follow BAM-torch examples/oc25_mptrj
(dataset_id -> task; get_enr_avg_per_element_*.txt -> stream.atom_energies_by_task).

    python -I scripts/train/make_multidataset_configs.py --name multi_smoke --model polar --frames 300 --one-shard
    python -I scripts/train/make_multidataset_configs.py --name multi_1M --model polar          (BAM-torch frame counts)

--frames N: max_frames_per_shard of every source (default: the BAM-torch input.json values = about 1M frames per
weight unit); --one-shard: first shard of every source only (smoke tests); --sources: subset of the source names.
-> configs/multi/<name>.json
"""
import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'input' / 'input_3bpa_omat24_abl3_full_s42_cue.json'
# name: (train dir, valid dir, task, frames per shard train / valid (BAM-torch input.json), extra source options)
# meta: BAM-torch meta-data root (<root>/<split directory>/<shard stem>.npz: exclusions and charge groups)
SOURCES = {
    'oc25': ('oc25/train', 'oc25/val', 1, 50000, 250, {'meta': 'oc25_meta_v2', 'slab': True, 'dipole_sign': -1.0}),
    'mptrj': ('mptrj/train', 'mptrj/val', 2, 200000, 2500, {'task_rules': {'mp2020_hubbard': 5},
                                                             'meta': 'mptrj_meta_v2'}),
    'alex': ('alex/train', 'alex/val', 2, 2880, 8, {'task_rules': {'mp2020_hubbard': 5}, 'meta': 'alex_meta_v2'}),
    'omol25': ('omol25/train_4M', 'omol25/neutral_val', 3, 25000, 30,
               {'charge_label_key': 'nbo_charges', 'charge_label_scale': 0.85, 'meta': 'omol25_meta_v2'}),
    'omol25_nbo': ('omol25/train_nbo', None, 3, 250000, None,
                   {'charge_label_key': 'nbo_charges', 'charge_label_scale': 0.85, 'meta': 'omol25_meta_v2'}),
    'opoly26': ('opoly26/train', 'opoly26/val', 4, 37500, 30, {'meta': 'opoly26_meta_v2'}),
    'omc25': ('omc25/train', 'omc25/val', 6, 25000, 30, {'meta': 'omc25_meta_v2'}),
}
E0_FILES = {1: 'oc25', 2: 'pbe_gga', 3: 'omol25', 4: 'opoly26', 5: 'pbe_gga_u', 6: 'omc25'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--data', type=Path, default=Path.home() / 'Datasets')
    ap.add_argument('--e0-dir', type=Path, default=Path.home() / 'Python/BAM-torch/examples/oc25_mptrj')
    ap.add_argument('--model', default='polar', choices=['race', 'race_cs', 'polar'],
                    help='race: plain RACE; race_cs: + charge/spin embedding and task table; polar: RACE_POLAR')
    ap.add_argument('--sources', nargs='+', default=list(SOURCES))
    ap.add_argument('--frames', type=int, default=None)
    ap.add_argument('--valid-frames', type=int, default=None)
    ap.add_argument('--one-shard', action='store_true')
    ap.add_argument('--subset-seed', type=int, default=0)
    ap.add_argument('--cutoff', type=float, default=6.0)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--steps-per-epoch', type=int, default=None)
    ap.add_argument('--layers', type=int, default=None)
    ap.add_argument('--hidden', type=int, default=None)
    ap.add_argument('--lmax', type=int, default=None)
    ap.add_argument('--batch', type=int, default=8, help='graphs_per_pack (mean structures per batch)')
    ap.add_argument('--buckets', type=int, default=4)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--optimizer', default='amsgrad')
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--x64', action='store_true')
    ap.add_argument('--tp-backend', default='e3nn', help="'cue' needs cuequivariance-jax")
    ap.add_argument('--stress-weight', type=float, default=10.0)
    ap.add_argument('--charge-weight', type=float, default=0.08)
    ap.add_argument('--dipole-weight', type=float, default=0.1)
    ap.add_argument('--avg-n-neighbors', type=float, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--polar', default=None, help='JSON object merged into "polar"')
    ap.add_argument('--sizes-dir', default=None,
                    help='directory of the size tables (<dir>/train, <dir>/valid), shared by configs with the same '
                         'sources and cutoff; make them with scripts/data/multi_sizes.py (default: made in the run dir)')
    ap.add_argument('--patience', type=int, default=5, help='lr_schedule_patience in (virtual) epochs')
    ap.add_argument('--phase1', action='store_true',
                    help='polar without charge groups, contact charge transfer and the slab correction')
    args = ap.parse_args()

    z_to_index = ast.literal_eval((args.e0_dir / 'get_enr_avg_per_element_omol25.txt').read_text())[1]
    atomic_numbers = sorted(int(z) for z in z_to_index)

    def e0(tag):
        e_by_index, idx = ast.literal_eval((args.e0_dir / f'get_enr_avg_per_element_{tag}.txt').read_text())
        return [float(e_by_index[idx[z]]) for z in atomic_numbers]

    def source(nm, split):
        tr, va, task, n_tr, n_va, extra = SOURCES[nm]
        sub = tr if split == 'train' else va
        if sub is None:
            return None
        path = args.data / sub
        if args.one_shard:
            path = sorted(path.glob('*.aselmdb'))[0]
        n = (args.frames or n_tr) if split == 'train' else (args.valid_frames or n_va)
        s = {'name': nm, 'path': f'aselmdb:{path}', 'task': task, 'max_frames_per_shard': n,
             'subset_seed': args.subset_seed}
        for k, v in extra.items():
            if k == 'meta':
                md = args.data / v / Path(sub).name
                if md.is_dir():
                    s['meta_dir'] = str(md)
            else:
                s[k] = v
        return s

    c = {k: v for k, v in json.load(open(BASE)).items() if not k.startswith('_') and k not in ('valid_frac', 'note')}
    for k in ('train_path', 'valid_path', 'test_paths', 'soap_species_adam'):
        c.pop(k, None)
    name = args.name
    tasks = sorted({SOURCES[nm][2] for nm in args.sources} | {t for nm in args.sources
                                                              for t in (SOURCES[nm][5].get('task_rules') or {}).values()})
    c.update({
        'atomic_numbers': atomic_numbers, 'atom_energies': e0('pbe_gga'), 'e0_mode': 'isolated',
        'cutoff': args.cutoff, 'periodic': True, 'stress_loss': True, 'stress_weight': args.stress_weight,
        'x64': bool(args.x64), 'pad_mode': 'fixed', 'n_epochs': args.epochs, 'optimizer': args.optimizer,
        'tp_backend': args.tp_backend, 'seed': args.seed, 'restart': True, 'ckpt_last_every': 1,
        'eval_test_every': 0, 'gauge_refit': False, 'lr_warmup_epochs': 1, 'lr_schedule_patience': 5,
        'total_charge_key': 'charge', 'total_spin_key': 'spin',
        'log_file': f'runs/multi/{name}/loss.out', 'ckpt_name': f'runs/multi/{name}/ckpt.pkl',
        'data_format': 'stream',
        'stream': {
            'train': [source(nm, 'train') for nm in args.sources],
            'valid': [s for s in (source(nm, 'valid') for nm in args.sources) if s],
            'mixed': True, 'dipole': args.model == 'polar',
            'atom_energies_by_task': {str(t): str(args.e0_dir / f'get_enr_avg_per_element_{E0_FILES[t]}.txt')
                                      for t in tasks},
            'workers': args.workers, 'eval_workers': max(2, args.workers // 2), 'n_buckets': args.buckets,
            'graphs_per_pack': args.batch, 'pack_max_graphs': 8 * args.batch, 'block_rows': 64, 'mix_blocks': 64,
            'stats_sample': 4000,
        },
        'batch_size': args.batch,
        'note': f'multi-dataset stream training ({", ".join(args.sources)}), model {args.model}, cutoff {args.cutoff:g} A, '
                f'E0 per task from {args.e0_dir}; made by scripts/train/make_multidataset_configs.py',
    })
    c['lr_schedule_patience'] = args.patience
    if args.sizes_dir:
        for split in ('train', 'valid'):
            c['stream'][split] = {'sources': c['stream'][split], 'sizes': f'{args.sizes_dir}/{split}'}
    if args.steps_per_epoch:
        c['stream']['steps_per_epoch'] = args.steps_per_epoch
    if args.avg_n_neighbors:
        c['avg_n_neighbors'] = args.avg_n_neighbors
    else:
        c.pop('avg_n_neighbors', None)
        c['use_dataset_stats'] = c.get('use_dataset_stats', False)
    if args.layers is not None:
        c['n_layers'] = args.layers
    if args.hidden is not None:
        c['hidden_size'] = args.hidden
        c.pop('hidden_irreps', None)
    if args.lmax is not None:
        c['lmax'] = args.lmax
        c.pop('hidden_irreps', None)
    if args.lr is not None:
        c['learning_rate'] = args.lr
    lr = {'charge_spin_embedding': True, 'n_tasks': max(tasks) + 1}
    if args.model == 'race':
        c['model'] = 'RACE'
    elif args.model == 'race_cs':
        c['model'], c['long_range'] = 'RACE_LR', dict(lr, qeq=False)
    else:
        c['model'], c['long_range'] = 'RACE_POLAR', lr
        c['polar'] = {'sigma': 1.0, 'thole_width': 2.0, 'polarizability_init': 0.7, 'min_polarizability': 0.001,
                      'max_polarizability': 4.0, 'max_polarizability_per_element': {'H': 1.0},
                      'quadrupoles': True, 'cap_induced_width': True, 'cg_max_iterations': 25, 'cg_tolerance': 1e-4,
                      'ewald_k_tolerance': 1e-5, 'multipole_init_scale': 0.1}
        if not args.phase1:      # the BAM-torch input.json settings (charge groups from the meta-data, slabs)
            c['polar'].update({'charge_constraint': 'electrode', 'charge_transfer': 'contact',
                               'transfer_cutoff': 4.0, 'contact_radius': 2.5, 'contact_pairs': 'charged',
                               'slab_correction': True, 'slab_axis': 2})
        if args.polar:
            c['polar'].update(json.loads(args.polar))
        c['charge_weight'], c['dipole_weight'] = args.charge_weight, args.dipole_weight
    c = {k: v for k, v in c.items() if v is not None}
    out = ROOT / 'input' / 'multi' / f'{name}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(c, open(out, 'w'), indent=1)
    print(out, '|', c['model'], '| sources', args.sources, '| tasks', tasks, '| species', len(atomic_numbers))


if __name__ == '__main__':
    main()
