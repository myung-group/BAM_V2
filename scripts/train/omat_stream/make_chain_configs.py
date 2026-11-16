"""Configs of the chained run (chain_oam.sh) for a given number of GPUs and data location.

Reads configs/omat_pretrain/race_v2_omat24_pretrain_continue.json and
configs/oam_finetune/race_v2_oam_finetune_after_continue.json (written for 48 GPUs and the paths of account t510a13)
and writes <out-dir>/<tag>_pre.json and <out-dir>/<tag>_ft.json.  Only paths, run names and the virtual-epoch numbers
change; the model, loss and optimiser settings are copied.

Virtual epochs (5,000 steps each) scale with 1 / GPUs:
  pretraining : one OMat24 data epoch = 139,000 steps at 24 GPUs; --pre-epochs data epochs (default 2), linear decay
                over the last third;
  fine-tuning : one data epoch (MPtrj x8 + sAlex x1) = 6.1 virtual epochs at 24 GPUs; --ft-epochs data epochs
                (default 6.2), constant learning rate for --ft-flat data epochs (default 4.6), then linear decay.

    python -I scripts/train/omat_stream/make_chain_configs.py --gpus 48 --tag oam4 --out-dir runs/omat_stream \
        --map /scratch/t510a13/data/oam_lmdb=<raw OMat24/sAlex> --map /scratch/t510a13/data/mptrj_lmdb=<raw MPtrj> \
        --map /scratch/t510a13/data/oam_sizes=<remapped size tables> --init runs/mbd/snap/omat_pre_final.pkl
    --test : small check of the chain (2 virtual epochs of 100 steps, 1,000 validation structures)
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser()
ap.add_argument('--gpus', type=int, required=True); ap.add_argument('--tag', required=True)
ap.add_argument('--out-dir', default='runs/omat_stream')
ap.add_argument('--map', action='append', default=[], help='OLD=NEW path prefix in the stream sources and sizes')
ap.add_argument('--init', default='runs/mbd/snap/omat_pre_final.pkl', help='checkpoint the pretraining starts from')
ap.add_argument('--pre-epochs', type=float, default=2.0); ap.add_argument('--ft-epochs', type=float, default=6.2)
ap.add_argument('--ft-flat', type=float, default=4.6)
ap.add_argument('--test', action='store_true')
a = ap.parse_args()
maps = [tuple(m.split('=', 1)) for m in a.map]


def remap(s):
    for old, new in maps:
        if old in s:
            return s.replace(old, new)
    return s


def paths(split):
    for p in split.get('parts', [split]):
        p['sources'] = [remap(s) for s in p['sources']]
        p['sizes'] = remap(p['sizes'])
        left = [s for s in p['sources'] + [p['sizes']] if '/t510a13/' in s]
        assert not left, f'no --map for {left}'


def named(c, name):
    run = Path('runs') / f'{name}_cue'
    c['log_file'] = str(run / f'loss_{name}_cue.out'); c['ckpt_name'] = str(run / f'ckpt_{name}_cue.pkl')


pre = json.load(open(REPO / 'configs/omat_pretrain/race_v2_omat24_pretrain_continue.json'))
ft = json.load(open(REPO / 'configs/oam_finetune/race_v2_oam_finetune_after_continue.json'))
assert pre['stream']['steps_per_epoch'] == ft['stream']['steps_per_epoch'] == 5000
s = 24.0 / a.gpus
n_pre = round(a.pre_epochs * 139000 * s / 5000); cd_pre = round(n_pre * 2 / 3)
n_ft = round(a.ft_epochs * 6.1 * s); cd_ft = round(a.ft_flat * 6.1 * s)
if a.test:
    n_pre = n_ft = 2; cd_pre = cd_ft = 1
    for c in (pre, ft):
        c['stream']['steps_per_epoch'] = 100; c['stream']['valid']['max_structures'] = 1000
for c, n, cd in ((pre, n_pre, cd_pre), (ft, n_ft, cd_ft)):
    paths(c['stream']['train']); paths(c['stream']['valid'])
    c['n_epochs'] = n; c['lr_cooldown_from_epoch'] = cd
    c['note'] += (f"  make_chain_configs.py: {a.gpus} GPUs, n_epochs {n}, decay from {cd}"
                  + (" (chain test: 100 steps per virtual epoch)" if a.test else ""))
named(pre, f'{a.tag}_pre'); named(ft, f'{a.tag}_ft')
pre['init_params_from'] = a.init
ft['init_params_from'] = f'runs/mbd/snap/{a.tag}_pre_final.pkl'
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
for c, k in ((pre, 'pre'), (ft, 'ft')):
    Path(c['log_file']).parent.mkdir(parents=True, exist_ok=True)       # the trainer opens the log first
    json.dump(c, open(out / f'{a.tag}_{k}.json', 'w'), indent=1)
    print(out / f'{a.tag}_{k}.json', 'n_epochs', c['n_epochs'], 'decay from', c['lr_cooldown_from_epoch'],
          'steps per epoch', c['stream']['steps_per_epoch'], 'lr', c['learning_rate'], 'init', c['init_params_from'])
