"""Latent Qeq charges of trained RACE_LR checkpoints vs the DFT Hirshfeld charges (never used in training).

usage: python -m runs.diag.local_4g.charges   is not needed; run as a file from the repo root:
       python runs/diag/local_4g/charges.py <config.json> <ckpt.pkl> <test.xyz> <out.npz>

Prints the per-atom correlation and RMSE of q vs Hirshfeld per subset, and writes q / Hirshfeld / subset per frame.
"""
import functools, sys
import numpy as np
import jax, jax.numpy as jnp, jraph
from flax import nnx
sys.path.insert(0, '.')
from bam_omat24 import train_3bpa as T                                      # noqa: E402
from bam_omat24.data_3bpa import load_frames, GraphListDataset, atomic_numbers_to_indices, preprocess_graph  # noqa
from bam_omat24.data_nnx import BucketedDataLoader, default_collate_fn       # noqa: E402
from bam_omat24.training_utils import load_checkpoint                        # noqa: E402
from bam_omat24.analyze_errors import _tag_frames                            # noqa: E402
import json

cfg_path, ckpt_path, xyz, out = sys.argv[1:5]
cfg = json.load(open(cfg_path))
if cfg.get('x64'):
    jax.config.update('jax_enable_x64', True)
(Z, E0, _tr, _va, _te, stats) = T.build_datasets(cfg, open('/dev/null', 'w'))
graphdef, *_ = T.build_model(cfg, len(Z), E0, stats, open('/dev/null', 'w'))
model = nnx.merge(graphdef, jax.tree.map(jnp.asarray, load_checkpoint(ckpt_path)['ema_params']))
frames = load_frames(xyz)
ai = atomic_numbers_to_indices(Z)
graphs = [preprocess_graph(a, ai, E0, float(cfg['cutoff']), charge_key=cfg['total_charge_key']) for a in frames]
if cfg.get('x64'):
    graphs = T._cast_graphs(graphs, np.float64)
graphs = _tag_frames(graphs)
loader = BucketedDataLoader(GraphListDataset(graphs), batch_size=32, n_buckets=1, shuffle=False, drop_last=False,
                            rngs=nnx.Rngs(0))
loader.collate_fn = functools.partial(default_collate_fn, node_bucket_size=int(cfg.get('node_bucket_size', 128)),
                                      edge_bucket_size=int(cfg.get('edge_bucket_size', 4096)))
step = jax.jit(lambda b: model(b)['charges'])
q = [None] * len(frames)
for b in loader:
    qb = np.asarray(step(b)); gmask = np.asarray(jraph.get_graph_padding_mask(b))
    fid = np.asarray(b.globals['frame_id']).reshape(-1); off = np.concatenate([[0], np.cumsum(np.asarray(b.n_node))])
    for g in np.flatnonzero(gmask):
        q[fid[g]] = qb[off[g]:off[g + 1]]
lab = np.array([a.info['subset'] for a in frames])
h = [a.arrays['hirshfeld_charges'] for a in frames]
for k in sorted(set(lab)):
    qq = np.concatenate([q[i] for i in np.flatnonzero(lab == k)]); hh = np.concatenate([h[i] for i in np.flatnonzero(lab == k)])
    tot = np.array([q[i].sum() for i in np.flatnonzero(lab == k)])
    print(f'{k:22s} atoms {len(qq):6d}  corr(q, Hirshfeld) {np.corrcoef(qq, hh)[0, 1]:+.3f}  RMSE {np.sqrt(((qq - hh) ** 2).mean()):.3f} e  '
          f'|q| mean {np.abs(qq).mean():.3f} vs {np.abs(hh).mean():.3f}  sum q [{tot.min():+.4f}, {tot.max():+.4f}]')
np.savez(out, q=np.array(q, dtype=object), h=np.array(h, dtype=object), subset=lab,
         symbols=np.array([a.get_chemical_symbols() for a in frames], dtype=object))
