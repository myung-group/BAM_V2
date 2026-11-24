"""Reference data for the PyTorch port of RACEv2 (2026-11-13).  Runs the JAX model (GPU job, env bam_cue) and writes
everything a torch re-implementation needs to be checked layer by layer:

  <out>/config.json      training config stored in the checkpoint
  <out>/params.npz       every nnx.Param of the EMA weights, key = jax keystr of its path (float64)
  <out>/static.json      static (non-Param) attributes of every sub-module: irreps strings, scales, flags, tables
  <out>/constants.npz    e3nn-jax conventions: normalised-activation constants, Clebsch-Gordan tensors (l <= 3),
                         spherical harmonics of random vectors, a small channelwise tensor-product example
  <out>/frame_XX.npz     per structure: the graph (all node / edge / global leaves, padded, with n_node / n_edge),
                         prelude outputs (h0, radial basis, Y, atomic basis A, pre-strata), per layer the input and
                         output features, x features and the stratum energy, the final node energies, and the
                         model's energy / forces / stress
  <out>/frames.extxyz    the same structures (positions, cell, numbers) for the torch side

    python -I race_v2_torch/tools/dump_jax_reference.py --ckpt <ckpt.pkl> --out <dir> --val <aselmdb dir> [--n 12]
    python -I race_v2_torch/tools/dump_jax_reference.py --ckpt <ckpt.pkl> --out <dir> --xyz frames.extxyz

--xyz (2026-11-24): the structures are read from an ASE-readable file instead of a validation split.
A float32-trained checkpoint is dumped in float64 with BAM_PREDICT_X64=1 (parameters and graphs cast, as predict.py).
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--val', default=None); ap.add_argument('--xyz', default=None)
ap.add_argument('--n', type=int, default=12); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[2]))
a = ap.parse_args()
if (a.val is None) == (a.xyz is None):
    ap.error('give exactly one of --val / --xyz')
sys.path.insert(0, a.repo); sys.path.insert(0, os.path.join(a.repo, 'race_v2', 'data'))
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import e3nn_jax as e3nn
from flax import nnx
import ase, ase.io
from race_v2.inference.predict import Predictor
import stream_io as S

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
p = Predictor(a.ckpt, species_refit=False, batch_size=1)
model, cfg = p.model, p.cfg
json.dump(cfg, open(out / 'config.json', 'w'), indent=1)

# ---------------------------------------------------------------- parameters
params = {}
for path, leaf in jax.tree_util.tree_leaves_with_path(nnx.state(model, nnx.Param)):
    params[jax.tree_util.keystr(path)] = np.asarray(leaf)
np.savez(out / 'params.npz', **params)
print(f'{len(params)} parameter arrays, {sum(v.size for v in params.values()):,} values')

# ---------------------------------------------------------------- static attributes
def _js(v):
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, e3nn.Irreps):
        return 'Irreps:' + str(v)
    if isinstance(v, (tuple, list)):
        r = [_js(x) for x in v]
        return None if any(x is _SKIP for x in r) else r
    if isinstance(v, dict):
        r = {str(k): _js(x) for k, x in v.items()}
        return None if any(x is _SKIP for x in r.values()) else r
    if isinstance(v, np.ndarray) and v.size < 100000:
        return v.tolist()
    return _SKIP
_SKIP = object()
static = {}
for path, node in nnx.iter_graph(model):
    if not isinstance(node, nnx.Module):
        continue
    d = {}
    for k, v in vars(node).items():
        if isinstance(v, (nnx.Variable, nnx.Module)) or k.startswith('_object'):
            continue
        j = _js(v)
        if j is not _SKIP and j is not None or v is None:
            d[k] = j
    static['/'.join(str(x) for x in path) or '<root>'] = d
json.dump(static, open(out / 'static.json', 'w'), indent=1, default=str)

# ---------------------------------------------------------------- e3nn-jax conventions
rng = np.random.RandomState(1)
const = {}
for name, f in (('silu', jax.nn.silu), ('tanh', jnp.tanh), ('sigmoid', jax.nn.sigmoid)):
    g = e3nn.normalize_function(f)
    xs = jnp.linspace(-3.0, 3.0, 13)
    const[f'act_{name}_x'] = np.asarray(xs); const[f'act_{name}_normalized'] = np.asarray(jax.vmap(g)(xs))
for l1 in range(4):
    for l2 in range(4):
        for l3 in range(abs(l1 - l2), min(l1 + l2, 3) + 1):
            const[f'cg_{l1}_{l2}_{l3}'] = np.asarray(e3nn.clebsch_gordan(l1, l2, l3))
v = rng.randn(64, 3)
const['sh_vectors'] = v
const['sh_component_normalize'] = np.asarray(e3nn.spherical_harmonics('1x0e+1x1o+1x2e+1x3o', jnp.asarray(v), normalize=True, normalization='component').array)
# channelwise product with the mul-1 trick (as RACEConvolution x product): mul 3, irreps 0e+1o+2e+3o
ir = e3nn.Irreps('3x0e+3x1o+3x2e+3x3o')
m = e3nn.IrrepsArray(ir, jnp.asarray(rng.randn(5, ir.dim))); x = e3nn.IrrepsArray(ir, jnp.asarray(rng.randn(5, ir.dim)))
tp = e3nn.tensor_product(m.mul_to_axis(3), x.mul_to_axis(3), filter_ir_out=e3nn.Irreps('0e+1o+2e+3o')).axis_to_mul()
const['tp_example_m'] = np.asarray(m.array); const['tp_example_x'] = np.asarray(x.array); const['tp_example_out'] = np.asarray(tp.array)
const['tp_example_out_irreps'] = np.asarray(str(tp.irreps))
np.savez(out / 'constants.npz', **const)

# ---------------------------------------------------------------- structures from the validation split
if a.xyz is not None:
    frames = ase.io.read(a.xyz, ':')
    for k, at in enumerate(frames):
        at.info.setdefault('source', f'xyz_{k}')
else:
    files = sorted(str(q) for q in Path(a.val).glob('*.aselmdb'))
    src = S.AselmdbSource(files, lru=len(files))
    ids = [(f, r) for f in range(len(files)) for r in src.row_ids(f)]
    order = np.random.RandomState(a.seed).permutation(len(ids))
    frames, big = [], 0
    for k in order:
        if len(frames) >= a.n:
            break
        st = src.read_block(*ids[k][:1], [ids[k][1]])[0]
        n = len(st['numbers'])
        if n > 160 or (len(frames) >= a.n - 2 and big < 2 and n < 40):
            continue                                   # keep them small, but at least two with >= 40 atoms
        big += n >= 40
        at = ase.Atoms(numbers=st['numbers'], positions=st['positions'], cell=st['cell'], pbc=True)
        at.info['source'] = f'val{ids[k][0]:02d}_{ids[k][1]}'
        frames.append(at)
ase.io.write(out / 'frames.extxyz', frames, format='extxyz')

def arr(x):
    return np.asarray(x.array if isinstance(x, e3nn.IrrepsArray) else x)

for fi, at in enumerate(frames):
    g = p._graphs([at])
    import functools, jraph
    from race_v2.data.xyz_graphs import GraphListDataset
    from race_v2.data.loaders import BucketedDataLoader, default_collate_fn
    T = p._T
    loader = BucketedDataLoader(GraphListDataset(g), batch_size=1, n_buckets=1, shuffle=False, drop_last=False,
                                rngs=nnx.Rngs(0), max_nodes_per_batch=len(at) + 1)
    loader.collate_fn = functools.partial(default_collate_fn, node_bucket_size=int(T._get(cfg, 'node_bucket_size', 128)),
                                          edge_bucket_size=int(T._get(cfg, 'edge_bucket_size', 4096)))
    batch = next(iter(loader))
    rec = {}
    for name, part in (('nodes', batch.nodes), ('edges', batch.edges), ('globals', batch.globals)):
        if isinstance(part, dict):
            for k2, v2 in part.items():
                if v2 is not None and not isinstance(v2, dict):
                    rec[f'graph_{name}_{k2}'] = np.asarray(v2)
    for k2 in ('senders', 'receivers', 'n_node', 'n_edge'):
        rec[f'graph_{k2}'] = np.asarray(getattr(batch, k2))
    periodic = bool(cfg.get('periodic', False))
    pos = batch.nodes['positions']
    cell, cut = (batch.globals['cell'], batch.globals['cutoff']) if periodic else (None, None)
    ctx = model._prelude(pos, cell, cut, batch)
    for k2 in ('features', 'radial_basis', 'sh', 'atomic_basis', 'r_norm', 'shell_env', 'x_feats'):
        if ctx.get(k2) is not None:
            rec[f'prelude_{k2}'] = arr(ctx[k2])
    rec['prelude_strata'] = np.stack([arr(o) for o in ctx['outputs']]) if ctx['outputs'] else np.zeros((0,))
    feats, hist = ctx['features'], []
    for i in range(len(model.layers)):
        hist.append(feats)
        prev = hist[i - 1] if (model.residual == 'chebyshev' and i >= 2) else None
        rec[f'layer{i}_in'] = arr(feats)
        feats, stratum = model._layer_step(ctx, i, feats, prev)
        ctx['outputs'].append(stratum)
        rec[f'layer{i}_out'] = arr(feats); rec[f'layer{i}_stratum'] = arr(stratum); rec[f'layer{i}_x'] = arr(ctx['x_feats'])
        for part, lst in ctx['layer_parts'].items():
            rec[f'layer{i}_part_{part}'] = arr(lst[-1])
    fin = model._finalize(ctx, feats)
    rec['node_energies'] = arr(fin['node_energies'])
    res = model(batch)
    for k2 in ('energy', 'forces', 'stress'):
        if res.get(k2) is not None:
            rec[f'result_{k2}'] = np.asarray(res[k2])
    np.savez(out / f'frame_{fi:02d}.npz', **rec)
    print(f'frame {fi:02d} {at.info["source"]} {len(at)} atoms {at.get_chemical_formula()} edges {int(batch.n_edge[0])} '
          f'E {float(np.asarray(res["energy"])[0]):.6f} eV')
print('done ->', out)
