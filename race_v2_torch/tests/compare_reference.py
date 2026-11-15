"""Compare racev2_torch with a JAX reference dump (tools/dump_jax_reference.py) intermediate by intermediate.

    racev2_torch/tools/tpy racev2_torch/tests/compare_reference.py <ref_dir> [--frames 0,3,5] [--save model.pt]

<ref_dir>: config.json, params.npz, static.json, constants.npz, frame_XX.npz.  The params are converted with
racev2_torch.convert (params.npz path); every frame is evaluated on the exact padded graph stored in the dump
(nodes / edges / Sij / cell / cutoff / volume as used by JAX), and the max abs differences of every stored
quantity are printed over the real nodes / edges / graphs (padding excluded).  Energies exclude E0, as in the dump
(model built with l_train=True by predict.Predictor).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from racev2_torch import o3  # noqa: E402
from racev2_torch.convert import build_from_params, load_params_npz  # noqa: E402

torch.set_default_dtype(torch.float64)
ap = argparse.ArgumentParser()
ap.add_argument('ref')
ap.add_argument('--frames', default=None, help='comma-separated frame indices (default all)')
ap.add_argument('--edge-chunk', type=int, default=1024)
ap.add_argument('--save', default=None, help='also write the converted model file')
ap.add_argument('--tol', type=float, default=1e-10, help='flag absolute differences above this')
a = ap.parse_args()
ref = Path(a.ref)
worst = {}


def cmp(name, mine, theirs, scale_name=None):
    mine = np.asarray(mine.detach().cpu() if torch.is_tensor(mine) else mine, dtype=np.float64)
    theirs = np.asarray(theirs, dtype=np.float64)
    if mine.shape != theirs.shape:
        print(f"  SHAPE {name}: torch {mine.shape} vs jax {theirs.shape}")
        worst[name] = max(worst.get(name, 0.0), float('inf'))
        return float('inf')
    d = float(np.max(np.abs(mine - theirs))) if mine.size else 0.0
    s = float(np.max(np.abs(theirs))) if theirs.size else 0.0
    worst[name] = max(worst.get(name, 0.0), d)
    flag = '' if d <= a.tol else '   <-- above tol'
    print(f"  {name:22s} max|d| {d:10.3e}   max|ref| {s:10.3e}   rel {d / max(s, 1e-300):9.2e}{flag}")
    return d


# ------------------------------------------------------------------------------------------- conventions
print(f"== {ref}")
c = np.load(ref / 'constants.npz')
print("constants (e3nn-jax conventions):")
for nm, fn in (('silu', o3.silu), ('tanh', torch.tanh), ('sigmoid', torch.sigmoid)):
    x = torch.as_tensor(c[f'act_{nm}_x'])
    cmp(f"act_{nm}_normalized", fn(x) / o3.normalize_constant(nm), c[f'act_{nm}_normalized'])
cg_w = max(float(np.abs(o3.clebsch_gordan(*map(int, k.split('_')[1:])) - c[k]).max()) for k in c.files
           if k.startswith('cg_'))
print(f"  {'clebsch_gordan (all)':22s} max|d| {cg_w:10.3e}")
worst['clebsch_gordan'] = cg_w
cmp('spherical_harmonics', o3.spherical_harmonics(3, torch.as_tensor(c['sh_vectors'])), c['sh_component_normalize'])
from racev2_torch.model import ChannelwiseProduct  # noqa: E402
tp = ChannelwiseProduct(o3.Irreps('3x0e+3x1o+3x2e+3x3o'))
print(f"  tp_example irreps: torch {tp.irreps_out} jax {c['tp_example_out_irreps']}")
cmp('tp_example (x product)', tp(torch.as_tensor(c['tp_example_m']), torch.as_tensor(c['tp_example_x'])),
    c['tp_example_out'])

# ------------------------------------------------------------------------------------------- model
cfg = json.load(open(ref / 'config.json'))
t0 = time.time()
params = load_params_npz(ref / 'params.npz')
model, rep = build_from_params(params, cfg)
del params
model.edge_chunk = a.edge_chunk
print(f"converted {rep['n_converted']} arrays ({rep['n_values']:,} values, dtypes {rep['source_dtypes']}), "
      f"ignored {rep['ignored']} [{time.time() - t0:.1f} s]")
if a.save:
    model.save(a.save, config=cfg, extra=dict(source=dict(kind='npz', path=str(ref / 'params.npz')), report=rep))
    print(f"saved {a.save}")

static = json.load(open(ref / 'static.json'))
root = static.get('<root>', {})
print("static attributes:")


def _arr(v):
    return np.asarray(v, dtype=np.float64)


checks = [('species_shell_radii', model.species_shell_radii if model.radial_shells else None),
          ('species_radii_t', getattr(model, 'species_radii', None)),
          ('atom_energies', model.atom_energies)]
for k, mine in checks:
    if k in root and root[k] is not None and mine is not None:
        cmp(f"static {k}", mine, _arr(root[k]))
if root.get('radial_scales') is not None:
    cmp("static radial_scales", np.array([s for s in model.radial_scales]), _arr(root['radial_scales']))
for k in ('avg_n_neighbors', 'cutoff', 'radial_polynomial_p', 'pair_repulsion_switch', 'pair_repulsion_switch_width'):
    if k in root:
        mine = {'avg_n_neighbors': model.avg_n_neighbors, 'cutoff': model.cutoff,
                'radial_polynomial_p': model.radial_polynomial_p,
                'pair_repulsion_switch': model.pair_repulsion_switch,
                'pair_repulsion_switch_width': model.pair_repulsion_switch_width}[k]
        same = (mine == root[k]) or (mine is not None and root[k] is not None and float(mine) == float(root[k]))
        print(f"  static {k:20s} jax {root[k]!r:>10} torch {mine!r:>10} {'ok' if same else 'DIFFERENT'}")
        if not same:
            worst[f'static {k}'] = float('inf')
if 'species_z' in root:
    ok = list(root['species_z']) == model.atomic_numbers
    print(f"  static species_z         {'ok' if ok else 'DIFFERENT'}")
for i, layer in enumerate(model.layers):
    st = static.get(f'layers/{i}', {})
    if 'tp_irreps' in st:
        ok = o3.Irreps(st['tp_irreps']) == layer.tp_irreps
        print(f"  layers/{i} tp_irreps {'ok' if ok else 'DIFFERENT: jax ' + st['tp_irreps'] + ' torch ' + str(layer.tp_irreps)}")
        if not ok:
            worst[f'tp_irreps {i}'] = float('inf')

# ------------------------------------------------------------------------------------------- frames
files = sorted(ref.glob('frame_*.npz'))
if a.frames:
    want = {int(x) for x in a.frames.split(',')}
    files = [f for f in files if int(f.stem.split('_')[1]) in want]
for fp in files:
    z = np.load(fp)
    n_node, n_edge = z['graph_n_node'].astype(np.int64), z['graph_n_edge'].astype(np.int64)
    # jraph.get_number_of_padding_with_graphs_graphs: trailing empty graphs + the padding graph
    n_trail = int(np.argmax(np.cumsum(n_node[::-1]) > 0))
    n_real_g = len(n_node) - (n_trail + 1)
    n_real = int(n_node[:n_real_g].sum())
    e_real = int(n_edge[:n_real_g].sum())
    T = lambda k, dt=torch.float64: torch.as_tensor(z[k]).to(dt)
    pos, species = T('graph_nodes_positions'), T('graph_nodes_species', torch.long)
    snd, rcv = T('graph_senders', torch.long), T('graph_receivers', torch.long)
    node_mask = torch.arange(len(species)) < n_real
    periodic = 'graph_edges_Sij' in z.files and model.periodic
    t0 = time.time()
    out = model(pos, species, snd, rcv,
                shifts=T('graph_edges_Sij') if periodic else None, cell=T('graph_globals_cell') if periodic else None,
                n_node=torch.as_tensor(n_node), n_edge=torch.as_tensor(n_edge),
                cutoff=T('graph_globals_cutoff') if periodic else None,
                volume=T('graph_globals_volume') if periodic else None, node_mask=node_mask, record=True)
    dt_run = time.time() - t0
    ctx, rec = out['ctx'], out['ctx']['record']
    print(f"{fp.name}: {n_real} atoms, {e_real} edges (padded {len(species)} / {len(snd)}), {dt_run:.1f} s")
    N, E = slice(0, n_real), slice(0, e_real)
    cmp('prelude_features', ctx['features'][N], z['prelude_features'][N])
    cmp('prelude_radial_basis', ctx['radial_basis'][E], z['prelude_radial_basis'][E])
    cmp('prelude_sh', ctx['sh'][E], z['prelude_sh'][E])
    cmp('prelude_r_norm', ctx['r_norm'][E], z['prelude_r_norm'][E])
    if 'prelude_shell_env' in z.files:
        cmp('prelude_shell_env', ctx['shell_env'][E], z['prelude_shell_env'][E])
    cmp('prelude_atomic_basis', ctx['atomic_basis'][N], z['prelude_atomic_basis'][N])
    n_pre = z['prelude_strata'].shape[0] if z['prelude_strata'].ndim == 2 else 0
    if n_pre:
        cmp('prelude_strata (UFF)', torch.stack(ctx['outputs'][:n_pre])[:, N], z['prelude_strata'][:, N])
    for i in range(model.n_layers):
        for k in ('in', 'x', 'out', 'stratum'):
            cmp(f'layer{i}_{k}', rec[f'layer{i}_{k}'][N], z[f'layer{i}_{k}'][N])
        for k in ('node', 'x', 'f'):
            cmp(f'layer{i}_part_{k}', ctx['layer_parts'][k][i][N], z[f'layer{i}_part_{k}'][N])
    cmp('node_energies', out['model_out']['node_energies'][N], z['node_energies'][N])
    cmp('result_energy', out['energy'][:n_real_g], z['result_energy'][:n_real_g])
    cmp('result_forces', out['forces'][N], z['result_forces'][N])
    if 'stress' in out and 'result_stress' in z.files:
        cmp('result_stress', out['stress'][:n_real_g], z['result_stress'][:n_real_g])
    del out, ctx, rec

# ------------------------------------------------------------------------------------------- ASE calculator
if (ref / 'frames.extxyz').exists():
    import ase.io
    from racev2_torch.calculator import RACEv2Calculator
    frames = ase.io.read(ref / 'frames.extxyz', ':')
    calc = RACEv2Calculator(model, jax_compat=True, edge_chunk=a.edge_chunk)
    print("ASE calculator (own neighbour list; geometry = the positions of the stored graph and the extxyz cell, "
          "since frames.extxyz is not bit-identical to them; E0 float64 added to the reference as predict.py does):")
    from racev2_torch.calculator import neighbor_graph
    for fp in files:
        fi = int(fp.stem.split('_')[1])
        z = np.load(fp)
        at = frames[fi].copy()
        n = len(at)
        g_pos = z['graph_nodes_positions'][:n]
        dpos = float(np.abs(at.positions.astype(np.float32).astype(np.float64) - g_pos).max())
        dcell = float(np.abs(at.cell.array.astype(np.float32).astype(np.float64) - z['graph_globals_cell'][0]).max())
        at.positions = g_pos          # the cell stays the extxyz one: jax_compat rounds it (and V) to float32 like JAX
        at.calc = calc
        e = at.get_potential_energy()
        sp = z['graph_nodes_species'][:n]
        e_ref = float(z['result_energy'][0]) + float(np.sum(model.atom_energies.numpy()[sp]))
        n_e = len(neighbor_graph(at, model.cutoff)[0])
        print(f"  frame {fi:02d}: edges {n_e} (jax {int(z['graph_n_edge'][0])}); info: extxyz positions differ from "
              f"the graph positions by up to {dpos:.1e} A after float32 rounding, cell {dcell:.1e}")
        if n_e != int(z['graph_n_edge'][0]):
            worst['calc edge count'] = float('inf')
        cmp('calc energy (E0 incl.)', np.array([e]), np.array([e_ref]))
        cmp('calc forces', at.get_forces(), z['result_forces'][:n])
        if 'result_stress' in z.files and np.all(at.pbc):
            cmp('calc stress', at.get_stress(), z['result_stress'][0])

print("== worst over all frames (max abs difference)")
bad = 0
for k, v in worst.items():
    flag = '' if v <= a.tol else '   <-- above tol'
    bad += v > a.tol
    print(f"  {k:28s} {v:10.3e}{flag}")
print(f"{len(worst)} quantities, {bad} above {a.tol:g}")
