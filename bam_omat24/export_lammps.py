"""Export a trained RACEv2 model as StableHLO for a C++ (LAMMPS) runtime -- step 1 of the LAMMPS plan
(docs/RACEv2_lammps_export_plan_2026-11-06.md).

Exported function, one per padded shape bucket (n_pad atoms, e_pad edges), float64:

    (species i32[n_pad], receivers i32[e_pad], senders i32[e_pad], rvec f64[e_pad, 3], n_real i32[], e_real i32[])
        -> (e_total f64[], e_atom f64[n_pad], dE_drvec f64[e_pad, 3])

  * receivers = centre atom i, senders = neighbour j (owned index of the image), rvec = x_j(image) - x_i, i.e. the
    edge vector of the training graphs (data_3bpa.preprocess_graph: r = x_sender - (x_receiver + Sij . cell)).
    The model is evaluated with zero positions, identity cell and Sij = -rvec, so its edge vectors are exactly rvec;
    nothing else in RACEv2 depends on absolute positions.
  * padding: atoms n_real.. are padding (species 0); edges e_real.. must be (n_real, n_real, rvec = 0) -- r = 0
    edges carry no message (race_nnx, 2026-11-06) and padding-atom energies are masked out.
  * energies exclude the isolated-atom / regression E0 (l_train); the per-species E0 table is in the metadata, add it
    in the caller.  Forces and virial follow from dE_drvec: F_i += g_e, F_j -= g_e (F = -dE/dx, r_e = x_j - x_i),
    virial W = -sum_e r_e (x) g_e.

    python -m bam_omat24.export_lammps <config.json> --ckpt <ckpt.pkl> --out <dir> [--backend e3nn|cue]
        [--buckets 128:4096 256:8192 ...] [--check <extxyz>]

--check compares, for every frame of the extxyz, the exported function (deserialised and run by JAX) against the
normal model call (energy, forces, stress); run it on a GPU node (no JAX on the login nodes).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def parse_buckets(items):
    return [tuple(int(v) for v in s.split(':')) for s in items]


def build(cfg_path, ckpt, backend):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from bam_omat24 import train_3bpa as T
    from bam_omat24.training_utils import load_checkpoint
    cfg = json.load(open(cfg_path))
    cfg['x64'] = True
    cfg['periodic'] = True                       # the weights do not depend on it; periodic path takes Sij / cell
    cfg['tp_backend'] = backend
    jax.config.update('jax_enable_x64', True)
    (zs, e0, tr, va, te, stats) = T.build_datasets(cfg, sys.stderr)
    graphdef, _p, *_ = T.build_model(cfg, len(zs), e0, stats, sys.stderr)
    params = jax.tree.map(lambda x: jnp.asarray(x, jnp.float64) if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating)
                          else jnp.asarray(x), load_checkpoint(ckpt)['ema_params'])
    model = nnx.merge(graphdef, params)
    return cfg, model, [int(z) for z in zs], [float(x) for x in e0]


def make_fn(model, cutoff, n_pad, e_pad):
    import jax
    import jax.numpy as jnp
    import jraph

    def fn(species, receivers, senders, rvec, n_real, e_real):
        n_node = jnp.stack([n_real, n_pad - n_real]).astype(jnp.int32)
        n_edge = jnp.stack([e_real, e_pad - e_real]).astype(jnp.int32)
        node_mask = jnp.arange(n_pad) < n_real
        cell = jnp.broadcast_to(jnp.eye(3, dtype=rvec.dtype), (2, 3, 3))
        cut = jnp.array([cutoff, 0.0], dtype=rvec.dtype)
        pos0 = jnp.zeros((n_pad, 3), rvec.dtype)

        def etot(rv):
            g = jraph.GraphsTuple(
                nodes={'species': species, 'positions': pos0}, edges={'Sij': -rv},
                senders=senders, receivers=receivers, n_node=n_node, n_edge=n_edge,
                globals={'cell': cell, 'cutoff': cut, 'volume': jnp.ones((2,), rv.dtype)})
            ne = model.node_energies(pos0, cell, cut, g)['node_energies'].reshape(-1)
            ne = jnp.where(node_mask, ne, 0.0)
            return jnp.sum(ne), ne
        (e, ne), g = jax.value_and_grad(etot, has_aux=True)(rvec)
        return e, ne, g
    return fn


def export(args):
    import jax
    import jax.numpy as jnp
    cfg, model, zs, e0 = build(args.config, args.ckpt, args.backend)
    cutoff = float(cfg.get('cutoff', 5.0))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta = {'config': str(args.config), 'ckpt': str(args.ckpt), 'backend': args.backend, 'dtype': 'float64',
            'cutoff': cutoff, 'n_layers': int(cfg.get('n_layers', 5)), 'atomic_numbers': zs, 'e0': e0,
            'signature': 'species,receivers,senders,rvec,n_real,e_real -> e_total,e_atom,dE_drvec',
            'buckets': []}
    fns = {}
    for n_pad, e_pad in parse_buckets(args.buckets):
        fn = jax.jit(make_fn(model, cutoff, n_pad, e_pad))
        sds = (jax.ShapeDtypeStruct((n_pad,), jnp.int32), jax.ShapeDtypeStruct((e_pad,), jnp.int32),
               jax.ShapeDtypeStruct((e_pad,), jnp.int32), jax.ShapeDtypeStruct((e_pad, 3), jnp.float64),
               jax.ShapeDtypeStruct((), jnp.int32), jax.ShapeDtypeStruct((), jnp.int32))
        exp = jax.export.export(fn, platforms=['cuda'])(*sds)
        stem = f'racev2_N{n_pad}_E{e_pad}'
        (out / f'{stem}.jaxexport').write_bytes(exp.serialize())                # for jax.export.deserialize
        (out / f'{stem}.mlirbc').write_bytes(exp.mlir_module_serialized)        # StableHLO bytecode for PJRT
        meta['buckets'].append({'n_pad': n_pad, 'e_pad': e_pad, 'file': f'{stem}.mlirbc',
                                'jaxexport': f'{stem}.jaxexport'})
        txt = exp.mlir_module()
        n_cc = txt.count('custom_call')
        print(f'exported {stem}: {len(exp.mlir_module_serialized) / 1e6:.1f} MB, custom_call ops {n_cc}', flush=True)
        if n_cc:
            targets = sorted({l.split('@')[1].split('(')[0] for l in txt.splitlines() if 'custom_call @' in l})
            print('  custom_call targets:', targets)
            meta['buckets'][-1]['custom_calls'] = targets
        fns[(n_pad, e_pad)] = exp
    json.dump(meta, open(out / 'racev2_lammps_meta.json', 'w'), indent=1)
    print('written', out / 'racev2_lammps_meta.json')
    if args.check:
        check(args, cfg, model, zs, e0, meta, fns)


def graph_inputs(atoms, zs, cutoff, n_pad, e_pad):
    from ase.neighborlist import neighbor_list
    idx = {z: i for i, z in enumerate(sorted(zs))}
    i, j, D = neighbor_list('ijD', atoms, cutoff)        # D = x_j(image) - x_i
    n, e = len(atoms), len(i)
    if n >= n_pad or e > e_pad:
        return None
    species = np.zeros(n_pad, np.int32); species[:n] = [idx[int(z)] for z in atoms.numbers]
    rec = np.full(e_pad, n, np.int32); snd = np.full(e_pad, n, np.int32); rv = np.zeros((e_pad, 3))
    rec[:e], snd[:e], rv[:e] = i, j, D
    return species, rec, snd, rv, np.int32(n), np.int32(e)


def check(args, cfg, model, zs, e0, meta, fns):
    import ase.io
    import jax
    import jax.numpy as jnp
    from bam_omat24.data_3bpa import preprocess_graph, atomic_numbers_to_indices
    from bam_omat24 import train_3bpa as T
    from bam_omat24.data_nnx import default_collate_fn
    cutoff = meta['cutoff']
    frames = ase.io.read(args.check, index=':%d' % args.n_check)
    worst = {'E': 0.0, 'F': 0.0, 'S': 0.0}
    for a in frames:
        a.pbc = True if a.cell.rank == 3 else a.pbc
        key = next(((n, e) for (n, e) in sorted(fns) if len(a) < n), None)
        ins = graph_inputs(a, zs, cutoff, *key) if key else None
        if ins is None:
            print('frame does not fit any bucket', len(a)); continue
        e, ne, g = fns[key].call(*[jnp.asarray(x) for x in ins])
        n, ne_ = int(ins[4]), int(ins[5])
        rec, snd, rv, g = ins[1][:ne_], ins[2][:ne_], ins[3][:ne_], np.asarray(g)[:ne_]
        F = np.zeros((n, 3)); np.add.at(F, rec, g); np.add.at(F, snd, -g)
        W = -(rv[:, :, None] * g[:, None, :]).sum(0)
        stress = W / a.get_volume() * -1.0          # ASE stress = (1/V) dE/d(strain) = -W / V
        # reference: the normal model call (training graph path)
        gr = preprocess_graph(a, atomic_numbers_to_indices(zs), e0, cutoff, targets=False)
        gr = T._cast_graphs([gr], np.float64)[0]
        b = default_collate_fn([gr], batch_size=1, node_bucket_size=64, edge_bucket_size=1024)
        r = model(jax.tree.map(jnp.asarray, b))
        Er, Fr, Sr = float(np.asarray(r['energy'])[0]), np.asarray(r['forces'])[:n], np.asarray(r['stress'])[0]
        S6 = stress[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
        worst['E'] = max(worst['E'], abs(float(e) - Er)); worst['F'] = max(worst['F'], np.abs(F - Fr).max())
        worst['S'] = max(worst['S'], np.abs(S6 - Sr).max())
    print(f"check on {len(frames)} frames of {args.check}: max |dE| {worst['E']:.2e} eV, max |dF| {worst['F']:.2e} eV/A, "
          f"max |dS| {worst['S']:.2e} eV/A^3", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('config'); ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--backend', choices=['e3nn', 'cue'], default='e3nn')
    ap.add_argument('--buckets', nargs='+', default=['128:4096', '256:8192', '512:16384', '1024:32768'])
    ap.add_argument('--check', default=None); ap.add_argument('--n-check', type=int, default=5)
    export(ap.parse_args())


if __name__ == '__main__':
    main()
