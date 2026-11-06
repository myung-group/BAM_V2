"""SOAP-curvature Laplace posterior over all weights (docs/SOAP_Laplace.md).  UNTESTED skeleton (2026-11-06,
written while the compute allocation was exhausted): run the checks of docs/SOAP_Laplace.md section 5 first.

Posterior precision of every SOAP matrix leaf, diagonal in the SOAP eigenbasis:
    Lambda = (Q_L (x) Q_R) diag(kappa * nu + lam) (Q_L (x) Q_R)^T      (nu: Adam second moment in the rotated basis)
Adam leaves: diag(kappa * nu + lam).  Skip leaves: fixed.  Samples are drawn around the EMA weights.
Molecular (non-periodic) graphs only for now (node_energies called with cell = cutoff = None, as in
bayes_lastlayer); check that the treedef of ckpt['params'] matches the nnx state of the merged model.

    python -m bam_omat24.soap_laplace <config.json> --ckpt <ckpt.pkl> --out <dir> [--samples 16]
        [--kappa 1e3 1e4 1e5 1e6] [--lam 1e2 1e3 1e4]      (grid; chosen on validation NLL)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def find_soap_state(opt_state):
    """The SOAP state inside an optax chain state (per-leaf ScaleBySoapState or ScaleBySoapGroupedState)."""
    import jax
    hits = []

    def visit(x):
        if hasattr(x, '_fields') and {'nu', 'ql', 'qr'} <= set(x._fields):
            hits.append(x)
            return
        if isinstance(x, (tuple, list)):
            for y in x:
                visit(y)
        elif isinstance(x, dict):
            for y in x.values():
                visit(y)
    visit(opt_state)
    if len(hits) != 1:
        raise ValueError(f"expected one SOAP state in the optimizer state, found {len(hits)}")
    return hits[0]


def per_leaf_state(cfg, params, st, n_species):
    """Per-leaf (mu, nu, gl, gr, ql, qr) lists, converting a grouped state with the trainer's options."""
    import jax
    leaves, treedef = jax.tree.flatten(params)
    if 'adam_mu' in st._fields:      # grouped layout
        from bam_omat24.soap_grouped import soap_grouped_build
        kw = dict(b1=float(cfg.get('soap_b1', 0.95)), b2=float(cfg.get('soap_b2', 0.95)),
                  precondition_frequency=int(cfg.get('soap_precondition_frequency', 10)),
                  max_precond_dim=int(cfg.get('soap_max_precond_dim', 10000)),
                  shampoo_beta=cfg.get('soap_shampoo_beta', None), orth=cfg.get('soap_orth', 'householder'),
                  state_dtype=cfg.get('soap_state_dtype', None),
                  adam_leading_dim=(n_species if cfg.get('soap_species_adam', False) else None))
        _tx, _to_grouped, to_per_leaf = soap_grouped_build(**kw)
        st = to_per_leaf(st, leaves, treedef)
    cols = [list(treedef.flatten_up_to(x)) for x in st[1:]]
    return leaves, treedef, cols


def leaf_specs(cfg, leaves):
    from bam_omat24.soap_optax import _spec
    import jax.numpy as jnp
    max_dim = int(cfg.get('soap_max_precond_dim', 10000))
    specs = [_spec(jnp.shape(x), jnp.result_type(x), max_dim, False) for x in leaves]
    if cfg.get('soap_species_adam', False):
        n_sp = len(cfg['atomic_numbers'])
        from bam_omat24.soap_optax import _Spec
        specs = [_Spec('adam') if (sp.kind == 'soap' and len(jnp.shape(x)) >= 2 and int(jnp.shape(x)[0]) == n_sp)
                 else sp for sp, x in zip(specs, leaves)]
    return specs


def sample_params(center_leaves, specs, cols, kappa, lam, rng):
    """One posterior sample (list of leaves) around center_leaves."""
    mu_l, nu_l, gl_l, gr_l, ql_l, qr_l = cols
    out = []
    for i, (c, sp) in enumerate(zip(center_leaves, specs)):
        c = np.asarray(c)
        if sp.kind == 'skip' or not np.issubdtype(c.dtype, np.floating):
            out.append(c)
            continue
        nu = np.asarray(nu_l[i], np.float64)
        if sp.kind == 'adam':
            z = rng.standard_normal(c.shape) / np.sqrt(kappa * nu.reshape(c.shape) + lam)
            out.append((c + z).astype(c.dtype))
            continue
        ms = sp.mat_shape
        z = rng.standard_normal(ms) / np.sqrt(kappa * nu.reshape(ms) + lam)
        ql, qr = np.asarray(ql_l[i]), np.asarray(qr_l[i])
        if sp.left and ql.ndim >= 2 and ql.shape[-1] == ms[-2]:
            z = ql @ z
        if sp.right and qr.ndim >= 2 and qr.shape[-1] == ms[-1]:
            z = z @ np.swapaxes(qr, -1, -2)
        out.append((c + z.reshape(c.shape)).astype(c.dtype))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('config'); ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--samples', type=int, default=16)
    ap.add_argument('--kappa', type=float, nargs='+', default=[1e3, 1e4, 1e5, 1e6])
    ap.add_argument('--lam', type=float, nargs='+', default=[1e2, 1e3, 1e4])
    ap.add_argument('--max-test', type=int, default=None)
    a = ap.parse_args()
    import jax
    from flax import nnx
    from bam_omat24.training_utils import load_checkpoint
    from bam_omat24.bayes_lastlayer import build, calib, load_xyz_graphs, collate
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg, load, model, zs, e0, tr, va, te = build(a.config, a.ckpt)
    ck = load_checkpoint(a.ckpt)
    st = find_soap_state(ck['opt_state'])
    leaves, treedef, cols = per_leaf_state(cfg, ck['params'], st, len(zs))
    specs = leaf_specs(cfg, leaves)
    ema_leaves = treedef.flatten_up_to(ck['ema_params'])
    graphdef, _ = nnx.split(model)
    print(f"SOAP state: {sum(sp.kind == 'soap' for sp in specs)} SOAP leaves, {sum(sp.kind == 'adam' for sp in specs)} "
          f"Adam leaves, {sum(sp.kind == 'skip' for sp in specs)} fixed", flush=True)

    import jraph
    import jax.numpy as jnp

    def _ef(state, b):
        # energy in the training target space (sum of node energies, E0 excluded) and forces, as bayes_lastlayer
        m = nnx.merge(graphdef, state)
        mask = jraph.get_node_padding_mask(b)

        def e(pos):
            ne = m.node_energies(pos, None, None, b)['node_energies'].reshape(-1)
            return jnp.sum(jnp.where(mask, ne, 0.0))
        en, g = jax.value_and_grad(e)(b.nodes['positions'])
        return en, -g
    ef = jax.jit(_ef)

    def predict(param_leaves, graphs):
        state = jax.tree.unflatten(treedef, [jnp.asarray(x) for x in param_leaves])
        E, F = [], []
        for g in graphs:
            n = int(np.asarray(g.n_node)[0])
            en, f = ef(state, collate(g, 32, 1024))
            E.append(float(en)); F.append(np.asarray(f)[:n].reshape(-1))
        return np.asarray(E), F

    def mc(graphs, kappa, lam, seed=0):
        rng = np.random.default_rng(seed)
        Es, Fs = [], []
        for _ in range(a.samples):
            E, F = predict(sample_params(ema_leaves, specs, cols, kappa, lam, rng), graphs)
            Es.append(E); Fs.append(np.concatenate(F))
        return np.stack(Es), np.stack(Fs)

    valid = [va[i] for i in range(len(va))]
    yE = np.array([float(np.asarray(g.globals['energy']).reshape(-1)[0]) for g in valid])
    yF = np.concatenate([np.asarray(g.nodes['forces'], np.float64).reshape(-1) for g in valid])
    best = None
    for kappa in a.kappa:
        for lam in a.lam:
            Es, Fs = mc(valid, kappa, lam)
            if not (np.isfinite(Es).all() and np.isfinite(Fs).all()):
                print(f"kappa {kappa:g} lam {lam:g}: non-finite predictions", flush=True); continue
            cE = calib(1e3 * (Es.mean(0) - yE), 1e3 * Es.std(0, ddof=1) + 1e-6)
            cF = calib(1e3 * (Fs.mean(0) - yF), 1e3 * Fs.std(0, ddof=1) + 1e-6)
            print(f"kappa {kappa:g} lam {lam:g}: valid E nll {cE['nll']:.3f} z {cE['rms_z']:.2f} rho {cE['spearman']:.2f}"
                  f" | F nll {cF['nll']:.3f} z {cF['rms_z']:.2f} rho {cF['spearman']:.2f}", flush=True)
            score = cE['nll'] + cF['nll']
            if best is None or score < best[0]:
                best = (score, kappa, lam)
    if best is None:
        raise RuntimeError('no finite (kappa, lam)')
    _, kappa, lam = best
    report = {'kappa': kappa, 'lam': lam, 'samples': a.samples, 'sets': {}}
    for name, ds in te.items():
        gs = [ds[i] for i in range(len(ds))][:a.max_test]
        Es, Fs = mc(gs, kappa, lam, seed=1)
        yE = np.array([float(np.asarray(g.globals['energy']).reshape(-1)[0]) for g in gs])
        yF = np.concatenate([np.asarray(g.nodes['forces'], np.float64).reshape(-1) for g in gs])
        report['sets'][name] = {'E': calib(1e3 * (Es.mean(0) - yE), 1e3 * Es.std(0, ddof=1) + 1e-6),
                                'F': calib(1e3 * (Fs.mean(0) - yF), 1e3 * Fs.std(0, ddof=1) + 1e-6)}
        print(name, json.dumps(report['sets'][name]), flush=True)
    json.dump(report, open(out / 'soap_laplace.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
