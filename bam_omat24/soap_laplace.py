"""SOAP-curvature Laplace posterior over all weights (docs/SOAP_Laplace.md).  Run 2026-11-09 on the local 4G carbon
chain and AuMgO (docs/local_test_report.md section 18): kappa -> inf reproduces the network, but the second moments
(optimizer nu, or recollected per-structure squared gradients at the EMA weights, --recollect) are an empirical
Fisher: tiny at convergence (median 1e-15) and concentrated in a few directions, so the posterior is either the
isotropic prior with the global `scale` frozen (small kappa) or over-confident (large kappa).  Not a usable
curvature; a GGN would be needed.  Kept for reference.

Posterior precision of every SOAP matrix leaf, diagonal in the SOAP eigenbasis:
    Lambda = (Q_L (x) Q_R) diag(kappa * nu + lam) (Q_L (x) Q_R)^T      (nu: Adam second moment in the rotated basis)
Adam leaves: diag(kappa * nu + lam).  Skip leaves: fixed.  Samples are drawn around the EMA weights.
Periodic cells supported since 2026-11-09 (cell and cutoff globals; forces only, no stress yet).

    python -m bam_omat24.soap_laplace <config.json> --ckpt <ckpt.pkl> --out <dir> [--samples 16]
        [--kappa 1e3 1e4 1e5 1e6] [--lam 1e2 1e3 1e4]      (grid; chosen on validation NLL)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
    ap.add_argument('--max-valid', type=int, default=None, help='validation subset for the (kappa, lam) grid')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-bucket', type=int, default=32); ap.add_argument('--e-bucket', type=int, default=1024)
    ap.add_argument('--recollect', type=int, default=0,
                    help='refinement 1: replace the optimizer nu by the mean squared per-structure gradient of the '
                         'training loss at the EMA weights, in the SOAP basis (empirical Fisher diagonal, EKFAC '
                         'style), from this many random training structures; kappa then scales as the number of '
                         'training structures')
    a = ap.parse_args()
    t_start = time.time()
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
    # scale of the second moments: kappa * nu competes with lam, so the grid must bracket lam / nu
    nu_all = np.concatenate([np.asarray(cols[1][i], np.float64).ravel() for i, sp in enumerate(specs)
                             if sp.kind == 'soap'])
    q = np.quantile(nu_all, [0.1, 0.5, 0.9, 0.999])
    print(f"nu over SOAP leaves: 10 % {q[0]:.2e}, median {q[1]:.2e}, 90 % {q[2]:.2e}, 99.9 % {q[3]:.2e}; "
          f"kappa * median(nu) = lam at kappa = lam / {q[1]:.2e}", flush=True)

    import jraph
    import jax.numpy as jnp

    periodic = bool(cfg.get('periodic', False))

    if a.recollect:
        from bam_omat24.training_utils import compute_loss
        t0 = time.time()
        train = [tr[i] for i in range(len(tr))]
        idx = np.sort(np.random.default_rng(a.seed).choice(len(train), min(a.recollect, len(train)), replace=False))
        ema_state = jax.tree.unflatten(treedef, [jnp.asarray(x) for x in ema_leaves])
        soap_idx = [i for i, sp in enumerate(specs) if sp.kind == 'soap']
        adam_idx = [i for i, sp in enumerate(specs) if sp.kind == 'adam']
        QL = [jnp.asarray(cols[4][i]) for i in soap_idx]; QR = [jnp.asarray(cols[5][i]) for i in soap_idx]
        lw = dict(energy_weight=float(cfg.get('energy_weight', 1.0)), force_weight=float(cfg.get('force_weight', 10.0)),
                  stress_weight=float(cfg.get('stress_weight', 1.0)) if cfg.get('stress_loss') else 0.0,
                  huber_delta=float(cfg.get('huber_delta', 0.1)), loss_type=cfg.get('loss_type', 'huber'),
                  periodic=bool(cfg.get('stress_loss', False)), energy_loss_per=cfg.get('energy_loss_per', 'atom'))

        def _sq_grads(state, b):
            g = jax.grad(lambda p: compute_loss(graphdef, p, b, lw['energy_weight'], lw['force_weight'],
                                                lw['stress_weight'], lw['huber_delta'], loss_type=lw['loss_type'],
                                                periodic=lw['periodic'], energy_loss_per=lw['energy_loss_per'])[0])(state)
            gl = treedef.flatten_up_to(g)
            out_s = []
            for k, i in enumerate(soap_idx):
                sp = specs[i]
                G = gl[i].astype(sp.dtype).reshape(sp.mat_shape)
                if sp.left and QL[k].ndim >= 2:
                    G = jnp.swapaxes(QL[k], -1, -2) @ G
                if sp.right and QR[k].ndim >= 2:
                    G = G @ QR[k]
                out_s.append(G ** 2)
            return out_s, [gl[i] ** 2 for i in adam_idx]
        sq = jax.jit(_sq_grads)
        acc_s = acc_a = None
        for j, r in enumerate(idx):
            b = collate(train[r], a.n_bucket, a.e_bucket)
            s_, a_ = sq(ema_state, b)
            acc_s = s_ if acc_s is None else [x + y for x, y in zip(acc_s, s_)]
            acc_a = a_ if acc_a is None else [x + y for x, y in zip(acc_a, a_)]
        M = len(idx)
        nu_new = list(cols[1])
        for k, i in enumerate(soap_idx):
            nu_new[i] = np.asarray(acc_s[k] / M, np.float64).reshape(np.shape(cols[1][i]))
        for k, i in enumerate(adam_idx):
            nu_new[i] = np.asarray(acc_a[k] / M, np.float64).reshape(np.shape(cols[1][i]))
        cols = [cols[0], nu_new] + list(cols[2:])
        nu_all = np.concatenate([nu_new[i].ravel() for i in soap_idx])
        q = np.quantile(nu_all, [0.1, 0.5, 0.9, 0.999, 1.0])
        print(f"recollected nu at the EMA weights from {M} training structures ({time.time() - t0:.0f} s): SOAP leaves "
              f"10 % {q[0]:.2e}, median {q[1]:.2e}, 90 % {q[2]:.2e}, 99.9 % {q[3]:.2e}, max {q[4]:.2e}; Adam leaves max "
              f"{max(float(np.max(nu_new[i])) for i in adam_idx):.2e}", flush=True)

    def _ef(state, b):
        # energy in the training target space (sum of node energies, E0 excluded) and forces, as bayes_lastlayer;
        # periodic cells (2026-11-09): node energies with the cell and cutoff globals of the graph
        m = nnx.merge(graphdef, state)
        mask = jraph.get_node_padding_mask(b)

        def e(pos):
            cell, cut = (b.globals['cell'], b.globals['cutoff']) if periodic else (None, None)
            ne = m.node_energies(pos, cell, cut, b)['node_energies'].reshape(-1)
            return jnp.sum(jnp.where(mask, ne, 0.0))
        en, g = jax.value_and_grad(e)(b.nodes['positions'])
        return en, -jnp.where(jnp.isfinite(g), g, 0.0)
    ef = jax.jit(_ef)

    def predict(param_leaves, graphs):
        state = jax.tree.unflatten(treedef, [jnp.asarray(x) for x in param_leaves])
        E, F = [], []
        for g in graphs:
            n = int(np.asarray(g.n_node)[0])
            en, f = ef(state, collate(g, a.n_bucket, a.e_bucket))
            E.append(float(en)); F.append(np.asarray(f)[:n].reshape(-1))
        return np.asarray(E), F

    def mc(graphs, kappa, lam, seed=0):
        rng = np.random.default_rng(seed)
        Es, Fs = [], []
        for _ in range(a.samples):
            E, F = predict(sample_params(ema_leaves, specs, cols, kappa, lam, rng), graphs)
            Es.append(E); Fs.append(np.concatenate(F))
        return np.stack(Es), np.stack(Fs)

    def targets(gs):
        return (np.array([float(np.asarray(g.globals['energy']).reshape(-1)[0]) for g in gs]),
                np.concatenate([np.asarray(g.nodes['forces'], np.float64).reshape(-1) for g in gs]))

    def evaluate(gs, kappa, lam, seed, netE, netF):
        """Calibration with two centres: the sample mean ('mean') and the network itself ('map', sigma = sample std)."""
        Es, Fs = mc(gs, kappa, lam, seed)
        if not (np.isfinite(Es).all() and np.isfinite(Fs).all()):
            return None
        yE, yF = targets(gs)
        sE, sF = 1e3 * Es.std(0, ddof=1) + 1e-6, 1e3 * Fs.std(0, ddof=1) + 1e-6
        return {'E_mean': calib(1e3 * (Es.mean(0) - yE), sE), 'F_mean': calib(1e3 * (Fs.mean(0) - yF), sF),
                'E_map': calib(1e3 * (netE - yE), sE), 'F_map': calib(1e3 * (netF - yF), sF),
                'net_E_rmse_meV': 1e3 * float(np.sqrt(np.mean((netE - yE) ** 2))),
                'net_F_rmse_meVA': 1e3 * float(np.sqrt(np.mean((netF - yF) ** 2)))}

    valid = [va[i] for i in range(len(va))]
    if a.max_valid and len(valid) > a.max_valid:
        idx = np.sort(np.random.default_rng(a.seed).choice(len(valid), a.max_valid, replace=False))
        valid = [valid[i] for i in idx]
    # check: kappa -> infinity gives the network back (samples collapse onto the EMA weights)
    netE, netF = predict(ema_leaves, valid); netF = np.concatenate(netF)
    E_inf, F_inf = predict(sample_params(ema_leaves, specs, cols, 1e300, 1e300, np.random.default_rng(0)), valid)
    dev = max(np.abs(E_inf - netE).max(), np.abs(np.concatenate(F_inf) - netF).max())
    yE, yF = targets(valid)
    print(f"check: kappa, lam -> inf reproduces the network to {dev:.1e} (E eV / F eV/A); network on valid "
          f"({len(valid)}): E rmse {1e3 * np.sqrt(np.mean((netE - yE) ** 2)):.2f} meV, "
          f"F rmse {1e3 * np.sqrt(np.mean((netF - yF) ** 2)):.2f} meV/A; {time.time() - t_start:.0f} s", flush=True)
    best, grid = None, {}
    for kappa in a.kappa:
        for lam in a.lam:
            r = evaluate(valid, kappa, lam, a.seed, netE, netF)
            if r is None:
                print(f"kappa {kappa:g} lam {lam:g}: non-finite predictions", flush=True); continue
            grid[f'{kappa:g},{lam:g}'] = r
            cE, cF, mF = r['E_map'], r['F_map'], r['F_mean']
            print(f"kappa {kappa:g} lam {lam:g}: valid F(map) nll {cF['nll']:.3f} z {cF['rms_z']:.2f} "
                  f"sigma {cF['mean_sigma']:.2f} rho {cF['spearman']:.2f} | F(mean) rmse {mF['rmse']:.2f} "
                  f"| E(map) nll {cE['nll']:.3f} z {cE['rms_z']:.2f} rho {cE['spearman']:.2f}; "
                  f"{time.time() - t_start:.0f} s", flush=True)
            score = cF['nll']                       # forces: no per-composition energy offsets in the ranking
            if best is None or score < best[0]:
                best = (score, kappa, lam)
    if best is None:
        raise RuntimeError('no finite (kappa, lam)')
    _, kappa, lam = best
    print(f"chosen kappa {kappa:g} lam {lam:g} (validation force NLL, network as the mean)", flush=True)
    report = {'kappa': kappa, 'lam': lam, 'samples': a.samples, 'check_inf': float(dev), 'grid': grid, 'sets': {}}
    for name, ds in te.items():
        gs = [ds[i] for i in range(len(ds))][:a.max_test]
        nE, nF = predict(ema_leaves, gs)
        r = evaluate(gs, kappa, lam, a.seed + 1, nE, np.concatenate(nF))
        report['sets'][name] = r
        print(f"{name} n={len(gs)}: net F {r['net_F_rmse_meVA']:.2f} | F(map) sigma {r['F_map']['mean_sigma']:.2f} "
              f"z {r['F_map']['rms_z']:.2f} cov68 {r['F_map']['cov68']:.2f} cov95 {r['F_map']['cov95']:.2f} "
              f"rho {r['F_map']['spearman']:.2f} | F(mean) rmse {r['F_mean']['rmse']:.2f} z {r['F_mean']['rms_z']:.2f}"
              f" | E(map) rmse {r['E_map']['rmse']:.2f} sigma {r['E_map']['mean_sigma']:.2f} z {r['E_map']['rms_z']:.2f} "
              f"rho {r['E_map']['spearman']:.2f}", flush=True)
    report['time_s'] = time.time() - t_start
    json.dump(report, open(out / 'soap_laplace.json', 'w'), indent=1)
    print(f"written {out / 'soap_laplace.json'} ({report['time_s']:.0f} s)")


if __name__ == '__main__':
    main()
