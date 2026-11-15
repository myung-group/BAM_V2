"""SOAP-basis Laplace posterior over all weights (docs/SOAP_Laplace.md).

Posterior precision of every SOAP matrix leaf, diagonal in the SOAP eigenbasis:
    Lambda = (Q_L (x) Q_R) diag(kappa * nu + lam) (Q_L (x) Q_R)^T
Adam leaves: diag(kappa * nu + lam).  Skip leaves: fixed.  Samples are drawn around the EMA weights.
Periodic cells supported (cell and cutoff globals; forces only, no stress yet).

Three curvatures for nu (docs/local_test_report.md sections 18-19):
  * optimizer nu (default): the SOAP/Adam second moment of the loss gradient.  Run 2026-11-09: an empirical Fisher,
    tiny at convergence (median 1e-15) and concentrated in a few directions -> isotropic prior or over-confident.
  * --recollect N: the same quantity recomputed at the EMA weights from N training structures.  Same picture.
  * --ggn N [--ggn-probes K]: the generalised Gauss-Newton  sum_k J_k^T J_k / sigma_k^2  (J: Jacobian of the
    energy and forces wrt the weights, sigma: noise of the Gaussian likelihood, --sigma-e/--sigma-f), diagonal in
    the SOAP basis (EKFAC-style), averaged over N training structures.  No residual factor, so it does not vanish
    at convergence.  K > 0: Hutchinson estimate with K Gaussian probes per structure (one VJP each);
    K = 0: exact, one VJP per output (1 + 3 n_atoms).  With --ggn the proper Laplace precision is
    n_train * nu + lam; --kappa-times-ntrain makes the kappa grid relative to n_train (kappa = 1 is the untempered
    posterior).

    python -m race_v2.eval.soap_laplace <config.json> --ckpt <ckpt.pkl> --out <dir> [--samples 16]
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
        from race_v2.training.soap_grouped import soap_grouped_build
        kw = dict(b1=float(cfg.get('soap_b1', 0.95)), b2=float(cfg.get('soap_b2', 0.95)),
                  precondition_frequency=int(cfg.get('soap_precondition_frequency', 10)),
                  max_precond_dim=int(cfg.get('soap_max_precond_dim', 10000)),
                  shampoo_beta=cfg.get('soap_shampoo_beta', None), orth=cfg.get('soap_orth', 'householder'),
                  state_dtype=cfg.get('soap_state_dtype', None),
                  adam_leading_dim=(n_species if cfg.get('soap_species_adam', False) else None),
                  adam_any_axis=(cfg.get('soap_species_adam', False) == 'any_axis'))
        _tx, _to_grouped, to_per_leaf = soap_grouped_build(**kw)
        st = to_per_leaf(st, leaves, treedef)
    cols = [list(treedef.flatten_up_to(x)) for x in st[1:]]
    return leaves, treedef, cols


def leaf_specs(cfg, leaves):
    from race_v2.training.soap_optax import _spec
    import jax.numpy as jnp
    max_dim = int(cfg.get('soap_max_precond_dim', 10000))
    specs = [_spec(jnp.shape(x), jnp.result_type(x), max_dim, False) for x in leaves]
    if cfg.get('soap_species_adam', False):
        n_sp = len(cfg['atomic_numbers'])
        from race_v2.training.soap_optax import _Spec
        from race_v2.training.soap_grouped import is_species_table
        any_axis = cfg.get('soap_species_adam') == 'any_axis'
        specs = [_Spec('adam') if (sp.kind == 'soap' and is_species_table(jnp.shape(x), n_sp, any_axis))
                 else sp for sp, x in zip(specs, leaves)]
    return specs


def sample_params(center_leaves, specs, cols, kappa, lam, rng):
    """One posterior sample (list of leaves) around center_leaves.  lam: scalar, or one prior precision per leaf."""
    mu_l, nu_l, gl_l, gr_l, ql_l, qr_l = cols
    per_leaf = np.ndim(lam) > 0
    lam_all = lam
    out = []
    for i, (c, sp) in enumerate(zip(center_leaves, specs)):
        c = np.asarray(c)
        if sp.kind == 'skip' or not np.issubdtype(c.dtype, np.floating):
            out.append(c)
            continue
        lam = float(lam_all[i]) if per_leaf else lam_all
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


def rotated_squares(grad_leaves, specs, cols, soap_idx, adam_idx):
    """Squares of a gradient (list of leaves) in the SOAP eigenbasis: (Q_L^T G Q_R)^2 for SOAP leaves, G^2 for Adam."""
    import jax.numpy as jnp
    out_s = []
    for i in soap_idx:
        sp = specs[i]
        G = grad_leaves[i].astype(sp.dtype).reshape(sp.mat_shape)
        ql, qr = cols[4][i], cols[5][i]
        if sp.left and jnp.ndim(ql) >= 2:
            G = jnp.swapaxes(ql, -1, -2) @ G
        if sp.right and jnp.ndim(qr) >= 2:
            G = G @ qr
        out_s.append(G ** 2)
    return out_s, [grad_leaves[i] ** 2 for i in adam_idx]


def ggn_diag(out_fn, state, batches, specs, cols, treedef, w_e, w_f, probes, seed=0, chunk=16, log=None):
    """Diagonal of the generalised Gauss-Newton  sum_k w_k J_k^T J_k  in the SOAP eigenbasis, averaged over batches.

    out_fn(state, b) -> (E, F, mask): energy (scalar), forces (n_pad, 3) and the node padding mask of batch b.
    w_e = 1 / sigma_E^2, w_f = 1 / sigma_F^2 (Gaussian likelihood precisions).
    probes > 0: Hutchinson, E_v[(J^T v)_i^2] with v ~ N(0, diag(w)) (one VJP per probe);
    probes = 0: exact, one VJP per output, vectorised in chunks.
    Returns (nu_soap, nu_adam): lists of float64 arrays in the shapes of cols[1] for SOAP and Adam leaves."""
    import jax
    import jax.numpy as jnp
    soap_idx = [i for i, sp in enumerate(specs) if sp.kind == 'soap']
    adam_idx = [i for i, sp in enumerate(specs) if sp.kind == 'adam']
    colsj = [[jnp.asarray(x) if hasattr(x, 'shape') else x for x in c] if c is not None else None for c in cols]
    se, sf = float(np.sqrt(w_e)), float(np.sqrt(w_f))

    def rot_sq(g):
        return rotated_squares(treedef.flatten_up_to(g), specs, colsj, soap_idx, adam_idx)

    def _hutch(state, b, key):
        mask = out_fn(state, b)[2]
        (_, F), vjp = jax.vjp(lambda p: out_fn(p, b)[:2], state)
        k1, k2 = jax.random.split(key)
        eps_e = jax.random.normal(k1, (), F.dtype)
        eps_f = jax.random.normal(k2, F.shape, F.dtype) * mask[:, None].astype(F.dtype)
        (g,) = vjp((eps_e * se, eps_f * sf))
        return rot_sq(g)

    def _exact(state, b):
        mask = out_fn(state, b)[2]
        (_, F), vjp = jax.vjp(lambda p: out_fn(p, b)[:2], state)
        n_pad = F.shape[0]
        K = 1 + 3 * n_pad
        Kp = -(-K // chunk) * chunk
        rows = jnp.eye(K, Kp, dtype=F.dtype).T                   # (Kp, K); zero rows pad the last chunk
        wrow = jnp.concatenate([jnp.ones(1, F.dtype), jnp.repeat(mask.astype(F.dtype), 3), jnp.zeros(Kp - K, F.dtype)])

        def one(row):
            (g,) = vjp((row[0] * se, row[1:].reshape(n_pad, 3) * sf))
            return rot_sq(g)

        def body(acc, rw):
            r, w = rw
            sq = jax.vmap(one)(r)                                  # (chunk, ...) trees
            add = jax.tree.map(lambda x: jnp.tensordot(w, x, axes=1), sq)
            return jax.tree.map(jnp.add, acc, add), None
        init = jax.tree.map(jnp.zeros_like, rot_sq(jax.tree.map(jnp.zeros_like, state)))
        acc, _ = jax.lax.scan(body, init, (rows.reshape(-1, chunk, K), wrow.reshape(-1, chunk)))
        return acc
    hutch, exact = jax.jit(_hutch), jax.jit(_exact)
    key = jax.random.PRNGKey(seed)
    acc_s = acc_a = None
    n = 0
    t0 = time.time()
    for j, b in enumerate(batches):
        if probes > 0:
            for _ in range(probes):
                key, sub = jax.random.split(key)
                s_, a_ = hutch(state, b, sub)
                acc_s = s_ if acc_s is None else [x + y for x, y in zip(acc_s, s_)]
                acc_a = a_ if acc_a is None else [x + y for x, y in zip(acc_a, a_)]
            n += probes
        else:
            s_, a_ = exact(state, b)
            acc_s = s_ if acc_s is None else [x + y for x, y in zip(acc_s, s_)]
            acc_a = a_ if acc_a is None else [x + y for x, y in zip(acc_a, a_)]
            n += 1
        if log and (j == 0 or (j + 1) % 20 == 0 or j + 1 == len(batches)):
            log(f"  ggn {j + 1}/{len(batches)} structures, {time.time() - t0:.0f} s")
    # mean over all draws (structures x probes, or structures): the per-structure mean GGN diagonal
    nu_s = [np.asarray(x, np.float64) / n for x in acc_s]
    nu_a = [np.asarray(x, np.float64) / n for x in acc_a]
    return nu_s, nu_a


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
    ap.add_argument('--ggn', type=int, default=0,
                    help='refinement 2: replace nu by the GGN diagonal in the SOAP basis from this many random '
                         'training structures (energy and force Jacobians, Gaussian likelihood)')
    ap.add_argument('--ggn-probes', type=int, default=2, help='Hutchinson probes per structure (0: exact)')
    ap.add_argument('--ggn-chunk', type=int, default=16, help='outputs per vectorised VJP chunk (exact mode)')
    ap.add_argument('--sigma-e', type=float, default=None, help='energy noise, eV (default: network valid RMSE)')
    ap.add_argument('--sigma-f', type=float, default=None, help='force noise, eV/A (default: network valid RMSE)')
    ap.add_argument('--predictive', choices=['mc', 'glm'], default='mc',
                    help='mc: re-evaluate the network at each weight sample; glm: linearised predictive, '
                         'f(theta*) + J (theta - theta*) by one JVP per sample (Immer et al. 2021)')
    ap.add_argument('--lam-evidence', action='store_true',
                    help='with --ggn: add the evidence-optimal lam (kappa = n_train) to the lam grid')
    ap.add_argument('--jvp-chunk', type=int, default=4, help='glm: samples per vectorised JVP chunk (GPU memory)')
    ap.add_argument('--lam-per-leaf', action='store_true',
                    help='with --ggn: one prior precision per leaf from the evidence fixed point '
                         'lam_g = gamma_g / |theta_g|^2 (MacKay); the --lam grid then multiplies these values')
    ap.add_argument('--kappa-times-ntrain', action='store_true',
                    help='kappa grid relative to the number of training structures (1 = untempered posterior)')
    a = ap.parse_args()
    t_start = time.time()
    import jax
    from flax import nnx
    from race_v2.training.utils import load_checkpoint
    from race_v2.eval.bayes_lastlayer import build, calib, load_xyz_graphs, collate
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
        return en, -jnp.where(jnp.isfinite(g), g, 0.0), mask
    ef = jax.jit(lambda state, b: _ef(state, b)[:2])

    def predict(param_leaves, graphs):
        state = jax.tree.unflatten(treedef, [jnp.asarray(x) for x in param_leaves])
        E, F = [], []
        for g in graphs:
            n = int(np.asarray(g.n_node)[0])
            en, f = ef(state, collate(g, a.n_bucket, a.e_bucket))
            E.append(float(en)); F.append(np.asarray(f)[:n].reshape(-1))
        return np.asarray(E), F

    ema_state_j = jax.tree.unflatten(treedef, [jnp.asarray(x) for x in ema_leaves])
    zero_leaves = [np.zeros_like(np.asarray(c)) for c in ema_leaves]

    def _ef_jvp(state, dstates, b):
        f = lambda p: _ef(p, b)[:2]
        # samples in chunks of --jvp-chunk (GPU memory: the vectorised JVP of a 128-node cell needs ~1 GiB per sample)
        return jax.lax.map(lambda d: jax.jvp(f, (state,), (d,))[1], dstates, batch_size=a.jvp_chunk)
    ef_jvp = jax.jit(_ef_jvp)

    def mc_lin(graphs, kappa, lam, seed=0, netE=None, netF=None):
        # linearised (GLM) predictive: network + J (theta_s - theta*), one JVP per sample, vectorised over samples
        rng = np.random.default_rng(seed)
        dz = [sample_params(zero_leaves, specs, cols, kappa, lam, rng) for _ in range(a.samples)]
        dstates = jax.tree.map(lambda *xs: jnp.stack(xs),
                               *[jax.tree.unflatten(treedef, [jnp.asarray(x) for x in z]) for z in dz])
        if netE is None:
            netE, netF = predict(ema_leaves, graphs); netF = np.concatenate(netF)
        dEs, dFs = [], []
        for g in graphs:
            n = int(np.asarray(g.n_node)[0])
            dE, dF = ef_jvp(ema_state_j, dstates, collate(g, a.n_bucket, a.e_bucket))
            dEs.append(np.asarray(dE)); dFs.append(np.asarray(dF)[:, :n].reshape(a.samples, -1))
        return netE[None] + np.stack(dEs, 1), netF[None] + np.concatenate(dFs, 1)

    def targets(gs):
        return (np.array([float(np.asarray(g.globals['energy']).reshape(-1)[0]) for g in gs]),
                np.concatenate([np.asarray(g.nodes['forces'], np.float64).reshape(-1) for g in gs]))

    valid = [va[i] for i in range(len(va))]
    if a.max_valid and len(valid) > a.max_valid:
        idx = np.sort(np.random.default_rng(a.seed).choice(len(valid), a.max_valid, replace=False))
        valid = [valid[i] for i in idx]
    netE, netF = predict(ema_leaves, valid); netF = np.concatenate(netF)
    yE, yF = targets(valid)
    net_rmse_E = float(np.sqrt(np.mean((netE - yE) ** 2))); net_rmse_F = float(np.sqrt(np.mean((netF - yF) ** 2)))
    print(f"network on valid ({len(valid)}): E rmse {1e3 * net_rmse_E:.2f} meV, F rmse {1e3 * net_rmse_F:.2f} meV/A; "
          f"{time.time() - t_start:.0f} s", flush=True)
    n_train = len(tr)

    if a.recollect:
        from race_v2.training.utils import compute_loss
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

    if a.ggn:
        t0 = time.time()
        train = [tr[i] for i in range(len(tr))]
        idx = np.sort(np.random.default_rng(a.seed + 7).choice(len(train), min(a.ggn, len(train)), replace=False))
        batches = [collate(train[r], a.n_bucket, a.e_bucket) for r in idx]
        ema_state = jax.tree.unflatten(treedef, [jnp.asarray(x) for x in ema_leaves])
        sig_e = a.sigma_e if a.sigma_e else net_rmse_E
        sig_f = a.sigma_f if a.sigma_f else net_rmse_F
        print(f"GGN curvature from {len(idx)} training structures, "
              f"{'exact' if a.ggn_probes == 0 else f'{a.ggn_probes} Hutchinson probes'}, sigma_E {1e3 * sig_e:.1f} meV, "
              f"sigma_F {1e3 * sig_f:.2f} meV/A", flush=True)
        nu_s, nu_a = ggn_diag(_ef, ema_state, batches, specs, cols, treedef, 1 / sig_e ** 2, 1 / sig_f ** 2,
                              a.ggn_probes, seed=a.seed, chunk=a.ggn_chunk, log=lambda m: print(m, flush=True))
        soap_idx = [i for i, sp in enumerate(specs) if sp.kind == 'soap']
        adam_idx = [i for i, sp in enumerate(specs) if sp.kind == 'adam']
        nu_new = list(cols[1])
        for k, i in enumerate(soap_idx):
            nu_new[i] = nu_s[k].reshape(np.shape(cols[1][i]))
        for k, i in enumerate(adam_idx):
            nu_new[i] = nu_a[k].reshape(np.shape(cols[1][i]))
        cols = [cols[0], nu_new] + list(cols[2:])
        nu_all = np.concatenate([nu_new[i].ravel() for i in soap_idx])
        q = np.quantile(nu_all, [0.1, 0.5, 0.9, 0.999, 1.0])
        print(f"GGN nu ({time.time() - t0:.0f} s): SOAP leaves 10 % {q[0]:.2e}, median {q[1]:.2e}, 90 % {q[2]:.2e}, "
              f"99.9 % {q[3]:.2e}, max {q[4]:.2e}; Adam leaves max "
              f"{max(float(np.max(nu_new[i])) for i in adam_idx):.2e}; n_train {n_train}: n_train * median = "
              f"{n_train * q[1]:.2e}", flush=True)
        np.savez(out / 'ggn_nu.npz', **{f'leaf{i}': nu_new[i] for i in soap_idx + adam_idx})
        # Laplace marginal likelihood at kappa = n_train (untempered), as a function of the prior precision lam:
        #   log Z(lam) = log p(D | theta*) - 1/2 sum_i log(1 + n_train nu_i / lam) - lam/2 |theta*|^2   (+ const)
        # (P/2 log lam from the prior normalisation cancels against the log det), maximised over lam in closed 1-D form
        from scipy.optimize import minimize_scalar
        nu_tot = n_train * np.concatenate([nu_new[i].ravel() for i in soap_idx + adam_idx])
        theta_sq = float(sum(np.sum(np.asarray(ema_leaves[i], np.float64) ** 2) for i in soap_idx + adam_idx))
        P_eff = nu_tot.size

        def neg_log_ev(loglam):
            lam_ = np.exp(loglam)
            return 0.5 * np.sum(np.log1p(nu_tot / lam_)) + 0.5 * lam_ * theta_sq
        r_ev = minimize_scalar(neg_log_ev, bounds=(np.log(1e-4), np.log(1e10)), method='bounded')
        lam_ev = float(np.exp(r_ev.x))
        ev_grid = {f'{l:g}': float(-neg_log_ev(np.log(l))) for l in a.lam}
        print(f"evidence (kappa = n_train): |theta*|^2 {theta_sq:.1f} over {P_eff} weights, optimum lam {lam_ev:.3g} "
              f"(log Z rel. {-r_ev.fun:.1f}); at the grid lam: " + ', '.join(f'{k}: {v:.1f}' for k, v in ev_grid.items()),
              flush=True)
        if a.lam_evidence:
            a.lam = list(a.lam) + [lam_ev]
        evidence = {'lam_opt': lam_ev, 'log_ev_rel_opt': float(-r_ev.fun), 'log_ev_rel_grid': ev_grid,
                    'theta_sq': theta_sq, 'n_weights': int(P_eff)}
        lam_leaf = None
        if a.lam_per_leaf:
            lam_leaf = np.ones(len(specs))
            gam_tot = 0.0
            for i in soap_idx + adam_idx:
                nu_i = n_train * nu_new[i].ravel()
                th2 = float(np.sum(np.asarray(ema_leaves[i], np.float64) ** 2))
                lg = 1.0
                for _ in range(500):
                    gam = float(np.sum(nu_i / (nu_i + lg)))
                    lg_new = max(gam, 1e-3) / max(th2, 1e-12)
                    if abs(np.log(lg_new / lg)) < 1e-8:
                        lg = lg_new; break
                    lg = lg_new
                lam_leaf[i] = lg; gam_tot += gam
            evidence['lam_per_leaf'] = {str(i): float(lam_leaf[i]) for i in soap_idx + adam_idx}
            evidence['gamma_total'] = gam_tot
            q = np.quantile([lam_leaf[i] for i in soap_idx + adam_idx], [0, 0.5, 1])
            print(f"per-leaf evidence prior: lam_g min {q[0]:.3g} median {q[1]:.3g} max {q[2]:.3g}; well-determined "
                  f"directions gamma {gam_tot:.0f} of {P_eff}; the --lam grid multiplies lam_g", flush=True)

    def mc(graphs, kappa, lam, seed=0):
        rng = np.random.default_rng(seed)
        Es, Fs = [], []
        for _ in range(a.samples):
            E, F = predict(sample_params(ema_leaves, specs, cols, kappa, lam, rng), graphs)
            Es.append(E); Fs.append(np.concatenate(F))
        return np.stack(Es), np.stack(Fs)

    def evaluate(gs, kappa, lam, seed, netE, netF, dump=None):
        """Calibration with two centres: the sample mean ('mean') and the network itself ('map', sigma = sample std).
        dump: path of an npz with the per-structure arrays (OOD analyses, research/diag/ood_summary.py)."""
        Es, Fs = mc(gs, kappa, lam, seed) if a.predictive == 'mc' else mc_lin(gs, kappa, lam, seed, netE, netF)
        if not (np.isfinite(Es).all() and np.isfinite(Fs).all()):
            return None
        yE, yF = targets(gs)
        sE, sF = 1e3 * Es.std(0, ddof=1) + 1e-6, 1e3 * Fs.std(0, ddof=1) + 1e-6
        if dump is not None:
            ns = [3 * int(np.asarray(g.n_node)[0]) for g in gs]; cut = np.cumsum([0] + ns)
            err = netF - yF
            np.savez(dump, yE=yE, netE=netE, meanE=Es.mean(0), sigE=sE / 1e3,
                     F_err_rms=np.array([np.sqrt(np.mean(err[p:q] ** 2)) for p, q in zip(cut[:-1], cut[1:])]),
                     F_sig_mean=np.array([np.mean(sF[p:q]) for p, q in zip(cut[:-1], cut[1:])]) / 1e3,
                     F_err=err, F_sig=sF / 1e3)
        from scipy.stats import spearmanr
        # confound check: does sigma just track the force magnitude, and does |err| itself track |F|?
        absF = np.abs(yF); err = np.abs(netF - yF)
        return {'E_mean': calib(1e3 * (Es.mean(0) - yE), sE), 'F_mean': calib(1e3 * (Fs.mean(0) - yF), sF),
                'E_map': calib(1e3 * (netE - yE), sE), 'F_map': calib(1e3 * (netF - yF), sF),
                'F_spearman_sigma_absF': float(spearmanr(sF, absF).correlation),
                'F_spearman_err_absF': float(spearmanr(err, absF).correlation),
                'net_E_rmse_meV': 1e3 * float(np.sqrt(np.mean((netE - yE) ** 2))),
                'net_F_rmse_meVA': 1e3 * float(np.sqrt(np.mean((netF - yF) ** 2)))}

    # check: kappa -> infinity gives the network back (samples collapse onto the EMA weights)
    E_inf, F_inf = predict(sample_params(ema_leaves, specs, cols, 1e300, 1e300, np.random.default_rng(0)), valid)
    dev = max(np.abs(E_inf - netE).max(), np.abs(np.concatenate(F_inf) - netF).max())
    print(f"check: kappa, lam -> inf reproduces the network to {dev:.1e} (E eV / F eV/A)", flush=True)
    kappas = [(k, k * n_train if a.kappa_times_ntrain else k) for k in a.kappa]
    if a.kappa_times_ntrain:
        print(f"kappa grid x n_train = {n_train}: " + ' '.join(f'{k:g}->{ke:.3g}' for k, ke in kappas), flush=True)
    best, grid = None, {}
    lam_leaf = lam_leaf if a.ggn else None
    lams = [(l, l * lam_leaf if lam_leaf is not None else l) for l in a.lam]
    for kappa_label, kappa in kappas:
        for lam_label, lam in lams:
            r = evaluate(valid, kappa, lam, a.seed, netE, netF)
            if r is None:
                print(f"kappa {kappa_label:g} lam {lam_label:g}: non-finite predictions", flush=True); continue
            grid[f'{kappa_label:g},{lam_label:g}'] = r
            cE, cF, mF = r['E_map'], r['F_map'], r['F_mean']
            print(f"kappa {kappa_label:g} lam {lam_label:g}: valid F(map) nll {cF['nll']:.3f} z {cF['rms_z']:.2f} "
                  f"sigma {cF['mean_sigma']:.2f} rho {cF['spearman']:.2f} | F(mean) rmse {mF['rmse']:.2f} "
                  f"| E(map) nll {cE['nll']:.3f} z {cE['rms_z']:.2f} rho {cE['spearman']:.2f}; "
                  f"{time.time() - t_start:.0f} s", flush=True)
            score = cF['nll']                       # forces: no per-composition energy offsets in the ranking
            if best is None or score < best[0]:
                best = (score, kappa_label, kappa, lam_label, lam)
    if best is None:
        raise RuntimeError('no finite (kappa, lam)')
    _, kappa_label, kappa, lam_label, lam = best
    print(f"chosen kappa {kappa_label:g} lam {lam_label:g} (validation force NLL, network as the mean)", flush=True)
    report = {'kappa': kappa_label, 'kappa_effective': kappa, 'lam': lam_label, 'lam_per_leaf': lam_leaf is not None, 'n_train': n_train, 'predictive': a.predictive,
              'evidence': evidence if a.ggn else None,
              'curvature': 'ggn' if a.ggn else ('recollect' if a.recollect else 'optimizer'), 'samples': a.samples, 'check_inf': float(dev), 'grid': grid, 'sets': {}}
    for name, ds in te.items():
        gs = [ds[i] for i in range(len(ds))][:a.max_test]
        nE, nF = predict(ema_leaves, gs)
        r = evaluate(gs, kappa, lam, a.seed + 1, nE, np.concatenate(nF), dump=out / f'per_structure_{name}.npz')
        report['sets'][name] = r
        print(f"{name} n={len(gs)}: net F {r['net_F_rmse_meVA']:.2f} | F(map) sigma {r['F_map']['mean_sigma']:.2f} "
              f"z {r['F_map']['rms_z']:.2f} cov68 {r['F_map']['cov68']:.2f} cov95 {r['F_map']['cov95']:.2f} "
              f"rho {r['F_map']['spearman']:.2f} | F(mean) rmse {r['F_mean']['rmse']:.2f} z {r['F_mean']['rms_z']:.2f}"
              f" | E(map) rmse {r['E_map']['rmse']:.2f} sigma {r['E_map']['mean_sigma']:.2f} z {r['E_map']['rms_z']:.2f} "
              f"rho {r['E_map']['spearman']:.2f} | confound: rho(sigma_F, |F|) {r['F_spearman_sigma_absF']:.2f}, "
              f"rho(|err_F|, |F|) {r['F_spearman_err_absF']:.2f}", flush=True)
    report['time_s'] = time.time() - t_start
    json.dump(report, open(out / 'soap_laplace.json', 'w'), indent=1)
    print(f"written {out / 'soap_laplace.json'} ({report['time_s']:.0f} s)")


if __name__ == '__main__':
    main()
