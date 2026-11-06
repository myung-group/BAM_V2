"""Last-layer Bayesian linear regression for RACEv2 (exact Bayesian inference over the final readout weights).

In RACEv2 every layer t has a stratum readout  e_{t,i} = w_t . g_{t,i} + b_t  (g = gated scalar features of node i),
and the energy is  E = scale * sum_t sum_i e_{t,i} + (terms that do not depend on w, b).  E and the forces
F = -dE/dx are therefore exactly linear in theta = (w_1..w_L, sum_t b_t):

    E = rest_E + phi . theta,   phi = scale * [sum_i g_{1,i}, ..., sum_i g_{L,i}, N]
    F = rest_F + J theta,       J   = -scale * d[sum_i g_{t,i}] / dx   (zero column for the bias)

With a Gaussian prior theta ~ N(0, diag(alpha)^-1) (one precision per layer and one for the bias) and Gaussian
likelihoods for energies (precision beta_E) and force components (beta_F), the posterior is Gaussian and closed form.
The hyperparameters maximise the log evidence (type-II maximum likelihood) on the training set.  The features g come
from the trained network (held fixed): the posterior is exact for the last layer *given* the features; uncertainty of
the feature extractor itself is not represented.

Outputs per evaluation set: accuracy of the posterior mean and calibration (z-score RMS, Gaussian NLL, 68/95 %
coverage, Spearman correlation between |error| and the predicted sigma), compared with the deep ensemble of the
given seed checkpoints (mean, sample std).

    python -m bam_omat24.bayes_lastlayer <config.json> --ckpt <ckpt.pkl> [--ensemble <ckpt> <ckpt> ...]
        [--dihedral <test_dih.xyz>] --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------------------------- model / data
def build(cfg_path, ckpt, backend=None):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from bam_omat24 import train_3bpa as T
    from bam_omat24.training_utils import load_checkpoint
    cfg = json.load(open(cfg_path))
    cfg['x64'] = True
    if backend:
        cfg['tp_backend'] = backend
    jax.config.update('jax_enable_x64', True)
    zs, e0, tr, va, te, stats = T.build_datasets(cfg, sys.stderr)
    graphdef, _p, *_ = T.build_model(cfg, len(zs), e0, stats, sys.stderr)

    def load(path):
        params = jax.tree.map(lambda x: jnp.asarray(x, jnp.float64) if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating)
                              else jnp.asarray(x), load_checkpoint(path)['ema_params'])
        return nnx.merge(graphdef, params)
    return cfg, load, load(ckpt), zs, e0, tr, va, te


def zero_readout(model):
    """Copy of the model with every stratum readout's final Linear (weights and biases) set to zero."""
    import jax
    import jax.numpy as jnp
    from flax import nnx
    graphdef, state = nnx.split(model)
    leaves, treedef = jax.tree_util.tree_flatten_with_path(state)
    out, n = [], 0
    for path, x in leaves:
        p = jax.tree_util.keystr(path)
        hit = "['layers']" in p and "['readout']" in p and "['readout1']" not in p and "['readout_sym" not in p
        if hit:
            n += 1
        out.append(jnp.zeros_like(x) if hit else x)
    if n == 0:
        raise RuntimeError("no stratum readout parameters found")
    return nnx.merge(graphdef, jax.tree_util.tree_unflatten(treedef, out)), n


def load_xyz_graphs(path, zs, e0, cutoff):
    """Graphs of an extxyz file; forces only when present (dihedral scans are energy-only)."""
    import ase.io
    from bam_omat24.data_3bpa import preprocess_graph, atomic_numbers_to_indices
    idx = atomic_numbers_to_indices(zs)
    out, has_f = [], True
    for a in ase.io.read(path, index=':'):
        g = preprocess_graph(a, idx, e0, cutoff, targets=False)
        sp = np.asarray(g.nodes['species'])
        energy = a.get_potential_energy() - float(np.sum(np.asarray(e0)[sp]))
        try:
            f = a.get_forces()
        except Exception:
            f = None
        has_f = has_f and f is not None
        nodes = dict(g.nodes)
        nodes['forces'] = (np.asarray(f, np.float32) if f is not None else np.zeros((len(a), 3), np.float32))
        out.append(g._replace(nodes=nodes, globals={'energy': np.array([energy], np.float32)}))
    return out, has_f


# ----------------------------------------------------------------------------------------------- per-structure arrays
def make_fns(model, model0, n_bucket, e_bucket):
    """jit-compiled per-structure functions on a fixed padded shape."""
    import jax
    import jax.numpy as jnp
    import jraph
    from bam_omat24 import race_nnx as R

    def energy(m, pos, b, mask):
        ne = m.node_energies(pos, None, None, b)['node_energies'].reshape(-1)
        return jnp.sum(jnp.where(mask, ne, 0.0))

    def feats(pos, b, mask):
        R.READOUT_TAP = []
        try:
            model.node_energies(pos, None, None, b)
            taps = list(R.READOUT_TAP)
        finally:
            R.READOUT_TAP = None
        g = jnp.concatenate([jnp.sum(jnp.where(mask[:, None], t, 0.0), axis=0) for t in taps])
        return model.scale[...] * g

    def full(b):
        mask = jraph.get_node_padding_mask(b)
        pos = b.nodes['positions']
        e, g = jax.value_and_grad(energy, argnums=1)(model, pos, b, mask)
        e0_, g0 = jax.value_and_grad(energy, argnums=1)(model0, pos, b, mask)
        phi = feats(pos, b, mask)
        jac = jax.jacfwd(feats)(pos, b, mask)                    # (P, n_pad, 3) = d phi / dx
        return e, -g, e0_, -g0, phi, -jac

    def energy_only(b):
        mask = jraph.get_node_padding_mask(b)
        pos = b.nodes['positions']
        return energy(model, pos, b, mask), energy(model0, pos, b, mask), feats(pos, b, mask)

    def ens(m, b):
        mask = jraph.get_node_padding_mask(b)
        e, g = jax.value_and_grad(energy, argnums=1)(m, b.nodes['positions'], b, mask)
        return e, -g
    return jax.jit(full), jax.jit(energy_only), jax.jit(ens)


def collate(g, n_bucket, e_bucket):
    import jax
    import jax.numpy as jnp
    from bam_omat24.data_nnx import default_collate_fn
    from bam_omat24 import train_3bpa as T
    g = T._cast_graphs([g], np.float64)[0]
    b = default_collate_fn([g], batch_size=1, node_bucket_size=n_bucket, edge_bucket_size=e_bucket)
    return jax.tree.map(jnp.asarray, b)


def arrays_for(graphs, fns, n_bucket, e_bucket, with_forces, ens_models=()):
    """Per-structure design rows and targets.  Energies relative to the isolated-atom E0 (training target space)."""
    full, energy_only, ens = fns
    rows = {'yE': [], 'eFull': [], 'eRest': [], 'phi': [], 'N': [], 'yF': [], 'fFull': [], 'fRest': [], 'J': [],
            'ensE': [], 'ensF': []}
    t0 = time.time()
    for k, g in enumerate(graphs):
        n = int(np.asarray(g.n_node)[0])
        b = collate(g, n_bucket, e_bucket)
        rows['yE'].append(float(np.asarray(g.globals['energy']).reshape(-1)[0])); rows['N'].append(n)
        if with_forces:
            e, f, e0_, f0, phi, jac = full(b)
            rows['yF'].append(np.asarray(g.nodes['forces'], np.float64).reshape(-1))
            rows['fFull'].append(np.asarray(f)[:n].reshape(-1)); rows['fRest'].append(np.asarray(f0)[:n].reshape(-1))
            rows['J'].append(np.transpose(np.asarray(jac)[:, :n, :], (1, 2, 0)).reshape(3 * n, -1))
        else:
            e, e0_, phi = energy_only(b)
        rows['eFull'].append(float(e)); rows['eRest'].append(float(e0_)); rows['phi'].append(np.asarray(phi))
        if ens_models:
            es, fs = zip(*[ens(m, b) for m in ens_models])
            rows['ensE'].append([float(x) for x in es])
            if with_forces:
                rows['ensF'].append(np.stack([np.asarray(x)[:n].reshape(-1) for x in fs], 0))
        if k == 0:
            t_first = time.time() - t0
    print(f"    {len(graphs)} structures in {time.time() - t0:.0f} s (first incl. compile {t_first:.0f} s)", flush=True)
    out = {k: v for k, v in rows.items() if v}
    out['phi'] = np.stack(out['phi'])
    out['phi'] = np.concatenate([out['phi'], np.zeros((len(out['phi']), 1))], axis=1)   # bias column, set by the caller
    if 'J' in out:                                                                      # the bias has no force column
        out['J'] = [np.concatenate([j, np.zeros((j.shape[0], 1))], axis=1) for j in out['J']]
    return out


# ----------------------------------------------------------------------------------------------- Bayesian regression
class BLR:
    """theta ~ N(0, diag(alpha)^-1), alpha per group; y_E ~ N(phi theta, 1/beta_E); y_F ~ N(J theta, 1/beta_F)."""

    def __init__(self, groups, AE, tE, AF, tF):
        self.groups = np.asarray(groups)
        self.nE, self.nF = len(tE), len(tF)
        self.SE, self.SF = AE.T @ AE, AF.T @ AF
        self.uE, self.uF = AE.T @ tE, AF.T @ tF
        self.qE, self.qF = float(tE @ tE), float(tF @ tF)
        self.P = AE.shape[1]

    def _post(self, log_alpha, lbE, lbF):
        alpha = np.exp(log_alpha)[self.groups]
        bE, bF = np.exp(lbE), np.exp(lbF)
        Lam = np.diag(alpha) + bE * self.SE + bF * self.SF
        C = np.linalg.cholesky(Lam)
        m = np.linalg.solve(C.T, np.linalg.solve(C, bE * self.uE + bF * self.uF))
        return alpha, bE, bF, C, m

    def log_evidence(self, x):
        ng = self.groups.max() + 1
        try:
            alpha, bE, bF, C, m = self._post(x[:ng], x[ng], x[ng + 1])
        except np.linalg.LinAlgError:
            return -1e300
        ssE = self.qE - 2 * m @ self.uE + m @ self.SE @ m
        ssF = self.qF - 2 * m @ self.uF + m @ self.SF @ m
        return float(0.5 * np.sum(np.log(alpha)) + 0.5 * self.nE * np.log(bE) + 0.5 * self.nF * np.log(bF)
                     - np.sum(np.log(np.diag(C))) - 0.5 * (bE * ssE + bF * ssF + m @ (alpha * m))
                     - 0.5 * (self.nE + self.nF) * np.log(2 * np.pi))

    def fit(self, x0):
        from scipy.optimize import minimize
        r = minimize(lambda x: -self.log_evidence(x), x0, method='L-BFGS-B', options={'maxiter': 2000})
        ng = self.groups.max() + 1
        self.x = r.x
        self.alpha, self.bE, self.bF, C, self.m = self._post(r.x[:ng], r.x[ng], r.x[ng + 1])
        Cinv = np.linalg.solve(C, np.eye(self.P))
        self.Sigma = Cinv.T @ Cinv
        self.logev = -r.fun
        return r


def calib(err, sig):
    from scipy.stats import spearmanr
    err, sig = np.asarray(err, np.float64), np.asarray(sig, np.float64)
    z = err / sig
    return {'rmse': float(np.sqrt(np.mean(err ** 2))), 'mean_sigma': float(np.mean(sig)),
            'rms_z': float(np.sqrt(np.mean(z ** 2))),
            'nll': float(np.mean(0.5 * z ** 2 + np.log(sig) + 0.5 * np.log(2 * np.pi))),
            'cov68': float(np.mean(np.abs(z) < 1.0)), 'cov95': float(np.mean(np.abs(z) < 1.96)),
            'spearman': float(spearmanr(np.abs(err), sig).correlation)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('config'); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--ensemble', nargs='*', default=[])
    ap.add_argument('--dihedral', default=None); ap.add_argument('--out', required=True)
    ap.add_argument('--backend', default=None); ap.add_argument('--n-bucket', type=int, default=32)
    ap.add_argument('--e-bucket', type=int, default=1024)
    ap.add_argument('--max-test', type=int, default=None, help='cap per test set (quick runs)')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg, load, model, zs, e0, tr, va, te = build(a.config, a.ckpt, a.backend)
    model0, n_ro = zero_readout(model)
    L = len(model.layers)
    ens_models = [load(p) for p in a.ensemble]
    print(f"model: {L} layers, {n_ro} readout parameter arrays zeroed in the reference copy; ensemble {len(ens_models)}",
          flush=True)
    fns = make_fns(model, model0, a.n_bucket, a.e_bucket)
    cutoff = float(cfg.get('cutoff', 5.0))

    def graphs_of(ds):
        return [ds[i] for i in range(len(ds))]
    sets = {'train': (graphs_of(tr), True), 'valid': (graphs_of(va), True)}
    for k, ds in te.items():
        gs = graphs_of(ds)
        sets[k] = (gs[:a.max_test] if a.max_test else gs, True)
    if a.dihedral:
        gs, hf = load_xyz_graphs(a.dihedral, zs, e0, cutoff)
        sets['dihedral'] = (gs[:a.max_test] if a.max_test else gs, hf)
    data = {}
    for name, (gs, wf) in sets.items():
        print(f"  {name}: computing design rows", flush=True)
        data[name] = arrays_for(gs, fns, a.n_bucket, a.e_bucket, wf, ens_models if name != 'train' else ())
        data[name]['phi'][:, -1] = np.asarray(data[name]['N'], np.float64) * float(model.scale[...])   # bias column

    P = data['train']['phi'].shape[1]
    d = (P - 1) // L
    groups = np.array([t for t in range(L) for _ in range(d)] + [L])
    D = data['train']
    AE, tE = D['phi'], np.asarray(D['yE']) - np.asarray(D['eRest'])
    AF, tF = np.concatenate(D['J']), np.concatenate(D['yF']) - np.concatenate(D['fRest'])

    # exactness check: the trained network's linear part reproduced by the features
    th_E = np.linalg.lstsq(AE, np.asarray(D['eFull']) - np.asarray(D['eRest']), rcond=None)[0]
    resE = np.abs(AE @ th_E - (np.asarray(D['eFull']) - np.asarray(D['eRest']))).max()
    resF = np.abs(AF @ th_E - (np.concatenate(D['fFull']) - np.concatenate(D['fRest']))).max()
    print(f"linearity check on train: max |E - rest - phi.theta_net| {resE:.2e} eV, forces with the same theta "
          f"{resF:.2e} eV/A ({P} features)", flush=True)

    blr = BLR(groups, AE, tE, AF, tF)
    x0 = np.concatenate([np.zeros(L + 1), [np.log(1 / np.var(tE - AE @ th_E + 1e-12) if len(tE) else 0.0),
                                            np.log(1 / max(np.var(tF), 1e-12))]])
    r = blr.fit(x0)
    print(f"evidence maximised ({r.message}): log evidence {blr.logev:.1f}; noise sigma_E {1e3 / np.sqrt(blr.bE):.2f} meV, "
          f"sigma_F {1e3 / np.sqrt(blr.bF):.2f} meV/A; prior sigma per layer {np.round(1 / np.sqrt(np.exp(r.x[:L + 1])), 4)}",
          flush=True)

    report = {'P': P, 'linearity': {'E': float(resE), 'F': float(resF)}, 'log_evidence': blr.logev,
              'sigma_E_meV': 1e3 / np.sqrt(blr.bE), 'sigma_F_meVA': 1e3 / np.sqrt(blr.bF), 'sets': {}}
    for name, D in data.items():
        if name == 'train':
            continue
        yE = np.asarray(D['yE']); phi = D['phi']
        mE = np.asarray(D['eRest']) + phi @ blr.m
        vE_epi = np.einsum('ij,jk,ik->i', phi, blr.Sigma, phi)
        rep = {'n': len(yE),
               'net_E_rmse_meV': 1e3 * float(np.sqrt(np.mean((np.asarray(D['eFull']) - yE) ** 2))),
               'E_blr_total': calib(1e3 * (mE - yE), 1e3 * np.sqrt(vE_epi + 1 / blr.bE)),
               'E_blr_epistemic_spearman': calib(1e3 * (mE - yE), 1e3 * np.sqrt(vE_epi))['spearman']}
        if 'ensE' in D:
            ensE = np.asarray(D['ensE'])
            rep['E_ensemble'] = calib(1e3 * (ensE.mean(1) - yE), 1e3 * ensE.std(1, ddof=1))
        if 'J' in D:
            yF = np.concatenate(D['yF'])
            mF = np.concatenate(D['fRest']) + np.concatenate([j @ blr.m for j in D['J']])
            vF_epi = np.concatenate([np.einsum('ij,jk,ik->i', j, blr.Sigma, j) for j in D['J']])
            rep['net_F_rmse_meVA'] = 1e3 * float(np.sqrt(np.mean((np.concatenate(D['fFull']) - yF) ** 2)))
            rep['F_blr_total'] = calib(1e3 * (mF - yF), 1e3 * np.sqrt(vF_epi + 1 / blr.bF))
            rep['F_blr_epistemic_spearman'] = calib(1e3 * (mF - yF), 1e3 * np.sqrt(vF_epi))['spearman']
            if 'ensF' in D:
                ensF = np.concatenate(D['ensF'], axis=1)
                rep['F_ensemble'] = calib(1e3 * (ensF.mean(0) - yF), 1e3 * ensF.std(0, ddof=1))
        report['sets'][name] = rep
        line = (f"{name:>10} n={rep['n']:5d} | net E {rep['net_E_rmse_meV']:.2f} | BLR E rmse {rep['E_blr_total']['rmse']:.2f} "
                f"z {rep['E_blr_total']['rms_z']:.2f} cov95 {rep['E_blr_total']['cov95']:.2f} rho {rep['E_blr_total']['spearman']:.2f}")
        if 'E_ensemble' in rep:
            line += (f" | ens E rmse {rep['E_ensemble']['rmse']:.2f} z {rep['E_ensemble']['rms_z']:.2f} "
                     f"rho {rep['E_ensemble']['spearman']:.2f}")
        if 'F_blr_total' in rep:
            line += (f" || net F {rep['net_F_rmse_meVA']:.2f} | BLR F rmse {rep['F_blr_total']['rmse']:.2f} "
                     f"z {rep['F_blr_total']['rms_z']:.2f} cov95 {rep['F_blr_total']['cov95']:.2f} "
                     f"rho {rep['F_blr_total']['spearman']:.2f}")
            if 'F_ensemble' in rep:
                line += (f" | ens F rmse {rep['F_ensemble']['rmse']:.2f} z {rep['F_ensemble']['rms_z']:.2f} "
                         f"rho {rep['F_ensemble']['spearman']:.2f}")
        print(line, flush=True)
    json.dump(report, open(out / 'bayes_lastlayer.json', 'w'), indent=1)
    np.savez(out / 'blr_posterior.npz', m=blr.m, Sigma=blr.Sigma, groups=groups, x=blr.x)
    print('written', out / 'bayes_lastlayer.json')


if __name__ == '__main__':
    main()
