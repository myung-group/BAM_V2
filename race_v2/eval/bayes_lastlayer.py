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

Periodic structures (2026-11-09): the cell enters the features; the stress is linear in theta too,
S = rest_S + (1/V) [(d phi/d cell)^T cell + sum_i d phi/d x_i (x) x_i] theta, and forms a third likelihood block with
its own precision beta_S when the training data carry stress labels (--stress auto|on|off).  Training-set force /
stress rows are accumulated into A^T A on the fly (--max-train for a random subset), so large crystal sets fit.

Outputs per evaluation set: accuracy of the posterior mean and calibration (z-score RMS, Gaussian NLL, 68/95 %
coverage, Spearman correlation between |error| and the predicted sigma), compared with the deep ensemble of the
given seed checkpoints (mean, sample std).

    python -m race_v2.eval.bayes_lastlayer <config.json> --ckpt <ckpt.pkl> [--ensemble <ckpt> <ckpt> ...]
        [--dihedral <test_dih.xyz>] [--max-train N] [--max-test N] [--stress auto|on|off] --out <dir>
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
    from race_v2.training import train as T
    from race_v2.training.utils import load_checkpoint
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
    from race_v2.data.xyz_graphs import preprocess_graph, atomic_numbers_to_indices
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
VOIGT = [0, 4, 8, 5, 2, 1]          # xx yy zz yz xz xy, as utils.energy_forces_stress


def make_fns(model, model0, periodic, jac_chunk=8):
    """jit-compiled per-structure functions (one structure per padded batch, graph 0).

    Periodic graphs (2026-11-09): node energies with the cell and cutoff of the batch; stress as in
    utils.energy_forces_stress, virial = (dE/dcell)^T cell + sum_i dE/dx_i (x) x_i, over the volume.  The same linear map
    applied to d phi / dx and d phi / dcell gives the stress rows of the design matrix (stress is linear in theta).
    The Jacobian d phi / d(x, cell) is built from P vector-Jacobian products, jac_chunk at a time (a full jacfwd over
    3 n + 9 directions of a 100-atom cell does not fit a 16 GB GPU)."""
    import jax
    import jax.numpy as jnp
    import jraph
    from race_v2.models import race as R

    def _ne(m, pos, cell, b):
        if periodic:
            return m.node_energies(pos, cell, b.globals['cutoff'], b)
        return m.node_energies(pos, None, None, b)

    def energy(m, pos, cell, b, mask):
        ne = _ne(m, pos, cell, b)['node_energies'].reshape(-1)
        return jnp.sum(jnp.where(mask, ne, 0.0))

    def feats(pos, cell, b, mask):
        R.READOUT_TAP = []
        try:
            _ne(model, pos, cell, b)
            taps = list(R.READOUT_TAP)
        finally:
            R.READOUT_TAP = None
        g = jnp.concatenate([jnp.sum(jnp.where(mask[:, None], t, 0.0), axis=0) for t in taps])
        return model.scale[...] * g

    def _clean(x):
        return jnp.where(jnp.isfinite(x), x, 0.0)

    def virial(gpos, gcell, pos, cell, vol, mask):
        """(..., n, 3) position gradients and (..., 3, 3) cell gradients of graph 0 -> (..., 6) stress, eV/A^3.
        vol: the graph's 'volume' global, as the model's own stress uses it."""
        gpos = jnp.where(mask[:, None], _clean(gpos), 0.0)
        v = (jnp.einsum('...ba,bc->...ac', _clean(gcell), cell[0])
             + jnp.einsum('...iu,iv->...uv', gpos, jnp.where(mask[:, None], pos, 0.0)))
        vol = jnp.where(vol > 0, vol, 1.0)
        return v.reshape(v.shape[:-2] + (9,))[..., jnp.array(VOIGT)] / vol

    def efs(m, b):
        mask = jraph.get_node_padding_mask(b)
        pos = b.nodes['positions']
        if periodic:
            cell = b.globals['cell']
            e, (gp, gc) = jax.value_and_grad(energy, argnums=(1, 2))(m, pos, cell, b, mask)
            return e, -_clean(gp), virial(gp, gc[0], pos, cell, b.globals['volume'][0], mask)
        e, gp = jax.value_and_grad(energy, argnums=1)(m, pos, None, b, mask)
        return e, -_clean(gp), jnp.zeros(6)

    def full(b):
        mask = jraph.get_node_padding_mask(b)
        pos = b.nodes['positions']
        e, f, s = efs(model, b)
        e0_, f0, s0 = efs(model0, b)
        if periodic:
            cell = b.globals['cell']
            phi, vjp = jax.vjp(lambda p_, c_: feats(p_, c_, b, mask), pos, cell)
            eye = jnp.eye(phi.shape[0], dtype=phi.dtype)
            jp, jc = jax.lax.map(vjp, eye, batch_size=jac_chunk)            # (P, n_pad, 3), (P, n_g, 3, 3)
            srows = virial(jp, jc[:, 0], pos, cell, b.globals['volume'][0], mask)                   # (P, 6)
        else:
            phi, vjp = jax.vjp(lambda p_: feats(p_, None, b, mask), pos)
            eye = jnp.eye(phi.shape[0], dtype=phi.dtype)
            jp = jax.lax.map(lambda v: vjp(v)[0], eye, batch_size=jac_chunk)
            srows = jnp.zeros((phi.shape[0], 6))
        return dict(e=e, f=f, s=s, e0=e0_, f0=f0, s0=s0, phi=phi, J=-_clean(jp), S=srows)

    def energy_only(b):
        mask = jraph.get_node_padding_mask(b)
        pos = b.nodes['positions']
        cell = b.globals['cell'] if periodic else None
        return energy(model, pos, cell, b, mask), energy(model0, pos, cell, b, mask), feats(pos, cell, b, mask)

    return jax.jit(full), jax.jit(energy_only), jax.jit(efs)


def collate(g, n_bucket, e_bucket):
    import jax
    import jax.numpy as jnp
    from race_v2.data.loaders import default_collate_fn
    from race_v2.training import train as T
    g = T._cast_graphs([g], np.float64)[0]
    b = default_collate_fn([g], batch_size=1, node_bucket_size=n_bucket, edge_bucket_size=e_bucket)
    return jax.tree.map(jnp.asarray, b)


class Gram:
    """Running sufficient statistics of one linear block y = A theta + noise (A^T A, A^T t, t.t, n)."""

    def __init__(self, P):
        self.S = np.zeros((P, P)); self.u = np.zeros(P); self.q = 0.0; self.n = 0

    def add(self, A, t):
        self.S += A.T @ A; self.u += A.T @ t; self.q += float(t @ t); self.n += len(t)


def arrays_for(graphs, fns, n_bucket, e_bucket, with_forces, ens_models=(), use_stress=False, accumulate=False,
               stress_labels=True):
    """Per-structure design rows and targets.  Energies relative to the isolated-atom E0 (training target space).
    accumulate=True (training set): force / stress rows go into Gram blocks instead of being stored (memory: crystals
    and large training sets).  use_stress: stress rows (periodic); stress_labels=False: no stress targets (NaN), the
    rows still give the linearity check and the epistemic stress sigma."""
    full, energy_only, efs = fns
    rows = {'yE': [], 'eFull': [], 'eRest': [], 'phi': [], 'N': [], 'yF': [], 'fFull': [], 'fRest': [], 'J': [],
            'yS': [], 'sFull': [], 'sRest': [], 'Srows': [], 'ensE': [], 'ensF': [], 'ensS': []}
    gram = {}
    t0 = time.time(); t_first = 0.0
    for k, g in enumerate(graphs):
        n = int(np.asarray(g.n_node)[0])
        b = collate(g, n_bucket, e_bucket)
        rows['yE'].append(float(np.asarray(g.globals['energy']).reshape(-1)[0])); rows['N'].append(n)
        if with_forces:
            o = {kk: np.asarray(v) for kk, v in full(b).items()}
            e, e0_, phi = float(o['e']), float(o['e0']), o['phi']
            P1 = phi.shape[0] + 1                                               # + bias column (no force / stress)
            J = np.concatenate([np.transpose(o['J'][:, :n, :], (1, 2, 0)).reshape(3 * n, -1),
                                np.zeros((3 * n, 1))], axis=1)
            yF = np.asarray(g.nodes['forces'], np.float64).reshape(-1)
            fF, fR = o['f'][:n].reshape(-1), o['f0'][:n].reshape(-1)
            blocks = [('F', J, yF, fF, fR)]
            if use_stress:
                Sr = np.concatenate([o['S'].T, np.zeros((6, 1))], axis=1)       # (6, P + 1)
                yS = (np.asarray(g.globals['stress'], np.float64).reshape(-1)[:6] if stress_labels
                      else np.full(6, np.nan))
                blocks.append(('S', Sr, yS, o['s'], o['s0']))
            for nm, A, y, full_, rest in blocks:
                if accumulate:
                    gram.setdefault(nm, Gram(P1)).add(A, np.nan_to_num(y - rest))
                else:
                    key = {'F': ('J', 'yF', 'fFull', 'fRest'), 'S': ('Srows', 'yS', 'sFull', 'sRest')}[nm]
                    for kk, v in zip(key, (A, y, full_, rest)):
                        rows[kk].append(v)
        else:
            e, e0_, phi = (np.asarray(x) for x in energy_only(b))
            e, e0_ = float(e), float(e0_)
        rows['eFull'].append(e); rows['eRest'].append(e0_); rows['phi'].append(phi)
        if ens_models:
            outs = [efs(m, b) for m in ens_models]
            rows['ensE'].append([float(x[0]) for x in outs])
            if with_forces:
                rows['ensF'].append(np.stack([np.asarray(x[1])[:n].reshape(-1) for x in outs], 0))
                if use_stress:
                    rows['ensS'].append(np.stack([np.asarray(x[2]) for x in outs], 0))
        if k == 0:
            t_first = time.time() - t0
    print(f"    {len(graphs)} structures in {time.time() - t0:.0f} s (first incl. compile {t_first:.0f} s)", flush=True)
    out = {k: v for k, v in rows.items() if v}
    out['phi'] = np.stack(out['phi'])
    out['phi'] = np.concatenate([out['phi'], np.zeros((len(out['phi']), 1))], axis=1)   # bias column, set by the caller
    out['gram'] = gram
    return out


# ----------------------------------------------------------------------------------------------- Bayesian regression
class BLR:
    """theta ~ N(0, diag(alpha)^-1), alpha per group; one Gaussian likelihood block per observable (energy, force
    components, stress components) with its own precision beta_b:  y_b ~ N(A_b theta, 1/beta_b).
    blocks: list of (A^T A, A^T t, t.t, n)."""

    def __init__(self, groups, blocks):
        self.groups = np.asarray(groups)
        self.blocks = [(np.asarray(S), np.asarray(u), float(q), int(n)) for S, u, q, n in blocks]
        self.P = self.blocks[0][0].shape[0]
        self.ng = int(self.groups.max()) + 1

    def _post(self, x):
        alpha = np.exp(x[:self.ng])[self.groups]
        beta = np.exp(x[self.ng:])
        Lam = np.diag(alpha) + sum(bb * S for bb, (S, *_) in zip(beta, self.blocks))
        C = np.linalg.cholesky(Lam)
        rhs = sum(bb * u for bb, (_, u, _, _) in zip(beta, self.blocks))
        m = np.linalg.solve(C.T, np.linalg.solve(C, rhs))
        return alpha, beta, C, m

    def log_evidence(self, x):
        try:
            alpha, beta, C, m = self._post(x)
        except np.linalg.LinAlgError:
            return -1e300
        val = 0.5 * np.sum(np.log(alpha)) - np.sum(np.log(np.diag(C))) - 0.5 * m @ (alpha * m)
        for bb, (S, u, q, n) in zip(beta, self.blocks):
            ss = q - 2 * m @ u + m @ S @ m
            val += 0.5 * n * np.log(bb) - 0.5 * bb * ss - 0.5 * n * np.log(2 * np.pi)
        return float(val)

    def fit(self, x0):
        from scipy.optimize import minimize
        r = minimize(lambda x: -self.log_evidence(x), x0, method='L-BFGS-B', options={'maxiter': 2000})
        self.x = r.x
        self.alpha, self.beta, C, self.m = self._post(r.x)
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
    ap.add_argument('--max-train', type=int, default=None, help='random training subset for the posterior')
    ap.add_argument('--jac-chunk', type=int, default=8, help='feature VJPs per vectorised chunk (GPU memory)')
    ap.add_argument('--stress', choices=['auto', 'on', 'off'], default='auto',
                    help='stress likelihood block (periodic data with a stress label; auto: when every training '
                         'structure has a nonzero stress label)')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg, load, model, zs, e0, tr, va, te = build(a.config, a.ckpt, a.backend)
    periodic = bool(cfg.get('periodic', False))
    model0, n_ro = zero_readout(model)
    L = len(model.layers)
    ens_models = [load(p) for p in a.ensemble]
    print(f"model: {L} layers, {n_ro} readout parameter arrays zeroed in the reference copy; ensemble {len(ens_models)}"
          f"; periodic {periodic}", flush=True)
    fns = make_fns(model, model0, periodic, a.jac_chunk)
    cutoff = float(cfg.get('cutoff', 5.0))

    def graphs_of(ds):
        return [ds[i] for i in range(len(ds))]
    train = graphs_of(tr)
    if a.max_train and len(train) > a.max_train:
        idx = np.sort(np.random.default_rng(0).choice(len(train), a.max_train, replace=False))
        train = [train[i] for i in idx]
    has_s = periodic and all('stress' in g.globals and np.any(np.asarray(g.globals['stress']) != 0) for g in train)
    fit_stress = periodic and (a.stress == 'on' or (a.stress == 'auto' and has_s))
    if fit_stress and not has_s:
        raise SystemExit('--stress on: some training structures have no (nonzero) stress label')
    print(f"stress likelihood block: {fit_stress} (labels present: {has_s}); stress rows: {periodic}", flush=True)
    sets = {'train': (train, True), 'valid': (graphs_of(va), True)}
    for k, ds in te.items():
        gs = graphs_of(ds)
        sets[k] = (gs[:a.max_test] if a.max_test else gs, True)
    if a.dihedral:
        gs, hf = load_xyz_graphs(a.dihedral, zs, e0, cutoff)
        sets['dihedral'] = (gs[:a.max_test] if a.max_test else gs, hf)
    data = {}
    for name, (gs, wf) in sets.items():
        if not gs:
            continue
        print(f"  {name}: computing design rows", flush=True)
        lab = fit_stress and all('stress' in g.globals for g in gs)
        data[name] = arrays_for(gs, fns, a.n_bucket, a.e_bucket, wf, ens_models if name != 'train' else (),
                                use_stress=periodic, accumulate=(name == 'train'), stress_labels=lab)
        data[name]['phi'][:, -1] = np.asarray(data[name]['N'], np.float64) * float(model.scale[...])   # bias column

    P = data['train']['phi'].shape[1]
    d = (P - 1) // L
    groups = np.array([t for t in range(L) for _ in range(d)] + [L])
    D = data['train']
    AE, tE = D['phi'], np.asarray(D['yE']) - np.asarray(D['eRest'])
    dE = np.asarray(D['eFull']) - np.asarray(D['eRest'])

    # exactness check: the trained network's linear part reproduced by the features (theta fitted on the energies,
    # then forces / stress with the same theta: RMS of the remainder)
    th_E = np.linalg.lstsq(AE, dE, rcond=None)[0]
    resE = np.abs(AE @ th_E - dE).max()
    lin = {'E_max_eV': float(resE)}
    # forces / stress with the same theta, directly on the stored rows of the first evaluation set (a residual from
    # the training Gram sums cancels below sqrt(eps) * |F|, so it cannot resolve exactness)
    ev = next((k for k in data if k != 'train' and 'J' in data[k]), None)
    if len(dE) <= P:
        print(f"linearity check skipped for forces / stress: {len(dE)} training structures <= {P} features (theta from "
              "the energies is not unique)", flush=True)
        ev = None
    if ev is not None:
        De = data[ev]
        for nm, (kA, kf, kr) in {'F': ('J', 'fFull', 'fRest'), 'S': ('Srows', 'sFull', 'sRest')}.items():
            if kA in De:
                lin[f'{nm}_max_{ev}'] = float(max(np.abs(A @ th_E - (f - r)).max()
                                                  for A, f, r in zip(De[kA], De[kf], De[kr])))
    print(f"linearity check ({P} features, theta from train energies): max |E - rest - phi.theta| {resE:.2e} eV"
          + ''.join(f"; {k} {v:.2e}" for k, v in lin.items() if k != 'E_max_eV')
          + " (F eV/A, S eV/A^3)", flush=True)

    blocks = [(AE.T @ AE, AE.T @ tE, float(tE @ tE), len(tE))]
    names = ['E']
    for nm in ('F', 'S') if fit_stress else ('F',):
        if nm in D['gram']:
            gm = D['gram'][nm]
            blocks.append((gm.S, gm.u, gm.q, gm.n)); names.append(nm)
    blr = BLR(groups, blocks)
    var0 = [np.var(tE - AE @ th_E) + 1e-12] + [max(D['gram'][nm].q / max(D['gram'][nm].n, 1), 1e-12)
                                                for nm in names[1:]]
    x0 = np.concatenate([np.zeros(L + 1), np.log(1 / np.asarray(var0))])
    r = blr.fit(x0)
    beta = dict(zip(names, blr.beta))
    unit = {'E': (1e3, 'meV'), 'F': (1e3, 'meV/A'), 'S': (160.21766, 'GPa')}
    print(f"evidence maximised ({r.message}): log evidence {blr.logev:.1f}; noise "
          + ', '.join(f"sigma_{k} {unit[k][0] / np.sqrt(v):.3g} {unit[k][1]}" for k, v in beta.items())
          + f"; prior sigma per layer {np.round(1 / np.sqrt(np.exp(r.x[:L + 1])), 4)}", flush=True)

    report = {'P': P, 'periodic': periodic, 'stress_block': fit_stress, 'n_train': len(train), 'linearity': lin,
              'log_evidence': blr.logev, 'sigma_E_meV': 1e3 / np.sqrt(beta['E']), 'sets': {}}
    if 'F' in beta:
        report['sigma_F_meVA'] = 1e3 / np.sqrt(beta['F'])
    if 'S' in beta:
        report['sigma_S_GPa'] = 160.21766 / np.sqrt(beta['S'])
    for name, D in data.items():
        if name == 'train':
            continue
        yE = np.asarray(D['yE']); phi = D['phi']
        mE = np.asarray(D['eRest']) + phi @ blr.m
        vE_epi = np.einsum('ij,jk,ik->i', phi, blr.Sigma, phi)
        rep = {'n': len(yE),
               'net_E_rmse_meV': 1e3 * float(np.sqrt(np.mean((np.asarray(D['eFull']) - yE) ** 2))),
               'E_blr_total': calib(1e3 * (mE - yE), 1e3 * np.sqrt(vE_epi + 1 / beta['E'])),
               'E_blr_epistemic_spearman': calib(1e3 * (mE - yE), 1e3 * np.sqrt(vE_epi))['spearman']}
        if 'ensE' in D:
            ensE = np.asarray(D['ensE'])
            rep['E_ensemble'] = calib(1e3 * (ensE.mean(1) - yE), 1e3 * ensE.std(1, ddof=1))
        for nm, (kA, ky, kf, kr, ke) in {'F': ('J', 'yF', 'fFull', 'fRest', 'ensF'),
                                        'S': ('Srows', 'yS', 'sFull', 'sRest', 'ensS')}.items():
            if kA not in D:
                continue
            c = unit[nm][0]
            y = np.concatenate(D[ky])
            mu = np.concatenate(D[kr]) + np.concatenate([A @ blr.m for A in D[kA]])
            v_epi = np.concatenate([np.einsum('ij,jk,ik->i', A, blr.Sigma, A) for A in D[kA]])
            rep[f'{nm}_blr_epistemic_mean_sigma'] = float(np.mean(c * np.sqrt(v_epi)))
            if np.isnan(y).any():                       # no labels (stress of a set without stress): sigma only
                continue
            rep[f'net_{nm}_rmse'] = c * float(np.sqrt(np.mean((np.concatenate(D[kf]) - y) ** 2)))
            if nm in beta:
                rep[f'{nm}_blr_total'] = calib(c * (mu - y), c * np.sqrt(v_epi + 1 / beta[nm]))
            rep[f'{nm}_blr_epistemic_spearman'] = calib(c * (mu - y), c * np.sqrt(v_epi))['spearman']
            if ke in D:
                ens = (np.concatenate(D[ke], axis=1) if nm == 'F' else
                       np.concatenate([x for x in D[ke]], axis=1))          # (n_models, components)
                rep[f'{nm}_ensemble'] = calib(c * (ens.mean(0) - y), c * ens.std(0, ddof=1))
        report['sets'][name] = rep
        # per-structure arrays for OOD analyses (research/diag/ood_summary.py): energies and structure-mean force sigma
        dump = {'yE': yE, 'mE': mE, 'netE': np.asarray(D['eFull']), 'sigE_epi': np.sqrt(vE_epi),
                'sigE_tot': np.sqrt(vE_epi + 1 / beta['E'])}
        if 'J' in D and 'F' in beta:
            ns = [A.shape[0] for A in D['J']]; cut = np.cumsum([0] + ns)
            yF = np.concatenate(D['yF']); muF = np.concatenate(D['fRest']) + np.concatenate([A @ blr.m for A in D['J']])
            vF = np.concatenate([np.einsum('ij,jk,ik->i', A, blr.Sigma, A) for A in D['J']])
            dump.update({'F_err_rms': np.array([np.sqrt(np.mean((muF[a:b] - yF[a:b]) ** 2)) for a, b in zip(cut[:-1], cut[1:])]),
                         'F_sig_epi_mean': np.array([np.mean(np.sqrt(vF[a:b])) for a, b in zip(cut[:-1], cut[1:])]),
                         'F_sig_tot_mean': np.array([np.mean(np.sqrt(vF[a:b] + 1 / beta['F'])) for a, b in zip(cut[:-1], cut[1:])]),
                         'F_err': muF - yF, 'F_sig_epi': np.sqrt(vF), 'F_sig_tot': np.sqrt(vF + 1 / beta['F'])})
        np.savez(out / f'per_structure_{name}.npz', **dump)
        line = (f"{name:>10} n={rep['n']:5d} | net E {rep['net_E_rmse_meV']:.2f} | BLR E rmse {rep['E_blr_total']['rmse']:.2f} "
                f"z {rep['E_blr_total']['rms_z']:.2f} cov95 {rep['E_blr_total']['cov95']:.2f} rho {rep['E_blr_total']['spearman']:.2f}")
        if 'E_ensemble' in rep:
            line += (f" | ens E rmse {rep['E_ensemble']['rmse']:.2f} z {rep['E_ensemble']['rms_z']:.2f} "
                     f"rho {rep['E_ensemble']['spearman']:.2f}")
        for nm in ('F', 'S'):
            if f'{nm}_blr_total' in rep:
                t = rep[f'{nm}_blr_total']
                line += (f" || net {nm} {rep[f'net_{nm}_rmse']:.3g} | BLR {nm} rmse {t['rmse']:.3g} z {t['rms_z']:.2f} "
                         f"cov95 {t['cov95']:.2f} rho {t['spearman']:.2f}")
                if f'{nm}_ensemble' in rep:
                    t = rep[f'{nm}_ensemble']
                    line += f" | ens {nm} rmse {t['rmse']:.3g} z {t['rms_z']:.2f} rho {t['spearman']:.2f}"
        print(line, flush=True)
    json.dump(report, open(out / 'bayes_lastlayer.json', 'w'), indent=1)
    np.savez(out / 'blr_posterior.npz', m=blr.m, Sigma=blr.Sigma, groups=groups, x=blr.x, blocks=np.array(names))
    print('written', out / 'bayes_lastlayer.json')


if __name__ == '__main__':
    main()


load_xyz_graphs = load_xyz_graphs          # old name
