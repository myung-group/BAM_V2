"""RACE_POLAR with polar.charges "qeq" (2026-11-11): the charges of the Thole block are the charge-equilibration
charges of RACE_LR.  Open-boundary model (the OMol25 case).  Checked here: exact total charges, independence of the
graphs of a batch, forces against central differences, the blocked Qeq path, and the energy bookkeeping (Qeq
contributes its on-site energy only; the pair Coulomb energy is counted once, in the Thole block)."""
import math

import numpy as np
import jax
import jraph
import pytest
from flax import nnx

from conftest import ATOMIC_NUMBERS, ATOM_ENERGIES
from race_v2.training.train import COVALENT_RADII  # noqa: F401  (imports training.train before the x64 fixture)
from race_v2.data import stream_io as S
from race_v2.data.stream_data import _to_graph, _cast

CUTOFF = 5.0
IDX = {z: i for i, z in enumerate(ATOMIC_NUMBERS)}
FIELDS = dict(charge_key='charge', spin_key='spin')
KW = dict(n_species=4, lmax=2, cutoff=CUTOFF, hidden_irreps="16x0e + 16x1o + 16x2e", n_layers=2, radial_basis_size=8,
          radial_mlp_size=16, radial_mlp_layers=2, mlp_init_scale=4.0, shift=0.0, scale=1.0, avg_n_neighbors=20.0,
          atom_energies=ATOM_ENERGIES, l_train=True, periodic=False, x_mode="species",
          species_radii=[0.31, 0.76, 0.71, 0.66], species_z=ATOMIC_NUMBERS)
LR = {'charge_spin_embedding': True, 'n_tasks': 3, 'chi_init': 'pauling'}
POLAR = {'charges': 'qeq', 'cg_tolerance': 1e-12, 'cg_max_iterations': 200, 'multipole_init_scale': 1.0,
         'max_polarizability_per_element': {'H': 1.0}}


def _name(path):
    return '/'.join(str(getattr(k, 'key', getattr(k, 'name', k))) for k in path)


def _perturbed(m, seed, noise=0.3, parts=('pol_', 'lr_head', 'chi0', 'hard0')):
    """float64 parameters; the electrostatic heads (zero-initialised in part) get noise so that every term is active."""
    import jax.numpy as jnp
    graphdef, params = nnx.split(m, nnx.Param)
    key = jax.random.PRNGKey(seed + 1)
    leaves, _ = jax.tree.flatten_with_path(params)
    out = []
    for i, (path, x) in enumerate(leaves):
        x = jnp.asarray(x, jnp.float64)
        if any(p in _name(path) for p in parts):
            x = x + noise * jax.random.normal(jax.random.fold_in(key, i), x.shape, jnp.float64)
        out.append(x)
    return nnx.merge(graphdef, jax.tree.unflatten(jax.tree.structure(params), out))


def _polar_model(seed=0, lr=None, polar=None):
    from race_v2.models.race_polar import RACE_POLAR
    return _perturbed(RACE_POLAR(**KW, rngs=nnx.Rngs(seed), long_range={**LR, **(lr or {})},
                                 polar={**POLAR, **(polar or {})}), seed)


def _with_params_of(src, dst):
    """dst with the parameter values of src for every leaf of dst (same attribute paths)."""
    _, ps = nnx.split(src, nnx.Param)
    gd, pd = nnx.split(dst, nnx.Param)
    by = {_name(p): x for p, x in jax.tree.flatten_with_path(ps)[0]}
    new = [by[_name(p)] for p, _ in jax.tree.flatten_with_path(pd)[0]]
    return nnx.merge(gd, jax.tree.unflatten(jax.tree.structure(pd), new))


def _molecule(seed, charge=0.0, n=9):
    rng = np.random.default_rng(seed)
    z = rng.choice(ATOMIC_NUMBERS, n)
    g = np.stack(np.meshgrid(*[np.arange(3)] * 3, indexing='ij'), -1).reshape(-1, 3)[rng.permutation(27)[:n]]
    pos = g * 1.6 + rng.normal(0, 0.15, (n, 3))
    return {'numbers': z, 'positions': pos, 'cell': np.zeros((3, 3)), 'pbc': np.zeros(3, bool), 'energy': 0.0,
            'forces': np.zeros((n, 3)), 'stress': None, 'info': {'charge': charge, 'spin': 1}, 'arrays': {}}


def _batch(sts, extra_nodes=3, extra_edges=11):
    gs = []
    for st in sts:
        d = _cast(S.graph_arrays(st, IDX, np.zeros(4), CUTOFF, dict(FIELDS)), np.float64)
        d['positions'] = np.asarray(st['positions'], np.float64)       # float64 coordinates for the differences
        gs.append(_to_graph(d))
    b = jraph.batch_np(gs)
    return jraph.pad_with_graphs(b, int(b.n_node.sum()) + extra_nodes, int(b.n_edge.sum()) + extra_edges, len(gs) + 1)


def test_options():
    from race_v2.models.race_polar import RACE_POLAR
    with pytest.raises(ValueError):          # the default (Fukui) charges do not go with the Qeq stratum
        RACE_POLAR(**KW, rngs=nnx.Rngs(0), long_range={**LR, 'qeq': True}, polar={})
    with pytest.raises(ValueError):          # Qeq charges need the Qeq stratum
        RACE_POLAR(**KW, rngs=nnx.Rngs(0), long_range={**LR, 'qeq': False}, polar={'charges': 'qeq'})
    with pytest.raises(ValueError):          # the pair Coulomb energy would be counted twice
        RACE_POLAR(**KW, rngs=nnx.Rngs(0), long_range={**LR, 'coulomb_energy': True}, polar={'charges': 'qeq'})
    with pytest.raises(ValueError):
        RACE_POLAR(**KW, rngs=nnx.Rngs(0), long_range=dict(LR), polar={'charges': 'qeq',
                                                                      'charge_constraint': 'electrode'})
    m = RACE_POLAR(**KW, rngs=nnx.Rngs(0), long_range=dict(LR), polar={'charges': 'qeq'})
    assert m.qeq and m.lr['coulomb_energy'] is False and m.polar['charges'] == 'qeq'


def test_charges_batch_forces_blocked(x64):
    sts = [_molecule(1), _molecule(2, charge=1.0), _molecule(3, charge=-1.0, n=8)]
    model = _polar_model()
    out = model(_batch(sts))
    n = [len(s['numbers']) for s in sts]
    off = np.concatenate([[0], np.cumsum(n)])
    q = np.asarray(out['charges'])
    for g, st in enumerate(sts):                       # exact total charge of every graph; padding nodes zero
        assert abs(q[off[g]:off[g + 1]].sum() - st['info']['charge']) < 1e-10
    assert np.all(q[off[-1]:] == 0.0) and np.all(np.isfinite(np.asarray(out['energy'])))
    assert float(np.abs(q[:off[-1]]).max()) > 1e-3
    # induction is active: the same parameters without induced dipoles give another energy
    no_ind = _with_params_of(model, _polar_model(polar={'induction': False}))
    assert abs(float(no_ind(_batch(sts))['energy'][0]) - float(out['energy'][0])) > 1e-6

    for g, st in enumerate(sts):                       # every graph alone = in the batch
        o1 = model(_batch([st], extra_nodes=2, extra_edges=5))
        assert abs(float(o1['energy'][0]) - float(out['energy'][g])) < 1e-8
        assert np.abs(np.asarray(o1['forces'])[:n[g]] - np.asarray(out['forces'])[off[g]:off[g + 1]]).max() < 1e-8
        assert np.abs(np.asarray(o1['charges'])[:n[g]] - q[off[g]:off[g + 1]]).max() < 1e-10
        assert np.abs(np.asarray(o1['dipole'])[0] - np.asarray(out['dipole'])[g]).max() < 1e-8

    def energy(sts2, m=model):
        return np.asarray(m(_batch(sts2))['energy'])[:len(sts2)].sum()
    h = 1e-5                                           # forces = -dE/dx: the charges move with the atoms
    f = np.asarray(out['forces'])
    for g, i, k in [(0, 2, 0), (1, 4, 1), (2, 7, 2), (1, 0, 0)]:
        ep = []
        for sgn in (+1, -1):
            s2 = [dict(s, positions=s['positions'].copy()) for s in sts]
            s2[g]['positions'][i, k] += sgn * h
            ep.append(energy(s2))
        fd = -(ep[0] - ep[1]) / (2 * h)
        assert abs(fd - f[off[g] + i, k]) < 2e-6 * max(1.0, abs(fd)), (g, i, k, fd, f[off[g] + i, k])

    # Qeq solved per graph (qeq_block) gives the same model
    blocked = _with_params_of(model, _polar_model(lr={'qeq_block': 12}))
    ob = blocked(_batch(sts))
    assert np.abs(np.asarray(ob['energy']) - np.asarray(out['energy'])).max() < 1e-8
    assert np.abs(np.asarray(ob['forces']) - np.asarray(out['forces'])).max() < 1e-8
    assert np.abs(np.asarray(ob['charges']) - q).max() < 1e-10


def test_energy_bookkeeping(x64):
    """No double counting.  With mu = Theta = 0 and no induction the Thole block is the Coulomb energy of Gaussian
    charges of width sigma (pair kernel erf(r / 2 sigma) / r, self energy q^2 / (2 sqrt(pi) sigma)):
        E(RACE_POLAR, charges qeq) - E(RACE_LR, coulomb_energy false) = that energy of the Qeq charges,
        E(RACE_LR) - E(RACE_LR, coulomb_energy false)                 = 1/2 q . gamma . q of the Qeq kernel."""
    from race_v2.models import long_range, thole
    from race_v2.models.race_lr import RACE_LR
    sigma_t = 0.8
    sts = [_molecule(4, charge=1.0), _molecule(5), _molecule(6, charge=-2.0, n=7)]
    batch = _batch(sts)
    a = _polar_model(polar={'induction': False, 'quadrupoles': False, 'multipole_init_scale': 0.0, 'sigma': sigma_t})
    b = _with_params_of(a, RACE_LR(**KW, rngs=nnx.Rngs(0), long_range={**LR, 'coulomb_energy': False}))
    c = _with_params_of(a, RACE_LR(**KW, rngs=nnx.Rngs(0), long_range=dict(LR)))
    oa, ob, oc = a(batch), b(batch), c(batch)
    q = np.asarray(oa['charges'])
    assert np.abs(q - np.asarray(ob['charges'])).max() < 1e-12 and np.abs(q - np.asarray(oc['charges'])).max() < 1e-12
    sig_q = np.exp(np.asarray(c.log_sigma[...], np.float64))
    off = 0
    for g, st in enumerate(sts):
        n = len(st['numbers'])
        qg, pos = q[off:off + n], st['positions']
        r = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        iu = np.triu_indices(n, 1)
        erf = np.vectorize(math.erf)
        e_thole = thole.K_COULOMB * ((qg[iu[0]] * qg[iu[1]] * erf(r[iu] / (2 * sigma_t)) / r[iu]).sum()
                                     + (qg ** 2).sum() / (2 * math.sqrt(math.pi) * sigma_t))
        assert abs(float(oa['energy'][g]) - float(ob['energy'][g]) - e_thole) < 1e-8 * max(1.0, abs(e_thole)), g
        s = sig_q[[IDX[int(z)] for z in st['numbers']]]
        sij = np.sqrt(s[:, None] ** 2 + s[None] ** 2)
        e_pair = long_range.K_COULOMB * (qg[iu[0]] * qg[iu[1]] * erf(r[iu] / (math.sqrt(2) * sij[iu])) / r[iu]).sum()
        assert abs(float(oc['energy'][g]) - float(ob['energy'][g]) - e_pair) < 1e-8 * max(1.0, abs(e_pair)), g
        off += n
    # the blocked Qeq path keeps the same bookkeeping
    bb = _with_params_of(a, RACE_LR(**KW, rngs=nnx.Rngs(0), long_range={**LR, 'coulomb_energy': False,
                                                                       'qeq_block': 12}))
    assert np.abs(np.asarray(bb(batch)['energy']) - np.asarray(ob['energy'])).max() < 1e-8
