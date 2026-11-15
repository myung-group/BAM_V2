"""Pairwise C6 dispersion of RACE_POLAR (polar.dispersion, 2026-11-15): C6 from the Thole polarizabilities,
Tkatchenko-Scheffler combination, Becke-Johnson damping.  Open-boundary model (the OMol25 case): the energy against
a numpy reference, graphs of a batch independent, forces against central differences, the r^-6 tail beyond the
network cutoff."""
import numpy as np
import pytest
from flax import nnx

from test_race_polar_qeq import KW, LR, POLAR, _batch, _molecule, _perturbed, _with_params_of  # noqa: F401

DISP = {'dispersion': True}


def _model(seed=0, polar=None, lr=None):
    from bam_omat24.race_polar_nnx import RACE_POLAR
    return _perturbed(RACE_POLAR(**KW, rngs=nnx.Rngs(seed), long_range={**LR, 'qeq_block': 12, **(lr or {})},
                                 polar={**POLAR, **DISP, **(polar or {})}), seed)


def _ne(model, b):
    """the full node-level output dict (with 'polar' and 'dispersion_node_energies')"""
    return model.node_energies(b.nodes['positions'], None, None, b)


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _reference(model, st, alpha):
    """numpy: E_i = 1/2 sum_j -C6_ij / (r^6 + R0^6) for one open-boundary structure"""
    from conftest import ATOMIC_NUMBERS
    sp = np.array([ATOMIC_NUMBERS.index(z) for z in st['numbers']])
    kap = _softplus(np.asarray(model.disp_kappa[...], np.float64))[sp]
    a1 = float(_softplus(np.asarray(model.disp_a1[...], np.float64)))
    c6 = kap * alpha ** 2
    rv = 0.529177 * 2.54 * (alpha / 0.148185) ** (1.0 / 7.0)
    pos = np.asarray(st['positions'], np.float64)
    n = len(sp)
    e = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cij = 2 * c6[i] * c6[j] / (alpha[j] / alpha[i] * c6[i] + alpha[i] / alpha[j] * c6[j])
            r0 = a1 * (rv[i] + rv[j])
            r = np.linalg.norm(pos[i] - pos[j])
            e[i] += -0.5 * cij / (r ** 6 + r0 ** 6)
    return e


def test_dispersion_options():
    from bam_omat24.race_polar_nnx import RACE_POLAR
    with pytest.raises(ValueError):                    # needs the per-graph blocks of qeq_block
        RACE_POLAR(**KW, rngs=nnx.Rngs(0), long_range=dict(LR), polar={**POLAR, **DISP})
    with pytest.raises(ValueError):
        RACE_POLAR(**KW, rngs=nnx.Rngs(0), long_range={**LR, 'qeq_block': 12},
                   polar={**POLAR, **DISP, 'disp_kappa_init': 0.0})
    m = RACE_POLAR(**KW, rngs=nnx.Rngs(0), long_range={**LR, 'qeq_block': 12}, polar={**POLAR, **DISP})
    assert m.polar['dispersion'] and m.disp_kappa[...].shape == (4,)


def test_dispersion_energy_batch_forces(x64):
    sts = [_molecule(1), _molecule(2, charge=1.0), _molecule(3, charge=-1.0, n=8)]
    model = _model()
    out = model(_batch(sts))
    ne = _ne(model, _batch(sts))
    n = [len(s['numbers']) for s in sts]
    off = np.concatenate([[0], np.cumsum(n)])
    ed = np.asarray(ne['dispersion_node_energies'])
    assert np.abs(np.asarray(out['dispersion_node_energies']) - ed).max() < 1e-12      # passed through to the outputs
    alpha = np.asarray(ne['polar']['polarizabilities'])
    assert np.all(ed[off[-1]:] == 0.0) and float(ed[:off[-1]].max()) < 0.0
    for g, st in enumerate(sts):                       # numpy reference, per atom
        ref = _reference(model, st, alpha[off[g]:off[g + 1]])
        assert np.abs(ed[off[g]:off[g + 1]] - ref).max() < 1e-10 * max(1.0, np.abs(ref).max())
    # the dispersion is in the energy: the same parameters without it differ by exactly its sum
    nod = _with_params_of(model, _model(polar={'dispersion': False}))
    on = nod(_batch(sts))
    for g in range(len(sts)):
        assert abs(float(out['energy'][g]) - float(on['energy'][g]) - ed[off[g]:off[g + 1]].sum()) < 1e-9
    for g, st in enumerate(sts):                       # every graph alone = in the batch
        o1 = model(_batch([st], extra_nodes=2, extra_edges=5))
        assert abs(float(o1['energy'][0]) - float(out['energy'][g])) < 1e-8
    h = 1e-5                                           # forces = -dE/dx including the dispersion
    f = np.asarray(out['forces'])
    for g, i, k in [(0, 2, 0), (2, 5, 1)]:
        ep = []
        for sgn in (+1, -1):
            s2 = [dict(s, positions=s['positions'].copy()) for s in sts]
            s2[g]['positions'][i, k] += sgn * h
            ep.append(np.asarray(model(_batch(s2))['energy'])[:len(s2)].sum())
        fd = -(ep[0] - ep[1]) / (2 * h)
        assert abs(fd - f[off[g] + i, k]) < 2e-6 * max(1.0, abs(fd)), (g, i, k, fd, f[off[g] + i, k])


def test_dispersion_overflow_is_nan(x64):
    """a structure larger than qeq_block (12 atoms here) gets NaN, not a silent omission of the extra atoms"""
    model = _model()
    ok = np.asarray(_ne(model, _batch([_molecule(6, n=8)]))['dispersion_node_energies'])
    big = np.asarray(_ne(model, _batch([_molecule(6, n=14)]))['dispersion_node_energies'])
    assert np.all(np.isfinite(ok)) and np.isnan(big).any()


def test_dispersion_tail_beyond_the_cutoff(x64):
    """two molecules far outside the 5 A network cutoff: their dispersion interaction is there and falls as r^-6"""
    a, b = _molecule(4, n=5), _molecule(5, n=5)
    model = _model(seed=1)

    def e_disp(d):
        pos = np.concatenate([a['positions'], b['positions'] + np.array([d, 0.0, 0.0])])
        st = dict(a, numbers=np.concatenate([a['numbers'], b['numbers']]), positions=pos, forces=np.zeros((10, 3)))
        return float(np.asarray(_ne(model, _batch([st]))['dispersion_node_energies']).sum())
    ea = float(np.asarray(_ne(model, _batch([a]))['dispersion_node_energies']).sum())
    eb = float(np.asarray(_ne(model, _batch([b]))['dispersion_node_energies']).sum())
    e15, e30 = e_disp(15.0) - ea - eb, e_disp(30.0) - ea - eb
    assert e15 < 0.0 and e30 < 0.0
    assert 0.8 < (e15 / e30) / 64.0 < 1.25, e15 / e30


def test_dispersion_float32_gradients_finite():
    """float32 (the training dtype) with padding slots in the blocks: energy, forces and parameter gradients finite
    (pre-flight 2026-11-15: a 0 / (0 + 1e-30) on the padding gave NaN parameter gradients in float32)"""
    import jax
    import jax.numpy as jnp
    from bam_omat24.race_polar_nnx import RACE_POLAR
    sts = [_molecule(1), _molecule(2, n=6)]
    res = {}
    for disp in (False, True):
        # the CG settings of the training configs (the test POLAR ones, 1e-12 / 200, are float64-only)
        m = RACE_POLAR(**KW, rngs=nnx.Rngs(0), long_range={**LR, 'qeq_block': 12},
                       polar={**POLAR, 'cg_tolerance': 1e-4, 'cg_max_iterations': 25, 'dispersion': disp})
        res[disp] = _f32_grads(m, sts)
    assert not res[False] and not res[True], ('without dispersion', res[False][:8], 'with', res[True][:8])


def _f32_grads(m, sts):
    import jax
    import jax.numpy as jnp
    b = jax.tree.map(lambda x: x.astype(jnp.float32) if hasattr(x, 'dtype') and x.dtype == np.float64 else x,
                     _batch(sts, extra_nodes=5, extra_edges=9))
    gd, st = nnx.split(m, nnx.Param)

    def loss(st):
        out = nnx.merge(gd, st)(b)
        return jnp.sum(out['energy'][:2] ** 2) + jnp.sum(out['forces'] ** 2)
    val, grads = jax.value_and_grad(loss)(st)
    bad = [jax.tree_util.keystr(k) for k, v in jax.tree_util.tree_leaves_with_path(grads)
           if not bool(jnp.all(jnp.isfinite(v)))]
    return bad if np.isfinite(float(val)) else ['loss'] + bad
