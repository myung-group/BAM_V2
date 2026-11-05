"""Faithful DeNS (docs/RACE_DeNS_audit_2026-10-29.md): additive l<=lmax force encoding after layer 0
(nodes['dens_force']), gated-MLP noise head, corruption / loss plumbing of the train step."""
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jax.sharding import Mesh

from conftest import make_model, as_float64, random_rotation
from test_equivariance import _transform, TOL
from test_round24 import V9, _fd_forces
from bam_omat24.data_nnx import stack_batches_for_devices, default_collate_fn
from bam_omat24.training_utils import make_sharded_train_step, replicate, replicate_pytree, to_host

HILL = dict(**V9, residual="chebyshev", layer_scale=0.5, x_norm="hill")
Sched = namedtuple("Sched", "scale")


def _with_dens_force(g, seed=0, frac=0.5):
    rng = np.random.default_rng(seed)
    f = np.asarray(g.nodes["forces"])
    m = (rng.random(f.shape[0]) < frac).astype(f.dtype)
    return g._replace(nodes={**g.nodes, "dens_force": f * m[:, None]})


def test_dens_off_equals_base(x64, graph):
    """Without nodes['dens_force'] the DeNS model is the base model (same energy-model parameters)."""
    base, g = as_float64(make_model(n_layers=3, **HILL), graph)
    dm, _ = as_float64(make_model(n_layers=3, dens=True, dens_seed=5, **HILL), graph)
    eb, fb = np.asarray(base(g)["energy"]), np.asarray(base(g)["forces"])
    ed, fd = np.asarray(dm(g)["energy"]), np.asarray(dm(g)["forces"])
    # equal up to the run-to-run GPU summation-order noise of the base model itself (~1e-7 relative)
    assert np.abs(eb - ed).max() <= TOL * max(1.0, np.abs(eb).max())
    assert np.abs(fb - fd).max() <= TOL * max(1.0, np.abs(fb).max())
    # zero reference forces: the encoding is exactly zero
    g0 = g._replace(nodes={**g.nodes, "dens_force": np.zeros_like(np.asarray(g.nodes["forces"]))})
    assert np.abs(np.asarray(dm(g0)["energy"]) - eb).max() <= TOL * max(1.0, np.abs(eb).max())
    # non-zero reference forces change the energy
    gd = _with_dens_force(g)
    assert np.abs(np.asarray(dm(gd)["energy"]) - eb).max() > 1e-6


def test_dens_e3_and_forces(x64, graph):
    """Energy invariant / forces and noise head equivariant under joint transformation of positions and
    reference forces; forces = -dE/dr with the encoding active."""
    dm, g = as_float64(make_model(n_layers=3, dens=True, dens_seed=5, **HILL), graph)
    g = _with_dens_force(g)
    out0 = dm(g)
    e0, f0 = np.asarray(out0["energy"]), np.asarray(out0["forces"])
    n0 = np.asarray(dm.dens_noise(out0["node_features"]))
    assert np.abs(n0).max() > 1e-8
    R = random_rotation(1)
    for kw in (dict(R=R), dict(t=np.array([1.3, -2.1, 0.7])), dict(invert=True), dict(R=R, invert=True)):
        g2, M = _transform(g, **kw)
        g2 = g2._replace(nodes={**g2.nodes, "dens_force": np.asarray(g.nodes["dens_force"]) @ M.T})
        out2 = dm(g2)
        assert np.abs(np.asarray(out2["energy"]) - e0).max() <= TOL * max(1.0, np.abs(e0).max())
        assert np.abs(f0 @ M.T - np.asarray(out2["forces"])).max() <= TOL * max(1.0, np.abs(f0).max())
        n2 = np.asarray(dm.dens_noise(out2["node_features"]))
        assert np.abs(n0 @ M.T - n2).max() <= TOL * max(1.0, np.abs(n0).max()), kw.keys()
    _fd_forces(dm, g)


def test_dens_validation():
    with pytest.raises(ValueError):
        make_model(n_layers=3, dens=True, noise_head=True, **HILL)


def _step(graphs, dens, base_model=False):
    model = make_model(n_layers=3, **HILL) if base_model else make_model(n_layers=3, dens=True, dens_seed=5, **HILL)
    gd, params = nnx.split(model, nnx.Param)
    params = jax.tree.map(lambda x: x.astype(jnp.float64), params)
    mesh = Mesh(np.array(jax.devices()[:1]), ("dp",))
    opt = optax.sgd(1.0)
    step = make_sharded_train_step(opt, mesh, energy_weight=1.0, force_weight=10.0, stress_weight=0.0,
                                   huber_delta=10.0, loss_type="huber", periodic=False, grad_clip=1e9,
                                   dens=dens)
    batch, _ = stack_batches_for_devices([default_collate_fn(graphs, len(graphs), 16, 256)], mesh)
    p = replicate_pytree(params, mesh)
    new, _, _, _, met = step(gd, p, replicate_pytree(opt.init(params), mesh),
                             replicate_pytree(Sched(jnp.asarray(1.0)), mesh), p, replicate(jnp.asarray(3), mesh), batch)
    return params, new, {k: float(to_host(v)) for k, v in met.items()}


def test_dens_step_plumbing(x64):
    from test_data_parallel import _graphs
    gs = _graphs([27, 20, 12, 25, 9])
    # p = 0: nothing corrupted -> the base update on the energy-model parameters, zero noise loss
    base_p, base_new, base_m = _step(gs, None, base_model=True)
    for noisy_ef in (True, False):
        p, new, m = _step(gs, dict(p=0.0, r=0.25, sigma=0.05, noisy_ef=noisy_ef, weight=1e-3))
        assert m["dens_loss"] == 0.0 and abs(m["loss"] - base_m["loss"]) < 1e-12 * abs(base_m["loss"])
        bl = {jax.tree_util.keystr(k): to_host(a) - to_host(b) for (k, a), (_, b) in
              zip(jax.tree_util.tree_leaves_with_path(base_p), jax.tree_util.tree_leaves_with_path(base_new))}
        dl = {jax.tree_util.keystr(k): to_host(a) - to_host(b) for (k, a), (_, b) in
              zip(jax.tree_util.tree_leaves_with_path(p), jax.tree_util.tree_leaves_with_path(new))}
        common = [k for k in bl if k in dl]
        assert len(common) == len(bl)
        gmax = max(np.abs(bl[k]).max() for k in common)
        assert max(np.abs(bl[k] - dl[k]).max() for k in common) < 1e-10 * gmax
    # p = 1, r = 0.5: corrupted atoms, positive noise loss, forces on ~half of the atoms only
    for noisy_ef in (True, False):
        _, _, m = _step(gs, dict(p=1.0, r=0.5, sigma=0.05, noisy_ef=noisy_ef, weight=1e-3))
        assert m["dens_loss"] > 0.1 and np.isfinite(m["loss"])
        n_all = sum([27, 20, 12, 25, 9])
        if noisy_ef:
            assert 0.2 * n_all < m["n_atoms"] < 0.8 * n_all, m["n_atoms"]
        else:
            assert m["n_atoms"] == n_all
