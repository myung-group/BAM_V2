"""Data parallel training (review §13): the multi-device step equals the single-device step on the
union of the device batches, and multi-device evaluation counts every graph exactly once.
Needs >= 2 devices (GPUs, or CPU with XLA_FLAGS=--xla_force_host_platform_device_count=4)."""
from collections import namedtuple
import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from flax import nnx
from jax.sharding import Mesh

from conftest import make_model, ATOMIC_NUMBERS, ATOM_ENERGIES, CUTOFF, XYZ
from bam_omat24.data_3bpa import preprocess_graph, load_frames, atomic_numbers_to_indices
from bam_omat24.data_nnx import stack_batches_for_devices, default_collate_fn
from bam_omat24.training_utils import (make_sharded_train_step, replicate, replicate_pytree, to_host,
                                       evaluate_sharded, make_sharded_evaluate_step)

SIZES = [27, 20, 12, 25, 9, 27, 15, 22]          # different atom counts per graph
Sched = namedtuple("Sched", "scale")
needs2 = pytest.mark.skipif(len(jax.devices()) < 2, reason="needs >= 2 devices")


def _graphs(sizes):
    frames = load_frames(XYZ)
    idx = atomic_numbers_to_indices(ATOMIC_NUMBERS)
    out = []
    for f, k in zip(frames, sizes):
        a = Atoms(numbers=f.numbers[:k], positions=f.positions[:k])
        a.calc = SinglePointCalculator(a, energy=f.get_potential_energy() * k / len(f), forces=f.get_forces()[:k])
        g = preprocess_graph(a, idx, ATOM_ENERGIES, CUTOFF)
        out.append(jax.tree.map(lambda x: x.astype(np.float64) if np.issubdtype(x.dtype, np.floating) else x,
                                jax.tree.map(np.asarray, g)))
    return out


def _model():
    model = make_model(n_layers=2)
    gd, params = nnx.split(model, nnx.Param)
    return gd, jax.tree.map(lambda x: x.astype(jnp.float64), params)


def _grads(groups, radial_l2=0.0):
    """Gradient (params - params after one SGD(lr=1) step) and loss of a data-parallel step with one
    device per group."""
    gd, params = _model()
    mesh = Mesh(np.array(jax.devices()[:len(groups)]), ("dp",))
    opt = optax.sgd(1.0)
    step = make_sharded_train_step(opt, mesh, energy_weight=1.0, force_weight=10.0, stress_weight=0.0,
                                   huber_delta=10.0, loss_type="huber", periodic=False, grad_clip=1e9,
                                   radial_l2=radial_l2)
    batch_size = max(len(g) for g in groups)
    batch, _ = stack_batches_for_devices([default_collate_fn(g, batch_size, 16, 256) for g in groups], mesh)
    p = replicate_pytree(params, mesh)
    new, _, _, _, metrics = step(gd, p, replicate_pytree(opt.init(params), mesh),
                                 replicate_pytree(Sched(jnp.asarray(1.0)), mesh), p,
                                 replicate(jnp.asarray(0), mesh), batch)
    g = jax.tree.map(lambda a, b: to_host(a) - to_host(b), params, new)
    return np.concatenate([x.ravel() for x in jax.tree.leaves(g)]), float(to_host(metrics["loss"]))


@needs2
@pytest.mark.parametrize("radial_l2", [0.0, 1e-3])
def test_two_devices_equal_union_batch(x64, radial_l2):
    gs = _graphs(SIZES)
    g_union, l_union = _grads([gs], radial_l2)
    g_dp, l_dp = _grads([gs[:4], gs[4:]], radial_l2)
    scale = np.abs(g_union).max()
    assert np.abs(g_dp - g_union).max() < 1e-10 * scale
    assert abs(l_dp - l_union) < 1e-10 * abs(l_union)
    # the previous pmean of per-device means is a different (atom-count weighted) update
    g_a, _ = _grads([gs[:4]], radial_l2)
    g_b, _ = _grads([gs[4:]], radial_l2)
    assert np.abs(0.5 * (g_a + g_b) - g_union).max() > 1e-3 * scale


@pytest.mark.skipif(len(jax.devices()) < 4, reason="needs >= 4 devices")
def test_four_devices_equal_union_batch(x64):
    gs = _graphs(SIZES)
    g_union, _ = _grads([gs])
    g_dp, _ = _grads([gs[0:2], gs[2:4], gs[4:6], gs[6:8]])
    assert np.abs(g_dp - g_union).max() < 1e-10 * np.abs(g_union).max()


class _DS:
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)


@needs2
def test_eval_counts_every_graph_once(x64):
    """11 graphs, batch 2 -> 6 batches: on 4 devices the last group has 2 padding copies."""
    gs = _graphs((SIZES * 2)[:11])
    gd, params = _model()
    collate = functools.partial(default_collate_fn, node_bucket_size=16, edge_bucket_size=256)
    res = {}
    for n in sorted({1, 2, min(4, len(jax.devices()))}):
        mesh = Mesh(np.array(jax.devices()[:n]), ("dp",))
        ev = make_sharded_evaluate_step(mesh, huber_delta=10.0, loss_type="huber", periodic=False)
        res[n] = evaluate_sharded(gd, params, [_DS(gs)], 2, n, mesh, energy_weight=1.0, force_weight=10.0,
                                  stress_weight=0.0, huber_delta=10.0, eval_step=ev, periodic=False,
                                  collate_fn=collate)
    for n, m in res.items():
        assert m["n_graphs"] == 11 and m["n_atoms"] == sum((SIZES * 2)[:11]), (n, m["n_graphs"])
        for k in ("energy_mae_total", "energy_rmse_total", "force_mae", "force_rmse", "total_loss"):
            assert abs(m[k] - res[1][k]) < 1e-10 * abs(res[1][k]), (n, k, m[k], res[1][k])
