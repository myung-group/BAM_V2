"""Round-19: first-order Taylor augmentation of the training set (training.train, config taylor_aug)."""
import json
import numpy as np
import pytest
from flax import nnx

from conftest import make_model, pad_single
from race_v2.training.utils import compute_loss


class _DS:
    def __init__(self, graphs):
        self.graphs = graphs


def _augment(graphs, n_aug, delta, seed=42):
    """Run the taylor_aug block of training.train.train on a list of graphs (extracted logic)."""
    rng = np.random.RandomState(seed + 7)
    out = []
    for g in graphs:
        nodes = dict(g.nodes)
        n = int(g.n_node[0])
        w_orig = nodes.get('loss_weight', np.ones((n,), np.float32))
        out.append(g._replace(nodes={**nodes, 'loss_weight': np.asarray(w_orig, np.float32) * (1 + n_aug)}))
        pos = np.asarray(nodes['positions']); frc = np.asarray(nodes['forces'], np.float64)
        e = np.asarray(g.globals['energy'])
        for _ in range(n_aug):
            d = rng.normal(size=(n, 3)); d = d / np.linalg.norm(d, axis=1, keepdims=True) * delta
            out.append(g._replace(nodes={**nodes, 'positions': (pos.astype(np.float64) + d).astype(pos.dtype),
                                         'loss_weight': np.zeros((n,), np.float32)},
                                  globals={**g.globals, 'energy': (e.astype(np.float64) - np.sum(frc * d)).astype(e.dtype)}))
    return out


def test_taylor_aug_source_matches_train_3bpa():
    """The augmentation logic above must be the one in training.train (guard against drift)."""
    import inspect
    from race_v2.training import train as T
    src = inspect.getsource(T.train)
    for frag in ("ta_cfg = _get(cfg, 'taylor_aug', None)", "np.sum(frc * d)", "'loss_weight': np.zeros((n,), np.float32)",
                 "* (1 + n_aug)", "rng.normal(size=(n, 3))"):
        assert frag in src, frag


def test_taylor_aug_labels_and_weights(graph):
    n_aug, delta = 2, 0.01
    out = _augment([graph], n_aug, delta)
    assert len(out) == 1 + n_aug
    n = int(graph.n_node[0])
    orig, augs = out[0], out[1:]
    assert np.all(np.asarray(orig.nodes['loss_weight']) == 1 + n_aug)
    assert np.array_equal(np.asarray(orig.nodes['positions']), np.asarray(graph.nodes['positions']))
    pos0 = np.asarray(graph.nodes['positions'], np.float64); f0 = np.asarray(graph.nodes['forces'], np.float64)
    e0 = float(np.asarray(graph.globals['energy']).reshape(-1)[0])
    for a in augs:
        d = np.asarray(a.nodes['positions'], np.float64) - pos0
        assert np.allclose(np.linalg.norm(d, axis=1), delta, rtol=1e-3)          # |d_i| = delta (float32 positions)
        e = float(np.asarray(a.globals['energy']).reshape(-1)[0])
        assert abs((e - e0) - (-np.sum(f0 * d))) < 1e-4                           # first-order Taylor label
        assert np.all(np.asarray(a.nodes['loss_weight']) == 0.0)
        assert np.array_equal(np.asarray(a.nodes['forces']), np.asarray(graph.nodes['forces']))
        assert np.array_equal(np.asarray(a.senders), np.asarray(graph.senders))
    # mean node weight over all atoms is 1
    w = np.concatenate([np.asarray(g.nodes['loss_weight']) for g in out])
    assert abs(w.mean() - 1.0) < 1e-6
    # the two copies differ
    assert not np.array_equal(np.asarray(augs[0].nodes['positions']), np.asarray(augs[1].nodes['positions']))


def test_taylor_aug_copy_has_no_force_loss(graph):
    m = make_model(n_layers=2, update="linear", readout_mode="gated", message_norm="avg",
                   avg_n_neighbors=16.7, x_mode="basis")
    graphdef, params = nnx.split(m, nnx.Param)
    orig, aug = _augment([graph], 1, 0.01)
    _, a_aug = compute_loss(graphdef, params, pad_single(aug), 1.0, 1.0, 0.0, 1e6, periodic=False)
    _, a_org = compute_loss(graphdef, params, pad_single(orig), 1.0, 1.0, 0.0, 1e6, periodic=False)
    _, a_ref = compute_loss(graphdef, params, pad_single(graph), 1.0, 1.0, 0.0, 1e6, periodic=False)
    assert float(a_aug["force_loss"]) == 0.0
    assert float(a_aug["energy_loss"]) > 0.0
    assert abs(float(a_org["force_loss"]) - 2 * float(a_ref["force_loss"])) / float(a_ref["force_loss"]) < 1e-5
