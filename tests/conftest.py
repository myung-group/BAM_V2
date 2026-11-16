"""Shared fixtures: one 3BPA frame as a graph, and a small RACE model factory."""
import os
import sys
from pathlib import Path

import jax
import jraph
import numpy as np
import pytest
from flax import nnx

jax.config.update("jax_enable_x64", False)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from race_v2.data.xyz_graphs import (  # noqa: E402
    preprocess_graph, load_frames, atomic_numbers_to_indices,
)
from race_v2.models.race import RACE  # noqa: E402

ATOMIC_NUMBERS = [1, 6, 7, 8]
ATOM_ENERGIES = [-13.587222780835477, -1029.4889999855063,
                 -1484.9814568572233, -2041.9816003861047]
CUTOFF = 5.0
XYZ = REPO / "dataset_3BPA" / "train_300K.xyz"


@pytest.fixture(scope="session")
def frame():
    frames = load_frames(XYZ)
    rng = np.random.RandomState(0)
    return frames[rng.randint(len(frames))]


@pytest.fixture(scope="session")
def graph(frame):
    """Single, un-padded 3BPA graph (n_graphs == 1)."""
    return preprocess_graph(frame, atomic_numbers_to_indices(ATOMIC_NUMBERS),
                            ATOM_ENERGIES, CUTOFF)


def pad_single(g: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Pad a single graph with one padding graph (as the data loader does)."""
    n_node = int(g.n_node.sum()) + 1
    n_edge = int(g.n_edge.sum()) + 8
    return jraph.pad_with_graphs(g, n_node, n_edge, 2)


def make_model(x_mode="species", cov_mode="scalar6", seed=0, **kw):
    """Small RACE (fast on CPU) with the production code path."""
    args = dict(
        n_species=len(ATOMIC_NUMBERS), lmax=2, cutoff=CUTOFF,
        hidden_irreps="16x0e + 16x1o + 16x2e", n_layers=2,
        radial_basis_size=8, radial_mlp_size=16, radial_mlp_layers=2,
        mlp_init_scale=4.0, shift=0.0, scale=1.0, avg_n_neighbors=20.0,
        atom_energies=ATOM_ENERGIES, l_train=True, periodic=False,
        x_mode=x_mode, cov_mode=cov_mode, rngs=nnx.Rngs(seed),
    )
    args.update(kw)
    return RACE(**args)


@pytest.fixture
def x64():
    """Enable float64 for one test (the training code forces float32 at import)."""
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def as_float64(model, graph):
    """Return (model, graph) with all parameters / positions cast to float64.
    Must be called with x64 enabled."""
    import jax.numpy as jnp
    graphdef, params = nnx.split(model, nnx.Param)
    params = jax.tree.map(lambda x: x.astype(jnp.float64), params)
    g = graph._replace(nodes={**graph.nodes,
                              "positions": np.asarray(graph.nodes["positions"], np.float64),
                              "forces": np.asarray(graph.nodes["forces"], np.float64)})
    return nnx.merge(graphdef, params), g


def random_rotation(seed=0):
    rng = np.random.RandomState(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)
