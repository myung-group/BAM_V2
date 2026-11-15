"""Round-11 options: orbital-radial species embedding (embed_mode descriptor / both) and the
shell-resolved radial basis (radial_shells), plus sanity checks of the atomic-shell table."""
import json
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from conftest import make_model, as_float64, pad_single, ATOMIC_NUMBERS
from test_equivariance import _check_e3, TOL
from race_v2.training.utils import compute_loss
from race_v2.training.train import element_descriptors, element_shell_radii, SHELL_TABLE_PATH, BOHR_A

Z = list(ATOMIC_NUMBERS)
DESC = element_descriptors(Z, "both")
SHELLS = element_shell_radii(Z)


def _energy(model, graph):
    return float(np.asarray(model(graph)["energy"])[0])


def test_shell_table_physics():
    """4f is localised, 3d is compact, the outer s shells are diffuse; eigenvalues are bound."""
    t = json.load(open(SHELL_TABLE_PATH))
    assert len(t) == 94
    fe, gd, h, c = t["26"]["shells"], t["64"]["shells"], t["1"]["shells"], t["6"]["shells"]
    assert gd["f"]["n"] == 4 and fe["d"]["n"] == 3 and fe["s"]["n"] == 4 and gd["s"]["n"] == 6
    assert gd["f"]["r_mean"] < fe["d"]["r_mean"] < fe["s"]["r_mean"] < gd["s"]["r_mean"]
    assert abs(h["s"]["eps"] - (-0.233471)) < 2e-3                       # NIST LDA H 1s
    assert c["p"]["r_mean"] > c["s"]["r_mean"] > 0.5                       # C 2p more diffuse than 2s
    for z in range(1, 95):
        for name, sh in t[str(z)]["shells"].items():
            if sh is not None:
                assert sh["eps"] < 0 and sh["r_mean"] > 0


def test_descriptor_rows_shape_and_finite():
    rows = np.array(DESC)
    assert rows.shape == (4, 4 * 4 + 1 + 1)
    assert np.all(np.isfinite(rows))
    assert np.array(element_descriptors(Z, "radii")).shape == (4, 1)
    assert np.array(element_descriptors(Z, "shells")).shape == (4, 17)
    # H has no p shell -> its p "present" entry equals the standardised 0 of the table
    sh = np.array(SHELLS)
    assert sh.shape == (4, 4) and sh[0, 1] == 0.0 and np.all(sh[1:, :2] > 0) and np.all(sh[:, 2:] == 0)
    assert abs(sh[0, 0] - 1.673 * BOHR_A) < 1e-2                            # H 1s <r> in A


@pytest.mark.parametrize("kw", [
    dict(embed_mode="both", species_descriptors=DESC),
    dict(embed_mode="descriptor", species_descriptors=DESC),
    dict(radial_shells=True, species_shell_radii=SHELLS),
    dict(embed_mode="both", species_descriptors=DESC, radial_shells=True, species_shell_radii=SHELLS,
         residual="identity", pair_term=True),
])
def test_e3_round11_options(x64, graph, kw):
    model, g = as_float64(make_model(n_layers=3, **kw), graph)
    _check_e3(model, g, TOL)


def test_forces_fd_round11(x64, graph):
    model, graph = as_float64(make_model(
        n_layers=3, embed_mode="both", species_descriptors=DESC,
        radial_shells=True, species_shell_radii=SHELLS), graph)
    forces = np.asarray(model(graph)["forces"])
    pos = np.asarray(graph.nodes["positions"], dtype=np.float64)

    def energy(p):
        g = graph._replace(nodes={**graph.nodes, "positions": p.astype(np.float64)})
        return float(np.asarray(model(g)["energy"])[0])

    cands = [(i, k) for i in range(pos.shape[0]) for k in range(3) if abs(forces[i, k]) > 0.05]
    rng = np.random.RandomState(3)
    picks = [cands[j] for j in rng.choice(len(cands), size=2, replace=False)]
    h = 1e-4
    for i, k in picks:
        pp, pm = pos.copy(), pos.copy(); pp[i, k] += h; pm[i, k] -= h
        fd = -(energy(pp) - energy(pm)) / (2 * h)
        assert abs(fd - forces[i, k]) / max(abs(forces[i, k]), 1e-6) < 1e-3, (fd, forces[i, k])


def test_options_change_energy_and_onehot_is_default(graph):
    base = make_model(n_layers=2, seed=7)
    same = make_model(n_layers=2, seed=7, embed_mode="onehot")
    assert _energy(base, graph) == _energy(same, graph)
    both = make_model(n_layers=2, seed=7, embed_mode="both", species_descriptors=DESC)
    shells = make_model(n_layers=2, seed=7, radial_shells=True, species_shell_radii=SHELLS)
    assert abs(_energy(base, graph) - _energy(both, graph)) > 1e-6
    assert abs(_energy(base, graph) - _energy(shells, graph)) > 1e-6


def test_round11_grads_finite_with_padding(graph):
    """Padded edges have r = 0 (envelopes = 1 there, masked by the cutoff); all gradients finite."""
    m = make_model(n_layers=2, embed_mode="both", species_descriptors=DESC,
                   radial_shells=True, species_shell_radii=SHELLS)
    graphdef, params = nnx.split(m, nnx.Param)
    g = pad_single(graph)

    def loss(p):
        l, _ = compute_loss(graphdef, p, g, 1.0, 10.0, 0.0, 10.0, periodic=False)
        return l

    val, grads = jax.value_and_grad(loss)(params)
    assert np.isfinite(float(val))
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(grads))
    out = nnx.merge(graphdef, params)(g)
    assert bool(jnp.all(jnp.isfinite(out["forces"])))


def test_shell_envelopes_vanish_at_cutoff(graph):
    """Moving an atom's only neighbour to the cutoff radius must give the same energy as
    removing the edge: the envelope channels carry the cutoff polynomial."""
    m = make_model(n_layers=2, radial_shells=True, species_shell_radii=SHELLS, seed=1)
    # two-atom graph: edge in both directions, separation exactly at the cutoff
    cutoff = float(m.cutoff)
    pos = np.array([[0.0, 0.0, 0.0], [cutoff - 1e-7, 0.0, 0.0]], np.float32)
    nodes = {"species": np.array([1, 0], np.int32), "positions": pos, "forces": np.zeros((2, 3), np.float32)}
    g_edge = graph._replace(n_node=np.array([2], np.int32), n_edge=np.array([2], np.int32), nodes=nodes,
                            senders=np.array([0, 1], np.int32), receivers=np.array([1, 0], np.int32),
                            globals={"energy": np.zeros((1,), np.float32)})
    g_none = g_edge._replace(n_edge=np.array([0], np.int32), senders=np.zeros((0,), np.int32),
                             receivers=np.zeros((0,), np.int32))
    assert abs(_energy(m, g_edge) - _energy(m, g_none)) < 1e-5
