"""Thole electrostatics (bam_omat24/thole.py) against the BAM-torch block (mode pbc_split).

tests/data/thole_torch_ref.npz holds inputs and outputs of bam_ufo.model.thole.TholeElectrostaticsBlock for a batch
of two periodic cells (one charged) and two molecules (one charged); made by tests/data/make_thole_torch_ref.py in
an environment with torch and BAM-torch.  The JAX functions are imported without the bam_omat24 package (no model).
"""
import importlib.util
import os

import numpy as np
import pytest

import jax
import jax.numpy as jnp


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("thole", os.path.join(_HERE, "..", "bam_omat24", "thole.py"))
TH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TH)
REF = np.load(os.path.join(_HERE, "data", "thole_torch_ref.npz"))

CASES = {"s10": {}, "s05": {"sigma": 0.5}, "s10_noquad": {"quadrupoles": False},
         "s10_nomutual": {"mutual_induction": False}, "s10_noind": {"induction": False},
         "s10_nocap_ktol": {"cap_induced_width": False, "ewald_k_tolerance": 1e-5}}


def _polar(**kw):
    p = dict(TH.POLAR_DEFAULTS)
    p.update(cg_max_iterations=200, cg_tolerance=1e-12)
    p.update(kw)
    return p


def _inputs(pad_nodes=0, pad_edges=0, pad_graphs=0):
    pos, cell, batch, pbc = REF["pos"], REF["cell"], REF["batch"], REF["pbc"]
    recv, send = REF["edge_index"]
    sij = -REF["edge_shifts"].astype(float)
    N, G, E = len(pos), len(cell), len(recv)
    d = dict(pos=pos, cell=cell, node_graph=batch, node_mask=np.ones(N, bool), pbc=pbc, send=send, recv=recv,
             sij=sij, edge_mask=np.ones(E, bool), q=REF["q"], mu=REF["mu"], theta=REF["theta"], alpha=REF["alpha"])
    if pad_nodes:        # jraph-style padding: extra nodes / edges / one padding graph (zero cell)
        d["pos"] = np.concatenate([pos, np.zeros((pad_nodes, 3))])
        d["node_graph"] = np.concatenate([batch, np.full(pad_nodes, G)])
        d["node_mask"] = np.concatenate([d["node_mask"], np.zeros(pad_nodes, bool)])
        for k, v in (("q", 0.3), ("alpha", 1.0)):
            d[k] = np.concatenate([d[k], np.full(pad_nodes, v)])
        d["mu"] = np.concatenate([d["mu"], np.full((pad_nodes, 3), 0.2)])
        d["theta"] = np.concatenate([d["theta"], np.zeros((pad_nodes, 3, 3))])
        d["cell"] = np.concatenate([cell, np.zeros((pad_graphs, 3, 3))])
        d["pbc"] = np.concatenate([pbc, np.zeros(pad_graphs, bool)])
        d["send"] = np.concatenate([send, np.full(pad_edges, N)])
        d["recv"] = np.concatenate([recv, np.full(pad_edges, N)])
        d["sij"] = np.concatenate([sij, np.zeros((pad_edges, 3))])
        d["edge_mask"] = np.concatenate([d["edge_mask"], np.zeros(pad_edges, bool)])
    return d


def _run(d, polar, pos=None, cell=None, n_k=None, grid=None):
    es = TH.ewald_settings(polar, float(REF["cutoff"]))
    L = np.linalg.norm(REF["cell"][REF["pbc"]], axis=-1).max(0)
    V = np.abs(np.linalg.det(REF["cell"][REF["pbc"]])).max()
    g, nk = TH.k_budget(es["k_max"], L, V)
    grid_m = TH.half_grid(grid or g)
    return TH.thole_energy(
        jnp.asarray(d["pos"] if pos is None else pos), jnp.asarray(d["cell"] if cell is None else cell),
        jnp.asarray(d["node_graph"]), jnp.asarray(d["node_mask"]), jnp.asarray(d["pbc"]), jnp.asarray(d["send"]),
        jnp.asarray(d["recv"]), jnp.asarray(d["sij"]), jnp.asarray(d["edge_mask"]), jnp.asarray(d["q"]),
        jnp.asarray(d["mu"]), jnp.asarray(d["theta"]) if polar["quadrupoles"] else None, jnp.asarray(d["alpha"]),
        cutoff=float(REF["cutoff"]), polar=polar, grid_m=grid_m, n_k=n_k or nk, open_dense=True)


@pytest.mark.parametrize("name", list(CASES))
def test_matches_torch(name):
    polar = _polar(**CASES[name])
    d = _inputs()
    G = len(d["cell"])
    out = _run(d, polar)
    assert not bool(np.any(out["overflow"]))
    e_graph = np.asarray(jax.ops.segment_sum(out["node_energy"], jnp.asarray(d["node_graph"]), num_segments=G))
    np.testing.assert_allclose(e_graph, REF[f"{name}:coulomb_energy"], rtol=0, atol=2e-8)
    np.testing.assert_allclose(np.asarray(out["node_energy"]), REF[f"{name}:node_coulomb"], rtol=0, atol=2e-8)
    np.testing.assert_allclose(np.asarray(out["induced_dipoles"]), REF[f"{name}:induced_dipoles"], rtol=0, atol=2e-8)
    np.testing.assert_allclose(np.asarray(out["total_dipole"]), REF[f"{name}:total_dipole"], rtol=0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(out["potential"]), REF[f"{name}:electrostatic_potential"], rtol=0, atol=2e-8)

    def total(pos, cell):
        return jnp.sum(_run(d, polar, pos, cell)["node_energy"])
    gp, gc = jax.grad(total, argnums=(0, 1))(jnp.asarray(d["pos"]), jnp.asarray(d["cell"]))
    np.testing.assert_allclose(-np.asarray(gp), REF[f"{name}:forces"], rtol=0, atol=5e-8)
    # the placeholder cells of the molecules never enter; the periodic cell gradients (stress) agree
    np.testing.assert_allclose(np.asarray(gc)[d["pbc"]], REF[f"{name}:cellgrad"][d["pbc"]], rtol=0, atol=5e-8)
    assert np.abs(np.asarray(gc)[~d["pbc"]]).max() == 0.0


def test_padding_invariant():
    polar = _polar()
    a = _run(_inputs(), polar)
    d = _inputs(pad_nodes=11, pad_edges=37, pad_graphs=1)
    b = _run(d, polar)
    n = len(REF["pos"])
    np.testing.assert_allclose(np.asarray(b["node_energy"])[:n], np.asarray(a["node_energy"]), rtol=0, atol=1e-10)
    assert np.all(np.asarray(b["node_energy"])[n:] == 0.0)
    assert np.all(np.isfinite(np.asarray(b["induced_dipoles"])))
    g = jax.grad(lambda p: jnp.sum(_run(d, polar, pos=p)["node_energy"]))(jnp.asarray(d["pos"]))
    assert np.all(np.isfinite(np.asarray(g))) and np.abs(np.asarray(g)[n:]).max() == 0.0


def test_k_overflow_is_nan_and_extra_budget_is_harmless():
    polar = _polar()
    d = _inputs()
    a = _run(d, polar)
    b = _run(d, polar, n_k=4000, grid=(12, 12, 14))        # larger grid and budget: same vectors selected
    np.testing.assert_allclose(np.asarray(b["node_energy"]), np.asarray(a["node_energy"]), rtol=0, atol=1e-10)
    c = _run(d, polar, n_k=50)
    e = np.asarray(c["node_energy"])
    assert np.all(np.isnan(e[REF["pbc"][REF["batch"]]])) and np.all(np.isfinite(e[~REF["pbc"][REF["batch"]]]))


def test_float32_close_to_float64():
    polar = _polar(cg_tolerance=1e-6, cg_max_iterations=50)
    d = _inputs()
    a = _run(d, polar)
    d32 = {k: (v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in d.items()}
    b = _run(d32, polar)
    assert b["node_energy"].dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(b["node_energy"]), np.asarray(a["node_energy"]), rtol=0, atol=2e-4)


def test_equilibrate_sums_to_targets():
    rng = np.random.default_rng(0)
    group = np.array([0] * 5 + [1] * 7 + [2] * 3)
    mask = np.array([True] * 12 + [False] * 3)
    q = TH.equilibrate(jnp.asarray(rng.normal(size=15)), jnp.asarray(rng.normal(size=15) * 3), jnp.asarray(group), 3,
                       jnp.asarray([1.0, -2.0, 0.0]), jnp.asarray(mask))
    s = np.asarray(jax.ops.segment_sum(q, jnp.asarray(group), num_segments=3))
    np.testing.assert_allclose(s, [1.0, -2.0, 0.0], atol=1e-12)
    assert np.all(np.asarray(q)[12:] == 0.0)
