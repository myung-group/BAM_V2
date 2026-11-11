"""Thole phase 2 (bam_omat24/thole.py, stream_io.py) against BAM-torch: electrode / charge-group constraints,
contact charge relaxation, fragment unwrapping and the slab dipole correction.  Reference values:
tests/data/thole_torch_ref2.npz (tests/data/make_thole_torch_ref2.py, run with torch and BAM-torch)."""
import importlib.util
import os
import sys

import numpy as np
import pytest

import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("thole", os.path.join(_HERE, "..", "bam_omat24", "thole.py"))
TH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TH)
sys.path.insert(0, os.path.join(_HERE, "..", "bam_omat24"))
import stream_io as S  # noqa: E402

REF = np.load(os.path.join(_HERE, "data", "thole_torch_ref2.npz"))
F = 16


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def _groups():
    batch = REF["batch"]
    lg, fx = np.zeros(len(batch), np.int32), np.zeros(len(batch), np.float32)
    for g in np.unique(batch):
        m = batch == g
        lg[m], fx[m] = S.local_groups(REF["electrode_mask"][m], REF["charge_group"][m], REF["fixed_charge"][m], F)
    return lg, fx


def _edges(pos):
    recv, send = REF["edge_index"]
    d = pos[recv] - pos[send] - jnp.einsum("ei,eij->ej", jnp.asarray(REF["edge_shifts"], pos.dtype),
                                           jnp.asarray(REF["cell"])[REF["batch"][recv]])
    return recv, send, jnp.sqrt(jnp.sum(d * d, axis=-1))


def test_electrode_groups_match_torch():
    lg, fx = _groups()
    N, G = len(lg), len(REF["cell"])
    mask = jnp.ones(N, bool)
    group, targets = TH.constraint_targets(jnp.asarray(lg), jnp.asarray(fx, jnp.float64), jnp.asarray(REF["batch"]), mask,
                                           jnp.asarray(REF["total_charge"]), F)
    np.testing.assert_allclose(np.asarray(targets).sum(1), REF["total_charge"], atol=1e-6)
    q = TH.equilibrate(jnp.asarray(REF["raw"]), jnp.asarray(REF["fukui"]), group, G * F, targets.reshape(-1), mask)
    np.testing.assert_allclose(np.asarray(q), REF["q_group"], atol=1e-7)       # fixed charges are float32 in the meta-data


@pytest.mark.parametrize("pairs", ["charged", "all"])
def test_contact_charges_match_torch(pairs):
    lg, fx = _groups()
    N, G = len(lg), len(REF["cell"])
    mask = jnp.ones(N, bool)
    group, targets = TH.constraint_targets(jnp.asarray(lg), jnp.asarray(fx, jnp.float64), jnp.asarray(REF["batch"]), mask,
                                           jnp.asarray(REF["total_charge"]), F)

    def charges(pos):
        recv, send, dist = _edges(pos)
        return TH.contact_charges(jnp.asarray(REF["raw"]), jnp.asarray(REF["fukui"]), group, targets, mask,
                                  jnp.asarray(recv), jnp.asarray(send), dist, jnp.ones(len(recv), bool),
                                  contact_radius=2.5, cutoff=4.0, pairs=pairs)
    q = charges(jnp.asarray(REF["pos"]))
    np.testing.assert_allclose(np.asarray(q), REF[f"q_contact_{pairs}"], atol=2e-7)
    assert np.abs(REF[f"q_contact_{pairs}"] - REF["q_group"]).max() > 0.1          # contacts do move charge here
    # total charge of every graph is exact; position derivative agrees
    tot = np.asarray(jax.ops.segment_sum(q, jnp.asarray(REF["batch"]), num_segments=G))
    np.testing.assert_allclose(tot, REF["total_charge"], atol=1e-6)
    g = jax.grad(lambda p: jnp.sum(charges(p) * jnp.asarray(REF[f"w_{pairs}"])))(jnp.asarray(REF["pos"]))
    np.testing.assert_allclose(np.asarray(g), REF[f"grad_contact_{pairs}"], atol=2e-6)


def test_contact_charges_padding_and_float32():
    lg, fx = _groups()
    N, G = len(lg), len(REF["cell"])
    recv, send, dist = _edges(jnp.asarray(REF["pos"]))
    kw = dict(contact_radius=2.5, cutoff=4.0, pairs="charged")

    def run(dt, eps, pad=0):
        n = N + pad
        mask = jnp.asarray(np.arange(n) < N)
        ng = np.concatenate([REF["batch"], np.full(pad, G)])
        z = lambda x, v=0: np.concatenate([np.asarray(x), np.full(pad, v, np.asarray(x).dtype)])
        Q = np.concatenate([REF["total_charge"], np.zeros(1 if pad else 0)])
        group, targets = TH.constraint_targets(jnp.asarray(z(lg)), jnp.asarray(z(fx), dt), jnp.asarray(ng), mask,
                                               jnp.asarray(Q, dt), F)
        r = np.concatenate([recv, np.full(3 * (pad > 0), N)]); s = np.concatenate([send, np.full(3 * (pad > 0), N)])
        dd = jnp.concatenate([dist, jnp.zeros(3 * (pad > 0))]).astype(dt)
        em = jnp.asarray(np.arange(len(r)) < len(recv))
        return np.asarray(TH.contact_charges(jnp.asarray(z(REF["raw"], 0.7), dt), jnp.asarray(z(REF["fukui"], 2.0), dt),
                                             group, targets, mask, jnp.asarray(r), jnp.asarray(s), dd, em, eps=eps, **kw))
    a = run(jnp.float64, 1e-6)
    b = run(jnp.float64, 1e-6, pad=5)
    np.testing.assert_allclose(b[:N], a, atol=1e-12)
    assert np.all(b[N:] == 0.0)
    c64, c32 = run(jnp.float64, 1e-3), run(jnp.float32, 1e-3)
    assert np.abs(c64 - a).max() < 1e-2                 # eps 1e-3 changes the contact limit by a few 1e-3 e
    np.testing.assert_allclose(c32, c64, atol=2e-4)


def test_unwrap_offsets_match_torch():
    recv, send = REF["edge_index"]
    off, perc = S.unwrap_offsets(recv, send, REF["edge_shifts"], len(REF["pos"]))
    assert np.array_equal(off, REF["offsets"]) and np.array_equal(perc, REF["percolating"])
    assert (off != 0).any()


def _slab_run(name, polar_kw, pos=None, cell=None):
    polar = dict(TH.POLAR_DEFAULTS)
    polar.update(cg_max_iterations=300, cg_tolerance=1e-13, slab_correction=True)
    polar.update(polar_kw)
    es = TH.ewald_settings(polar, float(REF["cutoff"]))
    L = np.linalg.norm(REF["cell"][REF["pbc"]], axis=-1).max(0)
    V = np.abs(np.linalg.det(REF["cell"][REF["pbc"]])).max()
    grid, nk = TH.k_budget(es["k_max"], L, V)
    pos = jnp.asarray(REF["pos"]) if pos is None else pos
    cell = jnp.asarray(REF["cell"]) if cell is None else cell
    batch = jnp.asarray(REF["batch"])
    recv, send = REF["edge_index"]
    pos_field = pos + jnp.einsum("ni,nij->nj", jnp.asarray(REF["offsets"], pos.dtype), cell[batch])
    return TH.thole_energy(pos, cell, batch, jnp.ones(len(REF["pos"]), bool), jnp.asarray(REF["pbc"]), jnp.asarray(send),
                           jnp.asarray(recv), jnp.asarray(-REF["edge_shifts"].astype(float)), jnp.ones(len(recv), bool),
                           jnp.asarray(REF["q"]), jnp.asarray(REF["mu"]), jnp.asarray(REF["theta"]), jnp.asarray(REF["alpha"]),
                           cutoff=float(REF["cutoff"]), polar=polar, grid_m=TH.half_grid(grid), n_k=nk, open_dense=True,
                           slab_on=jnp.asarray(REF["slab_on"]), pos_field=pos_field)


@pytest.mark.parametrize("name,kw", [("slab", {}), ("slab_nomutual", {"mutual_induction": False}),
                                     ("slab_noind", {"induction": False}), ("slab_axis0", {"slab_axis": 0})])
def test_slab_correction_matches_torch(name, kw):
    out = _slab_run(name, kw)
    G = len(REF["cell"])
    seg = lambda x: np.asarray(jax.ops.segment_sum(x, jnp.asarray(REF["batch"]), num_segments=G))
    np.testing.assert_allclose(seg(out["node_energy"]), REF[f"{name}:coulomb_energy"], atol=5e-8)
    np.testing.assert_allclose(np.asarray(out["node_energy"]), REF[f"{name}:node_coulomb"], atol=5e-8)
    np.testing.assert_allclose(seg(out["node_slab"]), REF[f"{name}:slab_energy"], atol=5e-8)
    np.testing.assert_allclose(np.asarray(out["induced_dipoles"]), REF[f"{name}:induced_dipoles"], atol=5e-8)
    np.testing.assert_allclose(np.asarray(out["total_dipole"]), REF[f"{name}:total_dipole"], atol=1e-7)
    np.testing.assert_allclose(np.asarray(out["normal_dipole"]), REF[f"{name}:normal_dipole"], atol=1e-7)
    assert REF[f"{name}:slab_energy"][REF["slab_on"]].min() > 1e-3
    gp, gc = jax.grad(lambda p, c: jnp.sum(_slab_run(name, kw, p, c)["node_energy"]), argnums=(0, 1))(
        jnp.asarray(REF["pos"]), jnp.asarray(REF["cell"]))
    np.testing.assert_allclose(-np.asarray(gp), REF[f"{name}:forces"], atol=1e-7)
    np.testing.assert_allclose(np.asarray(gc)[REF["pbc"]], REF[f"{name}:cellgrad"][REF["pbc"]], atol=1e-7)
