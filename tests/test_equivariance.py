"""Energy invariance, force equivariance and (NLL) loss invariance under E(3)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from conftest import make_model, pad_single, random_rotation, as_float64
from bam_omat24.training_utils import compute_loss, _traceless_symmetric_from_2e
import e3nn_jax as e3nn

# Measured on a 3BPA frame (2 layers, 16 channels): float64 -> dE ~ 1e-9, dF ~ 2e-8;
# float32 -> dE ~ 3e-4 (|E|~14), dF ~ 2e-4 (|F|~0.7).  The float32 error is round-off
# (it vanishes in float64), so the strict check runs in float64 and float32 only
# guards against gross breakage.
TOL = 1e-6        # float64
TOL32 = 2e-3      # float32, relative to max |F| / |E|


def _transform(graph, R=None, t=None, invert=False):
    pos = np.asarray(graph.nodes["positions"])
    frc = np.asarray(graph.nodes["forces"])
    dt = pos.dtype
    M = np.eye(3, dtype=dt)
    if R is not None:
        M = R.astype(dt) @ M
    if invert:
        M = -M
    pos2 = pos @ M.T
    frc2 = frc @ M.T
    if t is not None:
        pos2 = pos2 + np.asarray(t, dt)
    return graph._replace(nodes={**graph.nodes, "positions": pos2.astype(dt),
                                 "forces": frc2.astype(dt)}), M


def _run(model, g):
    out = model(g)
    return np.asarray(out["energy"]), np.asarray(out["forces"])


def _check_e3(model, graph, tol):
    e0, f0 = _run(model, graph)
    R = random_rotation(1)
    t = np.array([1.3, -2.1, 0.7])
    for kw in (dict(R=R), dict(t=t), dict(invert=True), dict(R=R, t=t, invert=True)):
        g2, M = _transform(graph, **kw)
        e2, f2 = _run(model, g2)
        escale = max(1.0, float(np.abs(e0).max()))
        fscale = max(1.0, float(np.abs(f0).max()))
        assert np.abs(e0 - e2).max() <= tol * escale, (kw.keys(), e0, e2)
        assert np.abs(f0 @ M.T - f2).max() <= tol * fscale, (kw.keys(), np.abs(f0 @ M.T - f2).max())


@pytest.mark.parametrize("x_mode", ["species", "node_scalar"])
@pytest.mark.parametrize("cov_mode", ["scalar6", "irreps"])
def test_energy_forces_e3_float64(x64, graph, x_mode, cov_mode):
    model, g = as_float64(make_model(x_mode=x_mode, cov_mode=cov_mode), graph)
    assert np.asarray(model(g)["energy"]).dtype == np.float64
    _check_e3(model, g, TOL)


@pytest.mark.parametrize("x_mode", ["species", "node_scalar"])
def test_energy_forces_e3_float32(graph, x_mode):
    _check_e3(make_model(x_mode=x_mode), graph, TOL32)


def test_2e_to_tensor_is_equivariant():
    """B(D(R) b) == R B(b) R^T  for the e3nn-jax 2e convention used in training_utils."""
    rng = np.random.RandomState(0)
    b = rng.normal(size=(7, 5)).astype(np.float32)
    R = random_rotation(3)
    b_rot = e3nn.IrrepsArray("1x2e", jnp.asarray(b)).transform_by_matrix(jnp.asarray(R)).array
    B = np.asarray(_traceless_symmetric_from_2e(jnp.asarray(b)))
    B_rot = np.asarray(_traceless_symmetric_from_2e(jnp.asarray(b_rot)))
    assert np.allclose(B_rot, R @ B @ R.T, atol=1e-5)
    assert np.allclose(np.trace(B, axis1=1, axis2=2), 0.0, atol=1e-6)


def _nll(model, g, loss_type):
    graphdef, params = nnx.split(model, nnx.Param)
    loss, aux = compute_loss(graphdef, params, pad_single(g), 1.0, 1.0, 0.0, 0.1,
                             loss_type=loss_type, periodic=False, cov_mode=model.cov_mode)
    return float(loss), {k: float(v) for k, v in aux.items()}


@pytest.mark.parametrize("loss_type", ["nll", "gauss_nll"])
def test_nll_loss_invariant_irreps(x64, graph, loss_type):
    """cov_mode='irreps': the force NLL is invariant under rotation + inversion."""
    model, graph = as_float64(make_model(cov_mode="irreps"), graph)
    l0, a0 = _nll(model, graph, loss_type)
    # the anisotropic part must be non-trivial, otherwise the test is vacuous
    fv = np.asarray(model(graph)["force_variation"])
    assert np.abs(fv[:, 1:]).max() > 1e-3
    for kw in (dict(R=random_rotation(5)), dict(invert=True),
               dict(R=random_rotation(6), t=np.array([0.5, 0.5, -1.0], np.float32))):
        g2, _ = _transform(graph, **kw)
        l2, a2 = _nll(model, g2, loss_type)
        assert abs(l0 - l2) <= TOL * max(1.0, abs(l0)), (kw.keys(), l0, l2)
        assert abs(a0["force_loss"] - a2["force_loss"]) <= TOL * max(1.0, abs(a0["force_loss"]))


@pytest.mark.xfail(strict=True, reason="cov_mode='scalar6' fixes the covariance to the lab frame "
                                       "(known issue #5); the NLL is NOT rotation invariant")
@pytest.mark.parametrize("loss_type", ["nll", "gauss_nll"])
def test_nll_loss_invariant_scalar6(x64, graph, loss_type):
    model, graph = as_float64(make_model(cov_mode="scalar6"), graph)
    l0, a0 = _nll(model, graph, loss_type)
    g2, _ = _transform(graph, R=random_rotation(5))
    l2, a2 = _nll(model, g2, loss_type)
    assert abs(a0["force_loss"] - a2["force_loss"]) <= TOL * max(1.0, abs(a0["force_loss"]))


@pytest.mark.parametrize("x_mode", ["species", "node_scalar"])
def test_energy_forces_e3_float64_gated_readout(x64, graph, x_mode):
    model, g = as_float64(make_model(x_mode=x_mode, readout_mode="gated"), graph)
    _check_e3(model, g, TOL)


FULL_PARITY = "16x0e+16x0o+16x1o+16x1e+16x2e+16x2o"


@pytest.mark.parametrize("x_mode", ["species", "node_irreps"])
def test_energy_forces_e3_float64_node_irreps_full_parity(x64, graph, x_mode):
    """Equivariant x product (x_mode node_irreps) and full-parity hidden irreps
    (0o/1e/2o channels, tanh odd activation) must stay E(3)-equivariant."""
    model, g = as_float64(make_model(x_mode=x_mode, hidden_irreps=FULL_PARITY), graph)
    _check_e3(model, g, TOL)


def test_energy_forces_e3_float64_node_irreps(x64, graph):
    model, g = as_float64(make_model(x_mode="node_irreps"), graph)
    _check_e3(model, g, TOL)


def test_energy_forces_e3_float64_readout_species(x64, graph):
    model, g = as_float64(make_model(readout_species=True), graph)
    _check_e3(model, g, TOL)


def test_node_irreps_product_is_not_trivial(graph):
    """The equivariant x product must actually use the l>0 part of x_i: zeroing the
    l>0 outputs of the layer-1 x projection must change the energy."""
    model = make_model(x_mode="node_irreps")
    e0 = float(np.asarray(model(graph)["energy"])[0])
    proj = model.x_projs[1]
    changed = 0
    for orig, attr in proj._weight_name_mapping.items():
        out_ir = orig.split()[-1].split(",")[-1]  # "w[i,j] 16x1o,16x1o" -> "16x1o"
        if "0e" not in out_ir:
            getattr(proj, attr).value = jnp.zeros_like(getattr(proj, attr).value)
            changed += 1
    assert changed > 0
    e1 = float(np.asarray(model(graph)["energy"])[0])
    assert abs(e0 - e1) > 1e-6, (e0, e1)


@pytest.mark.parametrize("residual", ["identity", "identity_cum"])
def test_energy_forces_e3_float64_residual(x64, graph, residual):
    model, g = as_float64(make_model(residual=residual, n_layers=3), graph)
    _check_e3(model, g, TOL)


def test_residual_changes_features_only_from_layer_1(graph):
    """residual='identity' must give the same layer-0 stratum energy as the original
    model (same parameters) and a different total energy (layers 1+ differ)."""
    m0 = make_model(n_layers=3, seed=3)
    m1 = make_model(n_layers=3, seed=3, residual="identity")
    e0 = np.asarray(m0(graph)["energy"])[0]
    e1 = np.asarray(m1(graph)["energy"])[0]
    assert abs(e0 - e1) > 1e-6, (e0, e1)
