"""Quadrupole labels / outputs and atomic spin populations (2026-11-13): stream labels from the ORCA multipoles and
spin populations, the traceless graph quadrupole of RACE_POLAR (rotation covariance, translation invariance of a
neutral graph, agreement with the per-atom definition for charges only), the spin head (sums to 2S), and the new loss
terms.  Run in a Slurm job (JAX)."""
import numpy as np

from conftest import ATOMIC_NUMBERS
from bam_omat24.train_3bpa import COVALENT_RADII  # noqa: F401  (imports train_3bpa before the x64 fixture)
from test_race_polar_qeq import _polar_model, _molecule, _batch, IDX, CUTOFF, FIELDS  # noqa: E402

BOHR = 0.529177210903


def test_stream_quadrupole_and_spin_labels():
    from bam_omat24 import stream_io as S
    st = _molecule(0, charge=0.0, n=6)
    st['info'] = {'charge': 0, 'spin': 3, 'unrestricted': True,
                  'multipoles': [[0.0, 0.0, 0.0], [0.1, -0.2, 0.3], [-30.0, -31.0, -29.0, 0.5, -0.25, 0.75]],
                  'lowdin_spins': [1.0, 0.5, 0.25, 0.25, 0.0, 0.0]}
    d = S.graph_arrays(st, IDX, np.zeros(4), CUTOFF, dict(FIELDS, quadrupole=True, spin_labels={'key': 'lowdin_spins'}))
    q = d['g:quad'][0].astype(np.float64)
    tr = (-30.0 - 31.0 - 29.0) / 3.0
    ref = np.array([-30.0 - tr, -31.0 - tr, -29.0 - tr, 0.5, -0.25, 0.75]) * BOHR ** 2
    assert np.allclose(q, ref, atol=1e-5) and abs(q[:3].sum()) < 1e-5 and d['g:quad_w'][0] == 1.0
    assert np.allclose(d['n:ref_spins'], [1.0, 0.5, 0.25, 0.25, 0.0, 0.0]) and np.all(d['n:ref_spins_w'] == 1.0)
    # restricted closed shell without spin labels: zero spin density, labelled; open shell without labels: weight 0
    st['info'] = {'charge': 0, 'spin': 1, 'unrestricted': False}
    d = S.graph_arrays(st, IDX, np.zeros(4), CUTOFF, dict(FIELDS, quadrupole=True, spin_labels={'key': 'lowdin_spins'}))
    assert np.all(d['n:ref_spins'] == 0) and np.all(d['n:ref_spins_w'] == 1) and d['g:quad_w'][0] == 0.0
    st['info'] = {'charge': 0, 'spin': 2, 'unrestricted': True}
    d = S.graph_arrays(st, IDX, np.zeros(4), CUTOFF, dict(FIELDS, spin_labels={'key': 'lowdin_spins'}))
    assert np.all(d['n:ref_spins_w'] == 0)


def _q6_to_3(v):
    xx, yy, zz, xy, xz, yz = v
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def test_graph_quadrupole_rotation_translation(x64):
    import jax
    import jax.numpy as jnp
    m = _polar_model(1)
    sts = [_molecule(3, charge=0.0), _molecule(4, charge=1.0)]
    b = _batch(sts)
    out = m(jax.tree.map(jnp.asarray, b))
    Q0 = np.asarray(out['quadrupole'])
    assert Q0.shape[1] == 6 and np.all(np.abs(Q0[:2, :3].sum(axis=1)) < 1e-9)       # traceless
    assert np.abs(Q0[:2]).max() > 1e-4
    # rotation: Q -> R Q R^T (graph by graph)
    u, _, vt = np.linalg.svd(np.random.default_rng(5).normal(size=(3, 3)))
    R = u @ vt
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    rot = [dict(st, positions=st['positions'] @ R.T) for st in sts]
    Q1 = np.asarray(m(jax.tree.map(jnp.asarray, _batch(rot)))['quadrupole'])
    for g in range(2):
        assert np.allclose(_q6_to_3(Q1[g]), R @ _q6_to_3(Q0[g]) @ R.T, atol=1e-7)
    # translation: unchanged (about the centre of mass), also for the charged graph
    tr = [dict(st, positions=st['positions'] + np.array([3.0, -2.0, 7.5])) for st in sts]
    Q2 = np.asarray(m(jax.tree.map(jnp.asarray, _batch(tr)))['quadrupole'])
    assert np.allclose(Q2[:2], Q0[:2], atol=1e-7)


def test_spin_head_sums_to_2S(x64):
    import jax
    import jax.numpy as jnp
    m = _polar_model(2, lr={'spin_head': True})
    m.sp_head2.kernel.value = jnp.asarray(np.random.default_rng(0).normal(0, 0.3, m.sp_head2.kernel.value.shape))
    sts = [_molecule(6), _molecule(7)]
    sts[0]['info']['spin'], sts[1]['info']['spin'] = 3, 1
    b = _batch(sts)
    out = m(jax.tree.map(jnp.asarray, b))
    s = np.asarray(out['spins'])
    n0, n1 = int(b.n_node[0]), int(b.n_node[1])
    assert abs(s[:n0].sum() - 2.0) < 1e-10 and abs(s[n0:n0 + n1].sum()) < 1e-10 and np.all(s[n0 + n1:] == 0)


def test_quadrupole_and_spin_loss_terms(x64):
    import jax
    import jax.numpy as jnp
    from bam_omat24.training_utils import compute_loss
    m = _polar_model(3, lr={'spin_head': True})
    sts = [_molecule(8), _molecule(9, charge=-1.0)]
    for st in sts:
        st['info'].update(multipoles=[[0, 0, 0], [0, 0, 0], [-30.0, -31.0, -29.0, 0.4, 0.0, -0.2]], unrestricted=False)
    gs = []
    from bam_omat24 import stream_io as S
    from bam_omat24.stream_data import _to_graph, _cast
    import jraph
    for st in sts:
        d = _cast(S.graph_arrays(st, IDX, np.zeros(4), CUTOFF, dict(FIELDS, quadrupole=True,
                                                                     spin_labels={'key': 'lowdin_spins'})), np.float64)
        gs.append(_to_graph(d))
    bb = jraph.batch_np(gs)
    b = jraph.pad_with_graphs(bb, int(bb.n_node.sum()) + 3, int(bb.n_edge.sum()) + 11, 3)
    b = jax.tree.map(jnp.asarray, b)
    from flax import nnx
    gd, params = nnx.split(m, nnx.Param)
    kw = dict(energy_weight=0.0, force_weight=0.0, stress_weight=0.0, huber_delta=0.02, periodic=False)
    _, a0 = compute_loss(gd, params, b, **kw)
    _, a1 = compute_loss(gd, params, b, quadrupole_weight=1.0, spin_weight=1.0, **kw)
    res = nnx.merge(gd, params)(b)
    # the documented formulas
    qw = np.asarray(b.globals['quad_w']).reshape(-1) * np.asarray(jraph.get_graph_padding_mask(b))
    dq = np.asarray(res['quadrupole']) - np.asarray(b.globals['quad']).reshape(-1, 6)
    qd = np.sum(qw[:, None] * np.array([1, 1, 1, 2, 2, 2.0]) * dq ** 2) / max(9 * qw.sum(), 1.0)
    nw = np.asarray(jraph.get_node_padding_mask(b), float) * np.asarray(b.nodes['ref_spins_w'])
    sp = np.sum(nw * (np.asarray(res['spins']) - np.asarray(b.nodes['ref_spins'])) ** 2) / max(nw.sum(), 1.0)
    assert qw[0] == 1.0 and qw[1] == 1.0 and qw[2] == 0.0
    assert abs(float(a1['qd_loss']) - qd) < 1e-9 and abs(float(a1['sp_loss']) - sp) < 1e-9 and qd > 0
    assert abs(float(a1['aux_loss']) - float(a0['aux_loss']) - qd - sp) < 1e-9
