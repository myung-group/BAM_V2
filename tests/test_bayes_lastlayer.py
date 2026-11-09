"""Last-layer Bayesian regression on periodic cells (2026-11-09): the design rows reproduce the network exactly.
Energy, forces and stress of a small periodic RACE_LR (Ewald + Qeq) are rest + phi.theta, rest + J theta and
rest + S theta with one theta (fitted on the energies of rattled cells), and the stress of the per-structure function
equals the model's own stress.  Run in a Slurm job or on a CPU (JAX)."""
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

from conftest import as_float64
from test_oam_stress import _cell, _graph
from test_race_lr import _periodic_lr_model


def test_periodic_design_rows_are_exact(x64):
    from bam_omat24 import bayes_lastlayer as B
    base = _cell(1)
    model, _ = as_float64(_periodic_lr_model({"ewald_alpha": 0.5, "ewald_n_real": 2, "ewald_n_recip": 6}), _graph(base))
    model0, n_ro = B.zero_readout(model)
    assert n_ro > 0
    full, _, efs = B.make_fns(model, model0, periodic=True, jac_chunk=8)
    rng = np.random.default_rng(0)
    rows = []
    for k in range(60):
        a = base.copy()
        a.positions += rng.normal(0, 0.08, a.positions.shape)
        a.set_cell(a.cell.array @ (np.eye(3) + rng.normal(0, 0.02, (3, 3))), scale_atoms=True)
        a.calc = SinglePointCalculator(a, energy=0.0, forces=np.zeros((len(a), 3)), stress=np.zeros(6))
        g = _graph(a)
        g = g._replace(globals={**g.globals, 'charge': np.array([0.0]), 'spin': np.array([1.0])})
        b = B.collate(g, 16, 1024)
        o = {kk: np.asarray(v) for kk, v in full(b).items()}
        if k == 0:
            ref = model(b)
            assert np.allclose(o['s'], np.asarray(ref['stress'])[0], rtol=1e-8, atol=1e-12)
            assert np.allclose(o['f'][:len(a)], np.asarray(ref['forces'])[:len(a)], rtol=1e-8, atol=1e-10)
        rows.append((o, len(a)))
    P = rows[0][0]['phi'].shape[0]
    assert len(rows) > P
    phi = np.stack([o['phi'] for o, _ in rows])
    th = np.linalg.lstsq(phi, np.array([o['e'] - o['e0'] for o, _ in rows]), rcond=None)[0]
    assert np.abs(phi @ th - np.array([o['e'] - o['e0'] for o, _ in rows])).max() < 1e-10
    for o, n in rows[:10]:
        J = np.transpose(o['J'][:, :n, :], (1, 2, 0)).reshape(3 * n, -1)
        assert np.abs(J @ th - (o['f'][:n] - o['f0'][:n]).reshape(-1)).max() < 1e-8
        assert np.abs(o['S'].T @ th - (o['s'] - o['s0'])).max() < 1e-10
        assert np.abs(o['s'] - o['s0']).max() > 1e-6            # the readout does contribute to the stress
