"""Forces must equal -dE/dr (central finite differences).

Run in float64: in float32 the total energy (a sum of ~27 node energies with
cancellation) has a round-off of ~1e-5 eV, which makes central differences with
h ~ 1e-3 A meaningless (error ~1e-2 eV/A).  In float64 the agreement is ~1e-7.
"""
import numpy as np
import pytest

from conftest import make_model, as_float64


@pytest.mark.parametrize("x_mode", ["species", "node_scalar"])
def test_forces_match_finite_difference(x64, graph, x_mode):
    model, graph = as_float64(make_model(x_mode=x_mode), graph)
    out = model(graph)
    forces = np.asarray(out["forces"])
    assert forces.dtype == np.float64
    pos = np.asarray(graph.nodes["positions"], dtype=np.float64)

    def energy(p):
        g = graph._replace(nodes={**graph.nodes, "positions": p.astype(np.float64)})
        return float(np.asarray(model(g)["energy"])[0])

    rng = np.random.RandomState(0)
    # pick 2 atoms x 1 component where the force is not tiny
    cands = [(i, k) for i in range(pos.shape[0]) for k in range(3) if abs(forces[i, k]) > 0.05]
    assert len(cands) >= 2
    picks = [cands[j] for j in rng.choice(len(cands), size=2, replace=False)]
    h = 1e-4
    for i, k in picks:
        p_plus, p_minus = pos.copy(), pos.copy()
        p_plus[i, k] += h
        p_minus[i, k] -= h
        fd = -(energy(p_plus) - energy(p_minus)) / (2 * h)
        rel = abs(fd - forces[i, k]) / max(abs(forces[i, k]), 1e-6)
        assert rel < 1e-3, (i, k, fd, forces[i, k], rel)
