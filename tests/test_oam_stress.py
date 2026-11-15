"""OAM-scale extensions (2026-11-05): UFF table for all OAM elements, stress = (1/V) dE/d(strain) for a
multi-element periodic cell with the final RACEv2 options, and the stress target stored in periodic graphs."""
import numpy as np
import jax
import pytest
import ase
from ase.calculators.singlepoint import SinglePointCalculator
from flax import nnx

from conftest import as_float64
from race_v2.models.race import RACE, UFF_VDW
from race_v2.data.xyz_graphs import preprocess_graph, atomic_numbers_to_indices
from race_v2.training.train import element_shell_radii, COVALENT_RADII

ZS = [3, 8, 26, 78]                     # Li, O, Fe, Pt: light, anion, 3d and 5d metal
OAM_Z = list(range(1, 84)) + list(range(89, 95))   # H..Bi and Ac..Pu (species_encoding of the OAM shards)


def test_uff_table_covers_oam_elements():
    assert all(z in UFF_VDW for z in OAM_Z)
    # spot values (kcal/mol -> eV) from Rappé et al. 1992 Table 1
    assert UFF_VDW[1][0] == 2.886 and abs(UFF_VDW[1][1] - 0.044 * 0.0433641) < 1e-12
    assert UFF_VDW[26][0] == 2.912 and UFF_VDW[78][0] == 2.754
    assert all(v[2] > 6.0 for v in UFF_VDW.values())          # exp-6 prefactor 6/(zeta-6) is finite


def _cell(seed=0):
    rng = np.random.RandomState(seed)
    L = 5.2
    frac = rng.uniform(0, 1, (10, 3))
    num = [3, 3, 8, 8, 8, 8, 26, 26, 78, 3]
    cell = np.eye(3) * L + rng.uniform(-0.3, 0.3, (3, 3))
    a = ase.Atoms(numbers=num, scaled_positions=frac, cell=cell, pbc=True)
    # push atoms apart so the configuration is not pathological
    from ase.geometry import get_distances
    for _ in range(200):
        _, d = get_distances(a.positions, cell=a.cell, pbc=True)
        np.fill_diagonal(d, 9.0)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if d[i, j] > 1.6:
            break
        a.positions[i] += rng.normal(0, 0.4, 3); a.wrap()
    a.calc = SinglePointCalculator(a, energy=-50.0, forces=np.zeros((len(a), 3)), stress=np.arange(6) * 1e-3)
    return a


def _model():
    return RACE(n_species=len(ZS), lmax=2, cutoff=5.0, hidden_irreps="16x0e + 16x1o + 16x2e", n_layers=2,
                radial_basis_size=8, radial_mlp_size=16, radial_mlp_layers=2, mlp_init_scale=4.0, shift=0.0,
                scale=1.0, avg_n_neighbors=30.0, atom_energies=[0.0] * len(ZS), l_train=True, periodic=True,
                x_mode="basis", x_norm="hill", residual="chebyshev", layer_scale=0.5, update="linear",
                readout_mode="gated", message_norm="avg", radial_prior={"type": "gaussian", "sigma": 0.224},
                radial_shells=True, species_shell_radii=element_shell_radii(ZS), pair_repulsion="uff",
                pair_repulsion_switch=1.25, pair_repulsion_switch_width=0.3, species_z=ZS,
                species_radii=[COVALENT_RADII[z] for z in ZS], rngs=nnx.Rngs(0))


def _graph(a):
    return preprocess_graph(a, atomic_numbers_to_indices(ZS), [0.0] * len(ZS), 5.0)


def _f64(g):
    return jax.tree.map(lambda x: np.asarray(x, np.float64) if np.issubdtype(np.asarray(x).dtype, np.floating) else x, g)


def test_periodic_graph_carries_stress_target():
    a = _cell()
    g = _graph(a)
    assert g.globals["stress"].shape == (1, 6)
    assert np.allclose(g.globals["stress"][0], np.arange(6) * 1e-3)


def test_stress_is_strain_derivative(x64):
    a = _cell(1)
    model, _ = as_float64(_model(), _graph(a))
    g = _f64(_graph(a))
    out = model(g)
    stress = np.asarray(out["stress"])[0]
    assert np.all(np.isfinite(stress)) and np.any(np.abs(stress) > 0)

    # finite differences: E(strained cell, strained positions) at FIXED neighbour list (Sij unchanged)
    cell0 = np.asarray(g.globals["cell"][0]); pos0 = np.asarray(g.nodes["positions"])
    vol = abs(np.linalg.det(cell0))
    def e_strain(eps):
        F = np.eye(3) + eps
        gg = g._replace(nodes={**g.nodes, "positions": pos0 @ F.T},
                        globals={**g.globals, "cell": (cell0 @ F.T)[None], "volume": np.array([vol * np.linalg.det(F)])})
        return float(np.asarray(model(gg)["energy"])[0])
    h = 1e-5
    voigt = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    fd = []
    for (u, v) in voigt:
        eps = np.zeros((3, 3)); eps[u, v] += h / 2; eps[v, u] += h / 2
        fd.append((e_strain(eps) - e_strain(-eps)) / (2 * h) / vol)
    fd = np.array(fd)
    assert np.allclose(stress, fd, rtol=1e-4, atol=1e-7), (stress, fd)


def test_fixed_shape_packing_covers_every_graph_once():
    """pad_mode "fixed": every batch has one shape and every graph is used exactly once per epoch."""
    from race_v2.data.loaders import BucketedDataLoader, fixed_collate_fn
    import functools
    rng = np.random.RandomState(3)
    graphs = []
    for k in range(60):
        n = int(rng.randint(1, 40)); e = int(rng.randint(0, 30 * n + 1))
        graphs.append(jraph_graph(n, e, tag=k))
    class _DS:
        pass
    ds = _DS(); ds.graphs = graphs
    pad_n, pad_e, max_g = 128, 2048, 16
    loader = BucketedDataLoader(ds, batch_size=max_g, n_buckets=4, shuffle=True, drop_last=False,
                                rngs=nnx.Rngs(0), max_nodes_per_batch=pad_n, max_edges_per_batch=pad_e)
    loader.collate_fn = functools.partial(fixed_collate_fn, n_node_pad=pad_n, n_edge_pad=pad_e)
    seen, shapes = [], set()
    for b in loader:
        shapes.add((b.nodes["positions"].shape[0], b.senders.shape[0], b.n_node.shape[0]))
        import jraph
        # the node-padding graph sits right after the real graphs, not at the end: use jraph's mask
        real = np.asarray(b.globals["tag"])[np.asarray(jraph.get_graph_padding_mask(b))]
        seen += [int(t) for t in real]
    assert shapes == {(pad_n, pad_e, max_g + 1)}
    assert sorted(seen) == list(range(60))


def jraph_graph(n, e, tag):
    import jraph
    return jraph.GraphsTuple(
        n_node=np.array([n], np.int32), n_edge=np.array([e], np.int32),
        nodes={"positions": np.zeros((n, 3), np.float32), "species": np.zeros(n, np.int32)},
        edges=None, senders=np.zeros(e, np.int32), receivers=np.zeros(e, np.int32),
        globals={"energy": np.zeros(1, np.float32), "tag": np.array([tag], np.int32)})


def test_many_padding_edges_leave_real_outputs_unchanged(x64):
    """pad_mode "fixed" puts thousands of r = 0 padding edges on one padding node: real energies, forces and stress
    must equal those of a lightly padded batch, and every graph energy must be finite (no nan from the padding).
    Compared in float64: in float32 (TF32 matmuls) the two batch shapes differ by ~5e-6 eV/A in the forces through
    the summation order alone (job 1222875)."""
    from race_v2.data.loaders import fixed_collate_fn
    gs = [_f64(_graph(_cell(s))) for s in (4, 5, 6)]
    n_n = sum(int(g.n_node[0]) for g in gs); n_e = sum(int(g.n_edge[0]) for g in gs)
    light = fixed_collate_fn(gs, 3, n_node_pad=n_n + 1, n_edge_pad=n_e)
    heavy = fixed_collate_fn(gs, 40, n_node_pad=n_n + 64, n_edge_pad=n_e + 30000)
    model, _ = as_float64(_model(), light)
    a, b = model(light), model(heavy)
    assert np.all(np.isfinite(np.asarray(b["energy"])))
    assert np.allclose(np.asarray(a["energy"])[:3], np.asarray(b["energy"])[:3], rtol=1e-10, atol=1e-10)
    assert np.allclose(np.asarray(a["forces"])[:n_n], np.asarray(b["forces"])[:n_n], rtol=1e-9, atol=1e-10)
    assert np.allclose(np.asarray(a["stress"])[:3], np.asarray(b["stress"])[:3], rtol=1e-9, atol=1e-12)
