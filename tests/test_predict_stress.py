"""Stress of the inference API (race_v2.inference.predict, 2026-11-24): Predictor.predict(stress=True) and the ASE
calculator return the model's stress (Voigt xx yy zz yz xz xy, eV/A^3, ASE sign), frame by frame; it is the strain
derivative of the predicted energy; no stress for a frame without full periodicity or a model built without cells."""
import io

import numpy as np
import pytest
import ase
from ase.calculators.calculator import PropertyNotImplementedError

from test_periodic import _water_box

ZS = [1, 6, 7, 8]
E0 = [-0.5, -1.0, -1.5, -2.0]
CFG = {"model": "RACE", "atomic_numbers": ZS, "atom_energies": E0, "cutoff": 4.0, "hidden_size": 8, "lmax": 2,
       "n_layers": 2, "radial_mlp_size": 8, "radial_mlp_layers": 1, "avg_n_neighbors": 12.0, "tp_backend": "e3nn",
       "update": "linear", "readout_mode": "gated", "message_norm": "avg", "x_mode": "basis", "x64": True,
       "periodic": True, "node_bucket_size": 16, "edge_bucket_size": 256}


def _checkpoint(tmp_path, **kw):
    from race_v2.training import train as T
    from race_v2.training.utils import save_checkpoint
    cfg = {**CFG, **kw}
    stats = {'shift_raw': 0.0, 'scale': 1.0, 'avg_n_neighbors': 12.0, 'n_graphs': 1, 'n_atoms': 1}
    graphdef, params, *_ = T.build_model({**cfg, 'learning_rate': 1e-3}, len(ZS), E0, stats, io.StringIO())
    ck = tmp_path / f"ckpt_{'p' if cfg['periodic'] else 'm'}.pkl"
    save_checkpoint({'config': cfg, 'ema_params': params}, str(ck))
    return ck


def _sheared(a, seed):
    r = np.random.RandomState(seed)
    b = a.copy(); b.calc = None
    b.set_cell(b.cell.array @ (np.eye(3) + 0.03 * r.randn(3, 3)), scale_atoms=True)
    return b


def test_predict_stress_and_ase_calculator(tmp_path, x64):
    from race_v2.inference.predict import load, ase_calculator
    p = load(_checkpoint(tmp_path))
    a, b = _sheared(_water_box(seed=0, n=8, L=6.5), 1), _sheared(_water_box(seed=3, n=5, L=6.0), 2)
    e, f = p.predict([a, b])                                   # the two-value call is unchanged
    e3, f3, s = p.predict([a, b], stress=True)
    assert np.allclose(e, e3, rtol=0, atol=1e-10) and all(np.allclose(x, y, rtol=0, atol=1e-10) for x, y in zip(f, f3))
    assert s.shape == (2, 6) and np.all(np.isfinite(s)) and np.abs(s).max() > 1e-6
    # every frame gets its own row, whatever the batch it was evaluated in
    for k, fr in enumerate((a, b)):
        assert np.allclose(p.predict([fr], stress=True)[2][0], s[k], rtol=1e-9, atol=1e-12)
    # stress = (1 / V) dE / d(strain) of the predicted energy (central differences; the graph builder rounds the
    # geometry to float32, which limits the differences to ~1e-5 eV/A^3 at this step)
    h, vol, fd = 2e-3, a.get_volume(), []
    for (u, v) in [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]:
        eps = np.zeros((3, 3)); eps[u, v] += h / 2; eps[v, u] += h / 2
        ep = []
        for sgn in (1.0, -1.0):
            c = a.copy(); c.set_cell(a.cell.array @ (np.eye(3) + sgn * eps).T, scale_atoms=True)
            ep.append(p.predict([c])[0][0])
        fd.append((ep[0] - ep[1]) / (2 * h) / vol)
    assert np.allclose(s[0], np.array(fd), rtol=2e-3, atol=3e-5), (s[0], fd)
    # ASE calculator: the same numbers, and the 3 x 3 form
    a.calc = ase_calculator(p)
    assert np.allclose(a.get_stress(), s[0], rtol=1e-9, atol=1e-12)
    assert np.allclose(a.get_stress(voigt=False), a.get_stress(voigt=False).T)
    assert abs(a.get_potential_energy() - e[0]) < 1e-9 and np.allclose(a.get_forces(), f[0])


def test_no_stress_without_full_periodicity(tmp_path, x64):
    from race_v2.inference.predict import load, ase_calculator
    a = _sheared(_water_box(seed=0, n=8, L=6.5), 1)
    slab = a.copy(); slab.pbc = [True, True, False]
    p = load(_checkpoint(tmp_path))
    s = p.predict([a, slab], stress=True)[2]
    assert np.all(np.isfinite(s[0])) and np.all(np.isnan(s[1]))
    slab.calc = ase_calculator(p)
    assert np.all(np.isfinite(slab.get_forces()))                 # energy and forces still work
    with pytest.raises(PropertyNotImplementedError):
        slab.get_stress()
    # a model built without cells (periodic false) returns none either
    mol = ase.Atoms(numbers=a.numbers[:9], positions=a.positions[:9])
    pm = load(_checkpoint(tmp_path, periodic=False))
    assert np.all(np.isnan(pm.predict([mol], stress=True)[2]))
    mol.calc = ase_calculator(pm)
    with pytest.raises(PropertyNotImplementedError):
        mol.get_stress()


def test_calculator_defaults_float64_and_core_exclusion(tmp_path):
    """ase_calculator(ckpt) runs a float32-trained checkpoint in float64 with the core exclusion on (one definition of
    the model for every benchmark task); the Predictor called directly keeps the training precision and the
    checkpoint's own setting.  The core exclusion changes nothing where no pair is inside r_in and takes over below."""
    import jax
    from race_v2.inference.predict import Predictor, ase_calculator, resolve_core_exclusion, DEFAULT_CORE_EXCLUSION
    ck = _checkpoint(tmp_path, x64=False)
    try:
        p0 = Predictor(ck)
        assert not p0.promoted and p0.core_exclusion is None
        calc = ase_calculator(ck)
        pd_ = calc.predictor
        assert pd_.promoted and pd_.cfg['x64'] and pd_.core_exclusion == DEFAULT_CORE_EXCLUSION
        assert pd_.core_exclusion_source.startswith('default table')
        off = ase_calculator(ck, core_exclusion=False)
        assert off.predictor.core_exclusion is None and off.predictor.promoted
        assert not ase_calculator(ck, x64=False, core_exclusion=False).predictor.promoted
        # an ordinary structure: no pair below r_in, the two settings give the same numbers
        # (water molecules on a 3.2 A grid; the random boxes of _water_box have overlapping molecules)
        pos, num = [], []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    o = 3.2 * np.array([i, j, k]) + 0.8
                    pos += [o, o + [0.96, 0.0, 0.0], o + [-0.24, 0.93, 0.0]]; num += [8, 1, 1]
        a = ase.Atoms(numbers=num, positions=np.array(pos), cell=np.eye(3) * 6.4, pbc=True)
        a.rattle(0.03, seed=2)
        b = a.copy(); a.calc, b.calc = calc, off
        assert abs(a.get_potential_energy() - b.get_potential_energy()) < 1e-9
        assert np.allclose(a.get_forces(), b.get_forces(), atol=1e-9) and np.allclose(a.get_stress(), b.get_stress(), atol=1e-11)
        # an O-O pair far inside r_in (0.5 (R_O + R_O) = 0.66 A at most): no ML edge is left, the pair energy is ZBL
        d = ase.Atoms('O2', positions=[[5.0, 5.0, 5.0], [5.0, 5.0, 5.2]], cell=[10.0] * 3, pbc=True)
        d2 = d.copy(); d.calc, d2.calc = calc, off
        r = float(np.float32(5.2)) - 5.0             # the graph builder rounds the positions to float32
        x = r / (0.46850 / (2 * 8 ** 0.23))
        zbl = 14.399645 * 64 / r * (0.18175 * np.exp(-3.19980 * x) + 0.50986 * np.exp(-0.94229 * x)
                                      + 0.28022 * np.exp(-0.40290 * x) + 0.02817 * np.exp(-0.20162 * x))
        # reference of the pair energy: the same two atoms beyond the cutoff (the model's energy of two isolated atoms)
        far = ase.Atoms('O2', positions=[[5.0, 5.0, 2.5], [5.0, 5.0, 7.5]], cell=[10.0] * 3, pbc=True); far.calc = calc
        e_on = d.get_potential_energy() - far.get_potential_energy()
        assert abs(e_on - zbl) < 1e-7 * zbl and abs(d2.get_potential_energy() - d.get_potential_energy()) > 1.0
        assert d.get_forces()[1, 2] > 100.0                                   # strongly repulsive
        # the checkpoint's own specification wins over the default table; RACE_LR checkpoints keep their config
        own = {'table': {'8-8': 0.3}, 'cap': 0.4, 'width': 0.05}
        assert resolve_core_exclusion({'model': 'RACE', 'core_exclusion': own}, True) == (own, 'config')
        assert resolve_core_exclusion({'model': 'RACE_LR'}, True)[0] is None
    finally:
        jax.config.update("jax_enable_x64", False)
