"""Per-term Huber deltas (2026-11-13; config keys huber_delta_energy / _force / _stress): the loss with
(dE, dF, dS) equals, term by term, the loss with one delta; a single number keeps the old behaviour.
Run in a Slurm job (JAX)."""
import numpy as np
import pytest

from conftest import ATOMIC_NUMBERS  # noqa: F401
from race_v2.training.train import COVALENT_RADII  # noqa: F401  (imports training.train before the x64 fixture)
from test_race_polar_qeq import _polar_model, _molecule, _batch  # noqa: E402


def test_huber_deltas_parsing():
    from race_v2.training.utils import huber_deltas
    assert huber_deltas(0.02) == (0.02, 0.02, 0.02)
    assert huber_deltas((0.05, 0.25, 0.05)) == (0.05, 0.25, 0.05)
    assert huber_deltas([0.1, 0.2, 0.3]) == (0.1, 0.2, 0.3)
    with pytest.raises(ValueError):
        huber_deltas((0.1, 0.2))


def test_per_term_deltas_match_single_delta_terms(x64):
    import jax
    import jax.numpy as jnp
    from race_v2.training.utils import _loss_terms
    m = _polar_model(11)
    b = jax.tree.map(jnp.asarray, _batch([_molecule(1), _molecule(2, charge=1.0)]))
    res = m(b)
    # forces of an untrained model: residuals of order 1, so the deltas matter
    kw = dict(loss_type='huber', periodic=False, cov_mode='scalar6', beta_nll=0.0)
    t3 = _loss_terms(res, b, (0.05, 0.25, 0.05), **kw)
    tE = _loss_terms(res, b, 0.05, **kw)
    tF = _loss_terms(res, b, 0.25, **kw)
    assert abs(float(t3['energy_loss']) - float(tE['energy_loss'])) < 1e-12
    assert abs(float(t3['force_loss']) - float(tF['force_loss'])) < 1e-12
    assert abs(float(tE['force_loss']) - float(tF['force_loss'])) > 1e-9      # the force delta does change it
    # metrics are independent of the deltas
    assert float(t3['force_ae']) == float(tE['force_ae'])
