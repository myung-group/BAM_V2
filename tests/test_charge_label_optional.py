"""charge_label_optional (2026-11-11): structures without reference charges get NaN labels and are skipped by the
charge loss; fully labelled data are unchanged."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bam_omat24'))


def test_missing_labels_nan_only_when_optional():
    from graph_fields import extra_fields, field_keys
    info = {'charge': 1, 'spin': 2}
    with pytest.raises(KeyError):
        extra_fields(info, {}, 3, charge_key='charge', spin_key='spin', charge_label_key='lowdin_charges')
    nx, gx = extra_fields(info, {}, 3, charge_key='charge', spin_key='spin', charge_label_key='lowdin_charges',
                          charge_label_optional=True)
    assert nx['ref_charges'].shape == (3,) and nx['ref_charges'].dtype == np.float32
    assert np.isnan(nx['ref_charges']).all() and gx['charge'][0] == 1.0 and gx['spin'][0] == 2.0
    # labels present: the option changes nothing (aselmdb rows keep per-atom labels in info)
    lab = {'lowdin_charges': [0.1, -0.2, 0.1], **info}
    a, _ = extra_fields(lab, {}, 3, charge_label_key='lowdin_charges')
    b, _ = extra_fields(lab, {}, 3, charge_label_key='lowdin_charges', charge_label_optional=True)
    np.testing.assert_array_equal(a['ref_charges'], b['ref_charges'])
    assert field_keys(charge_key='charge', charge_label_key='lowdin_charges', charge_label_optional=True) == \
        field_keys(charge_key='charge', charge_label_key='lowdin_charges')


@pytest.mark.skipif(not os.environ.get('SLURM_JOB_ID'), reason='JAX test: run inside a job')
def test_masked_charge_loss_skips_nan_labels():
    import jax
    import jax.numpy as jnp
    from bam_omat24.training_utils import masked_charge_loss
    q = jnp.array([0.3, -0.1, 0.2, 0.5, 0.0, 0.0])
    ref = jnp.array([0.1, 0.1, jnp.nan, jnp.nan, 0.0, 0.0])        # atoms 2, 3: structure without labels
    mask = jnp.array([True, True, True, True, False, False])       # atoms 4, 5: padding
    val, grad = jax.value_and_grad(masked_charge_loss)(q, ref, mask, 4.0)
    np.testing.assert_allclose(val, (0.2 ** 2 + 0.2 ** 2) / 4.0, rtol=1e-6)
    np.testing.assert_allclose(grad, [2 * 0.2 / 4, 2 * -0.2 / 4, 0.0, 0.0, 0.0, 0.0], rtol=1e-6, atol=1e-12)
    # fully labelled: equal to the old expression sum(mask * (q - ref)^2) / n_atoms
    ref2 = jnp.array([0.1, 0.1, 0.0, 0.4, 9.0, 9.0])
    old = jnp.sum(mask * (q - ref2) ** 2) / 4.0
    np.testing.assert_allclose(masked_charge_loss(q, ref2, mask, 4.0), old, rtol=1e-7)
