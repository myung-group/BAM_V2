"""Soft layer norm of the x readout input (RACE x_readout_norm_eps, 2026-11-15): x / sqrt(ms + e0^2).  The plain
norm (eps 1e-6) lifted the tiny neighbour density of an atom whose only neighbour sits near the cutoff to O(1)."""
import numpy as np
import e3nn_jax as e3nn


def _x(scale, seed=0):
    import jax.numpy as jnp
    irreps = e3nn.Irreps("8x0e + 4x1o + 2x2e")
    rng = np.random.default_rng(seed)
    return e3nn.IrrepsArray(irreps, jnp.asarray(scale * rng.normal(size=(5, irreps.dim))))


def test_default_eps_is_the_old_norm(x64):
    from race_v2.models.blocks import separated_layer_norm
    x = _x(0.3)
    a = np.asarray(separated_layer_norm(x).array)
    b = np.asarray(separated_layer_norm(x, eps=1e-6).array)
    assert np.array_equal(a, b)
    s = np.asarray(x.array)[:, :8]
    ref = (s - s.mean(-1, keepdims=True)) / np.sqrt(((s - s.mean(-1, keepdims=True)) ** 2).mean(-1, keepdims=True) + 1e-6)
    assert np.allclose(a[:, :8], ref, atol=1e-12)


def test_soft_norm_large_unchanged_small_linear(x64):
    from race_v2.models.blocks import separated_layer_norm
    e0 = 1e-2                                         # (the old eps 1e-6 is e0 = 1e-3)
    big = _x(1.0)                                     # |x| >> e0: the plain norm
    a = np.asarray(separated_layer_norm(big, eps=e0 ** 2).array)
    b = np.asarray(separated_layer_norm(big).array)
    assert np.abs(a - b).max() < 1e-3
    # |x| << e0: linear in x (the lone-neighbour regime), so it goes to 0 smoothly as the neighbour leaves
    for c in (1e-5, 1e-6):
        y = np.asarray(separated_layer_norm(_x(c), eps=e0 ** 2).array)
        y2 = np.asarray(separated_layer_norm(_x(2 * c), eps=e0 ** 2).array)
        assert np.allclose(y2, 2 * y, rtol=1e-5, atol=1e-15)
        assert np.abs(y).max() < 1e-2
    # the plain norm lifts the same tiny x to O(1) (the bug)
    plain = np.abs(np.asarray(separated_layer_norm(_x(3e-3)).array)).max()
    soft = np.abs(np.asarray(separated_layer_norm(_x(3e-3), eps=e0 ** 2).array)).max()
    assert plain > 0.5 and soft < 0.5 * plain


def test_readout_takes_the_eps():
    from flax import nnx
    from race_v2.models.race import RACEXReadout
    r = RACEXReadout(hidden_irreps=e3nn.Irreps("8x0e"), output_irreps=e3nn.Irreps("0e"),
                     x_irreps=e3nn.Irreps("8x0e + 4x1o"), n_species=2, avg_n_neighbors=10.0, rngs=nnx.Rngs(0),
                     norm=True, norm_eps=1e-4)
    assert r.norm_eps == 1e-4
