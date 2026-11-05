"""RACE model implementation using Flax NNX.

This module implements the RACE model
using Flax NNX framework with e3nn_jax for equivariant operations.

Changes from original:
- Fixed hardcoded irreps in RACEXReadout (now derived from hidden_irreps)
- Fixed avg_n_neighbors Param usage (now consistent)
- avg_n_neighbors / atom_energies are now non-trainable constants
  (static Python float / tuple) instead of nnx.Param.
- New option ``x_mode``: "species" (default, original behaviour: x_feats is a
  per-species scalar) or "node_scalar" (x_feats is a per-layer linear projection
  of the receiving node's current scalar (0e) features).
- New option ``cov_mode``: "scalar6" (default, original behaviour: 6 invariant
  scalars fill a 3x3 Cholesky factor, which is NOT rotation-equivariant) or
  "irreps" (predict ``1x0e + 1x2e`` and build Sigma = (a*I + B)^2 with B a
  traceless symmetric tensor, which is rotation-equivariant).

"""

from typing import Optional, Sequence, Tuple, Dict, Union

import e3nn_jax as e3nn
import flax.nnx as nnx
import math
import jax
import jax.numpy as jnp
import jraph

from bam_omat24.utils import (
    separated_layer_norm,
    MLP,
    compute_node_edge_features,
    energy_forces_stress,
    bessel_basis,
    polynomial_cutoff,
    node_graph_idx,
)
from bam_omat24.linear_nnx import Linear as e3nn_nnx_Linear


# -----------------------------------------------------------------------------
# Optional cuEquivariance backend for the edge tensor product
# -----------------------------------------------------------------------------
# tp_backend="cue" replaces
#     messages = TP(linear_1(x)[senders], sh) * radial(r_ij);  agg = scatter_sum(messages, receivers)
# by a single fused CUDA kernel (cuequivariance "channelwise tensor product" with
# gather/scatter indices), so the (n_edges, mul * n_paths) message tensor is never
# materialised.  Requires `cuequivariance-jax` (+ `cuequivariance-ops-jax-cu12`);
# `jax.devices()` must have been called before the import so that the pip cuBLAS
# is loaded first.  Descriptors are cached per (irreps_in, sh_irreps, irreps_out).
_CUE_CACHE: Dict[Tuple[str, str, str], object] = {}


def _cue_descriptor(irreps_in: e3nn.Irreps, sh_irreps: e3nn.Irreps, irreps_out: e3nn.Irreps):
    key = (str(irreps_in), str(sh_irreps), str(irreps_out))
    if key not in _CUE_CACHE:
        import jax as _jax
        _jax.devices()
        import cuequivariance as cue
        O3 = cue.O3
        e = cue.descriptors.channelwise_tensor_product(
            cue.Irreps(O3, key[0]), cue.Irreps(O3, key[1]), cue.Irreps(O3, key[2])
        )
        # non-uniform multiplicities (tapered hidden irreps, e.g. 128x0e+64x1o+32x2e+16x3o) are
        # not supported by the fused 'uniform_1d' kernel (equal segment extents per operand; the
        # channel mode u cannot be split because it is not the leading mode of the ir_mul
        # operands) -> use tp_backend="e3nn" for such irreps.
        muls = {mul for mul, _ in irreps_in} | {mul for mul, _ in irreps_out}
        if len(muls) > 1:
            raise NotImplementedError(
                f"tp_backend='cue' needs a uniform multiplicity, got {irreps_in} -> {irreps_out}; "
                "use tp_backend='e3nn' for tapered hidden irreps")
        _CUE_CACHE[key] = e
    return _CUE_CACHE[key]


def _mul_ir_to_ir_mul(irreps: e3nn.Irreps, x: jax.Array) -> jax.Array:
    """e3nn-jax layout (mul, ir.dim) per irrep block -> cuEquivariance (ir.dim, mul)."""
    chunks = []
    i = 0
    for mul, ir in irreps:
        n = mul * ir.dim
        chunks.append(x[..., i:i + n].reshape(x.shape[:-1] + (mul, ir.dim)).swapaxes(-1, -2).reshape(x.shape[:-1] + (n,)))
        i += n
    return jnp.concatenate(chunks, axis=-1)


def _ir_mul_to_mul_ir(irreps: e3nn.Irreps, x: jax.Array) -> jax.Array:
    chunks = []
    i = 0
    for mul, ir in irreps:
        n = mul * ir.dim
        chunks.append(x[..., i:i + n].reshape(x.shape[:-1] + (ir.dim, mul)).swapaxes(-1, -2).reshape(x.shape[:-1] + (n,)))
        i += n
    return jnp.concatenate(chunks, axis=-1)


def cue_edge_tensor_product(
    irreps_in: e3nn.Irreps,
    sh_irreps: e3nn.Irreps,
    irreps_out: e3nn.Irreps,
    node_feats: jax.Array,       # (n_nodes, irreps_in.dim)   e3nn (mul_ir) layout
    sh: jax.Array,               # (n_edges, sh_irreps.dim)   mul = 1, layout irrelevant
    weights: jax.Array,          # (n_edges, n_weights)
    senders: jax.Array,
    receivers: jax.Array,
    method: str = "uniform_1d",
) -> e3nn.IrrepsArray:
    """sum_{j in N(i)} w_ij * (x_j (x) Y(r_ij)), fused gather/TP/scatter.

    cuEquivariance operands use the ir_mul layout; the (cheap, per-node)
    transposes convert from/to the e3nn-jax mul_ir layout.
    """
    import cuequivariance_jax as cuex
    e = _cue_descriptor(irreps_in, sh_irreps, irreps_out)
    poly = e.polynomial
    out_irreps = e3nn.Irreps(str(e.outputs[0].irreps))
    x = _mul_ir_to_ir_mul(irreps_in, node_feats)
    [out] = cuex.segmented_polynomial(
        poly, [weights, x, sh],
        [jax.ShapeDtypeStruct((node_feats.shape[0], poly.outputs[0].size), node_feats.dtype)],
        [None, senders, None, receivers],
        method=method,
    )
    return e3nn.IrrepsArray(out_irreps, _ir_mul_to_mul_ir(out_irreps, out))



_CUE_SYM_CACHE: Dict[Tuple[str, str, Tuple[int, ...]], object] = {}


def _cue_symmetric_descriptor(irreps_in: e3nn.Irreps, irreps_out: e3nn.Irreps, degrees: Tuple[int, ...]):
    """cuEquivariance descriptor of the MACE symmetric contraction
    B_i = sum_nu W_nu^{Z_i} . (A_i)^{(x) nu}  for nu in degrees (operands: weights[u], A[u] x nu, B[u]);
    the multiplicity u is treated channelwise (requires uniform multiplicities)."""
    key = (str(irreps_in), str(irreps_out), tuple(int(d) for d in degrees))
    if key not in _CUE_SYM_CACHE:
        import cuequivariance as cue
        muls = {mul for mul, _ in irreps_in} | {mul for mul, _ in irreps_out}
        if len(muls) != 1:
            raise NotImplementedError(
                f"symmetric contraction needs one uniform multiplicity, got {irreps_in} -> {irreps_out}")
        _CUE_SYM_CACHE[key] = cue.descriptors.symmetric_contraction(
            cue.Irreps(cue.O3, key[0]), cue.Irreps(cue.O3, key[1]), key[2])
    return _CUE_SYM_CACHE[key]


def cue_symmetric_contraction(
    irreps_in: e3nn.Irreps,
    irreps_out: e3nn.Irreps,
    degrees: Tuple[int, ...],
    a: jax.Array,               # (n_nodes, irreps_in.dim)   e3nn (mul_ir) layout
    weights: jax.Array,         # (n_nodes, n_weights)       per-node (species-indexed) weights
    method: str = "uniform_1d",
) -> e3nn.IrrepsArray:
    """Symmetric contraction of the atomic basis with itself (degrees nu), species weights.
    method "uniform_1d" = fused CUDA kernel (GPU); "naive" = pure JAX (any device)."""
    jax.devices()            # load the pip cuBLAS before cuequivariance (see _cue_descriptor)
    import cuequivariance_jax as cuex
    e = _cue_symmetric_descriptor(irreps_in, irreps_out, degrees)
    poly = e.polynomial
    x = _mul_ir_to_ir_mul(irreps_in, a)
    [out] = cuex.segmented_polynomial(
        poly, [weights, x],
        [jax.ShapeDtypeStruct((a.shape[0], poly.outputs[0].size), a.dtype)],
        method=method,
    )
    return e3nn.IrrepsArray(irreps_out, _ir_mul_to_mul_ir(irreps_out, out))


def _jackson_gain(n: int, N: int) -> float:
    """Jackson kernel g_n for N moments (n = 0..N-1), KPM damping (Weisse et al. RMP 78, 275, eq. 71)."""
    q = math.pi / (N + 1)
    return ((N - n + 1) * math.cos(q * n) + math.sin(q * n) / math.tan(q)) / (N + 1)


def _remat_layers() -> bool:
    """Environment BAM_REMAT_LAYERS=1: rematerialise every interaction layer (read at trace time)."""
    import os
    return os.environ.get("BAM_REMAT_LAYERS", "0") == "1"


def _rms_normalize(x: e3nn.IrrepsArray, eps: float = 1e-6) -> e3nn.IrrepsArray:
    """x / sqrt(mean over all components of x^2 + eps), per node (an invariant rescaling: equivariant)."""
    arr = x.array
    return e3nn.IrrepsArray(x.irreps, arr * jax.lax.rsqrt(jnp.mean(arr * arr, axis=-1, keepdims=True) + eps))


def _hill_normalize(x: e3nn.IrrepsArray, ab: jax.Array, species: jax.Array,
                    eps: float = 1e-6) -> e3nn.IrrepsArray:
    """Per-node RMS normalisation with a bounded degree gate:  x_hat_i * 2 sigmoid(a_Z log r_i + b_Z),
    r_i = per-node RMS of x_i, ab = (2, n_species).  a = b = 0 is _rms_normalize; a = 1 gives
    2 x_i / (r_i + e^{-b}) (magnitude kept for r_i << e^{-b}, saturating at |x_hat| <= 2): the gate
    restores the local degree r_i that the plain normalisation discards, with the row bound kept."""
    arr = x.array
    ms = jnp.mean(arr * arr, axis=-1, keepdims=True) + eps
    g = 2.0 * jax.nn.sigmoid(ab[0][species][:, None] * 0.5 * jnp.log(ms) + ab[1][species][:, None])
    return e3nn.IrrepsArray(x.irreps, arr * jax.lax.rsqrt(ms) * g)


def _spectral_normalize(lin, x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
    """x = lin(A) with every output-irrep block of the (vanilla, bias-free) equivariant linear map
    divided by its operator norm  sigma_max([path_weight_k W_k]_k)  (stacked over the input blocks
    feeding it), i.e. ||W_hat||_2 = 1 per block: |x_i| <= |A_i|, and the magnitude of A_i (the local
    density) is kept -- one global (KPM-type) rescaling instead of the per-node one."""
    fl = lin._linear
    ws = {}
    for ins in fl.instructions:
        if ins.i_in == -1:
            raise ValueError("x_norm 'spectral' needs a bias-free x projection")
        name = f"w[{ins.i_in},{ins.i_out}] {fl.irreps_in[ins.i_in]},{fl.irreps_out[ins.i_out]}"
        ws.setdefault(ins.i_out, []).append(ins.path_weight * lin._get_weight(name))
    scales = []
    for k, (mul, ir) in enumerate(fl.irreps_out):
        if k in ws:
            m = jnp.concatenate(ws[k], axis=0).astype(x.array.dtype)        # (sum mul_in, mul_out)
            inv = jax.lax.rsqrt(jnp.linalg.eigvalsh(m.T @ m)[-1])
        else:
            inv = jnp.ones((), x.array.dtype)                                 # structurally zero block
        scales.append(jnp.broadcast_to(inv, (mul * ir.dim,)))
    return e3nn.IrrepsArray(x.irreps, x.array * jnp.concatenate(scales)[None, :])


def _dense(x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
    """Materialise zero (None) chunks, e.g. after a Linear with force_irreps_out=True
    whose output irreps are not reachable from its input (layer 0: only 0e in, so the
    l>0 / odd-parity outputs are structurally zero).  Downstream code (tensor products,
    layer norm) then sees ordinary zero blocks instead of dropping those irreps."""
    return e3nn.IrrepsArray(x.irreps, x.array)


# UFF van der Waals constants (Rappé, Casewit, Colwell, Goddard, Skiff, JACS 114, 10024 (1992), Table 1, as in the
# RDKit / Open Babel parameter files; Z = 1-102 parsed from openbabel/data/UFF.prm, first atom type of each element):
# Z -> (x_I nonbond distance A, D_I well depth eV, zeta_I scale).  x, D and zeta are the same for all atom types of an
# element in that file.  D converted from kcal/mol (1 kcal/mol = 0.0433641 eV).
UFF_VDW = {
    1: (2.886, 0.044 * 0.0433641, 12),            # H
    2: (2.362, 0.056 * 0.0433641, 15.24),         # He
    3: (2.451, 0.025 * 0.0433641, 12),            # Li
    4: (2.745, 0.085 * 0.0433641, 12),            # Be
    5: (4.083, 0.18 * 0.0433641, 12.052),         # B
    6: (3.851, 0.105 * 0.0433641, 12.73),         # C
    7: (3.66, 0.069 * 0.0433641, 13.407),         # N
    8: (3.5, 0.06 * 0.0433641, 14.085),           # O
    9: (3.364, 0.05 * 0.0433641, 14.762),         # F
    10: (3.243, 0.042 * 0.0433641, 15.44),         # Ne
    11: (2.983, 0.03 * 0.0433641, 12),             # Na
    12: (3.021, 0.111 * 0.0433641, 12),            # Mg
    13: (4.499, 0.505 * 0.0433641, 11.278),        # Al
    14: (4.295, 0.402 * 0.0433641, 12.175),        # Si
    15: (4.147, 0.305 * 0.0433641, 13.072),        # P
    16: (4.035, 0.274 * 0.0433641, 13.969),        # S
    17: (3.947, 0.227 * 0.0433641, 14.866),        # Cl
    18: (3.868, 0.185 * 0.0433641, 15.763),        # Ar
    19: (3.812, 0.035 * 0.0433641, 12),            # K
    20: (3.399, 0.238 * 0.0433641, 12),            # Ca
    21: (3.295, 0.019 * 0.0433641, 12),            # Sc
    22: (3.175, 0.017 * 0.0433641, 12),            # Ti
    23: (3.144, 0.016 * 0.0433641, 12),            # V
    24: (3.023, 0.015 * 0.0433641, 12),            # Cr
    25: (2.961, 0.013 * 0.0433641, 12),            # Mn
    26: (2.912, 0.013 * 0.0433641, 12),            # Fe
    27: (2.872, 0.014 * 0.0433641, 12),            # Co
    28: (2.834, 0.015 * 0.0433641, 12),            # Ni
    29: (3.495, 0.005 * 0.0433641, 12),            # Cu
    30: (2.763, 0.124 * 0.0433641, 12),            # Zn
    31: (4.383, 0.415 * 0.0433641, 11),            # Ga
    32: (4.28, 0.379 * 0.0433641, 12),             # Ge
    33: (4.23, 0.309 * 0.0433641, 13),             # As
    34: (4.205, 0.291 * 0.0433641, 14),            # Se
    35: (4.189, 0.251 * 0.0433641, 15),            # Br
    36: (4.141, 0.22 * 0.0433641, 16),             # Kr
    37: (4.114, 0.04 * 0.0433641, 12),             # Rb
    38: (3.641, 0.235 * 0.0433641, 12),            # Sr
    39: (3.345, 0.072 * 0.0433641, 12),            # Y
    40: (3.124, 0.069 * 0.0433641, 12),            # Zr
    41: (3.165, 0.059 * 0.0433641, 12),            # Nb
    42: (3.052, 0.056 * 0.0433641, 12),            # Mo
    43: (2.998, 0.048 * 0.0433641, 12),            # Tc
    44: (2.963, 0.056 * 0.0433641, 12),            # Ru
    45: (2.929, 0.053 * 0.0433641, 12),            # Rh
    46: (2.899, 0.048 * 0.0433641, 12),            # Pd
    47: (3.148, 0.036 * 0.0433641, 12),            # Ag
    48: (2.848, 0.228 * 0.0433641, 12),            # Cd
    49: (4.463, 0.599 * 0.0433641, 11),            # In
    50: (4.392, 0.567 * 0.0433641, 12),            # Sn
    51: (4.42, 0.449 * 0.0433641, 13),             # Sb
    52: (4.47, 0.398 * 0.0433641, 14),             # Te
    53: (4.5, 0.339 * 0.0433641, 15),              # I
    54: (4.404, 0.332 * 0.0433641, 12),            # Xe
    55: (4.517, 0.045 * 0.0433641, 12),            # Cs
    56: (3.703, 0.364 * 0.0433641, 12),            # Ba
    57: (3.522, 0.017 * 0.0433641, 12),            # La
    58: (3.556, 0.013 * 0.0433641, 12),            # Ce
    59: (3.606, 0.01 * 0.0433641, 12),             # Pr
    60: (3.575, 0.01 * 0.0433641, 12),             # Nd
    61: (3.547, 0.009 * 0.0433641, 12),            # Pm
    62: (3.52, 0.008 * 0.0433641, 12),             # Sm
    63: (3.493, 0.008 * 0.0433641, 12),            # Eu
    64: (3.368, 0.009 * 0.0433641, 12),            # Gd
    65: (3.451, 0.007 * 0.0433641, 12),            # Tb
    66: (3.428, 0.007 * 0.0433641, 12),            # Dy
    67: (3.409, 0.007 * 0.0433641, 12),            # Ho
    68: (3.391, 0.007 * 0.0433641, 12),            # Er
    69: (3.374, 0.006 * 0.0433641, 12),            # Tm
    70: (3.355, 0.228 * 0.0433641, 12),            # Yb
    71: (3.64, 0.041 * 0.0433641, 12),             # Lu
    72: (3.141, 0.072 * 0.0433641, 12),            # Hf
    73: (3.17, 0.081 * 0.0433641, 12),             # Ta
    74: (3.069, 0.067 * 0.0433641, 12),            # W
    75: (2.954, 0.066 * 0.0433641, 12),            # Re
    76: (3.12, 0.037 * 0.0433641, 12),             # Os
    77: (2.84, 0.073 * 0.0433641, 12),             # Ir
    78: (2.754, 0.08 * 0.0433641, 12),             # Pt
    79: (3.293, 0.039 * 0.0433641, 12),            # Au
    80: (2.705, 0.385 * 0.0433641, 12),            # Hg
    81: (4.347, 0.68 * 0.0433641, 11),             # Tl
    82: (4.297, 0.663 * 0.0433641, 12),            # Pb
    83: (4.37, 0.518 * 0.0433641, 13),             # Bi
    84: (4.709, 0.325 * 0.0433641, 14),            # Po
    85: (4.75, 0.284 * 0.0433641, 15),             # At
    86: (4.765, 0.248 * 0.0433641, 16),            # Rn
    87: (4.9, 0.05 * 0.0433641, 12),               # Fr
    88: (3.677, 0.404 * 0.0433641, 12),            # Ra
    89: (3.478, 0.033 * 0.0433641, 12),            # Ac
    90: (3.396, 0.026 * 0.0433641, 12),            # Th
    91: (3.424, 0.022 * 0.0433641, 12),            # Pa
    92: (3.395, 0.022 * 0.0433641, 12),            # U
    93: (3.424, 0.019 * 0.0433641, 12),            # Np
    94: (3.424, 0.016 * 0.0433641, 12),            # Pu
    95: (3.381, 0.014 * 0.0433641, 12),            # Am
    96: (3.326, 0.013 * 0.0433641, 12),            # Cm
    97: (3.339, 0.013 * 0.0433641, 12),            # Bk
    98: (3.313, 0.013 * 0.0433641, 12),            # Cf
    99: (3.299, 0.012 * 0.0433641, 12),            # Es
    100: (3.286, 0.012 * 0.0433641, 12),            # Fm
    101: (3.274, 0.011 * 0.0433641, 12),            # Md
    102: (3.248, 0.011 * 0.0433641, 12),            # No
}


def _bessel_safe(x: jax.Array, num_basis: int, r_max: jax.Array) -> jax.Array:
    """Bessel basis 2/r_max * sin(n pi x / r_max) / x, finite *and NaN-free in the gradient*
    at x = 0 (padded edges have r = 0).  utils.bessel_basis evaluates sin(.)/x at x = 0 inside
    a jnp.where: the value is masked but the 1/x cotangent is inf, and inf * 0 = NaN reaches
    any parameter upstream of x (here alpha_ij -> rho).  Divide by a safe x instead."""
    r_max = r_max[:, None]
    x = x[:, None]
    xs = jnp.where(x == 0.0, 1.0, x)
    w = jnp.linspace(1.0, num_basis, num_basis) * jnp.pi
    return (2.0 / r_max) * jnp.where(x == 0.0, w / r_max, jnp.sin(w * xs / r_max) / xs)


def _scale_irreps(x: e3nn.IrrepsArray, gamma: jax.Array) -> e3nn.IrrepsArray:
    """Multiply every irrep channel by its own scalar (gamma has x.irreps.num_irreps entries;
    LayerScale, CaiT).  Equivariant: a per-channel scalar commutes with the group action."""
    chunks, off = [], 0
    for (mul, ir), c in zip(x.irreps, x.chunks):
        g = gamma[off:off + mul]
        off += mul
        chunks.append(None if c is None else c * g[:, None])
    return e3nn.from_chunks(x.irreps, chunks, x.shape[:-1], x.dtype)


def _gated_activation(x: e3nn.IrrepsArray, output_irreps: e3nn.Irreps) -> e3nn.IrrepsArray:
    """Equiformer-style gate, identical to e3nn.gate(even_act=silu, even_gate_act=silu)
    for natural-parity hidden irreps, but the gates are always taken from the extra
    0e block.  With "full parity" hidden irreps (0o present) e3nn.gate would use the
    last scalars -- including the 0o ones -- as gates and flip the parity of the gated
    vectors; here 0o scalars get an odd activation (tanh) and are never gates."""
    n0e = output_irreps.count("0e")
    se = x.filter(keep="0e")
    so = x.filter(keep="0o")
    vec = x.filter(drop=["0e", "0o"])
    extra = se.slice_by_mul[:n0e]
    gates = se.slice_by_mul[n0e:]
    assert gates.irreps.num_irreps == vec.irreps.num_irreps, (gates.irreps, vec.irreps)
    parts = [e3nn.scalar_activation(extra, even_act=jax.nn.silu, normalize_act=True)]
    if so.irreps.num_irreps > 0:
        parts.append(e3nn.scalar_activation(so, odd_act=jnp.tanh, normalize_act=True))
    gates = e3nn.scalar_activation(gates, even_act=jax.nn.silu, normalize_act=True)
    parts.append(gates * vec)
    return e3nn.concatenate(parts, axis=-1)


class RACEXReadout(nnx.Module):
    """RACE x_node_feature's readout layer with equivariant message passing."""

    def __init__(
        self,
        hidden_irreps: e3nn.Irreps,
        output_irreps: e3nn.Irreps,
        x_irreps: e3nn.Irreps,
        n_species: int,
        avg_n_neighbors: float,
        rngs: nnx.Rngs,
        norm: bool = True,
        linear: bool = False,
    ):
        # norm=False (review §11.16): no layer norm on the input -> the readout is a genuine
        # function of the magnitude of x (with x = A_i a body-ordered quantity), not a degree-0 one.
        # linear=True: species-indexed Linear(0e part of x) -> 0e, i.e. an exactly linear function of
        # x; with x = Linear(A_i) this is a learned 2-body (pair) stratum sum_j R_Z(r_ij).
        self.norm = bool(norm)
        self.linear = bool(linear)
        self.x_readout1 = e3nn_nnx_Linear(
            irreps_in=x_irreps,
            irreps_out=(e3nn.Irreps("0e") if self.linear else hidden_irreps),
            linear_type="indexed",
            num_indexed_weights=n_species,
            force_irreps_out=True,
            biases=self.linear,
            rngs=rngs,
        )
        if self.linear:
            self.avg_n_neighbors = avg_n_neighbors
            return
        self.x_readout2 = e3nn_nnx_Linear(
            irreps_in=hidden_irreps,
            irreps_out=hidden_irreps // 2,
            biases=True,
            rngs=rngs,
            force_irreps_out=True,
        )
        self.x_readout3 = e3nn_nnx_Linear(
            irreps_in=hidden_irreps // 2,
            irreps_out=output_irreps,
            biases=True,
            rngs=rngs,
            force_irreps_out=True,
        )
        self.avg_n_neighbors = avg_n_neighbors

    def __call__(
        self,
        species: jax.Array,
        x_feats: e3nn.IrrepsArray,
    ) -> e3nn.IrrepsArray:
        if x_feats.irreps.dim > 1 and self.norm and not self.linear:
            x_feats_norm = separated_layer_norm(_dense(x_feats)) / jnp.sqrt(self.avg_n_neighbors)
        else:
            x_feats_norm = _dense(x_feats) / jnp.sqrt(self.avg_n_neighbors)
        if self.linear:
            return self.x_readout1(species, x_feats_norm)
        x_features = self.x_readout1(species, x_feats_norm)
        x_features = self.x_readout2(x_features)
        x_features = e3nn.gate(
            x_features,
            even_act=jax.nn.silu,
            odd_act=jax.nn.sigmoid,
            even_gate_act=jax.nn.silu,
        )
        x_energy = self.x_readout3(x_features)

        return x_energy


class RACEConvolution(nnx.Module):
    """RACE convolution layer with equivariant message passing."""

    def __init__(
        self,
        input_irreps: e3nn.Irreps,
        output_irreps: e3nn.Irreps,
        x_irreps: e3nn.Irreps,
        sh_irreps: e3nn.Irreps,
        n_species: int,
        radial_basis_size: int,
        radial_mlp_size: int,
        radial_mlp_layers: int,
        mlp_init_scale: float,
        avg_n_neighbors: float,
        tp_backend: str = "e3nn",
        readout_mode: str = "linear",
        readout_species: bool = False,
        residual: str = "none",
        layer_scale: Optional[float] = None,
        msg_lmax: Optional[int] = None,
        radial_extra: int = 0,
        in_layer_order: int = 1,
        update: str = "gate",
        message_norm: str = "layer",
        sym_contraction: bool = False,
        sym_init_scale: float = 0.1,
        sym_target: str = "features",
        radial_scale: Optional[Sequence[float]] = None,
        radial_species_mix: Optional[str] = None,
        *,
        rngs: nnx.Rngs,
    ):
        if tp_backend not in ("e3nn", "cue"):
            raise ValueError(f"tp_backend must be 'e3nn' or 'cue', got {tp_backend!r}")
        if readout_mode not in ("linear", "gated"):
            raise ValueError(f"readout_mode must be 'linear' or 'gated', got {readout_mode!r}")
        if residual not in ("none", "identity", "identity_cum", "chebyshev", "neumann"):
            raise ValueError(
                f"residual must be 'none', 'identity' or 'identity_cum', got {residual!r}"
            )
        # update "gate" (original, NequIP-style: features of the next layer are the gated
        # output) or "linear" (review §11 A, BOTNet-style body-ordered update: the next layer
        # receives the *ungated* linear combination z = W_2 m + S_z h, so h^(t+1) is a finite
        # polynomial of the atomic basis; the non-linearity lives only in the stratum readout,
        # which therefore has to be the gated MLP).
        if update not in ("gate", "linear"):
            raise ValueError(f"update must be 'gate' or 'linear', got {update!r}")
        if update == "linear" and readout_mode != "gated":
            raise ValueError("update='linear' needs readout_mode='gated' (otherwise the stratum energy is linear in the features)")
        # message_norm "layer" (original: per-node separated layer norm of h_j before the message
        # tensor product, sum / sqrt(avg_n_neighbors)) or "avg" (review §11 B, BOTNet/MACE: no
        # feature normalisation, sum / avg_n_neighbors; keeps the body order of the features).
        if message_norm not in ("layer", "avg"):
            raise ValueError(f"message_norm must be 'layer' or 'avg', got {message_norm!r}")
        self.update = update
        self.message_norm = message_norm
        # radial regularity prior (review §11.15): fixed per-channel rescaling 1/gamma(n) of the
        # radial basis entering the radial MLP (static tuple; the matching L2 on the MLP's first
        # layer is applied in the loss, training_utils.compute_loss radial_l2)
        self.radial_scale = None if radial_scale is None else tuple(float(s) for s in radial_scale)
        # neighbour-species radial mixing (review §12.3, GRACE's W_n(mu_j)): a per-layer linear mix
        # of the radial channels by the species of the sender ("neighbor") or of the pair ("pair"),
        # identity at init, applied after the regularity-prior rescaling -> R_{mu_j n}(r)
        if radial_species_mix not in (None, "neighbor", "pair"):
            raise ValueError(f"radial_species_mix must be None, 'neighbor' or 'pair', got {radial_species_mix!r}")
        self.radial_species_mix = radial_species_mix
        if radial_species_mix is not None:
            n_r = radial_basis_size + radial_extra
            shape = (n_species, n_r, n_r) if radial_species_mix == "neighbor" else (n_species, n_species, n_r, n_r)
            self.radial_mix = nnx.Param(jnp.array(jnp.broadcast_to(jnp.eye(n_r, dtype=jnp.float32), shape)))
        self.readout_mode = readout_mode
        self.readout_species = readout_species
        self.residual = residual
        self.output_irreps = output_irreps
        # Hyperbolic-cross truncation (ACE sparse grid): the *new* information of this
        # stratum is limited to l <= msg_lmax (message TP output filter); the skip term
        # still carries all channels of the previous features.
        self.msg_irreps = (output_irreps if msg_lmax is None
                           else e3nn.Irreps([(m, ir) for m, ir in output_irreps if ir.l <= msg_lmax]))
        # LayerScale (CaiT): learnable per-channel gain on the layer increment, small init.
        self.gamma = None if layer_scale is None else nnx.Param(
            jnp.full((output_irreps.num_irreps,), float(layer_scale), dtype=jnp.float32))
        self.x_irreps = e3nn.Irreps(x_irreps)
        # x_feats is either a single scalar (x_mode "species"/"node_scalar": the
        # "tensor product" with it is a plain scaling) or a full equivariant
        # feature of the receiving node (x_mode "node_irreps": channelwise TP
        # m_i (x) x_i, the recursive ACE product  B^(t+1) = B^(t) (x) A_i).
        self.x_equivariant = self.x_irreps.dim > 1
        if self.x_equivariant:
            if self.x_irreps != output_irreps:
                raise ValueError(
                    f"x_irreps must equal output_irreps for the equivariant x product, "
                    f"got {x_irreps} vs {output_irreps}"
                )
            muls = {mul for mul, _ in output_irreps}
            if len(muls) != 1:
                raise ValueError(
                    f"equivariant x product needs a uniform multiplicity, got {output_irreps}"
                )
            self.x_mul = muls.pop()
            # multiplicity-1 copy of output_irreps: the channelwise product is a
            # plain e3nn tensor product with the channel moved to a batch axis
            self.x_irreps_1 = e3nn.Irreps([(1, ir) for _, ir in output_irreps])
        self.avg_n_neighbors = avg_n_neighbors
        self.tp_backend = tp_backend
        self.input_irreps = e3nn.Irreps(input_irreps)
        self.sh_irreps = e3nn.Irreps(sh_irreps)

        if tp_backend == "cue":
            e = _cue_descriptor(self.input_irreps, self.sh_irreps, self.msg_irreps)
            tp_irreps = e3nn.Irreps(str(e.outputs[0].irreps))
            n_weights = e.polynomial.inputs[0].size
            assert n_weights == tp_irreps.num_irreps
        else:
            tp_irreps = e3nn.tensor_product(
                input_irreps, sh_irreps, filter_ir_out=self.msg_irreps
            )

        self.tp_irreps = e3nn.Irreps(tp_irreps)      # irreps of the aggregated message m_i (= A_i for layer 0)
        self.linear_1 = e3nn_nnx_Linear(
            irreps_in=input_irreps,
            irreps_out=input_irreps,
            rngs=rngs,
        )

        self.radial_mlp = MLP(
            sizes=[radial_basis_size + radial_extra]     # radial_extra: shell-envelope channels (RACE.radial_shells)
            + [radial_mlp_size] * radial_mlp_layers
            + [tp_irreps.num_irreps],
            activation=jax.nn.silu,
            use_bias=False,
            init_scale=mlp_init_scale,
            rngs=rngs,
        )

        # Add extra irreps to output to account for gate
        # one 0e gate per non-scalar irrep (0o scalars are activated, not gated)
        gate_irreps = e3nn.Irreps(
            f"{output_irreps.filter(drop=['0e', '0o']).num_irreps}x0e"
        )
        gated_output_irreps = (output_irreps + gate_irreps).regroup()
        # linear update: no gate scalars, the layer output is output_irreps itself
        update_irreps = output_irreps if update == "linear" else gated_output_irreps

        if self.x_equivariant:
            # tp_irreps -> output_irreps (one path per channel), then the channelwise
            # product with x_i produces tp2_irreps (all CG paths kept by the filter)
            self.linear_m = e3nn_nnx_Linear(
                irreps_in=tp_irreps,
                irreps_out=output_irreps,
                force_irreps_out=True,
                rngs=rngs,
            )
            tp2_irreps_1 = e3nn.tensor_product(
                self.x_irreps_1, self.x_irreps_1, filter_ir_out=self.x_irreps_1
            )
            tp2_irreps = e3nn.Irreps([(self.x_mul * mul, ir) for mul, ir in tp2_irreps_1])
            linear_2_in = tp2_irreps
        else:
            linear_2_in = tp_irreps

        # in_layer_order 2 ("BRACE-2", review §9): a channelwise symmetric square of the
        # aggregated message, (m_i (x) m_i) -> Linear, added to the layer increment.  The
        # stratum of layer t then carries body orders t+2 (m) and 2t+3 (m (x) m): body order
        # still grows with depth, but the strata are no longer pure single orders.
        if in_layer_order not in (1, 2, 3, 4):
            raise ValueError(f"in_layer_order must be 1, 2, 3 or 4, got {in_layer_order}")
        self.in_layer_order = int(in_layer_order)
        # in_layer_order 3 (review §11.11, "coarse RACE"): MACE symmetric contraction of the
        # aggregated message with itself, B_i = sum_{nu=2..order} W_nu^{Z_i} . (Linear A_i)^{(x) nu},
        # followed by a Linear and added to the layer increment; the nu = 1 term is linear_2.
        # The stratum of layer t then carries body orders t+2 .. order*(t+1)+1.  With
        # sym_contraction=True the same path is used for order 2 (instead of the legacy
        # channelwise square of r13), so that nu = 2 and nu = 3 differ only in the degrees.
        self.sym_contraction = bool(sym_contraction) or self.in_layer_order >= 3
        if self.sym_contraction and self.in_layer_order == 1:
            self.sym_contraction = False
        if self.sym_contraction:
            self.sym_degrees = tuple(range(2, self.in_layer_order + 1))
            self.sym_method = "uniform_1d" if tp_backend == "cue" else "naive"
            muls = {mul for mul, _ in output_irreps}
            if len(muls) != 1:
                raise ValueError(f"in_layer_order={in_layer_order} needs a uniform multiplicity, got {output_irreps}")
            # A_i channels mixed (tp_irreps -> output_irreps) before the product, as the
            # radial/path weights do in MACE; the product then maps output_irreps -> output_irreps
            self.linear_sym_in = e3nn_nnx_Linear(
                irreps_in=tp_irreps, irreps_out=output_irreps, force_irreps_out=True, rngs=rngs,
            )
            e = _cue_symmetric_descriptor(output_irreps, output_irreps, self.sym_degrees)
            self.sym_n_weights = int(e.polynomial.inputs[0].size)
            # species-indexed contraction weights (MACE: one set per element); the cue paths are
            # variance-normalised per output, so N(0, 1) weights keep the output at the input scale
            self.sym_weights = nnx.Param(
                jax.random.normal(rngs.params(), (n_species, self.sym_n_weights), dtype=jnp.float32))
            # one learnable gain per degree (init sym_init_scale): a stack of unnormalised cubic
            # layers is dynamically unstable (|h| -> |h|^3 per layer, review §11.11 diagnostics),
            # so the product terms start small relative to the linear term and the optimiser
            # scales them up where useful.  Weights are stacked per degree in the cue descriptor.
            self.sym_degree_sizes = tuple(
                int(_cue_symmetric_descriptor(output_irreps, output_irreps, (d,)).polynomial.inputs[0].size)
                for d in self.sym_degrees)
            assert sum(self.sym_degree_sizes) == self.sym_n_weights, (self.sym_degree_sizes, self.sym_n_weights)
            self.sym_gain = nnx.Param(jnp.full((len(self.sym_degrees),), float(sym_init_scale), dtype=jnp.float32))
            # sym_target "features": B_i enters the layer increment (MACE: the product basis is the
            # next layer's input; a stack of such layers is a cubic map h -> h^3 per layer and
            # Adam drives the gains to O(1) within an epoch -> overflow, review §11.11).
            # "readout": B_i feeds only this stratum's energy through its own gated readout; the
            # propagated features stay nu = 1 (v8 backbone), the strata gain orders t+2 .. 3(t+1)+1.
            if sym_target not in ("features", "readout"):
                raise ValueError(f"sym_target must be 'features' or 'readout', got {sym_target!r}")
            self.sym_target = sym_target
            self.linear_sym = e3nn_nnx_Linear(
                irreps_in=output_irreps,
                irreps_out=(update_irreps if sym_target == "features" else output_irreps),
                force_irreps_out=True, rngs=rngs,
            )
            if sym_target == "readout":
                sym_ro_irreps = (output_irreps // 2).filter(keep="0e")
                self.readout_sym1 = e3nn_nnx_Linear(irreps_in=output_irreps, irreps_out=sym_ro_irreps,
                                                    biases=True, force_irreps_out=True, rngs=rngs)
                self.readout_sym = e3nn_nnx_Linear(irreps_in=sym_ro_irreps, irreps_out=e3nn.Irreps("0e"),
                                                   biases=True, rngs=rngs)
        else:
            self.sym_target = "features"
        if self.in_layer_order == 2 and not self.sym_contraction:
            muls = {mul for mul, _ in output_irreps}
            if len(muls) != 1:
                raise ValueError(f"in_layer_order=2 needs a uniform multiplicity, got {output_irreps}")
            self.sq_mul = muls.pop()
            self.sq_irreps_1 = e3nn.Irreps([(1, ir) for _, ir in output_irreps])
            self.linear_sq_in = e3nn_nnx_Linear(
                irreps_in=tp_irreps, irreps_out=output_irreps, force_irreps_out=True, rngs=rngs,
            )
            sq_irreps_1 = e3nn.tensor_product(self.sq_irreps_1, self.sq_irreps_1,
                                              filter_ir_out=self.sq_irreps_1)
            sq_irreps = e3nn.Irreps([(self.sq_mul * mul, ir) for mul, ir in sq_irreps_1])
            self.linear_sq = e3nn_nnx_Linear(
                irreps_in=sq_irreps, irreps_out=update_irreps, force_irreps_out=True, rngs=rngs,
            )

        # force_irreps_out: at layer 0 (0e input) the odd-parity / l>0 outputs may be
        # unreachable; they must still exist (as zeros) to be added to the skip term.
        self.linear_2 = e3nn_nnx_Linear(
            irreps_in=linear_2_in,
            irreps_out=update_irreps,
            force_irreps_out=True,
            rngs=rngs,
        )

        # Skip connection has per-species weights
        self.skip = e3nn_nnx_Linear(
            irreps_in=input_irreps,
            irreps_out=update_irreps,
            linear_type="indexed",
            num_indexed_weights=n_species,
            force_irreps_out=True,
            rngs=rngs,
        )

        # Stratum readout of this layer.  "linear": Linear -> Linear(0e) (original; the
        # final Linear only sees the 0e part, so the energy is linear in the scalar
        # features).  "gated": Linear(0e hidden) -> silu -> Linear(0e), a small MLP on the
        # scalar features as in the x/f readouts (RACEXReadout) and MACE's last layer.
        output_irreps_2 = output_irreps // 2
        if readout_mode == "gated":
            output_irreps_2 = output_irreps_2.filter(keep="0e")
        self.readout1 = e3nn_nnx_Linear(
            irreps_in=output_irreps,
            irreps_out=output_irreps_2,
            biases=True,
            rngs=rngs,
            force_irreps_out=True,
        )
        # readout_species: ACE expansion coefficients depend on the central species,
        # E_i = sum_nu c_nu^{Z_i} B_nu  -> per-species weights (and bias) in the last Linear.
        if readout_species:
            self.readout = e3nn_nnx_Linear(
                irreps_in=output_irreps_2,
                irreps_out=e3nn.Irreps("0e"),
                biases=True,
                linear_type="indexed",
                num_indexed_weights=n_species,
                force_irreps_out=True,
                rngs=rngs,
            )
        else:
            self.readout = e3nn_nnx_Linear(
                irreps_in=output_irreps_2,
                irreps_out=e3nn.Irreps("0e"),
                biases=True,
                rngs=rngs,
            )

    def aggregate(
        self,
        features: e3nn.IrrepsArray,
        sh: e3nn.IrrepsArray,
        radial_basis: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        edge_scale: Optional[jax.Array] = None,
        species_pair: Optional[Tuple[jax.Array, jax.Array]] = None,
    ) -> e3nn.IrrepsArray:
        """Aggregated message  m_i = (1/norm) sum_j R(r_ij) (x) [W_1 n(h_j) (x) Y(r_ij)]  (before the
        x product).  n = separated layer norm and norm = sqrt(avg_n_neighbors) for
        message_norm="layer" (original); n = identity and norm = avg_n_neighbors for "avg".
        With the layer-0 input (species embedding) this is the atomic basis A_i (x_mode="basis")."""
        if self.message_norm == "layer":
            # Follow equiformer_v2: layer norm before message passing
            features_norm = separated_layer_norm(features)
            norm = jnp.sqrt(self.avg_n_neighbors)
        else:
            features_norm = features
            norm = self.avg_n_neighbors
        if self.radial_scale is not None:
            radial_basis = radial_basis * jnp.asarray(self.radial_scale, dtype=radial_basis.dtype)[None, :]
        if self.radial_species_mix is not None:
            # species_pair = (species of the receiver i, species of the sender j) per edge
            if species_pair is None:
                raise ValueError("radial_species_mix needs species_pair in aggregate()")
            sp_i, sp_j = species_pair
            w = self.radial_mix[...].astype(radial_basis.dtype)
            w_e = w[sp_j] if self.radial_species_mix == "neighbor" else w[sp_i, sp_j]     # (n_edges, n_r, n_r)
            radial_basis = jnp.einsum("en,enm->em", radial_basis, w_e)
        radial_message = jax.vmap(self.radial_mlp)(radial_basis)
        if edge_scale is not None:
            radial_message = radial_message * edge_scale[:, None]
        if self.tp_backend == "cue":
            node_in = self.linear_1(features_norm)
            messages_agg = cue_edge_tensor_product(
                node_in.irreps, sh.irreps, self.msg_irreps,
                node_in.array, sh.array, radial_message, senders, receivers,
            ) / norm
        else:
            messages = self.linear_1(features_norm)[senders]
            messages = e3nn.tensor_product(messages, sh, filter_ir_out=self.msg_irreps)
            messages = messages * radial_message
            messages_agg = e3nn.scatter_sum(
                messages, dst=receivers, output_size=features.shape[0]
            ) / norm
        return messages_agg

    def sym_term(self, messages_agg: e3nn.IrrepsArray, species: jax.Array) -> e3nn.IrrepsArray:
        """Symmetric-contraction increment  Linear( sum_nu W_nu^{Z_i} . (Linear A_i)^{(x) nu} ),
        nu in sym_degrees; an exact polynomial of degree 2..order in the aggregated message."""
        a = _dense(self.linear_sym_in(messages_agg))
        gain = jnp.concatenate([jnp.full((n,), g) for n, g in zip(self.sym_degree_sizes, self.sym_gain[...])])
        w = (self.sym_weights[...] * gain[None, :]).astype(a.array.dtype)[species]
        b = cue_symmetric_contraction(a.irreps, self.output_irreps, self.sym_degrees,
                                      a.array, w, method=self.sym_method)
        return self.linear_sym(b)

    def __call__(
        self,
        features: e3nn.IrrepsArray,
        x_feats: e3nn.IrrepsArray,
        species: jax.Array,
        sh: e3nn.IrrepsArray,
        radial_basis: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        edge_scale: Optional[jax.Array] = None,
        drop_scale: Optional[jax.Array] = None,
        prev_features: Optional[e3nn.IrrepsArray] = None,
    ) -> Tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]:
        """edge_scale: optional (n_edges,) multiplier of the radial weights (per-stratum
        cutoff envelope); drop_scale: optional (n_nodes,) stochastic-depth factor applied to
        the increment that enters the residual stream (0 = stratum dropped)."""
        messages_agg = self.aggregate(features, sh, radial_basis, senders, receivers, edge_scale,
                                      species_pair=(species[receivers], species[senders]))
        if not self.x_equivariant:
            # x_feats is 1x0e: the tensor product with it is a plain scaling
            if self.tp_backend == "cue":
                messages_agg = messages_agg * x_feats.array
            else:
                messages_agg = e3nn.tensor_product(messages_agg, x_feats, filter_ir_out=self.msg_irreps)
        sq_term = None
        sym_energy = None
        if self.sym_contraction and self.sym_target == "readout":
            b = self.sym_term(messages_agg, species)
            sym_energy = e3nn.gate(self.readout_sym1(b), even_act=jax.nn.silu, odd_act=jnp.tanh,
                                   even_gate_act=jax.nn.silu)
            sym_energy = self.readout_sym(sym_energy)
        elif self.sym_contraction:
            sq_term = self.sym_term(messages_agg, species)
        elif self.in_layer_order == 2:
            # channelwise symmetric square of the aggregated message (same channel->batch trick
            # as the equivariant x product); antisymmetric paths of m (x) m vanish identically
            m1 = _dense(self.linear_sq_in(messages_agg)).mul_to_axis(self.sq_mul)
            sq = e3nn.tensor_product(m1, m1, filter_ir_out=self.sq_irreps_1).axis_to_mul()
            sq_term = self.linear_sq(sq)
        if self.x_equivariant:
            # Recursive ACE product at the receiving node:  m_i (x) x_i, channelwise.
            # Channel -> batch axis so that e3nn's (multiplicity-1) tensor product
            # keeps every CG path  (l1, l2) -> l3 in output_irreps.
            # _dense: structurally-zero blocks (layer 0) must stay as zero arrays so
            # that every output irrep of the product exists (fixed irreps for linear_2)
            m = _dense(self.linear_m(messages_agg)).mul_to_axis(self.x_mul)
            xa = _dense(x_feats).mul_to_axis(self.x_mul)
            prod = e3nn.tensor_product(m, xa, filter_ir_out=self.x_irreps_1)
            messages_agg = prod.axis_to_mul()
        delta = self.linear_2(messages_agg) + self.skip(species, features)
        if sq_term is not None:
            delta = delta + sq_term
        delta = _dense(delta)

        if self.update == "gate":
            # Follow equiformer: gated activation (see _gated_activation; identical to
            # e3nn.gate for natural-parity irreps, parity-safe for full-parity irreps)
            delta = _gated_activation(delta, self.output_irreps)
        # update "linear": delta = z stays linear in (m, h); the stratum readout below is the
        # only non-linearity (readout_mode "gated")
        if self.gamma is not None:
            delta = _scale_irreps(delta, self.gamma[...])

        # residual: "none" (original) -> next-layer features are the gated output;
        # "identity" / "identity_cum" -> h_{t+1} = h_t + delta_t (pre-norm ResNet; the
        # residual stream is the cumulative cluster expansion).  Layer 0 has different
        # input/output irreps and keeps the original behaviour.  The stratum readout
        # reads the increment delta_t ("identity") or the cumulative h_{t+1} ("identity_cum").
        stream_delta = delta if drop_scale is None else delta * drop_scale[:, None]
        if self.residual == "chebyshev":
            # three-term recurrence T_{t+1} = 2 x T_t - T_{t-1} (review §10 A): the strata become
            # orthogonal (Chebyshev-type) polynomials in the atomic basis instead of monomials.
            # delta plays the role of x T_t; prev_features = h_{t-1} (None for layers 0, 1).
            if prev_features is not None and self.input_irreps == self.output_irreps:
                new_features = 2.0 * stream_delta - prev_features
            else:
                new_features = stream_delta
            readout_in = new_features
        elif self.residual == "neumann":
            # truncated resolvent (review §12.4 C): h_{t+1} = u_0 + M h_t, i.e. with a shared operator
            # h_{t+1} = sum_{k<=t} M^k u_0 -> (1 - M)^{-1} u_0; prev_features = u_0 (None for layer 0)
            new_features = stream_delta + prev_features if prev_features is not None else stream_delta
            readout_in = new_features
        else:
            if self.residual != "none" and self.input_irreps == self.output_irreps:
                new_features = features + stream_delta
            else:
                new_features = stream_delta
            readout_in = new_features if self.residual == "identity_cum" else delta
        features = new_features

        node_energies = self.readout1(readout_in)
        if self.readout_mode == "gated":
            node_energies = e3nn.gate(
                node_energies,
                even_act=jax.nn.silu,
                odd_act=jnp.tanh,
                even_gate_act=jax.nn.silu,
            )
        if self.readout_species:
            node_energies = self.readout(species, node_energies)
        else:
            node_energies = self.readout(node_energies)
        if sym_energy is not None:
            node_energies = node_energies + sym_energy

        return node_energies, features


class RACE(nnx.Module):
    """RACE model for predicting energies and forces.

    Args:
        n_species: Number of atom species
        lmax: Maximum angular momentum for spherical harmonics
        cutoff: Radial cutoff distance
        hidden_irreps: Hidden feature dimension
        n_layers: Number of convolution layers
        radial_basis_size: Number of radial basis functions
        radial_mlp_size: Hidden size of radial MLP
        radial_mlp_layers: Number of radial MLP layers
        radial_polynomial_p: Polynomial cutoff order
        mlp_init_scale: Initialization scale for MLPs
        shift: Energy shift
        scale: Energy scale
        avg_n_neighbors: Average number of neighbors for normalization
        atom_energies: Isolated atom energies for each species
        l_train: Whether in training mode
        periodic: Whether system is periodic
        rngs: Flax NNX random number generator
    """

    def __init__(
        self,
        n_species: int,
        lmax: int = 3,
        cutoff: float = 5.0,
        hidden_irreps: str = "64x0e + 64x1o + 64x2e",
        n_layers: int = 5,
        radial_basis_size: int = 8,
        radial_mlp_size: int = 64,
        radial_mlp_layers: int = 3,
        radial_polynomial_p: float = 2.0,
        mlp_init_scale: float = 4.0,
        shift: float = 0.0,
        scale: float = 1.0,
        avg_n_neighbors: float = 25.0,
        atom_energies: Optional[Sequence[float]] = None,
        l_train: bool = True,
        periodic: bool = True,
        x_mode: str = "species",
        cov_mode: str = "scalar6",
        tp_backend: str = "e3nn",
        pair_term: bool = False,
        stratum_gain: str = "uniform",
        readout_mode: str = "linear",
        readout_species: bool = False,
        residual: str = "none",
        layer_scale: Optional[float] = None,
        noise_head: bool = False,
        layer_lmax: Optional[Sequence[int]] = None,
        layer_cutoff: Optional[Sequence[float]] = None,
        radial_species_scale: bool = False,
        radial_slater: bool = False,
        species_radii: Optional[Sequence[float]] = None,
        embed_mode: str = "onehot",
        species_descriptors: Optional[Sequence[Sequence[float]]] = None,
        radial_shells: bool = False,
        species_shell_radii: Optional[Sequence[Sequence[float]]] = None,
        radial_shell_scale: bool = False,
        in_layer_order: int = 1,
        shell_sk_gate: bool = False,
        strata_resum: Optional[str] = None,
        pair_repulsion: Union[bool, str] = False,
        pair_repulsion_scale: float = 1.0,
        pair_repulsion_switch: Optional[float] = None,
        pair_repulsion_switch_width: float = 0.1,
        species_z: Optional[Sequence[int]] = None,
        radial_gaussians: Optional[Dict] = None,
        update: str = "gate",
        message_norm: str = "layer",
        in_layer_order_layers: Optional[int] = None,
        radial_species_mix: Optional[str] = None,
        x_norm: Optional[str] = None,
        feature_norm: Optional[str] = None,
        tie_operator: bool = False,
        bond_order: Optional[str] = None,
        bond_order_p: Optional[float] = None,
        bond_order_shape: bool = True,
        sym_contraction: bool = False,
        sym_init_scale: float = 0.1,
        sym_method: Optional[str] = None,
        sym_target: str = "features",
        x_basis_lmax: Optional[int] = None,
        x_basis_layers: Optional[Sequence[int]] = None,
        radial_prior: Optional[Dict] = None,
        angular_prior: Optional[Dict] = None,
        x_readout: str = "mlp",
        x_readout_norm: bool = True,
        f_readout: bool = True,
        force_encoding: bool = False,
        dens: bool = False,
        dens_seed: int = 0,
        *,
        rngs: nnx.Rngs,
    ):
        """
        update        : "gate" (original: next-layer features = gated layer output) or "linear"
                        (review §11 A, BOTNet-style body-ordered update: next-layer features are the
                        ungated z = W_2 m + S_z h; needs readout_mode="gated", the only non-linearity).
        message_norm  : "layer" (original: separated layer norm of h_j before the message TP and
                        sum / sqrt(avg_n_neighbors)) or "avg" (review §11 B: no feature normalisation,
                        sum / avg_n_neighbors, as BOTNet/MACE; keeps the features body-ordered).
        x_norm        : None | "rms" | "hill" | "spectral"; x_mode "basis": bound the factor
                        x_i = Linear_t(A_i) before the product (review §12.3: the unnormalised product
                        compounds |A_i| per layer).  "rms": divide by its per-node RMS (over all
                        components; unit size in every layer, the local degree |x_i| is lost);
                        "hill": rms times a bounded learned degree gate 2 sigmoid(a_{t,Z} log r_i + b_{t,Z})
                        (init a = b = 0 = "rms"); "spectral": no per-node rescaling, every irrep block of
                        Linear_t divided by its operator norm (global, keeps |A_i|; review §12.5).
        feature_norm  : None | "rms"; divide the features passed to the next layer by their per-node
                        RMS (the stratum readout still reads the unnormalised layer output).
        tie_operator  : layers 1..n_layers-1 share one operator (radial MLP, message / product /
                        update linears, skip, layer scale) and one x projection; each layer keeps its
                        own stratum readout.  With update="linear" the shared layer is a linear map
                        h -> M(R) h for fixed geometry, so the strata are polynomials p_t(M) u_0
                        (residual "chebyshev": T_t(M) u_0, review §12.4 Chebyshev-RACE).
        bond_order    : None | "power" | "log"; one homogeneous bond-order scale stratum (moments
                        theorem, review §12.7): E_bo = -sum_i s_Z phi(b_i) (1 + g(q_i)) with the
                        second-moment width b_i = (eps + sum_j beta_ZZ'(r_ij)^2)^(1/2),
                        beta_ZZ'(r) = sum_k softplus(c_k^ZZ') exp(-alpha_k r) f_cut(r), alpha = 1, 2, 4 / A
                        (positive, monotonically decaying); phi = b^p ("power", Finnis-Sinclair at
                        p = 1/2, tight-binding degree 1 at p = 1) or log b ("log", the unbounded control);
                        g = small MLP of the scale-free shape invariants q_i (per-layer RMS-normalised
                        0e features, concatenated), last layer zero -> starts as phi(b).  s_Z init 0, so
                        the model starts identical to the one without the term.
        bond_order_p  : None (learned, p = 0.5 + min(softplus(pi), 1), init 1) or a fixed exponent.
        bond_order_shape : include the shape factor (1 + g(q_i)) (default True).
        x_mode "basis": (review §11 C) x_i = Linear_t(A_i) with A_i the layer-0 atomic basis
                        (the aggregated layer-0 message, equivariant, ungated, computed once); every
                        layer forms the channelwise product m_i^(t) (x) x_i -> one atom-centred factor
                        per layer (body order +2 per layer, strata stay pure under a linear update).
        pair_repulsion: fixed-form Born–Mayer repulsion stratum  E_rep = 1/2 sum_ij A_ij exp(-r_ij/rho_ij) f_cut(r_ij),
                        A_ij = exp(a_i + a_j) (a_i learnable, init 0 -> 1 eV), rho_ij = rho_i + rho_j with
                        rho_i = softplus(p_i) learnable, init 0.2 * species_radii + 0.05 A.  Anchors the
                        repulsive wall physically so the strata only learn corrections.
                        "zbl": fixed (parameter-free) Ziegler–Biersack–Littmark universal screened Coulomb
                        repulsion  E = 1/2 sum_ij 14.3996 Z_i Z_j / r phi(r / a_ij) f_cut(r)  with
                        a_ij = 0.4685 / (Z_i^0.23 + Z_j^0.23) A (needs species_z), times pair_repulsion_scale;
                        a prior for the short-range wall outside the training distances (dihedral collision
                        analysis 2026-11-02).  pair_repulsion_switch = c: multiply by the smooth switch
                        sigmoid((r - c (R_i + R_j)) / w), w = pair_repulsion_switch_width (A, default 0.1), R = covalent radii, so the prior
                        acts on non-bonded contacts only and bonded pairs are left to the strata.
        radial_gaussians : {"centers": [r_1..r_K] (A), "width": sigma (A)}: K Gaussian channels
                        exp(-(r - r_k)^2 / 2 sigma^2) f_cut(r) appended to the radial MLP input
                        (radial resolution concentrated near the bond lengths).
        embed_mode    : "onehot" (original: Linear on the one-hot species), "descriptor" (MLP on a fixed
                        per-element descriptor row, shared over elements), "both" (sum of the two).
        species_descriptors : (n_species, m) fixed descriptor rows (standardised), needed unless "onehot".
        radial_shells : shell-resolved radial basis: Slater envelopes exp(-r / (rho_a(i) + rho_b(j)))
                        for every pair of valence shells (a, b in s, p, d, f) of the two atoms, times the
                        cutoff, appended to the radial MLP input (the mixing is learned once, shared by
                        all element pairs).  Non-periodic only.
        species_shell_radii : (n_species, n_shells) <r> of the outermost shell per l in A, 0 = absent.
        radial_shell_scale : learnable per-shell-pair length scale s_ab (softplus, init 1) on the
                        envelope, exp(-r / (s_ab (rho_a(i) + rho_b(j)))): keeps the physical
                        initialisation, lets a pair sharpen or widen (review §8, heavy-atom bonds).
        in_layer_order : 1 (RACE: one tensor product per layer, body order +1 per layer), 2
                        (adds the channelwise symmetric square m_i (x) m_i of the aggregated
                        message inside the layer; legacy r13 path unless sym_contraction) or 3
                        (MACE symmetric contraction of degrees 2 and 3 with species weights,
                        review §11.11; body order +3 per layer, "coarse RACE").
        in_layer_order_layers : None (all layers) or the number of leading layers that use
                        in_layer_order; the remaining layers use order 1 (hybrid: low orders by
                        products, higher orders by depth).
        sym_contraction : use the cue symmetric-contraction path also for in_layer_order 2
                        (control for order 3 that differs only in the degrees).
        sym_init_scale : init of the learnable per-degree gain of the contraction terms (default 0.1).
        sym_target : "features" (B_i added to the layer increment, MACE-like; unstable when stacked
                        over many layers) or "readout" (B_i feeds only the stratum energy through its
                        own gated readout; propagated features stay nu = 1).
        x_basis_lmax : x_mode "basis": only the l <= lmax components of the atomic basis A_i enter
                        the x factor (0 -> per-channel scalar gating by the local density, no
                        angular product at the centre; review §11.12).
        x_basis_layers : x_mode "basis": indices of the layers that use A_i; the other layers use
                        the per-species x (the v7_ab behaviour).
        radial_prior : {"type": "algebraic"|"exponential"|"gaussian", "sigma": s, "p": p}: regularity
                        prior on the radial functions (Darby et al. 2026, arXiv:2601.15072) as a fixed
                        rescaling of Bessel channel n (n = 1..N) by 1/gamma(n), gamma = (1+s n)^p |
                        exp(s n) | exp(s^2 n^2); the shell-envelope channels are not rescaled.  With a
                        uniform L2 on the radial MLP's first layer (training radial_l2) this is the
                        Tikhonov prior gamma(n)^2 |c_n|^2.  "sigma" may be a list with one value per
                        layer.
        angular_prior : {"sigma_l": s}: Y_l edge attributes scaled by exp(-s^2 l^2) (static), the
                        angular counterpart of radial_prior (review §11.18).
        x_readout : "mlp" (original: per-layer MLP readout of x_feats added to every stratum), "off"
                        (no x energy), "layer0" (MLP readout in layer 0 only), "linear" (species-indexed
                        linear readout of the 0e part of x_feats; with x_mode "basis" an exact 2-body
                        stratum) — review §11.16.
        x_readout_norm : layer norm on the x readout input (original True); False keeps the magnitude.
        f_readout : per-layer MLP readout of the layer *input* features added to the stratum (original
                        True; redundant with the previous layer's readout, constant for layer 0).
        force_encoding : DeNS (Liao et al., TMLR 2024): layer-0 input gets a 1o block Linear(F_ref) in
                        the denoising pass (``denoise``), zeros in the energy pass; needs noise_head.
        shell_sk_gate : Slater–Koster selection rule (review §10 B): per layer and spherical-harmonic
                        degree l, Y_l(r_ij) is scaled by 1 + sum_ab c_{ab,l} env_ab(r) over the shell
                        pairs (a, b) allowed by |l_a-l_b| <= l <= l_a+l_b, l_a+l_b+l even (two-centre
                        integral expansion); c init 0 (= v6 at init).  Needs radial_shells.
        strata_resum  : None | "aitken" (review §10 C): Aitken/Shanks geometric-tail estimate of the
                        body-order series from the last two layer strata, saturated (|rho| < 0.5) and
                        added as an extra stratum with learnable weight lambda (init 0).
        layer_scale   : LayerScale init (e.g. 0.1): learnable per-channel gain on each layer's
                        increment (layers >= 1).  None = off.
        noise_head    : add an equivariant 1o head on the final features (Noisy-Nodes denoising
                        auxiliary task, see ``denoise``).
        dens          : faithful DeNS (Liao et al., TMLR 2024; audit docs/RACE_DeNS_audit_2026-10-29.md):
                        nodes['dens_force'] (reference forces on the corrupted atoms, 0 elsewhere) is
                        encoded as (|F|/sqrt 3) Y^l(F^), l <= min(3, lmax), by a bias-free Linear and
                        *added* to the node features after layer 0 (absent key -> the model is exactly the
                        model without DeNS); ``dens_noise`` = gated equivariant MLP noise head on the final
                        features.  The DeNS modules use their own RNG stream (dens_seed), so the energy
                        model initialises as without DeNS.  Replaces force_encoding / noise_head.
        layer_lmax    : per-layer max l of the *new* stratum information (hyperbolic-cross /
                        sparse-grid truncation, e.g. [3,3,2,2,1]).  None = lmax for all.
        layer_cutoff  : per-layer cutoff (A) of the message radial envelope (<= cutoff), e.g.
                        [5,5,4.5,4,3.5]: higher body order = shorter range.  Non-periodic only.
        radial_species_scale : Bessel argument r * alpha_ij, alpha_ij = 2 rho_bar / (rho_i + rho_j)
                        with learnable per-species radii rho (init species_radii, A); the cutoff
                        envelope stays on r.  (ACE1 species-pair distance transform.)
        radial_slater : extra Slater factor exp(-zeta_ij r), zeta_ij = softplus(z_i + z_j) learnable.
        stochastic depth is driven by ``data.globals['stratum_keep']`` (n_graphs, n_layers)
                        when present (training step inserts it); absent = no dropping.
        residual      : "none" (original: next-layer features = gated layer output),
                        "identity" (h_{t+1} = h_t + delta_t, stratum readout reads the increment
                        delta_t) or "identity_cum" (same stream, readout reads h_{t+1}).
                        Pre-norm ResNet; layer 0 unchanged (different irreps).  "chebyshev":
                        h_{t+1} = 2 M_t h_t - h_{t-1}; "neumann": h_{t+1} = u_0 + M_t h_t with
                        u_0 = layer-0 output (truncated resolvent; review §12.4); both read h_{t+1}.
        x_mode        : "species" (per-species scalar x_i, original), "node_scalar" (scalar
                        projection of the receiving node's 0e features) or "node_irreps"
                        (x_i = Linear(h_i) with the full hidden irreps; the layer then forms the
                        channelwise equivariant product  m_i (x) x_i  -- the recursive ACE
                        product B^(t+1) = B^(t) (x) A_i -- instead of a scalar scaling).
        readout_mode  : per-layer stratum readout, "linear" (original) or "gated"
                        (Linear -> silu -> Linear on the scalar features; see RACEConvolution).
        readout_species: per-species weights in the last stratum readout Linear
                        (ACE coefficients c_nu^{Z_i}).  Default off (original behaviour).
        hidden_irreps : may contain both parities ("full parity", e.g.
                        "128x0e+128x0o+128x1o+128x1e+128x2e+128x2o+128x3o+128x3e") so that no
                        CG path of the per-layer tensor product is dropped.
        pair_term     : add an explicit 2-body stratum  E_2 = 1/2 sum_ij c[s_i,s_j] * MLP(R(r_ij))
                        (R = Bessel basis x polynomial cutoff), i.e. the first term of the
                        cluster expansion made explicit.  Default off (original behaviour).
        stratum_gain  : static multiplier on the energy of layer i (i = 0..n_layers-1):
                        "uniform" -> 1 (original), "inverse" -> 1/(i+1), "inverse_sq" -> 1/(i+1)^2,
                        "jackson" -> the Jackson kernel g_i of N = n_layers moments (KPM damping,
                        Weisse et al. RMP 78, 275: the body-order analogue of the radial prior).
                        Lower strata (low body order) explain first, deeper strata correct.
        """
        if x_mode not in ("species", "node_scalar", "node_irreps", "basis"):
            raise ValueError(
                f"x_mode must be 'species', 'node_scalar', 'node_irreps' or 'basis', got {x_mode!r}"
            )
        self.update = update
        self.message_norm = message_norm
        if x_norm not in (None, "rms", "hill", "spectral"):
            raise ValueError(f"x_norm must be None, 'rms', 'hill' or 'spectral', got {x_norm!r}")
        if feature_norm not in (None, "rms"):
            raise ValueError(f"feature_norm must be None or 'rms', got {feature_norm!r}")
        if x_norm is not None and x_mode != "basis":
            raise ValueError("x_norm needs x_mode='basis'")
        self.x_norm = x_norm
        self.feature_norm = feature_norm
        if cov_mode not in ("scalar6", "irreps"):
            raise ValueError(f"cov_mode must be 'scalar6' or 'irreps', got {cov_mode!r}")

        # Store static configuration
        self.lmax = lmax
        self.cutoff = cutoff
        self.periodic = periodic
        self.n_species = n_species
        self.radial_basis_size = radial_basis_size
        self.radial_polynomial_p = radial_polynomial_p
        self.l_train = l_train
        self.x_mode = x_mode
        if x_readout not in ("mlp", "off", "layer0", "linear"):
            raise ValueError(f"x_readout must be mlp|off|layer0|linear, got {x_readout!r}")
        self.x_readout_mode = str(x_readout)
        self.use_f_readout = bool(f_readout)
        self.x_basis_lmax = None if x_basis_lmax is None else int(x_basis_lmax)
        self.x_basis_layers = None if x_basis_layers is None else tuple(int(i) for i in x_basis_layers)
        self.cov_mode = cov_mode
        self.tp_backend = tp_backend
        self.pair_term = pair_term
        if stratum_gain not in ("uniform", "inverse", "inverse_sq", "jackson"):
            raise ValueError(f"unknown stratum_gain {stratum_gain!r}")
        self.stratum_gain = stratum_gain
        self.readout_mode = readout_mode
        self.readout_species = readout_species
        self.residual = residual
        self.n_layers = n_layers
        self.use_noise_head = noise_head
        if layer_lmax is not None and len(layer_lmax) != n_layers:
            raise ValueError("layer_lmax must have n_layers entries")
        if layer_cutoff is not None:
            if len(layer_cutoff) != n_layers:
                raise ValueError("layer_cutoff must have n_layers entries")
            if periodic:
                raise NotImplementedError("layer_cutoff is implemented for non-periodic systems only")
            if max(layer_cutoff) > cutoff + 1e-6:
                raise ValueError("layer_cutoff entries must be <= cutoff")
        self.layer_lmax = None if layer_lmax is None else tuple(int(l) for l in layer_lmax)
        self.layer_cutoff = None if layer_cutoff is None else tuple(float(c) for c in layer_cutoff)
        self.radial_species_scale = radial_species_scale
        self.radial_slater = radial_slater
        if (radial_species_scale or radial_slater) and periodic:
            raise NotImplementedError("radial_species_scale/radial_slater: non-periodic only")
        if radial_species_scale:
            if species_radii is None or len(species_radii) != n_species:
                raise ValueError("radial_species_scale needs species_radii (one per species, A)")
            self.species_rho = nnx.Param(jnp.asarray(species_radii, dtype=jnp.float32))
        if radial_slater:
            # softplus(z_i + z_j) ~ 0.1 1/A initially (weak localisation; learnable)
            z0 = float(jnp.log(jnp.expm1(0.1)) / 2.0)
            self.species_zeta = nnx.Param(jnp.full((n_species,), z0, dtype=jnp.float32))
        # orbital-radial species embedding / shell-resolved radial basis (review §8).  The
        # descriptor rows and shell radii are fixed constants (tuples -> part of the graphdef).
        if embed_mode not in ("onehot", "descriptor", "both"):
            raise ValueError(f"embed_mode must be 'onehot', 'descriptor' or 'both', got {embed_mode!r}")
        self.embed_mode = embed_mode
        self.radial_shells = radial_shells
        # radial_shells with periodic cells: edge distances in node_energies include the image shifts
        # (review §12.4, tests/test_periodic.py)
        if embed_mode != "onehot":
            if species_descriptors is None or len(species_descriptors) != n_species:
                raise ValueError("embed_mode != 'onehot' needs species_descriptors (one row per species)")
            self.species_descriptors = tuple(tuple(float(v) for v in row) for row in species_descriptors)
            self.n_descriptors = len(self.species_descriptors[0])
        if radial_shells:
            if species_shell_radii is None or len(species_shell_radii) != n_species:
                raise ValueError("radial_shells needs species_shell_radii (one row per species, A)")
            self.species_shell_radii = tuple(tuple(float(v) for v in row) for row in species_shell_radii)
            self.n_shells = len(self.species_shell_radii[0])
        self.radial_shell_scale = bool(radial_shell_scale)
        if radial_shell_scale:
            if not radial_shells:
                raise ValueError("radial_shell_scale needs radial_shells=True")
            # softplus(raw) = 1 at init -> identical to the fixed envelopes at initialisation
            self.shell_scale_raw = nnx.Param(jnp.full((self.n_shells, self.n_shells),
                                                      math.log(math.e - 1.0), jnp.float32))
        self.shell_sk_gate = bool(shell_sk_gate)
        if shell_sk_gate:
            if not radial_shells:
                raise ValueError("shell_sk_gate needs radial_shells=True")
            mask = []
            for la in range(self.n_shells):
                for lb in range(self.n_shells):
                    mask.append(tuple(1.0 if (abs(la - lb) <= l <= la + lb and (la + lb + l) % 2 == 0) else 0.0
                                      for l in range(lmax + 1)))
            self.sk_mask = tuple(mask)                                   # (n_shells**2, lmax+1), static
            self.sk_coeff = nnx.Param(jnp.zeros((n_layers, self.n_shells ** 2, lmax + 1), jnp.float32))
        self.strata_resum = strata_resum
        if strata_resum is not None:
            if strata_resum != "aitken":
                raise ValueError(f"strata_resum must be None or 'aitken', got {strata_resum!r}")
            if n_layers < 2:
                raise ValueError("strata_resum needs at least 2 layers")
            self.resum_lambda = nnx.Param(jnp.zeros((), jnp.float32))
        n_radial_extra = self.n_shells ** 2 if radial_shells else 0
        # bond-length Gaussian radial channels (review §7.14 lever 2)
        self.radial_gaussians = None
        if radial_gaussians is not None:
            centers = tuple(float(c) for c in radial_gaussians["centers"])
            width = float(radial_gaussians["width"])
            if not centers or width <= 0:
                raise ValueError("radial_gaussians needs non-empty 'centers' and a positive 'width'")
            self.radial_gaussians = (centers, width)
            n_radial_extra += len(centers)
        # fixed-form Born–Mayer repulsion stratum (review §7.14 lever 1)
        self.pair_repulsion = pair_repulsion
        self.pair_repulsion_scale = float(pair_repulsion_scale)
        self.pair_repulsion_switch = None if pair_repulsion_switch is None else float(pair_repulsion_switch)
        self.pair_repulsion_switch_width = float(pair_repulsion_switch_width)
        self.species_z = None if species_z is None else tuple(int(z) for z in species_z)
        self.species_radii_t = None if species_radii is None else tuple(float(r) for r in species_radii)
        if pair_repulsion:
            # periodic too: r_norm below is built from the same minimum-image edge vectors as the messages
            if species_radii is None or len(species_radii) != n_species:
                raise ValueError("pair_repulsion needs species_radii (one per species, A)")
            if pair_repulsion in ("zbl", "uff"):
                if self.species_z is None or len(self.species_z) != n_species:
                    raise ValueError(f"pair_repulsion {pair_repulsion!r} needs species_z (one atomic number per species)")
                if pair_repulsion == "uff":
                    missing = [z for z in self.species_z if z not in UFF_VDW]
                    if missing:
                        raise ValueError(f"pair_repulsion 'uff': no UFF vdW parameters for Z={missing}")
            elif pair_repulsion is True:
                rho0 = 0.2 * jnp.asarray(species_radii, dtype=jnp.float32) + 0.05
                self.rep_loga = nnx.Param(jnp.zeros((n_species,), jnp.float32))      # A_ij = exp(a_i + a_j)
                self.rep_rho_raw = nnx.Param(jnp.log(jnp.expm1(rho0)))                # rho_i = softplus(.)
            else:
                raise ValueError(f"pair_repulsion must be False, True (Born–Mayer), 'zbl' or 'uff', got {pair_repulsion!r}")

        # Store trainable scale parameters
        self.shift = nnx.Param(jnp.array(shift))
        self.scale = nnx.Param(jnp.array(scale))

        # Non-trainable constants.  Stored as plain Python objects so they are
        # part of the (hashable) graphdef and never receive gradients/updates.
        self.avg_n_neighbors = float(avg_n_neighbors)
        if atom_energies is not None:
            atom_energies = tuple(float(e) for e in atom_energies)
            if len(atom_energies) != n_species:
                raise ValueError(
                    f"len(atom_energies)={len(atom_energies)} != n_species={n_species}"
                )
        else:
            atom_energies = tuple(0.0 for _ in range(n_species))
        self.atom_energies = atom_energies

        self.hidden_irreps = e3nn.Irreps(hidden_irreps)
        embed_irreps = self.hidden_irreps.filter(keep="0e")
        # force_encoding (DeNS, Liao et al. 2024, review §11.17): the layer-0 input carries an extra
        # 1o block = Linear(reference forces) during the denoising pass and zeros otherwise, so the
        # denoising target of a non-equilibrium structure is well defined; the energy / force pass
        # never sees reference forces.
        self.force_encoding = bool(force_encoding)
        input_irreps = (embed_irreps + e3nn.Irreps(f"{embed_irreps.num_irreps}x1o")
                        if self.force_encoding else embed_irreps)
        self.embed_irreps = embed_irreps
        # "node_irreps" / "basis": x_i carries the full hidden irreps (equivariant product in the layer)
        x_irreps = self.hidden_irreps if x_mode in ("node_irreps", "basis") else e3nn.Irreps("1x0e")
        # x_basis_lmax: 0/1 mask over the mul_ir layout of the atomic basis (static tuple, not a
        # Param; nnx treats a Python tuple as static metadata)
        self.x_basis_mask = None
        if self.x_basis_lmax is not None:
            self.x_basis_mask = tuple(
                float(ir.l <= self.x_basis_lmax) for mul, ir in self.hidden_irreps for _ in range(mul * ir.dim))

        # x_mode == "species": a single per-species scalar (original behaviour)
        self.x_linear = e3nn_nnx_Linear(
            irreps_in=input_irreps,
            irreps_out=x_irreps,
            linear_type="indexed",
            num_indexed_weights=n_species,
            force_irreps_out=True,
            rngs=rngs,
        )

        self.sh_irreps = e3nn.s2_irreps(lmax)

        self.species_embedding = e3nn_nnx_Linear(
            irreps_in=e3nn.Irreps(f"{n_species}x0e"),
            irreps_out=embed_irreps,
            rngs=rngs,
        )
        if self.force_encoding:
            self.force_enc = e3nn_nnx_Linear(
                irreps_in=e3nn.Irreps("1x1o"), irreps_out=e3nn.Irreps(f"{embed_irreps.num_irreps}x1o"),
                force_irreps_out=True, rngs=rngs,
            )
        if self.embed_mode != "onehot":
            # fixed element descriptor -> hidden scalars; one MLP shared by all elements
            self.descriptor_mlp = MLP(
                sizes=[self.n_descriptors, radial_mlp_size, embed_irreps.num_irreps],
                activation=jax.nn.silu, use_bias=True, init_scale=1.0, rngs=rngs,
            )

        radial_scales = [None] * n_layers
        self.radial_prior = None if radial_prior is None else dict(radial_prior)
        if radial_prior:
            rp_type = str(radial_prior.get("type", "algebraic"))
            sig = radial_prior.get("sigma", 0.5)
            # a single sigma or one per layer (review §11.18: deeper layers can be smoothed more)
            sigmas = [float(sig)] * n_layers if not isinstance(sig, (list, tuple)) else [float(x) for x in sig]
            if len(sigmas) != n_layers:
                raise ValueError(f"radial_prior sigma list must have n_layers={n_layers} entries, got {len(sigmas)}")
            p_rp = float(radial_prior.get("p", 2.0))
            n_idx = range(1, radial_basis_size + 1)
            for t, s_rp in enumerate(sigmas):
                if rp_type == "algebraic":
                    gam = [(1.0 + s_rp * n) ** p_rp for n in n_idx]
                elif rp_type == "exponential":
                    gam = [math.exp(s_rp * n) for n in n_idx]
                elif rp_type == "gaussian":
                    gam = [math.exp(s_rp ** 2 * n ** 2) for n in n_idx]
                else:
                    raise ValueError(f"radial_prior type must be algebraic|exponential|gaussian, got {rp_type!r}")
                radial_scales[t] = tuple(1.0 / g for g in gam) + (1.0,) * n_radial_extra
        self.radial_scale = radial_scales[0]
        self.radial_scales = tuple(radial_scales)
        # angular regularity prior (review §11.18): Y_l components scaled by exp(-sigma_l^2 l^2)
        # (static); the analogue of Darby's angular-degree factor for the spherical-harmonic edge
        # attributes, applied once to sh so that the atomic basis and every layer see it
        self.angular_prior = None if angular_prior is None else dict(angular_prior)
        self.angular_scale = None
        if angular_prior:
            s_l = float(angular_prior.get("sigma_l", 0.3))
            self.angular_scale = tuple(math.exp(-(s_l ** 2) * (ir.l ** 2))
                                       for mul, ir in self.sh_irreps for _ in range(mul * ir.dim))

        layers = []
        x_readouts = []
        f_readouts = []
        x_projs = []
        for i in range(n_layers):
            layer = RACEConvolution(
                input_irreps=input_irreps if i == 0 else self.hidden_irreps,
                output_irreps=self.hidden_irreps,
                x_irreps=x_irreps,
                sh_irreps=self.sh_irreps,
                n_species=n_species,
                radial_basis_size=radial_basis_size,
                radial_mlp_size=radial_mlp_size,
                radial_mlp_layers=radial_mlp_layers,
                radial_extra=n_radial_extra,
                mlp_init_scale=mlp_init_scale,
                avg_n_neighbors=avg_n_neighbors,
                tp_backend=tp_backend,
                readout_mode=readout_mode,
                readout_species=readout_species,
                residual=residual,
                layer_scale=(layer_scale if i >= 1 else None),
                msg_lmax=(None if layer_lmax is None else int(layer_lmax[i])),
                in_layer_order=(in_layer_order if (in_layer_order_layers is None
                                                   or i < int(in_layer_order_layers)) else 1),
                update=update,
                message_norm=message_norm,
                sym_contraction=sym_contraction,
                sym_init_scale=sym_init_scale,
                sym_target=sym_target,
                radial_scale=radial_scales[i],
                radial_species_mix=radial_species_mix,
                rngs=rngs,
            )
            if sym_method is not None and getattr(layer, "sym_contraction", False):
                layer.sym_method = str(sym_method)      # debug override: "naive" (pure JAX) | "uniform_1d"
            layers.append(layer)

            # FIX: Use derived readout_hidden_irreps instead of hardcoded "64x0e"
            readout_hidden_irreps = self.hidden_irreps.filter(keep="0e")
            if x_readout == "mlp" or x_readout == "linear" or (x_readout == "layer0" and i == 0):
                x_readouts.append(RACEXReadout(
                    hidden_irreps=readout_hidden_irreps, #e3nn.Irreps("64x0e"),
                    output_irreps=e3nn.Irreps("0e"),
                    x_irreps=x_irreps,
                    n_species=n_species,
                    avg_n_neighbors=avg_n_neighbors,
                    rngs=rngs,
                    norm=x_readout_norm,
                    linear=(x_readout == "linear"),
                ))

            if f_readout:
                f_readouts.append(RACEXReadout(
                    hidden_irreps=readout_hidden_irreps, #e3nn.Irreps("64x0e"),
                    output_irreps=e3nn.Irreps("0e"),
                    x_irreps=input_irreps if i == 0 else self.hidden_irreps,
                    n_species=n_species,
                    avg_n_neighbors=avg_n_neighbors,
                    rngs=rngs,
                ))

            # x_mode == "node_scalar": project the receiving node's current
            # scalar features (after layer norm) onto x_irreps, per layer.
            if x_mode == "node_scalar":
                layer_input_irreps = input_irreps if i == 0 else self.hidden_irreps
                x_projs.append(
                    e3nn_nnx_Linear(
                        irreps_in=layer_input_irreps.filter(keep="0e"),
                        irreps_out=x_irreps,
                        biases=True,
                        force_irreps_out=True,
                        rngs=rngs,
                    )
                )
            elif x_mode == "node_irreps":
                # x_i = Linear(h_i) with the full hidden irreps (layer 0: only 0e is
                # available, the l>0 parts are zero -> per-channel scaling, 2-body stratum)
                layer_input_irreps = input_irreps if i == 0 else self.hidden_irreps
                x_projs.append(
                    e3nn_nnx_Linear(
                        irreps_in=layer_input_irreps,
                        irreps_out=x_irreps,
                        biases=True,
                        force_irreps_out=True,
                        rngs=rngs,
                    )
                )
            elif x_mode == "basis":
                # x_i = W_t A_i: a per-layer linear mix of the fixed layer-0 atomic basis
                # (no bias: x stays exactly degree 1 in the density)
                x_projs.append(
                    e3nn_nnx_Linear(
                        # A_i has the layer-0 message irreps (= hidden irreps for a scalar layer-0 input;
                        # extra path blocks with the DeNS force-encoding 1o input)
                        irreps_in=layers[0].tp_irreps,
                        irreps_out=x_irreps,
                        biases=False,
                        force_irreps_out=True,
                        rngs=rngs,
                    )
                )

        self.tie_operator = bool(tie_operator)
        if bond_order not in (None, "power", "log"):
            raise ValueError(f"bond_order must be None, 'power' or 'log', got {bond_order!r}")
        if bond_order is not None and periodic:
            raise NotImplementedError("bond_order: non-periodic only")
        self.bond_order = bond_order
        self.bond_order_p = None if bond_order_p is None else float(bond_order_p)
        self.bond_order_shape = bool(bond_order_shape)
        self.bo_alpha = (1.0, 2.0, 4.0)
        if bond_order is not None:
            # softplus(c) = 1 at init (symmetrised over the species pair in the forward pass)
            self.bo_c = nnx.Param(jnp.full((n_species, n_species, len(self.bo_alpha)),
                                           math.log(math.expm1(1.0)), jnp.float32))
            self.bo_s = nnx.Param(jnp.zeros((n_species,), jnp.float32))
            if bond_order == "power" and self.bond_order_p is None:
                self.bo_pi = nnx.Param(jnp.asarray(math.log(math.expm1(0.5)), jnp.float32))   # p = 1
            if self.bond_order_shape:
                n0e = self.hidden_irreps.filter(keep="0e").dim
                self.bo_shape_mlp = MLP(sizes=[n_layers * n0e, 32, 1], activation=jax.nn.silu,
                                        use_bias=True, rngs=rngs)
                last = self.bo_shape_mlp.layers[-1]
                last.weights[...] = jnp.zeros_like(last.weights[...])
        if self.tie_operator:
            if n_layers < 3:
                raise ValueError("tie_operator needs n_layers >= 3 (layers 1.. share one operator)")
            if layer_lmax is not None and len(set(int(l) for l in layer_lmax[1:])) > 1:
                raise ValueError("tie_operator needs one layer_lmax for layers 1..")
            if len(set(tuple(radial_scales[i]) if radial_scales[i] is not None else None
                       for i in range(1, n_layers))) > 1:
                raise ValueError("tie_operator needs one radial scale for layers 1..")
            # every sub-module of layer 1 except its stratum readout is shared by layers 2..
            shared = [k for k, v in vars(layers[1]).items()
                      if isinstance(v, (nnx.Module, nnx.Variable)) and not k.startswith("readout")]
            for layer in layers[2:]:
                for k in shared:
                    setattr(layer, k, getattr(layers[1], k))
            if x_mode in ("basis", "node_irreps", "node_scalar"):
                x_projs[2:] = [x_projs[1]] * (n_layers - 2)
        self.layers = nnx.List(layers)
        self.x_readouts = nnx.List(x_readouts)
        self.f_readouts = nnx.List(f_readouts)
        self.x_projs = nnx.List(x_projs)
        if x_norm == "hill":
            # degree gate (a, b) per layer and species; layers 2.. use row 1 with tie_operator
            self.x_hill = nnx.Param(jnp.zeros((n_layers, 2, n_species), jnp.float32))
        if noise_head:
            self.noise_head = e3nn_nnx_Linear(
                irreps_in=self.hidden_irreps, irreps_out=e3nn.Irreps("1x1o"),
                force_irreps_out=True, rngs=rngs,
            )
        self.dens = bool(dens)
        if self.dens:
            if force_encoding or noise_head:
                raise ValueError("dens replaces force_encoding / noise_head")
            d_rngs = nnx.Rngs(dens_seed)
            l_enc = min(3, max(ir.l for _, ir in self.hidden_irreps))
            self.dens_enc_irreps = e3nn.Irreps([(1, (l, (-1) ** l)) for l in range(l_enc + 1)])
            self.dens_enc = e3nn_nnx_Linear(irreps_in=self.dens_enc_irreps, irreps_out=self.hidden_irreps,
                                            force_irreps_out=True, rngs=d_rngs)
            self.dens_head1 = e3nn_nnx_Linear(irreps_in=self.hidden_irreps, irreps_out=e3nn.Irreps("128x0e + 64x1o"),
                                              force_irreps_out=True, rngs=d_rngs)
            self.dens_head2 = e3nn_nnx_Linear(irreps_in=e3nn.Irreps("64x0e + 64x1o"),
                                              irreps_out=e3nn.Irreps("64x0e + 32x1o"), force_irreps_out=True, rngs=d_rngs)
            self.dens_head3 = e3nn_nnx_Linear(irreps_in=e3nn.Irreps("32x0e + 32x1o"), irreps_out=e3nn.Irreps("1x1o"),
                                              force_irreps_out=True, rngs=d_rngs)

        if pair_term:
            self.pair_mlp = MLP(
                sizes=[radial_basis_size + n_radial_extra] + [radial_mlp_size] * radial_mlp_layers + [1],
                activation=jax.nn.silu, use_bias=False, init_scale=mlp_init_scale, rngs=rngs,
            )
            # symmetric species-pair coefficients, start small so the pair term grows from ~0
            self.pair_coeff = nnx.Param(
                0.1 * jax.random.normal(rngs.params(), (n_species, n_species), jnp.float32)
            )

        if cov_mode == "scalar6":
            # 6 invariant scalars -> 3x3 lower-triangular Cholesky factor.
            # NOTE: not rotation-equivariant (kept for backward compatibility).
            self.frc_logvar = e3nn_nnx_Linear(
                irreps_in=hidden_irreps,
                irreps_out=e3nn.Irreps("6x0e"),
                biases=False,
                rngs=rngs,
            )
        else:
            # 1x0e (isotropic) + 1x2e (traceless symmetric) -> equivariant Sigma
            self.frc_logvar = e3nn_nnx_Linear(
                irreps_in=hidden_irreps,
                irreps_out=e3nn.Irreps("1x0e + 1x2e"),
                biases=False,
                force_irreps_out=True,
                rngs=rngs,
            )

        self.energy_logvar = e3nn_nnx_Linear(
            irreps_in=hidden_irreps,
            irreps_out=e3nn.Irreps("1x0e"),
            biases=False,
            rngs=rngs,
        )

        # Diagonal-only stress variance: 6 params (one per Voigt component)
        self.stress_logvar = e3nn_nnx_Linear(
            irreps_in=hidden_irreps,
            irreps_out=e3nn.Irreps("6x0e"),
            biases=False,
            rngs=rngs,
        )

    def node_energies(
        self,
        positions: jax.Array,
        cell: jax.Array,
        cutoff: jax.Array,
        data: jraph.GraphsTuple,
        force_input: Optional[jax.Array] = None,
    ) -> Dict:
        """Compute per-node energies.

        force_input: (n_atoms, 3) reference forces for the DeNS force encoding (denoising pass
        only); None -> the 1o input block is zero (energy / force pass, inference).

        Args:
            positions: Atomic positions of shape (n_atoms, 3)
            cell: (n_graph, 3, 3) for periodic, None for molecular
            cutoff: (n_graph,) for periodic, None for molecular
            data: Graph data structure

        Returns:
            Per-node energies of shape (n_atoms, 1)
        """
        # Compute edge features using shared utility
        features, radial_basis, sh = compute_node_edge_features(
            positions=positions,
            cell=cell,
            cutoff=cutoff,
            data=data,
            n_species=self.n_species,
            model_cutoff=self.cutoff,
            radial_basis_size=self.radial_basis_size,
            radial_polynomial_p=self.radial_polynomial_p,
            lmax=self.lmax,
            periodic=self.periodic,
        )
        if self.angular_scale is not None:
            sh = e3nn.IrrepsArray(sh.irreps, sh.array * jnp.asarray(self.angular_scale, dtype=sh.array.dtype)[None, :])

        species = data.nodes["species"]
        need_r = (self.layer_cutoff is not None or self.radial_species_scale or self.radial_slater
                  or self.radial_shells or self.radial_gaussians is not None or self.pair_repulsion
                  or self.bond_order is not None)
        if need_r:
            # edge distances (non-periodic): same convention as compute_node_edge_features;
            # the inner where keeps the gradient of sqrt finite on padded r = 0 edges
            rvec = positions[data.senders] - positions[data.receivers]
            if self.periodic:
                # same convention as utils.compute_node_edge_features: r = x_s - (x_r + S . cell)
                n_e = data.senders.shape[0]
                shift = jnp.einsum('ei,eij->ej', data.edges['Sij'],
                                   jnp.repeat(cell, data.n_edge, axis=0, total_repeat_length=n_e))
                rvec = rvec - shift
            r2 = jnp.sum(rvec ** 2, axis=-1)
            r_norm = jnp.where(r2 > 0.0, jnp.sqrt(jnp.where(r2 > 0.0, r2, 1.0)), 0.0)
            r_c = jnp.full((r_norm.shape[0],), float(self.cutoff))
        if self.radial_species_scale or self.radial_slater:
            # species-pair radial transform: basis(alpha_ij r) * f_cut(r) [* exp(-zeta_ij r)]
            x = r_norm
            if self.radial_species_scale:
                rho = self.species_rho[...]
                rho_bar = jnp.mean(rho)
                alpha = 2.0 * rho_bar / (rho[species[data.senders]] + rho[species[data.receivers]])
                x = r_norm * alpha
            radial_basis = (_bessel_safe(x, self.radial_basis_size, r_c)
                            * polynomial_cutoff(r_norm, r_c, self.radial_polynomial_p)[:, None])
            if self.radial_slater:
                z = self.species_zeta[...]
                zeta = jax.nn.softplus(z[species[data.senders]] + z[species[data.receivers]])
                radial_basis = radial_basis * jnp.exp(-zeta * r_norm)[:, None]
        shell_env = None
        if self.radial_shells:
            # shell-resolved Slater envelopes exp(-r / (rho_a(i) + rho_b(j))) for every shell pair
            # (a, b), zero where either shell is absent, times the cutoff -> extra radial channels
            rho = jnp.asarray(self.species_shell_radii, dtype=radial_basis.dtype)      # (n_species, n_shells)
            rho_i = rho[species[data.receivers]]                                       # (n_edges, n_shells)
            rho_j = rho[species[data.senders]]
            denom = rho_i[:, :, None] + rho_j[:, None, :]                              # (n_edges, a, b)
            if self.radial_shell_scale:
                # learnable per-shell-pair length scale, softplus(raw) = 1 at init
                denom = denom * jax.nn.softplus(self.shell_scale_raw[...]).astype(denom.dtype)[None]
            present = (rho_i[:, :, None] > 0.0) & (rho_j[:, None, :] > 0.0)
            env = jnp.where(present, jnp.exp(-r_norm[:, None, None] / jnp.where(present, denom, 1.0)), 0.0)
            env = env.reshape(env.shape[0], self.n_shells ** 2)         # explicit size: valid for 0 edges too
            env = env * polynomial_cutoff(r_norm, r_c, self.radial_polynomial_p)[:, None]
            shell_env = env
            radial_basis = jnp.concatenate([radial_basis, env], axis=-1)
        if self.radial_gaussians is not None:
            centers, width = self.radial_gaussians
            c = jnp.asarray(centers, dtype=radial_basis.dtype)
            gauss = jnp.exp(-0.5 * ((r_norm[:, None] - c[None, :]) / width) ** 2)
            gauss = gauss * polynomial_cutoff(r_norm, r_c, self.radial_polynomial_p)[:, None]
            radial_basis = jnp.concatenate([radial_basis, gauss], axis=-1)
        # Padded edges: jraph.pad_with_graphs puts every padding edge on the first padding node with r = 0, and
        # their radial basis (bessel(0) * f_cut(0)) and Y_0 are not zero, so they carry messages.  With thousands
        # of them (pad_mode "fixed", 2026-11-05 OAM subsets) the padding node runs to inf / nan in float32 and
        # node_mask * nan = nan reaches the graph energies.  Zero the radial and angular features of r = 0 edges;
        # real edges always have r > 0, so real outputs are unchanged.
        if not need_r:
            rvec = positions[data.senders] - positions[data.receivers]
            if self.periodic:
                n_e = data.senders.shape[0]
                rvec = rvec - jnp.einsum('ei,eij->ej', data.edges['Sij'],
                                         jnp.repeat(cell, data.n_edge, axis=0, total_repeat_length=n_e))
            r2 = jnp.sum(rvec ** 2, axis=-1)
        edge_ok = (r2 > 0.0).astype(radial_basis.dtype)
        radial_basis = radial_basis * edge_ok[:, None]
        sh = e3nn.IrrepsArray(sh.irreps, sh.array * edge_ok[:, None].astype(sh.array.dtype))
        if shell_env is not None:
            shell_env = shell_env * edge_ok[:, None]
        # stochastic depth: per-graph keep factors (n_graphs, n_layers) -> per node
        keep = None
        if isinstance(data.globals, dict) and 'stratum_keep' in data.globals:
            keep = data.globals['stratum_keep'][node_graph_idx(data)]   # (n_nodes, n_layers)
        elif isinstance(data.nodes, dict) and 'stratum_keep' in data.nodes:
            keep = data.nodes['stratum_keep']                           # per-node masks

        if self.embed_mode == "onehot":
            features = self.species_embedding(features)
        else:
            # fixed element descriptor rows -> MLP -> hidden scalars (shared over elements)
            desc = jnp.asarray(self.species_descriptors, dtype=features.array.dtype)[species]
            desc_feats = e3nn.IrrepsArray(self.hidden_irreps.filter(keep="0e"),
                                          jax.vmap(self.descriptor_mlp)(desc))
            features = desc_feats if self.embed_mode == "descriptor" else self.species_embedding(features) + desc_feats
        if self.force_encoding:
            mul = self.embed_irreps.num_irreps
            if force_input is None:
                f1o = e3nn.IrrepsArray(e3nn.Irreps(f"{mul}x1o"),
                                       jnp.zeros((features.shape[0], 3 * mul), dtype=features.array.dtype))
            else:
                f1o = _dense(self.force_enc(e3nn.IrrepsArray(
                    e3nn.Irreps("1x1o"), jnp.asarray(force_input, dtype=features.array.dtype))))
            features = e3nn.concatenate([features, f1o], axis=-1)
        # x_mode == "species": per-species scalar, identical for all layers.
        x_feats = self.x_linear(data.nodes['species'], features)

        outputs = []
        layer_parts = {"node": [], "x": [], "f": []}   # per-stratum diagnostics
        if self.pair_repulsion:
            # fixed-form repulsion as the lowest stratum (each pair twice in the edge list -> 1/2)
            sp_i = species[data.receivers]
            sp_j = species[data.senders]
            f_cut = polynomial_cutoff(r_norm, r_c, self.radial_polynomial_p)
            if self.pair_repulsion == "zbl":
                # ZBL universal screened Coulomb repulsion, parameter-free; r = 0 on padded edges -> use a
                # safe r inside and mask the value (keeps the gradient NaN-free, cf. _bessel_safe)
                z = jnp.asarray(self.species_z, dtype=r_norm.dtype)
                zi, zj = z[sp_i], z[sp_j]
                a_ij = 0.46850 / (zi ** 0.23 + zj ** 0.23)
                rs = jnp.where(r_norm > 0.0, r_norm, 1.0)
                x = rs / a_ij
                phi = (0.18175 * jnp.exp(-3.19980 * x) + 0.50986 * jnp.exp(-0.94229 * x)
                       + 0.28022 * jnp.exp(-0.40290 * x) + 0.02817 * jnp.exp(-0.20162 * x))
                e_edge = self.pair_repulsion_scale * 14.399645 * zi * zj / rs * phi * f_cut
                if self.pair_repulsion_switch is not None:
                    rad = jnp.asarray(self.species_radii_t, dtype=r_norm.dtype)
                    r_sw = self.pair_repulsion_switch * (rad[sp_i] + rad[sp_j])
                    e_edge = e_edge * jax.nn.sigmoid((r_norm - r_sw) / self.pair_repulsion_switch_width)
                e_edge = jnp.where(r_norm > 0.0, e_edge, 0.0)
            elif self.pair_repulsion == "uff":
                # UFF exponential-6 repulsive part (Rappé et al. 1992, Table 1 constants, geometric combination
                # rules): E = D_ij 6/(zeta_ij - 6) exp(zeta_ij (1 - r / x_ij)) f_cut; closed-shell exchange
                # repulsion at non-bonded contact distances (1.5–3 A), softer than ZBL in range and
                # parameter-free from our point of view
                tab = jnp.asarray([UFF_VDW[z] for z in self.species_z], dtype=r_norm.dtype)   # (n_species, 3): x, D, zeta
                xi, xj = tab[sp_i, 0], tab[sp_j, 0]
                di, dj = tab[sp_i, 1], tab[sp_j, 1]
                zi, zj = tab[sp_i, 2], tab[sp_j, 2]
                x_ij = jnp.sqrt(xi * xj); d_ij = jnp.sqrt(di * dj); z_ij = jnp.sqrt(zi * zj)
                e_edge = self.pair_repulsion_scale * d_ij * (6.0 / (z_ij - 6.0)) * jnp.exp(z_ij * (1.0 - r_norm / x_ij)) * f_cut
                if self.pair_repulsion_switch is not None:
                    rad = jnp.asarray(self.species_radii_t, dtype=r_norm.dtype)
                    r_sw = self.pair_repulsion_switch * (rad[sp_i] + rad[sp_j])
                    e_edge = e_edge * jax.nn.sigmoid((r_norm - r_sw) / self.pair_repulsion_switch_width)
                e_edge = jnp.where(r_norm > 0.0, e_edge, 0.0)
            else:
                a = self.rep_loga[...]
                rho = jax.nn.softplus(self.rep_rho_raw[...])
                amp = jnp.exp(a[sp_i] + a[sp_j])
                e_edge = amp * jnp.exp(-r_norm / (rho[sp_i] + rho[sp_j])) * f_cut
            e_rep = 0.5 * jax.ops.segment_sum(e_edge, data.receivers, num_segments=features.shape[0])
            outputs.append(e_rep)
        if self.pair_term:
            # explicit 2-body stratum (each pair appears twice in the edge list -> 1/2)
            species = data.nodes["species"]
            c = self.pair_coeff[...]
            c = 0.5 * (c + c.T)
            e_edge = jax.vmap(self.pair_mlp)(radial_basis)[:, 0] * c[species[data.receivers], species[data.senders]]
            e_pair = 0.5 * jax.ops.segment_sum(e_edge, data.receivers, num_segments=features.shape[0])
            outputs.append(e_pair)
        feats_hist = []
        dens_force = (data.nodes.get('dens_force') if (self.dens and isinstance(data.nodes, dict)) else None)
        bo_q = []
        atomic_basis = None
        if self.x_mode == "basis":
            # layer-0 atomic basis A_i = aggregated layer-0 message of the species embedding
            # (equivariant, ungated, linear in the density); computed once, used by every layer
            edge_scale0 = None
            if self.layer_cutoff is not None and self.layer_cutoff[0] < self.cutoff - 1e-6:
                edge_scale0 = polynomial_cutoff(
                    r_norm, jnp.full_like(r_norm, self.layer_cutoff[0]), self.radial_polynomial_p)
            atomic_basis = _dense(self.layers[0].aggregate(
                features, sh, radial_basis, data.senders, data.receivers, edge_scale0,
                species_pair=(species[data.receivers], species[data.senders])))
            if self.x_basis_lmax is not None:
                # keep only l <= x_basis_lmax of A_i (fixed 0/1 mask over the mul_ir layout)
                mask = jnp.asarray(self.x_basis_mask, dtype=atomic_basis.array.dtype)
                atomic_basis = e3nn.IrrepsArray(atomic_basis.irreps, atomic_basis.array * mask[None, :])
        x_species = x_feats
        for i, layer in enumerate(self.layers):
            x_readout = (self.x_readouts[i] if self.x_readout_mode in ("mlp", "linear")
                         else (self.x_readouts[0] if (self.x_readout_mode == "layer0" and i == 0) else None))
            f_readout = self.f_readouts[i] if self.use_f_readout else None
            feats_hist.append(features)
            if self.x_mode == "node_scalar":
                # Receiving node's current scalar (0e) features -> 1x0e
                x_feats = self.x_projs[i](
                    separated_layer_norm(features).filter(keep="0e")
                )
            elif self.x_mode == "node_irreps":
                # Receiving node's current features -> hidden irreps (equivariant x_i)
                x_feats = self.x_projs[i](separated_layer_norm(features))
            elif self.x_mode == "basis":
                if self.x_basis_layers is None or i in self.x_basis_layers:
                    x_feats = _dense(self.x_projs[i](atomic_basis))
                    if self.x_norm == "rms":
                        x_feats = _rms_normalize(x_feats)
                    elif self.x_norm == "hill":
                        x_feats = _hill_normalize(
                            x_feats, self.x_hill[...][min(i, 1) if self.tie_operator else i], species)
                    elif self.x_norm == "spectral":
                        x_feats = _spectral_normalize(self.x_projs[i], x_feats)
                else:
                    x_feats = x_species          # per-species x (scalar-only), as in v7_ab
            f_energy = (f_readout(data.nodes["species"], features) if f_readout is not None else None)
            edge_scale = None
            if self.layer_cutoff is not None and self.layer_cutoff[i] < self.cutoff - 1e-6:
                edge_scale = polynomial_cutoff(
                    r_norm, jnp.full_like(r_norm, self.layer_cutoff[i]), self.radial_polynomial_p)
            drop_scale = None if keep is None else keep[:, i]
            sh_i = sh
            if self.shell_sk_gate:
                # Slater–Koster gate: Y_l scaled per edge by 1 + g_l(r), g_l = sum over allowed
                # shell pairs of c_{ab,l} env_ab(r) (equivalent to an l-selective radial modulation
                # of every tensor-product path of degree l)
                coeff = self.sk_coeff[...][i] * jnp.asarray(self.sk_mask, dtype=shell_env.dtype)   # (n_shells**2, lmax+1)
                g = shell_env @ coeff                                                                # (n_edges, lmax+1)
                factor = jnp.concatenate([jnp.repeat(1.0 + g[:, l:l + 1], 2 * l + 1, axis=1)
                                          for l in range(self.lmax + 1)], axis=1)                   # (n_edges, sh.dim)
                sh_i = e3nn.IrrepsArray(sh.irreps, sh.array * factor)
            prev_features = (feats_hist[i - 1] if (self.residual == "chebyshev" and i >= 2) else
                             feats_hist[1] if (self.residual == "neumann" and i >= 1) else None)
            if _remat_layers():
                # recompute the layer (per-edge radial weights, messages) in the backward pass instead of storing
                # it: the per-edge activations of all layers dominate device memory (2026-11-06: ~140 kB per edge
                # for L5 lmax3 128 channels float64); environment BAM_REMAT_LAYERS=1, read at trace time
                def _call(lyr, f, xf, sp, y, rb, snd, rcv, es, ds, pf):
                    return lyr(f, xf, sp, y, rb, snd, rcv, edge_scale=es, drop_scale=ds, prev_features=pf)
                node_energies, features = nnx.remat(_call)(
                    layer, features, x_feats, data.nodes["species"], sh_i, radial_basis, data.senders,
                    data.receivers, edge_scale, drop_scale, prev_features)
            else:
                node_energies, features = layer(
                    features,
                    x_feats,
                    data.nodes["species"],
                    sh_i,
                    radial_basis,
                    data.senders,
                    data.receivers,
                    edge_scale=edge_scale,
                    drop_scale=drop_scale,
                    prev_features=prev_features,
                )
            if self.feature_norm == "rms":
                features = _rms_normalize(_dense(features))
            if i == 0 and dens_force is not None:
                features = _dense(features)
                features = features + self._dens_encoding(dens_force, features)
            if self.bond_order is not None and self.bond_order_shape:
                q = _dense(features).filter(keep="0e").array
                bo_q.append(q * jax.lax.rsqrt(jnp.mean(q * q, axis=-1, keepdims=True) + 1e-6))
            x_energy = (x_readout(data.nodes["species"], x_feats) if x_readout is not None else None)
            if self.stratum_gain == "uniform":
                gain = 1.0
            elif self.stratum_gain == "jackson":
                gain = _jackson_gain(i, len(self.layers))
            elif self.stratum_gain == "inverse":
                gain = 1.0 / (i + 1)
            else:
                gain = 1.0 / (i + 1) ** 2
            e_node = node_energies.array[:, 0]
            e_x = x_energy.array.squeeze() if x_energy is not None else jnp.zeros_like(e_node)
            e_f = f_energy.array.squeeze() if f_energy is not None else jnp.zeros_like(e_node)
            if drop_scale is not None:
                gain = gain * drop_scale
            layer_parts["node"].append(gain * e_node)
            layer_parts["x"].append(gain * e_x)
            layer_parts["f"].append(gain * e_f)
            outputs += [gain * (e_node + e_x + e_f)]

        if self.strata_resum == "aitken":
            # geometric-tail (Aitken / Shanks e1) estimate of the body-order series from the last two
            # layer strata: rho = e_L / e_{L-1} (saturated to |rho| < 0.5), tail = e_L rho / (1 - rho);
            # added as an extra stratum with learnable weight lambda (init 0 -> plain partial sum)
            e_last, e_prev = outputs[-1], outputs[-2]
            rho = e_last * e_prev / (e_prev * e_prev + 1e-8)
            rho = 0.5 * jnp.tanh(rho / 0.5)
            tail = e_last * rho / (1.0 - rho)
            outputs = outputs + [self.resum_lambda[...] * tail]

        if self.bond_order is not None:
            # homogeneous bond-order scale stratum (review §12.7): second-moment width
            # b_i = (eps + sum_j beta_ZZ'(r_ij)^2)^(1/2), beta > 0 and decaying with r
            sp_i, sp_j = species[data.receivers], species[data.senders]
            c = jax.nn.softplus(self.bo_c[...])
            c = 0.5 * (c + jnp.swapaxes(c, 0, 1))[sp_i, sp_j]                                   # (n_edges, K)
            alpha = jnp.asarray(self.bo_alpha, dtype=r_norm.dtype)
            env = polynomial_cutoff(r_norm, r_c, self.radial_polynomial_p)
            beta = jnp.sum(c * jnp.exp(-r_norm[:, None] * alpha[None, :]), axis=-1) * env
            beta = jnp.where(r_norm > 0.0, beta, 0.0)                                            # padded edges
            b2 = 1e-6 + jax.ops.segment_sum(beta * beta, data.receivers, num_segments=species.shape[0])
            if self.bond_order == "log":
                phi = 0.5 * jnp.log(b2)
            else:
                p = (self.bond_order_p if self.bond_order_p is not None
                     else 0.5 + jnp.minimum(jax.nn.softplus(self.bo_pi[...]), 1.0))
                phi = b2 ** (0.5 * p)
            if self.bond_order_shape:
                phi = phi * (1.0 + jax.vmap(self.bo_shape_mlp)(jnp.concatenate(bo_q, axis=-1))[:, 0])
            outputs = outputs + [-self.bo_s[...][species] * phi]

        # (n_strata, n_atoms): stratum 0 is the pair term when enabled
        strata = jnp.stack(outputs, axis=0) * self.scale[...]
        node_energies = jnp.sum(
            jnp.stack(outputs, axis=1),
            axis=-1,
            keepdims=True
        )
        node_energies = e3nn.IrrepsArray(e3nn.Irreps("0e"), node_energies)

        # Scale and shift energies
        node_energies = node_energies * self.scale[...] + self.shift[...]

        frc_logvar = self.frc_logvar(features)
        if self.cov_mode == "scalar6":
            # Force variation: 6 params for 3x3 lower triangular Cholesky L_f
            # [diag0, diag1, diag2, off0, off1, off2] -> softplus on diagonal, min 0.04
            frc_var = jnp.hstack([
                jax.nn.softplus(frc_logvar.array[:, :3]) + 0.04,
                frc_logvar.array[:, 3:]
            ])
        else:
            # Force variation (equivariant): [a, b_0..b_4]
            #   a : isotropic std (0e), softplus + min 0.04
            #   b : 5 components of a 2e irrep (traceless symmetric B)
            # Sigma = (a*I + B')^2 is assembled in training_utils.build_force_L
            frc_var = jnp.hstack([
                jax.nn.softplus(frc_logvar.array[:, :1]) + 0.04,
                frc_logvar.array[:, 1:]
            ])

        # Energy variation: per-atom sigma, min 0.04
        energy_var = jax.nn.softplus(self.energy_logvar(features).array) + 0.04  # (n_atoms, 1)

        # Stress variation: 6 diagonal sigma values (one per Voigt component)
        stress_raw = self.stress_logvar(features).array  # (n_atoms, 6)
        stress_var = jax.nn.softplus(stress_raw) + 0.04

        return {
            'node_energies': node_energies.array,
            'node_features': features,
            'frc_variation': frc_var,
            'energy_variation': energy_var,
            'stress_variation': stress_var,
            # diagnostics (not used by the loss)
            'strata': strata,                                   # (n_strata, n_atoms), scaled
            'strata_parts': {k: jnp.stack(v, axis=0) * self.scale[...] for k, v in layer_parts.items()},
        }

    def _dens_encoding(self, force: jax.Array, features: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        """DeNS force encoding (|F|/sqrt 3) Y^l_component(F^), l = 0..l_enc -> bias-free Linear; F = 0 gives 0."""
        f = jnp.asarray(force, dtype=features.array.dtype)
        fn = jnp.sqrt(jnp.sum(f * f, axis=-1, keepdims=True))
        y = e3nn.spherical_harmonics(self.dens_enc_irreps, f / jnp.maximum(fn, 1e-12), normalize=False,
                                     normalization="component")
        enc = _dense(self.dens_enc(e3nn.IrrepsArray(y.irreps, y.array * (fn / jnp.sqrt(3.0)))))
        assert enc.irreps == features.irreps, (enc.irreps, features.irreps)
        return enc

    def dens_noise(self, node_features: e3nn.IrrepsArray) -> jax.Array:
        """DeNS noise head: (n_atoms, 3) prediction of eps/sigma from the final node features
        (RMS-normalised -> Linear -> gate -> Linear -> gate -> Linear to 1x1o)."""
        h = _rms_normalize(_dense(node_features))
        h = e3nn.gate(_dense(self.dens_head1(h)), even_act=jax.nn.silu, even_gate_act=jax.nn.silu)
        h = e3nn.gate(_dense(self.dens_head2(h)), even_act=jax.nn.silu, even_gate_act=jax.nn.silu)
        return _dense(self.dens_head3(h)).array

    def denoise(self, data: jraph.GraphsTuple) -> jax.Array:
        """Noisy-Nodes head: equivariant per-node vector (n_atoms, 3) from the final features
        of a plain forward pass (no energy gradient).  Trained to predict the (normalised)
        displacement noise added to the positions."""
        if not self.use_noise_head:
            raise ValueError("model was built without noise_head=True")
        force_input = data.nodes['forces'] if (self.force_encoding and 'forces' in data.nodes) else None
        out = self.node_energies(data.nodes['positions'], None, None, data, force_input=force_input)
        return self.noise_head(out['node_features']).array

    def strata_energies_forces(self, data: jraph.GraphsTuple):
        """Per-stratum total energies (n_strata, n_graphs) and forces (n_strata, n_atoms, 3)
        for a (possibly padded) batch.  Stratum 0 is the explicit pair term when
        ``pair_term`` is on, otherwise stratum i is layer i."""
        from bam_omat24.utils import node_graph_idx
        n_graphs = data.n_node.shape[0]
        node_mask = (jraph.get_node_padding_mask(data) if n_graphs > 1
                     else jnp.ones(data.nodes['positions'].shape[0], bool)).astype(jnp.float32)
        gidx = node_graph_idx(data)

        def strata_fn(positions):
            out = self.node_energies(positions, None, None, data)
            return out['strata'] * node_mask[None, :]            # (S, n_atoms)

        strata = strata_fn(data.nodes['positions'])
        energies = jax.vmap(lambda s: jraph.segment_sum(s, gidx, num_segments=n_graphs))(strata)
        # forces of each stratum: -d(sum_atoms strata_s)/d positions, one VJP per stratum
        n_strata = strata.shape[0]
        _, vjp = jax.vjp(lambda p: jnp.sum(strata_fn(p), axis=1), data.nodes['positions'])
        forces = jnp.stack([-vjp(jnp.eye(n_strata, dtype=strata.dtype)[s])[0] for s in range(n_strata)], axis=0)
        forces = forces * node_mask[None, :, None]
        return energies, forces

    @property
    def frc_variation_fill(self) -> jax.Array:
        """Neutral (identity covariance) fill value for padded nodes."""
        if self.cov_mode == "scalar6":
            return jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        return jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def __call__(
        self, data: jraph.GraphsTuple
    ) -> Dict:
        """Compute energies and forces for a batch of graphs.

        Args:
            data: Graph data structure with nodes containing 'positions' and 'species'.
                  For periodic systems, globals must contain 'cell', 'cutoff', 'volume'.
                  For molecular systems, these are not required.

        Returns:
            Tuple of (graph_energies, forces, stress) where:
                - graph_energies: Total energy per graph, shape (n_graphs,)
                - forces: Forces on each atom, shape (n_atoms, 3)
                - stress: Stress tensor per graph, shape (n_graphs, 6)
                          (Voigt: xx, yy, zz, yz, xz, xy)
                          Zero for molecular systems.
        """
        return energy_forces_stress(
            self.node_energies, data, self.periodic,
            self.l_train, jnp.asarray(self.atom_energies, dtype=jnp.float32),
            frc_variation_fill=self.frc_variation_fill,
        )


if __name__ == "__main__":
    # Example usage
    import numpy as np
    import pickle
    from bam_omat24.atom_energies import ATOM_ENERGIES

    print("=" * 70)
    print("RACE-Style Equivariant Potential")
    print("=" * 70)

    with open('omat24_val_0.pkl', 'rb') as f:
        graphset = pickle.load(f)
    data = graphset[0]

    # Create model
    rngs = nnx.Rngs(42)
    model = RACE(
        n_species=len(ATOM_ENERGIES),
        lmax=3,
        hidden_irreps="64x0e + 64x1o + 64x2e",
        n_layers=3,
        radial_basis_size=8,
        radial_mlp_size=64,
        radial_mlp_layers=3,
        mlp_init_scale=4.0,
        shift=0.0,
        scale=1.0,
        avg_n_neighbors=25,
        atom_energies=ATOM_ENERGIES,
        l_train=True,
        periodic=True,
        rngs=rngs,
    )

    # Count parameters
    graphdef, state = nnx.split(model)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(state))
    print(f"\nNumber of parameters: {n_params:,}")

    # Run forward pass
    #energies, forces, stress = model(data)
    result = model(data)
    print(f"Energies shape: {result['energy'].shape}")
    print(f"Forces shape: {result['forces'].shape}")
    print(f"Total energy: {result['energy'][0]:.6f}")
    print(f"Force magnitude: {np.linalg.norm(result['forces'], axis=-1).mean():.6f}")
    print(f"Stress: {result['stress']}")
    print(f"Target energies: {data.globals['energy']}")
    print(f"Target stress: {data.globals['stress']} eV/A^3")
