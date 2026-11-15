"""RACEv2 interatomic potential, PyTorch inference re-implementation of race_v2/models/race.py (class RACE).

Faithful port in float64 of the configuration family of the MPtrj runs (x_mode basis, x_norm hill, residual chebyshev,
LayerScale, linear update, message_norm avg, gated stratum readouts, x / f readouts, radial prior, shell-radius radial
channels, UFF repulsion with switch, periodic cells with stress).  Features use the e3nn-jax ``mul_ir`` layout
(per irrep block ``(mul, 2l+1)``), so every weight matrix of the JAX checkpoint maps one-to-one onto a parameter of the
same name here (``layers.3.linear_2.weight_w_0_0_512x0e_128x0e`` <-> ``layers/3/linear_2/weight_w_0_0_512x0e_128x0e``).

The forward pass keeps the split of the JAX code: ``prelude`` (edge features, embedding, repulsion, atomic basis),
``layer_step`` (one interaction layer + its energy stratum) and ``finalize`` (sum of the strata, scale / shift), so
that a domain-decomposed MD driver can exchange node features after every layer.

Graph convention (race_v2/data_3bpa.preprocess_graph): receivers = i, senders = j, integer shifts ``Sij`` (n_edges, 3)
with edge vector  r_ij = x_j - (x_i + Sij . cell)  (= x_j - x_i + S . cell for the ASE shift S = -Sij).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from . import o3
from .o3 import Irreps, NormalizedSiLU, silu

FORMAT = 'racev2_torch'
FORMAT_VERSION = 1


# ===================================================================================================== config
def _get(cfg: Dict, key: str, default=None):
    v = cfg.get(key, default)
    return default if v is None else v


# (key, default as in training.train.build_model, allowed values): options the port implements.  Anything else raises.
_FIXED_OPTIONS = [
    ('model', 'RACE', ('RACE',)),
    ('x_mode', 'species', ('basis',)),
    ('x_norm', None, ('hill',)),
    ('residual', 'none', ('chebyshev',)),
    ('update', 'gate', ('linear',)),
    ('message_norm', 'layer', ('avg',)),
    ('readout_mode', 'linear', ('gated',)),
    ('embed_mode', 'onehot', ('onehot',)),
    ('x_readout', 'mlp', ('mlp',)),
    ('x_readout_norm', True, (True,)),
    ('x_readout_norm_eps', None, (None,)),     # soft readout norm (2026-11-15): the port has eps 1e-6 only
    ('f_readout', True, (True,)),
    ('readout_species', False, (False,)),
    ('stratum_gain', 'uniform', ('uniform',)),
    ('in_layer_order', 1, (1,)),
    ('in_layer_order_layers', None, (None,)),
    ('sym_contraction', False, (False,)),
    ('layer_lmax', None, (None,)),
    ('layer_cutoff', None, (None,)),
    ('radial_species_scale', False, (False,)),
    ('radial_slater', False, (False,)),
    ('radial_shell_scale', False, (False,)),
    ('shell_sk_gate', False, (False,)),
    ('strata_resum', None, (None,)),
    ('pair_term', False, (False,)),
    ('radial_gaussians', None, (None,)),
    ('radial_species_mix', None, (None,)),
    ('feature_norm', None, (None,)),
    ('tie_operator', False, (False,)),
    ('bond_order', None, (None,)),
    ('x_basis_lmax', None, (None,)),
    ('x_basis_layers', None, (None,)),
    ('angular_prior', None, (None,)),
    ('force_encoding', False, (False,)),
    ('dens', False, (False,)),
    ('noisy_nodes', False, (False,)),
]


def resolve_config(cfg: Dict) -> Dict:
    """Training config (as stored in the checkpoint) -> the hyper-parameters of the model, resolved exactly as
    race_v2/train_3bpa.build_model does.  Raises NotImplementedError for options the port does not implement."""
    for key, default, allowed in _FIXED_OPTIONS:
        v = _get(cfg, key, default)
        if all(isinstance(a, bool) for a in allowed):
            ok = bool(v) in allowed                      # flags (build_model reads them with bool(...))
        else:
            ok = (not isinstance(v, (bool, dict, list))) and v in allowed
        if not ok:
            raise NotImplementedError(f"race_v2_torch: option {key}={v!r} is not implemented (supported: {allowed})")
    if not _get(cfg, 'x64', False):
        import warnings
        warnings.warn("config has x64 false: the JAX model ran in float32, the torch port runs in float64 "
                      "(differences of order 1e-6 relative are expected)")
    lmax = int(_get(cfg, 'lmax', 3))
    hidden_irreps = _get(cfg, 'hidden_irreps', None)
    if hidden_irreps is None:
        hidden_irreps = str(o3.s2_irreps(lmax).scale_mul(int(_get(cfg, 'hidden_size', 128))))
    hidden = Irreps(hidden_irreps)
    if len({m for m, _, _ in hidden}) != 1 or any(p != (-1) ** l for _, l, p in hidden):
        raise NotImplementedError(f"hidden irreps {hidden}: only a uniform multiplicity and natural parity are "
                                  f"implemented (equivariant x product)")
    pr = _get(cfg, 'pair_repulsion', False)
    if pr not in (False, 'uff'):
        raise NotImplementedError(f"pair_repulsion={pr!r}: only 'uff' (or off) is implemented")
    rp = _get(cfg, 'radial_prior', None)
    if rp:
        rp = dict(rp)
        if str(rp.get('type', 'algebraic')) not in ('algebraic', 'exponential', 'gaussian'):
            raise NotImplementedError(f"radial_prior {rp}")
    atomic_numbers = [int(z) for z in cfg['atomic_numbers']]
    atom_energies = cfg.get('atom_energies')
    if atom_energies is None:
        atom_energies = [0.0] * len(atomic_numbers)
    sr = _get(cfg, 'species_rank', None)
    switch = _get(cfg, 'pair_repulsion_switch', None)
    return dict(
        atomic_numbers=atomic_numbers,
        atom_energies=[float(e) for e in atom_energies],
        lmax=lmax,
        cutoff=float(_get(cfg, 'cutoff', 5.0)),
        hidden_irreps=str(hidden),
        n_layers=int(_get(cfg, 'n_layers', 5)),
        radial_basis_size=int(_get(cfg, 'radial_basis_size', 8)),
        radial_mlp_size=int(_get(cfg, 'radial_mlp_size', 64)),
        radial_mlp_layers=int(_get(cfg, 'radial_mlp_layers', 3)),
        radial_polynomial_p=float(_get(cfg, 'radial_polynomial_p', 2.0)),
        avg_n_neighbors=float(_get(cfg, 'avg_n_neighbors', 25.0)),
        periodic=bool(_get(cfg, 'periodic', False)),
        layer_scale=(None if _get(cfg, 'layer_scale', None) is None else float(cfg['layer_scale'])),
        radial_shells=bool(_get(cfg, 'radial_shells', False)),
        radial_prior=(dict(rp) if rp else None),
        pair_repulsion=(pr or False),
        pair_repulsion_scale=float(_get(cfg, 'pair_repulsion_scale', 1.0)),
        pair_repulsion_switch=(None if switch is None else float(switch)),
        pair_repulsion_switch_width=float(_get(cfg, 'pair_repulsion_switch_width', 0.1)),
        species_rank=(None if sr is None else int(sr)),
        species_rank_prune=bool(_get(cfg, 'species_rank_prune', False)) and sr is not None,
    )


# ===================================================================================================== building blocks
def _sanitize_name(name: str) -> str:
    """models.linear.Linear._sanitize_name: 'w[0,0] 128x0e,128x0e' -> 'weight_w_0_0_128x0e_128x0e'."""
    s = name.replace("[", "_").replace("]", "_").replace(",", "_").replace(" ", "_").replace(".", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return "weight_" + s.strip("_")


class SpeciesGroups:
    """Nodes sorted by species (stable), for per-species matmuls of the element-indexed linears."""

    def __init__(self, species: torch.Tensor):
        self.perm = torch.argsort(species, stable=True)
        self.inv = torch.empty_like(self.perm)
        self.inv[self.perm] = torch.arange(self.perm.numel(), device=species.device)
        uniq, counts = torch.unique_consecutive(species[self.perm], return_counts=True)
        self.species = [int(z) for z in uniq.tolist()]
        self.counts = [int(c) for c in counts.tolist()]


class EquivariantLinear(nn.Module):
    """race_v2/linear_nnx.Linear (+ e3nn_jax.FunctionalLinear) in the mul_ir layout.

    y_out = sum_{blocks in -> out} path_weight * W^T x_in  (+ bias on 0e outputs), path_weight = sqrt(1 / fan_in),
    fan_in = sum of the input multiplicities feeding the output block (path_normalization "element", gradient
    normalization "path": e3nn-jax defaults).  Element-indexed (``num_indexed``): one W per species; with
    ``index_rank`` K the table is W[z] = K**-0.5 sum_k coeff[z, k] basis[k] (``index_rank_lpos`` for l > 0 outputs).
    """

    def __init__(self, irreps_in, irreps_out, biases: bool = False, force_irreps_out: bool = False,
                 num_indexed: Optional[int] = None, index_rank: Optional[int] = None,
                 index_rank_lpos: Optional[int] = None):
        super().__init__()
        self.irreps_in = Irreps(irreps_in).regroup()
        irreps_out = Irreps(irreps_out)
        if not force_irreps_out:
            irreps_out = irreps_out.filter_keep(self.irreps_in.irs()).simplify()
        self.irreps_out = irreps_out
        self.num_indexed = num_indexed
        self.index_rank = None if index_rank is None else int(index_rank)
        self.index_rank_lpos = (self.index_rank if (index_rank_lpos is None or index_rank is None)
                                else int(index_rank_lpos))
        ins = [(i, o) for i, (_, li, pi) in enumerate(self.irreps_in) for o, (_, lo, po) in enumerate(irreps_out)
               if (li, pi) == (lo, po)]
        self.instructions = []
        for i, o in ins:
            x = (self.irreps_in[i][0] ** 0.0) * sum(float(self.irreps_in[i2][0]) ** 1.0 for i2, o2 in ins if o2 == o)
            alpha = 1 / x if x > 0 else 1.0
            path_weight = math.sqrt(alpha) ** 1.0
            mi, mo = self.irreps_in[i][0], irreps_out[o][0]
            name = _sanitize_name(f"w[{i},{o}] {Irreps.mulir_str(*self.irreps_in[i])},{Irreps.mulir_str(*irreps_out[o])}")
            rank = None
            if num_indexed is not None and self.index_rank is not None:
                rank = self.index_rank_lpos if irreps_out[o][1] > 0 else self.index_rank
            if num_indexed is None:
                shape = (mi, mo)
            elif rank is None:
                shape = (num_indexed, mi, mo)
            else:
                shape = (rank, mi, mo)
                self.register_parameter("coeff_" + name[len("weight_"):],
                                        nn.Parameter(torch.zeros(num_indexed, rank, dtype=torch.float64)))
            self.register_parameter(name, nn.Parameter(torch.zeros(shape, dtype=torch.float64)))
            self.instructions.append((i, o, name, path_weight, rank))
        self.biases = []
        if biases:
            for o, (mo, lo, po) in enumerate(irreps_out):
                if lo == 0 and po == 1:
                    name = _sanitize_name(f"b[{o}] {Irreps.mulir_str(mo, lo, po)}")
                    shape = (mo,) if num_indexed is None else (num_indexed, mo)
                    self.register_parameter(name, nn.Parameter(torch.zeros(shape, dtype=torch.float64)))
                    self.biases.append((o, name))
        self._in_slices = self.irreps_in.slices()

    def species_weight(self, name: str, rank: Optional[int], z: int) -> torch.Tensor:
        """W[z] (mul_in, mul_out) of an indexed block (models.linear.Linear._get_weight)."""
        w = getattr(self, name)
        if rank is None:
            return w[z]
        c = getattr(self, "coeff_" + name[len("weight_"):])
        return torch.tensordot(c[z], w, dims=1) * (w.shape[0] ** -0.5)

    def forward(self, x: torch.Tensor, species: Optional[torch.Tensor] = None,
                groups: Optional[SpeciesGroups] = None) -> torch.Tensor:
        n = x.shape[0]
        outs: List[Optional[torch.Tensor]] = [None] * len(self.irreps_out)
        if self.num_indexed is not None:
            if species is None:
                raise ValueError("indexed linear needs species")
            if groups is None:
                groups = SpeciesGroups(species)
        for i, o, name, pw, rank in self.instructions:
            a, b = self._in_slices[i]
            mi, li, _ = self.irreps_in[i]
            xb = x[:, a:b].reshape(n, mi, 2 * li + 1)
            if self.num_indexed is None:
                y = pw * torch.einsum("uw,nui->nwi", getattr(self, name), xb)
            else:
                xs = xb[groups.perm]
                parts, off = [], 0
                for z, cnt in zip(groups.species, groups.counts):
                    parts.append(torch.einsum("uw,nui->nwi", self.species_weight(name, rank, z), xs[off:off + cnt]))
                    off += cnt
                y = pw * torch.cat(parts, dim=0)[groups.inv]
            outs[o] = y if outs[o] is None else outs[o] + y
        for o, name in self.biases:
            bvec = getattr(self, name)
            bvec = bvec.view(1, -1, 1) if self.num_indexed is None else bvec[species].unsqueeze(-1)
            y = 1.0 * bvec
            outs[o] = y.expand(n, -1, -1) if outs[o] is None else outs[o] + y
        chunks = []
        for o, (mo, lo, _) in enumerate(self.irreps_out):
            if outs[o] is None:
                chunks.append(x.new_zeros(n, mo * (2 * lo + 1)))
            else:
                chunks.append(outs[o].reshape(n, mo * (2 * lo + 1)))
        return torch.cat(chunks, dim=-1) if chunks else x.new_zeros(n, 0)


class DenseLinear(nn.Module):
    """race_v2/utils.DenseLinear without bias: x @ weights, weights (in, out)."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_in, n_out, dtype=torch.float64))

    def forward(self, x):
        return x @ self.weights


class RadialMLP(nn.Module):
    """race_v2/utils.MLP (use_bias False): Dense -> silu -> ... -> Dense (plain silu, not normalised)."""

    def __init__(self, sizes: Sequence[int]):
        super().__init__()
        self.layers = nn.ModuleList([DenseLinear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = silu(x)
        return x


def separated_layer_norm(x: torch.Tensor, irreps: Irreps, eps: float = 1e-6) -> torch.Tensor:
    """race_v2/utils.separated_layer_norm: each 0e block centred + RMS-normalised over its channels, all
    non-scalar blocks share one RMS."""
    n = x.shape[0]
    sl = irreps.slices()
    xi_s = x.new_zeros(n, 1)
    cnt = 0
    for (mul, l, p), (a, b) in zip(irreps, sl):
        if not (l == 0 and p == 1):
            xi_s = xi_s + torch.sum(x[:, a:b] ** 2, dim=-1, keepdim=True)
            cnt += mul * (2 * l + 1)
    inv_ns = torch.rsqrt(xi_s / max(cnt, 1) + eps)
    chunks = []
    for (mul, l, p), (a, b) in zip(irreps, sl):
        xi = x[:, a:b]
        if l == 0 and p == 1:
            xc = xi - torch.mean(xi, dim=-1, keepdim=True)
            chunks.append(xc * torch.rsqrt(torch.mean(xc ** 2, dim=-1, keepdim=True) + eps))
        else:
            chunks.append(xi * inv_ns)
    return torch.cat(chunks, dim=-1)


def bessel_basis(x: torch.Tensor, num_basis: int, r_max: torch.Tensor) -> torch.Tensor:
    """utils.bessel_basis 2/r_max sin(n pi x / r_max) / x (limit n pi / r_max at x = 0), NaN-free gradient."""
    r_max = r_max[:, None]
    x = x[:, None]
    prefactor = 2.0 / r_max
    w = torch.linspace(1.0, num_basis, num_basis, dtype=x.dtype, device=x.device) * math.pi
    xs = torch.where(x == 0.0, torch.ones_like(x), x)
    return prefactor * torch.where(x == 0.0, w / r_max, torch.sin(w * xs / r_max) / xs)


def polynomial_cutoff(x: torch.Tensor, r_max, p: float) -> torch.Tensor:
    """utils.polynomial_cutoff."""
    factor = 1.0 / r_max
    x = x * factor
    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
    out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))
    return out * torch.where(x < 1.0, 1.0, 0.0)


def _safe_norm(r2: torch.Tensor) -> torch.Tensor:
    """sqrt(r2) with value 0 and gradient 0 at r2 = 0 (JAX: where(r2 == 0, 0, sqrt(r2)))."""
    ok = r2 > 0.0
    return torch.where(ok, torch.sqrt(torch.where(ok, r2, torch.ones_like(r2))), torch.zeros_like(r2))


def _tp_paths(irreps1: Irreps, irreps2: Irreps, filter_irs) -> List[Tuple[int, int, int, int, int]]:
    """Paths (i1, i2, l3, p3, position) of e3nn_jax.tensor_product / cue channelwise_tensor_product, in the order
    of the output irreps after the (stable) sort: loop i1, i2, l3 in ir1 * ir2, filtered, then sorted by irrep."""
    paths = []
    for i1, (_, l1, p1) in enumerate(irreps1):
        for i2, (_, l2, p2) in enumerate(irreps2):
            for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                p3 = p1 * p2
                if filter_irs is not None and (l3, p3) not in filter_irs:
                    continue
                paths.append((i1, i2, l3, p3))
    order = sorted(range(len(paths)), key=lambda k: (o3._ir_key(paths[k][2], paths[k][3]), k))
    return [paths[k] for k in order]


class EdgeTensorProduct(nn.Module):
    """Channelwise edge tensor product with gather / scatter (models.race.cue_edge_tensor_product, or the e3nn path
    linear_1(h)[senders] (x) Y -> * R -> scatter_sum):  m_i = sum_{j} R_ij,p,u  (h_j,u (x)_p Y(r_ij)), output blocks in
    the sorted path order (path-major: block mul index = path * mul + u), CG with 'component' normalisation."""

    def __init__(self, irreps_in: Irreps, sh_irreps: Irreps, filter_irreps: Irreps):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.sh_irreps = Irreps(sh_irreps)
        paths = _tp_paths(self.irreps_in, self.sh_irreps, filter_irreps.irs())
        self.irreps_out = Irreps([(self.irreps_in[i1][0] * self.sh_irreps[i2][0], l3, p3)
                                  for i1, i2, l3, p3 in paths])
        if any(m != 1 for m, _, _ in self.sh_irreps):
            raise NotImplementedError("edge tensor product with multiplicity > 1 spherical harmonics")
        self.n_weights = self.irreps_out.num_irreps
        in_sl, sh_sl = self.irreps_in.slices(), self.sh_irreps.slices()
        # group the paths by input block: one batched matmul per input block
        self.groups = []          # (i1, a, b, mul, d1, [(sorted position, l2 slice, d3, q offset, w offset, cg name)])
        w_off = []
        acc = 0
        for i1, i2, l3, p3 in paths:
            w_off.append(acc)
            acc += self.irreps_in[i1][0]
        by_block: Dict[int, list] = {}
        for pos, (i1, i2, l3, p3) in enumerate(paths):
            by_block.setdefault(i1, []).append((pos, i2, l3))
        self.out_index = [None] * len(paths)           # pos -> (group index, q offset, d3)
        for gi, i1 in enumerate(sorted(by_block)):
            mul, l1, _ = self.irreps_in[i1]
            entries, q = [], 0
            for pos, i2, l3 in by_block[i1]:
                l2 = self.sh_irreps[i2][1]
                cg = o3.clebsch_gordan(l1, l2, l3) * math.sqrt(2 * l3 + 1)
                name = f"cg_{gi}_{pos}"
                self.register_buffer(name, torch.as_tensor(cg, dtype=torch.float64), persistent=False)
                entries.append((pos, sh_sl[i2], 2 * l3 + 1, q, w_off[pos], name))
                self.out_index[pos] = (gi, q, 2 * l3 + 1)
                q += 2 * l3 + 1
            self.groups.append((i1, in_sl[i1][0], in_sl[i1][1], mul, 2 * l1 + 1, q, entries))
        self.paths = paths

    def forward(self, x: torch.Tensor, sh: torch.Tensor, w: torch.Tensor, senders: torch.Tensor,
                receivers: torch.Tensor, n_nodes: int) -> torch.Tensor:
        e = senders.shape[0]
        xs = x[senders]
        aggs = []
        for i1, a, b, mul, d1, q_tot, entries in self.groups:
            xb = xs[:, a:b].reshape(e, mul, d1)
            m = torch.cat([torch.einsum("ijk,ej->eik", getattr(self, name).to(x.dtype), sh[:, s0:s1])
                           for pos, (s0, s1), d3, q, wo, name in entries], dim=-1)          # (E, d1, Q)
            out = torch.bmm(xb, m)                                                             # (E, mul, Q)
            wexp = torch.cat([w[:, wo:wo + mul, None].expand(e, mul, d3)
                              for pos, (s0, s1), d3, q, wo, name in entries], dim=-1)          # (E, mul, Q)
            out = out * wexp
            aggs.append(torch.zeros(n_nodes, mul, q_tot, dtype=x.dtype, device=x.device).index_add(0, receivers, out))
        chunks = []
        for pos, (i1, i2, l3, p3) in enumerate(self.paths):
            gi, q, d3 = self.out_index[pos]
            chunks.append(aggs[gi][:, :, q:q + d3].reshape(n_nodes, -1))
        return torch.cat(chunks, dim=-1)


class ChannelwiseProduct(nn.Module):
    """Equivariant product m_i (x) x_i of RACEConvolution (x_mode basis): the e3nn-jax tensor product of the
    multiplicity-1 copies (mul_to_axis), all CG paths into the hidden irreps, axis_to_mul (channel-major: block mul
    index = channel * n_paths + path)."""

    def __init__(self, irreps: Irreps):
        super().__init__()
        self.irreps = Irreps(irreps)
        muls = {m for m, _, _ in self.irreps}
        assert len(muls) == 1
        self.mul = muls.pop()
        ir1 = Irreps([(1, l, p) for _, l, p in self.irreps])
        paths = _tp_paths(ir1, ir1, ir1.irs())
        out1 = Irreps([(1, l3, p3) for _, _, l3, p3 in paths]).regroup()          # e.g. 4x0e+6x1o+7x2e+6x3o
        self.irreps_out = out1.scale_mul(self.mul)
        self.blocks = []        # per output block: [(i1, i2, cg name)] in path order, d3
        k = 0
        for nb, (np_, l3, p3) in enumerate(out1):
            entries = []
            for _ in range(np_):
                i1, i2, l3_, p3_ = paths[k]
                assert (l3_, p3_) == (l3, p3)
                cg = o3.clebsch_gordan(ir1[i1][1], ir1[i2][1], l3) * math.sqrt(2 * l3 + 1)
                name = f"cg_{nb}_{len(entries)}"
                self.register_buffer(name, torch.as_tensor(cg, dtype=torch.float64), persistent=False)
                entries.append((i1, i2, name))
                k += 1
            self.blocks.append((entries, 2 * l3 + 1))
        self._sl = self.irreps.slices()

    def forward(self, m: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        n, c = m.shape[0], self.mul
        mb = [m[:, a:b].reshape(n, c, (b - a) // c) for a, b in self._sl]
        xb = [x[:, a:b].reshape(n, c, (b - a) // c) for a, b in self._sl]
        out = []
        for entries, d3 in self.blocks:
            ys = [torch.einsum("nci,ncj,ijk->nck", mb[i1], xb[i2], getattr(self, name).to(m.dtype))
                  for i1, i2, name in entries]
            out.append(torch.stack(ys, dim=2).reshape(n, -1))            # (n, c, n_paths, d3) -> channel-major
        return torch.cat(out, dim=-1)


def _expand_per_irrep(gamma: torch.Tensor, irreps: Irreps) -> torch.Tensor:
    """(num_irreps,) per-channel gains -> (dim,) over the mul_ir layout (models.race._scale_irreps)."""
    parts, off = [], 0
    for mul, l, _ in irreps:
        parts.append(gamma[off:off + mul, None].expand(mul, 2 * l + 1).reshape(-1))
        off += mul
    return torch.cat(parts)


# ===================================================================================================== modules
class RACEXReadout(nn.Module):
    """models.race.RACEXReadout (norm=True, linear=False): separated LN / sqrt(<N>) -> indexed Linear -> Linear(+b)
    -> normalised silu -> Linear(+b) -> 0e."""

    def __init__(self, hidden_irreps: Irreps, x_irreps: Irreps, n_species: int, avg_n_neighbors: float,
                 species_rank: Optional[int]):
        super().__init__()
        self.x_irreps = Irreps(x_irreps)
        self.avg_n_neighbors = float(avg_n_neighbors)
        self.x_readout1 = EquivariantLinear(x_irreps, hidden_irreps, force_irreps_out=True, num_indexed=n_species,
                                            index_rank=species_rank)
        self.x_readout2 = EquivariantLinear(hidden_irreps, hidden_irreps.div_mul(2), biases=True,
                                            force_irreps_out=True)
        self.x_readout3 = EquivariantLinear(hidden_irreps.div_mul(2), "0e", biases=True, force_irreps_out=True)
        self.act = NormalizedSiLU()

    def forward(self, species, x, groups=None):
        if self.x_irreps.dim > 1:
            xn = separated_layer_norm(x, self.x_irreps) / math.sqrt(self.avg_n_neighbors)
        else:
            xn = x / math.sqrt(self.avg_n_neighbors)
        h = self.x_readout1(xn, species, groups)
        h = self.x_readout2(h)
        h = self.act(h)
        return self.x_readout3(h)


class RACEConvolution(nn.Module):
    """models.race.RACEConvolution for x_mode basis, update linear, message_norm avg, readout gated, residual
    chebyshev (in_layer_order 1, no sym contraction, no msg_lmax)."""

    def __init__(self, input_irreps: Irreps, hidden_irreps: Irreps, sh_irreps: Irreps, n_species: int,
                 n_radial: int, radial_mlp_size: int, radial_mlp_layers: int, avg_n_neighbors: float,
                 layer_scale: Optional[float], radial_scale: Optional[Sequence[float]], species_rank: Optional[int],
                 species_rank_lpos: Optional[int]):
        super().__init__()
        self.input_irreps = Irreps(input_irreps)
        self.output_irreps = Irreps(hidden_irreps)
        self.avg_n_neighbors = float(avg_n_neighbors)
        self.radial_scale = None if radial_scale is None else tuple(float(s) for s in radial_scale)
        if self.radial_scale is not None:
            self.register_buffer("radial_scale_t", torch.tensor(self.radial_scale, dtype=torch.float64),
                                 persistent=False)
        self.tp = EdgeTensorProduct(self.input_irreps, sh_irreps, self.output_irreps)
        self.tp_irreps = self.tp.irreps_out                      # not regrouped (cue output irreps)
        self.linear_1 = EquivariantLinear(self.input_irreps, self.input_irreps)
        self.radial_mlp = RadialMLP([n_radial] + [radial_mlp_size] * radial_mlp_layers + [self.tp.n_weights])
        if layer_scale is not None:
            self.gamma = nn.Parameter(torch.full((self.output_irreps.num_irreps,), float(layer_scale),
                                                 dtype=torch.float64))
        else:
            self.gamma = None
        self.linear_m = EquivariantLinear(self.tp_irreps, self.output_irreps, force_irreps_out=True)
        self.xprod = ChannelwiseProduct(self.output_irreps)
        self.linear_2 = EquivariantLinear(self.xprod.irreps_out, self.output_irreps, force_irreps_out=True)
        self.skip = EquivariantLinear(self.input_irreps, self.output_irreps, force_irreps_out=True,
                                      num_indexed=n_species, index_rank=species_rank,
                                      index_rank_lpos=species_rank_lpos)
        ro_irreps = self.output_irreps.div_mul(2).filter_keep({(0, 1)})
        self.readout1 = EquivariantLinear(self.output_irreps, ro_irreps, biases=True, force_irreps_out=True)
        self.readout = EquivariantLinear(ro_irreps, "0e", biases=True)
        self.act = NormalizedSiLU()

    def aggregate(self, features, sh, radial_basis, senders, receivers, edge_chunk: int = 0):
        """m_i = (1 / <N>) sum_j R(r_ij) (W_1 h_j (x) Y(r_ij)) (message_norm avg); with the species embedding as
        input this is the atomic basis A_i."""
        if self.radial_scale is not None:
            radial_basis = radial_basis * self.radial_scale_t.to(radial_basis.dtype)[None, :]
        node_in = self.linear_1(features)
        n = features.shape[0]

        def run(rb, y, s, r, h):
            return self.tp(h, y, self.radial_mlp(rb), s, r, n)

        e = senders.shape[0]
        if edge_chunk and e > edge_chunk:
            from torch.utils.checkpoint import checkpoint
            agg = None
            for a in range(0, e, edge_chunk):
                b = min(e, a + edge_chunk)
                part = checkpoint(run, radial_basis[a:b], sh[a:b], senders[a:b], receivers[a:b], node_in,
                                  use_reentrant=False)
                agg = part if agg is None else agg + part
        else:
            agg = run(radial_basis, sh, senders, receivers, node_in)
        return agg / self.avg_n_neighbors

    def forward(self, features, x_feats, species, groups, sh, radial_basis, senders, receivers,
                prev_features=None, edge_chunk: int = 0):
        m = self.aggregate(features, sh, radial_basis, senders, receivers, edge_chunk)
        m = self.linear_m(m)
        prod = self.xprod(m, x_feats)
        delta = self.linear_2(prod) + self.skip(features, species, groups)
        if self.gamma is not None:
            delta = delta * _expand_per_irrep(self.gamma, self.output_irreps)[None, :]
        if prev_features is not None and self.input_irreps == self.output_irreps:
            new_features = 2.0 * delta - prev_features
        else:
            new_features = delta
        e = self.readout1(new_features)
        e = self.act(e)
        e = self.readout(e)
        return e, new_features


class RACEv2(nn.Module):
    """RACEv2 (models.race.RACE) in float64.  Build with ``RACEv2(resolve_config(cfg))`` or ``RACEv2.load(path)``."""

    def __init__(self, hp: Dict):
        super().__init__()
        self.hp = dict(hp)
        from . import constants as C
        z = [int(v) for v in hp['atomic_numbers']]
        self.atomic_numbers = z
        self.n_species = n_species = len(z)
        self.lmax = int(hp['lmax'])
        self.cutoff = float(hp['cutoff'])
        self.periodic = bool(hp['periodic'])
        self.radial_basis_size = int(hp['radial_basis_size'])
        self.radial_polynomial_p = float(hp['radial_polynomial_p'])
        self.avg_n_neighbors = float(hp['avg_n_neighbors'])
        self.n_layers = int(hp['n_layers'])
        self.hidden_irreps = Irreps(hp['hidden_irreps'])
        self.sh_irreps = o3.s2_irreps(self.lmax)
        self.embed_irreps = self.hidden_irreps.filter_keep({(0, 1)})
        species_rank = hp.get('species_rank')
        prune = bool(hp.get('species_rank_prune', False)) and species_rank is not None
        self.register_buffer("atom_energies", torch.tensor(hp['atom_energies'], dtype=torch.float64),
                             persistent=False)
        # radial shells (n_species, 4) [A]
        self.radial_shells = bool(hp.get('radial_shells', False))
        n_extra = 0
        if self.radial_shells:
            rho = C.shell_radii(z)
            self.n_shells = len(rho[0])
            n_extra = self.n_shells ** 2
            self.register_buffer("species_shell_radii", torch.tensor(rho, dtype=torch.float64), persistent=False)
        # pair repulsion (UFF + switch)
        self.pair_repulsion = hp.get('pair_repulsion') or False
        self.pair_repulsion_scale = float(hp.get('pair_repulsion_scale', 1.0))
        self.pair_repulsion_switch = hp.get('pair_repulsion_switch')
        self.pair_repulsion_switch_width = float(hp.get('pair_repulsion_switch_width', 0.1))
        if self.pair_repulsion:
            self.register_buffer("uff_table", torch.tensor(C.uff_rows(z), dtype=torch.float64), persistent=False)
            self.register_buffer("species_radii", torch.tensor(C.covalent_radii(z), dtype=torch.float64),
                                 persistent=False)
        # radial regularity prior (models.race.RACE.__init__ radial_scales)
        radial_scales = [None] * self.n_layers
        rp = hp.get('radial_prior')
        if rp:
            rp_type = str(rp.get('type', 'algebraic'))
            sig = rp.get('sigma', 0.5)
            sigmas = [float(sig)] * self.n_layers if not isinstance(sig, (list, tuple)) else [float(s) for s in sig]
            if len(sigmas) != self.n_layers:
                raise ValueError("radial_prior sigma list must have n_layers entries")
            p_rp = float(rp.get('p', 2.0))
            n_idx = range(1, self.radial_basis_size + 1)
            for t, s_rp in enumerate(sigmas):
                if rp_type == "algebraic":
                    gam = [(1.0 + s_rp * n) ** p_rp for n in n_idx]
                elif rp_type == "exponential":
                    gam = [math.exp(s_rp * n) for n in n_idx]
                else:
                    gam = [math.exp(s_rp ** 2 * n ** 2) for n in n_idx]
                radial_scales[t] = tuple(1.0 / g for g in gam) + (1.0,) * n_extra
        self.radial_scales = tuple(radial_scales)

        self.species_embedding = EquivariantLinear(f"{n_species}x0e", self.embed_irreps)
        self.sph = o3.SphericalHarmonics(self.lmax)
        layers, x_readouts, f_readouts, x_projs = [], [], [], []
        ro_hidden = self.hidden_irreps.filter_keep({(0, 1)})
        for i in range(self.n_layers):
            inp = self.embed_irreps if i == 0 else self.hidden_irreps
            layers.append(RACEConvolution(
                inp, self.hidden_irreps, self.sh_irreps, n_species, self.radial_basis_size + n_extra,
                int(hp['radial_mlp_size']), int(hp['radial_mlp_layers']), self.avg_n_neighbors,
                layer_scale=(hp.get('layer_scale') if i >= 1 else None), radial_scale=radial_scales[i],
                species_rank=species_rank, species_rank_lpos=(1 if (prune and i == self.n_layers - 1) else None)))
            x_readouts.append(RACEXReadout(ro_hidden, self.hidden_irreps, n_species, self.avg_n_neighbors,
                                           species_rank))
            f_readouts.append(RACEXReadout(ro_hidden, inp, n_species, self.avg_n_neighbors, species_rank))
        self.layers = nn.ModuleList(layers)
        self.x_readouts = nn.ModuleList(x_readouts)
        self.f_readouts = nn.ModuleList(f_readouts)
        for i in range(self.n_layers):
            x_projs.append(EquivariantLinear(self.layers[0].tp_irreps, self.hidden_irreps, force_irreps_out=True))
        self.x_projs = nn.ModuleList(x_projs)
        self.x_hill = nn.Parameter(torch.zeros(self.n_layers, 2, n_species, dtype=torch.float64))
        self.shift = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.scale = nn.Parameter(torch.ones((), dtype=torch.float64))
        self.edge_chunk = 0                     # > 0: edge tensor products in chunks with recomputation (memory)

    # ----------------------------------------------------------------------------------------------- I/O
    @classmethod
    def from_config(cls, cfg: Dict) -> "RACEv2":
        return cls(resolve_config(cfg))

    def save(self, path, config: Optional[Dict] = None, extra: Optional[Dict] = None):
        torch.save({'format': FORMAT, 'format_version': FORMAT_VERSION, 'hyperparameters': self.hp,
                    'config': config, 'state_dict': {k: v.detach().cpu() for k, v in self.state_dict().items()},
                    'extra': extra or {}}, path)

    @classmethod
    def load(cls, path, map_location='cpu') -> "RACEv2":
        d = torch.load(path, map_location=map_location, weights_only=False)
        if d.get('format') != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT} model file")
        if int(d.get('format_version', 0)) > FORMAT_VERSION:
            raise ValueError(f"{path}: format version {d['format_version']} is newer than this code ({FORMAT_VERSION})")
        model = cls(d['hyperparameters'])
        model.load_state_dict(d['state_dict'], strict=True)
        model.source_config = d.get('config')
        model.extra = d.get('extra', {})
        return model.eval()

    # ----------------------------------------------------------------------------------------------- forward pieces
    def edge_vectors(self, positions, senders, receivers, shifts=None, cell=None, n_edge=None):
        """Edge vectors of utils.compute_node_edge_features  r = x_s - (x_r + S.cell)  and of the _prelude distance
        branch  (x_s - x_r) - S.cell  (the same up to rounding; both kept for bit-faithfulness)."""
        if shifts is not None and cell is not None:
            e = senders.shape[0]
            g_of_e = torch.repeat_interleave(torch.arange(cell.shape[0], device=cell.device), n_edge, output_size=e)
            sh = torch.einsum('ei,eij->ej', shifts.to(positions.dtype), cell[g_of_e])
            r_feat = positions[senders] - (positions[receivers] + sh)
            r_dist = (positions[senders] - positions[receivers]) - sh
        else:
            r_feat = positions[senders] - positions[receivers]
            r_dist = r_feat
        return r_feat, r_dist

    def prelude(self, species, senders, receivers, r_feat, r_dist, cutoff_e=None) -> Dict:
        """models.race.RACE._prelude: edge features, species embedding, UFF repulsion stratum, atomic basis A_i.

        r_feat / r_dist: edge vectors (see edge_vectors); cutoff_e: per-edge cutoff of the Bessel basis and its
        envelope (periodic graphs: globals['cutoff'] per graph, clamped to >= 1; default the model cutoff)."""
        n = species.shape[0]
        dtype = r_feat.dtype
        e = senders.shape[0]
        if cutoff_e is None:
            cutoff_e = torch.full((e,), self.cutoff, dtype=dtype, device=r_feat.device)
        cutoff_e = torch.where(cutoff_e > 1.0, cutoff_e, torch.ones_like(cutoff_e))
        r_norm_f = _safe_norm(torch.sum(r_feat ** 2, dim=-1))
        radial_basis = (bessel_basis(r_norm_f, self.radial_basis_size, cutoff_e)
                        * polynomial_cutoff(r_norm_f, cutoff_e, self.radial_polynomial_p)[:, None])
        sh = self.sph(r_feat)
        r2 = torch.sum(r_dist ** 2, dim=-1)
        r_norm = _safe_norm(r2)
        r_c = torch.full((e,), self.cutoff, dtype=dtype, device=r_feat.device)
        shell_env = None
        if self.radial_shells:
            rho = self.species_shell_radii.to(dtype)
            rho_i = rho[species[receivers]]
            rho_j = rho[species[senders]]
            denom = rho_i[:, :, None] + rho_j[:, None, :]
            present = (rho_i[:, :, None] > 0.0) & (rho_j[:, None, :] > 0.0)
            env = torch.where(present, torch.exp(-r_norm[:, None, None] / torch.where(present, denom,
                                                                                     torch.ones_like(denom))),
                              torch.zeros_like(denom))
            env = env.reshape(e, self.n_shells ** 2)
            env = env * polynomial_cutoff(r_norm, r_c, self.radial_polynomial_p)[:, None]
            shell_env = env
            radial_basis = torch.cat([radial_basis, env], dim=-1)
        edge_ok = (r2 > 0.0).to(dtype)
        radial_basis = radial_basis * edge_ok[:, None]
        sh = sh * edge_ok[:, None]
        if shell_env is not None:
            shell_env = shell_env * edge_ok[:, None]
        onehot = torch.nn.functional.one_hot(species.long(), self.n_species).to(dtype)
        features = self.species_embedding(onehot)
        outputs = []
        if self.pair_repulsion:
            sp_i, sp_j = species[receivers], species[senders]
            f_cut = polynomial_cutoff(r_norm, r_c, self.radial_polynomial_p)
            tab = self.uff_table.to(dtype)
            xi, xj = tab[sp_i, 0], tab[sp_j, 0]
            di, dj = tab[sp_i, 1], tab[sp_j, 1]
            zi, zj = tab[sp_i, 2], tab[sp_j, 2]
            x_ij = torch.sqrt(xi * xj)
            d_ij = torch.sqrt(di * dj)
            z_ij = torch.sqrt(zi * zj)
            e_edge = (self.pair_repulsion_scale * d_ij * (6.0 / (z_ij - 6.0)) * torch.exp(z_ij * (1.0 - r_norm / x_ij))
                      * f_cut)
            if self.pair_repulsion_switch is not None:
                rad = self.species_radii.to(dtype)
                r_sw = self.pair_repulsion_switch * (rad[sp_i] + rad[sp_j])
                e_edge = e_edge * torch.sigmoid((r_norm - r_sw) / self.pair_repulsion_switch_width)
            e_edge = torch.where(r_norm > 0.0, e_edge, torch.zeros_like(e_edge))
            e_rep = 0.5 * torch.zeros(n, dtype=dtype, device=r_feat.device).index_add(0, receivers, e_edge)
            outputs.append(e_rep)
        groups = SpeciesGroups(species)
        atomic_basis = self.layers[0].aggregate(features, sh, radial_basis, senders, receivers, self.edge_chunk)
        return dict(features=features, radial_basis=radial_basis, sh=sh, species=species, groups=groups,
                    senders=senders, receivers=receivers, r_norm=r_norm, shell_env=shell_env,
                    atomic_basis=atomic_basis, outputs=outputs, x_feats=None,
                    layer_parts={"node": [], "x": [], "f": []})

    def x_features(self, ctx: Dict, i: int) -> torch.Tensor:
        """x_i of layer i: hill-normalised x_proj_i(A_i) (models.race._hill_normalize)."""
        x = self.x_projs[i](ctx['atomic_basis'])
        ab = self.x_hill[i]
        sp = ctx['species']
        ms = torch.mean(x * x, dim=-1, keepdim=True) + 1e-6
        g = 2.0 * torch.sigmoid(ab[0][sp][:, None] * 0.5 * torch.log(ms) + ab[1][sp][:, None])
        return x * torch.rsqrt(ms) * g

    def layer_step(self, ctx: Dict, i: int, features, prev_features=None):
        """models.race.RACE._layer_step: interaction layer i -> (new node features, energy stratum of the layer).
        prev_features: input features of layer i-1 (Chebyshev residual, i >= 2), else None."""
        layer = self.layers[i]
        sp, groups = ctx['species'], ctx['groups']
        x_feats = self.x_features(ctx, i)
        e_f = self.f_readouts[i](sp, features, groups)[:, 0]
        e_node, features = layer(features, x_feats, sp, groups, ctx['sh'], ctx['radial_basis'], ctx['senders'],
                                 ctx['receivers'], prev_features=prev_features, edge_chunk=self.edge_chunk)
        e_x = self.x_readouts[i](sp, x_feats, groups)[:, 0]
        e_node = e_node[:, 0]
        gain = 1.0
        ctx['layer_parts']['node'].append(gain * e_node)
        ctx['layer_parts']['x'].append(gain * e_x)
        ctx['layer_parts']['f'].append(gain * e_f)
        ctx['x_feats'] = x_feats
        return features, gain * (e_node + e_x + e_f)

    def finalize(self, ctx: Dict, features) -> Dict:
        """models.race.RACE._finalize: node energy = scale * (sum of the strata) + shift (variance heads omitted)."""
        outputs = ctx['outputs']
        strata = torch.stack(outputs, dim=0) * self.scale
        node_energies = torch.sum(torch.stack(outputs, dim=1), dim=-1, keepdim=True)
        node_energies = node_energies * self.scale + self.shift
        return {'node_energies': node_energies, 'node_features': features, 'strata': strata,
                'strata_parts': {k: torch.stack(v, dim=0) * self.scale for k, v in ctx['layer_parts'].items()}}

    def run_layers(self, ctx: Dict) -> Dict:
        """All layers with the Chebyshev history, then finalize (models.race.RACE.node_energies)."""
        features = ctx['features']
        hist = []
        rec = ctx.get('record')                 # dict: keep layer inputs / outputs / x features (validation)
        for i in range(self.n_layers):
            hist.append(features)
            prev = hist[i - 1] if i >= 2 else None
            if rec is not None:
                rec[f'layer{i}_in'] = features
            features, stratum = self.layer_step(ctx, i, features, prev)
            ctx['outputs'].append(stratum)
            if rec is not None:
                rec[f'layer{i}_out'] = features
                rec[f'layer{i}_x'] = ctx['x_feats']
                rec[f'layer{i}_stratum'] = stratum
        return self.finalize(ctx, features)

    def node_energies(self, positions, species, senders, receivers, shifts=None, cell=None, n_edge=None,
                      cutoff=None, record: bool = False) -> Dict:
        """Per-node energies (n_nodes, 1) without E0.  Periodic: shifts (E, 3) = Sij, cell (G, 3, 3), n_edge (G,),
        cutoff (G,) (default model cutoff)."""
        r_feat, r_dist = self.edge_vectors(positions, senders, receivers, shifts, cell, n_edge)
        cutoff_e = None
        if cutoff is not None and n_edge is not None:
            cutoff_e = torch.repeat_interleave(cutoff.to(positions.dtype), n_edge, output_size=senders.shape[0])
        ctx = self.prelude(species, senders, receivers, r_feat, r_dist, cutoff_e)
        if record:
            ctx['record'] = {}
        out = self.run_layers(ctx)
        out['ctx'] = ctx
        return out

    def forward(self, positions, species, senders, receivers, shifts=None, cell=None, n_node=None, n_edge=None,
                cutoff=None, volume=None, node_mask=None, compute_forces: bool = True,
                compute_stress: Optional[bool] = None, e0: Optional[str] = None, record: bool = False) -> Dict:
        """Energies (G,), forces (N, 3), stress (G, 6) (Voigt xx yy zz yz xz xy, eV/A^3, = (1/V) dE/d strain as in
        utils.energy_forces_stress).  ``energy`` excludes E0 (as the JAX model built with l_train=True, i.e. what
        ``RACE.__call__`` returns in predict.py); e0="float64" adds sum E0 in float64 (predict.py), "float32" adds
        float32-rounded E0 per atom (utils.energy_forces_stress with l_train=False).
        node_mask: (N,) bool of real nodes (padded graphs); default all."""
        n = positions.shape[0]
        species, senders, receivers = species.long(), senders.long(), receivers.long()
        if n_node is None:
            n_node = torch.tensor([n], device=positions.device)
        if n_edge is None and cell is not None:
            n_edge = torch.tensor([senders.shape[0]], device=positions.device)
        g = int(n_node.shape[0])
        periodic = shifts is not None and cell is not None
        if compute_stress is None:
            compute_stress = periodic
        pos = positions.detach().clone().requires_grad_(compute_forces or compute_stress)
        cel = None
        if periodic:
            cel = cell.detach().clone().to(positions.dtype).requires_grad_(compute_stress)
        out = self.node_energies(pos, species, senders, receivers, shifts, cel, n_edge, cutoff, record=record)
        ne = out['node_energies'].reshape(-1)
        if node_mask is not None:
            ne = torch.where(node_mask, ne, torch.zeros_like(ne))
        node_graph = torch.repeat_interleave(torch.arange(g, device=positions.device), n_node, output_size=n)
        res = {'node_energies': ne, 'strata': out['strata'], 'ctx': out['ctx'], 'model_out': out}
        if compute_forces or compute_stress:
            inputs = [pos] + ([cel] if (periodic and compute_stress) else [])
            grads = torch.autograd.grad(ne.sum(), inputs, allow_unused=True)
            grad = grads[0] if grads[0] is not None else torch.zeros_like(pos)
            grad = torch.nan_to_num(grad, nan=0.0)
            if node_mask is not None:
                grad = torch.where(node_mask[:, None], grad, torch.zeros_like(grad))
            res['forces'] = -grad
            if periodic and compute_stress:
                cellgrad = grads[1] if grads[1] is not None else torch.zeros_like(cel)
                cellgrad = torch.nan_to_num(cellgrad, nan=0.0)
                if volume is None:
                    volume = torch.abs(torch.linalg.det(cel.detach()))
                volume = torch.where(volume > 0.0, volume, torch.ones_like(volume))
                stress_cell = torch.transpose(cellgrad, 1, 2) @ cel.detach()
                sg = torch.einsum('iu,iv->iuv', grad, pos.detach())
                stress_grad = torch.zeros(g, 3, 3, dtype=sg.dtype, device=sg.device).index_add(0, node_graph, sg)
                virial = (stress_cell + stress_grad).reshape(-1, 9)[:, [0, 4, 8, 5, 2, 1]]
                res['stress'] = virial / volume[:, None]
        ne_tot = ne
        if e0 == 'float64':
            ne_tot = ne_tot + self.atom_energies.to(ne.dtype)[species]
        elif e0 == 'float32':
            ne_tot = ne_tot + self.atom_energies.to(torch.float32).to(ne.dtype)[species]
        elif e0 is not None:
            raise ValueError("e0 must be None, 'float64' or 'float32'")
        if e0 is not None and node_mask is not None:
            ne_tot = torch.where(node_mask, ne_tot, torch.zeros_like(ne_tot))
        res['node_energies_total'] = ne_tot
        res['energy'] = torch.zeros(g, dtype=ne.dtype, device=ne.device).index_add(0, node_graph, ne_tot)
        return res
