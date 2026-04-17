"""NequIP model — PyTorch port using cuequivariance instead of e3nn."""

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import cuequivariance as cue
import cuequivariance_torch as cuet

from bam_torch.data import TorchGraphData
from bam_torch_cue.linear import Linear
from bam_torch_cue.utils import (
    LAYOUT,
    MLP,
    build_cg_tp,
    build_norm_info,
    compute_node_edge_features,
    cue_irreps,
    energy_forces_stress,
    filter_scalar_irreps,
    is_scalar_ir,
    scatter_sum,
    separated_layer_norm,
    spherical_harmonics_irreps,
)


# =============================================================================
# Gated activation (replaces e3nn.nn.Gate)
# =============================================================================

class EquivariantGate(nn.Module):
    """Gated equivariant activation.

    Splits input into scalars, gate values, and gated features.
    - Scalars: silu activation
    - Gates: silu → multiply into gated features
    - Output: activated_scalars + gated_features = output_irreps

    Matches e3nn_jax gate(even_act=silu, even_gate_act=silu).

    Uses ir_mul layout: within each (mul, ir) segment, data is arranged as
    [comp0_mul0..mulM, comp1_mul0..mulM, ...].
    """

    def __init__(self, output_irreps: cue.Irreps):
        super().__init__()
        scalars = [(mi.mul, mi.ir) for mi in output_irreps if is_scalar_ir(mi.ir)]
        gated   = [(mi.mul, mi.ir) for mi in output_irreps if not is_scalar_ir(mi.ir)]

        if not gated:
            # All-scalar: no gate module needed, use plain silu
            self.irreps_in = output_irreps
            self.has_gate = False
            return

        self.has_gate = True
        n_gates = sum(mul for mul, _ in gated)

        self.irreps_scalars = cue.Irreps("O3", scalars)
        self.irreps_gated   = cue.Irreps("O3", gated)
        self.irreps_gates   = cue_irreps(f"{n_gates}x0e")

        # Input irreps: scalars + gates + gated
        self.irreps_in = self.irreps_scalars + self.irreps_gates + self.irreps_gated

        self.scalar_dim = self.irreps_scalars.dim
        self.gate_dim   = self.irreps_gates.dim   # = n_gates
        self.gated_dim  = self.irreps_gated.dim

        # Store (mul, ir.dim) for each gated segment for reshaping
        self._gated_segments = [(mi.mul, mi.ir.dim) for mi in self.irreps_gated]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.has_gate:
            return F.silu(x)

        scalars = x[:, :self.scalar_dim]
        gates   = x[:, self.scalar_dim:self.scalar_dim + self.gate_dim]
        gated   = x[:, self.scalar_dim + self.gate_dim:]

        activated_scalars = F.silu(scalars)
        gate_values = F.silu(gates)

        # Apply gates to each gated irrep segment.
        # ir_mul layout: (mul, ir) → [comp0_mul0..M, comp1_mul0..M, ...comp{D-1}_mul0..M]
        gate_idx = 0
        gated_parts = []
        pos = 0
        for mul, ir_dim in self._gated_segments:
            seg = gated[:, pos:pos + mul * ir_dim]                # (batch, D*M)
            g   = gate_values[:, gate_idx:gate_idx + mul]         # (batch, M)
            # Reshape to (batch, D, M), gate broadcasts over D
            seg = seg.view(-1, ir_dim, mul) * g.unsqueeze(1)     # (batch, D, M)
            gated_parts.append(seg.view(-1, mul * ir_dim))
            gate_idx += mul
            pos += mul * ir_dim

        return torch.cat([activated_scalars] + gated_parts, dim=-1)


def _build_gate(output_irreps: cue.Irreps):
    """Build an EquivariantGate module.

    If output_irreps is all-scalar, returns None (caller uses plain silu).
    """
    has_nonscalar = any(not is_scalar_ir(mi.ir) for mi in output_irreps)
    if not has_nonscalar:
        return None
    return EquivariantGate(output_irreps)


# =============================================================================
# NequIP convolution layer
# =============================================================================

class NequipConvolution(nn.Module):
    """Single NequIP equivariant message-passing layer.

    Uses a channelwise CG tensor product with external radial weights
    (from cuequivariance), replacing the old e3nn TensorProduct +
    scale_irreps_by_scalar pattern.
    """

    def __init__(
        self,
        input_irreps: cue.Irreps,
        output_irreps: cue.Irreps,
        sh_irreps: cue.Irreps,
        n_species: int,
        radial_basis_size: int,
        radial_mlp_size: int,
        radial_mlp_layers: int,
        mlp_init_scale: float,
        avg_n_neighbors: float,
    ):
        super().__init__()
        self.output_irreps   = output_irreps
        self.avg_n_neighbors = avg_n_neighbors
        self.input_irreps    = input_irreps

        # Channelwise CG tensor product with external weights
        self.tp, self.tp_irreps = build_cg_tp(input_irreps, sh_irreps, output_irreps)
        # Radial MLP: output size = tp.weight_numel (one weight per TP path)
        radial_out = self.tp.weight_numel

        self.linear_1 = Linear(input_irreps, input_irreps)

        self.radial_mlp = MLP(
            [radial_basis_size] + [radial_mlp_size] * radial_mlp_layers + [radial_out],
            init_scale=mlp_init_scale,
            use_bias=False,
        )

        # Gate (may be None if output_irreps is all-scalar)
        gate = _build_gate(output_irreps)
        self.gate = gate
        gate_input_irreps = gate.irreps_in if gate is not None else output_irreps

        # linear_2 projects from tp_irreps → gate_input_irreps
        self.linear_2 = Linear(self.tp_irreps, gate_input_irreps)
        self.skip      = Linear(input_irreps, gate_input_irreps,
                                num_indexed_weights=n_species,
                                force_irreps_out=True)

        # Precomputed norm info
        self._input_norm_info = build_norm_info(input_irreps)

    def forward(
        self,
        features: torch.Tensor,        # (N, input_irreps.dim)
        species: torch.Tensor,          # (N,) long
        sh: torch.Tensor,               # (E, sh_irreps.dim)
        radial_basis: torch.Tensor,     # (E, radial_basis_size)
        senders: torch.Tensor,          # (E,)
        receivers: torch.Tensor,        # (E,)
    ) -> torch.Tensor:                  # (N, output_irreps.dim)
        n_nodes = features.shape[0]

        features_norm = separated_layer_norm(features, self._input_norm_info)
        messages = self.linear_1(features_norm)
        # Channelwise CG TP with radial weights (absorbs CG + radial scaling)
        radial_weights = self.radial_mlp(radial_basis)                  # (E, weight_numel)
        messages = self.tp(messages[senders], sh, weight=radial_weights) # (E, tp_irreps.dim)

        messages_agg = scatter_sum(messages, receivers, n_nodes) / (self.avg_n_neighbors ** 0.5)

        out = self.linear_2(messages_agg) + self.skip(features, species)
        if self.gate is not None:
            return self.gate(out)
        return F.silu(out)


# =============================================================================
# NequIP model
# =============================================================================

class Nequip(nn.Module):
    """NequIP model for energies, forces, and stress.

    Args match bam_omat24/nequip_nnx.py:Nequip exactly.
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
    ):
        super().__init__()
        self.lmax                = lmax
        self.cutoff              = cutoff
        self.periodic            = periodic
        self.n_species           = n_species
        self.radial_basis_size   = radial_basis_size
        self.radial_polynomial_p = radial_polynomial_p
        self.l_train             = l_train

        self.shift          = nn.Parameter(torch.tensor(shift))
        self.scale          = nn.Parameter(torch.tensor(scale))
        self.avg_n_neighbors = avg_n_neighbors

        if atom_energies is not None:
            self.register_buffer('atom_energies', torch.tensor(atom_energies, dtype=torch.float32))
        else:
            self.register_buffer('atom_energies', torch.zeros(n_species))

        hidden = cue_irreps(hidden_irreps)
        self.hidden_irreps = hidden
        input_irreps = filter_scalar_irreps(hidden)
        self.sh_irreps = spherical_harmonics_irreps(lmax)

        # Spherical harmonics module (created once, reused every forward pass)
        self.sh_module = cuet.SphericalHarmonics(
            list(range(lmax + 1)), normalize=True,
        )

        self.species_embedding = Linear(
            cue_irreps(f"{n_species}x0e"), input_irreps
        )

        layers = []
        for i in range(n_layers):
            inp = input_irreps if i == 0 else hidden
            layers.append(NequipConvolution(
                input_irreps=inp,
                output_irreps=hidden,
                sh_irreps=self.sh_irreps,
                n_species=n_species,
                radial_basis_size=radial_basis_size,
                radial_mlp_size=radial_mlp_size,
                radial_mlp_layers=radial_mlp_layers,
                mlp_init_scale=mlp_init_scale,
                avg_n_neighbors=avg_n_neighbors,
            ))
        self.layers = nn.ModuleList(layers)

        # Readout
        hidden_scalar_dim = filter_scalar_irreps(hidden).dim
        hidden2_irreps = cue_irreps(f"{hidden_scalar_dim // 2}x0e")
        self.readout1 = Linear(hidden, hidden2_irreps, biases=True)
        self.readout  = Linear(hidden2_irreps, cue_irreps("1x0e"), biases=True)

        # Variance heads
        self.frc_logvar    = Linear(hidden, cue_irreps("6x0e"))
        self.energy_logvar = Linear(hidden, cue_irreps("1x0e"))
        self.stress_logvar = Linear(hidden, cue_irreps("6x0e"))

    def node_energies(
        self,
        positions: torch.Tensor,
        cell: Optional[torch.Tensor],
        cutoff: Optional[torch.Tensor],
        data: TorchGraphData,
    ) -> Dict:
        node_features, radial_basis, sh = compute_node_edge_features(
            positions=positions, cell=cell, cutoff=cutoff, data=data,
            n_species=self.n_species,
            model_cutoff=self.cutoff,
            radial_basis_size=self.radial_basis_size,
            radial_polynomial_p=self.radial_polynomial_p,
            lmax=self.lmax, periodic=self.periodic,
            sh_module=self.sh_module,
        )

        features = self.species_embedding(node_features)
        for layer in self.layers:
            features = layer(
                features, data.nodes['species'],
                sh, radial_basis,
                data.senders, data.receivers,
            )

        # Variance heads
        frc_raw = self.frc_logvar(features)     # (N,6)
        frc_var = torch.cat([
            F.softplus(frc_raw[:, :3]) + 0.04,
            frc_raw[:, 3:],
        ], dim=-1)

        energy_var = F.softplus(self.energy_logvar(features)) + 0.04   # (N,1)
        stress_var = F.softplus(self.stress_logvar(features)) + 0.04   # (N,6)

        # Energy readout
        h = self.readout1(features)
        ne = self.readout(h) * self.scale + self.shift   # (N,1)

        return {
            'node_energies':    ne.squeeze(-1),
            'node_features':    features,
            'frc_variation':    frc_var,
            'energy_variation': energy_var,
            'stress_variation': stress_var,
        }

    def forward(self, data: TorchGraphData) -> Dict:
        return energy_forces_stress(
            self.node_energies, data,
            self.periodic, self.l_train,
            self.atom_energies,
            create_graph=self.training,
        )


if __name__ == "__main__":
    import pickle
    import numpy as np
    from bam_torch.atom_energies import ATOM_ENERGIES
    from bam_torch.data import GraphData, batch_graphs

    def jraph_to_graphdata(g) -> GraphData:
        """Convert a jraph.GraphsTuple (from existing pkl files) to GraphData."""
        return GraphData(
            n_node=np.asarray(g.n_node),
            n_edge=np.asarray(g.n_edge),
            nodes={k: np.asarray(v) for k, v in g.nodes.items()},
            edges={k: np.asarray(v) for k, v in g.edges.items()},
            senders=np.asarray(g.senders),
            receivers=np.asarray(g.receivers),
            globals={k: np.asarray(v) for k, v in g.globals.items()},
        )

    print("=" * 70)
    print("NequIP Equivariant Potential (PyTorch + cuequivariance)")
    print("=" * 70)

    with open("omat24_val_0.pkl", "rb") as f:
        graphset = pickle.load(f)

    data_np = jraph_to_graphdata(graphset[0])
    data = batch_graphs([data_np]).to(torch.device("cpu"))

    model = Nequip(
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
        avg_n_neighbors=25.0,
        atom_energies=ATOM_ENERGIES.tolist(),
        periodic=True,
        l_train=True,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nNumber of parameters: {n_params:,}")

    results = model(data)

    energies = results["energy"]
    forces   = results["forces"]
    stress   = results["stress"]

    print(f"Energies shape : {energies.shape}")
    print(f"Forces shape   : {forces.shape}")
    print(f"Total energy   : {energies[0].item():.6f} eV")
    print(f"Force magnitude: {forces.norm(dim=-1).mean().item():.6f} eV/Å")
    print(f"Stress (Voigt) : {stress[0].detach().numpy()}")
    print(f"\nTarget energy  : {data_np.globals['energy'][0]:.6f} eV")
    print(f"Target stress  : {data_np.globals['stress'][0]} eV/Å³")
