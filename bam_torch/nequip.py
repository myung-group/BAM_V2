"""NequIP model — PyTorch port of bam_omat24/nequip_nnx.py."""

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn import Gate

from bam_torch.data import TorchGraphData
from bam_torch.linear import Linear
from bam_torch.utils import (
    MLP,
    build_cg_tp,
    build_norm_info,
    build_weight_expand_index,
    compute_node_edge_features,
    energy_forces_stress,
    scale_irreps_by_scalar,
    scatter_sum,
    separated_layer_norm,
)


def _build_gate(output_irreps: o3.Irreps):
    """Build a Gate module + its required input irreps.

    Mirrors the JAX:
        gate_irreps = (num_irreps - count_0e) x 0e
        gated_output = (output_irreps + gate_irreps).regroup()

    If output_irreps is all-scalar, returns None (caller uses plain silu).
    """
    irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in output_irreps if ir.is_scalar()])
    irreps_gated   = o3.Irreps([(mul, ir) for mul, ir in output_irreps if not ir.is_scalar()])

    if len(irreps_gated) == 0:
        # All-scalar: no gate module needed
        return None

    n_gates      = sum(mul for mul, _ in irreps_gated)
    irreps_gates = o3.Irreps(f"{n_gates}x0e")

    act_scalars = [F.silu] * len(irreps_scalars)
    #act_gates   = [F.silu] * len(irreps_gates)
    act_gates = [F.sigmoid] * len(irreps_gates)
    
    gate = Gate(
        irreps_scalars=irreps_scalars,
        act_scalars=act_scalars,
        irreps_gates=irreps_gates,
        act_gates=act_gates,
        irreps_gated=irreps_gated,
    )
    return gate


class NequipConvolution(nn.Module):
    """Single NequIP equivariant message-passing layer.

    Uses an **unweighted** CG tensor product (no TP weights) followed by
    per-irrep-instance radial scaling — matching the JAX implementation.
    """

    def __init__(
        self,
        input_irreps: o3.Irreps,
        output_irreps: o3.Irreps,
        sh_irreps: o3.Irreps,
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

        # Unweighted CG tensor product (no learnable weights in TP)
        self.tp, self.tp_irreps = build_cg_tp(input_irreps, sh_irreps, output_irreps)
        # Radial MLP: one scalar per output irrep instance
        radial_out = self.tp_irreps.num_irreps

        self.linear_1 = Linear(str(input_irreps), str(input_irreps))

        self.radial_mlp = MLP(
            [radial_basis_size] + [radial_mlp_size] * radial_mlp_layers + [radial_out],
            init_scale=mlp_init_scale,
            use_bias=False,
        )

        # Gate (may be None if output_irreps is all-scalar)
        gate = _build_gate(output_irreps)
        self.gate = gate
        gate_input_irreps = gate.irreps_in if gate is not None else output_irreps

        # linear_2 projects from *expanded* tp_irreps → gate_input_irreps
        self.linear_2 = Linear(str(self.tp_irreps), str(gate_input_irreps))
        self.skip      = Linear(str(input_irreps), str(gate_input_irreps),
                                num_indexed_weights=n_species,
                                force_irreps_out=True)

        # Precomputed index tensors for fast irreps operations
        self._input_norm_info = build_norm_info(input_irreps)
        self.register_buffer(
            '_tp_expand_idx', build_weight_expand_index(self.tp_irreps),
        )

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
        # Unweighted CG TP (no weight argument)
        messages = self.tp(messages[senders], sh)                       # (E, tp_irreps.dim)
        # Per-irrep-instance radial scaling
        radial_weights = self.radial_mlp(radial_basis)                  # (E, num_irreps)
        messages = scale_irreps_by_scalar(messages, radial_weights, self._tp_expand_idx)

        messages_agg = scatter_sum(messages, receivers, n_nodes) / (self.avg_n_neighbors ** 0.5)

        out = self.linear_2(messages_agg) + self.skip(features, species)
        if self.gate is not None:
            return self.gate(out)
        return F.silu(out)


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

        hidden = o3.Irreps(hidden_irreps)
        self.hidden_irreps = hidden
        input_irreps = hidden.filter(keep=['0e'])
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax)

        self.species_embedding = Linear(
            f"{n_species}x0e", str(input_irreps)
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
        hidden_scalar_dim = hidden.filter(keep=['0e']).dim
        hidden2_irreps = o3.Irreps(f"{hidden_scalar_dim // 2}x0e")
        self.readout1 = Linear(str(hidden), str(hidden2_irreps), biases=True)
        self.readout  = Linear(str(hidden2_irreps), "1x0e",      biases=True)

        # Variance heads
        self.frc_logvar    = Linear(str(hidden), "6x0e")
        self.energy_logvar = Linear(str(hidden), "1x0e")
        self.stress_logvar = Linear(str(hidden), "6x0e")

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
            sh_irreps=self.sh_irreps,
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
    print("NequIP Equivariant Potential (PyTorch)")
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
