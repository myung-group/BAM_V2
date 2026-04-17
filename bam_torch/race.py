"""RACE model — PyTorch port of bam_omat24/race_nnx.py."""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn import Gate

from bam_torch.data import TorchGraphData
from bam_torch.linear import Linear
from bam_torch.nequip import _build_gate
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


class RACEXReadout(nn.Module):
    """Per-species readout of x-features → scalar energy contribution.

    Mirrors bam_omat24/race_nnx.py:RACEXReadout.

    hidden_irreps here is scalar-only (e.g. 64x0e), matching the JAX
    readout_hidden_irreps = hidden_irreps.filter(keep="0e").
    """

    def __init__(
        self,
        hidden_irreps: o3.Irreps,
        output_irreps: o3.Irreps,
        x_irreps: o3.Irreps,
        n_species: int,
        avg_n_neighbors: float,
    ):
        super().__init__()
        self.avg_n_neighbors = avg_n_neighbors
        self.x_irreps        = x_irreps

        # Layer 1: indexed per-species, x_irreps → hidden_irreps
        self.x_readout1 = Linear(
            str(x_irreps), str(hidden_irreps),
            num_indexed_weights=n_species,
            force_irreps_out=True,
        )
        # hidden_irreps is scalar-only → just silu, no Gate needed
        hidden_dim  = hidden_irreps.dim
        hidden_half = o3.Irreps(f"{hidden_dim // 2}x0e")

        self.x_readout2 = Linear(str(hidden_irreps), str(hidden_half), biases=True,
                                 force_irreps_out=True)
        self.x_readout3 = Linear(str(hidden_half), str(output_irreps), biases=True,
                                 force_irreps_out=True)

        self.hidden_irreps = hidden_irreps
        self._x_norm_info = build_norm_info(x_irreps)

    def forward(
        self,
        species: torch.Tensor,   # (N,) long
        x_feats: torch.Tensor,   # (N, x_irreps.dim)
    ) -> torch.Tensor:           # (N, output_irreps.dim)
        scale = self.avg_n_neighbors ** 0.5
        if x_feats.shape[-1] > 1:
            x_norm = separated_layer_norm(x_feats, self._x_norm_info) / scale
        else:
            x_norm = x_feats / scale

        h = self.x_readout1(x_norm, species)  # (N, hidden_dim)  — scalar-only
        h = F.silu(h)                          # plain activation for scalars
        h = self.x_readout2(h)                 # (N, hidden_dim // 2)
        return self.x_readout3(h)              # (N, output_irreps.dim)


class RACEConvolution(nn.Module):
    """RACE convolution layer with extra x-feature tensor product.

    Mirrors bam_omat24/race_nnx.py:RACEConvolution.
    Uses an unweighted CG TP + radial scaling (same as NequipConvolution),
    plus a second TP with scalar x_feats (simple element-wise multiply).
    """

    def __init__(
        self,
        input_irreps: o3.Irreps,
        output_irreps: o3.Irreps,
        x_irreps: o3.Irreps,
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
        self.x_irreps        = x_irreps

        # First TP: unweighted CG (features ⊗ sh)
        self.tp1, self.tp_irreps = build_cg_tp(input_irreps, sh_irreps, output_irreps)
        radial_out = self.tp_irreps.num_irreps

        self.linear_1   = Linear(str(input_irreps), str(input_irreps))
        self.radial_mlp = MLP(
            [radial_basis_size] + [radial_mlp_size] * radial_mlp_layers + [radial_out],
            init_scale=mlp_init_scale, use_bias=False,
        )

        # Second TP with x_feats (1x0e = scalar): this is just element-wise
        # multiplication.  Build an unweighted CG TP for it as well.
        self.tp2, self.tp2_irreps = build_cg_tp(self.tp_irreps, x_irreps, output_irreps)
        # tp2_irreps ⊗ 1x0e simply preserves all irreps, so tp2_irreps == tp_irreps.

        # Gate for final output (may be None if all-scalar)
        gate = _build_gate(output_irreps)
        self.gate         = gate
        gate_input_irreps = gate.irreps_in if gate is not None else output_irreps
        # linear_2 projects from expanded tp2_irreps → gate_input_irreps
        self.linear_2     = Linear(str(self.tp2_irreps), str(gate_input_irreps))
        self.skip         = Linear(str(input_irreps),  str(gate_input_irreps),
                                   num_indexed_weights=n_species, force_irreps_out=True)

        # Per-layer energy readout
        half = o3.Irreps(f"{output_irreps.filter(keep=['0e']).dim // 2}x0e")
        self.readout1 = Linear(str(output_irreps), str(half), biases=True)
        self.readout  = Linear(str(half), "1x0e", biases=True)

        # Precomputed index tensors for fast irreps operations
        self.register_buffer(
            '_input_norm_info_ns_range',
            torch.tensor(build_norm_info(input_irreps)['nonscalar_range'] or [-1, -1]),
        )
        self._input_norm_info = build_norm_info(input_irreps)
        self.register_buffer(
            '_tp_expand_idx', build_weight_expand_index(self.tp_irreps),
        )

    def forward(
        self,
        features: torch.Tensor,    # (N, input_irreps.dim)
        x_feats: torch.Tensor,     # (N, x_irreps.dim)
        species: torch.Tensor,     # (N,) long
        sh: torch.Tensor,          # (E, sh_irreps.dim)
        radial_basis: torch.Tensor,# (E, radial_basis_size)
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (node_energies (N,), new_features (N, output_irreps.dim))."""
        n_nodes = features.shape[0]

        features_norm = separated_layer_norm(features, self._input_norm_info)
        messages = self.linear_1(features_norm)
        # Unweighted CG TP
        messages = self.tp1(messages[senders], sh)                      # (E, tp_irreps.dim)
        radial_weights = self.radial_mlp(radial_basis)                  # (E, num_irreps)
        messages = scale_irreps_by_scalar(messages, radial_weights, self._tp_expand_idx)

        messages_agg = scatter_sum(messages, receivers, n_nodes) / (self.avg_n_neighbors ** 0.5)

        # Second TP with x_feats (1x0e scalar): unweighted CG
        messages_agg = self.tp2(messages_agg, x_feats)

        h = self.linear_2(messages_agg) + self.skip(features, species)
        new_features = self.gate(h) if self.gate is not None else F.silu(h)

        # Per-layer energy readout
        ne = self.readout(self.readout1(new_features))  # (N, 1)
        return ne.squeeze(-1), new_features


class RACE(nn.Module):
    """RACE model for energies, forces, and stress.

    Mirrors bam_omat24/race_nnx.py:RACE.
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

        self.shift           = nn.Parameter(torch.tensor(shift))
        self.scale           = nn.Parameter(torch.tensor(scale))
        self.avg_n_neighbors = avg_n_neighbors

        if atom_energies is not None:
            self.register_buffer('atom_energies', torch.tensor(atom_energies, dtype=torch.float32))
        else:
            self.register_buffer('atom_energies', torch.zeros(n_species))

        hidden       = o3.Irreps(hidden_irreps)
        self.hidden_irreps = hidden
        input_irreps = hidden.filter(keep=['0e'])
        x_irreps     = o3.Irreps("1x0e")
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax)

        self.species_embedding = Linear(f"{n_species}x0e", str(input_irreps))
        self.x_linear = Linear(str(input_irreps), "1x0e",
                               num_indexed_weights=n_species, force_irreps_out=True)

        layers, x_readouts, f_readouts = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        readout_hidden = hidden.filter(keep=['0e'])

        for i in range(n_layers):
            inp = input_irreps if i == 0 else hidden
            layers.append(RACEConvolution(
                input_irreps=inp, output_irreps=hidden,
                x_irreps=x_irreps, sh_irreps=self.sh_irreps,
                n_species=n_species,
                radial_basis_size=radial_basis_size,
                radial_mlp_size=radial_mlp_size,
                radial_mlp_layers=radial_mlp_layers,
                mlp_init_scale=mlp_init_scale,
                avg_n_neighbors=avg_n_neighbors,
            ))
            x_readouts.append(RACEXReadout(
                hidden_irreps=readout_hidden, output_irreps=o3.Irreps("1x0e"),
                x_irreps=x_irreps, n_species=n_species,
                avg_n_neighbors=avg_n_neighbors,
            ))
            f_readouts.append(RACEXReadout(
                hidden_irreps=readout_hidden, output_irreps=o3.Irreps("1x0e"),
                x_irreps=inp,
                n_species=n_species,
                avg_n_neighbors=avg_n_neighbors,
            ))

        self.layers     = layers
        self.x_readouts = x_readouts
        self.f_readouts = f_readouts

        # Variance heads
        self.frc_logvar    = Linear(str(hidden), "6x0e")
        self.energy_logvar = Linear(str(hidden), "1x0e")
        self.stress_logvar = Linear(str(hidden), "6x0e")

        self.input_irreps = input_irreps
        self.x_irreps     = x_irreps

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
        x_feats  = self.x_linear(features, data.nodes['species'])

        outputs: List[torch.Tensor] = []
        for layer, x_ro, f_ro in zip(self.layers, self.x_readouts, self.f_readouts):
            #f_energy = f_ro(data.nodes['species'], features)
            ne, features = layer(
                features, x_feats, data.nodes['species'],
                sh, radial_basis, data.senders, data.receivers,
            )
            #x_energy = x_ro(data.nodes['species'], x_feats)
            #outputs.append(ne + x_energy.squeeze(-1) + f_energy.squeeze(-1))
            outputs.append(ne)
            
        node_energies = torch.stack(outputs, dim=-1).sum(dim=-1)  # (N,)
        node_energies = node_energies * self.scale + self.shift

        frc_raw  = self.frc_logvar(features)
        frc_var  = torch.cat([F.softplus(frc_raw[:, :3]) + 0.04, frc_raw[:, 3:]], dim=-1)
        energy_var = F.softplus(self.energy_logvar(features)) + 0.04
        stress_var = F.softplus(self.stress_logvar(features)) + 0.04

        return {
            'node_energies':    node_energies,
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
    import torch
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
    print("RACE Equivariant Potential (PyTorch)")
    print("=" * 70)

    with open("omat24_val_0.pkl", "rb") as f:
        graphset = pickle.load(f)

    #data_np = jraph_to_graphdata(graphset[0])
    #data = batch_graphs([data_np]).to(torch.device("cpu"))

    graphs = [jraph_to_graphdata(graphset[i]) for i in range(5)]
    data = batch_graphs(graphs).to(torch.device("cpu"))

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
    for i, e in enumerate(energies):
        print(f"Total energy [{i}]: {e.item():.6f} eV")
    print(f"Force magnitude: {forces.norm(dim=-1).mean().item():.6f} eV/Å")
    print(f"Stress (Voigt) : {stress[0].detach().numpy()}")
    print(f"\nTarget energy  : {graphs[0].globals['energy'][0]:.6f} eV")
    print(f"Target stress  : {graphs[0].globals['stress'][0]} eV/Å³")
