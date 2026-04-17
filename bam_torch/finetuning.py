"""FineTuning potential — PyTorch port of bam_omat24/finetuning_nnx.py.

Delta-learning wrapper:
    E_ft = MLP_scale(features) * E_foundation + MLP_shift(features)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3

from bam_torch.data import TorchGraphData
from bam_torch.linear import Linear
from bam_torch.race import RACE
from bam_torch.utils import MLP, energy_forces_stress


class FineTuningPotential(nn.Module):
    """Fine-tuning wrapper around a frozen RACE foundation model.

    Trainable parameters are prefixed with ``ft_``.
    Foundation model parameters should be frozen externally via
    ``requires_grad_(False)`` on ``self.foundation``.
    """

    def __init__(
        self,
        foundation_model: RACE,
        mlp_init_scale: float = 4.0,
        mlp_hidden_size: int = 64,
        mlp_hidden_layers: int = 2,
    ):
        super().__init__()
        self.foundation = foundation_model

        hidden_irreps = foundation_model.hidden_irreps
        n_species     = foundation_model.n_species
        output_irreps = hidden_irreps.filter(keep=['0e'])
        scalar_dim    = output_irreps.dim

        # Per-species readout: hidden → scalar features
        self.ft_readout = Linear(
            str(hidden_irreps), str(output_irreps),
            num_indexed_weights=n_species,
            biases=True,
            force_irreps_out=True,
        )

        # Affine scale and shift MLPs
        sizes_s = [scalar_dim] + [mlp_hidden_size] * mlp_hidden_layers + [1]
        self.ft_scale_mlp = MLP(sizes_s, init_scale=mlp_init_scale, use_bias=True)
        self.ft_shift_mlp = MLP(sizes_s, init_scale=mlp_init_scale, use_bias=True)

        # Zero-init last layers so initial output is identity (scale=1, shift=0)
        with torch.no_grad():
            self.ft_scale_mlp.layers[-1].weight.zero_()
            if self.ft_scale_mlp.layers[-1].bias is not None:
                self.ft_scale_mlp.layers[-1].bias.fill_(1.0)
            self.ft_shift_mlp.layers[-1].weight.zero_()
            if self.ft_shift_mlp.layers[-1].bias is not None:
                self.ft_shift_mlp.layers[-1].bias.zero_()

        # Variance heads (own parameters, trainable)
        self.ft_frc_logvar    = Linear(str(hidden_irreps), "6x0e")
        self.ft_energy_logvar = Linear(str(hidden_irreps), "1x0e")
        self.ft_stress_logvar = Linear(str(hidden_irreps), "6x0e")

        # Copy pretrained logvar weights from foundation
        self._copy_params_if_exists('frc_logvar',    'ft_frc_logvar')
        self._copy_params_if_exists('energy_logvar', 'ft_energy_logvar')
        self._copy_params_if_exists('stress_logvar', 'ft_stress_logvar')

    def _copy_params_if_exists(self, src_name: str, dst_name: str):
        src = getattr(self.foundation, src_name, None)
        dst = getattr(self, dst_name, None)
        if src is None or dst is None:
            return
        try:
            dst.load_state_dict(src.state_dict())
        except Exception:
            pass  # shape mismatch — skip

    def node_energies(
        self,
        positions: torch.Tensor,
        cell: Optional[torch.Tensor],
        cutoff: Optional[torch.Tensor],
        data: TorchGraphData,
    ) -> Dict:
        # Foundation: run with grad enabled (forces must flow through scale*E_f)
        results        = self.foundation.node_energies(positions, cell, cutoff, data)
        foundation_ne  = results['node_energies']   # (N,)
        node_features  = results['node_features']   # (N, hidden_dim)

        # Affine transformation
        scalar_feats = self.ft_readout(node_features, data.nodes['species'])  # (N, scalar_dim)
        scale = self.ft_scale_mlp(scalar_feats)  # (N,1)
        shift = self.ft_shift_mlp(scalar_feats)  # (N,1)
        node_energies = scale.squeeze(-1) * foundation_ne + shift.squeeze(-1)

        # Variance heads
        frc_raw    = self.ft_frc_logvar(node_features)
        frc_var    = torch.cat([F.softplus(frc_raw[:, :3]) + 0.04, frc_raw[:, 3:]], dim=-1)
        energy_var = F.softplus(self.ft_energy_logvar(node_features)) + 0.04
        stress_var = F.softplus(self.ft_stress_logvar(node_features)) + 0.04

        return {
            'node_energies':    node_energies,
            'frc_variation':    frc_var,
            'energy_variation': energy_var,
            'stress_variation': stress_var,
        }

    def forward(self, data: TorchGraphData) -> Dict:
        return energy_forces_stress(
            self.node_energies, data,
            self.foundation.periodic,
            self.foundation.l_train,
            self.foundation.atom_energies,
            create_graph=self.training,
        )

    def freeze_foundation(self):
        """Freeze all foundation model parameters."""
        for p in self.foundation.parameters():
            p.requires_grad_(False)

    def ft_parameters(self):
        """Return only the fine-tuning parameters (all non-foundation)."""
        foundation_ids = {id(p) for p in self.foundation.parameters()}
        return [p for p in self.parameters() if id(p) not in foundation_ids]
