"""BAM-OMAT24: Equivariant ML potentials with JAX/Flax NNX."""

# Models
from bam_omat24.nequip_nnx import Nequip
from bam_omat24.race_nnx import RACE
from bam_omat24.finetuning_nnx import FineTuningPotential, filter_delta_params

# Model building blocks
from bam_omat24.utils import (
    bessel_basis,
    polynomial_cutoff,
    separated_layer_norm,
    DenseLinear,
    MLP,
    compute_node_edge_features,
    node_graph_idx,
    energy_forces_stress,
)
from bam_omat24.linear_nnx import Linear as EquivariantLinear

# Data
from bam_omat24.data_nnx import (
    Dataset,
    BucketedDataLoader,
    MultiDeviceDataLoader,
    setup_mesh,
)
from bam_omat24.atom_energies import (
    ATOM_ENERGIES,
    ATOMIC_NUMBERS,
    ATOMIC_NUMBER_TO_INDEX,
    atomic_numbers_to_indices,
)

# Training utilities
from bam_omat24.training_utils import (
    huber_loss,
    compute_loss,
    squeeze_batch,
    replicate,
    replicate_pytree,
    unreplicate,
    unreplicate_pytree,
    make_sharded_train_step,
    make_sharded_evaluate_step,
    evaluate_sharded,
    save_checkpoint,
    load_checkpoint,
)

# Training entry points
from bam_omat24.train_sharded import train_sharded
from bam_omat24.train_finetuning_sharded import train_finetuning_sharded

__all__ = [
    # Models
    "Nequip",
    "RACE",
    "FineTuningPotential",
    "filter_delta_params",
    # Model building blocks
    "bessel_basis",
    "polynomial_cutoff",
    "separated_layer_norm",
    "DenseLinear",
    "MLP",
    "EquivariantLinear",
    "compute_node_edge_features",
    "node_graph_idx",
    "energy_forces_stress",
    # Data
    "Dataset",
    "BucketedDataLoader",
    "MultiDeviceDataLoader",
    "setup_mesh",
    "ATOM_ENERGIES",
    "ATOMIC_NUMBERS",
    "ATOMIC_NUMBER_TO_INDEX",
    "atomic_numbers_to_indices",
    # Training utilities
    "huber_loss",
    "compute_loss",
    "squeeze_batch",
    "replicate",
    "replicate_pytree",
    "unreplicate",
    "unreplicate_pytree",
    "make_sharded_train_step",
    "make_sharded_evaluate_step",
    "evaluate_sharded",
    "save_checkpoint",
    "load_checkpoint",
    # Training entry points
    "train_sharded",
    "train_finetuning_sharded",
]
