"""race_v2: the RACE v2 equivariant ML potentials (JAX / Flax NNX) of the BAM_V2 repository.

Subpackages: models, data, training, inference, eval, export.  The old package name ``race_v2`` and its module
names (``race_v2.training.train`` ...) remain importable through the compatibility aliases in ``race_v2/``.
"""

import os as _os
if _os.environ.get("BAM_DISTRIBUTED") == "1":
    # multi-node data parallel: jax.distributed must start before the imports below touch the backend
    from race_v2.training.distributed import init_distributed as _init_distributed
    _init_distributed()

# Models
from race_v2.models.nequip import Nequip
from race_v2.models.race import RACE
from race_v2.models.finetuning import FineTuningPotential, filter_ft_params

# Model building blocks
from race_v2.models.blocks import (
    bessel_basis,
    polynomial_cutoff,
    separated_layer_norm,
    DenseLinear,
    MLP,
    compute_node_edge_features,
    node_graph_idx,
    energy_forces_stress,
)
from race_v2.models.linear import Linear as EquivariantLinear

# Data
from race_v2.data.loaders import (
    Dataset,
    BucketedDataLoader,
    MultiDeviceDataLoader,
    setup_mesh,
)
from race_v2.data.atom_energies import (
    ATOM_ENERGIES,
    ATOMIC_NUMBERS,
    ATOMIC_NUMBER_TO_INDEX,
    atomic_numbers_to_indices,
)

# Training utilities
from race_v2.training.utils import (
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

__all__ = [
    # Models
    "Nequip",
    "RACE",
    "FineTuningPotential",
    "filter_ft_params",
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
]
