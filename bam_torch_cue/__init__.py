"""bam_torch_cue — PyTorch port of bam_omat24 with cuequivariance backend.

Models
------
Nequip             : bam_torch_cue.nequip.Nequip
RACE               : bam_torch_cue.race.RACE
FineTuningPotential: bam_torch_cue.finetuning.FineTuningPotential

Data
----
GraphData          : bam_torch_cue.data.GraphData
TorchGraphData     : bam_torch_cue.data.TorchGraphData
GraphDataset       : bam_torch_cue.data.GraphDataset
BucketedDataLoader : bam_torch_cue.data.BucketedDataLoader

Training
--------
compute_loss       : bam_torch_cue.training_utils.compute_loss
make_train_step    : bam_torch_cue.training_utils.make_train_step
evaluate           : bam_torch_cue.training_utils.evaluate
save_checkpoint    : bam_torch_cue.training_utils.save_checkpoint
load_checkpoint    : bam_torch_cue.training_utils.load_checkpoint

Distributed
-----------
setup_distributed  : bam_torch_cue.distributed.setup_distributed
cleanup_distributed: bam_torch_cue.distributed.cleanup_distributed
wrap_model_ddp     : bam_torch_cue.distributed.wrap_model_ddp
unwrap_model       : bam_torch_cue.distributed.unwrap_model
is_main_process    : bam_torch_cue.distributed.is_main_process
"""

from bam_torch.atom_energies import (
    ATOM_ENERGIES,
    ATOMIC_NUMBER_TO_INDEX,
    ATOMIC_NUMBERS,
    get_atom_energy,
    get_total_reference_energy,
)
from bam_torch_cue.data import (
    BucketedDataLoader,
    GraphData,
    GraphDataset,
    TorchGraphData,
    batch_graphs,
)
from bam_torch_cue.distributed import (
    cleanup_distributed,
    is_main_process,
    setup_distributed,
    unwrap_model,
    wrap_model_ddp,
)
from bam_torch_cue.finetuning import FineTuningPotential
from bam_torch_cue.nequip import Nequip
from bam_torch_cue.race import RACE
from bam_torch_cue.training_utils import (
    compute_loss,
    evaluate,
    init_ema,
    load_checkpoint,
    make_train_step,
    save_checkpoint,
)

__all__ = [
    # atoms
    "ATOM_ENERGIES", "ATOMIC_NUMBER_TO_INDEX", "ATOMIC_NUMBERS",
    "get_atom_energy", "get_total_reference_energy",
    # data
    "GraphData", "TorchGraphData", "GraphDataset", "BucketedDataLoader",
    "batch_graphs",
    # models
    "Nequip", "RACE", "FineTuningPotential",
    # training
    "compute_loss", "make_train_step", "evaluate", "init_ema",
    "save_checkpoint", "load_checkpoint",
    # distributed
    "setup_distributed", "cleanup_distributed",
    "wrap_model_ddp", "unwrap_model", "is_main_process",
]
