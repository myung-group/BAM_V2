"""Improved RACE training script with scan-based training support.

This module provides efficient training using jax.lax.scan for epoch-level
optimization, reducing Python overhead and improving training throughput.
"""

from typing import Dict, Tuple, Optional, NamedTuple
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import pickle
import jraph
import time
import numpy as np

from bam_3bpa.data import DataLoader, Dataset
from bam_3bpa.race_nnx import RACE
from bam_3bpa.train_3bpa import TrainState, evaluate, load_checkpoint

jax.config.update("jax_enable_x64", False)


# =============================================================================
# Main Training Function
# =============================================================================

def main_test(config: Dict):
    """Main testing function."""

    # Load datasets
    atom_energies = config["atom_energies"]
    test_dataset = Dataset(
        file_path=config["test_path"],
        atomic_numbers=config["atomic_numbers"],
        cutoff=config["cutoff"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        seed=42,
        shuffle=False,
        drop_last=False,
    )

    shift = 0.0
    scale = 1.0
    n_neighbors = 25
    # Create model
    rngs = nnx.Rngs(config.get('seed', 42))
    model = RACE(
        n_species=len(config["atomic_numbers"]),
        hidden_size=config["hidden_size"],
        lmax=config["lmax"],
        cutoff=config["cutoff"],
        n_layers=config["n_layers"],
        radial_basis_size=config["radial_basis_size"],
        radial_mlp_size=config["radial_mlp_size"],
        radial_mlp_layers=config["radial_mlp_layers"],
        radial_polynomial_p=config["radial_polynomial_p"],
        mlp_init_scale=config["mlp_init_scale"],
        shift=shift,
        scale=scale,
        avg_n_neighbors=n_neighbors,
        atom_energies=atom_energies,
        rngs=rngs
    )

    graphdef, params = nnx.split(model, nnx.Param)
    ema_params = params
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"Model parameters: {n_params:,}")

    # Setup optimizer
    optim = optax.amsgrad(config["learning_rate"])
    schedule = optax.contrib.reduce_on_plateau(
        factor=config["lr_schedule_factor"],
        patience=config["lr_schedule_patience"],
        min_scale=config["lr_schedule_min_scale"],
    )

    opt_state = optim.init(params)
    schedule_state = schedule.init(params)
    rng_key = jax.random.key(config.get('seed', 42))
    train_state = TrainState(
            graphdef=graphdef,
            params=params,
            opt_state=opt_state,
            schedule_state=schedule_state,
            ema_params=ema_params,
            step=jnp.array(0),
            rng_key=rng_key,
        )

    # Training loop
    best_val_loss, best_state = load_checkpoint(train_state, 'ckpt_best.pkl')
    ema_params = best_state.ema_params

    test_metrics = evaluate(
            graphdef, ema_params, test_loader,
            energy_weight=config["energy_weight"],
            force_weight=config["force_weight"],
            huber_delta=config.get('huber_delta', 0.1)
        )

    test_loss = test_metrics["total_loss"]
    print(f"  Test | Loss: {test_loss:.8f} | "
              f"E_MAE:  {test_metrics['energy_mae']*1000:.2f} meV| "
              f"F_MAE:  {test_metrics['force_mae']*1000:.2f} meV/A| "
              f"E_RMSE: {test_metrics['energy_rmse']*1000:.2f} meV| "
              f"F_RMSE: {test_metrics['force_rmse']*1000:.6f} meV/A")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import json
    from pathlib import Path

    config_path = Path('input/input_3bpa.json')
    if config_path.exists():
        print(f"Loading config from {config_path}")
        with open(config_path) as f:
            config = json.load(f)
            main_test(config)
