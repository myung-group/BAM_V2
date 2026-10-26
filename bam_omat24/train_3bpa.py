"""Training script for the 3BPA benchmark with the ``bam_omat24`` RACE model.

Molecular (non-periodic) systems, single-node (one or more GPUs via JAX sharding).

Usage:
    python -m bam_omat24.train_3bpa input/input_3bpa.json

Config keys (in addition to the model hyper-parameters, see README):
    train_path, test_path | test_paths, valid_frac, atomic_numbers, atom_energies
    hidden_size (-> ``hidden_size * s2_irreps(lmax)``) or hidden_irreps
    loss_type        : "huber" | "nll" | "gauss_nll"          (default "huber")
    cov_mode         : "scalar6" | "irreps"                   (default "scalar6")
    x_mode           : "species" | "node_scalar"              (default "species")
    beta_nll         : beta for beta-NLL (gauss_nll only)     (default 0.0)
    use_dataset_stats: init shift/scale/avg_n_neighbors from the training set
                                                              (default false)
    e0_mode          : "isolated" | "regression"              (default "isolated")
    loss_schedule    : [{"until_epoch": N, "energy_weight": .., "force_weight": ..}, ...]
    grad_clip        : global-norm clip (null disables)       (default 1.0)
    restart          : resume from ckpt_name                  (default false)
    eval_test_every  : epochs between test-set evaluations    (default 100)
"""

from typing import Dict, Any, List
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import time
import numpy as np
from pathlib import Path
import json
import math
import sys
import functools

from bam_omat24.data_nnx import BucketedDataLoader, MultiDeviceDataLoader, setup_mesh, default_collate_fn
from bam_omat24.data_3bpa import (
    load_3bpa_graphs, load_frames, fit_e0, compute_dataset_stats,
    split_train_valid, GraphListDataset, atomic_numbers_to_indices,
)
from bam_omat24.nequip_nnx import Nequip
from bam_omat24.race_nnx import RACE
from bam_omat24.training_utils import (
    replicate,
    replicate_pytree,
    unreplicate_pytree,
    make_sharded_train_step,
    make_sharded_evaluate_step,
    evaluate_sharded,
    save_checkpoint,
    load_checkpoint,
)

jax.config.update("jax_enable_x64", False)

# 3BPA defaults: H, C, N, O
DEFAULT_ATOMIC_NUMBERS = [1, 6, 7, 8]
DEFAULT_ATOM_ENERGIES = [
    -13.587222780835477,
    -1029.4889999855063,
    -1484.9814568572233,
    -2041.9816003861047,
]


def _get(cfg: Dict, key: str, default: Any = None) -> Any:
    v = cfg.get(key, default)
    return default if v is None else v


# =============================================================================
# Loss-weight schedule
# =============================================================================

def loss_weights_for_epoch(cfg: Dict, epoch: int) -> Dict[str, float]:
    """Return {energy_weight, force_weight, stress_weight} for a 0-based epoch.

    ``loss_schedule`` is a list of stages, each with ``until_epoch`` (exclusive,
    may be omitted for the last stage) and optional weights; missing weights
    fall back to the global ``energy_weight``/``force_weight``/``stress_weight``.
    """
    base = {
        'energy_weight': float(_get(cfg, 'energy_weight', 1.0)),
        'force_weight': float(_get(cfg, 'force_weight', 1.0)),
        'stress_weight': float(_get(cfg, 'stress_weight', 1.0)),
    }
    sched = _get(cfg, 'loss_schedule', None)
    if not sched:
        return base
    for stage in sched:
        until = stage.get('until_epoch', None)
        if until is None or epoch < until:
            out = dict(base)
            for k in base:
                if k in stage:
                    out[k] = float(stage[k])
            return out
    # past the last stage: keep the last stage's weights
    out = dict(base)
    for k in base:
        if k in sched[-1]:
            out[k] = float(sched[-1][k])
    return out


# =============================================================================
# Data
# =============================================================================

def build_datasets(cfg: Dict, fout):
    atomic_numbers = list(_get(cfg, 'atomic_numbers', DEFAULT_ATOMIC_NUMBERS))
    atom_indices = atomic_numbers_to_indices(atomic_numbers)
    cutoff = float(_get(cfg, 'cutoff', 5.0))
    train_path = Path(_get(cfg, 'train_path', 'dataset_3BPA/train_300K.xyz'))

    train_frames = load_frames(train_path)
    ref_e0 = np.asarray(_get(cfg, 'atom_energies', DEFAULT_ATOM_ENERGIES), dtype=np.float64)
    e0_mode = _get(cfg, 'e0_mode', 'isolated')
    if e0_mode == 'regression':
        # NOTE: 3BPA has a single stoichiometry, so the composition matrix is
        # rank-1 and lstsq returns the minimum-norm correction to the reference
        # isolated-atom energies (only the mean per-atom shift is identifiable).
        atom_energies = fit_e0(train_frames, atom_indices, reference_e0=ref_e0)
        print(f"E0 (regression, delta vs isolated): "
              f"{np.array2string(atom_energies - ref_e0, precision=4)}", file=fout)
    elif e0_mode == 'isolated':
        atom_energies = ref_e0
    else:
        raise ValueError(f"unknown e0_mode {e0_mode!r}")
    atom_energies = [float(x) for x in atom_energies]

    all_graphs = load_3bpa_graphs(train_path, atomic_numbers, atom_energies, cutoff,
                                  frames=train_frames)
    train_graphs, valid_graphs = split_train_valid(
        all_graphs, float(_get(cfg, 'valid_frac', 0.1)), seed=int(_get(cfg, 'split_seed', 42))
    )

    test_paths = _get(cfg, 'test_paths', None)
    if test_paths is None:
        tp = _get(cfg, 'test_path', None)
        test_paths = [tp] if tp else []
    test_sets = {
        Path(p).stem: GraphListDataset(load_3bpa_graphs(p, atomic_numbers, atom_energies, cutoff))
        for p in test_paths
    }

    stats = compute_dataset_stats(train_graphs, n_species=len(atomic_numbers))
    print("Training-set statistics:", file=fout)
    print(f"  n_graphs={stats['n_graphs']} n_atoms={int(stats['n_atoms'])} "
          f"avg_n_neighbors={stats['avg_n_neighbors']:.3f}", file=fout)
    print(f"  per-atom energy mean (shift_raw) = {stats['shift_raw']:.6f} eV/atom", file=fout)
    print(f"  force RMS (scale)                = {stats['scale']:.6f} eV/A", file=fout)
    print(f"  residual E0 (lstsq)              = "
          f"{np.array2string(stats['e0_residual'], precision=4)}", file=fout)

    return (atomic_numbers, atom_energies, GraphListDataset(train_graphs),
            GraphListDataset(valid_graphs), test_sets, stats)


# =============================================================================
# Model / optimizer
# =============================================================================

def build_model(cfg: Dict, n_species: int, atom_energies: List[float], stats: Dict, fout):
    seed = int(_get(cfg, 'seed', 42))
    model_rngs = nnx.Rngs(seed)
    lmax = int(_get(cfg, 'lmax', 3))

    import e3nn_jax as e3nn
    hidden_irreps = _get(cfg, 'hidden_irreps', None)
    if hidden_irreps is None:
        hidden_size = int(_get(cfg, 'hidden_size', 128))
        hidden_irreps = str(hidden_size * e3nn.s2_irreps(lmax))

    if _get(cfg, 'use_dataset_stats', False):
        shift = float(_get(cfg, 'shift', stats['shift_raw']))
        scale = float(_get(cfg, 'scale', stats['scale']))
        avg_n_neighbors = float(_get(cfg, 'avg_n_neighbors', stats['avg_n_neighbors']))
    else:
        shift = float(_get(cfg, 'shift', 0.0))
        scale = float(_get(cfg, 'scale', 1.0))
        avg_n_neighbors = float(_get(cfg, 'avg_n_neighbors', 25.0))
    print(f"Model init: shift={shift:.6f} scale={scale:.6f} avg_n_neighbors={avg_n_neighbors:.3f}",
          file=fout)

    model_type = _get(cfg, 'model', 'RACE')
    common = dict(
        n_species=n_species,
        lmax=lmax,
        cutoff=float(_get(cfg, 'cutoff', 5.0)),
        hidden_irreps=hidden_irreps,
        n_layers=int(_get(cfg, 'n_layers', 5)),
        radial_basis_size=int(_get(cfg, 'radial_basis_size', 8)),
        radial_mlp_size=int(_get(cfg, 'radial_mlp_size', 64)),
        radial_mlp_layers=int(_get(cfg, 'radial_mlp_layers', 3)),
        radial_polynomial_p=float(_get(cfg, 'radial_polynomial_p', 2.0)),
        mlp_init_scale=float(_get(cfg, 'mlp_init_scale', 4.0)),
        shift=shift,
        scale=scale,
        avg_n_neighbors=avg_n_neighbors,
        atom_energies=atom_energies,
        l_train=True,
        periodic=False,
        rngs=model_rngs,
    )
    if model_type == 'Nequip':
        model = Nequip(**common)
    else:
        model = RACE(**common,
                     x_mode=_get(cfg, 'x_mode', 'species'),
                     cov_mode=_get(cfg, 'cov_mode', 'scalar6'))

    graphdef, params = nnx.split(model, nnx.Param)
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"Model: {model_type} hidden_irreps={hidden_irreps} "
          f"x_mode={_get(cfg, 'x_mode', 'species')} cov_mode={_get(cfg, 'cov_mode', 'scalar6')}",
          file=fout)
    print(f"Model parameters: {n_params:,}", file=fout)

    grad_clip = _get(cfg, 'grad_clip', 1.0)
    chain = []
    if grad_clip is not None and grad_clip > 0:
        chain.append(optax.clip_by_global_norm(float(grad_clip)))
    opt_name = _get(cfg, 'optimizer', 'amsgrad')
    lr = float(cfg['learning_rate'])
    wd = float(_get(cfg, 'weight_decay', 0.0))
    if opt_name == 'amsgrad':
        chain.append(optax.amsgrad(lr))
    elif opt_name == 'adam':
        chain.append(optax.adam(lr))
    elif opt_name == 'adamw':
        chain.append(optax.adamw(lr, weight_decay=wd))
    else:
        raise ValueError(f"unknown optimizer {opt_name!r}")
    optimizer = optax.chain(*chain)
    print(f"Optimizer: {opt_name} lr={lr} grad_clip={grad_clip}", file=fout)

    schedule = optax.contrib.reduce_on_plateau(
        factor=float(_get(cfg, "lr_schedule_factor", 0.8)),
        patience=int(_get(cfg, "lr_schedule_patience", 10)),
        min_scale=float(_get(cfg, "lr_schedule_min_scale", 0.001)),
    )
    return graphdef, params, optimizer, schedule, grad_clip


# =============================================================================
# Entry point
# =============================================================================

def _fmt_metrics(m: Dict) -> str:
    return (f"E_MAE: {m['energy_mae_total']*1e3:7.2f} meV | "
            f"E_RMSE: {m['energy_rmse_total']*1e3:7.2f} meV | "
            f"F_MAE: {m['force_mae']*1e3:7.2f} meV/A | "
            f"F_RMSE: {m['force_rmse']*1e3:7.2f} meV/A")


def train(cfg: Dict):
    mesh, n_devices = setup_mesh()
    log_path = _get(cfg, 'log_file', 'loss_3bpa.out')
    fout = open(log_path, 'w', buffering=1)
    print(json.dumps(cfg, indent=1), file=fout)

    loss_type = _get(cfg, 'loss_type', 'huber')
    cov_mode = _get(cfg, 'cov_mode', 'scalar6')
    beta_nll = float(_get(cfg, 'beta_nll', 0.0))
    huber_delta = float(_get(cfg, 'huber_delta', 0.1))
    ema_decay = float(_get(cfg, 'ema_decay', 0.99))
    batch_size = int(_get(cfg, 'batch_size', 5))
    n_epochs = int(_get(cfg, 'n_epochs', 1000))
    ckpt_name = _get(cfg, 'ckpt_name', 'ckpt_best.pkl')
    eval_test_every = int(_get(cfg, 'eval_test_every', 100))
    use_schedule = bool(_get(cfg, 'loss_schedule', None))

    (atomic_numbers, atom_energies, train_dataset, valid_dataset,
     test_sets, stats) = build_datasets(cfg, fout)

    graphdef, params, optimizer, schedule, grad_clip = build_model(
        cfg, len(atomic_numbers), atom_energies, stats, fout)
    ema_params = params

    train_step = make_sharded_train_step(
        optimizer=optimizer, mesh=mesh,
        energy_weight=float(_get(cfg, 'energy_weight', 1.0)),
        force_weight=float(_get(cfg, 'force_weight', 1.0)),
        stress_weight=0.0,
        huber_delta=huber_delta, ema_decay=ema_decay,
        loss_type=loss_type, periodic=False, cov_mode=cov_mode, beta_nll=beta_nll,
        grad_clip=float(grad_clip) if grad_clip else 1.0,
        dynamic_loss_weights=use_schedule,
    )
    eval_step = make_sharded_evaluate_step(
        mesh, huber_delta=huber_delta, loss_type=loss_type, periodic=False,
        cov_mode=cov_mode, beta_nll=beta_nll)

    opt_state = optimizer.init(params)
    schedule_state = schedule.init(params)
    best_val_loss = math.inf
    start_epoch = 0
    if _get(cfg, 'restart', False) and Path(ckpt_name).exists():
        ckpt = load_checkpoint(ckpt_name)
        params = ckpt['params']
        ema_params = ckpt['ema_params']
        opt_state = ckpt['opt_state']
        schedule_state = ckpt['schedule_state']
        best_val_loss = ckpt.get('best_val_loss', math.inf)
        start_epoch = int(ckpt.get('epoch', 0))
        print(f"Resumed from {ckpt_name}: epoch={start_epoch} step={ckpt.get('step')} "
              f"best_val_loss={best_val_loss:.6f}", file=fout)

    params = replicate_pytree(params, mesh)
    ema_params = replicate_pytree(ema_params, mesh)
    opt_state = replicate_pytree(opt_state, mesh)
    schedule_state = replicate_pytree(schedule_state, mesh)
    step = replicate(jnp.array(int(_get(cfg, 'step', 0))), mesh)

    loader_rngs = nnx.Rngs(int(_get(cfg, 'seed', 42)) + 100)
    base_loader = BucketedDataLoader(
        train_dataset, batch_size=batch_size, n_buckets=8,
        shuffle=True, drop_last=True, rngs=loader_rngs,
    )
    # Padding granularity. The default collate pads nodes to multiples of 128
    # and edges to multiples of 4096, which wastes ~50 % of the compute for
    # small molecules; smaller buckets mean a few more JIT shapes but less work.
    node_bucket = int(_get(cfg, 'node_bucket_size', 128))
    edge_bucket = int(_get(cfg, 'edge_bucket_size', 4096))
    collate = functools.partial(default_collate_fn, node_bucket_size=node_bucket,
                                edge_bucket_size=edge_bucket)
    base_loader.collate_fn = collate
    train_loader = MultiDeviceDataLoader(base_loader, n_devices=n_devices, mesh=mesh,
                                         drop_incomplete=True)

    def _evaluate(dataset, params_rep, weights):
        return evaluate_sharded(
            graphdef=graphdef, params=params_rep, files_pkl=[dataset],
            batch_size=batch_size, n_devices=n_devices, mesh=mesh,
            energy_weight=weights['energy_weight'], force_weight=weights['force_weight'],
            stress_weight=0.0, huber_delta=huber_delta, loss_type=loss_type,
            eval_step=eval_step, params_already_replicated=True, periodic=False,
            cov_mode=cov_mode, beta_nll=beta_nll, collate_fn=collate,
        )

    def _log(*args, **kwargs):
        print(*args, file=fout, **kwargs)

    _log(f"\nStarting training: epochs {start_epoch+1}..{n_epochs}, "
         f"{len(base_loader)} batches/epoch, loss_type={loss_type}")
    global_step = int(np.asarray(step))
    t_train_start = time.time()
    prev_weights = loss_weights_for_epoch(cfg, start_epoch)
    for epoch in range(start_epoch, n_epochs):
        weights = loss_weights_for_epoch(cfg, epoch)
        if weights != prev_weights:
            # New loss-schedule stage: the validation loss changes scale, so the
            # best-checkpoint tracking and the plateau counter must restart
            # (the LR scale itself is kept).
            _log(f"--- loss schedule stage change at epoch {epoch+1}: {prev_weights} -> {weights}")
            best_val_loss = math.inf
            old_sched = unreplicate_pytree(schedule_state)
            fresh = schedule.init(unreplicate_pytree(params))
            schedule_state = replicate_pytree(fresh._replace(scale=old_sched.scale), mesh)
            prev_weights = weights
        lw = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in weights.items()}

        acc_energy = acc_force = 0.0
        acc_graphs = acc_atoms = acc_grad = acc_applied = 0.0
        n_steps = 0
        epoch_start = time.time()
        for batch, _info in train_loader:
            if use_schedule:
                params, opt_state, ema_params, step, metrics = train_step(
                    graphdef, params, opt_state, schedule_state, ema_params, step, batch, lw)
            else:
                params, opt_state, ema_params, step, metrics = train_step(
                    graphdef, params, opt_state, schedule_state, ema_params, step, batch)
            n_g = float(metrics['n_graphs'])
            n_a = float(metrics['n_atoms'])
            acc_energy += float(metrics['energy_loss']) * n_g
            acc_force += float(metrics['force_loss']) * 3 * n_a
            acc_graphs += n_g
            acc_atoms += n_a
            acc_grad += float(metrics['grad_norm'])
            acc_applied += float(metrics['update_applied'])
            n_steps += 1
            global_step += 1
        epoch_time = time.time() - epoch_start

        e_loss = acc_energy / max(acc_graphs, 1)
        f_loss = acc_force / max(3 * acc_atoms, 1)
        t_loss = weights['energy_weight'] * e_loss + weights['force_weight'] * f_loss
        lr_now = cfg['learning_rate'] * float(unreplicate_pytree(schedule_state).scale)

        _log(f"Epoch {epoch+1}/{n_epochs} ({epoch_time:.1f}s) Step {global_step} "
             f"LR: {lr_now:.2e} wE={weights['energy_weight']:g} wF={weights['force_weight']:g}")
        _log(f"  Train | Loss: {t_loss:.6f} | E_LOSS: {e_loss:.6f} | F_LOSS: {f_loss:.6f} | "
             f"Grad: {acc_grad / max(n_steps, 1):.4f} | unclipped: {acc_applied / max(n_steps, 1):.2f}")

        val_start = time.time()
        val_metrics = _evaluate(valid_dataset, ema_params, weights)
        val_loss = val_metrics['total_loss']
        _log(f"  Valid ({time.time()-val_start:.1f}s) | Loss: {val_loss:.7f} | {_fmt_metrics(val_metrics)}")

        unrep_params = unreplicate_pytree(params)
        _, new_schedule = schedule.update(
            updates=unrep_params, state=unreplicate_pytree(schedule_state), value=val_loss)
        schedule_state = replicate_pytree(new_schedule, mesh)

        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            best_state = {
                'params': unrep_params,
                'ema_params': unreplicate_pytree(ema_params),
                'opt_state': unreplicate_pytree(opt_state),
                'schedule_state': new_schedule,
                'step': int(np.asarray(step)),
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'metrics': val_metrics,
                'config': cfg,
            }
            save_checkpoint(best_state, ckpt_name)
            _log(f"  *** New best val loss: {best_val_loss:.6f} "
                 f"(improved by {improvement:.6f}) *** -> saved to {ckpt_name}")
        else:
            _log(f"  No improvement for {int(new_schedule.plateau_count)} evaluations "
                 f"(best: {best_val_loss:.6f})")

        if eval_test_every > 0 and (epoch + 1) % eval_test_every == 0:
            for name, ds in test_sets.items():
                tm = _evaluate(ds, ema_params, weights)
                _log(f"  Test[{name}] (EMA, current) | {_fmt_metrics(tm)}")

    _log(f"\nTraining finished in {(time.time()-t_train_start)/3600:.2f} h")

    # ---- final evaluation with the best checkpoint (EMA weights) ----
    if Path(ckpt_name).exists():
        ckpt = load_checkpoint(ckpt_name)
        best_ema = replicate_pytree(ckpt['ema_params'], mesh)
        weights = loss_weights_for_epoch(cfg, n_epochs - 1)
        _log(f"Final evaluation with best checkpoint (epoch {ckpt.get('epoch')}, "
             f"val loss {ckpt.get('best_val_loss'):.6f}):")
        summary = {}
        for name, ds in list(test_sets.items()) + [('valid', valid_dataset)]:
            tm = _evaluate(ds, best_ema, weights)
            summary[name] = {k: float(v) for k, v in tm.items()}
            _log(f"  {name:>12s} | {_fmt_metrics(tm)}")
        with open(Path(log_path).with_suffix('.summary.json'), 'w') as f:
            json.dump(summary, f, indent=1)
    fout.close()


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'input/input_3bpa.json'
    with open(cfg_path) as f:
        config = json.load(f)
    train(config)
