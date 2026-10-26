# `bam_omat24/race_nnx.py` review and 3BPA reproduction setup

Branch: `refactor/race-omat24-canonical`. Environment: conda env `bam_v2`
(Python 3.12.15, jax 0.12.0.1 + CUDA 12, e3nn-jax 0.21.0, flax 0.12.10,
optax 0.2.8), RTX 4060 Ti 16 GB. Install was done with
`PIP_CONFIG_FILE=/dev/null` because the user-level pip config points to an
unreachable NVIDIA extra index (no version pins were changed).

## 1. Code map (state before this branch, commit `9c6d1a4`)

| File | Content |
| --- | --- |
| `bam_omat24/race_nnx.py` | `RACEXReadout` (x-feature readout: indexed Linear → Linear → gate → Linear), `RACEConvolution` (layer-norm → Linear → TP with SH × radial MLP → scatter → **second TP with x_feats** → Linear + per-species skip → gate → per-layer energy readout), `RACE` (species embedding, `x_linear`, `n_layers` convolutions, `x_readouts`, `f_readouts`, variance heads `frc_logvar` 6x0e / `energy_logvar` / `stress_logvar`). |
| `bam_omat24/utils.py` | `bessel_basis`, `polynomial_cutoff`, `separated_layer_norm`, `DenseLinear`, `MLP`, `compute_node_edge_features` (one-hot, displacements, radial basis, SH), `node_graph_idx`, `energy_forces_stress` (forces via `jax.grad`, stress, variance aggregation). |
| `bam_omat24/linear_nnx.py` | NNX wrappers of e3nn `FunctionalLinear` (`vanilla`/`indexed`) and `FullyConnectedTensorProduct`. |
| `bam_omat24/training_utils.py` | `huber_loss`, `_build_force_L` (Cholesky), `compute_loss`, `make_sharded_train_step` (shard_map + pmean), `make_sharded_evaluate_step`, `evaluate_sharded`, pickle checkpoints. |
| `bam_omat24/data_nnx.py` | `Dataset` (pickle of GraphsTuple), `BucketedDataLoader`, `MultiDeviceDataLoader`, `setup_mesh`. |
| `bam_omat24/train_omat24.py`, `finetuning_nnx.py`, `build_pkl.py` | OMAT24 pre-training / delta fine-tuning (periodic, stress). Not touched. |
| `bam_omat24/train_3bpa.py` | Port of the OMAT24 loop. **Could not run 3BPA**: expected non-existent `dataset_3BPA/*_cutoff_6.0.pkl`, ignored `hidden_size`/`cutoff` from `input_3bpa.json`, hard-coded `clip_by_global_norm(1.0)`. |
| `bam_3bpa/*` | Original 3BPA code that produced the README table (`hidden_irreps = hidden_size * s2_irreps(lmax)` = `128x0e+128x1o+128x2e+128x3o`, cutoff 5 Å, power-of-two padding, train/valid split `RandomState(42)` 90/10, raw energies with E0 added inside the model with `stop_gradient`). Read only. |

Data flow (3BPA, non-periodic): extxyz → `data_3bpa.preprocess_graph`
(matscipy neighbour list, `E - ΣE0`, float32) → list of `GraphsTuple` →
`BucketedDataLoader` (pads to bucket sizes, adds one padding graph) →
`MultiDeviceDataLoader` (leading device axis) → `shard_map` train step →
`compute_loss` → `RACE.__call__` → `energy_forces_stress` → `node_energies`.

### Issues 1–8 → code locations (before the fix)

| # | Issue | Location (commit `9c6d1a4`) | Fix |
| --- | --- | --- | --- |
| 1 | BRACE advertised but absent | `README.md` "Model Architecture" | README: "(planned)". |
| 2 | `x_feats` is a per-element constant | `race_nnx.py:392` `x_feats = self.x_linear(species, features)`; second TP `race_nnx.py:186-190` | `x_mode="node_scalar"` option (`race_nnx.py` `x_projs`, per layer). Default `"species"`. |
| 3 | Constants declared as `nnx.Param` | `race_nnx.py:264` (`avg_n_neighbors`), `:268-270` (`atom_energies`) | Plain Python float / tuple (static graphdef). `nnx.split(model, nnx.Param)` unchanged. |
| 4 | Data statistics unused (shift 0, scale 1, 25 neighbours) | `train_3bpa.py:68-70`, `bam_3bpa/train_3bpa.py:544-546` | `data_3bpa.compute_dataset_stats`, `fit_e0`; `use_dataset_stats`, `e0_mode` in `train_3bpa.py`. |
| 5 | Force covariance not equivariant | `race_nnx.py:337-342` (`6x0e` head), `:436-440`; `training_utils.py:59-78`, `:162-170` | `cov_mode="irreps"`: head `1x0e+1x2e`; `training_utils._build_force_L_irreps`: `L = aI + B'`, `Σ = L²`, `‖B'‖_F ≤ 0.9a`. |
| 6 | NLL is Huber on whitened residuals (+ `2·log σ`) | `training_utils.py:146-208` | `loss_type="gauss_nll"`: `0.5‖z‖² + log|L|`, optional `beta_nll`. Legacy `"nll"` kept (for `irreps` the Huber acts on `‖z‖`). |
| 7 | E:F weighting fixed, energy term negligible | `train_3bpa.py:97-100` | `loss_schedule` stages; dynamic weights passed into the jitted step. Best-checkpoint tracking and plateau counter restart at a stage change. |
| 8a | `grad_clip` ignored | `train_3bpa.py:88` `clip_by_global_norm(1.0)` | Read from config (`null` disables). |
| 8b | tanh clipping leaks into MAE/RMSE; energy metrics per-atom | `training_utils.py:138-139, 431-432, 511-516` | Metrics from raw residuals; `energy_*_total` (eV, 3BPA/MACE convention) added. Also: component-wise tanh of force residuals was not rotation equivariant → now applied to the residual norm. |

## 2. New options (all default to the original behaviour)

| Option | Values | Default | Where |
| --- | --- | --- | --- |
| `x_mode` | `species`, `node_scalar` | `species` | `RACE.__init__` |
| `cov_mode` | `scalar6`, `irreps` | `scalar6` | `RACE.__init__`, `compute_loss`, `make_sharded_*` |
| `loss_type` | `huber`, `nll`, `gauss_nll` | `huber` | `compute_loss` |
| `beta_nll` | float ≥ 0 | `0.0` | `compute_loss` |
| `use_dataset_stats` | bool | `false` | `train_3bpa.build_model` |
| `e0_mode` | `isolated`, `regression` | `isolated` | `train_3bpa.build_datasets` |
| `loss_schedule` | list of stages | none | `train_3bpa.loss_weights_for_epoch` |
| `grad_clip` | float or `null` | `1.0` | `train_3bpa.build_model` |
| `optimizer` / `weight_decay` | `amsgrad`, `adam`, `adamw` | `amsgrad` / 0 | `train_3bpa.build_model` |
| `node_bucket_size` / `edge_bucket_size` | int | 128 / 4096 | `train_3bpa.train` |
| `test_paths`, `eval_test_every`, `restart`, `ckpt_name`, `log_file` | | | `train_3bpa.train` |

Model API additions: `RACE.frc_variation_fill` (neutral covariance for padded
nodes), `energy_forces_stress(..., frc_variation_fill=None)`,
`training_utils.build_force_L / force_whiten / _loss_terms`,
`evaluate_sharded(..., collate_fn=None)` and in-memory datasets,
`make_sharded_train_step(..., cov_mode, beta_nll, grad_clip, dynamic_loss_weights)`.

## 3. Numerical verification (`pytest tests`, 16 passed, 2 xfailed, ~7 min on GPU)

Measured on a random 3BPA frame with a small RACE (2 layers, 16 channels,
lmax 2):

| Check | float64 | float32 |
| --- | --- | --- |
| Energy invariance (rotation, inversion, translation) | 6e-10 | 3e-4 (|E| ≈ 14) |
| Force equivariance | 2e-8 | 2e-4 (|F| ≈ 0.7) |
| Forces vs central finite differences (h = 1e-4 Å) | rel 1e-7 | meaningless (float32 round-off of ΣE_node ≈ 1e-5 eV) |
| `_traceless_symmetric_from_2e`: `B(D(R)b) = R B(b) Rᵀ` | 1e-15 | – |
| NLL / gauss-NLL loss invariance, `cov_mode="irreps"` | passes (1e-6) | – |
| NLL / gauss-NLL loss invariance, `cov_mode="scalar6"` | **fails** (strict xfail): Δloss ≈ 0.027 of 21.6 under a rotation | – |

The float32 errors are round-off (they disappear in float64), so the strict
tests run in float64 (params and positions cast) and the float32 test only
guards against gross breakage at 2e-3 relative. The requested 1e-4 float32
tolerance is **not** met by this architecture in float32 for ≥ 2 layers
(1 layer: 1e-7); the amplification source was not pinned down (layer-norm
scalar variances are 0.05–0.15, so it is not the `eps`). Recorded as a question.

`compute_dataset_stats` on synthetic data: E0 recovered to 1e-3 eV, force RMS
to 3 %, neighbour count exactly. `x_mode="species"`: identical x for atoms of
the same element; `node_scalar`: geometry dependent, same output shapes.

## 4. End-to-end smoke run (20 epochs, `input/input_3bpa_omat24_smoke.json`)

```
conda activate bam_v2
python -m bam_omat24.train_3bpa input/input_3bpa_omat24_smoke.json
```
Model: 128x0e+128x1o+128x2e+128x3o, 5 layers, cutoff 5 Å, batch 5, AMSGrad 1e-2,
Huber δ = 0.1, E:F = 1:1, 90 steps/epoch, ≈ 27 s/epoch on the RTX 4060 Ti
(9.4 GB with the default 128/4096 padding buckets).

Selected lines of `runs/smoke/loss_smoke.out` (validation = 50 held-out training frames, EMA weights; energies are total energies in meV):

```
Model parameters: 6,194,705
Optimizer: amsgrad lr=0.01 grad_clip=1.0
Epoch 1/20 (229.8s) Step 90 LR: 1.00e-02 wE=1 wF=1
  Train | Loss: 0.090282 | E_LOSS: 0.047873 | F_LOSS: 0.042409 | Grad: 0.6790 | unclipped: 1.00
  Valid (22.1s) | Loss: 0.2044515 | E_MAE: 39741.26 meV | E_RMSE: 39741.63 meV | F_MAE:  717.17 meV/A | F_RMSE:  955.45 meV/A
Epoch 2/20 (26.1s) Step 180 LR: 1.00e-02 wE=1 wF=1
  Train | Loss: 0.012698 | E_LOSS: 0.000624 | F_LOSS: 0.012074 | Grad: 0.2062 | unclipped: 1.00
  Valid (1.2s) | Loss: 0.0896686 | E_MAE: 15403.47 meV | E_RMSE: 15403.84 meV | F_MAE:  426.65 meV/A | F_RMSE:  578.48 meV/A
Epoch 5/20 (29.0s) Step 450 LR: 1.00e-02 wE=1 wF=1
  Train | Loss: 0.003139 | E_LOSS: 0.000341 | F_LOSS: 0.002799 | Grad: 0.1765 | unclipped: 1.00
  Valid (1.3s) | Loss: 0.0031512 | E_MAE: 1069.37 meV | E_RMSE: 1069.89 meV | F_MAE:   52.51 meV/A | F_RMSE:   74.99 meV/A
Epoch 10/20 (27.5s) Step 900 LR: 1.00e-02 wE=1 wF=1
  Train | Loss: 0.001113 | E_LOSS: 0.000035 | F_LOSS: 0.001078 | Grad: 0.0613 | unclipped: 1.00
  Valid (1.2s) | Loss: 0.0005804 | E_MAE:   23.12 meV | E_RMSE:   26.99 meV | F_MAE:   23.78 meV/A | F_RMSE:   34.78 meV/A
Epoch 15/20 (26.0s) Step 1350 LR: 1.00e-02 wE=1 wF=1
  Train | Loss: 0.001086 | E_LOSS: 0.000055 | F_LOSS: 0.001031 | Grad: 0.0725 | unclipped: 1.00
  Valid (1.1s) | Loss: 0.0004140 | E_MAE:   27.33 meV | E_RMSE:   30.20 meV | F_MAE:   19.88 meV/A | F_RMSE:   29.18 meV/A
Epoch 20/20 (27.6s) Step 1800 LR: 1.00e-02 wE=1 wF=1
  Train | Loss: 0.000663 | E_LOSS: 0.000029 | F_LOSS: 0.000633 | Grad: 0.0529 | unclipped: 1.00
  Valid (1.1s) | Loss: 0.0003329 | E_MAE:   18.11 meV | E_RMSE:   20.85 meV | F_MAE:   17.61 meV/A | F_RMSE:   26.08 meV/A
Training finished in 0.26 h
Final evaluation with best checkpoint (epoch 20, val loss 0.000333):
     test_300K | E_MAE:   17.91 meV | E_RMSE:   20.89 meV | F_MAE:   18.15 meV/A | F_RMSE:   26.58 meV/A
         valid | E_MAE:   18.09 meV | E_RMSE:   20.85 meV | F_MAE:   17.60 meV/A | F_RMSE:   26.07 meV/A
```

The loss decreases monotonically; the best checkpoint is written every improving epoch and a restart from it (`restart: true`, 1 more epoch) continued the descent (val loss 0.000333 → 0.000318). Variants run for 5 epochs each on the same config, only to confirm the code paths (not a performance comparison): `x_mode="node_scalar"`, `cov_mode="irreps"` + `loss_type="gauss_nll"`, `e0_mode="regression"` and a two-stage `loss_schedule` all run (see `runs/mini/loss_mini.out` for the combined 3-epoch check with a 16-channel model).

## 5. Questions (not changed; design intent unknown)

1. The second tensor product with `x_feats` (`RACEConvolution`) is, with
   `x_mode="species"`, a per-element scaling of the aggregated messages. Is the
   intended design a learned per-element *gate* or a genuinely geometry
   dependent feature (`node_scalar`)? Only the latter adds information.
2. Legacy `loss_type="nll"` adds `2·log σ` per component (twice the Gaussian
   term) and the loss floors (`MIN_VAR`) differ per term — kept verbatim.
3. Float32 equivariance error grows from 1e-7 (1 layer) to 2e-4 (2 layers).
   Which operation amplifies the round-off (gate, `rsqrt` in the layer norm,
   per-species indexed weights)?
4. OMAT24 energy targets subtract `ATOM_ENERGIES` *before* training, while
   `bam_3bpa` adds E0 inside the model. The 3BPA pipeline follows the OMAT24
   convention (`l_train=True`). For inference with `l_train=False` the model
   adds `atom_energies` itself — the two must not be combined.
5. `BucketedDataLoader` stops a batch early when `max_nodes_per_batch` (256) is
   reached without re-queuing the skipped graphs; harmless for 3BPA (135 nodes
   per batch) but a potential data-dropping bug for OMAT24.

## 6. Next steps (separate sessions)

* Full 1500-epoch runs: baseline (`input/input_3bpa_omat24_baseline.json`) and
  `v2` (dataset statistics + two-stage loss schedule) — see section 7.
* MACE comparison under identical conditions, 5 seeds, UQ / calibration with
  `cov_mode="irreps"` + `gauss_nll`.
* BRACE (symmetric contractions) and a structural `x_feats` redesign.
