# Repository layout (since 2026-11-16)

Main branch: **`main`** (until 2026-11-16 `racev2-thole-merge`).  It contains the former lines `racev2` (Matbench
Discovery work) and `thole-merge` (OMol25 / RACE v2 LR-POLAR); new work starts from `main`.

Distribution name: **BAM_V2** (`pyproject.toml`).  Python package: **`race_v2`** (was `bam_omat24`).
PyTorch inference port: **`race_v2_torch`** (was `racev2_torch`).

| Folder | Content |
|---|---|
| `race_v2/` | the package: `models`, `data`, `training`, `inference`, `eval`, `export` |
| `race_v2_torch/` | PyTorch inference port (its own project folder: `PYTHONPATH=<repo>/race_v2_torch`) |
| `bam_omat24/`, `racev2_torch/` | compatibility aliases for the old names (generated; see below) |
| `scripts/train`, `scripts/eval`, `scripts/data` | launchers and helpers in use; `scripts/data/prep_results` holds E0 / meta / size-table summaries per dataset |
| `configs/` | representative configs only (see `configs/README.md`) |
| `research/` | diagnostics (`research/diag`, was `runs/diag`), old configs (`research/input`, was `input/`), `research/dataset_3BPA` |
| `docs/` | documentation; results per model under `docs/results/` |
| `tests/`, `lammps/` | unchanged |

## Package

| `race_v2/models/` | Network definitions (RACE, RACE_LR, RACE_POLAR, Nequip) and their building blocks. |
| `race_v2/data/` | Structures to graphs, loaders and the stream loader.  The JAX-free modules of this folder (neighbors, graph_fields, formal_assign, stream_*, graph_sizes) are also imported as top-level modules by the worker processes, so they must stay together in one folder. |
| `race_v2/training/` | Training entry point (train.py), loss / train steps / checkpoints (utils.py), SOAP optimiser. |
| `race_v2/inference/` | Prediction, molecular dynamics and batched relaxation. |
| `race_v2/eval/` | Error analysis and uncertainty estimates (last-layer Bayes, SOAP Laplace). |
| `race_v2/export/` | Export to other runtimes (LAMMPS). |

| Old file | New file | New module |
|---|---|---|
| `bam_omat24/atom_energies.py` | `race_v2/data/atom_energies.py` | `race_v2.data.atom_energies` |
| `bam_omat24/build_pkl.py` | `race_v2/data/build_pkl.py` | `race_v2.data.build_pkl` |
| `bam_omat24/build_pkl_v2.py` | `race_v2/data/build_pkl_v2.py` | `race_v2.data.build_pkl_v2` |
| `bam_omat24/formal_assign.py` | `race_v2/data/formal_assign.py` | `race_v2.data.formal_assign` |
| `bam_omat24/graph_fields.py` | `race_v2/data/graph_fields.py` | `race_v2.data.graph_fields` |
| `bam_omat24/graph_sizes.py` | `race_v2/data/graph_sizes.py` | `race_v2.data.graph_sizes` |
| `bam_omat24/data_nnx.py` | `race_v2/data/loaders.py` | `race_v2.data.loaders` |
| `bam_omat24/neighbors.py` | `race_v2/data/neighbors.py` | `race_v2.data.neighbors` |
| `bam_omat24/oam_subset.py` | `race_v2/data/oam_subset.py` | `race_v2.data.oam_subset` |
| `bam_omat24/stream_data.py` | `race_v2/data/stream_data.py` | `race_v2.data.stream_data` |
| `bam_omat24/stream_io.py` | `race_v2/data/stream_io.py` | `race_v2.data.stream_io` |
| `bam_omat24/stream_plan.py` | `race_v2/data/stream_plan.py` | `race_v2.data.stream_plan` |
| `bam_omat24/stream_worker.py` | `race_v2/data/stream_worker.py` | `race_v2.data.stream_worker` |
| `bam_omat24/data_3bpa.py` | `race_v2/data/xyz_graphs.py` | `race_v2.data.xyz_graphs` |
| `bam_omat24/analyze_errors.py` | `race_v2/eval/analyze_errors.py` | `race_v2.eval.analyze_errors` |
| `bam_omat24/bayes_lastlayer.py` | `race_v2/eval/bayes_lastlayer.py` | `race_v2.eval.bayes_lastlayer` |
| `bam_omat24/dump_frame_errors.py` | `race_v2/eval/dump_frame_errors.py` | `race_v2.eval.dump_frame_errors` |
| `bam_omat24/eval_sets.py` | `race_v2/eval/eval_sets.py` | `race_v2.eval.eval_sets` |
| `bam_omat24/md_lips_analysis.py` | `race_v2/eval/md_lips_analysis.py` | `race_v2.eval.md_lips_analysis` |
| `bam_omat24/soap_laplace.py` | `race_v2/eval/soap_laplace.py` | `race_v2.eval.soap_laplace` |
| `bam_omat24/export_lammps.py` | `race_v2/export/lammps.py` | `race_v2.export.lammps` |
| `bam_omat24/md_run.py` | `race_v2/inference/md_run.py` | `race_v2.inference.md_run` |
| `bam_omat24/predict.py` | `race_v2/inference/predict.py` | `race_v2.inference.predict` |
| `bam_omat24/relax_batched.py` | `race_v2/inference/relax_batched.py` | `race_v2.inference.relax_batched` |
| `bam_omat24/relax_wbm.py` | `race_v2/inference/relax_wbm.py` | `race_v2.inference.relax_wbm` |
| `bam_omat24/atomic_shells.py` | `race_v2/models/atomic_shells.py` | `race_v2.models.atomic_shells` |
| `bam_omat24/utils.py` | `race_v2/models/blocks.py` | `race_v2.models.blocks` |
| `bam_omat24/finetuning_nnx.py` | `race_v2/models/finetuning.py` | `race_v2.models.finetuning` |
| `bam_omat24/linear_nnx.py` | `race_v2/models/linear.py` | `race_v2.models.linear` |
| `bam_omat24/long_range.py` | `race_v2/models/long_range.py` | `race_v2.models.long_range` |
| `bam_omat24/nequip_nnx.py` | `race_v2/models/nequip.py` | `race_v2.models.nequip` |
| `bam_omat24/race_nnx.py` | `race_v2/models/race.py` | `race_v2.models.race` |
| `bam_omat24/race_lr_nnx.py` | `race_v2/models/race_lr.py` | `race_v2.models.race_lr` |
| `bam_omat24/race_polar_nnx.py` | `race_v2/models/race_polar.py` | `race_v2.models.race_polar` |
| `bam_omat24/thole.py` | `race_v2/models/thole.py` | `race_v2.models.thole` |
| `bam_omat24/distributed.py` | `race_v2/training/distributed.py` | `race_v2.training.distributed` |
| `bam_omat24/logger.py` | `race_v2/training/logger.py` | `race_v2.training.logger` |
| `bam_omat24/soap_grouped.py` | `race_v2/training/soap_grouped.py` | `race_v2.training.soap_grouped` |
| `bam_omat24/soap_optax.py` | `race_v2/training/soap_optax.py` | `race_v2.training.soap_optax` |
| `bam_omat24/train_3bpa.py` | `race_v2/training/train.py` | `race_v2.training.train` |
| `bam_omat24/training_utils.py` | `race_v2/training/utils.py` | `race_v2.training.utils` |

Renamed function: `load_3bpa_graphs` -> `load_xyz_graphs` (the old name is kept as an alias).
Config JSON keys are unchanged.

## Commands

| Old | New |
|---|---|
| `python -m bam_omat24.train_3bpa <config>` | `python -m race_v2.training.train <config>` |
| `python -m bam_omat24.predict ...` | `python -m race_v2.inference.predict ...` |
| `python -I bam_omat24/stream_io.py sizes ...` | `python -I race_v2/data/stream_io.py sizes ...` |
| `python -m bam_omat24.export_lammps ...` | `python -m race_v2.export.lammps ...` |
| `python -m racev2_torch.convert ...` | `python -m race_v2_torch.convert ...` |

## Compatibility aliases

- `bam_omat24/` is a generated alias package: every old module name is the same module object as the new module.
  `import bam_omat24...`, `python -m bam_omat24.<old>`, `python bam_omat24/<old>.py` and the top-level import of the
  JAX-free worker modules after `sys.path.insert(0, "<repo>/bam_omat24")` keep working.
- Checkpoints written by the old code pickle the SOAP optimiser state as
  `bam_omat24.soap_grouped.ScaleBySoapGroupedState`.  `race_v2.training.utils.load_checkpoint` maps old module paths
  itself (`race_v2/_compat.py`); a plain `pickle.load` needs the `bam_omat24` alias package on the path.
- Checkpoints written by the new code name `race_v2.training.soap_grouped...`: the old code trees cannot read their
  optimiser state.
- `racev2_torch`: `racev2_torch/racev2_torch/` (old folder on the path) and `race_v2_torch/racev2_torch.py` (new folder
  on the path) alias `race_v2_torch`.  Saved model files keep the format tag `racev2_torch`.
- Work folders that run the code through a symbolic link (`work/bam_omat24 -> <repo>/bam_omat24`) keep working with
  the old commands.  For the new commands add `work/race_v2 -> <repo>/race_v2`, or put the repository root on
  `PYTHONPATH`.
- The JAX-free modules of `race_v2/data` (neighbors, graph_fields, formal_assign, stream_io, stream_plan, stream_data,
  stream_worker, graph_sizes) are imported as top-level modules by worker processes (no package import, no JAX), so
  they must stay together in that folder.

## Matbench Discovery scripts moved from branch `racev2` (2026-11-16)

| Was (racev2 work folder / untracked) | Now |
|---|---|
| `runs/helpers/{eval_x64.sh, eval_loop.sh, final_eval.sh, next_epoch.sh, val_table.py}` | `scripts/eval/mbd/` |
| `runs/diag/uq/{ood_laplace.py, ood_laplace.sh, plot_ood.py}` | `scripts/eval/uq/` |
| `runs/diag/bench/{mem_bench.py, ref_rattled.py, bench.sh}` | `scripts/eval/bench/` |
| `runs/torch_port/tools/compare_pred.py` | `race_v2_torch/tools/compare_pred.py` |
| `runs/helpers/` one-off launchers of the OAM run (start, restart, backups) | `research/diag/oam_run_2026-11/` (record only) |
| `docs/racev2_mbd/{report, figures}` | `docs/results/race_v2_oam_mbd/{report, figures}` |

These scripts are run from a work folder that holds `race_v2 -> <repo>/race_v2` and `scripts -> <repo>/scripts`
(the launchers `cd` to `/scratch/t510a13/race_v2/work`).  Check of the move:
`research/diag/oam_run_2026-11/verify_port.sh`.
