# RACE v2 Matbench Discovery results (MPtrj-only, OAM)

These are the results of training RACE v2 and evaluating it the Matbench Discovery way in the compute window of
2026-11-11~16 (KISTI HANGANG, exclusive06, ended 11/16 09:00). The document that sets out the methods (equations) and the
results is `report/RACEv2_MBD_methods_results.pdf` (14 pages).

Training was done with the code of branch `racev2` (old package name `bam_omat24`). On the main branch `main` the package name is `race_v2`,
and the mapping between old and new names is in `docs/LAYOUT.md`. Checkpoints, size tables and raw evaluation outputs are not included here (Section 5).

## 1. Results

Full WBM (215,488 unique prototypes), no correction, float64 inference. The final checkpoint is the last epoch, fixed in advance.

| Model | Parameters | Epochs | F1 | MAE (eV/atom) | R2 | RMSD (Å) | κ SRME | CPS | Leaderboard rank |
|---|---|---|---|---|---|---|---|---|---|
| RACE v2 OAM | 17.8M | pretraining 2 + fine-tuning 6.2 | 0.893 | 0.0214 | 0.872 | 0.0643 | 0.229 | 0.858 | about 20th of 49 |
| RACE v2 MPtrj-only | 17.8M | 40 | 0.777 | 0.040 | 0.800 | 0.0785 | 0.375 | 0.761 | about 6th of 19 MPtrj-only |
| BAM-MP-DeNS (previous model) | - | - | 0.761 | 0.043 | 0.797 | 0.0832 | 0.766 | 0.672 | |
| BAM-OAM (previous model) | 10.5M | 4 + 64 | - | 0.027 | - | - | - | - | |

- The leaderboard values were read on 2026-11-15 (`scripts/eval/mbd/mbd_leaderboard_2026-11-12.tsv`). The comparison table with other models is
  `figures/sota_table_racev2_oam.png`.
- Uncertainty (SOAP-eigenbasis Laplace): the maximum of the force σ separates OMol25, OPoly26 and OC25 from the MPtrj and sAlex validation sets
  with an AUROC of 0.96. The AUROC for picking out structures with a large error in WBM is weak, 0.56~0.59. The absolute size of σ is smaller than the error (about 16 times for energy).
- Size that can be computed at once on a single GPU (GH200): 410,000~560,000 edges in float32, 170,000~260,000 in float64.

## 2. Summary of the training settings

- Model: RACE, 7 Å, 128 channels, lmax 3, 5 layers, per-element table rank 32, 17,757,551 parameters. float32 model + float64 SOAP state.
- Loss: Huber (energy 0.05 eV/atom, forces 0.25 eV/Å, stress 0.05 eV/Å³), weights 1 : 1 : 1, EMA 0.999.
- Optimiser: SOAP (b1 = b2 = 0.95, preconditioner update every 10 steps), gradient clipping 1.0.
- MPtrj-only: 20 epochs, then 20 epochs of linear decay. Configs in `configs/mptrj/`.
- OAM: 2 epochs of OMat24 pretraining (`configs/omat_pretrain/`) → MPtrj × 8 + sAlex × 1 fine-tuning (`configs/oam_finetune/`).
  One virtual epoch is 5,000 steps. The repeated copies are read spread over the whole epoch (`stream.train.parts`).

## 3. Folder contents

| Path | Content |
|---|---|
| `report/RACEv2_MBD_methods_results.pdf` | Methods and results document |
| `report/build_report.py`, `pdfkit.py`, `assets/` | Scripts that build the PDF, and the equation images |
| `report/methods_formulas.md` | Equations transcribed from the code (file names and line numbers refer to branch `racev2`, commit 3545c1d) |
| `report/final_results.json`, `wbm_points.json` | Final numbers that go into the PDF, and the 3k sample tracking values |
| `report/plot_*.py`, `table_sota.py` | Scripts that make the figures and the comparison table |
| `figures/` | Figures of the document: training curves, pretraining, fine-tuning, y=x scatter plot, OOD, comparison table |

Rebuilding the PDF (login node, uses only reportlab and matplotlib):

```bash
cd docs/results/race_v2_oam_mbd/report
python -I build_report.py <NanumGothic font folder> ../figures RACEv2_MBD_methods_results.pdf final_results.json
```

## 4. How to rerun the evaluation

Run from a working folder (on HANGANG, `/scratch/t510a13/race_v2/work`) that holds the symbolic links `race_v2 -> <repository>/race_v2` and
`scripts -> <repository>/scripts`.

| Task | Script |
|---|---|
| Full evaluation of one checkpoint (float64: 3k sample, κ, full WBM, RMSD, CPS) | `bash scripts/eval/mbd/eval_x64.sh <checkpoint> <tag>` (submits 5 jobs) |
| 3k sample evaluation at every epoch during training | `scripts/eval/mbd/eval_loop.sh` |
| Validation error table | `python -I scripts/eval/mbd/val_table.py <run name>` |
| Leaderboard rank | `python -I scripts/eval/mbd/rank_mbd.py F1=.. MAE=.. R2=.. RMSD=.. kSRME=..` |
| Laplace uncertainty and OOD | `scripts/eval/uq/ood_laplace.sh`, figures with `scripts/eval/uq/plot_ood.py` |
| Memory and speed measurement, reference values for the PyTorch comparison | `scripts/eval/bench/bench.sh` |
| Comparison of the PyTorch conversion with JAX | `race_v2_torch/tools/compare_pred.py` |

- κ and tight relaxation are inferred with `BAM_PREDICT_X64=1` (`docs/inference_precision.md`). In float32 the κ SRME comes out as 0.861.
- Pretraining continued, fine-tuning and evaluation in one allocation: `scripts/train/omat_stream/chain_oam.sh`
  (configs in `configs/README.md`).

## 5. What is not here (HANGANG)

| Content | Location |
|---|---|
| Checkpoints (final, pretraining epochs 1 and 2, fine-tuning intermediate), PyTorch model | `/appsdata/t510a13/race_v2_backup_2026-11-15/{checkpoints, torch}` |
| 7 Å size tables, list of next tasks | `/appsdata/t510a13/race_v2_backup_2026-11-15/{oam_sizes, NEXT_TASKS.md}` |
| Training logs, per-structure predictions, OOD results | `/appsdata/t510a13/archive/race_v2_oam_mbd_20261116.tar.gz` |
| Bundle for resuming the work (code, configs, key checkpoints, notes) | `/appsdata/t510a13/archive/race_v2_resume_20261116.tar.gz` |
| Description of the preprocessing results | `scripts/data/prep_results/README.md`, `prep/README.md` in the export folder |

## 6. Known limitations

- The pretraining is short at 2 epochs. The top models have 4~6 epochs or more and 30 million parameters or more.
- The stress loss is a sum over the 6 Voigt components, so it enters 6 times larger than intended.
- The difference between the PyTorch conversion and JAX float64 is 4e-7 eV in energy and 2e-6 eV/Å in forces (1e-14 for the 5 Å model). The cause has not been checked.
- The Laplace σ cannot be used as an error bar because the prior precision was not tuned. The energy σ per atom depends on the structure size, so
  the force σ or the total energy σ is used as the OOD score.
