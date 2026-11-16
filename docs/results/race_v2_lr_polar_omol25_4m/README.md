# RACE v2 LR-POLAR (OMol25 4M) results

Name: the model is called **RACE v2 LR-POLAR**. The `v3` that remains in the names of the training run folder and the checkpoint folder
(e.g. `BAM_V2_run_v3`, `omol4m_v3_polarqeq_...`) is the internal run name used at training time. Inside this folder the names were changed
to `epoch020`, `epoch070` and `final` (the original evaluation folders are `md_water/eval/v3_e20`, `v3_e70`, `v3_final`).

This folder collects, separately inside the package, the results of the 2026-11-15~16 compute window (exclusive06, ended 11/16 09:00).
The originals are in `/scratch/t510a13/md_water`, but `/scratch` is purged after 15 days without use, so only the small result files
(text, JSON, figures) were copied here. Checkpoints, size tables and extxyz samples are not included.

The document that summarises the methods (equations) and the results is `report/RACEv2_LR-POLAR_methods_results.pdf`.

## Training setup summary

- Model: RACE_POLAR, 19,919,663 parameters. Qeq + conductance charge transfer (learned decay), local charge / spin inputs
  (cs_local [4.5, 6.0]), spin energy, NBO head, induced polarisation, C6 dispersion. cutoff 6 Å, 5 layers x 128 channels, lmax 3.
- Data: OMol25 4M (3,936,920 training structures). 3 nodes x 4 GPUs, 1306 steps per epoch.
- Schedule: 170 epochs, cosine learning rate (max 0.01), SOAP optimiser, energy weight 10. About 448 s per epoch, 21.1 hours in total.
- Code: working folder `BAM_V2_run_v3`, commit d90bdc5. The full configuration is at the very top of `training/loss_log.txt`.
- Slurm job: 1280748 (restarted 11/15 07:15, finished normally 11/16 04:20).
- Final checkpoint: `/appsdata/t510a13/race_runs/omol4m/omol4m_v3_polarqeq_c6_cs_ct_L5x128_r12p6_spec_hub_spinE_nbo_csloc_decL_disp_xrn3_lr1e-2_soap_g48x12_s42`

## Folder layout

| Folder | Contents |
|---|---|
| `report/` | The PDF and the scripts that build it (`build_report.py`, `pdfkit.py`), the figure scripts, and the formulas extracted from the code (`methods_formulas.md`) |
| `training/loss_log.txt` | Original training log (`loss.out`): configuration, per-epoch loss table, neutral_val test every 16 epochs, final evaluation |
| `eval/<label>/` | Full test per checkpoint. The labels are `epoch020` (20 epochs), `epoch070` (70 epochs), `final` (170 epochs) |
| `official_val/` | Evaluation of a 32,000-structure sample of the official OMol25 val set (formal charges assigned automatically): per-structure results (`.json.gz`), per-group table (`_summary.txt`), first 200 |
| `figures/` | Figures of the final test: per-epoch loss, summary, comparison with public models (UMA, eSEN) |

Files inside `eval/<label>/`:

| File | Contents |
|---|---|
| `groups.json` | E/F MAE per group (domain, charge, spin, metal) for the 6000 internal validation structures |
| `bench_report.txt`, `bench.json`, `bench_report.png` | Physics benchmark curves (NaCl, LiF, Li+...I-, salt bridge, benzene, halogen bond, noble gases, etc.) |
| `iw/` | Ion-water interaction curves (compared with the DFT reference) |
| `iw_step_0p05A.json` | Ion-water scan at 0.05 Å spacing (step check) |
| `size_consistency.txt`, `.json` | Size consistency: difference between the energy of two separated fragments and classical electrostatics |
| `energy_decomposition.json` | Decomposition of the energy by term |
| `spin_eval.txt`, `.json` | Spin gaps (O2, Fe(H2O)6 2+, etc.) |
| `../formal_assign_check_train300.json` | Agreement rate between the automatic formal-charge assignment rules and the label q0 (sample of training shards) |

## Key numbers

Validation per epoch (stream val = all domains, neutral_val = neutral-molecule test set; meV, meV/Å):

| Epoch | train loss | val loss | val E MAE | val F MAE | neutral_val E MAE | neutral_val F MAE |
|---|---|---|---|---|---|---|
| 16 | 0.00456 | 0.00349 | 187.9 | 32.8 | 138.5 | 28.2 |
| 32 | 0.00371 | 0.00292 | 155.4 | 29.3 | 111.3 | 25.0 |
| 48 | 0.00329 | 0.00268 | 148.4 | 27.7 | 109.2 | 23.6 |
| 64 | 0.00296 | 0.00254 | 136.6 | 27.0 | 102.0 | 23.0 |
| 80 | 0.00267 | 0.00246 | 120.8 | 26.1 | 89.5 | 22.1 |
| 96 | 0.00238 | 0.00239 | 109.3 | 25.7 | 79.9 | 21.7 |
| 112 | 0.00216 | 0.00234 | 99.9 | 25.4 | 69.7 | 21.2 |
| 128 | 0.00194 | 0.00232 | 96.2 | 25.2 | 66.3 | 21.0 |
| 144 | 0.00179 | 0.00230 | 94.0 | 25.1 | 63.2 | 20.8 |
| 160 | 0.00169 | 0.00230 | 92.9 | 25.0 | 62.3 | 20.7 |
| 170 (final) | 0.00166 | 0.00229 | 92.3 | 25.0 | 62.2 | 20.7 |

Full test (70 epochs and the final 170 epochs; Slurm jobs 1285135~1285139, 1285153):

| Item | 70 epochs | Final |
|---|---|---|
| Internal 6000 structures (E meV / F meV/Å) | 136.7 / 26.0 | 90.9 / 24.2 |
| Official val, 32,000 structures, formal charges assigned automatically | 448.7 / 24.8 | 363.8 / 23.3 |
| Official val: neutral / q = ±1 / \|q\| ≥ 2 (E meV) | 278.5 / 342.9 / 1272 | 192.5 / 262.1 / 1178 |
| Official val, first 200 (automatic assignment) | 274.2 / 19.2 | 183.0 / 17.2 |
| Salt bridge MAE (kcal/mol) | 0.74 | 0.48 |
| NaCl / LiF dissociation MAE | 26.1 / 12.2 | 9.5 / 20.0 |
| Li+...I- / benzene / halogen bond MAE | 4.0 / 0.36 / 1.2 | 2.9 / 0.35 / 1.2 |
| Ar2 / Li+...Xe MAE | 3.8 / 31.0 | 3.0 / 31.5 |
| Ion-water curve MAE (kcal/mol) | 0.70~1.93 | 0.33~1.04 |
| Size consistency (6 Å, kcal/mol) | within ±0.5 | within ±0.6 |
| Spin gap O2 (DFT -34.7) / Fe(H2O)6 2+ (DFT -76.6) | -32.4 / -40.4 | -32.5 / -40.9 |

On the same official val sample the public models give 44.9 meV / 7.9 meV/Å (UMA-s-1.2.1) and 47.9 / 4.8 (eSEN-md).
The bad groups on the official val set are electrolytes (453 meV) and |q| >= 2 (1178 meV); the rows where the automatically assigned
formal charge is wrong (about 10 %) are the suspected cause. The re-evaluation with the label formal charges has not been done.

## Preprocessing results

The dataset preprocessing results (per-element E0, meta, size table summaries) are not in this folder but separately in `scripts/data/prep_results/`.

## How to rebuild

- Collecting the results: `bash collect_results.sh v3_e20 v3_e70 v3_final` (it only does `cp` of small files). The figures are taken from
  the session temporary folder (`SP` in the script), so in another session change `SP` to the place where the figures are.
- PDF: `python -I report/build_report.py <font folder> <figure folder> <output.pdf>`. The fonts are the two NanumGothic files
  (`docs/fonts/`, not put in git) and the figure folder is `figures/`. Only reportlab and matplotlib are used.
