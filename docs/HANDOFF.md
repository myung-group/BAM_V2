# RACEv2 / BAM_V2 — handoff for continuing on another machine

Written 2026-11-06 at the end of a long Claude Code session on KISTI HANGANG (GH200, aarch64), when the project
allocation (SRU) ran out. A new Claude Code session on any server should read this file first, then
`README.md` (results, English) and `docs/RACEv2_methods_summary.pdf` (methods, Korean).
Update the "State" and "Next steps" sections whenever something finishes.

## 1. Project in one paragraph

RACEv2 is an E(3)-equivariant message-passing MLIP (JAX + Flax NNX, package `bam_omat24`, model
`bam_omat24/race_nnx.py`). Each layer has a stratum readout, so the energy is a sum of per-layer strata (body
order grows with depth). The final recipe (README "Changes" table) beats MACE/TACE/GRACE on 3BPA temperature
extrapolation (except the dihedral scan) and the best published rMD17 values on most molecules. Paper working title (the user's choice):
**"Highly locally resolved machine learning interatomic potential: RACEv2"**. The paper must not claim a universal model
until OAM / Matbench results exist.

## 2. Rules from the user (keep them)

- Reply to the user **in Korean**. `README.md` and its figures in **English**. The methods PDF stays Korean.
- Keep `docs/RACEv2_methods_summary.pdf` current. Edit `docs/methods_pdf_data.json` (the only place its numbers
  live), then run `python docs/make_methods_pdf.py` (needs matplotlib, reportlab, PIL and NanumGothic fonts in
  `docs/fonts/`, which are not in git). Keep `README.md` in sync with new results.
- **Fair protocol**: train on the official training split only. No pretraining, no extra data, no data augmentation
  (no synthetic dimers or perturbed frames), and no ensembles for the reported numbers. Width and depth may change
  per dataset, but the methodology stays the same.
- Never report "works" without numbers. Results live in `runs/<name>_cue/loss_<name>_cue.summary.json`.
- Reporting convention: summary['gauge'] energies + summary[test] forces, best-validation EMA checkpoint, mean of 3
  seeds. Also give the raw energies and the best-validation seed. Disclose that test sets were monitored during
  development (`eval_test_every`).
- The user wants **principled new methodology** (mathematics or physics), not recipe sweeps. Do not launch
  screening waves without their go-ahead.
- Git: work on branch `racev2`. Commits are allowed. The **user pushes** (SSH key with passphrase; never open
  `~/.ssh`). End commit messages with the Co-Authored-By / Claude-Session lines that Claude Code provides.
- Never cancel the user's own jobs (e.g. `bamjax_oam`, `md_stab*`).
- Cluster etiquette (HANGANG policy; follow the local policy elsewhere): no JAX and no heavy work on login nodes;
  ask before every sbatch, scancel or salloc and show the exact command; one small test job first; no sbatch loops or
  auto-resubmission; poll at most once a minute; at most 5 jobs per partition.

## 3. Environment

- Python 3.12, conda env `bam_cue`. Versions: jax / jaxlib / jax-cuda12-plugin 0.12.0.1, flax 0.12.10,
  optax 0.2.8, e3nn-jax 0.21.0, cuequivariance(-jax, -ops-jax-cu12) 0.13.0, jraph 0.0.6.dev0, ase 3.29.0,
  matscipy 1.3.1, numpy 2.5.4, scipy 1.18.1.
- Two local compatibility patches, to be redone if the environment is rebuilt:
  1. `race_nnx.py` calls `jax.devices()` before importing `cuequivariance_jax` (in the repo).
  2. A shim for the DCE-rule signature in site-packages
     `cuequivariance_jax/segmented_polynomials/segmented_polynomial.py` (original kept as `.orig`). On a new machine,
     diff against `.orig` on HANGANG, or re-derive it from the error the first cue run raises.
- **Always** `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`: the float64 Triton GEMM fusion miscompiles the cue
  energy+forces program (rMD17 azobenzene). `runs/run_cue.sh` sets it.
- Importing `bam_omat24` initialises the XLA backend (cue needs `jax.devices()`). Stand-alone tools that must run
  without JAX are scripts run with `python -I` (`bam_omat24/oam_subset.py`, `bam_omat24/graph_sizes.py`).
- Environment switches: `BAM_INDEXED_LINEAR=ragged` (species-indexed Linear via `ragged_dot`, −25 % memory),
  `BAM_REMAT_LAYERS=1` (per-layer rematerialisation, −50 % memory, −36 % speed), `BAM_DISTRIBUTED=1` (multi-node),
  `RACEV2_TIMING=1` / `RACEV2_PREALLOCATE=1` (LAMMPS runtime).

## 4. What is not in git, and must be copied or regenerated

`runs/` (all checkpoints, logs, summaries), `*.pkl`, `*.out` and `docs/fonts/` are git-ignored (some `runs/diag`
scripts are force-added). Copy with a data-transfer node (HANGANG: dm01–dm04) what you need:

| What | HANGANG path | Note |
|---|---|---|
| final 3BPA runs (3 seeds) | `runs/abl3_full_s4{2,3,4}_cue/` | `ckpt_*.gauge.pkl` = reported checkpoints |
| ablations | `runs/abl3_<variant>_s4{2,3,4}_cue/` | full, xspecies, nohill, nocheb, nols, msgnorm, gate, noprior done |
| rMD17 / AcAc / Si / LiPS / water finals | `runs/rmd17f_*`, `runs/acacf_*`, `runs/siF_*`, `runs/lipsF_*`, `runs/wF*` | see README |
| OAM subsets 10k | `/scratch/t589a02/datasets_ext/oam_subsets/` | xyz + meta json (pkl caches can be dropped) |
| OAM subset 100k (OMat24) | `/scratch/t589a02/datasets_ext/oam_subsets_100k/` | 372 MB xyz |
| OAM raw shards (BAM-GAMP format) | `/scratch/t589a02/oam_data/{omat24,salex}_{train,val}`, refs `oam_data/refs/*_ls.npz`; MPtrj `/scratch/t589a02/mp_data/jax_data_v3_raw/` | regenerate subsets with `oam_subset.py` |
| 3BPA dihedral test | `/scratch/t589a02/datasets_ext/BOTNet-datasets/dataset_3BPA/test_dih.xyz` | public (BOTNet datasets) |
| LiPS | `/scratch/t589a02/datasets_ext/lips/` | splits.npz, md_reference.npz |
| LAMMPS source and builds | `/scratch/t589a02/src/lammps-stable_30Sep2026_update1`, `.../lammps-racev2-{build,mpi-build}` | rebuild with `lammps/build_lammps.sh` |
| exported LiPS model for LAMMPS | `runs/lammps_export/lips_cue_halo4/` | re-export with `export_lammps.py` (section 6) |

`/scratch` on HANGANG deletes files not accessed for 15 days, so copy checkpoints you need elsewhere.
3BPA xyz files are in git (`dataset_3BPA/`).

**Transfer package** (2026-11-06): `/scratch/t589a02/transfer_racev2_2026-11-06/` holds the git bundle, the results
text of all runs, the exact processed data splits (325 MB compressed), fonts, the environment versions and patch,
the Claude memory files and checkpoint manifests (core 10.7 GB, all 302.6 GB). Read its `RESTORE.md`.
**Rebuilding the datasets from the raw public sources**: `docs/DATASETS.md`, which has the download URLs, md5s, the
exact conversion commands and splits, and the md5s of the outputs. The converters are in `data_prep/`.

## 5. Key files

| File | Role |
|---|---|
| `bam_omat24/race_nnx.py` | model. `node_energies` = `_prelude` → `_layer_step` × L → `_finalize`; `READOUT_TAP` (last-layer features) |
| `bam_omat24/train_3bpa.py` | training / eval / gauge refit; all config keys documented at the top |
| `bam_omat24/soap_grouped.py`, `soap_optax.py` | SOAP optimizer (`soap_species_adam`: species tables by Adam) |
| `bam_omat24/data_3bpa.py`, `neighbors.py` | graphs (`build_graphs` parallel, `graph_cache`, size TSVs) |
| `bam_omat24/graph_sizes.py` | stand-alone size table + padding report (no JAX) |
| `bam_omat24/oam_subset.py` | OAM shard → xyz subsets |
| `bam_omat24/export_lammps.py` | StableHLO export (monolithic buckets + `--staged-buckets` per-layer functions) |
| `lammps/` | PJRT C++ runtime (`racev2_pjrt.*`), `pair_style race` (`pair_race.*`), `build_lammps.sh` |
| `bam_omat24/bayes_lastlayer.py` | last-layer Bayesian linear regression (not run yet) |
| `runs/diag/make_oam_configs.py` | OAM configs (`--x64 --species-adam --layers --hidden --optimizer --warmup`) |
| `runs/diag/size_screen.sh`, `select_size.py` | 100k size screening and the pre-registered selection rule |
| `input/input_3bpa_omat24_abl3_full_s42_cue.json` | reference config of the final recipe (3BPA) |
| `docs/size_screening_protocol_2026-11-03.md`, `docs/ablation_protocol_2026-11-05.md` | pre-registered rules |
| `docs/RACE_benchmark_audit_2026-11-03.md` | verified SOTA numbers and reporting rules (Korean) |

## 6. State on 2026-11-06 (allocation exhausted; no new jobs possible)

**Running when the session ended** (HANGANG job ids):
- `1224124 sz100k` (4 GPUs): L5x64, L5x128, L5x192, L4x128, 30 epochs on 100k OMat24.
- `1224125 sz100k_b` (2 GPUs): L6x128 and L5x128 with AMSGrad (an optimizer control, not a size candidate).
- `1221822 bamjax_oam`: the user's own BAM-GAMP job. Do not touch it.

When the size screening finishes, apply the rule (validation only):
`python -I runs/diag/select_size.py sz100k_L5x64_omat24_v2_s42 sz100k_L5x128_omat24_v2_s42 sz100k_L5x192_omat24_v2_s42 sz100k_L4x128_omat24_v2_s42 sz100k_L6x128_omat24_v2_s42`.
Report the AMSGrad control separately (SOAP vs AMSGrad on OAM), then update README section "Model size on 100k OMat24 structures".
Epoch times measured: 834 s (L4x128), 1010 s (L5x128), 1196 s (L6x128), 1419 s (L5x192) on one GH200, float64.

**Done in this session** (numbers in README):
- OAM 10k subsets: RACEv2 beats V1 on all three datasets. Settings adopted for OAM: float64 (SOAP diverges in
  float32), stress weight 10, species-Adam, cutoff 5 Å.
- Ablations: 7 of 10 done, plus noprior (no radial prior: dihedral +68 %, 1200 K force +30 %).
- LAMMPS `pair_style race`:
  - owner mode validated against Python;
  - halo mode validated on 1/2/4 ranks to 1e-14 eV/Å;
  - weak scaling 1→8 GPUs on 2 nodes at 88 % (42,496 atoms);
  - halo is still 3.3× slower than owner on one GPU.

## 7. Next steps (in the order the user agreed)

1. **Remaining ablations** (pre-registered, 3 seeds, 500 epochs): `noshell`, `nouff`, `amsgrad`. Configs:
   `input/input_3bpa_omat24_abl3_{noshell,nouff,amsgrad}_s4{2,3,4}_cue.json`. Also add a plain Adam run and
   AMSGrad/Adam at lr 5e-3 (1 seed) as a sensitivity check, labelled "not pre-registered". Launcher:
   `runs/diag/train_runs.sh <run> ...` (one GPU per run). Add the results to the README ablation table and to table 6b
   of the methods PDF.
2. **Bayesian last layer** (the user asked for this, then step 3): `runs/diag/bayes_lastlayer.sh`. Run a quick check
   first with `MAX_TEST=200`.
   - Method: the readout weights of every stratum (5 × 64) plus a shared bias give an exactly linear model for E
     and F; the posterior is closed form, with hyperparameters from evidence maximisation on the training set.
   - The script first checks linearity (features × trained weights must reproduce the network). Then it reports
     calibration (z RMS, NLL, coverage, Spearman |err|–σ) against the 3-seed deep ensemble on valid, 300/600/1200 K
     and the dihedral scan.
   - Possible issues: `jax.jacfwd` through the cue kernels (if it fails, use `BACKEND=e3nn`), and memory on the
     dihedral set (energies only).
3. **SOAP-curvature Laplace**: use the Kronecker factors that SOAP keeps (`soap_grouped.py` state) as a KFAC-Laplace
   posterior over all weights. This is a research idea: the GGN under the Huber loss and the prior precision need to
   be worked out first.
4. **Pickle-free data loading** (code done, not yet run on a GPU node):
   - `runs/diag/graph_loading_check.sh` runs `tests/test_graph_loading.py` and writes the size tables of the 100k
     subset.
   - New configs can then use `"graph_cache": false, "graph_workers": 0`.
   - For OMol25 / OC25 / full OAM, the BAM-GAMP shard pipeline (offline packed shards, a few buckets, streaming) has
     to be wired into the RACEv2 training loop. This is not started.
5. **LAMMPS halo speed-ups** (the user said "we can improve the JAX path"):
   - compute layer outputs for the owned rows only (separate n_local bucket);
   - keep the forward residuals on the device so the VJPs do not recompute the layers;
   - device-resident or GPU-aware MPI exchange instead of full host round trips;
   - finer, independent node and edge buckets;
   - XLA command buffers.
   - Validate with `runs/diag/lammps_multi.py --modes halo` and `runs/diag/lammps_weak.py`.
   - After multi-GPU is final, the user wants the C++ part moved to a new private repo "LAMMPS-JAX" (the user
     creates it), with only the export wrapper and examples kept in BAM_V2.
6. Later: full OAM training (OMat24 pretraining → MPtrj + sAlex fine-tuning) at the chosen size; water rerun with
   repulsion; MD22; MD observables for 3BPA and water; an inference-speed table.

## 8. Pitfalls already paid for

- SOAP diverges in float32 on OAM data (30–80 steps) and on 3BPA (epochs 107–337); use float64 or `soap_state_dtype`.
- The repulsion prior is multiplied by the learned global scale (trained s ≈ 0.25). Never call it parameter-free.
- Padding edges must be masked: r = 0 edges carry no message, and padding-node energies are masked with `where`.
- ASE `lammps-data` rotates non-lower-triangular cells: compare LAMMPS forces in the frame read back from the data file.
- `jax.export`: cue custom calls need `DisabledSafetyCheck.custom_call`. `jax.jit` drops unused arguments unless
  `keep_unused=True` (the staged LAMMPS functions need every argument).
- C++ cue replay: the kernels must be recompiled in the recorded order (handles are sequential), and the arity
  depends on the cue version (34 arguments in 0.13).
- `runs/` is git-ignored: add diagnostic scripts with `git add -f`.
- Login-node JAX imports crashed the node once (core dump). Run JAX only inside jobs.
