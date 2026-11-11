# RACEv2 / BAM_V2 — handoff for continuing on another machine

Written 2026-11-06 at the end of a long Claude Code session on KISTI HANGANG (GH200, aarch64), when the project
allocation (SRU) ran out. A new Claude Code session on any server should read this file first, then
`README.md` (results, English) and `docs/RACEv2_methods_summary.pdf` (methods, Korean).
Update the "State" and "Next steps" sections whenever something finishes.
Why RACE_LR is being pretrained on OMol25 (the protein–ligand use case, requirements and success criteria):
`docs/RESEARCH_PURPOSE.md`.

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
| `bam_omat24/bayes_lastlayer.py` | last-layer Bayesian linear regression; periodic cells and a stress block since 2026-11-09 (`tests/test_bayes_lastlayer.py`; run locally on 4G carbon chain and AuMgO, `docs/local_test_report.md` §17; OMat24: `runs/diag/bayes_lastlayer_oam.sh`) |
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

## 6b. State on 2026-11-11 (user's local PC, RTX 4090, conda env `jax`)

**Read `docs/HANDOFF_multidataset_thole.md`** for the details, the verification and the open items.

- **Multi-dataset aselmdb reading** (user request): `data_format: "stream"` now takes per-source options
  (`max_frames_per_shard` seeded subsets, `meta_dir` exclusions, `task` / `task_rules`, `energy_key`,
  per-source charge labels), `stream.mixed` (periodic + open-boundary structures in one batch),
  `stream.atom_energies_by_task` and `stream.dipole`. All six corpora of `~/Datasets` (alex, mptrj, oc25, omc25,
  omol25, opoly26) read and train together; configs from `runs/diag/make_multidataset_configs.py`.
- **Thole-type electrostatics, phase 1** (user request): `bam_omat24/thole.py` + `"model": "RACE_POLAR"`
  (`race_polar_nnx.py`), a JAX port of the `pbc_split` mode of BAM-torch `bam_ufo/model/thole.py`. It equals the
  torch block to 5e-8 (energies, forces, cell gradients) in `tests/test_thole.py`.
- **Thole phase 2** (same day): slab dipole correction, electrode / charge-group constraints from the `*_meta_v2`
  files, contact charge transfer and the OC25 slab-dipole labels; equal to the torch functions to 1e-7
  (`tests/test_thole_phase2.py`). Only 2-epoch plumbing runs so far.
- The `jax` env got `lmdb`, `vesin`, `pytest` (user approved). It has no cuequivariance: use `tp_backend: "e3nn"`.
- **First real run prepared, not started**: `input/multi/multi_full_L4x96.json` / `multi_full_L5x128.json`,
  `runs/diag/multi_sizes.py`, `runs/diag/run_multi.sh`; benchmarks and commands in
  `docs/HANDOFF_multidataset_thole.md` §5b. Stream workers no longer create CUDA contexts
  (`stream_io.worker_context`; before, about 5 GB of GPU memory went to 12 workers).
- This checkout is not a git repository; nothing was committed.
- Local runs: `runs/multi/` (plumbing checks; numbers in `docs/HANDOFF_multidataset_thole.md` §5).

## 7. Work done without compute (2026-11-06 afternoon, allocation exhausted)

The user asked for this list ("4번 빼고 순서대로", 2026-11-06) and asked to record it so that Claude Code can continue.
Each item: what exists, whether it was tested, and the **validation job to run first** (ask the user before every
sbatch). Every one of those jobs was written but not submitted.

| # | Item | Files | Tested? | Validation job (not yet run) |
|---|---|---|---|---|
| 1a | Paper draft v0 (methods with equations, results, leave-one-out analysis, limitations, reporting statement) | `docs/paper/RACEv2_draft.md` | n/a (text; ablation percentages recomputed by hand) | — |
| 1b | Audit cross-check of README/PDF against the 2026-11-03 audit; corrections applied (3BPA table + notes, rMD17 test sets, water final recipe wFu1/wFu2, labels, disclosure paragraphs, raw energies, ablation bold, ddof=1 std) | `docs/paper/audit_crosscheck.md`, `README.md`, `docs/methods_pdf_data.json` (PDF rebuilt) | numbers recomputed from the summary JSONs (scratch scripts in the audit report) | — |
| 2a | LAMMPS halo: saved-residual VJPs (`--staged-residuals`, runtime `RACEV2_STAGED_RESIDUALS`) and owned-row buckets (`--staged-buckets n:e:k`, `race_nnx.NODE_LIMIT`) | `bam_omat24/export_lammps.py`, `bam_omat24/race_nnx.py`, `lammps/racev2_pjrt.*`, `lammps/pair_race.cpp` | compiles; LAMMPS rebuilt; **no run** | `runs/diag/lammps_halo_res.sh`: Python staged check (residual vs recompute, owned rows vs full), then LiPS 664 / 5312 atoms on 4 ranks with residuals on/off |
| 2b | LAMMPS-JAX repository | `/home01/t589a02/prog/LAMMPS-JAX` (clone of the private github.com/myung-group/LAMMPS-JAX; local history merged, 3 commits ahead, **user pushes**). `/scratch/t589a02/LAMMPS-JAX` = temporary local copy, delete after the push | runtime compiles in the new layout | build LAMMPS from it once (`tools/build_lammps.sh`) |
| 2c | Streaming (pickle-free) data pipeline: JAX-free readers (extxyz offsets, aselmdb), `graph_arrays` (= `preprocess_graph`), size pass CLI, per-epoch planner (hosts, blocks, packing, lock-step), single-host `StreamLoader` | `bam_omat24/stream_io.py`, `stream_plan.py`, `stream_data.py`, `tests/test_stream_io.py`, design `docs/stream_loader_design.md` | JAX-free tests pass on the login node (4/4); graph and loader equivalence tests skip outside a job | `runs/diag/stream_check.sh` (all stream + graph-loading tests, size pass of the 100k subset with 64 workers). The aselmdb source needs `pip install lmdb` in `bam_cue` (ask first) |
| 2c' | **Done 2026-11-09 (single host):** `train_3bpa` `data_format: "stream"` (train / valid / test streamed, charge / spin / fragment / charge-label fields, resumable data cursor, gauge refit on a seeded subset, block mixing), opt-in vesin neighbour list. Multi-host code written, untested | `docs/HANDOFF_stream_loader.md` (read first), `bam_omat24/stream_data.py`, `graph_fields.py`, `train_3bpa.py` | local CPU tests + GPU end-to-end (xyz vs stream equal; resume works) | HANGANG T6 throughput smoke; OMol25 aselmdb field check; 2-process test (T5) |
| 2d | SOAP-basis Laplace with the GGN diagonal (`--ggn`) and the linearised predictive (`--predictive glm`): self-calibrating posterior over all weights. In the V1 paper's 3BPA OOD setting (local reduced model, §20) it is the only method whose σ grows with the extrapolation (z 0.9 / 1.7 / 3.6 at 300 / 600 / 1200 K; ② BLR: 3.2 / 7.6 / 22 with AUROC 1.00 for OOD detection). Next: the same chain on the full-recipe HANGANG checkpoint. **Read `docs/HANDOFF_uq_omol25.md`** (Korean, 2026-11-10): UQ status, 3BPA OOD reading vs the V1 paper, and the plan the user chose for training on the **full** OMol25 dataset. | `docs/SOAP_Laplace.md` §6, `bam_omat24/soap_laplace.py`, `runs/diag/ood/`, `docs/local_test_report.md` §18–20 | **run locally, works; not yet on the HANGANG checkpoint** | `sbatch ... runs/diag/ood/ood_3bpa_hangang.sh` (MAX_TEST=400 for a quick pass) |
| 2e | PyTorch port scope (no torch RACEv2 exists; V1 port is approximate; ~10–15 person-days; recommended LAMMPS path ML-IAP unified + Kokkos) | `docs/torch_port_scope.md` | read-only survey | decision for the user: torch port vs keep the PJRT halo path |
| 4 | RACE_LR: charge-dependent long-range extension (user request; RACE unchanged): Qeq with total charge per graph, Gaussian-charge Coulomb (direct / batched Ewald), `"model": "RACE_LR"`, `total_charge_key` | `bam_omat24/long_range.py`, `bam_omat24/race_lr_nnx.py`, edits in `data_3bpa.py` / `train_3bpa.py` (off by default), `tests/test_race_lr.py`, design `docs/RACE_LR_design.md` | **CPU tests pass** on the user's local PC (2026-11-06, `docs/local_test_report.md`; one test-precision fix, `charges` now returned); GPU job not run | `runs/diag/race_lr_check.sh` (Madelung, α independence, Qeq optimality, forces / rotation / charge dependence, plus the existing periodic and stress tests) |
| 3 | Analysis of finished results | — | — | the 100k size screening was still running (see §6); apply `select_size.py` when it ends |

## 8. Next steps (in the order the user agreed)

1. **When the allocation is back**, run the validation jobs of §7 in this order (one small test job first, as the
   site policy requires):
   1. `runs/diag/stream_check.sh` (CPU work only);
   2. `runs/diag/lammps_halo_res.sh`;
   3. `runs/diag/bayes_lastlayer.sh` (`MAX_TEST=200`, then full);
   4. `runs/diag/soap_laplace.sh`.
2. **Remaining ablations** (pre-registered, 3 seeds, 500 epochs): `noshell`, `nouff`, `amsgrad`. Configs:
   `input/input_3bpa_omat24_abl3_{noshell,nouff,amsgrad}_s4{2,3,4}_cue.json`.
   - Add a plain Adam run, and AMSGrad / Adam at lr 5e-3 (1 seed each) as a sensitivity check, labelled "not
     pre-registered".
   - Launcher: `runs/diag/train_runs.sh <run> ...` (one GPU per run).
   - Results go into the README ablation table and table 6b of the methods PDF.
3. **100k size screening**: apply `select_size.py` (§6). Report the AMSGrad control separately, judged on
   **validation only**: final-epoch values, plus the mean of the last 5 epochs and the force MAE as secondary numbers.
   The validation curves fluctuate by 5–10 % per epoch, and the train loss (raw weights, running mean) is not
   comparable with validation (EMA weights).
   - Interim reading at epoch 23 (L5x128, 1 seed): SOAP and AMSGrad tie on validation. Validation loss 0.433 vs
     0.419 and force MAE 154.8 vs 154.8 meV/Å (epochs 19–23). AMSGrad has the lower train loss (0.187 vs 0.238) and
     is about 31 % faster per epoch (520 vs 760 s).
   - On 3BPA (500 epochs) SOAP was clearly better: 1200 K force 49.3 vs AMSGrad 60.1 vs Adam β=0.95 82.8. So the
     optimizer may be data-regime dependent; report it that way.
4. **Optimizer / precision screening for OAM pretraining** (agreed with the user 2026-11-06; 1 GPU each, about 4.5 h
   per run at about 520 s/epoch, 100k OMat24, 30 epochs):
   ```
   sbatch -J opt100k -p normal --comment="field=sciai;appl=jax" --gres=gpu:2 -t 08:00:00 runs/diag/size_screen.sh 30 opt100k L5x128:adam L5x128:amsgrad:f32
   SEED=43 sbatch -J opt100k_s43 -p normal --comment="field=sciai;appl=jax" --gres=gpu:2 -t 10:00:00 --export=ALL,SEED=43 runs/diag/size_screen.sh 30 sz100k L5x128 L5x128:amsgrad
   ```
   - Plain Adam on OAM has not been tried.
   - AMSGrad in float32 has never completed an OAM run: check that it does not diverge (SOAP does), then measure the
     speed.
   - The extra seed (43) of SOAP and AMSGrad decides the tie.
   - Rule: if AMSGrad (or float32 AMSGrad) matches SOAP on validation within the seed spread, use it for OAM
     pretraining (speed). Keep SOAP for the small-data benchmarks and state the regime dependence in the paper.
   - Both lr = 0.01 (not tuned per optimizer); state this.
   - **V1 on 100k** (user request 2026-11-06): the RACE V1 recipe (all RACEv2 additions removed, AMSGrad, float64)
     on the same 100k OMat24 data, as the 10k comparison:
     `sbatch -J v1100k -p normal --comment="field=sciai;appl=jax" --gres=gpu:1 -t 08:00:00 runs/diag/size_screen.sh 30 v1100k L5x128:v1`
     (V1 was 2× faster per epoch at 10k, so it should take about 4 h). Compare it with sz100k L5x128 (SOAP) and
     L5x128amsgrad: the AMSGrad control separates the architecture effect from the optimizer effect, which the 10k
     comparison mixed.
5. **RACE_LR** (user request 2026-11-06, "순서대로 진행"):
   - step 1: the unit-test job above (CPU tests already pass locally; the GPU job is optional);
   - step 2: the 4G-HDNNP benchmark systems vs RACEv2. The data are converted on the local PC
     (`data_prep/convert_4ghdnnp.py`, `docs/DATASETS.md` §16; copy `data_local/4ghdnnp/` to HANGANG). Decisions
     (2026-11-06): `atom_energies` = the train-only offsets of `e0.json`, the seeded 90/10 split of `split.json`, and
     plain 3D Ewald with the default α 0.3 / n_real 2 / n_recip 6 for Au2/MgO (converged; no dipole correction);
     A local small-budget GPU test exists (`docs/local_test_report.md` §7). Ag3 shows a large gain for RACE_LR, the
     carbon chain about 25 %, and NaCl none at that budget. Raw energy errors of the multi-composition systems are
     dominated by a per-composition offset, so report them with a train-only per-composition gauge refit (or add a
     per-species `gauge_refit`: decision for the user). Evaluate with `dump_frame_errors` / `eval_sets` from
     2026-11-06 on, since they now pass `total_charge_key`. The charge diagnostics (`runs/diag/local_4g/charges.py`,
     report §8) show non-local charge transfer captured on the carbon chain, but charges about six times too small on
     NaCl. README has a preliminary RACE_LR section. **The methods PDF is not updated yet**, because the NanumGothic
     fonts are not on the local PC: add the RACE_LR section to `docs/methods_pdf_data.json` and the PDF on HANGANG;
   - step 3: SPICE / OMol25 and LAMMPS long-range coupling.
   - **OMol25 pretraining pilot** (user, 2026-11-06): pretrain RACEv2 on OMol25 with the total charge and spin from
     the start, and fine-tune for long range later.
     - Tooling: `data_prep/convert_omol25.py`, `runs/diag/make_omol_configs.py` (arms race / race_cs / racelr /
       racelr_q × soap_f64 / amsgrad_f32), and `runs/diag/check_ewald.py` for periodic data.
     - Blocked on Hugging Face access to the gated dataset. Run `inspect` first on the real files.
     - Then run a 10–100k pilot before any full run.
     - Full OMol scale needs the stream loader integrated (2c'), and a block-diagonal Qeq solve. The current solve is
       dense over the whole batch.
     - Ewald fixes for jit, float32 and the periodic guard: `docs/local_test_report.md` §9.
     - Local pilot on the public OpenFF redistribution of the neutral OMol25 ANI-2x part (report §11):
       - AMSGrad f32 is about 10× faster than SOAP f64 on the RTX 4060 Ti, at equal or slightly better accuracy.
       - The Löwdin charge loss fixes the charges (r 0.93) but not the errors there.
       - Latent charges on neutral data are meaningless (r −0.15).
     - ctx-RACE requirements (2026-11-07, `docs/RACE_LR_feedback_reply_2026-11-07_ko.md`):
       - per-fragment charge constraints (`fragment_key` / `fragment_charges_key` / `n_fragments`);
       - the short-range switch `long_range.sr_switch`;
       - `chi_init` defaulting to Pauling, with old configs pinned to zero;
       - the inference API `bam_omat24/predict.py`: `load` / `predict` / `charges`, and the per-element `refit`.
       - Still open: the HOMO > 0 filter and the PDB leakage check, both after `inspect` of the official OMol25.
     - Block-diagonal Qeq: `long_range.qeq_block` (2026-11-07). It equals the dense path and is needed for large
       float64 batches (8 × 1000 atoms: dense OOM, block 5.3 GB). Set it to the largest structure, e.g. 352 for OMol.
     - Overnight local queue (report §13):
       - RACE_LR gains hold over 3 seeds (carbon chain −49 % force, NaCl −11 %).
       - The short-range switch [3, 5] Å removes the NaCl gain, so do not use the [4, 6] Å example.
       - The charge-loss weight is best at 0.1 for the molecule and 1.0 for NaCl; 0.01 is too weak.
     - Local 2026-11-08 (report §14):
       - With 4500 NaCl structures RACE and RACE_LR fail on the doublet Na8Cl8+ (108 / 71 meV/A).
       - The spin input via the charge/spin embedding fixes it (4.8 meV/A), so pretrain with spin.
       - More data makes the NaCl MD stable (6/6).
       - Au2/MgO: RACE_LR halves the doped-slab energy error. The Au2 charge shift has the right sign but is 13 % of
         DFT.
   See `docs/RACE_LR_design.md` §3.
5b. **OMol25 full dataset** (user decision 2026-11-10): follow `docs/HANDOFF_uq_omol25.md` §4 (HF access → inspect → size pass → T6 → 100k–1M pilot → multi-host T5 → full run → Val Comp / leaderboard → UQ with ② and ③ on subsets). The OMol25 configs use the universal element list
   (Z 1..94) and a task embedding slot (`long_range.n_tasks`), so that the model can later be continued on, or get
   heads for, OMat24 / MPtrj / Alexandria / OC20 (§7 of that document). The data-side task key (cfg `task`) and
   freezing (cfg `train_only`) exist since 2026-11-10; per-dataset readout heads and partial checkpoint loading do
   not. A small local OMol25 → OMat24 transfer test (§7.4 there) confirms the plumbing and that replay prevents
   forgetting; it shows no gain from the 10k-molecule pretraining.
5c. **Multi-dataset training with RACE_POLAR** (user request 2026-11-11; `docs/HANDOFF_multidataset_thole.md` §6):
   - a first real run on the BAM-torch frame counts (about 1M frames per weight unit and epoch): choose the model
     size and batch for the 24 GB GPU first (the e3nn backend is memory hungry; cuequivariance is not installed);
   - compare with the BAM-torch results of `~/Python/BAM-torch/examples/oc25_mptrj` on the same validation subsets.
   Show the user the exact command before a long run.
6. **Stream loader integration** (2c'), then full OAM training: OMat24 pretraining, then MPtrj + sAlex fine-tuning, at
   the chosen size.
7. **Paper draft**: fill the TODO/CHECK markers of `docs/paper/RACEv2_draft.md`; mechanism tests of draft §5.4
   (radial curvature, stratum magnitudes vs T, layer CKA) when compute is available.
8. Later: water and MD22 (locality), MD observables for 3BPA and water, an inference-speed table, and the 99k
   evaluation of the final rMD17 models (audit C3).

## 9. Pitfalls already paid for

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
