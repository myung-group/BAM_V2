# HANGANG, 2026-11-11: OMol25 4M training set-up, multi-node check, Thole merge

Account t510a13, partition `exclusive06` (GH200 nodes, 4 GPUs of 96 GB and 288 cores each). All numbers are from
the Slurm jobs named in the text. Short runs are 3 × 200 optimizer steps; their errors say nothing about final
accuracy. The long run of section 5 was still in progress when this was written.

Code state: branch `thole-merge` (GitHub `racev2` 5830202 + the streaming-loader commits made on HANGANG + the
Thole / multi-dataset snapshot + the changes of section 7). Job scripts: `/scratch/t510a13/jobs/` (not in the
repository).

## 1. Environment

- conda env `bam_cue` (Python 3.12, aarch64) from the pinned package list of the previous account, plus
  `lmdb 3.0.0` and `vesin 0.6.2`; JAX 0.12.0.1 sees the GPU; cuequivariance-jax DCE shim applied (job 1257317, 4 min 44 s).
- A user may hold 8,192 processes + threads on a node (`ulimit -u`). XLA, LLVM and vesin start one thread per
  visible core, so several unpinned JAX processes on a 288-core node exceed it. Three jobs failed on this limit
  (section 6).

## 2. Data preparation (OMol25 4M subset, official train split)

Job 1257463 (9 min 17 s, 64 worker processes).

| Split | Source | Structures | Mean neighbours (5 Å) | Largest structure |
|---|---|---|---|---|
| train | shards 0000–0078 of `train_4M` | 3,936,920 | 23.24 | 350 atoms, 17,880 edges |
| valid (checkpoint selection) | shard 0079 of `train_4M` | 49,834 | 23.26 | 348 atoms |
| test | `neutral_val` (official validation, neutral part) | 27,697 | 20.79 | 110 atoms |

- The 80 shards are a random partition: the share of each of the 34 data sources in shard 0079 lies inside the
  range of the other shards (e.g. `omol` 35.80 % vs 35.85 % overall). The validation shard is in distribution; errors
  on it are lower than on unseen compositions.
- Size pass 97 s, E0 pass 85 s. E0: least squares per element on the training shards only, 83 elements, full rank;
  residual 0.42 eV/atom (standard deviation).
- Rows carry `charge`, `spin` and per-atom Löwdin / Mulliken charges in the row data, read by `stream_io.decode_row`
  unchanged. 87 training rows (0.002 %) have no Löwdin charges; 51 % of the rows are neutral singlets.
- Neighbour list: the default backend (matscipy for molecules). vesin 0.6.2 starts one thread per core in every
  worker whatever `n_threads` / `OMP_NUM_THREADS` say (job 1257426 failed); for molecules it gives no speed-up.

## 3. Throughput, one node (racelr_q unless stated; L5×64, lmax 3, 12.28 M parameters, SOAP float64)

Job 1257553. Structures per second from epochs 2–3 of 200 steps.

| Run | GPUs | Graphs per GPU and step | 200 steps (s) | Structures / s |
|---|---|---|---|---|
| racelr_q (Qeq + Löwdin charge loss) | 1 | 16 | 19.0 | 154 |
| racelr_q | 1 | 64 | 43.0 | 289 |
| racelr (Qeq, no charge loss) | 1 | 64 | 43.4 | 286 |
| race_cs (charge / spin embedding only) | 1 | 64 | 41.0 | 303 |
| racelr_q | 4 (one process) | 64 | 61.9 | 802 |

- Qeq (solved per structure, `qeq_block` 352) and the charge loss cost about 5 %.
- 64 graphs per step are 1.9× faster than 16; GPU memory about 43 GB.
- 4 GPUs in one process give 2.8× of one GPU. The loss is in the main process (batch assembly), not in the GPUs:
  the same 4 GPUs split over two processes are 1.4× faster (section 4), and pinning a 1-GPU run to 72 cores made it
  15 % faster (16.2 s instead of 19.0 s per 200 steps at 16 graphs).

## 4. Two nodes (one process per node, `BAM_DISTRIBUTED=1`)

Job 1258065. NCCL over the Slingshot plugin (`aws-ofi-nccl/1.20.0`, `NCCL_PROTO=^LL128`, `srun --mpi=pmix
--network=disable_rdzv_get`); a cross-node reduction with a known answer is exact with the plugin (1.3 ms) and with
TCP sockets (15.6 ms).

| Configuration | Graphs per step | 200 steps (s) | Structures / s | One pass over the data |
|---|---|---|---|---|
| 1 node × 4 GPUs | 256 | 61.9 | 802 | 82 min |
| 2 nodes × 2 GPUs | 256 | 44.6 | 1,130 | 58 min |
| 2 nodes × 4 GPUs | 512 | 51.5 | 1,950 | 34 min |

- Same device count and batch, 600 steps: train loss 0.357 (2 × 2) vs 0.353 (1 × 4), validation force RMSE 257 vs
  253 meV/Å. The packs differ (files are split over the hosts), so the runs are not bit-identical.
- Whole life of a run on a 24,000-structure subset (job 1258334): 3 epochs, final evaluation, gauge refit, then the
  same directory resumed to 6 epochs. Passed on 1 node × 4 GPUs and on 2 nodes × 4 GPUs: resume at the saved step
  and data cursor, cosine schedule continued, three checkpoints written.
- Not tested: more than two processes (the other HANGANG session reports a hang in the first compilation with four
  or more), runs of days.

## 5. Main run (in progress)

Job 1257815, started 2026-11-11 13:11, one node × 4 GPUs: racelr_q, universal element list Z 1–94, 8 task rows,
64 graphs per GPU, SOAP float64, lr 0.01 with cosine decay (warm-up 7,926 steps), 697,488 steps = 44 passes; one
log epoch = 3,963 steps = 1/4 pass (about 19 min). Validation: 10,000 structures of shard 0079.

| Log epoch | Passes | Force RMSE (meV/Å) | Force MAE (meV/Å) | Energy MAE (meV / structure) | Train loss |
|---|---|---|---|---|---|
| 1 | 0.25 | 251.8 | 81.4 | 972 | 2.420 |
| 2 | 0.50 | 157.8 | 58.1 | 712 | 0.178 |
| 4 | 1.00 | 136.3 | 47.4 | 600 | 0.115 |
| 8 | 2.00 | 126.0 | 42.6 | 555 | 0.094 |

- The weighted charge-loss term has stayed at 0.0037–0.0038 since log epoch 2 (about 4 % of the loss), while
  energies and forces improve: with weight 0.1 the charges hardly train. To be checked on the checkpoint (charge
  correlation with Löwdin) and to be considered for the next run (larger weight).
- Reference values of the OMol25 paper (arXiv 2505.08762, Table 3, models trained on the 4M subset for 80 epochs,
  Comp test split, total-energy MAE / force MAE): eSEN-sm-cons 141 meV / 10.0 meV/Å, eSEN-md-d 89 / 6.1, GemNet-OC
  60 / 7.8. Those are for unseen compositions; our validation shard is in distribution. Val Comp needs Hugging Face
  access (not yet on disk).

## 6. Failures found and fixed

| Job | Symptom | Cause | Fix |
|---|---|---|---|
| 1257426 | size pass: `Resource temporarily unavailable` in vesin | vesin 0.6.2: one thread per core in each of 64 workers | default neighbour backend for molecules |
| 1257517 | `BrokenProcessPool`, `tf_XLAEigen` thread creation failed | the loader workers imported the package (JAX): started with `python -m`, and the worker functions lived in `bam_omat24.stream_data` | workers in the top-level module `stream_worker`; no re-import of `__main__` |
| 1258661 | `LLVM ERROR: pthread_create failed` 15 s after start | tests (pytest) and four trainings at once, unpinned | tests first; every process pinned to its own cores (`taskset`) |

## 7. Thole merge, RACE_POLAR and the combined model

- The snapshot `racev2.tgz` (based on 5830202; `docs/HANDOFF_multidataset_thole.md`) is commit 296173d; merged with
  the HANGANG streaming changes in 909af86 (conflicts in `stream_io.size_pass`, the worker module, the charge loss:
  both sides kept).
- `charge_label_optional` (b9ce029): structures without charge labels get NaN labels and are left out of the charge
  loss only.
- Combined model (e51a176): `"model": "RACE_POLAR"` with `"polar": {"charges": "qeq"}`. The charges are the Qeq
  charges of RACE_LR; Qeq contributes its on-site energy only (`long_range.coulomb_energy` false) and the Thole block
  computes the Coulomb energy of these charges with dipoles, quadrupoles and induced dipoles once. Open-boundary
  data only so far.
- Tests on CPU (jobs 1258661, 1258700): the 69 existing tests pass on the merged code (Thole against the BAM-torch
  reference values, RACE_POLAR, multi-dataset stream, stream loader, RACE_LR, optional labels). Of the three new
  tests of the combined model, the option checks and the energy bookkeeping (same charges as RACE_LR, no double
  counting) pass; the third stopped at a wrong line of the test itself (fixed in 7897ce8). **Its remaining checks
  (batch independence, forces against central differences, blocked Qeq) have not run yet.**

First runs under the recipe of section 3 (job 1258700; one GPU each, pinned; charge loss 0.1 on Löwdin charges; the
RACE_POLAR variants also have the ORCA dipole loss 0.1):

| Model | Graphs per step | 200 steps (s) | Peak GPU memory | Valid force RMSE after 600 steps (meV/Å) | Test charge RMSE (me) |
|---|---|---|---|---|---|
| racelr_q | 16 | 16.2 | – | 399.5 | 284 |
| RACE_POLAR (Fukui charges) | 16 | 17.6 | 18 GB | 398.3 | 241 |
| RACE_POLAR + Qeq charges | 16 | 18.2 | 18 GB | 399.6 | 239 |
| RACE_POLAR, AMSGrad float32 | 16 | 11.2 | – | 445.1 | 238 |
| RACE_POLAR / + Qeq charges | 32 | 29.7 / 31.1 | 44 GB | 354.7 / 349.5 | 240 / 245 |
| RACE_POLAR / + Qeq charges | 64 | 51.8 / 53.4 | 80 GB | 307.9 / 312.7 | 240 / 249 |

- The Thole block runs with SOAP float64, the cuequivariance backend and the universal element list, and costs about
  10 % at 16 graphs per step. Its dense pair sums need 80 GB at 64 graphs per step (racelr_q: 43 GB).
- Force errors are the same for the three models at this length; no statement about accuracy is possible yet.

## 8. Open

- Val Comp (Hugging Face access) for numbers comparable with the paper.
- The remaining tests of the combined model; a 2-node run of it (prepared: `jobs/hybrid_2node.sh`, 48 graphs per
  GPU).
- SOAP cost breakdown (prepared: `jobs/soap_profile.sh`): CholeskyQR, refresh interval, float32 model with float64
  SOAP state, batch size, and `soap_distributed_refresh` (7897ce8: the QRs of the basis refresh split over the
  devices; connected, not yet run).
- vesin for periodic data (OMat24): an older single-threaded version or one thread per call is needed.
