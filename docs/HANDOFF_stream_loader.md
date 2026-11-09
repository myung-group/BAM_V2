# Handoff: streaming (pickle-free) training data path

Written 2026-11-09 for the next Claude Code session. Code on branch `racev2` of BAM_V2.

Related documents:
- design: `docs/stream_loader_design.md`
- speed measurement: `docs/local_test_report.md` §15
- overall status: `docs/HANDOFF.md` §7 row 2c / 2c'

## 1. What it is

`"data_format": "stream"` in a training config makes `bam_omat24.train_3bpa` train without graphs on disk or in
memory. The default `"xyz"` path is unchanged: graphs are built once, pickled next to the xyz and held in memory.

- **Workers:** forkserver processes, JAX-free. Each reads blocks of structures (extxyz by byte offset, or ASE lmdb by
  row id) and builds the graph arrays with the same code as the in-memory path.
  - neighbour list: `neighbors.neighbor_pairs`
  - optional fields (total charge, spin, fragments, per-atom charge labels): `graph_fields.extra_fields`, shared with
    `data_3bpa.preprocess_graph`
- **Main process:** packs the graphs into fixed-shape batches and runs the step.
- **Disk:** only a size table (n_atoms and n_edges per structure).
  - per structure: about 18 B (npz) + about 30 B (tsv)
  - pickles were up to 7.6× the xyz size (Au2/MgO: 61 MB xyz, 468 MB pickle)
- **RAM:** no full dataset in memory.
- **Splits:** train, valid and test all stream.
  - valid: optional fixed seeded subset (`max_structures`)
  - `gauge_refit`: a seeded training subset (`gauge_structures`, default 20000)

## 2. How compilation stays bounded (batch shapes)

XLA compiles one program per input shape. A batch of variable-size structures would recompile for every new
(nodes, edges, graphs) combination.

- **Size pass first** (`stream_io.size_pass`): one neighbour search per structure records n_atoms and n_edges. This
  happens before training, without keeping graphs.
- **Buckets** (`stream_plan.derive_buckets`): the structures are split into `n_buckets` ranges of edge count (equal
  counts). Each bucket gets one padded shape:
  - `pad_nodes`: at least graphs_per_pack × mean atoms, and the largest structure + 1; rounded to 64
  - `pad_edges`: same rule for edges, rounded to 1024
  - `max_graphs` = `pack_max_graphs`
- **Packing** (`stream_plan._pack`, greedy next-fit in stream order): structures go into a pack until the node,
  edge or graph budget would overflow. Every batch is then padded to exactly the bucket shape
  (`data_nnx.fixed_collate_fn`).
- **Result:** one compilation per bucket for training, plus one per bucket for evaluation. An eval split with a
  structure larger than every training bucket adds one extra shape.
- **Packs vary in size.** The number of real graphs per step varies (the OMol subset: 16.3 on average, up to 64).
  The loss is a per-graph / per-atom mean, so this changes the step noise, not the scale.
- **The plan is per data epoch.** It is deterministic in (seed, data epoch, hosts, devices) and is computed from the
  size table only (`stream_plan.plan_data_epoch`). This gives resume by cursor and lock-step across hosts.
- **Same scheme as in-memory** `pad_mode: "fixed"`, which computes the budgets from the graphs in memory instead of a
  size table.

## 3. Config

```json
"data_format": "stream",
"stream": {
  "train": ["extxyz:data/omol/train.xyz"],
  "valid": {"sources": ["extxyz:data/omol/val.xyz"], "max_structures": 100000},
  "test":  {"test": ["extxyz:data/omol/test.xyz"]},
  "workers": 8, "eval_workers": 4,
  "n_buckets": 1, "graphs_per_pack": 16, "pack_max_graphs": 64,
  "block_rows": 128, "steps_per_epoch": null, "stats_sample": 2000, "gauge_structures": 20000
}
```

- **Source specs:** `extxyz:<file>` or `aselmdb:<file or directory>`. The directory is listed recursively for
  `*.aselmdb`, and needs `pip install lmdb`.
- **Size tables:**
  - A split's `"sizes": <dir>` names its size table.
  - Without it, process 0 makes `<log dir>/stream_sizes_<split>` at start.
  - For large data, make tables beforehand with many CPUs:
    `python -I bam_omat24/stream_io.py sizes --source aselmdb:<dir> --cutoff 6.0 --out <dir> --workers 64`
  - A table made for other sources is refused.
- **Epochs:** `steps_per_epoch` null means one epoch = one pass over the data. A number makes virtual epochs (needed
  when one pass exceeds the 24 h wall time). The cursor (data epoch, step) is saved in every checkpoint as
  `data_cursor`, and `restart: true` resumes there.
- **Statistics:**
  - shift / scale / residual E0 come from `stats_sample` structures;
  - `avg_n_neighbors`, the structure count and the atom count come from the full table.
- **E0:** must be given (`atom_energies`, `e0_mode: "isolated"`).
- **Refused in stream mode** (they need the training graphs in memory): `valid_frac`, `boltzmann_tilt`,
  `node_weight`, `taylor_aug`, `e0_mode: "regression"`, `train_path`.
- **Neighbour list:** `BAM_NEIGHBORS=vesin` in the environment uses vesin (`pip install vesin`). It gives the same
  pairs, is 57× faster for periodic 110-atom cells and 2× for molecules.
  - Without it, a periodic worker makes only ~60 structures/s, too slow for OAM on a GH200.
  - The edge count of every structure is checked against the size table when its graph is built, so a backend
    mismatch cannot silently overflow a shape.

## 4. Files

| file | role |
|---|---|
| `bam_omat24/graph_fields.py` | new, JAX-free: charge / spin / fragment / charge-label fields (used by both paths) |
| `bam_omat24/neighbors.py` | `BAM_NEIGHBORS` = `ase` (default) or `vesin` |
| `bam_omat24/stream_io.py` | readers carry the optional fields (`info`, `arrays`); `graph_arrays(..., fields)` adds `n:`/`g:` keys; aselmdb rows put `key_value_pairs` + `data` into `info`; the size table records the neighbour backend |
| `bam_omat24/stream_plan.py` | unchanged (buckets, plan, packing) |
| `bam_omat24/stream_data.py` | `StreamLoader` (fields, `start_step`), `StreamSteps` (cursor across data epochs), `StreamEvalSet` (every structure once; filler copies with weight 0), `stack_local`, `counts_allgather_jax` |
| `bam_omat24/training_utils.py` | `evaluate_sharded` accepts a `StreamEvalSet` (`iter_groups`) |
| `bam_omat24/train_3bpa.py` | `build_stream_datasets`, `stream_mode` branches (datasets, loader, checkpoint cursor, gauge refit, close) |
| `tests/test_stream_train.py` | fields equal the in-memory graphs (molecule and periodic), aselmdb row fields, eval coverage (also with 2 CPU devices), resume cursor across a data-epoch boundary |
| `tests/test_neighbors_vesin.py` | vesin pairs = ASE pairs (skips without vesin) |
| `runs/diag/bench_stream/` | pickle vs stream speed benchmark and figures |
| `input/stream_e2e/*.json`, `runs/stream_e2e/queue.sh` | local end-to-end check (below) |

## 5. Verified (local PC, 2026-11-09)

- **CPU tests:**
  - `tests/test_stream_train.py` 5/5 pass; with `XLA_FLAGS=--xla_force_host_platform_device_count=2`, 9/9 together
    with `test_stream_io.py`;
  - `test_stream_io.py` 6/6 with `SLURM_JOB_ID=local` (its JAX tests are skipped otherwise);
  - `test_graph_loading.py`, `test_periodic.py`, `test_race_lr.py` pass;
  - `test_neighbors_vesin.py` 4/4 with vesin.
- **Speed** (report §15): see the benchmark summary there.
- **End-to-end**, local GPU (RTX 4060 Ti, 50 % SMs).
  - **Setup:** RACE_LR + charge loss (0.1) + total charge + spin; OMol25 public subset, 10k train / 1k valid /
    2.9k test; 5 epochs; configs `input/stream_e2e/`, runner `runs/stream_e2e/queue.sh`.

  | 5 epochs | xyz (pickle) | stream, mix_blocks 1 | stream, mix_blocks 16 (default) |
  |---|---|---|---|
  | train loss, epoch 5 | 0.440 | 0.552 | 0.437 |
  | valid F RMSE (meV/A) | 274 | 282 | 243 |
  | valid E MAE (meV) | 838 | 701 | 638 |
  | time per epoch | 11.7 s | 11.1 s | 11.1 s |

  - **Why mixing:** without it a pack comes from one block of 128 consecutive rows. In this file those rows are
    conformers of the same molecule, so the batches were low in diversity and the first epochs were worse. Hence
    `mix_blocks` (default 16).
  - **Noise:** two identical runs differ by about 1 % in epoch-1 loss on both paths (GPU atomics). The streaming
    data order itself is exactly reproducible (`test_stream_steps_resume_cursor`).
  - **Resume:** 3 epochs, stop, `restart: true` to 5 epochs. The log shows `Resumed ... epoch=3 step=1764` and
    `stream: start at data epoch 2 step 587`. The data epochs hold 588 / 589 / 587 steps: the packing differs per
    data epoch, so virtual epochs drift slightly against data epochs, as intended. Final valid F RMSE 243 meV/A.

## 6. Not verified, and what to do next

1. **OMol25 / OMat24 aselmdb field layout.**
   - `decode_row` takes total charge, spin and per-atom labels from the row's `key_value_pairs` / `data`.
   - Whether OMol25 rows hold `charge`, `spin` and Löwdin charges there is unknown until the files are available.
   - Run `python -I data_prep/convert_omol25.py inspect <dir>` and adjust `decode_row` and the config keys.
   - `tests/test_stream_train.py::test_aselmdb_row_fields` is synthetic.
2. **Several hosts** (design T5). `stack_local` uses `make_array_from_process_local_data`, and the eval weights and
   the plan counts are allgathered. Not run on more than one process. First test: 2 processes × 1 GPU against
   1 × 2 on 20 float64 steps.
3. **HANGANG T6 throughput smoke**, about 20 min on one GPU node; ask the user before sbatch:
   - install `vesin` and `lmdb` in `bam_cue` (ask first);
   - loader only, W = 2, 4, 8, 16;
   - 500 steps;
   - read from Lustre.
4. **Extxyz offsets are rescanned by every worker** that opens a file (one sequential pass). For one huge xyz this
   costs W passes; split big xyz files or use aselmdb.
5. **Model equivalence on a trained checkpoint** (design T4): stream eval vs in-memory eval of the same structures,
   to 1e-9 relative in float64.
6. **Saving `.last.pkl` every N steps.** Not done; it is saved every `ckpt_last_every` (virtual) epochs. Pick
   `steps_per_epoch` so that an epoch fits well inside the wall time.
