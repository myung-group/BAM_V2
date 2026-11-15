# Handoff: multi-dataset aselmdb training and Thole-type electrostatics (RACE_POLAR)

Written 2026-11-11 on the user's local PC (RTX 4090, 12 cores, conda env `jax`). The request: train `race_v2`
on the corpora in `~/Datasets` (alex, mptrj, oc25, omc25, omol25, opoly26; fairchem-style `.aselmdb` shards), as
was done with BAM-torch (`~/Python/BAM-torch/examples/oc25_mptrj`), and port the Thole-type electrostatics of
`~/Python/BAM-torch/bam_ufo/model/thole.py`.

Related: `docs/HANDOFF_stream_loader.md` (the streaming loader this builds on), `docs/HANDOFF.md` §7–8.

## 1. Decisions of the user (2026-11-11)

- `pip install --no-deps lmdb vesin pytest` into the `jax` env: approved and done.
- Thole port in stages. **Phase 1 (done here):** permanent Gaussian charges, dipoles and quadrupoles, Fukui charge
  partition with the exact total charge per graph, atomic polarizabilities, induced dipoles by CG with implicit
  differentiation, `pbc_split` Ewald + open boundaries, charge and dipole losses.
  **Phase 2 (done later the same day, user request "Thole 2단계 구현"; §4b):** slab dipole correction, electrode /
  charge-group constraints (`charge_constraint: "electrode"`, meta-data `charge_group` / `fixed_charge`), contact
  charge transfer, OC25 slab-dipole labels.
- Frame selection: a fixed seeded subset per shard (`max_frames_per_shard`), with `subset_seed` to draw another
  subset for a continued run. No full size table of the 10^7-frame corpora.

## 2. Environment on this PC

- `conda activate jax`: Python 3.14, jax 0.12.0.1 (CUDA 13 plugin), flax 0.12.12, e3nn-jax 0.21.0, no
  cuequivariance (so `"tp_backend": "e3nn"`), now also lmdb 3.0.0, vesin 0.6.2, pytest 9.2.0.
- The conda **base** env has torch 2.14 + BAM-torch + lmdb but no JAX: use it for torch reference values
  (`tests/data/make_thole_torch_ref.py`).
- This checkout (`~/Python/BAM_V2-racev2`) is **not a git repository** (a copy): nothing was committed.
- GPU runs: `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false BAM_NEIGHBORS=vesin XLA_PYTHON_CLIENT_PREALLOCATE=false`.
  With the default 75 % preallocation the first RACE_POLAR step ran out of memory at a size that peaks at 15 GB
  without it.

## 3. Reading the datasets

All six corpora decode with `stream_io.decode_row` (checked on 200 rows of every source, including
`omol25/train_4M`, `train_nbo` and `neutral_val`):

| source | shards | rows in shard 0 | boundary | labels | notes |
|---|---|---|---|---|---|
| oc25/train | 20 | 369,776 | periodic slabs | E, F, VASP dipole, tags | no stress |
| mptrj/train | 5 | 284,458 | periodic | E, F, stress | `mp2020_corrected_energy` in key_value_pairs |
| alex/train | 350 | 25,260 | periodic | E, F, stress | 711 rows flagged `exclude` in `alex_meta_v2` |
| omol25/train_4M | 80 | 49,835 | open | E, F, charge, spin, NBO / Löwdin / Mulliken charges (68 % with NBO), ORCA multipoles | |
| omol25/train_nbo | 5 | 896,073 | open | as above, every row with NBO, no multipoles | |
| opoly26/train | 80 | 73,786 | open | E, F, charge, spin, Löwdin / Mulliken, multipoles | |
| omc25/train | 40 | 621,756 | periodic | E, F, stress | |

New in `race_v2/data/stream_io.py` (module docstring has the full list):

- A source is `"aselmdb:<dir>"` as before, or a dict `{"path": "aselmdb:<dir>", "name", "max_frames_per_shard",
  "subset_seed", "meta_dir", "task", "task_rules", "energy_key", "charge_label_key", "charge_label_scale",
  "dipole"}`.
  - `max_frames_per_shard`: seeded subset per shard (seed = `[subset_seed, shard index]`); the size pass and the
    training read only these rows.
  - `meta_dir`: rows flagged `exclude` in the BAM-torch meta-data (`<shard stem>.npz`) are dropped.
  - `task` + `task_rules`: the task index per structure; `{"mp2020_hubbard": 5}` routes Materials Project GGA+U
    compositions to task 5 (same rule as BAM-torch `dataset_id_rules`).
- Stream options (`cfg["stream"]`): `"mixed": true` (periodic and open-boundary structures in one batch: every
  graph gets `Sij`, a cell, `globals['pbc']` and `globals['stress_w']`; open structures get zero shifts and a
  placeholder box), `"dipole": true` (ORCA dipole labels of open structures, about the centre of mass, in e Å),
  `"atom_energies_by_task": {task: file | list | {Z: E0}}` (the BAM-torch files
  `get_enr_avg_per_element_*.txt` are read directly).
- The size table also records the cell volume and lengths (`sizes.npz` `cell`, meta `max_cell_volume`,
  `max_cell_length`, `n_periodic`); RACE_POLAR sizes its reciprocal sums from them.
- Loss (`training/utils.py`): the stress term and the stress metrics use `globals['stress_w']` (OC25 and the
  molecules have no stress); the charge loss uses `nodes['ref_charges_w']` (rows without labels).
- Speed (one process, vesin): 0.3–1.9 ms per structure for read + graph at cutoff 6 Å, so a size pass of about
  1 M structures per source takes minutes with 12 workers.

Config generator: `scripts/train/make_multidataset_configs.py` (frame counts per shard default to the BAM-torch
`input.json` values; `--one-shard --frames N` for small tests). Tasks: oc25 1, PBE (mptrj + alex) 2, omol25 3,
opoly26 4, PBE+U 5, omc25 6, as the BAM-torch `dataset_id`s.

## 4. Thole electrostatics

- `race_v2/models/thole.py`: the electrostatics (pure `jax.numpy`). Module docstring = the energy functional and the
  differences from the torch code:
  - static shapes: every periodic graph keeps its `n_k` reciprocal vectors of smallest |k| inside the sphere
    `k_max` out of a fixed integer half grid; a graph that needs more gets NaN energies (raise `polar.n_k`);
  - open-boundary graphs use dense N × N pair arrays of the padded batch;
  - one CG for the whole batch (`jax.scipy.sparse.linalg.cg`, implicit differentiation, also for the second
    derivative that force training needs).
- `race_v2/models/race_polar.py`: `"model": "RACE_POLAR"` = RACE_LR (charge / spin embedding, task table; Qeq off)
  + the heads of the torch block (multipole, quadrupole, polarizability, Fukui) + `thole.thole_energy`. Returns
  `charges` and `dipole` next to the RACE outputs. Config `"polar"` (`thole.POLAR_DEFAULTS`).
- Loss: `"charge_weight"` (per-source `charge_label_key`), `"dipole_weight"`; validation lines print `q_RMSE` and
  `mu_RMSE`.

Verified (CPU, float64 unless noted):

- `tests/test_thole.py` (10 tests): energies, node energies, induced dipoles, total dipoles, potentials, forces
  and cell gradients equal the BAM-torch block to 2e-8 … 5e-8 on a batch of two periodic cells (one charged) and
  two molecules (one charged), for six settings (σ = 1.0 and 0.5, without quadrupoles, without mutual induction,
  without induction, without the width cap and with `ewald_k_tolerance` 1e-5); padding invariance; NaN on a too
  small `n_k`; float32 within 2e-4 eV of float64.
- `tests/test_race_polar.py` (4 tests): exact total charge per graph, every graph of a mixed batch equals the graph
  alone, forces and stress equal central differences, rotation / translation, non-periodic model = mixed model on
  a molecule, a 2 × 1 × 1 supercell has twice the energy, loss terms.
- `tests/test_multidataset_stream.py` (3 tests, JAX-free, synthetic aselmdb).
- Existing tests still pass: `test_stream_io`, `test_stream_train`, `test_race_lr`, `test_graph_loading`,
  `test_periodic`, `test_oam_stress` (51 tests, `SLURM_JOB_ID=local JAX_PLATFORMS=cpu`).
- GPU smoke (RTX 4090, float32, 7 sources × 300 frames of shard 0, cutoff 5 Å, 2 layers × 32 channels, lmax 3,
  e3nn backend): see §5.

## 4b. Phase 2 (2026-11-11)

The settings of the BAM-torch `input.json` "polar" block that phase 1 left out. `make_multidataset_configs.py`
switches them on by default (`--phase1` gives the phase-1 model).

| option (`"polar"`) | what it does | code |
|---|---|---|
| `"charge_constraint": "electrode"`, `"max_groups": 128` | every charge group of a structure gets its target charge (sum of the atoms' `fixed_charge`); group 0 (the electrode / reservoir atoms) takes the rest of the total charge | `thole.constraint_targets`, `thole.equilibrate` |
| `"charge_transfer": "contact"`, `contact_radius` 2.5, `transfer_cutoff` 4.0, `contact_pairs` "charged", `contact_eps` | groups in contact exchange charge; one (F−1)×(F−1) solve per graph | `thole.contact_charges` |
| `"slab_correction": true`, `slab_axis` 2 | E_slab = 2π k_e (M·n)² / V on the graphs flagged `slab`; its field acts on the induced dipoles inside the CG (rank-one term; Sherman–Morrison without mutual induction) | `thole.thole_energy(slab_on, pos_field)` |

Data side (`stream_io.py`, `stream_data.py`):

- Source options `"slab": true`, `"dipole_sign": -1.0` (OC25: VASP counts electrons positive), `"charge_groups":
  false` (ignore the meta-data groups). `meta_dir` must be the directory of the split
  (`oc25_meta_v2/train`, `omol25_meta_v2/train_4M`, …); a root with several files of the same name is refused.
- The stream worker looks up `electrode_mask`, `charge_group`, `fixed_charge` of every row in the meta-data
  (`ShardMeta`: the arrays of the uncompressed npz are memory-mapped) and writes nodes `charge_group` (0 =
  reservoir, 1.. = other groups) and `fixed_charge`. A source without meta-data is one group per structure.
- Slab graphs get nodes `unwrap` (image offsets that make bonded fragments contiguous, `unwrap_offsets` =
  torch `unwrap_fragments`) and globals `slab`. Their dipole label (the row's `dipole`) is used along the slab
  normal only; open-boundary graphs keep all three components (`training_utils._dipole_sq`, the torch
  `dipole_components: "auto"`). `mu_RMSE` is now per labelled element.

Verified:

- `tests/test_thole_phase2.py` (9 tests) against BAM-torch (`tests/data/thole_torch_ref2.npz`): group charges
  (1e-7), contact charges for `pairs` "charged" and "all" with their position derivative (2e-7 / 2e-6), unwrap
  offsets (equal), slab correction with / without mutual induction, without induction and with `slab_axis` 0
  (energies, slab energies, induced dipoles, dipoles, forces, cell gradients to 1e-7).
- `tests/test_race_polar.py::test_phase2_groups_contact_slab`: exact group and graph charges, graph alone = in
  the batch, forces and stress = central differences with everything on, dipole loss elements.
- `tests/test_multidataset_stream.py::test_shard_meta_groups_and_slab_fields`.
- GPU smoke `smoke_polar2` (the phase-1 smoke data and model, all phase-2 options on): train loss 3.34 → 1.66 in
  2 epochs, 23.4 s per epoch (22.8 s in phase 1), peak 16.0 GB.

Differences from torch, and limits:

- **`contact_eps`**: the conductance of a full contact is 1/eps. Torch solves in float64 with 1e-6. Float32
  training here uses 1e-3 (set by `train_3bpa` unless given), which moves the charges by up to about 5e-3 e in the
  test system; float64 (`x64`) uses 1e-6 and equals torch.
- The slab label is projected on the slab normal (label · n) instead of taking the Cartesian `slab_axis`
  component; the same for cells with a and b in the xy plane (OC25).
- No check that slab cells are neutral (torch raises). OC25 cells are neutral.
- Not ported: the reactive charge solve, `charge_transfer: "split_charge"`, `charge_constraint: "component"`,
  external fields, `annotate: "compute"` (groups are read from the meta files only),
  `molecular_groups: "whole"`, group-wise scaling of the reference charges (`reference_charge_scale` by fragment /
  contact; the labels here are scaled per structure).

## 5. Local GPU runs so far

Small plumbing checks only; they say nothing about accuracy.

| run | model | structures / batch | steps per epoch | time per epoch | peak GPU memory | note |
|---|---|---|---|---|---|---|
| `smoke_cs` | RACE_LR (embedding only) | 8 | 292 | 38.6 s (with compilation) | 10.6 GB | 1 epoch |
| `smoke_polar` | RACE_POLAR | 8 | 292 | 22.8 s (epoch 2) | 15.1 GB | train loss 3.34 → 1.65 in 2 epochs |
| `smoke_polar_b4` | RACE_POLAR | 4 | — | — | 10.5 GB | 1 epoch |

Reciprocal budget of that data: largest cell 19,222 Å³ (OC25), lengths 30 / 40 / 61 Å → `n_k` = 7307 vectors per
graph at `ewald_k_tolerance` 1e-5. The Thole block alone (1088 nodes, 35,840 edges, gradient of forces): 38 ms,
1.0 GB.

`pilot_polar_10k` (RACE_POLAR, 7 sources × 1500 frames of shard 0 = 10,500 structures, validation 6 × 150, cutoff
5 Å, 3 layers × 64 channels, lmax 2, 5.57 M parameters, AMSGrad lr 0.01, float32, charge weight 0.08, dipole weight
0.1; 1393 steps per epoch, 127–131 s per epoch after the first = 94 ms per step, 17.4 GB GPU memory in use):

| epoch | train loss | valid loss | valid F MAE (meV/Å) | valid E MAE per atom (meV) |
|---|---|---|---|---|
| 1 | 1.859 | 0.827 | 193.9 | — |
| 2 | 0.915 | 0.634 | 164.5 | — |
| 3 | 0.724 | 0.541 | 152.5 | — |
| 4 | 0.653 | 0.495 | 141.8 | — |
| 5 | 0.575 | 0.467 | 138.8 | 108.3 |

Final validation (all sources together): stress MAE 1.02 GPa, charge RMSE 335 me, dipole RMSE 0.593 e Å (0.805
after 2 epochs of the smoke run). The charge error hardly moved in this short run (358 → 335 me, close to the RMS
of the scaled NBO labels themselves): with weight 0.08 the charge term is about 2 % of the loss. Whether the
charges train at a useful rate needs a longer run and probably a larger `charge_weight`; BAM-torch needed the same
tuning (`qatom_lambda`, its notes of 2026-10-18).

## 5b. Prepared for the first real run (2026-11-11; **not started**: show the user the commands first)

**Bug fixed while benchmarking:** started with `python -m race_v2.training.train`, every stream worker re-imported
the package and created its own CUDA context (about 0.4–0.6 GB of GPU memory per worker, and a
`BrokenProcessPool` when the GPU was full). `stream_io.worker_context()` now starts the fork server with
`JAX_PLATFORMS=cpu`, `CUDA_VISIBLE_DEVICES=` and without `BAM_DISTRIBUTED`. **The peak-memory numbers of §4b and
§5 were measured before this fix and include about 5 GB of worker contexts** (8 + 4 workers).

Benchmark after the fix (RACE_POLAR with all phase-2 options, 7 sources × 1500 frames of shard 0, cutoff 6 Å,
lmax 2, 4 edge buckets, float32, e3nn backend; 150 steps of the second epoch; `nvidia-smi` memory is an upper bound
because the allocator grows in large steps):

| model | parameters | structures per batch | largest batch shape | ms per step | ms per structure | hours per 1M structures | GPU memory |
|---|---|---|---|---|---|---|---|
| 3 layers × 64 | 5.57 M | 8 | 1408 nodes, 77,824 edges | 109 | 13.7 | 3.8 | 20.2 GB |
| 3 layers × 64 | 5.57 M | 4 | 704 nodes, 38,912 edges | 61 | 15.2 | 4.2 | 11.2 GB |
| 4 layers × 96 | 16.9 M | 4 | same | 104 | 26.0 | 7.2 | ≤ 20.1 GB |
| 4 layers × 128 | 29.8 M | 4 | same | 131 | 32.7 | 9.1 | ≤ 20.1 GB |
| 5 layers × 128 | 37.7 M | 4 | same | 155 | 38.8 | 10.8 | ≤ 20.2 GB |

For comparison, the BAM-torch run notes give 284 ms per step of 8 structures (35.5 ms per structure) for 5 layers,
128 channels, lmax 3 on the same GPU. The times above will rise with the full data: the reciprocal budget `n_k`
follows the largest cell of the whole subset (25,544 Å³ in 6000 OC25 rows against 19,222 Å³ in the benchmark
subset).

Configs (made with `scripts/train/make_multidataset_configs.py --name multi_full_<size> --model polar --layers ..
--hidden .. --lmax 2 --cutoff 6.0 --batch 4 --buckets 4 --workers 8 --steps-per-epoch 25000 --epochs 1000
--patience 10 --sizes-dir runs/multi/sizes_full_c6`):

- `research/input/multi/multi_full_L4x96.json`, `research/input/multi/multi_full_L5x128.json`: the BAM-torch frame counts
  (10,249,550 training structures: oc25 1.00 M, mptrj 1.00 M, alex 1.00 M, omol25 2.00 M, omol25_nbo 1.25 M,
  opoly26 3.00 M, omc25 1.00 M; validation 16,284), all with meta-data, OC25 as slabs.
- One virtual epoch = 25,000 steps of about 4 structures = about 100,000 structures (about 43 min for 4 × 96,
  65 min for 5 × 128, plus the validation pass); a checkpoint after every virtual epoch; about 102 virtual epochs
  per pass over the subset (3 days / 4.6 days). `n_epochs` 1000 is an upper limit, not a plan.
- Untuned choices carried over, to be watched in the first hours: AMSGrad lr 0.01 in float32 (the recipe value;
  stable in the 5-epoch pilot at 8 structures per batch, not tried at this scale), `charge_weight` 0.08 and
  `dipole_weight` 0.1 (BAM-torch values; the charges hardly trained in the pilot), plateau schedule with
  patience 10 virtual epochs.

Commands (from the repository root):

```
conda activate jax
BAM_NEIGHBORS=vesin python -I scripts/data/multi_sizes.py research/input/multi/multi_full_L4x96.json --workers 11
bash scripts/train/run_multi.sh multi_full_L4x96
```

The first makes the size tables in `runs/multi/sizes_full_c6/{valid,train}` (CPU only; 3000–6000 structures per
second with 10 workers in a 6000-row test per corpus, so roughly an hour for 10.2 M random-access rows) and both
configs share them. The second trains; the same command resumes after a stop (`restart: true`). Check after the
size pass: `max_cell_volume` and `max_cell_length` in `sizes_meta.json` (they set `n_k` and the k grid), and the
bucket lines at the top of `loss.out`.

## 6. Not done, and what to do next

1. **A real training run with the phase-2 model** has not been made: only 2-epoch plumbing runs; §5b has the
   prepared configs and commands. The pilot of §5 is the phase-1 model. Open questions for it: the charge weight (§5), `contact_eps` in float32 (§4b), and the
   model size that fits 24 GB with the e3nn backend.
2. **Per-epoch resampling.** The subset is fixed for a run. To continue on other frames: change `subset_seed`
   in the sources, point `stream.train.sizes` to a new directory (or delete the old table) and restart with
   `restart: true`.
3. **Memory and speed of the open-boundary part.** Dense N × N pairs per batch; batches of many small molecules
   would be cheaper with per-graph blocks or pair lists. The reciprocal arrays are N × n_k for every node,
   sized by the largest cell of the data.
4. **cuequivariance backend** is not installed in `jax` (CUDA 13); the e3nn backend needs much more memory
   (10.6 GB for 2 layers × 32 channels at 36k edges).
5. **Meta embedding / ontology of BAM-torch** (`meta_embedding`, `dataset_cards.yaml`) is not ported; the task
   table of RACE_LR (`long_range.n_tasks`) plus one E0 table per task stands in for it.
6. **Torch checkpoints cannot be loaded** (different backbone); the heads follow the torch block but the
   parameters are new.
7. **Inference tools** (`predict.py`, `eval_sets.py`, `dump_frame_errors.py`) rebuild the model from the config;
   the reciprocal budget taken from the training data is stored in `<ckpt_name>.polar.json` and read back. They
   were not run with a RACE_POLAR checkpoint, and their in-memory graphs carry no `pbc` flag (one boundary type
   per call).
8. README and the methods PDF are unchanged: there is no new benchmark result yet.
