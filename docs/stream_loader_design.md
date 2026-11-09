# Large-scale streaming data pipeline for the RACEv2 trainer (BAM_V2)

Date: 2026-11-06. Read-only investigation; nothing in either repository was changed.

Repositories: `BG` = `/home01/t589a02/prog/BAM-GAMP-jax`, `BV` = `/scratch/t589a02/BAM_V2`.
Every `path:line` is relative to one of them. Data measurements are marked **[measured]** and come from manifest
reads, one unpickled shard, `ls -l` of two source directories and a 200-row lmdb probe (glogin01, no JAX).

**Requirement (updated by the user):** the RACEv2 loader must not depend on pre-pickled, padded super-batch shards.
The design is:
1. a size-metadata pass that writes a readable per-structure table (index, n_atoms, n_edges at the model cutoff);
2. a streaming loader whose worker processes read raw structures, compute neighbour lists on the fly and pack them
   into a few fixed bucket shapes, with prefetch, shuffling per epoch and sharding across hosts.

The BAM-GAMP shards remain an optional, read-only source. Sections 1-2 describe the shard format and loader as
background. Section 3 lists the convention gaps. Section 4 is the plan.

---

## 0. Summary

* **Shards (background).** One pickle holds up to 200 padded super-batches with the same `(B, N_pad, E_pad)`. The
  edges are stored at **6 Å**. RACEv2 runs at **5 Å**. Energies are raw float32 E_DFT. There are 25 shape
  directories `B{B}_N{N}_E{E}`. OMat24-train shards take 2.99 TB [measured]. The raw aselmdb sources are roughly
  7-15x smaller.
* **New pipeline.**
  * Source readers: aselmdb (random access by row id), extxyz (byte-offset index), and the BAM-GAMP shards as a
    reader only.
  * A size pass writes `sizes.tsv` (readable) and `sizes.npz` (fast) per source.
  * Each data epoch has a plan built from the size table only: a block permutation, blocks dealt to hosts, greedy
    packing into 1-4 fixed buckets, and per-step bucket labels.
  * Each host runs a forkserver pool of JAX-free workers. They read blocks, run the ASE neighbour list and build
    graph arrays. A packer thread runs `fixed_collate_fn`, stacks the local devices and builds the global array with
    `jax.make_array_from_process_local_data`.
  * Training runs in virtual epochs of `steps_per_epoch` steps. A data cursor is saved in the checkpoint, so a run
    can resume in the middle of a data epoch.
* **Throughput.** The GPU uses about 180 OMat24 structures/s per GPU in float64 (measured step time on the OAM
  subset). An ASE neighbour-list worker does about 300 structures/s. So 1 worker per GPU is enough in float64 and
  about 3-4 in float32. Plan for 8 per GPU (32 of 288 cores per node).
* **Wall-time limit.** One OMat24 epoch on one 4-GPU node takes about 39 h, which is more than the 24 h wall-time
  limit. Mid-epoch resume is therefore required.

---

## 1. BAM-GAMP-jax shard format

### 1.1 File layout and naming

* Each data root has one directory per shape `B{B}_N{N}_E{E}` (`BG/scripts/reshard.py:189-191`;
  `BG/scripts/convert_aselmdb_to_shards.py:63, 456-459`). The loader parses the name back with
  `shape_key_from_path` (`BG/bam_jax/data/loader.py:290-298`).
  * `omat24_train` has 25 shape directories, from `B2_N64_E24576` to `B128_N512_E32768` [measured: `ls`].
  * `mp_data/jax_data_v3_raw/train` has the same shapes plus `B2_N1024_E49152` and `B2_N1024_E65536` [measured].
* File names:
  * `shard_<subset>.t<k>_<kkk>.pkl` for a converter task (`convert_aselmdb_to_shards.py:466`);
  * `shard_rs<GG>_<iiii>.pkl` after a re-split. Shapes with fewer than `--min_shards` = 64 files are re-split
    (`:77-85, 597-644, 890`).
* Side files next to the shape directories:
  * `convert_config.json`, which pins the options, including `cutoff` and the ordered `files` list of input aselmdb
    files (`:942-959`);
  * `manifest.json`, with per-shape and per-file counts and the loader-readable `"shards"` table (`:787-830`);
  * `_progress/<tag>.json` and `_progress/<tag>.graphs.npz`, one sidecar per task (`:60-68, 493-501`). The
    `.graphs.npz` sidecar holds, for every **kept** graph, `src` (index into `convert_config.files`), `row_id`,
    `n_atoms`, `n_edges` at 6 Å, the raw energy, E/atom, fmax and stress statistics. There are 20,174 files in
    `omat24_train/_progress` [measured]. **These sidecars already record which rows passed the filters, and the
    size pass can reuse them (§4.2).**
* Pickle top level: `{"schema_version": "v2", "shape_key": [B, N, E], "batches": [...], "energy_reference": "none",
  "dataset_id": k}` (`convert_aselmdb_to_shards.py:467-469`; `BG/bam_jax/data/schema.py:116-129`). Pickle protocol
  5 (`HIGHEST_PROTOCOL`, `:384`) needs Python ≥ 3.8. The login node's `python3` 3.6 cannot read the shards; use the
  `bam_jax` env [measured].

### 1.2 Super-batch keys and shapes

Shapes and dtypes are listed in `schema.py:25-46, 50-89`. A sample file `salex_val/B8_N512_E16384/shard_rs01_0061.pkl`
was inspected [measured]:

| key | shape | dtype | meaning |
|---|---|---|---|
| positions | (N_pad, 3) | f32 | Cartesian, **not wrapped** (`convert_aselmdb_to_shards.py:31-32`) |
| species | (N_pad,) | i32 | `species_encoding[Z]`, see §3.5 |
| node_mask | (N_pad,) | bool | real atom |
| batch | (N_pad,) | i32 | graph slot; padding = `PAD_GRAPH_INDEX` = 2^31−1 (`schema.py:20-22`) |
| forces | (N_pad, 3) | f32 | eV/Å |
| edge_index | (2, E_pad) | i32 | row 0 = i, row 1 = j of ASE `ijS` (§3.2) |
| cell_offsets | (E_pad, 3) | f32 | integer image shift S (values −1..1 in the sample) [measured] |
| edge_mask, edge_batch | (E_pad,) | bool, i32 | padding edge_batch = PAD sentinel |
| cell | (B, 3, 3) | f32 | unused slots hold the identity (`reshard.py:240-241`) |
| energy | (B,) | f32 | raw total E_DFT, eV |
| stress | (B, 6) | f32 | eV/Å³, ASE sign, Voigt xx yy zz yz xz xy |
| graph_mask, num_atoms, num_edges, dataset_id | (B,) | bool, i32 | real graph, counts, source id (2 = OMat24, 3 = sAlex) |

Layout details:

* **Per-graph slot layout.** Graph g occupies node rows `[g·N/B, g·N/B + n_g)` and edge rows `[g·E/B, …)`. The
  padding rows are **interleaved** between graphs, not trailing (`reshard.py:248-270`). In the sample, slot 64,
  graph 0 has 48 atoms and `batch[60:64]` = PAD, then graph 1 starts at row 64 [measured].
* **Padding values are not neutral.** Padding atoms have species 0 (= H) and padding edges point to atom 0
  (`reshard.py:231-239`). Always use the masks.
* **Graphs per super-batch.** B is a power of two in [2, 128]: the largest B with B·n_pad ≤ `atom_budget` = 512 and
  B·e_pad ≤ `edge_budget` = 32768 (`reshard.py:48-52, 69-73, 109-112`; manifest budgets [measured]). The last
  super-batch of a slot group can be partial (`convert_aselmdb_to_shards.py:474-476`). Each file holds at most 200
  super-batches (`--batches_per_shard`, `:487-488, 889`).
* **Edges are stored, not computed at load.** `atoms_to_graph_numpy` calls ASE `neighbor_list("ijS", cutoff)`
  (`BG/bam_jax/data/preprocess.py:276-346`; `convert_aselmdb_to_shards.py:347-362`). The loader only unpickles
  (`loader.py:1-6, 328-342`).
* **Cutoff.** All four OAM trees use `cutoff: 6.0` (their `convert_config.json` [measured]; default
  `convert_aselmdb_to_shards.py:869`). In the sample, the largest stored edge length is 5.79 Å [measured]. The MPtrj
  raw manifest has no cutoff field [measured]; it inherits the BAM-torch layout (`convert_aselmdb_to_shards.py:29-31`).
* **Stress.** eV/Å³, ASE sign (positive = tensile), ASE Voigt order (`schema.py:41`;
  `convert_aselmdb_to_shards.py:23-25`). A 3x3 row stress is converted at `:166-168, 320`. Rows with any component
  above 80 GPa = 0.4993 eV/Å³ are dropped (`:338-340, 926`; `convert_config.json` filters).
* **Energy reference.**
  * OMat24 and sAlex: raw, uncorrected E_DFT (OMat24's MP2020-corrected energy is ignored),
    `energy_reference: "none"` (`convert_aselmdb_to_shards.py:18-21, 165`).
  * MPtrj `jax_data_v3_raw`: E0 added back, stored as `float32(float64(E_ref) + Σ atom_ref)` (`BG/scripts/mptrj_v3_to_raw.py:1-20`).
    The manifest records `check_max_abs_energy_dev` = 1.2e-4 eV (float32 rounding) [measured].
* **Filters** (`convert_aselmdb_to_shards.py:41-58, 298-344`): E/atom > 0, energy floor −5 eV/atom, |F| > 50 eV/Å,
  stress > 80 GPa, non-periodic, |det cell| < 1e-3, Z outside the 89-element encoding.

### 1.3 Data volume [measured from the manifests and `ls -l`]

| tree | graphs | atoms | atoms/str | edges/atom at 6 Å | shard bytes |
|---|---|---|---|---|---|
| omat24_train | 100,195,354 | 1.866e9 | 18.6 | 39.3 | 2.99 TB (58,556 files) |
| omat24_val | 1,019,060 | 1.90e7 | 18.7 | 39.4 | 30.9 GB |
| salex_train | 10,416,201 | 1.07e8 | 10.3 | 44.9 | 214 GB |
| salex_val | 551,591 | 5.70e6 | 10.3 | 45.0 | 11.4 GB |
| mptrj raw train / valid | 1,501,443 / 73,202 | 4.68e7 / 2.42e6 | 31.2 | — | 97.6 / 5.1 GB |

Raw sources for comparison:

* `oam_lmdb/omat24_train/aimd-from-PBE-1000-npt`: 490 files, 39.4 GB, 21.27M rows, about 219 B per atom
  [measured]. Extrapolated to OMat24 train: about 0.2-0.4 TB.
* `oam_lmdb/salex_train/train`: 350 files, 11.2 GB.

The raw sources are therefore 7-20x smaller than the shards, mostly because the shards store 6 Å edges. At 5 Å the
OAM subset gives 27.9 edges/atom (`BV/runs/oam64_omat24_v2_s42_cue/loss_oam64_omat24_v2_s42_cue.out:257`), so
OMat24 has about 52e9 edges per epoch.

---

## 2. How BAM-GAMP-jax streams the shards (background)

* **Two-phase epoch.**
  * `plan_epoch` (`loader.py:671-803`) runs in the main thread because it issues the cross-rank min collective
    (`:680-684`).
  * `iter_planned` (`:823-918`) does the unpickling and is safe in a prefetch thread.
  * The trainer calls them at `BG/bam_jax/training/trainer.py:1557-1574, 1627-1629`.
* **Shuffling.**
  * The order of shape buckets is permuted per epoch (`loader.py:729-732`).
  * Files are permuted inside each bucket (`:749-754`).
  * Super-batches are shuffled inside each file (`:891-899`).
  * The graphs inside a super-batch were fixed at conversion: seeded per task (`convert_aselmdb_to_shards.py:449,
    455`) and never re-mixed.
* **Mixing.** Per-source replication follows the `sqrt`, `uniform`, `proportional` or `repeat` rule
  (`loader.py:8-22, 577-640`). Validation uses `proportional` (`trainer.py:1741-1753`).
* **Multiple hosts.** Each rank takes the files `entries[pid::P]` of each bucket (stride), or an LPT-balanced set
  (`loader.py:131-141, 656-668, 756-764`). The per-bucket super-batch count is reduced to the minimum over ranks
  (`:790-793`), using `process_allgather` (`BG/bam_jax/training/distributed.py:70-85`). Each rank yields a
  self-consistent local batch, and `prepare_batch_for_step` builds global arrays with
  `jax.make_array_from_process_local_data` (`distributed.py:173-214`).
* **Prefetch.** A background thread with a bounded queue, depth 3. It does the unpickling and the host-to-device copy
  (`loader.py:1003-1033`; `trainer.py:1577-1580, 1629`).
* **Epoch definition.** Every file of every source is read once (times its replication), minus the min-truncation per
  bucket. The plan fingerprint (`loader.py:811-820`) and `skip_steps` allow a mid-epoch resume without unpickling the
  skipped files (`:836-846`; `trainer.py:1583-1606`).
* **Recompilation.** One `(B, N, E)` per directory, so jit compiles once per shape (`schema.py:7-9`). The OAM
  converters reuse MPtrj v3's shape table to share the compile cache (`convert_aselmdb_to_shards.py:35-39`).
  `pack_super_batches` concatenates k same-shape super-batches into one step (`loader.py:975-1000`).
* **Evaluation.** No shuffle, every shard once, but still min-truncated across ranks (`trainer.py:1737-1755`). A
  multi-rank validation therefore drops some batches, so it is not exact.

---

## 3. Gaps between the shard / raw formats and RACEv2

### 3.1 What RACEv2 reads from a `jraph.GraphsTuple`

* `nodes['species']` and `nodes['positions']` (`BV/bam_omat24/utils.py:299-301, 204-207`;
  `BV/bam_omat24/race_nnx.py:1687`).
* `senders` and `receivers` (`race_nnx.py:1694, 1747`).
* `n_node`: nodes must be contiguous per graph, with the padding graph last. It is used by `node_graph_idx`
  (`utils.py:259-267`) and `segment_sum(..., indices_are_sorted=True)` (`utils.py:385-390`).
* `n_edge`: edges must be contiguous per graph, because the cell is repeated per edge by `n_edge`
  (`utils.py:213-228`; `race_nnx.py:1698-1699`).
* Periodic graphs also need:
  * `edges['Sij']`;
  * `globals['cell']` (n_graph, 3, 3), `globals['cutoff']` (n_graph,) and `globals['volume']`
    (`utils.py:309-336, 395-398`);
  * `globals['stress']` for `stress_loss` (`BV/bam_omat24/training_utils.py:261-263`).
* The loss reads `globals['energy']` and `nodes['forces']`, plus the optional `loss_weight` entries
  (`training_utils.py:224-243`).

The reference builder of these fields is `data_3bpa.preprocess_graph` (`BV/bam_omat24/data_3bpa.py:39-94`). Batches
are padded by `jraph.pad_with_graphs` in `fixed_collate_fn` (`BV/bam_omat24/data_nnx.py:98-114`), which needs
`n_nodes < n_node_pad` (the padding graph needs at least one node).

**Gap.** The shard slot layout interleaves padding (§1.2), and padding atoms reuse species 0 and atom 0. Converting a
shard super-batch means unpacking the real graphs (as `reshard.unpack_graphs` does, `BG/scripts/reshard.py:194-225`)
and re-batching them with jraph. Shard tensors cannot be re-labelled in place.

### 3.2 Edge direction and Sij sign

* BAM-GAMP: `edge_index[0]` = i, `edge_index[1]` = j, and D_ij = x_j − x_i + S·cell
  (`convert_aselmdb_to_shards.py:29-31, 795-796`; `preprocess.py:331-335`). In the sample, `edge_index[0]` is sorted
  and the edge lengths are 2.35-5.79 Å [measured].
  * Note: `schema.py:33` calls row 0 "sender". That label is misleading.
* RACEv2: receivers = i, senders = j, `Sij = −S` (`data_3bpa.py:52-55, 73, 88-92`; `BV/bam_omat24/neighbors.py:4-7`).
  The model forms r = x_s − (x_r + Sij·cell) (`utils.py:229`), which equals D_ij.
* **Mapping from a shard:** receivers = row 0, senders = row 1, `Sij = −cell_offsets`. The ASE full list is symmetric,
  so the opposite pairing (senders = row 0 with `Sij = +S`) gives the same edge set. Mixing the two conventions
  (senders = row 0 with `−S`) gives wrong boundary vectors. This exact bug happened before, in `build_pkl.py`
  (`BV/bam_omat24/build_pkl_v2.py:1-13`).
* **Test:** for every edge, |r| < r_c and |r| equals the ASE distance.

### 3.3 Cutoff mismatch

The shards are at 6 Å; RACEv2 OAM runs are at 5 Å (`BV/input/input_3bpa_omat24_oam64_omat24_v2_s42_cue.json`:
`cutoff 5.0`). The per-edge radial cutoff comes from `globals['cutoff']` (`utils.py:223-228`), and the envelope is
zero for r ≥ r_c (`utils.py:57`).

* Feeding 6 Å edges with cutoff = 5 would be mostly harmless numerically, but it adds about 41% more edges (39.3 vs
  27.9 edges/atom) of wasted work and memory.
* It also risks terms that are not multiplied by the envelope, for example the UFF pair repulsion or `layer_cutoff`
  edges (`race_nnx.py:1688-1703, 1829`).
* **Rule:** edges must be rebuilt at the model cutoff, or the stored edges filtered to r < r_c. Filtering is exact
  only when the stored cutoff is at least r_c and is known; the MPtrj shards do not declare theirs (§1.2).
* The streaming design always builds the neighbour list at r_c with `neighbors.neighbor_pairs` (`neighbors.py:14-30`),
  the same function as the in-memory path.

### 3.4 Energy reference (E0)

* RACEv2 stores E − Σ_i E0[species_i] and trains with `l_train=True`; the model does not add E0
  (`data_3bpa.py:8-11, 58, 66`; `utils.py:365-368`; `BV/bam_omat24/train_3bpa.py:424`).
* The raw shards and aselmdb store raw E. **The worker subtracts E0 in float64.**
* E0 = `atom_ref` of `/scratch/t589a02/oam_data/refs/<ds>_ls.npz` (keys `atom_ref` (89,), `species_encoding` (95,),
  …; [measured]). These are the least-squares references of the full training split (`BV/bam_omat24/oam_subset.py:17-19,
  108-114, 125`). `atom_ref` is indexed by the encoded species index (§3.5).
* Mixing sources with different references (MPtrj + sAlex) needs one table (BAM-GAMP's "k = 1" convention,
  `convert_aselmdb_to_shards.py:826-828`) or a per-source table.
* **Precision.** Shard energies and positions are float32, so |E| ≈ 1000 eV has an ulp of about 6e-5 eV (the MPtrj
  dev is 1.2e-4 eV [measured]). The aselmdb rows are float64 JSON [measured probe]. For float64 training, prefer the
  aselmdb source.

### 3.5 Species index mapping

* Shards: `species = species_encoding[Z]`, with H → 0, …, Bi → 82, Ac..Pu → 83..88, and −1 for unsupported Z
  (`BG/bam_jax/data/preprocess.py:226-268`; `convert_aselmdb_to_shards.py:26-28`). The encoding is Z-sorted and
  contiguous; `oam_subset.py:109-113` asserts this.
* Do **not** use `Z_to_idx`, the identity map; it shifts every element by one slot (`preprocess.py:229-240`).
* RACEv2: species index = rank of Z in the sorted `atomic_numbers` (`data_3bpa.py:34-36`). The OAM configs list the
  same 89 Z values, so the two indices coincide.
* In general, use the lookup table `lut[enc_idx] = atom_indices[idx_to_z[enc_idx]]`, and refuse elements the run does
  not cover.
* **Hazard.** `build_model` indexes the per-species tables (`species_radii`, `species_z`, descriptors, shell radii)
  by the position in `cfg['atomic_numbers']` (`train_3bpa.py:456-476`), but the graphs use the sorted rank. An
  unsorted list would silently mismatch. **Assert that the list is sorted in stream mode.**
* E0 per RACE species: `atom_energies[i] = atom_ref[species_encoding[Z_i]]`.

### 3.6 Stress

* Model: Voigt order [0, 4, 8, 5, 2, 1] = xx, yy, zz, yz, xz, xy, computed as virial / V (`utils.py:392-409`). This is
  (1/V) dE/dε, verified by `BV/tests/test_oam_stress.py:73` (`test_stress_is_strain_derivative`).
* Targets: ASE `get_stress(voigt=True)` (`data_3bpa.py:77-85`).
* Shards and aselmdb use the same order and sign (§1.2). **No conversion is needed.** Units are eV/Å³ everywhere; the
  metrics multiply by 160.2 to report GPa (`train_3bpa.py:655-657`).

---

## 4. Plan

### 4.1 Overview

```
offline (Slurm job, once per source and cutoff)
  raw source ──► size pass (JAX-free workers: read + neighbour count at r_c)
              ──► <sizes_dir>/<file-tag>.sizes.tsv  (readable: index, n_atoms, n_edges, file, key)
                  <sizes_dir>/sizes.npz             (same columns, fast load)
                  <sizes_dir>/sizes_meta.json       (cutoff, source list, filters, sums, sha of the tables)

training (every host process)
  sizes.npz ──► plan_data_epoch(seed, data_epoch, host, n_hosts, n_local)   [pure numpy]
                 blocks → hosts → per-bucket packs → per-step bucket labels
  host's blocks ──► worker pool (forkserver, W procs, JAX-free):
                     read rows → ASE neighbour list at r_c → graph arrays (E−ΣE0, −S, cell, volume, stress)
              ──► packer thread: wait for each planned pack's members → fixed_collate_fn → stack n_local packs
                     → jax.make_array_from_process_local_data → bounded queue (depth 3-4)
              ──► main loop: train_step (shard_map over 'dp'), as now
```

### 4.2 Raw sources (how to read them)

**aselmdb (OMat24, sAlex, OMat24-1M)**

* Open: `lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=4)`
  (`convert_aselmdb_to_shards.py:174-178`).
* Row ids come from `nextid`/`deleted_ids` with a fallback key walk (`:230-251`).
* Random access by id: `txn.get(str(rid).encode("ascii"))` (`:417`). Values are zlib(JSON) (`:181-187`). Fields
  arrive as nested lists or as ASE `__ndarray__` (`:190-227`).
* Probe on `omat24_train/aimd-from-PBE-1000-npt/db_0.aselmdb` [measured]: 43,848 entries, 4 KB pages, B-tree depth 3,
  mean value 1.0 KB; 200 random gets took 57 ms (0.29 ms each) and decoding took 5 ms.
* Keys are ASCII decimal, so they are **sorted lexicographically**. Define blocks as runs of K consecutive keys in
  lexicographic order and read each block with one cursor `set_range(first_key)` followed by `next()`. This gives
  near-sequential page reads instead of 3-4 random page reads per row.
* Filters. The kept set must equal the converter's, so both pipelines train on the same data. Reuse the kept
  `(src, row_id)` pairs from `<shard_root>/_progress/*.graphs.npz` (`:493-501`); `src` indexes the
  `convert_config.json` `files` list (`:948`). This also avoids re-implementing the energy-floor table, whose path is
  pinned in `convert_config.json` filters. For sources without sidecars, a copied `check_row` (`:298-344`) applies
  the same filters.

**extxyz (OAM subsets, all other benchmarks)**

* The size pass records the byte offset of each frame.
* A worker seeks to the offset, reads `natoms + 2` lines into a `StringIO` and calls
  `ase.io.read(buf, format="extxyz")`. This is O(frame).
* Do not use `ase.io.read(path, index=k)` or `iread` from an offset; they scan the file.
* `graph_sizes.py` reads every frame into memory (`BV/bam_omat24/graph_sizes.py:41`), which is fine only for the
  small benchmark files.

**BAM-GAMP shards (optional, reader only; currently the only on-disk MPtrj source)**

* The raw MPtrj download is not on disk (`BV/docs/DATASETS.md:42`).
* The unit of access is a whole file: one unpickle, about 30 MB on average for MPtrj (97.6 GB / 3,273 files)
  [measured].
* Locator: `(file, super-batch b, slot g)`. Unpack as `reshard.unpack_graphs` does (`reshard.py:194-225`), then
  rebuild the edges at r_c from the float32 positions. Filtering stored edges is an optional speed-up when the tree's
  `convert_config.json` declares a cutoff ≥ r_c.
* Block = one file (it cannot be split cheaply).

### 4.3 Size pass

* New CLI: `python -I bam_omat24/stream_io.py sizes …`. It generalises `graph_sizes.py` (`graph_sizes.py:1-16,
  32-52`) and keeps its JAX-free import trick: it imports `neighbors` directly, because the package `__init__` imports
  JAX and the model (`graph_sizes.py:28-29`; `BV/bam_omat24/__init__.py:9-50`).
* Per input file it writes `<tag>.sizes.tsv` in the existing format: header `# source`, `# cutoff`,
  `index n_atoms n_edges` (`data_3bpa.py:280-290`), plus extra columns `file key`. `read_sizes_tsv` reads columns 1
  and 2, so it stays compatible.
* At the end it writes `sizes.npz`, with dtypes `n_atoms` u16, `n_edges` u32, `file` u16/u32 and `key` u32/i64
  (about 12 B per structure, 1.2 GB for OMat24). It also writes `sizes_meta.json`, which holds:
  * cutoff, source paths, converter/filter provenance and per-file sha256 of the TSVs;
  * N, Σ atoms, Σ edges;
  * `avg_n_neighbors` = Σ edges / Σ atoms over the **full** split;
  * histograms.
* A 100M-row `np.loadtxt` (as `read_sizes_tsv` does) would be far too slow. The trainer reads the `.npz`, and the TSVs
  stay the human-readable record.
* Cost: OMat24 + sAlex train and val are about 112M structures. At about 300 structures/s per process that is about
  104 process-hours, or about 25 min on one GH200 node with about 256 workers. The converter measured about 50 min
  for 100.8M rows at 6 Å on 288 cores, with neighbour lists taking 93-97% of the time
  (`convert_aselmdb_to_shards.py:95-98`); treat that as an upper bound.
* It must run as a Slurm job (`graph_sizes.py:3-4`).
  * The `bam_jax` env is aarch64, so the job runs on a GPU node with `--cpus-per-task` on the srun line, as in the
    converter example (`convert_aselmdb_to_shards.py:99-107`).
  * The alternative is an x86 env for `normal_cpu`.
  * Per the HANGANG policy: submit from /scratch, add the comment tag, ask the user before sbatch, and run a small
    test job (one subset directory) first.

### 4.4 Buckets from the size table

* `derive_buckets(n_atoms, n_edges, n_buckets, graphs_per_pack, max_graphs)` reuses `graph_sizes.budgets` (node pad =
  max(batch·mean, largest+1) rounded to 64; edge pad = max(batch·mean, largest) rounded to 1024) and its
  equal-count edge quantile split (`graph_sizes.py:55-58, 100-114`). It reports the expected fill with
  `pack_fill` (`:61-70`).
* **Default `n_buckets = 1`.** This is the current `pad_mode: fixed` behaviour (`train_3bpa.py:926-953`).
  * The largest OMat24 structure has 192 atoms (manifest quantiles [measured]), about 5k edges at 5 Å.
  * A budget of about 2048 nodes / 49152 edges holds about 110 average structures, with the largest about 10% of the
    budget, so next-fit fill should be about 93-95%.
  * MPtrj goes up to 444 atoms (the `N1024` slot rung [measured]), so it needs a larger budget or a second bucket.
* `max_graphs` = graph cap per pack; the collate pads to `max_graphs + 1` graphs (`data_nnx.py:111-114`).
* Buckets are stored in `sizes_meta.json` or set explicitly in the config. Each bucket is one jit shape for training
  and one for evaluation.

### 4.5 Per-data-epoch plan (pure numpy, deterministic, no graphs in memory)

`plan_data_epoch(tables, spec, buckets, data_epoch, host, n_hosts, n_local) -> HostPlan`:

1. **Blocks.**
   * aselmdb / extxyz: per source, cut each file's kept structures into blocks of K = 64-256 consecutive keys.
   * Shards: one block per file.
   * Replicate per source `repeat` (mixing as `loader.py:577-640`, if wanted).
   * Permute with `seed ⊕ data_epoch`, using a portable seed hash like `loader.py:1039-1045`.
2. **Hosts.** Deal the files (all blocks of a file to one host per epoch, re-dealt every epoch) LPT-balanced by edge
   total (as `loader.py:656-668`). Each file is then opened by one host and one worker per epoch (§4.6), which keeps
   Lustre metadata load to about one open per file per epoch.
3. **Packing.**
   * Order the host's structures as the concatenation of its blocks, each permuted internally with
     `(seed, epoch, block)`.
   * Split by bucket. Next-fit greedy packing per bucket, with one `np.searchsorted` on the cumulative sums of nodes
     and edges per pack (exactly the `BucketedDataLoader` rule, `data_nnx.py:260-276`, but without a Python loop over
     structures). About 1M packs for OMat24 take about 10 s per host.
4. **Lock-step.**
   * Per bucket, steps_b = min over hosts ⌊packs_{h,b} / n_local⌋. The per-bucket counts are found with one
     `process_allgather` in the **main thread** at the data-epoch boundary, together with a plan signature (as
     `loader.py:680-684, 790-793, 811-820`).
   * The extra packs (≤ imbalance, expected < 0.1%) are dropped, and the count is logged.
5. **Step order.** Bucket labels are interleaved by a stratified schedule (step t of bucket b gets key
   (t + ½)/steps_b), identical on all hosts. With more than one bucket the plan also computes the host's peak
   reorder-buffer size. If it is above a limit, increase K or reduce the number of buckets. With one bucket the
   schedule is trivial.
6. **Data cursor.** `(data_epoch, step)`. Resume rebuilds the plan, starts the reader at the first block that holds a
   member of an unconsumed pack, and discards structures of consumed packs (the same idea as `loader.py:836-846`,
   at structure granularity).

### 4.6 Workers, prefetch and multiple hosts

* **Worker pool.** One per host process: `mp.get_context("forkserver")` (the parent has initialised CUDA, as
  `data_3bpa.py:255-269` already does). The pool has W processes with a per-worker task queue; a file's blocks go to
  the worker `hash(file) % W`. Each worker keeps an LRU cache of open lmdb envs.
* **Back-pressure.** Do not use `Pool.imap`: it consumes the whole task iterator eagerly and has no back-pressure on
  results. Keep at most 2·W blocks in flight and reassemble them by block sequence number.
* **Worker code is JAX-free.** It lives in the top-level module `stream_io` (bam_omat24 is added to `sys.path`,
  forkserver preload), not in the `bam_omat24.` package, so a worker does not import JAX, flax and the model.
* **Worker output** is plain numpy records built by the same function as `preprocess_graph`. Factor its body out into
  a JAX-free `graph_arrays(...)` and keep `preprocess_graph` as the jraph wrapper, so both paths share one code path.
* **Edge check.** Each record carries `n_edges`. The packer asserts that it equals the table value; a mismatch would
  overflow the fixed shape at `data_nnx.py:111-113`.
* **Packer thread.**
  * Builds `jraph.GraphsTuple` objects and runs `fixed_collate_fn` (`data_nnx.py:98-114`).
  * Casts to the run dtype (replaces `_cast_graphs`, `train_3bpa.py:660-665, 748-751`).
  * Stacks the n_local packs and calls `jax.make_array_from_process_local_data(NamedSharding(mesh, P('dp')), local,
    global_shape)` (as `BG/bam_jax/training/distributed.py:173-214`).
  * Pushes to a queue of depth 3-4 (as `loader.py:1003-1033`).
  * It issues no collectives.
* **Why the stacking changes.** The current multi-process path stacks the *same* global group on every process and
  uses `global_from_host` (`data_nnx.py:31-36, 385-395`), so every host collates every pack. That is fine for 10k
  structures but not at OMat24 scale. Host h takes packs `[h·n_local, (h+1)·n_local)` of each step. Which pack goes
  to which device does not affect the update, because gradients are psum-ed with global counts
  (`training_utils.py:729-762`).
* **Mesh.** The existing 1-D 'dp' mesh over all devices is unchanged (`data_nnx.py:529-540`;
  `BV/bam_omat24/distributed.py:43-87`).

### 4.7 Throughput estimate

| quantity | value | source |
|---|---|---|
| GPU, float64, SOAP, 1 GPU, 512 nodes / 14,336 edges per step | 0.55 s/step; 10k structures (302k atoms) in 88-90 s ⇒ ~3.4k atoms/s, ~180 OMat24 structures/s (18.6 atoms) | `runs/oam64_omat24_v2_s42_cue/loss_*.out:270-283` |
| GPU, float32 | assumed 3-5x float64 ⇒ ~550-900 structures/s/GPU | assumption; measure |
| ASE neighbour list per process | ~300 structures/s (user-measured); slower on aimd-3000 (68 atoms/str): 47-177 rows/s at 6 Å | user; `convert_aselmdb_to_shards.py:96-98` |
| lmdb get + decode | 0.29 + 0.03 ms per row | probe [measured] |
| local 2026-11-09 (RTX 4060 Ti, 50 % SMs) | extxyz stream = pickle speed for molecules (1 worker); periodic 110-atom cells need `BAM_NEIGHBORS=vesin` (ASE list ~62 structures/s per worker, vesin ~860) | `docs/local_test_report.md` §15 |
| workers needed per GPU | float64 ≈ 0.6; float32 ≈ 2-3 | ratio of the above |
| **provision** | 8 workers per GPU (32 of 288 cores per node) | headroom for aimd-3000 stretches |
| host-to-device per step | about 2 MB per device (49k edges × (Sij 24 B + 8 B)) | estimate |
| OMat24 epoch, float64 | 1 node (4 GPUs) ≈ 39 h; 16 nodes ≈ 2.4 h | 100.2M / (4 × 180) |
| neighbour-list CPU per OMat24 epoch | ~93 core-hours (≈ 8% of 512 workers over 2.4 h) | 100.2M / 300 |
| Lustre reads per OMat24 epoch | about 0.2-0.4 TB of lmdb pages, vs 2.99 TB of shards | §1.3 |

The main-process collate and `GraphsTuple` construction is the Python-level unknown at float32 speeds. If the queue
runs empty, move `fixed_collate_fn` into the workers, so they return padded arrays.

### 4.8 New modules (signatures)

**`bam_omat24/stream_io.py`** (JAX-free; imported as a top-level module by workers and the CLI)

```python
Structure = dict  # numbers i64 (n,), positions f64 (n,3), cell f64 (3,3), pbc, energy f64, forces f64 (n,3), stress f64 (6,)

class AselmdbSource:   # files from convert_config.json["files"] order or a sorted listing of the user's root
    def __init__(self, root: str, files: list[str], z_ok: np.ndarray, lru: int = 64): ...
    def read_block(self, file: int, keys: np.ndarray) -> list[Structure]: ...   # cursor range read
class ExtxyzSource:
    def __init__(self, path: str, offsets: np.ndarray): ...
    def read_block(self, file: int, keys: np.ndarray) -> list[Structure]: ...
class ShardSource:     # BAM-GAMP shards, reader only; species via idx_to_z of species_encoding
    def __init__(self, root: str, files: list[str], idx_to_z: np.ndarray): ...
    def read_block(self, file: int, keys: np.ndarray) -> list[Structure]: ...   # key = b << 16 | g

def graph_arrays(st: Structure, lut_z: np.ndarray, atom_energies: np.ndarray, cutoff: float) -> dict:
    """Body of data_3bpa.preprocess_graph without jraph: species, positions, forces, senders=j, receivers=i,
    Sij=-S, energy=E-sum E0 (float64 sum), cell, cutoff, volume, stress, n_node, n_edge."""
def read_block_task(task: tuple) -> tuple[int, np.ndarray, list[dict]]:   # worker entry point
def size_pass(source_spec: dict, cutoff: float, out_dir: str, workers: int,
              kept_from: str | None = None) -> None:                       # writes .sizes.tsv, sizes.npz, meta
def main(argv=None) -> int:   # "sizes" sub-command
```

**`bam_omat24/stream_data.py`** (main process; jraph and JAX)

```python
@dataclass(frozen=True)
class Bucket: pad_nodes: int; pad_edges: int; max_graphs: int
@dataclass
class SizeTable: n_atoms: np.ndarray; n_edges: np.ndarray; file: np.ndarray; key: np.ndarray; meta: dict
def read_size_table(sizes_dir: str, cutoff: float) -> SizeTable          # refuses a cutoff mismatch
def derive_buckets(tables: list[SizeTable], n_buckets: int, graphs_per_pack: int, max_graphs: int) -> list[Bucket]
@dataclass
class HostPlan: blocks: list; packs: list[list[np.ndarray]]; step_buckets: np.ndarray; signature: int; dropped: int
def plan_data_epoch(tables, spec, buckets, data_epoch, host, n_hosts, n_local, counts_allgather=None) -> HostPlan
class StreamLoader:
    def __init__(self, spec, tables, buckets, mesh, n_devices, atom_indices, atom_energies, cutoff, dtype,
                 workers: int, prefetch: int = 3, shuffle: bool = True, seed: int = 0): ...
    def iter_steps(self, n_steps: int) -> Iterator[tuple[jraph.GraphsTuple, MultiDeviceBatchInfo]]: ...
    cursor: tuple[int, int]          # (data_epoch, step); settable for resume
    steps_per_data_epoch: int
    def close(self) -> None: ...
class StreamEvalSet:                 # shuffle=False, every structure exactly once
    def iter_groups(self) -> Iterator[tuple[jraph.GraphsTuple, np.ndarray]]   # batch, per-device weight mask
def build_stream_datasets(cfg: dict, fout, mesh, n_devices) -> tuple   # same tuple as build_datasets + loaders
```

**Evaluation.**

* Every structure exactly once. Hosts with fewer packs feed a duplicate with weight 0, the same mechanism as
  `multi_device_iterator` and `device_weights` (`data_nnx.py:443-452`; `training_utils.py:882-894, 996-999`).
* Generalise `device_weights(n_valid)` to an explicit per-device mask, because padding devices can sit on any host.
* `evaluate_sharded` assumes `dataset.graphs` (`training_utils.py:969-985`). Add an `if hasattr(ds, "iter_groups")`
  branch; the accumulation in `:996-1032` is unchanged.

**Reuse from BAM-GAMP-jax.**

* *Import:* nothing on the training path (it is not installed; it is used through PYTHONPATH), only in tests:
  * `bam_jax.data.schema.validate_super_batch` (`schema.py:157-197`); `bam_jax.data/__init__` imports only the schema
    (`BG/bam_jax/data/__init__.py:1-9`);
  * `bam_jax.data.preprocess.load_species_encoding` (`preprocess.py:226-268`);
  * the converter, as the reference for the copied functions.
* *Copy* (pure numpy, about 150 lines, each with a "keep in sync with converter v2" note and a test against the
  original):
  * `open_lmdb`, `decode_value`, `_arr`, `_scalar`, `decode_row`, `list_row_ids` (`convert_aselmdb_to_shards.py:174-251`);
  * `check_row` (`:298-344`);
  * `unpack_graphs` (`reshard.py:194-225`);
  * `_combine_seeds` and `_balanced_positions` (`loader.py:1039-1045, 656-668`);
  * the `prefetch` pattern (`:1003-1033`).
  * The converter script itself should not be imported: it sets thread environment variables and imports `reshard`
    and `bam_jax` at import time (`convert_aselmdb_to_shards.py:127-143`).
* *Not reused:* `plan_epoch` / `iter_planned` (their unit is a file of fixed super-batches) and
  `prepare_batch_for_step` (keyed by the schema dict).

### 4.9 `train_3bpa.py` integration

New config keys (document them in the docstring, `train_3bpa.py:1-139`):

```json
"data_format": "stream",
"stream": {
  "train": [{"name": "omat24", "type": "aselmdb", "root": "/scratch/t589a02/oam_lmdb/omat24_train",
             "sizes": "/scratch/t589a02/oam_sizes/omat24_train_c5", "repeat": 1}],
  "valid": [{"name": "omat24", "type": "aselmdb", "root": "/scratch/t589a02/oam_lmdb/omat24_val",
             "sizes": "/scratch/t589a02/oam_sizes/omat24_val_c5", "max_structures": 100000}],
  "test":  {"omat24_val_full": {"type": "aselmdb", "root": "...", "sizes": "..."}},
  "species_encoding": "/scratch/t589a02/oam_data/refs/omat24_train_ls.npz",
  "atom_energies_from": "/scratch/t589a02/oam_data/refs/omat24_train_ls.npz",
  "block_rows": 128, "n_buckets": 1, "buckets": null, "graphs_per_pack": 110, "pack_max_graphs": 512,
  "workers_per_host": 32, "prefetch": 4, "steps_per_epoch": 20000, "seed": 0
}
```

For MPtrj, use `{"type": "shards", "root": "/scratch/t589a02/mp_data/jax_data_v3_raw/train", …}`. A shard root as an
OMat24 source is optional.

Edits:

1. `build_datasets` (`:316-379`): branch to `build_stream_datasets` when `data_format == "stream"`.
   * Statistics come from `sizes_meta.json`: `avg_n_neighbors` from the full split, not the subset. The subset is
     biased: 30.2 vs 18.6 atoms per structure, because `oam_subset.draw` takes ≤ 50 per shard file
     (`oam_subset.py:55-74`; log `:257, 270`).
   * `shift`/`scale` come from the config (`use_dataset_stats` false in the OAM configs, `:397-404`).
   * `atom_energies` is mapped as in §3.5.
   * Assert that `atomic_numbers` is sorted.
2. Refuse the in-memory-only options in stream mode: `valid_frac` split (`:355-358`), `boltzmann_tilt`
   (`:757-775`), `node_weight` (`:781-815`), `taylor_aug` (`:822-848`).
   * `gauge_refit` evaluates `train_dataset` (`:1154-1167`); use a fixed seeded train subset `StreamEvalSet`.
3. Model and optimiser: the cosine schedule uses `stats['n_graphs'] // batch_size` (`:615-618`). Use
   `steps_per_epoch` instead.
4. Loader construction (`:924-980`): replace with `StreamLoader`. The fixed shapes come from the buckets, and no
   `sizes_<split>.tsv` is written (the size table is the record).
5. Epoch loop (`:1014-1059`):
   * `for batch, _info in train_loader.iter_steps(steps_per_epoch):` with an unchanged body;
   * `n_epochs` counts virtual epochs;
   * log the data epoch and the dropped-pack count.
6. Checkpoints (`:1099-1109, 1126-1131`): add `'data_cursor': loader.cursor`.
   * Resume (`:901-916`) restores it, instead of advancing `base_loader.rngs` (`:964-965`).
   * **Needed:** one OMat24 data epoch is more than 24 h on one node, so it must be resumable.
   * Save `<ckpt>.last.pkl` every N steps, not only every N epochs.
7. `_evaluate` (`:982-990`): pass a `StreamEvalSet`. With a validation set of 1M structures, evaluate a fixed
   `max_structures` subset per virtual epoch and the full sets only at the end.
8. The barrier around the dataset caches (`:741-747`) is not needed. Every host reads the same `sizes.npz`; check its
   sha256 with the allgathered plan signature.

### 4.10 Test plan

Tests T1-T3 and T7 are CPU and JAX-free and go in `tests/test_stream_data.py`. T4-T6 need GPUs and run as small
Slurm jobs.

* **T1 Source equivalence.** Use one non-re-split OMat24-val task.
  * Its sidecar `_progress/<tag>.graphs.npz` lists `(src, row_id)` in output order (`convert_aselmdb_to_shards.py:484-501`).
  * Read the first 50 graphs of `shard_<tag>_000.pkl` with `ShardSource`, and the same `(file, row)` with
    `AselmdbSource`.
  * Assert equal Z and cell, positions/forces/stress to float32 tolerance, and raw energy within one float32 ulp.
  * Also check that the copied `decode_row`/`check_row` agree with the converter's functions on 200 rows (import the
    script by path).
* **T2 Graph equivalence with the xyz path.**
  * Take the 1,000 structures of `/scratch/t589a02/datasets_ext/oam_subsets/omat24_valid.xyz`, which `oam_subset.py`
    made from the shards.
  * Compare each `stream_io.graph_arrays` record against `data_3bpa.preprocess_graph` on the same `ase.Atoms`.
  * Assert equal edge multisets (sorted by (receiver, sender, Sij)), |r| < r_c for all edges, equal energy (E − ΣE0),
    stress, cell, volume and cutoff.
  * Extend the existing `tests/test_periodic.py:53` (minimum-image check) to stream graphs.
* **T3 Plan properties.** Use a synthetic table with 4 hosts × 4 devices simulated in one process.
  * Every structure appears exactly once per data epoch, minus the reported leftovers.
  * No pack exceeds its bucket (reuse `tests/test_oam_stress.py:99`, `test_fixed_shape_packing_covers_every_graph_once`).
  * The plan is identical across two runs, and the signature changes with seed, epoch, n_hosts or the table.
  * Resume at step k gives the same packs as steps k..end.
* **T4 Model equivalence (GPU, float64, small).** Evaluate a trained checkpoint
  (`runs/oam64_omat24_v2_s42_cue/ckpt_*.pkl`) on `omat24_valid.xyz` in two ways:
  * the in-memory path (`load_3bpa_graphs` + `evaluate_sharded`);
  * a `StreamEvalSet` over an extxyz source with the same structures, and over the lmdb rows of the same structures.
  * Per-structure E/F/S and the summary metrics must agree to 1e-9 relative between the xyz paths, and to
    float32-input tolerance against lmdb.
  * Padding edges with r = 0 are zeroed (`race_nnx.py:1741-1757`), so the pack composition must not matter.
* **T5 Data-parallel equivalence.** Compare 2 processes × 1 GPU against 1 process × 2 GPUs: 20 float64 steps from the
  same plan must give the same parameters. Extend `tests/test_data_parallel.py:68-98` with the
  `make_array_from_process_local_data` stacking and the generalised device mask.
* **T6 Throughput smoke (one GPU node, about 20 min).**
  * Loader only: structures/s against W = 2, 4, 8, 16.
  * Then 500 training steps, logging the fraction of steps that waited on the queue.
  * Size pass on one subset directory (the policy's "small test job first").
* **T7 Resume.** Run 300 steps uninterrupted, and 150 + restart + 150. The float64 parameters must be bit-identical
  (or within 1e-12).

### 4.11 Risks

1. **Neighbour-count drift** between the size pass and training (ASE version, positions dtype, wrapping) would
   overflow a fixed shape. Use the same function and dtype in both, assert per structure, and store the ASE version
   in `sizes_meta.json`. Shard sources use float32 positions, so boundary pairs can differ against lmdb by
   |r − r_c| < 1e-4 Å. The envelope makes their contribution about 1e-17.
2. **Lustre load.** Random 4 KB reads and many lmdb opens. Mitigations: lexicographic cursor blocks (K ≥ 64), file
   affinity per host and worker per epoch, an LRU of open envs, and file lists from `convert_config.json` or a
   sorted listing of the user's own directory only. Never scan beyond it (site policy).
3. **Main-process Python overhead** (GraphsTuple construction, `jraph.batch_np`, device put) at float32 step rates.
   Measure it in T6; the fallback is collating in the workers.
4. **Wall-time limit** (24 h) against one data epoch: virtual epochs and cursor checkpoints are required, not
   optional.
5. **Multi-bucket reorder buffer** could deadlock or blow up memory if the schedule runs ahead of the stream. The plan
   computes the peak; the default is one bucket.
6. **Determinism across hosts.** Same numpy version and same `sizes.npz`: sha256 and plan signature are allgathered
   at each data-epoch boundary; mismatch raises an error.
7. **Data selection.** The aselmdb path must reproduce the converter's filters. Reusing the sidecar kept lists does
   this exactly. The energy-floor table lives in the BAM-GAMP repo (`convert_config.json` filters), so record its
   sha256.
8. **MPtrj has no raw source on disk** (`DATASETS.md:42`). MPtrj streaming depends on the shard reader with float32
   energies and positions (energy dev up to 1.2e-4 eV); keep it in mind for float64 fine-tuning.
9. **Statistics.** `avg_n_neighbors` (26.965 in the OAM configs) came from a biased subset. Recompute it from the
   full table for pretraining, and keep the pretraining value when fine-tuning.
10. **Species and E0 ordering** (§3.5): assert sorted `atomic_numbers`, never use `Z_to_idx`, and map `atom_ref`
    through `species_encoding`.
11. **Worker memory.** Workers that import `bam_omat24.*` pull in JAX and the model (`bam_omat24/__init__.py:9-50`).
    The `stream_io` top-level module avoids this, and forkserver avoids forking a CUDA process.
12. **Evaluation exactness.** Do not copy BAM-GAMP's min-truncated validation (`trainer.py:1741-1755`); use
    weight-0 padding devices.

### 4.12 Suggested order of work

1. `stream_io.graph_arrays` refactor, plus T2.
2. Sources, plus T1.
3. Size pass on omat24_val and salex_val (small Slurm test job, user approval).
4. Planner, plus T3.
5. Loader and evaluation set, plus T4 and T5 on 1-2 GPUs.
6. Cursor checkpoints, plus T7.
7. Throughput smoke T6.
8. Full size pass on omat24_train.
9. First OMat24 run.
