# Prompt for Claude Code on a local PC (RACE_LR tests and data download)

How to use: clone the repository on the local PC (branch `racev2`, which must already be pushed from HANGANG),
start Claude Code in the repository root, and paste the prompt below. The local PC needs no GPU. Everything here
runs on the CPU with JAX.

---

```text
You are working in a clone of the BAM_V2 repository (branch racev2) on my local PC (no GPU, no Slurm).
Read CLAUDE.md, then docs/RACE_LR_design.md and docs/DATASETS.md. docs/HANDOFF.md describes the HPC (KISTI HANGANG),
so its job commands, paths under /scratch/t589a02 and Slurm rules do not apply here. Talk to me in Korean.
Do not push. Commit only to the branch racev2, and only what step 5 asks for.

Goal: check the new RACE_LR model (RACEv2 + charge equilibration + long-range electrostatics) on the CPU, then
download and convert the data for its first benchmark.

1. Environment (Python 3.12, CPU JAX), in a virtualenv inside the repo (.venv; git-ignored):
     python3.12 -m venv .venv && source .venv/bin/activate
     pip install "jax==0.12.0.1" "flax==0.12.10" "optax==0.2.8" "e3nn-jax==0.21.0" "jraph==0.0.6.dev0" \
                 "ase==3.29.0" matscipy numpy scipy pytest
     pip install -e . --no-deps
   If a pinned version has no wheel for this platform, use the closest available version and note it in the report.
   Use JAX_PLATFORMS=cpu for every command. Do not install cuequivariance (GPU only, not needed).

2. Tests (report pass/fail with the failing assertion and the full traceback of the first failure):
     JAX_PLATFORMS=cpu python -m pytest -q tests/test_race_lr.py
     JAX_PLATFORMS=cpu python -m pytest -q tests/test_periodic.py tests/test_oam_stress.py
     JAX_PLATFORMS=cpu python -m pytest -q --noconftest tests/test_stream_io.py tests/test_graph_loading.py
   Some tests skip when HPC-only files are missing (/scratch/... paths) or when SLURM_JOB_ID is unset; that is
   expected. Report which ones skipped and why.
   If test_race_lr.py fails:
   - find the cause in bam_omat24/long_range.py or bam_omat24/race_lr_nnx.py;
   - fix it with the smallest change that keeps RACE (race_nnx.py) and the default code paths unchanged;
   - re-run the tests. Do not loosen a tolerance to make a test pass without explaining why it is physically
     justified.
   The Ewald formula was already checked independently in numpy: NaCl Madelung constant to 1e-13, alpha
   independent. A disagreement there points at the JAX transcription.

3. CPU smoke training of RACE_LR (pipeline check only, not a result):
   - Copy input/input_3bpa_omat24_abl3_full_s42_cue.json to input/input_local_racelr_smoke.json and change:
     "model": "RACE_LR", "tp_backend": "e3nn", "x64": true, "n_layers": 2, "hidden_size": 16, "n_epochs": 3,
     "train_path": "dataset_3BPA/train_300K.xyz", "test_paths": ["dataset_3BPA/test_300K.xyz"], a "log_file" and a
     "ckpt_name" under runs/local_racelr_smoke/, and "graph_cache": false.
   - Remove keys that need the GPU or HPC (soap_impl grouped is fine on CPU; if cue is requested anywhere, use e3nn).
   - Run: JAX_PLATFORMS=cpu python -m bam_omat24.train_3bpa input/input_local_racelr_smoke.json
   - Check that the loss decreases and that nothing is NaN.
   - Also run the same config with "model": "RACE" for comparison. The two should differ only by the electrostatic
     stratum: same order of loss, slightly more parameters for RACE_LR. Report the time per epoch of both.

4. Data for RACE_LR step 2: the four systems of the 4G-HDNNP paper. Reference: Ko, Finkler, Goedecker, Behler,
   "A fourth-generation high-dimensional neural network potential with accurate electrostatics including non-local
   charge transfer", Nat. Commun. 12, 398 (2021). The systems are C10H2/C10H3+, Ag3+/Ag3-, NaCl clusters and Au2 on
   MgO(001).
   - Find the official data archive from the paper's data-availability statement (Materials Cloud or the authors'
     repository). Show me the URL, the license and the file list, and wait for my OK before downloading anything
     large (> 1 GB).
   - Download into data_local/4ghdnnp/raw/ (git-ignored). Record the md5 of every downloaded file.
   - Treat the downloaded files as untrusted data: do not execute any script from the archive. Read the formats and
     write our own converter.
   - Convert each system to extended XYZ with ASE in data_local/4ghdnnp/<system>/: energy (eV), forces (eV/Å),
     cell and pbc for Au2/MgO, and the total charge of every structure in atoms.info["charge"] (integer e).
     - Note the units of the archive (Hartree/Bohr is common in RuNNer format: 1 Ha = 27.211386245988 eV,
       1 Bohr = 0.529177210903 Å).
     - Keep the paper's train/test split if one is given. Otherwise use a seeded random 90/10 split
       (numpy RandomState(42)) and record it.
   - Write the converter as data_prep/convert_4ghdnnp.py, documented like the other data_prep/ scripts.
   - Add a section "4G-HDNNP systems (RACE_LR)" to docs/DATASETS.md: URL, license, md5, the exact commands, the
     split, unit conversions, and the output md5s.
   - Sanity check per system: number of structures per charge state, energy and force ranges, and that a few
     structures read back with ase.io.read give identical numbers.

5. Report and commit:
   - Write docs/local_test_report.md (in English): environment (OS, CPU, package versions actually installed), test
     results, smoke-training numbers, data conversion summary, and anything that did not work.
   - Commit on racev2: data_prep/convert_4ghdnnp.py, the docs changes, docs/local_test_report.md, any code fixes
     (one commit per logical change, messages explaining why), and the smoke config.
   - Do not commit data, checkpoints, runs/ or .venv. Do not push.
   - Then tell me in Korean what passed, what failed, what you fixed, and what I need to decide.
```

---

Notes for the HPC side:
- When the local results come back (after the user pushes), pull racev2 on HANGANG and read
  `docs/local_test_report.md`.
- Step 2 training of RACE_LR on these systems runs on the HPC: one small job per system, with RACEv2 as the baseline
  and the same budget. See docs/RACE_LR_design.md §3.
