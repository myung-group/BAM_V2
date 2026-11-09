# Local CPU test of RACE_LR and 4G-HDNNP data (2026-11-06)

This report covers the run of `LOCAL_TEST_PROMPT.md` on the user's local PC. It used the CPU only, with no GPU and no
Slurm, on branch `racev2` starting from 692aa83.

## Summary

- **Tests.** All RACE_LR unit tests pass after one test fix. The fix concerns the precision of the test's rotation
  matrix, and the model was not changed. The periodic, stress, stream and graph-loading tests pass.
- **Smoke training.** RACE_LR and RACE both train for 3 epochs on 3BPA on the CPU. Loss decreases, nothing is NaN, and
  the two runs are close, with RACE_LR slightly ahead.
- **Data.** The four 4G-HDNNP systems were downloaded from Materials Cloud (CC BY 4.0, 40 MB) and converted to extxyz
  with total charges. The details are in `docs/DATASETS.md` §16.
- **Follow-up.** RACE_LR now returns its charges. The E0 offsets, the split and the Ewald settings for Au2/MgO
  are decided and checked (section 5).

## 1. Environment

| item | value |
|---|---|
| OS | Ubuntu 22.04.5 LTS, Linux 6.8.0-138-generic x86_64 |
| CPU | Intel Core i7-13700K, 24 logical CPUs, 62 GiB RAM |
| Python | 3.12.15 (the base interpreter is the user's conda env `bam_v2`; there is no system python3.12) |
| venv | `.venv` in the repo, git-ignored |
| JAX | jax 0.12.0.1, jaxlib 0.12.0.1 (CPU), `JAX_PLATFORMS=cpu` for every command |
| others | flax 0.12.10, optax 0.2.8, e3nn-jax 0.21.0, jraph 0.0.6.dev0, ase 3.29.0, matscipy 1.3.1, numpy 2.5.4, scipy 1.18.1, pytest 9.2.0, tqdm 4.70.1, natsort 8.4.0 |
| repo | `pip install -e . --no-deps` equivalent (`uv pip install --no-deps -e .`) |

Every pinned version had a wheel, so all of them were installed as given. There are three deviations from the
prompt:
- **Installer.** The global pip config of this PC adds the extra index `pypi.ngc.nvidia.com`. That host did not resolve
  in DNS, so pip spent its time retrying. The packages were therefore installed with
  `uv pip install --python .venv/bin/python --index-url https://pypi.org/simple ...` with the same version pins.
- **Missing packages.** `tqdm` and `natsort` are declared dependencies in `pyproject.toml` but are missing from the
  prompt's list. Without `tqdm`, `tests/conftest.py` fails at import, so both were added.
- **cuequivariance.** It was not installed.

## 2. Tests

Commands, as in the prompt:

```bash
JAX_PLATFORMS=cpu python -m pytest -q tests/test_race_lr.py
JAX_PLATFORMS=cpu python -m pytest -q tests/test_periodic.py tests/test_oam_stress.py
JAX_PLATFORMS=cpu python -m pytest -q --noconftest tests/test_stream_io.py tests/test_graph_loading.py
```

| file(s) | result |
|---|---|
| test_race_lr.py, first run | 5 passed, **1 failed** (`test_race_lr_forces_rotation_and_charge`), 23 s |
| test_race_lr.py, after the test fix | **6 passed**, 23 s |
| test_periodic.py + test_oam_stress.py | 9 passed, 65 s |
| test_stream_io.py + test_graph_loading.py | 7 passed, 2 skipped, 4 s |

**Skipped.** `test_stream_io.py:99` and `:126` skip with "imports JAX: run inside a Slurm job" because `SLURM_JOB_ID` is
unset. This is expected. Run with a dummy `SLURM_JOB_ID=local`, they also pass (6 passed). No test skipped because
of missing `/scratch` files.

**The failure:**

```
tests/test_race_lr.py:100: AssertionError
>       assert abs(float(rot['energy'][0]) - float(r0['energy'][0])) < 1e-9
E       assert 1.7324950363217795e-09 < 1e-09
E        +  where 1.7324950363217795e-09 = abs((-13.79222661236037 - -13.792226614092865))
```

**Cause.** The cause is in the test, not in `long_range.py` or `race_lr_nnx.py`.
- `conftest.random_rotation` returns a **float32** matrix. Casting it to float64 does not make it orthogonal again:
  its error |RᵀR − I| is 4.8e-8.
- The "rotated" geometry is therefore slightly strained, which moves the energy by about 1e-9 eV.
- Measured with the same graph and model in float64:

| model | ΔE with float32-rounded R | ΔE with an exact float64 R |
|---|---|---|
| RACE_LR | 1.73e-9 eV | 0.0 |
| RACE (`conftest.make_model`) | 1.98e-9 eV | 1.8e-15 eV |

**Fix.** In `tests/test_race_lr.py`, the float32 matrix is snapped to the nearest exact rotation (its polar factor
`U Vᵀ` from the SVD) before the check. The tolerance stays at 1e-9. `conftest.py`, the models and the default code
paths are unchanged. The Ewald, Madelung and α-independence tests passed on the first run.

## 3. CPU smoke training (pipeline check only, not a result)

**Configs.**
- `input/input_local_racelr_smoke.json` is `input_3bpa_omat24_abl3_full_s42_cue.json` with the changes the prompt
  lists: RACE_LR, e3nn, x64, 2 layers, hidden 16, 3 epochs, test_300K only, `graph_cache: false`, and logs under
  `runs/local_racelr_smoke/`.
- `input/input_local_race_smoke.json` is the same config with `"model": "RACE"`.
- Apart from `tp_backend`, no key needed the GPU or the HPC. `soap_impl: grouped` runs on the CPU.
- The command was `JAX_PLATFORMS=cpu python -m bam_omat24.train_3bpa input/input_local_<tag>_smoke.json`.

| | RACE_LR | RACE |
|---|---|---|
| parameters | 81,206 | 80,584 (+622 for RACE_LR: the χ/J head, χ0, J0, σ) |
| epoch time, epochs 1 / 2 / 3 (s) | 26.1 / 13.4 / 13.6 | 26.2 / 13.8 / 13.8 |
| wall time, whole run | 2:15 | 2:15 |
| peak RSS | 2.08 GB | 2.01 GB |
| train loss, epochs 1 / 2 / 3 | 7.349 / 0.365 / 0.110 | 7.340 / 0.394 / 0.125 |
| valid loss, epochs 1 / 2 / 3 | 12.47 / 3.732 / 0.932 | 12.47 / 3.820 / 0.989 |
| valid F RMSE, epoch 3 (meV/Å) | 412.9 | 425.2 |
| test_300K F MAE / RMSE (meV/Å) | 311.8 / 414.0 | 324.9 / 426.7 |
| test_300K E MAE after gauge refit (meV) | 73.3 | 76.9 |
| NaN | none | none |

Epoch 1 includes the JIT compilation.
- **Loss.** It decreases monotonically in both runs, and the two runs are of the same order, as expected.
- **Why they match.** 3BPA is neutral, and the Qeq stratum starts with χ = 0 and a zero-initialised head.
- **The differences.** These small gaps come from the extra stratum and from parameter initialisation. They are not
  evidence about the model at 3 epochs.
- **Speed.** The dense 27-atom Qeq solve costs nothing measurable on the CPU at this size.

## 4. 4G-HDNNP data (RACE_LR step 2)

**Source and conversion.** The full record is in `docs/DATASETS.md` §16.
- Data: Materials Cloud Archive 2020.137, DOI 10.24435/materialscloud:f3-yh, at
  https://archive.materialscloud.org/records/bk6kw-4b895, under CC BY 4.0.
- Files: `readme.txt`, `files_description.md`, and `datasets.tar.gz` (39.7 MB, md5 0aa11ea305b3af6a47bb3c577c2729bc,
  equal to the record's checksum).
- The archive is under the 1 GB limit, so no approval was needed before downloading.
- The archive contains only four RuNNer `input.data` files. Nothing was executed from it.
- Our converter `data_prep/convert_4ghdnnp.py` converts Bohr / Hartree to Å / eV with CODATA 2018 factors.
- It writes `all/train/test.xyz`, `split.json` and `e0.json` per system into `data_local/4ghdnnp/<system>/`.

| system | structures (charge states) | pbc |
|---|---|---|
| carbon_chain | C10H2 (0) 5008, C10H3+ (+1) 5011 | no |
| ag_cluster | Ag3+ (+1) 5014, Ag3− (−1) 5999 | no |
| nacl | Na8Cl8+ 2500, Na9Cl8+ 2500 (both +1) | no |
| aumgo | Au2/MgO undoped 2500, Al-doped 2500 (both 0), 110 atoms | yes (9.05 × 9.05 × 26.46 Å) |

**Split.** The paper used a random 90/10 split without published indices. This conversion uses
`RandomState(42).permutation`, taking the first 90 % per system for training, giving 9017/1002, 9912/1101,
4500/500 and 4500/500.

**Sanity checks.**
- Counts match the readme.
- The Hirshfeld charges sum to the total charge in every structure.
- `ase.io.read` read-back agrees with an independent re-parse of the raw files to 5e-9, the 8-decimal limit of the
  extxyz writer.
- A RACE_LR forward pass on one structure of each charge state was checked (untrained, `total_charge_key: "charge"`).
  It gave finite energies, forces and stress and conserved charge exactly, including the periodic Au2/MgO cell.
- Energy and force ranges per charge state are in `DATASETS.md` §16.

## 5. Follow-up: decisions taken (user: "proceed as you recommend", 2026-11-06)

1. **Model charges are now returned.**
   - The design doc said the model returns `charges`, but the shared `bam_omat24/utils.energy_forces_stress` copied
     only fixed keys into its aux, so `model(graph)` had no `charges` key.
   - It now passes `charges` through only when the node-energy function provides them. Every other model returns
     exactly the keys it returned before.
   - New checks in `tests/test_race_lr.py`: the returned charges sum to the total charge (0 and +1) with padding at 0,
     and plain RACE has no `charges` key.
   - Results: `test_race_lr.py` 7 passed. `test_periodic.py`, `test_oam_stress.py` and `test_equivariance.py` gave
     28 passed and 2 xfailed, the known issue #5. A RACE_LR smoke run with the patch gives the same losses as before
     (see below).
2. **E0 offsets.** Use the train-only least-squares offsets of `e0.json` as `atom_energies`, for RACE_LR and the
   RACEv2 baseline alike.
   - They are a constant per composition, so energy differences within a composition, including the charge-state
     gaps, are untouched.
   - No extra data is needed, which keeps the fair protocol. Isolated-atom DFT would need FHI-aims runs and would
     not change what the models learn.
3. **Split.** Keep the seeded RandomState(42) 90/10 split of `split.json`. The paper's indices are not published, and
   the per-state counts are within about ±1 % of 90/10.
4. **Ewald on Au2/MgO.** This was measured on the CPU with the Hirshfeld charges, on one undoped and one Al-doped
   structure.
   - The defaults (α 0.3, n_real 2, n_recip 6) differ from α 0.3 / n_real 4 / n_recip 24 by 0.008 meV or less.
   - With α from 0.2 to 0.4, the energies agree to below 1e-4 meV. **Keep the defaults.**
   - **Correction (2026-11-06, later):** this holds for the smooth Hirshfeld charges only.
     - With random charges (std 0.5 e, `runs/diag/check_ewald.py`), the defaults err by up to 1.9 meV on this cell.
       The reciprocal sum along the 26.5 Å axis is too short.
     - α 0.3 / n_real 1 / n_recip 8 errs by at most 0.06 meV and costs less. The Au2/MgO configs use it from now on.
     - Fitted charges need not be smooth, so check with `check_ewald.py`.
   - The slab dipole along z is large for the Al-doped slab, and the vacuum gap is about 10 Å:

     | subset | M_z (eÅ) | Yeh–Berkowitz term 2πK M_z²/V (meV) |
     |---|---|---|
     | Al-doped | +1.25 … +3.74 | 65 … 584 (std 92) |
     | undoped | −1.09 … +1.72 | 0 … 124 (std 14) |

   - The paper names no dipole correction and FHI-aims applies none by default, so the DFT reference is plain 3D
     periodic. **Keep plain 3D Ewald without a dipole correction**, which matches the reference. This should be
     revisited only if the FHI-aims control files show `use_dipole_correction`.

## 6. Environment notes (not blocking)

- Python 3.12 came from conda, because the system Python is 3.10.
- pip's NVIDIA extra index fails DNS on this PC, so uv was used.
- `tqdm` and `natsort` must be added to the install line of `docs/LOCAL_TEST_PROMPT.md`.

## 7. Local GPU test on the 4G-HDNNP systems (reduced budget, 2026-11-06 evening)

The user asked whether step 2 can run on this PC and said that a data subset is fine for a local test. These
numbers are a **small-budget check, not the step-2 result**. The HPC runs (full training split, full recipe) remain
the result.

**Setup.**
- GPU: NVIDIA RTX 4060 Ti 16 GB (driver 535, CUDA 12.2), shared with the desktop.
- Environment: the user's conda env `bam_cue` (jax 0.12.0.1 with jax-cuda12, cuequivariance 0.13.0), run unchanged.
  The repo's `bam_omat24` was the one imported, which was checked. That env also has an editable install of
  another clone.
- **Limited GPU use** (`runs/diag/local_4g/run_limited.sh`), because full GPU use had lagged the desktop:
  - CUDA MPS from a user-level daemon with its own pipe directory, at most 50 % of the SMs
    (`GPU_SM_PCT`, set it to 30 if the desktop still lags);
  - XLA with no preallocation, at most 40 % of the GPU memory;
  - `nice 10` for the CPU side.
- Model: the r37_uffw03 recipe (cue, float64, SOAP, UFF repulsion) with a **reduced size**: hidden 64, 3 layers,
  lmax 2.
  - With the full size (5.9 M parameters), an epoch of 1000 carbon-chain structures took 183 s. The reduced model
    took 11 s.
  - Float64 on a GeForce card is slow, and float64 was kept on purpose.
  - RACE_LR and RACE use exactly the same settings. `avg_n_neighbors` is the value of each training subset.
- Data (`data_prep/subset_4ghdnnp.py`):
  - The training set is 1000 structures, 500 of each charge state or structure family. For Au2/MgO it is 100
    structures, 50 of each, because one Au2/MgO structure costs about 76 carbon-chain structures.
  - The test set is the full test split.
  - Configs are in `input/local_4g/`. The runs used 200 epochs, and 20 for Au2/MgO, run by
    `runs/diag/local_4g/queue.sh`.
- Wall time: Ag3 25 min, carbon chain 37 min and NaCl 39 min per model, and Au2/MgO 6 min. No NaN occurred and
  every run exited 0.

**Evaluation.**
- Per-frame errors come from `bam_omat24.dump_frame_errors`.
- Fix: it, and `eval_sets`, now pass the config's `total_charge_key`. Before, they built every graph as neutral, so
  RACE_LR was evaluated with the wrong total charge. The trainer's own test evaluation was already correct.
  The per-frame numbers match the trainer's numbers.
- The raw energy errors of the carbon chain and NaCl were dominated by a constant offset per composition, in the
  training set as well. The trainer's `gauge_refit` is a single per-atom shift, which cannot remove an offset
  between two compositions.
- So the table also gives a **per-composition gauge refit**: a least-squares fit of the training residuals on the
  element counts, applied to the test set (`runs/diag/local_4g/by_state.py`).
  - It uses no test labels.
  - For Ag3 (one element) it is a single shift and cannot hide a charge-state error.

Test split, energy RMSE in meV (raw → after the per-composition refit), force RMSE in meV/Å:

| system | state | RACE_LR E | RACE E | RACE_LR F | RACE F |
|---|---|---|---|---|---|
| Ag3 | Ag3+ | 2.0 → 2.2 | 116 → 299 | 2.9 | 164 |
| Ag3 | Ag3− | 39 → 39 | 1601 → 1567 | 18.7 | 5651 |
| Ag3 | all | **28.9 → 28.8** | **1178 → 1169** | **13.9** | **4152** |
| carbon chain | C10H2 | 41.5 → 3.2 | 273 → 4.6 | 17.5 | 25.4 |
| carbon chain | C10H3+ | 2.1 → 2.1 | 121 → 2.5 | 12.2 | 13.1 |
| carbon chain | all | 29.3 → **2.7** | 211 → **3.7** | **15.1** | **20.2** |
| NaCl | Na8Cl8+ | 72 → 30 | 144 → 29 | 47.0 | 44.8 |
| NaCl | Na9Cl8+ | 220 → 23 | 28 → 25 | 37.0 | 41.7 |
| NaCl | all | 163 → 26.8 | 104 → 27.1 | 42.4 | 43.3 |
| Au2/MgO | all (100 train, 20 ep) | 1032 → 284 | 268 → 269 | 140.7 | 141.5 |

**Reading.**
1. **Ag3: a clear, structural gain.** RACE cannot tell Ag3+ from Ag3− where their geometries overlap.
   - Its Ag3− errors have a long tail: the median |ΔE| is 31 meV, but the 99th percentile is 7.1 eV. Single frames
     reach a force RMSE of 76 eV/Å.
   - Its training was unstable. The best checkpoint is epoch 45, and the validation force RMSE was 14.9 eV/Å at
     epoch 200.
   - RACE_LR is stable and about 40 times better in energy and 300 times better in force.
2. **Carbon chain: a modest gain.**
   - The large raw energy difference (29 vs 211 meV) is a composition offset and not a physics effect: a local
     model can represent the offset between C10H2 and C10H3+.
   - After the refit, RACE_LR is better by about 25 % in energy and in force. The gain is largest for the neutral
     C10H2 force (17.5 vs 25.4 meV/Å).
   - **An earlier chat message claimed a 7–10× energy gain here. That was this offset artifact.**
3. **NaCl: no difference at this budget** (26.8 vs 27.1 meV, 42.4 vs 43.3 meV/Å).
   - The paper shows the NaCl effect along the path of one Na atom. The random test split may not resolve it, and
     both models are far from converged.
   - On the HPC, also evaluate the path or scan if it can be built from the data. The archive marks no path.
4. **Au2/MgO: invalid run (corrected 2026-11-06, later).**
   - The configs lacked `"periodic": true`, so the trainer built a molecular model for both RACE_LR and RACE. The
     image shifts and the cell were ignored, and RACE_LR used the direct Coulomb sum instead of Ewald.
   - The trainer now refuses periodic data with `periodic` false.
   - The Ewald path also failed under jit and gave NaN in float32. Both are fixed (§9).
   - The runs are repeated with `periodic: true` and the checked Ewald settings. The old outputs are kept as
     `runs/local_4g/aumgo_*_nonperiodic_round1`.

**Consequences for the HPC step 2.**
- Report the energy errors with a per-composition gauge refit (train only), or train long enough that the
  composition offset converges. Otherwise multi-composition systems (carbon chain, NaCl, Au2/MgO) compare offsets
  rather than models.
  - The cleanest fix is a per-species version of `gauge_refit` in the trainer. It is not implemented, and it is a
    decision for the user.
- Use `dump_frame_errors` / `eval_sets` from this commit on, so that RACE_LR gets the total charge.
- Watch for RACE instability on Ag3. Its validation loss jumps after epoch 45, so the best-checkpoint selection
  matters.

## 8. Does RACE_LR capture the long-range physics? (charge diagnostics)

Energy and force errors alone do not show it, because knowing the total charge already fixes most of Ag3. The
latent Qeq charges of the trained RACE_LR checkpoints were therefore compared with the DFT Hirshfeld charges on the
test splits (`runs/diag/local_4g/charges.py`). The Hirshfeld charges are never used in training.

| system | state | corr(q, Hirshfeld) | mean abs(q), model / DFT |
|---|---|---|---|
| Ag3 | Ag3+ | −0.05 (both nearly uniform at 1/3) | 0.333 / 0.333 |
| Ag3 | Ag3− | +0.91 | 0.333 / 0.333 |
| carbon chain | C10H2 | +0.85 | 0.004 / 0.035 |
| carbon chain | C10H3+ | −0.63 | 0.077 / 0.077 |
| NaCl | Na8Cl8+ / Na9Cl8+ | +0.99 / +0.96 | 0.06 / 0.36–0.41 |
| Au2/MgO | both (20 epochs, non-periodic run, invalid, see §7) | undefined: q = 0 everywhere | 0.000 / 0.35 |

Carbon chain: these are mean charges of C1…C10 along the chain, with C10 the protonated end in C10H3+. The change
from C10H2 to C10H3+ is shown:

```
model dq  +0.115 +0.103 +0.096 +0.085 +0.090 +0.080 +0.093 +0.103 +0.091 +0.089
DFT   dq  +0.132 +0.053 +0.097 +0.051 +0.096 +0.049 +0.101 +0.038 +0.141 +0.061
```

**Reading.**
- **Captured: non-local charge transfer.** The far-end carbon is 11.4 Å from the protonated carbon, and the cutoff
  is 5 Å. Its charge changes by +0.115 e in the model and +0.132 e in DFT. A purely local model cannot see the
  protonation from there.
- **Not captured: the atomic pattern.** The model distributes the extra charge almost uniformly and misses the
  alternation of the DFT charges. For C10H3+ this gives the negative correlation.
- **Not captured: the magnitude in ionic systems.** On NaCl the signs are right, but the charges are about six times
  too small, so the short-range strata still carry most of the ionic energy.
- **Au2/MgO.** This was the invalid non-periodic run (§7). χ and its head were still at zero, so it says nothing.
- **Likely causes.** Latent charges are not unique: the short-range strata can absorb electrostatic energy. In
  addition, χ0 and the χ head start at zero, and the hardness starts large (J0 = 5 eV/e², plus a Gaussian self term
  K/(√π σ) of 5–11 eV/e² for σ = covalent radius), which favours small, uniform charges.

**How to test it further, in order of cost:**
1. Run the charge diagnostics above on every RACE_LR run.
2. Pull apart charged fragments beyond the cutoff, such as the NaCl cluster or the Ag3 ions. E(r) should follow
   K q1 q2 / r, while RACE is flat beyond 5 Å. Also watch for fractional charges at dissociation, a known Qeq
   artefact.
3. Compare RACE and RACE_LR at reduced cutoff (3–4 Å). A model that captures long range should lose less.
4. Check the Au2/MgO doping effect from the paper's Table II: the wetting / non-wetting energy order flips with the
   dopant. This needs the full HPC training.
5. Model options for the user to decide:
   - initialise χ0 with element electronegativities and use a smaller J0;
   - optionally add a Hirshfeld charge loss, as 4G-HDNNP does in its first training step. The labels are part of
     the same dataset, but this changes the "latent charges" claim.

## 9. Periodic support and OMol25 pilot tooling (2026-11-06, late)

**Bugs found and fixed in the periodic path.** Earlier tests ran eagerly and on large cells only, so none of these
was caught.
1. **jit.** `long_range.ewald_matrix_rows` built the k-vector list by boolean indexing of a jnp array, which fails
   under jit (`NonConcreteBooleanIndexError`), and every training step is jitted. The list is now a numpy constant.
2. **float32 NaN.** Periodic RACE_LR gave NaN energies in float32. Two changes fix it:
   - The reciprocal weight was sqrt(… exp(−k²/4α²)/k²). In float32 the exponential underflows to 0 at large k,
     and the derivative of sqrt at 0 is infinite. It is now exp(−k²/8α²)·sqrt(4πK/V)/|k|.
   - Padding graphs have a zero cell, whose inverse is infinite. Their rows are masked, but 0·∞ still reaches the
     gradients, so padding graphs now get a large finite cell.
   - Which of the two was decisive was not isolated.
3. **Silent molecular training.** Periodic data with `periodic` false trained a molecular model (see §7). The
   trainer now raises in that case. No existing config of the repo is affected (checked).

New tests (`tests/test_race_lr.py`, 15 passed):
- periodic stress against the strain derivative, forces, and invariance under a lattice translation;
- eager equal to jit;
- finite float32 gradients with several padding graphs and large k.

The periodic, OAM stress, round 10/14/15, equivariance, stream and graph-loading tests pass (66 + 2 xfailed, 9).
Float64 results are unchanged: the Madelung tests still pass at 1e-7.

**New model options (off by default).**
- `long_range.charge_spin_embedding`: MLP(Q, 2S+1 − 1), zero-initialised, added to the layer-0 scalars. The spin
  comes from the cfg `total_spin_key`, read from the extxyz info and defaulting to 1.
- `long_range.qeq: false` switches the Qeq stratum off, giving an embedding-only variant.
- `chi_init_missing: "mean"` covers elements without a Pauling value.

**Tools.**
- `runs/diag/check_ewald.py <periodic.xyz>`: the Ewald error of given settings, and the cheapest converged setting.
- `data_prep/convert_omol25.py`: aselmdb to extxyz subsets.
  - Modes: `inspect` shows the row layout, and `subset` samples the official train and val splits.
  - The total charge and spin are required. `--neutral-if-missing` is explicit and counted.
  - Partial charges, if present, go to `ref_charges`.
  - `--allow-periodic` keeps the cell, pbc and stress.
  - Tested on synthetic aselmdb files, molecular and periodic. The real OMol25 row layout is not seen yet.
- `runs/diag/make_omol_configs.py`: the pilot arms.
  - Models: race, race_cs, racelr, and racelr_q.
  - Optimizers: soap_f64 and amsgrad_f32.
  - Periodic subsets get the stress loss and `--ewald`.
- `runs/diag/make_oam_configs.py --model racelr --ewald ...`: RACE_LR on the OAM subsets.
- All arms trained one epoch on synthetic data (CPU), molecular and periodic, without NaN after the fixes above.

**Blocked on the user.**
- OMol25 is gated on Hugging Face: accept the terms with your account and create a read token. There is no token
  on this PC.
- This PC has 52 GB free (95 % used). Check the archive sizes before downloading.

## 10. Charge handling variants on the 4G-HDNNP systems (round 2, 2026-11-07)

**Setup.** Same as §7: reduced model, 1000 training structures (100 for Au2/MgO), 200 epochs (20 for Au2/MgO), one
seed. The Au2/MgO runs are now periodic, with Ewald α 0.3 / n_real 1 / n_recip 8.
- `racelr`: the original latent RACE_LR.
- `pauling`: χ0 from Pauling electronegativities.
- `pauling_q`: Pauling init plus the Hirshfeld charge loss, weight 0.1.

The numbers below are test RMSE after the train-only per-composition refit (`runs/local_4g/summary_round2.txt`). The
charge column gives the correlation of q with Hirshfeld per state, and in brackets the mean |q|, model / DFT.

| system | model | E (meV) | F (meV/Å) | charges |
|---|---|---|---|---|
| Ag3 | RACE | 1169 | 4152 | – |
| Ag3 | racelr | 28.8 | 13.9 | −0.05 ; +0.91 |
| Ag3 | pauling | 28.5 | 13.9 | −0.10 ; +0.92 |
| Ag3 | **pauling_q** | 28.8 | 13.8 | **+0.99 ; +1.00** |
| carbon chain | RACE | 3.7 | 20.2 | – |
| carbon chain | racelr | 2.7 | 15.1 | +0.85 (0.004/0.035) ; −0.63 |
| carbon chain | pauling | 4.6 | 16.8 | +0.99 ; +0.78 |
| carbon chain | **pauling_q** | **2.2** | **9.2** | **+0.99 ; +0.98** |
| NaCl | RACE | 27.1 | 43.3 | – |
| NaCl | racelr | 26.8 | 42.4 | +0.99 (0.062/0.361) ; +0.96 (0.059/0.407) |
| NaCl | pauling | 25.9 | 41.5 | +0.99 (0.062/0.361) ; +0.98 (0.059/0.407) |
| NaCl | **pauling_q** | **23.2** | **35.8** | **+1.00 (0.355/0.361) ; +1.00 (0.375/0.407)** |
| Au2/MgO (periodic) | RACE | 287 | 94.6 | – |
| Au2/MgO (periodic) | racelr | 279 | 94.8 | q = 0 everywhere |
| Au2/MgO (periodic) | pauling | 266 | 97.3 | +0.99 (0.58/0.35) |
| Au2/MgO (periodic) | pauling_q | 263 | 99.6 | +0.99 (0.37/0.35) |

**Reading.**
1. **The Hirshfeld charge loss is what makes the charges physical.**
   - With it, the correlation is at least 0.98 and the magnitudes match DFT in every system.
   - Without it, the latent charges stay too small on NaCl, at 0.06 e, whatever the χ0 initialisation. Both latent
     runs end at the same |q| although their χ0 differ (−3.1 / +3.3 vs −1.4 / +2.8 eV).
2. **Physical charges also improve energies and forces.**
   - Carbon chain: RACE 3.7 meV / 20.2 meV/Å, and pauling_q 2.2 / 9.2, which halves the force error.
   - NaCl: 14 % better in energy and 17 % in force than RACE. The latent RACE_LR showed no gain there.
   - Ag3: unchanged in energy, but the charges are now right.
3. **Au2/MgO, 100 structures and 20 epochs, is too small to conclude.**
   - The energies are within the noise of each other. Pauling-initialised RACE_LR is slightly better in energy on the
     Al-doped slab (294 and 278 vs 362 meV), but its force is slightly worse.
   - Training is about 1.5× slower per epoch with Ewald.
4. **Design flaw found: with the default zero χ initialisation, a neutral-only dataset can never develop charges.**
   - For Q = 0, the minimised Qeq energy is −½ χᵀBχ, a quadratic form in χ. Its gradient is exactly zero at
     χ = 0, so the χ head and χ0 receive no gradient.
   - This is the q = 0 of the periodic Au2/MgO racelr run, and of the round-1 run.
   - The carbon chain escaped only because the charged C10H3+ shares the parameters.
   - Any neutral dataset (OMat24, the OpenFF OMol part) therefore needs `chi_init: "pauling"`, or the charge loss.
     The OMol pilot and `make_oam_configs.py --model racelr` already use Pauling.
   - The default is left unchanged so that old configs reproduce. It is a decision for the user whether Pauling
     should become the default.

These are one-seed, reduced-budget numbers. The HPC runs at full size decide.

## 11. OMol25 pilot on the public OpenFF redistribution (2026-11-07)

**Setup.**
- Data: `data_local/omol25_openff/sub10k`, the OMol25 ANI-2x part as redistributed by OpenFF. All structures are
  neutral and closed-shell, with Löwdin charges.
- Split: 10,000 training and 1,000 validation structures from the official train rows; the test set is the 2,932
  official val rows.
- Model: hidden 64, 3 layers, lmax 2, 30 epochs, one seed, the same GPU limits as before.
- The energy refit is per composition, fitted on the validation split, so it uses train-split rows only.

| model | optimiser | E (meV/atom) | F RMSE (meV/Å) | time | q vs Löwdin r | mean abs(q), model / DFT |
|---|---|---|---|---|---|---|
| RACE | SOAP f64 | 86.8 | 833 | 59 min | – | – |
| RACE | AMSGrad f32 | 85.6 | 782 | 6 min | – | – |
| RACE_LR (Pauling) | SOAP f64 | 92.0 | 776 | 60 min | −0.15 | 0.011 / 0.253 |
| RACE_LR (Pauling) | AMSGrad f32 | **78.6** | **749** | 7 min | −0.16 | 0.153 / 0.253 |
| + Löwdin loss 0.1 | SOAP f64 | 94.0 | 808 | 60 min | +0.93 | 0.198 / 0.253 |
| + Löwdin loss 0.1 | AMSGrad f32 | 88.2 | 795 | 7 min | +0.89 | 0.195 / 0.253 |

**Reading.**
- **Optimiser.** AMSGrad in float32 is about 10× faster on this GPU (GeForce float64 is slow) and as accurate or
  slightly better. It never diverged.
- **Training length.** All runs are far from converged. The median per-structure force RMSE is about 160–175 meV/Å,
  while the RMSE is 750–830 meV/Å, so a few frames carry very large errors.
- **Latent charges are meaningless on neutral molecules.** They are anti-correlated with Löwdin (r −0.15).
- **The Löwdin loss fixes the charges but not the errors.** It gives r 0.89–0.93, but unlike on the 4G systems
  (§10) the energies and forces do not improve. Possible causes are undertraining, neutral-only data, and the Löwdin
  partitioning. With one seed the differences may be noise.
- **Consequences for the OMol pretraining plan.**
  - Prefer AMSGrad float32, pending the OAM seed check of the HANDOFF.
  - The value of a charge loss for pretraining is not shown on neutral data. Decide it on the official OMol25 with
    charged states.

## 12. Physics checks with the local checkpoints (2026-11-07)

Scripts and figures: `runs/diag/local_4g/{dissociation,md_nve,plot_*}.py`, `runs/diag/plot_omol_ko.py`. Figures are
in `runs/local_4g/plots/` (not in git; Korean single-message versions `ko_*.png`). One seed, reduced models.

**Dissociation tail.** One atom was pulled out of Na9Cl8+ to 30 Å with the 4G NaCl checkpoint (Pauling + Hirshfeld
loss).

| | Cl⁻ pulled, rest +2 | Na⁺ pulled, rest 0 |
|---|---|---|
| RACE | flat beyond the 5 Å cutoff | flat beyond the 5 Å cutoff |
| RACE_LR, total-charge constraint | weak tail, −0.13 e left on Cl | spurious repulsive 1/r tail, +0.53 e on Na |
| RACE_LR, fragment constraints | charge −1.000, tail → monopole −2K/R (ratio 1.11 at 40 Å) | charge +1.000, energy → 0 (polarisation well only) |

Fractional charges on separated fragments are the known Qeq artefact. The fragment constraints remove it, which the
protein–ligand use case needs.

**NVE MD, 2 ps from a 300 K start.**
- **C10H3+:** RACE and RACE_LR conserve the total energy within about 3 meV, and the RACE_LR charge sum is exact.
- **Na9Cl8+:** both models fall into a hole once the minimum distance drops below the training minimum (2.21 Å).
  RACE_LR fails at 0.64 ps and RACE at 1.52 ps.
  - The total energy is conserved up to that point.
  - This is extrapolation of models trained on 1000 structures of a narrow geometry set, not a Qeq-specific
    failure.
  - MD stability must still be checked for the pretrained model.

**OMol25 public part, 100 epochs, AMSGrad f32.** Test set, energy refit on the validation split, force = median
per-frame RMSE.

| model | 30 epochs E / F | 100 epochs E / F | q vs Löwdin r |
|---|---|---|---|
| RACE | 85.6 / 173 | **70.5** / 146 | – |
| RACE_LR (latent) | 78.6 / 176 | 80.0 / 149 | −0.16 |
| RACE_LR + Löwdin loss | 88.2 / 175 | 78.6 / **144** | +0.94 |

E is in meV/atom and F in meV/Å.
- More training helps RACE by 15–18 %.
- On these neutral molecules the long-range stratum does not improve the energy. The 30-epoch edge of the latent
  RACE_LR was not robust.
- With the charge loss the charges are right, and the force is about 2 % lower (within the one-seed noise).

**Cutoff 3 Å vs 5 Å** (4G systems, the same recipe otherwise; test RMSE after the per-composition refit).

| system | model | F 5 Å → 3 Å (meV/Å) | E 5 Å → 3 Å (meV) |
|---|---|---|---|
| carbon chain | RACE | 20.2 → 42.4 (+110 %) | 3.7 → 9.9 |
| carbon chain | RACE_LR | 9.2 → 27.2 (+196 %) | 2.2 → 7.6 |
| NaCl | RACE | 43.3 → 47.3 (+9 %) | 27.1 → 27.7 |
| NaCl | RACE_LR | 35.8 → 36.7 (+2.5 %) | 23.2 → 20.1 |

- **Ionic NaCl.** RACE_LR barely loses accuracy at 3 Å, and at 3 Å it beats RACE at 5 Å. The long-range
  electrostatics compensates the short cutoff.
- **Covalent carbon chain.** RACE_LR is still 36 % better than RACE at 3 Å, but it loses more in relative terms, and
  at 3 Å it is worse than RACE at 5 Å. The Qeq stratum does not replace the 3–5 Å many-body information there.
- **Consequence.** Do not shorten the cutoff for RACE_LR on organic or biomolecular data. The long-range stratum
  adds to the local model; it does not replace its range.

## 13. Overnight local queue: seeds, short-range switch, charge-loss weight (2026-11-07/08)

Same 4G recipe as §10: reduced model, 1000 training structures, 200 epochs. Test RMSE after the train-only
per-composition refit (`runs/local_4g/summary_night.json`). Figures: `runs/local_4g/plots/ko_night_*.png`.

**Seeds 42 / 43 / 44** (mean ± sample std over 3 seeds):

| system | model | E (meV) | F (meV/Å) | F per seed |
|---|---|---|---|---|
| carbon chain | RACE | 3.4 ± 0.3 | 19.2 ± 1.7 | 20.2, 17.3, 20.2 |
| carbon chain | RACE_LR (Pauling + Hirshfeld loss 0.1) | 2.3 ± 0.2 | 9.8 ± 0.6 | 9.2, 10.0, 10.2 |
| NaCl | RACE | 26.8 ± 0.2 | 42.8 ± 0.5 | 43.3, 42.4, 42.9 |
| NaCl | RACE_LR | 23.9 ± 1.4 | 38.1 ± 2.0 | 35.8, 38.7, 39.8 |

The gains are robust to the seed: −49 % force on the carbon chain and −11 % on NaCl. Every RACE_LR seed beats every
RACE seed.

**Short-range switch `[3, 5]` Å** (RACE_LR, seed 42; force RMSE):
- carbon chain: 9.2 → 8.7 meV/Å, within the seed noise;
- NaCl: 35.8 → 42.7 meV/Å (+19 %), back to the RACE level (43.3).

Switching off the Coulomb energy below 5 Å removes the Na–Cl electrostatics at 2.4–5 Å, which is where the gain on
the ionic system comes from. For a model trained jointly from scratch, use no switch or a short one (r_off well
below the ion-pair distances). The `[4, 6]` Å example given to ctx-RACE is withdrawn.

**Charge-loss weight** (seed 42; force RMSE in meV/Å, and in brackets the mean |q| vs DFT):

| system | 0.01 | 0.1 | 1.0 |
|---|---|---|---|
| carbon chain | 10.9 | **9.2** | 11.1 |
| NaCl | 39.7 (0.23 / 0.39) | 35.8 | **31.7** (0.39 / 0.39) |

- 0.01 is too weak for both systems.
- The best weight is 0.1 for the covalent molecule and 1.0 for the ionic cluster.
- For pretraining, start at 0.1. For data rich in charged or ionic structures, also try 0.3–1.0.
- This comparison has one seed.

## 14. More data, MD stability, spin input, Au2/MgO doping (2026-11-08)

Reduced model as before; one seed unless stated. Figures: `runs/local_4g/plots/ko_nacl_*.png`, `ko_aumgo_*.png`.

**NaCl on the full training split (4500 structures, 100 epochs).** Test force RMSE per charge state, in meV/Å:

| model | data | all | Na9Cl8+ | Na8Cl8+ |
|---|---|---|---|---|
| RACE | 1000 | 43.3 | 41.7 | 44.8 |
| RACE_LR (Pauling + Hirshfeld 0.1) | 1000 | 35.8 | 34.1 | 37.4 |
| RACE | 4500 | 77.0 | 3.8 | 108.0 |
| RACE_LR | 4500 | 50.5 | 2.8 | 70.8 |
| RACE + charge/spin embedding | 4500 | **3.5** | 1.1 | **4.8** |
| RACE_LR + charge/spin embedding | 4500 | 4.0 | 1.1 | 5.5 |

- **With more data the local model fits Na9Cl8+ almost exactly but gives up Na8Cl8+.** The latter is bad on the
  training structures too, so the model cannot fit it at all; this is not a generalisation failure.
  - Both states have total charge +1, so the total-charge input cannot separate them.
  - Na8Cl8+ has an odd electron count (223, doublet) and Na9Cl8+ an even one (234, singlet).
  - RACE_LR does 34 % better than RACE, but it still fails.
- **The spin input resolves it.** The multiplicity is taken from the electron-count parity (`data_prep/add_spin_parity.py`;
  this assumes the lowest multiplicity in the DFT) and fed through the charge/spin embedding.
  - The Na8Cl8+ force error drops about 20× (108 → 4.8 meV/Å), and the energy RMSE drops to 2.4 meV.
  - RACE with the embedding alone is as good as RACE_LR with it. The missing information was the electronic state,
    not the electrostatics.
- **Consequence for pretraining.** The spin (and total charge) input is essential. OMol25 provides both.
  - Caveat: in this dataset the spin label is perfectly correlated with the composition (Na8 vs Na9), so it also
    acts as a global composition flag.
  - The Al-doped Au2/MgO slabs are doublets too. The spin was therefore not used there, because it would reveal the
    doping directly.

**NVE MD, 2 ps, 3 velocity seeds** (start: a Na9Cl8+ test frame):

| model | 1000 structures | 4500 structures |
|---|---|---|
| RACE | 2 / 3 stable | **3 / 3** |
| RACE_LR | 2 / 3 stable | **3 / 3** |

- The 1000-structure models fail on the same seed, both after leaving the training range.
- With 4500 structures every run conserves the energy to about 0.01 meV RMS. The holes of §12 were a data-size
  effect.

**Au2 on MgO(001), periodic, 1000 structures, 60 epochs** (test energy RMSE per 110-atom structure in meV; force in
meV/Å):

| model | undoped E | Al-doped E | F |
|---|---|---|---|
| RACE | 55.2 | 153.1 | 4.6 |
| RACE_LR | 46.2 | **76.1** | 4.9 |

- RACE cannot see the dopant 10 Å below the Au2, so its doped-slab error is three times its undoped error. RACE_LR
  halves it.
- **Au2 charge** (sum over the two Au atoms):

  | | undoped | doped | shift |
  |---|---|---|---|
  | DFT (Hirshfeld) | +0.088 | −0.197 | −0.285 e |
  | RACE_LR | −0.032 | −0.070 | −0.038 e |

  The direction is right and the per-structure correlation is 0.83, but the size is about 13 % of DFT. Long-range
  charge transfer is under-represented at this budget; a larger charge-loss weight is the next thing to try (NaCl
  preferred 1.0).

## 15. Pickle vs streaming data path: speed (2026-11-09)

Question: before OMol25 / OAM pretraining with the streaming loader (`bam_omat24/stream_data.py`, not yet wired into
`train_3bpa`), is it slower than the pickle path (graphs built once, cached, held in memory)?

Setup: `runs/diag/bench_stream/bench_stream_vs_pkl.py` (GPU limited: MPS 50 % SMs, RTX 4060 Ti; RACE float32,
AMSGrad).
- Both paths share the model, the step and the fixed batch shape (`stream_plan.derive_buckets`, one bucket).
- Each path is timed for one full epoch, twice; the table shows the mean.
- Ideal = the same batches already collated in memory, GPU only.
- The JAX equivalence tests of the stream loader (`tests/test_stream_io.py`, previously skipped outside Slurm) pass
  locally on CPU: 6/6.

Training speed (structures/s; % of ideal):

| set | ideal | pickle | stream W=1 | W=2 | W=4 | W=8 |
|---|---|---|---|---|---|---|
| OMol25 public subset, 10k molecules (16 atoms) | 959–995 | 989 (103 %) | 1016 (102 %) | 1022 | 1004 | 1012 |
| 4G Au2/MgO, 4,500 periodic cells (110 atoms), ASE neighbour list | 100 | 102 (101 %) | **61 (61 %)** | 114 | 112 | 109 |
| same, vesin neighbour list | 100 | 102 | 104 (103 %) | 104 | 103 | 103 |

The stream runs at 109–114 % of ideal on the ASE periodic set (both repeats). This is unexplained; treat it as the
±10 % noise of a desktop.

Loader alone (structures/s, no model):

| set | pickle | stream W=1 | W=2 | W=4 | W=8 |
|---|---|---|---|---|---|
| molecules | 3,900 | 5,200 | 10,000 | 16,300 | 15,200 |
| periodic, ASE | 3,550 | 62 | 119 | 215 | 300 |
| periodic, vesin | 3,840 | 860 | 1,630 | 2,680 | 3,330 |

One-time preprocessing:

| set | pickle build, 1 core | size pass, 8 workers | pickle size |
|---|---|---|---|
| molecules | 1.5 s | 0.8 s | 26 MB |
| periodic, ASE | 81 s | 16 s | 490 MB |
| periodic, vesin | 4 s | 1.5 s | 490 MB |

Findings:
- **Molecules (OMol-like):** streaming costs nothing. One worker already outruns the GPU 5×.
  - The stream packs fill 16.3 graphs per step against 15.7 for the bucketed pickle loader.
- **Periodic (OAM-like):** the ASE periodic neighbour list (`neighbors.neighbor_pairs`, ~18 ms per 110-atom cell) is
  the whole cost.
  - The pickle path pays it once, the stream every epoch.
  - With 8 workers the loader has only 3× headroom over this GPU at 50 % SMs. A GH200 running a float32 model would
    outrun it.
- **Fix:** `BAM_NEIGHBORS=vesin` (opt-in, needs `pip install vesin`).
  - It gives the same (i, j, S) pair set as ASE (200 Au2/MgO cells; `tests/test_neighbors_vesin.py` with small
    skewed cells and a molecule).
  - It is 57× faster on periodic cells and 2× on molecules.
  - With it, one worker matches the pickle path, and 8 workers give 33× headroom.
  - Pairs are sorted by (i, j, S). The edge order therefore differs from the ASE backend: the graph is the same and
    only the order of the sums changes. Use one backend for the size pass and training of a run.
- **Main-thread cost:** in every path, "get next batch" is about 70 % of the wall time on the periodic set. This is
  the host waiting for the GPU queue, not host work:
  - `stack_batches_for_devices` alone is 2.6 ms of a 79 ms step;
  - collating 8 cells takes 0.24 ms.
  - The data path is not the limit here.
- **Not measured locally:**
  - aselmdb sources (OMat24 / OMol25 lmdb; `lmdb` is not installed here);
  - Lustre reads;
  - the Grace CPU per-core speed;
  - multi-GPU.
  - These belong to the T6 throughput smoke on HANGANG (`docs/stream_loader_design.md` §4.10). Install `vesin` in
    `bam_cue` first, after asking.

## 16. Streaming path wired into training (2026-11-09)

- **What:** `data_format: "stream"` in `train_3bpa` (train / valid / test). Details and the next steps are in
  `docs/HANDOFF_stream_loader.md`.
- **End-to-end on the OMol25 public subset** (RACE_LR + charge loss + charge + spin, 5 epochs):
  - Valid F RMSE: 274 meV/Å with pickle, 243 meV/Å with stream.
  - Same time per epoch.
  - Resume from the data cursor works.
- **Block mixing:** a pack drawn from one 128-row block (same-molecule conformers) made training worse (282 meV/Å)
  until the rows of 16 blocks were shuffled together (`mix_blocks`, default 16).

## 17. Last-layer Bayesian regression on crystals (2026-11-09)

`bam_omat24/bayes_lastlayer.py` now handles periodic cells.
- **Features:** the cell enters the features.
- **Stress:** stress is linear in the readout weights θ:
  S = S_rest + (1/V) [(∂φ/∂cell)ᵀ cell + Σ_i ∂φ/∂x_i ⊗ x_i] θ.
  It forms a third likelihood block with its own noise when the training data carry stress labels (`--stress auto|on|off`).
- **Training rows:** force and stress rows are summed into AᵀA on the fly, so crystal sets fit in memory (`--max-train`).
- **Jacobian:** built from P vector-Jacobian products, `--jac-chunk` at a time. A full forward Jacobian of a 110-atom cell ran out of memory on the 16 GB GPU.

**Exactness** (θ fitted on training energies; then forces and stress on the validation rows with the same θ; needs more training structures than features):

| case | energy max | force max | stress max |
|---|---|---|---|
| carbon chain (molecule, 97 features, 100 train) | 6e-15 eV | 6e-13 eV/Å | — |
| AuMgO slab (periodic, 97 features, 110 train) | 7e-14 eV | 1e-12 eV/Å | 2e-14 eV/Å³ |
| `tests/test_bayes_lastlayer.py` (periodic RACE_LR, Ewald + Qeq) | < 1e-10 | < 1e-8 | < 1e-10 |

The test also checks that the stress of the per-structure function equals the model's own stress. Both use the graph's `volume` global.

**Calibration**

| set | net F RMSE | BLR F RMSE | mean σ_F | of which epistemic | rms z | 68 / 95 % coverage | Spearman(\|err\|, σ) |
|---|---|---|---|---|---|---|---|
| carbon chain test (300 train) | 19.5 | 13.1 | 12.4 | 1.0 | 1.06 | 0.75 / 0.93 | 0.37 |
| carbon chain test, 3-seed ensemble | 16.3 | — | 7.9 | — | 4.95 | 0.40 / 0.62 | 0.36 |
| AuMgO test (200 train) | 4.57 | 4.57 | 4.50 | 0.16 | 1.01 | 0.80 / 0.95 | 0.37 |

All force values in meV/Å.

- **Size is right, ranking is weak.** The force σ has the right size on average: rms z ≈ 1 and 95 % coverage 0.93–0.95. The 3-seed ensemble is over-confident (z ≈ 5).
- **Almost all of σ is the fitted noise term.** The epistemic part is 1 meV/Å for the carbon chain and 0.16 meV/Å for AuMgO. In distribution, σ is therefore nearly constant per component. The ranking of errors is weak (Spearman 0.37), the same as the ensemble's.
- **Next test:** out-of-distribution structures, where the epistemic term should grow.
- **Energies:** the energy σ of AuMgO (≈ 115–170 meV, ranking negative) reflects per-composition energy offsets between the Au2/MgO subsets, which the single bias column (∝ N) cannot fit. This is the gauge problem of section 14, not the method.
- **Cost:** about 3.3 s per 110-atom cell in float64 at 50 % of the RTX 4060 Ti. OMat24 on HANGANG: `runs/diag/bayes_lastlayer_oam.sh` (not run).

## 18. SOAP-curvature Laplace: first runs (2026-11-09)

`bam_omat24/soap_laplace.py` on the SOAP-trained local checkpoints (carbon chain, AuMgO slab; periodic cells added).
Grid over κ (curvature weight) and λ (isotropic prior precision) on a validation subset, 8 samples, σ = sample std
around the network (the "MAP" centre); the sample mean is also reported.

**Checks passed.** κ, λ → ∞ gives the network back (2e-14). The sampler's spread in the SOAP basis is 1/√(κν+λ)
(`tests/test_soap_laplace.py`). Whole grid of 15 points: 144 s (carbon chain), 383 s (AuMgO).

**Result: the SOAP second moments are not a usable curvature.**

| ν over the SOAP leaves | 10 % | median | 99.9 % | max |
|---|---|---|---|---|
| carbon chain, optimizer ν | 0 | 1.4e-15 | 2.8e-8 | 2.8e-3 |
| carbon chain, recollected at θ* (`--recollect 50`, per-structure squared gradients) | 0 | 1.6e-15 | 8.5e-8 | 5.4e-3 |
| AuMgO, optimizer ν | 0 | 1.3e-16 | 7.0e-9 | 5.9e-4 |

ν is an **empirical Fisher**: squared loss gradients, which are residual-weighted and therefore tiny at convergence
and concentrated in a handful of directions. Recollecting them at the EMA weights gives the same picture, so this is
not an EMA artefact. Consequences on the κ grid:

| regime | what happens | carbon chain test (net F RMSE 19.5 meV/Å) |
|---|---|---|
| κ ≤ 1e8, λ = 1e4 | κν < λ in every SOAP direction; the only parameter the curvature term touches is the global `scale` (Adam leaf, ν = 0.42, value 0.109). With κ = 0 `scale` is shaken by 10 % and all forces follow (σ_F 79, ρ 0.33). With κ = 1e6 it is frozen: σ_F 17–18, z 1.5, ρ **0.51–0.53** | best NLL; this is isotropic weight noise with `scale` fixed, not a curvature posterior |
| κ ≥ 1e12 | the top directions freeze, the rest keep 1/√λ, and the forces barely move | σ_F 2.5 → 0.00, z 8 → 14000, ρ 0.27–0.46 |
| AuMgO, best κ = 1e10, λ = 1e3 | σ_F 6.6 vs RMSE 4.6, z 1.1, ρ 0.33; energy σ 2.8 eV (z 0.04) because isotropic weight noise adds coherently over 110 atoms | no better than κ = 0 (ρ 0.27–0.31) |

Where the theory puts κ (κ ≈ number of training structures ≈ 1e3, with recollected ν) the posterior is the prior
λ in all but a few directions.

**Comparison of the three methods on the carbon chain test set** (forces, meV/Å):

| method | σ_F | rms z | 95 % coverage | Spearman(\|err\|, σ) | cost |
|---|---|---|---|---|---|
| ② last-layer BLR (section 17) | 12.4 | 1.06 | 0.93 | 0.37 | P backward passes per structure |
| 3-seed ensemble | 7.9 | 4.95 | 0.62 | 0.36 | 3 trainings |
| ③ SOAP-Laplace, κ = 1e6, λ = 1e4, 32 samples | 18.5 | 1.54 | 0.94 | 0.53 | 32 forward+force passes per structure |
| ③ with κ = 0 (isotropic noise) | 29.3 | 1.83 | 0.91 | 0.36 | same |

The ③ ranking (0.53) is the best of the three, but it comes from isotropic weight noise with the global scale fixed,
which is a SWAG-like perturbation, not from the SOAP curvature.

**Conclusion.** The design assumption "ν ≈ Fisher / B" (docs/SOAP_Laplace.md section 2) fails for the empirical
Fisher. A Laplace posterior over all weights needs the GGN (output Jacobians, no residuals), which SOAP does not
store; that is what BAM V1's torch Laplace (curvlinops GGN, diagonal) computes. Options: (a) GGN diagonal in the SOAP
eigenbasis from per-structure output Jacobians (cost ≈ ② times the number of outputs; not cheap), (b) keep ② for the
calibrated σ and the ③ κ = 1e6 perturbation as a cheap ranking heuristic, (c) stop here. ③ is on hold again.

## 19. SOAP-basis Laplace with the GGN curvature (option (a) of section 18, 2026-11-09)

`soap_laplace.py --ggn N [--ggn-probes K]` replaces ν by the diagonal of the generalised Gauss-Newton
Σ_k J_kᵀJ_k/σ_k² (energy and force Jacobians wrt the weights, Gaussian likelihood with the network's validation
RMSEs as σ_E, σ_F) in the SOAP eigenbasis, averaged over N training structures. Two estimators: exact (one VJP per
output, 1 + 3 n_atoms per structure) and Hutchinson (K Gaussian probes, one VJP each). The untempered posterior
precision is n_train·ν + λ (`--kappa-times-ntrain`, κ = 1). Two predictives: MC (re-evaluate the network at each
weight sample) and GLM (linearised, f(θ*) + J(θ−θ*), one JVP per sample, Immer et al. 2021; `--predictive glm`).
Prior precision: a scalar λ from a validation grid, the closed-form evidence optimum at κ = n_train
(`--lam-evidence`), or one λ per leaf from the MacKay fixed point λ_g = γ_g/|θ_g|² (`--lam-per-leaf`).

**Checks.** Exact GGN equals the dense-Jacobian result on a toy model (`tests/test_soap_laplace.py`); Hutchinson
with 2 probes × 20 carbon structures agrees with the exact diagonal over 5.7e5 entries (Spearman 0.999, ratio
within ±0.1 dex for 68 % of the entries). Cost: exact 2 s per 10-atom structure, Hutchinson 0.2 s per probe;
GLM predictive 16 samples ≈ MC 8 samples per structure.

**The curvature is now real, but it determines few directions.** n_train·ν exceeds 1 in 3e4 of 5.7e5 directions
(carbon chain) and the evidence counts γ = 6.6e4 well-determined directions (AuMgO: 1.9e5 of 6.8e5). Everything else
is the prior. The evidence-optimal isotropic λ is 0.10 (carbon) and 0.29 (AuMgO), far below the validation-chosen
10–100: an isotropic Gaussian is a poor model of the trained weights (rms ≈ 1.1, unused leaves such as the
logvar heads and the vector channels of the last layer have γ = 0). Per-leaf λ_g range from 6e-8 (unused leaves)
to 84 (`scale`), median 0.17 (carbon) / 0.56 (AuMgO).

**Carbon chain test set (n = 100, network F RMSE 19.5 meV/Å, E RMSE 207 meV):**

| predictive | κ·n_train, λ | σ_F | rms z | cov95 | ρ(\|err\|, σ) | E σ (meV) | E rms z | ρ(σ_F, \|F\|) |
|---|---|---|---|---|---|---|---|---|
| MC, 8 samples | 1, 100 (valid) | 16.0 | 1.55 | 0.91 | 0.51 | 405 | 0.51 | – |
| MC, 16 samples | 10, 30 (valid) | 19.7 | 1.11 | 0.99 | **0.58** | 423 | 0.49 | 0.58 |
| GLM, 16 samples | 0.1, 10 (valid) | 21.9 | 1.07 | 0.97 | 0.43 | 177 | 1.22 | 0.56 |
| GLM | 1, λ_ev = 0.10 (no tuning) | 25.0 | 0.85 | 0.96 | 0.34 | 81 | 2.56 | – |
| GLM, per-leaf | 1, 10·λ_g (valid) | 17.1 | 1.32 | 0.93 | 0.37 | 63 | 3.32 | – |
| ② BLR (section 17) | – | 12.4 | 1.06 | 0.93 | 0.37 | – | – | – |
| ③ old, κ = 1e6 (scale frozen, section 18) | – | 18.5 | 1.54 | 0.94 | 0.53 | 480 | 0.43 | – |

ρ(|err_F|, |F|) = 0.29 on this set, so σ ∝ |F| alone would give about 0.3; both predictives have σ tracking |F|
at 0.56–0.58.

**AuMgO test set (n = 50, network F RMSE 4.6 meV/Å, E RMSE 122 meV; all GGN from 60 structures):**

| predictive | κ·n_train, λ | σ_F | rms z | cov95 | ρ_F | E σ (meV) | E rms z | ρ_E |
|---|---|---|---|---|---|---|---|---|
| MC, 8 samples | 0.1, 100 (valid) | 4.6 | 1.20 | 0.92 | 0.34 | 1635 | 0.08 | −0.58 |
| GLM, 16 samples | 0.1, 10 (valid) | 4.6 | 1.01 | 0.95 | 0.35 | 305 | 0.40 | 0.26 |
| GLM | 1, λ_ev = 0.29 (no tuning; valid) | 3.2 | 1.40 | – | 0.34 | – | 0.75 | 0.66 |
| GLM, per-leaf | 1, 0.3·λ_g (valid) | 3.6 | 1.25 | 0.91 | 0.34 | 146 | 0.82 | 0.36 |
| ③ old, κ = 1e10 (section 18) | – | 6.6 | 1.09 | 0.94 | 0.33 | 2783 | 0.04 | 0.33 |

On the AuMgO validation set the GLM energy ranking is 0.65–0.84 at every (κ, λ) with κ ≥ 1; the MC energy σ is
1.6–5 eV because isotropic weight noise adds coherently over 110 atoms.

**Reading.**
1. The GGN fixes the pathology of section 18: κ now matters (σ_F falls monotonically with κ·n_train), and the
   untempered posterior with the evidence prior is calibrated to within a factor 1.2–1.4 on both systems without
   validation tuning (GLM). That is the behaviour a Laplace approximation should have.
2. The force ranking does not improve: 0.34–0.43 for the GLM on both systems, the same as ② (0.37) and the
   ensemble (0.36). AuMgO is pinned at 0.34–0.35 in every variant (its force error, 4.6 meV/Å, is at the label noise
   floor). The MC predictive reaches 0.58 on the carbon chain, but its σ tracks |F| at 0.58 and the error itself
   tracks |F| at 0.29, so part of that is the force magnitude, and its energy σ is unusable on cells.
3. The GLM predictive is the first method here with a usable energy σ on AuMgO (z 0.4–0.8, ρ_E 0.3–0.8), where ②
   failed on per-composition offsets and MC blew up.
4. Per-leaf evidence priors change little; most of the weight space is unconstrained by the data and only the ≈ 1e5
   well-determined directions matter, which the GLM already handles correctly.

**Conclusion.** With the GGN the SOAP-basis Laplace is a correct, cheap (one pass of VJPs over a training subset,
then one JVP per sample per structure) and self-calibrating posterior over all weights; it is not a better error
ranker than the last-layer BLR on these in-distribution sets. The question it was built for, calibration under
extrapolation (3BPA 600 K / 1200 K, dihedral scan; OMat24 out-of-distribution), is open and is where the
whole-weight posterior should separate from ②. Recommended next run on HANGANG: `--ggn 500 --ggn-probes 2
--predictive glm --kappa 1 --kappa-times-ntrain --lam-evidence --lam 1 10` on the 3BPA checkpoint, against the
ensemble and ② on the 600 K / 1200 K sets.

## 20. Out-of-distribution check in the BAM V1 paper's 3BPA setting (2026-11-10)

The V1 paper (arXiv:2510.03046, Table 4 / S6) trains on 3BPA 300 K (450 + 50 structures) and tests on 300 K
(in-distribution) and 600 K / 1200 K (extrapolation); it reports RMSEs only (RACE 12.1 / 31.8 / 115.3 meV/Å forces,
3.4 / 11.7 / 37.5 meV energies; RACE-LA 18.2 / 51.0 / 171.8 and 4.8 / 15.3 / 60.8). Its uncertainty metrics
(calibration error, OOD-detection AUROC) are given only for oBN25, which is not available locally. Here the same
3BPA split is run with the reduced local model (hidden 64, 3 layers, lmax 2, 300 epochs, 58 min on the limited
GPU; `input/local_3bpa/race_s42.json`), then ② and ③ on 400 structures per test set
(`runs/local_3bpa/uq_s42/`, chain in the scratchpad; HANGANG version `runs/diag/ood/ood_3bpa_hangang.sh`;
summary script `runs/diag/ood/ood_summary.py`).

**Network (final EMA, gauge refit):** forces 10.6 / 25.7 / 75.6 meV/Å, energies 6.7 / 12.6 / 40.7 meV at
300 / 600 / 1200 K. Better than the paper's full RACE on forces at every temperature, worse on the 300 K energy.

**Uncertainty per temperature** (forces meV/Å, 400 structures each; ③ GLM: κ = n_train, λ = 10 chosen on 300 K
validation; ③ MC: κ = 10 n_train, λ = 30; ② hyperparameters by evidence on the training set):

| method | set | F RMSE | σ_F | rms z | cov95 | ρ(\|err\|, σ) | ρ per structure | E RMSE | σ_E | E rms z |
|---|---|---|---|---|---|---|---|---|---|---|
| ② last-layer BLR | 300 K | 11.3 | 3.5 | 3.19 | 0.64 | 0.54 | 0.64 | 3.9 | 2.1 | 1.83 |
| | 600 K | 27.0 | 3.5 | 7.59 | 0.42 | 0.58 | 0.75 | 13.8 | 2.2 | 6.42 |
| | 1200 K | 80.4 | 3.6 | 22.3 | 0.21 | 0.62 | 0.76 | 37.1 | 2.2 | 17.1 |
| ③ GGN Laplace, linearised | 300 K | 11.3 | 10.7 | 0.90 | 0.96 | 0.50 | 0.47 | 3.9 | 15.2 | 0.26 |
| | 600 K | 27.0 | 12.5 | 1.71 | 0.82 | 0.54 | 0.67 | 13.8 | 15.8 | 0.90 |
| | 1200 K | 80.7 | 16.3 | 3.64 | 0.58 | 0.56 | 0.68 | 37.1 | 16.8 | 2.24 |
| ③ GGN Laplace, MC | 300 K | 11.3 | 10.5 | 1.06 | 0.93 | 0.37 | 0.43 | 3.9 | 177 | 0.02 |
| | 600 K | 27.0 | 11.6 | 2.10 | 0.78 | 0.42 | 0.50 | 13.8 | 177 | 0.08 |
| | 1200 K | 80.7 | 13.6 | 4.58 | 0.51 | 0.49 | 0.64 | 37.1 | 178 | 0.21 |

**OOD detection** (AUROC of the structure-level σ, 300 K vs 600 K / 1200 K, the paper's oBN25 metric):

| method | σ_F, 600 K | σ_F, 1200 K | σ_E, 600 K | σ_E, 1200 K |
|---|---|---|---|---|
| ② BLR (total or epistemic part alone) | 0.99 | 1.00 | 0.86 | 1.00 |
| ③ GGN Laplace, linearised | 0.98 | 1.00 | 0.64 | 0.72 |
| ③ GGN Laplace, MC | 0.98 | 1.00 | 0.60 | 0.67 |

(The paper's oBN25 values: DE 1.00, IVON 1.00, SWAG 0.58, MVE 0.54.)

Figure `docs/figures/ood_3bpa_local_sigma_vs_error.png` (numbers: `docs/figures/ood_3bpa_local_summary.json`): structure-mean σ_F against structure force RMSE
per temperature.

**Reading.**
1. **② is a perfect OOD detector with the wrong scale.** Its σ is 3.5 meV/Å at every temperature because the
   evidence-fitted noise (training residuals, 3.5 meV/Å) dominates and the epistemic part is 0.17 → 0.21 → 0.32
   meV/Å. That epistemic part ranks structures correctly (AUROC 0.99 / 1.00, per-structure ρ 0.64–0.76, the best of
   the three), but the total σ does not grow with the error (z 3.2 → 7.6 → 22). Two causes: the training residuals
   underestimate the test error of an over-fitted 450-structure model (train F ≈ 3.5 vs test 11.3 meV/Å), and the
   last-layer posterior only sees the readout, so the feature extrapolation error is invisible to it.
2. **③ (linearised) is the only method whose σ moves with the extrapolation**: 10.7 → 12.5 → 16.3 meV/Å while the
   error goes 11.3 → 27 → 81. Calibrated at 300 K (z 0.90, cov95 0.96), 1.7× over-confident at 600 K, 3.6× at
   1200 K; its energy σ is honest at 300 / 600 K (z 0.26 / 0.90). The slope of σ against error is about 1/3 in log
   space: a Laplace posterior around one mode under-reacts to the feature shift, as expected, but it reacts.
3. **③ MC** adds nothing over the linearised predictive here (lower ρ, worse z, energy σ useless), confirming
   section 19.
4. The force-σ AUROC is ≈ 1 for every method because the 1200 K structures are far from 300 K; the energy σ
   separates the sets only for ②.

**Conclusion.** In the paper's own OOD setting, ③ with the GGN and the linearised predictive is the usable
posterior: calibrated in distribution and under-confident by a factor that grows with the extrapolation (1.7 at
600 K, 3.6 at 1200 K), while ② is a good ranker whose σ scale is fixed by the training noise. A natural
combination is ②'s ranking with ③'s scale, or ③ with the noise term of ② added (both omit the aleatoric floor
differently). The same chain on the full-recipe HANGANG checkpoint (`runs/diag/ood/ood_3bpa_hangang.sh`, with the
3-seed ensemble as the reference) is the next run; the paper's RACE-LA numbers (RMSE only) can be compared with the
"F(mean)" rows of `soap_laplace.json` there.
