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
