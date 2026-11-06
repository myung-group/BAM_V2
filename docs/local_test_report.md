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
- **Open points.** These are listed in the last section: model charges are not returned, the E0 offsets are a
  choice, and the Ewald settings for the Au2/MgO slab are not checked.

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

## 5. What did not work or needs a decision

1. **Model charges are not returned.**
   - The design doc says "The model also returns `charges`". `RACE_LR.node_energies` does put them in its output.
   - The shared `bam_omat24/utils.energy_forces_stress` copies only fixed keys into its aux, so `model(graph)` has no
     `charges` key.
   - A fix would pass `results.get('charges')` through when it is not None. That touches the shared default code
     path, and no test failed, so it was **not** changed here.
   - It is needed to analyse the charges by charge state in step 2.
2. **E0 offsets.**
   - The archive has no isolated-atom energies, and the trainer stores energy targets in float32 after subtracting
     `atom_energies`. Raw totals reach −1.48e6 eV, where float32 resolves only about 0.1 eV.
   - `e0.json` therefore holds per-element least-squares offsets fitted on the training split only. For Ag3 and
     Au2/MgO these are minimum-norm values, because the compositions do not determine each element.
   - The decision is whether to use these offsets as `atom_energies`, or to compute isolated-atom energies with
     FHI-aims PBE light.
3. **Ewald settings for Au2/MgO.**
   - Au2/MgO is a slab in a 3D periodic cell with vacuum along c, and it may carry a net dipole, especially the Al-doped
     slab with Au.
   - The default α = 0.3 Å⁻¹, n_real = 2 and n_recip = 6 were not checked for convergence on this cell.
   - A dipole correction is not implemented. Run the α-independence check on this cell before training.
4. **Split stratification.** The split is not stratified by charge state. The per-state counts are within about ±1 %
   of 90/10, as listed in §16.
5. **Environment notes.** These do not block the run.
   - Python 3.12 came from conda, because the system has 3.10.
   - pip's NVIDIA extra index fails DNS on this PC.
   - `tqdm` and `natsort` must be added to the install line of the prompt.
