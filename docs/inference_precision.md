# Inference precision: when a float32-trained model must run in float64

Models trained with `"x64": false` (float32 parameters, optionally a float64 SOAP state) are evaluated in float32 by
default. Setting the environment variable `BAM_PREDICT_X64=1` makes `race_v2.inference.predict.Predictor` (and everything
built on it: `relax_wbm`, `relax_batched.RaceEvaluator`, `scripts/eval/mbd/kappa.py`) enable `jax_enable_x64`, cast the
parameters to float64 and feed float64 graphs. The parameters are the same numbers; only the arithmetic changes.

## Rule

| task | precision |
|---|---|
| phonons, force constants, thermal conductivity (kappa SRME), any relaxation to fmax ~1e-4 eV/A | **float64 required** |
| WBM-type relaxations (fmax 0.05 eV/A), energies, stability classification | float32 is enough; float64 is ~2 % better in MAE / RMSD |
| training | float32 model + float64 SOAP state (`soap_state_dtype`), see `docs/HANDOFF_soap_speed_2026-11-11.md` on branch racev2 |

## Measurements (2026-11-15/16, RACEv2 OAM fine-tuning checkpoint, virtual epoch 18, same parameters)

Thermal conductivity, 103 PhononDB structures (displacement 0.03 A, fmax 1e-4, symprec 1e-5):

| arithmetic | kappa SRME | relaxations at the 300-step limit | imaginary modes |
|---|---|---|---|
| float32 | 0.861 | 33 of 99 | 6 |
| float64 | 0.220 | 0 | 0 |

In float32 the force noise (~1e-4 eV/A between two runs of the same model) is as large as the convergence threshold
and the finite-difference force differences; kappa comes out too small (median factor 1.7).

WBM 3k tracking sample (2,536 unique prototypes), same checkpoint:

| arithmetic | MAE (eV/atom) | RMSE | R2 | F1 | RMSD (A) |
|---|---|---|---|---|---|
| float32 | 0.0199 | 0.0510 | 0.920 | 0.893 | 0.0647 |
| float64 | 0.0195 | 0.0497 | 0.924 | 0.893 | 0.0631 |

Per-structure energy difference: median 1.3e-4 eV/atom; a few structures relax to a different minimum in float32.

## Memory

float64 doubles the memory. With a 7 A cutoff the largest PhononDB supercells (300-512 atoms) do not fit when several
are batched: rerun those with a smaller batch (`APC=256` in `scripts/eval/mbd/kappa.sh`; the supercells and the
displacements are unchanged, only the number of structures per GPU call).
