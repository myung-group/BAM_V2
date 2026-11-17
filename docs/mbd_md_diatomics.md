# Matbench Discovery molecular dynamics (CMDS) and diatomics (CDS) with a RACE v2 checkpoint

Besides the discovery metrics (F1, MAE, RMSD, κ SRME, combined into CPS), the Matbench Discovery leaderboard has two
more scores. Neither is part of CPS.

| Score | Task | What it measures |
|---|---|---|
| **CMDS** | 17 NVT rollouts of the DynaMat v1.0 systems (80,000 steps of 0.25 fs = 20 ps each) | vibrational density of states, angular distribution, pressure, and the run time |
| **CDS** | energy curves of the homonuclear dimers (119 distances from 0.1 to 6 Å, 87 scored elements) | accuracy against PBE, geometry of the curve, smoothness, and the run time |

`scripts/eval/mbd/md_diatomics.py` runs both with the benchmark's own code and adds only an ASE calculator for the
model and the two combined scores. Results of the final RACE v2 OAM model are in
[results/race_v2_oam_mbd](results/race_v2_oam_mbd/README.md).

## 1. What comes from the benchmark and what is ours

- From the benchmark (`janosh/matbench-discovery`, read at commit `6d587c1`): the rollouts
  (`matbench_discovery.md.run_md_benchmark`), the diatomic sweep (`predict_curves`, `write_results` of
  `models/run_diatomics.py`) and every metric (`matbench_discovery.metrics.{md,diatomics}`).
- Ours: `RaceV2Calculator`, an ASE calculator around `race_v2.inference.relax_batched.RaceEvaluator` that returns
  energy, forces and stress. The calculator of `race_v2.inference.predict` has no stress, and the pressure metric
  needs it.
- Ours: the combined scores. The benchmark computes them on its web site, not in the Python package, so the formulas
  and constants were copied from `site/src/lib/combined-scores.svelte.ts` and `site/src/lib/labels.ts`.

```
CMDS = 0.3 (1 - vDOS error / 100) + 0.3 (1 - pressure error / 100) + 0.2 (1 - ADF error / 100) + 0.2 speed
       speed = 1 at 9,000 s and 0 at 300,000 s of summed rollout time, linear in log(time)

CDS  = (4 accuracy + 2 geometry + 1 speed + 2 physicality) / 9 x (fraction of the 87 scored elements completed)
       accuracy    = 5/9 s(E MAE; 0 at 4 eV) + 4/9 s(F MAE; 0 at 5 eV/Å)
       geometry    = 1/2 s(wall distance; 0.5 Å) + 1/3 s(bond length; 0.8 Å) + 1/6 s(well depth; 3.5 eV)
       speed       = 1 at 50 s and 0 at 5,000 s, linear in log(time)
       physicality = 2/5 s(energy jump; 3) + 2/5 s(force flips; 1 to 3.5) + 1/5 s(tortuosity; 1 to 2)
       s(x; b) = 1 - x / b clamped to [0, 1]; every x is the plain mean over the elements
```

Because the errors are plain means over the elements, a few diverging curves can set a pillar to 0.

The model is not registered in the benchmark, so the curated per-model exclusions of diatomic curves do not exist for
it. Curves the model cannot give (unknown element, non-finite value in the scored range) are counted as excluded, as
the benchmark does for exclusions found at run time. The final OAM model gives all 87 scored curves.

## 2. Requirements

| Item | Value |
|---|---|
| Python | 3.14 or newer (the benchmark code needs it) with the model stack (JAX, cuEquivariance) and the benchmark's dependencies. On HANGANG: conda environment `mbd314` |
| `MBD_REPO` | clone of `janosh/matbench-discovery` |
| `MBD_CACHE_DIR` | folder with `md/2026-06-29-dynamat-v1.0-reference-trajectories.h5` and `diatomics/diatomics-dft.json.gz` |
| Inference | float64 (`BAM_PREDICT_X64=1`, set by the launcher), `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` |
| Hardware | one node with 4 GPUs: one process per GPU for the rollouts |

## 3. How to run

From the work folder (the one with the links `race_v2` and `scripts`):

```bash
# short check first: 2,000-step rollouts and the diatomics dry run (about 6 minutes)
sbatch -J mdtest -p normal --comment="field=sciai;appl=jax" --nodes=1 --ntasks-per-node=1 --exclusive --gres=gpu:4 \
    -t 00:30:00 --export=ALL,TEST=1,CKPT=<ckpt.pkl>,TAG=<test name> scripts/eval/mbd/md_diatomics.sh

# full run: 17 rollouts, diatomics, summaries (about 2.5 hours on 4 GH200 GPUs)
sbatch -J mdbench -p normal --comment="field=sciai;appl=jax" --nodes=1 --ntasks-per-node=1 --exclusive --gres=gpu:4 \
    -t 08:00:00 --export=ALL,CKPT=<ckpt.pkl>,TAG=<name> scripts/eval/mbd/md_diatomics.sh
```

| Variable | Meaning |
|---|---|
| `CKPT`, `TAG` | checkpoint and the name of the output folder `runs/mbd/<TAG>` |
| `MD_TIGHT_PAD` | `1` (default): pad each force call to the structure size + 6 %; `0`: the evaluator's padding ladder (Section 5) |
| `MD_PAD_MARGIN` | margin of the tight padding on the number of edges (default 0.06) |
| `TEST=1` | short check; prints the padded shapes and the time per call |
| `PROFILE=1` | `md_profile.py` only: time per stage of one force call |
| `BAM_NEIGHBORS` | neighbour list backend (`vesin` by default, `ase`) |

A rerun with the same `TAG` reuses finished rollouts.

Outputs in `runs/mbd/<TAG>`:

| File | Content |
|---|---|
| `md/md_summary_<key>.json` | mean errors, summed run time, CMDS |
| `md/*-md-metrics-*.csv.gz` | metrics and run time per system (one file per process) |
| `md/<system>-nvt-<key>.extxyz` | the trajectories |
| `diatomics/diatomics_summary_<key>.json` | mean metrics, pillars, CDS, metrics per element |
| `diatomics/diatomics-<key>.json.gz` | the curves, in the format of the benchmark |

The values of the other models and the comparison tables:

```bash
python -I scripts/eval/mbd/leaderboard_from_yaml.py <matbench-discovery clone> <out.json>   # needs /models/**/*.yml
cd docs/results/race_v2_oam_mbd && python -I report/table_sota_all.py figures/sota_table_racev2_oam_all_metrics
```

## 4. Results of the final RACE v2 OAM model

Checkpoint `oamft_final.pkl` (OMat24 2 epochs, then MPtrj ×8 + sAlex ×1 for 6.2 epochs), float64, NVIDIA GH200,
job 1290168 with tight padding. Ranks are among the 49 models shown on the leaderboard page (read on 2026-11-12)
that report the value, plus RACE v2.

| | Value | Rank |
|---|---|---|
| **CMDS** | **0.721** | 13 / 47 |
| vDOS error | 22.54 % | 21 / 48 |
| ADF error | 1.48 % | 21 / 48 |
| pressure error | 52.40 % | 27 / 48 |
| summed rollout time | 22,174 s (speed 0.743) | 12 / 47 |
| **CDS** | **0.376** | 36 / 48 |
| accuracy pillar | 0.000 (E MAE 45.52 eV, F MAE 345.8 eV/Å) | 44 / 48 |
| geometry pillar | 0.683 | 13 / 48 |
| speed pillar | 0.999 (50.3 s) | 4 / 48 |
| physicality pillar | 0.508 | 35 / 48 |

- Run times of the other models are on other hardware, so the speed terms compare submissions, not architectures.
- A first run with the evaluator's padding ladder (job 1288683) gave CMDS 0.714 (26,710 s) and CDS 0.353 (129.8 s for
  the curves). The curve metrics of the two runs are equal; the MD errors differ in the second or third digit, as
  expected for chaotic trajectories.

### Why the CDS accuracy pillar is 0

- The energies of Ne, Ar and Kr diverge to ±1e9~1e10 eV below about 1 Å. Mean energy error per element: Ne 3,758 eV,
  Kr 102 eV, Ar 29.1 eV, Xe 6.3 eV, He 5.1 eV. The median over the 87 elements is 0.66 eV, and 82 elements are below
  the 4 eV at which the pillar reaches 0.
- With the same formula and these elements left out of the means (for reference only, not a leaderboard value):
  without Ne, Ar, Kr 0.85 eV and 1.69 eV/Å, CDS 0.768; without the five noble gases 0.73 eV and 0.91 eV/Å, CDS 0.831.
- On the 82 elements that are not noble gases the errors are close to those of BAM-MP-DeNS: mean E MAE 0.730 against
  0.725 eV, mean F MAE 0.912 against 0.984 eV/Å. BAM-MP-DeNS does not diverge for the noble gases (4.7~11.7 eV).
- These distances are not in the training data, and the model's short-range repulsion does not hold the curve there.
  The model has a UFF pair repulsion (`pair_repulsion: "uff"`), but it is multiplied by a switch
  `sigmoid((r - 1.25 (R_i + R_j)) / 0.3)` with the covalent radii `R` (`race_v2/models/race.py`), so it is switched
  off below about 1.25 times the sum of the covalent radii. It therefore does not act at collision distances, and it
  is applied with the same switch to elements that form no bonds. The switch centres are 1.45 Å (Ne), 2.65 Å (Ar)
  and 2.90 Å (Kr).
- Removing the switch at inference would not be enough: the UFF term is of the order of thousands of eV at these
  distances, the divergence of the learned part is ±1e9 eV. A fix needs a change of the model (repulsion that stays
  on at short range, damping of the learned part there, or short-distance dimer data) and a new training.
- 15 elements have an attractive hole below 0.7 times the bond length. Six of them are of practical concern (Kr, Ar,
  Ne, He, Pa, U); for the other nine the hole is only at 0.03~0.16 times the bond length. The other 72 curves have
  no such hole.

## 5. Speed of the MD force call

Profile of one rollout step, 500 atoms, float64, one GH200 (`PROFILE=1`, jobs 1288945 and 1289017):

| Part | Time per step |
|---|---|
| whole step | 41 ms |
| calculator | 39.5 ms |
| model call on the device | 33~35 ms |
| neighbour list (vesin) | 4.3 ms |
| padding and collation | 0.4 ms |
| integrator and frame output | below 2 ms |

- Writing the frames to Lustre or to `/dev/shm` makes no difference.
- **Tight padding** (`MD_TIGHT_PAD=1`, the default). The evaluator pads every batch to a coarse ladder of sizes made for
  relaxations of many structures (for the 500-atom system 49,152 edges for 33,704 real ones). In MD the size hardly
  changes, so the calculator pads to the structure size + 6 % (nodes rounded to 16, edges to 512) and keeps one shape
  per atom count; a shape grows, with one recompilation, only when a later frame does not fit. The device call
  becomes 7~30 % faster. Energies agree within 1e-13 eV, forces within 4e-15 eV/Å and stresses within 2e-16 eV/Å³,
  comparable to the difference between two calls with identical input.
- **No skin list.** A neighbour list with a 0.5 Å skin gives the same energies and forces, but it has 15~23 % more
  edges, and the device call scales with the number of edges. What it would save, the neighbour search, is 4 ms of a
  41 ms step. It was not implemented.

## 6. Not verified

- The combined scores were computed with formulas copied from the site code, not by the benchmark's submission
  pipeline. The scores of the other models in the tables are recomputed in the same way from their YAML files;
  they were not compared one by one with the numbers on the page.
- The run time is measured on NVIDIA GH200 in float64. The leaderboard does not normalise run times by hardware.
- The diatomic results were not checked against an independent implementation of the model (the PyTorch conversion).
