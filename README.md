# RACEv2: Highly locally resolved MLIP

RACEv2 is an improved version of RACE, an E(3)-equivariant message-passing interatomic potential. The code lives in the
`bam_omat24` package (JAX + Flax NNX). Symmetry operations (irreps, spherical harmonics) use e3nn-jax; the heaviest
operation, the edge tensor product, runs on the CUDA kernels of NVIDIA cuequivariance-jax (`"tp_backend": "cue"`).
The `_cue` suffix of the config files refers to these kernels; it is not a PyTorch package.

Every result below was trained on the official training split of its dataset only: no pretraining, no extra data,
no data augmentation. All equations, tables and figures are collected in
[`docs/RACEv2_methods_summary.pdf`](docs/RACEv2_methods_summary.pdf) (in Korean).

![RACEv2 architecture](docs/RACEv2_architecture_en.png)

## Changes with respect to RACE V1 (final recipe)

| Component | RACE V1 | RACEv2 |
|---|---|---|
| Atomic feature x | species embedding (`x_mode=species`) | per-layer linear projection of the neighbourhood summary (`x_mode=basis`) |
| Scaling of x | none | hill: x/r · 2σ(a log r + b), a and b learned per layer and element (`x_norm=hill`) |
| Layer update | plain residual | Chebyshev-type h⁽ᵗ⁾ = 2γ⊙δ⁽ᵗ⁾ − h⁽ᵗ⁻²⁾ with LayerScale 0.5 (`residual=chebyshev`) |
| Message normalisation | LayerNorm + 1/√⟨N⟩ | division by the mean neighbour count (`message_norm=avg`) |
| Update / readout | gate / linear | linear / gated |
| Radial basis | Bessel | Bessel + element shell-radius channels, Gaussian prior against high-order oscillations (σ = 0.224) |
| Repulsion | none | UFF non-bonded repulsion (fixed functional form, smoothly switched off for bonded pairs), one learned global scale s |
| Training | Adam | SOAP (lr 1e-2, β 0.95), Huber δ = 10, EMA 0.99, float64, closed-form energy-constant refit δ* after training |

Default size: 5 layers, lmax 3, cutoff 5 Å, 128 channels (256 for rMD17 and MD22, 64 for water).

## Results (mean of 3 seeds, checkpoint selected on validation)

### 3BPA (RMSE, E in meV/molecule / F in meV/Å, trained on 500 frames at 300 K)

| Model | 300 K | 600 K | 1200 K | Dihedral |
|---|---|---|---|---|
| **RACEv2** | **1.56 / 6.31** | **6.05 / 15.92** | **22.20 / 45.16** | 16.67 / 19.09 |
| RACE V1 | 3.4 / 12.1 | 11.7 / 31.8 | 37.5 / 115.3 | — |
| GRACE (PRX 2024) | 1.7 / 7.6 | 7.3 / 19.2 | 25.8 / 61.7 | **7.4 / 15.3** |
| TACE | 2.69 / 8.68 | 8.10 / 19.88 | 23.42 / 54.26 | 11.0 / 16.45 |
| MACE | 3.0 / 8.8 | 9.7 / 21.8 | 29.8 / 62.0 | 7.8 / 16.5 |

### rMD17 (MAE, official split, 1000 training frames, L5h256) vs the best published values

For every molecule the lowest published energy error and the lowest published force error were picked separately.
Bold marks the better value.

| Molecule | RACE V1 E | RACEv2 E | Best published E | RACE V1 F | RACEv2 F | Best published F |
|---|---|---|---|---|---|---|
| aspirin | 3.3 | **1.19** | 1.55 (GotenNet) | 9.6 | **4.01** | 5.26 (FreeCG) |
| azobenzene | — | **0.36** | 0.5 (ACE(g)/PACE) | — | **1.41** | 2.09 (GotenNet) |
| benzene | — | 0.02 | **0.01** (ACE(g)) | — | 0.22 | **0.1** (ACE(g)) |
| ethanol | 0.8 | 0.21 | 0.2 (ACE(g)) | 3.6 | 1.41 | 1.4 (ACE(g)) |
| malonaldehyde | 1.4 | **0.36** | 0.5 (ACE(g)/PACE) | 7.2 | **2.49** | 3.1 (ACE(g)) |
| naphthalene | 0.9 | **0.12** | 0.17 (GotenNet) | 2.6 | **0.65** | 0.9 (ACE(g)) |
| paracetamol | — | **0.61** | 0.9 (ACE(g)/PACE) | — | **2.72** | 3.64 (FreeCG) |
| salicylic acid | 1.1 | **0.36** | 0.5 (ACE(g)/PACE) | 5.4 | **2.03** | 2.7 (ACE(g)) |
| toluene | 0.6 | **0.13** | 0.19 (GotenNet) | 2.7 | **0.78** | 1.04 (FreeCG) |
| uracil | 0.7 | 0.20 | 0.2 (ACE(g)) | 4.4 | **1.43** | 1.7 (ACE(g)) |

Against the best published values RACEv2 wins 7, ties 2 and loses 1 in energy (meV) and wins 8, ties 1 and loses 1
in forces (meV/Å); differences within the rounding digit count as ties. RACE V1 reported 7 molecules; on those
RACEv2 is 64–87 % lower in energy and 58–75 % lower in forces. The reference models were tested on the remaining
~99k frames, our table uses the official 1000 test frames; evaluating on the ~99k frames gives nearly identical values.

### Other benchmarks

| Benchmark | RACEv2 | Reference |
|---|---|---|
| AcAc 300 K / 600 K (RMSE E/F) | 0.46/3.06 · 5.32/15.55 | PACE 0.73/4.2 · 3.8/16.9 |
| Si GAP-18 → RSS (RMSE meV/atom, meV/Å) | 15.65 / 106.7 | XPOT-ACE 20.5 / 139 |
| Si GAP-18 → compressed MD | 4.47 / 69.0 | GAP-18 15.7 / 125 |
| LiPS 19k (MAE meV/atom, meV/Å) | 0.020 / 1.20 | FreeCG force 1.2, GemNet-T 1.3 |
| LiPS learning curve 2500/1000/100/10 (force MAE) | 2.02 / 3.17 / 9.46 / 34.1 | NequIP 4.7, CarNet 5.6, CAMP 21.6 / 76.4 |
| LiPS MD 50 ps × 5 (stability / RDF / Li diffusivity error) | 50 ps / 0.044 / 0.43 | NequIP 50 / 0.04 / 0.34, GemNet-T 0.24 |
| Water split 1 / 2 (RMSE meV/H₂O, meV/Å; rerun with repulsion in progress) | 3.38/63.6 · 0.64/21.96 | MACE 64-0 on the same splits 4.5/64.8 · 0.9/23.2 |

### Ablation (3BPA, 500 epochs, mean of 3 seeds)

One component at a time is reverted to RACE V1. Each cell: refitted energy RMSE (meV/molecule) / force RMSE (meV/Å).

| Variant | 300 K | 600 K | 1200 K | Dihedral | 1200 K force change |
|---|---|---|---|---|---|
| **Final recipe** | **1.87 / 6.70** | **6.52 / 16.63** | **22.89 / 46.42** | 16.74 / **19.28** | reference |
| V1 atomic feature (species embedding) | 2.55 / 9.52 | 9.80 / 22.69 | 29.86 / 61.44 | **15.36** / 19.95 | +32 % |
| No Gaussian radial prior | 2.05 / 7.22 | 7.34 / 18.93 | 28.22 / 60.12 | 28.06 / 23.32 | +30 % |
| No hill (plain RMS) | 2.20 / 7.36 | 7.60 / 19.08 | 26.68 / 56.67 | 21.04 / 20.17 | +22 % |
| No Chebyshev residual | 1.94 / 6.85 | 6.81 / 17.32 | 27.75 / 52.39 | 17.79 / 20.16 | +13 % |
| No LayerScale | 2.27 / 7.24 | 7.19 / 18.27 | 28.15 / 51.77 | 17.74 / 20.30 | +12 % |
| V1 message normalisation (LayerNorm + 1/√⟨N⟩) | 2.12 / 6.81 | 6.78 / 17.00 | 25.73 / 49.41 | 19.44 / 20.18 | +6 % |
| V1 update (gate) | 1.80 / 6.39 | 6.41 / 16.26 | 24.15 / 46.91 | 15.07 / 18.60 | +1 % |

Apart from the gate update, reverting any single component worsens both energies and forces at 600 K and 1200 K, and
the gap grows with temperature. The V1 atomic feature improves only the dihedral energy (15.36 vs 16.74), within the
seed spread of 2.8. Removing LayerScale also raises the seed spread of the 1200 K force error from 0.96 to 3.41.
Reverting to the gate update is equal or better except for the 1200 K energy (+6 %); the linear update shows no gain
in this setting. Removing the Gaussian radial prior gives the worst dihedral scan (28.06 vs 16.74 meV, +68 %) and
the second-largest 1200 K force increase. Remaining variants: no shell channels, no UFF, AMSGrad instead of SOAP.

### OAM subset comparison (RACEv2 vs V1, 2026-11-06)

Before training a universal model (OMat24 pretraining, MPtrj + sAlex fine-tuning), RACEv2 and V1 were compared on
random subsets of each dataset: 10k training, 1k validation, 1k test structures. Both recipes: float64, 100 epochs,
packed batches of about 16 structures, stress in the loss (weight 10), per-element reference energies from a least-
squares fit to the full training split of each dataset. V1 reverts every RACEv2 addition (including AMSGrad).
One seed.

![OAM subset curves](docs/fig_oam_subset_curves.png)

Test set (1,000 structures, checkpoint selected on validation):

| Data | Recipe | Energy MAE (meV/atom) | Force RMSE / MAE (meV/Å) | Stress MAE (GPa, per component) | Training time per epoch |
|---|---|---|---|---|---|
| OMat24 | **RACEv2** | **110** | **490 / 256** | **1.62** | 89 s |
| OMat24 | V1 | 160 | 751 / 353 | 2.02 | 41 s |
| MPtrj | **RACEv2** | **103** | **560 / 142** | **0.96** | 114 s |
| MPtrj | V1 | 131 | 847 / 169 | 1.12 | 59 s |
| sAlex | **RACEv2** | **166** | **306 / 104** | **1.13** | 71 s |
| sAlex | V1 | 178 | 344 / 120 | 1.34 | 30 s |

RACEv2 is better on every metric of all three datasets: energy MAE 7–31 % lower, force RMSE 11–35 % lower, stress
MAE 14–20 % lower. It is 2–2.4× slower per epoch: refreshing the SOAP eigenbases takes 40–60 % of a step
(`runs/diag/soap_profile.py`).
The per-atom energy errors (100–180 meV) are large for both recipes. With 89 elements and 10k structures, the
element-specific weights (89 % of the RACEv2 parameters) see little data.

Notes from the screening:
- SOAP diverges in float32 on these data within 30–80 steps, independently of learning rate, implementation and TF32;
  in float64 (or with a float64 optimizer state) it is stable, so this comparison used float64.
- Stress-loss weight: 10, 100 and 1000 (30 epochs) give nearly the same stress error, while 100 and 1000 make energies
  and forces worse. Stress is learned through the energy and forces; its error is not limited by the loss weight.
  Weight 10 was kept (chosen on validation).
- Speed-ups of the SOAP optimizer and the cutoff (30 epochs, validation; energy MAE meV/atom / force RMSE meV/Å,
  time per epoch):

  | Variant | OMat24 | MPtrj | sAlex |
  |---|---|---|---|
  | RACEv2 (SOAP on every tensor, refresh every 10 steps) | 127 / 528, 93 s | 152 / 337, 118 s | 186 / 365, 76 s |
  | refresh every 50 steps | 135 / 566, 67 s | 156 / 348, 91 s | 202 / 368, 53 s |
  | **species tables by Adam** | 135 / 522, 75 s | 150 / 341, 99 s | 188 / 365, 59 s |
  | both | 146 / 551, 64 s | 152 / 352, 88 s | 179 / 364, 46 s |
  | cutoff 6 Å | 135 / 561, 104 s | 148 / 350, 138 s | 196 / 365, 85 s |

  Adam for the species-indexed tables keeps the force error of full SOAP and is 16–22 % faster; it is used from
  here on. A 6 Å cutoff is not better on these subsets and is 12–17 % slower; 5 Å is kept.

#### Model size on 100k OMat24 structures (in progress)

100k training / 2k validation / 1k test structures, float64, species tables by Adam, 30 epochs. Candidates: L5×64,
L5×128, L5×192, L4×128, L6×128 (layers × channels), plus L5×128 with AMSGrad as an optimizer control. The selection
rule was fixed before the runs (`docs/size_screening_protocol_2026-11-03.md`, validation only): candidates within
+10 % of the best validation energy RMSE, then the lowest validation force RMSE, ties (< 2 %) to the smaller model.
Time per epoch on one GH200: 834 s (L4×128), 1010 s (L5×128), 1196 s (L6×128), 1419 s (L5×192).

Work in progress: 100k size comparison, ablations without shell channels / UFF / SOAP (AMSGrad), last-layer Bayesian
uncertainty, water rerun with repulsion, MD22.

## LAMMPS (`pair_style race`, experimental)

The trained model is exported to StableHLO (`bam_omat24/export_lammps.py`) and run from C++ through the PJRT CUDA
plugin that ships with jaxlib (`lammps/`). Neither Python nor JAX is needed at run time. The cuEquivariance kernels
are recompiled in the C++ process from a record written at export and registered with the plugin (FFI).

```
pair_style race <site-packages>/jax_plugins/xla_cuda12/xla_cuda_plugin.so <export dir> [gpu auto|<index>] [mode auto|owner|halo|ghost]
pair_coeff * * Li P S
```

- `mode owner` (one MPI rank): graph of the owned atoms, ghost neighbours mapped to their owned images. Fastest on
  one GPU.
- `mode halo` (any number of ranks, GPUs and nodes): per-layer halo exchange, as in other message-passing potentials
  with spatial decomposition (e.g. SevenNet). The export contains one function per layer and its vector-Jacobian
  product (`--staged-buckets`). Each rank holds its owned atoms plus one ghost shell (cutoff + skin), with edges for
  owned centres only. The node features are sent to the ghosts with LAMMPS forward communication after every layer.
  In the backward pass, the ghost-row cotangents are returned to their owners with reverse communication (including
  the Chebyshev term that reads layer i−2). Memory per rank scales with the owned atoms.
- `mode ghost`: single evaluation on owned + ghost atoms with the ghost cutoff raised to n_layers × cutoff + skin.
  Simple, but memory grows with the (5 × 5 Å) ghost shell; out of memory above ~1k atoms per rank.
- `mode auto`: owner for one rank, halo for several ranks when the export has the staged functions (ghost otherwise);
  `gpu auto` uses the node-local rank. Units: metal and real.

Validation (LiPS model, cue kernels, float64):

| Test | Result |
|---|---|
| owner, 83 atoms, vs Python | energy 2×10⁻¹³ eV, forces 9×10⁻¹⁴ eV/Å, pressure 10⁻³ bar; NVE drift 0.001 meV/atom per ps |
| owner speed, 1 GH200 | 83 atoms: 7.3 ms per force call; 664 atoms: 30 steps/s; 5,312 atoms: 4.1 steps/s |
| halo on 1, 2, 4 ranks vs owner (664 and 5,312 atoms) | \|ΔE\| ≤ 1×10⁻⁹ eV, max \|ΔF\| 1×10⁻¹⁴ eV/Å, \|ΔP\| ≤ 4×10⁻¹⁰ bar; NVE conserved |

Weak scaling of `mode halo` (5,312 atoms per GPU, replicated LiPS frame; energy per cell and forces of equivalent
atoms agree with the 1-GPU run within 4×10⁻¹² eV and 2×10⁻¹² eV/Å):

| GPUs (nodes) | Atoms | Steps/s | 10³ atom-steps/s | Efficiency vs 1 GPU |
|---|---|---|---|---|
| 1 (1) | 5,312 | 1.24 | 6.6 | 100 % |
| 2 (1) | 10,624 | 0.87 | 9.2 | 70 % |
| 4 (1) | 21,248 | 1.13 | 24.0 | 91 % |
| 8 (2) | 42,496 | 1.09 | 46.3 | 88 % |

On one GPU halo is 3.3× slower than owner (4.1 steps/s). Three costs cause this: the vector-Jacobian products
recompute each layer, the ghost shell doubles the nodes, and there are 12 sequential executions per step. Halo is
for systems that do not fit on one GPU. The 2-GPU point is low because of slower host↔device copies with two ranks;
not yet understood.
Build: `bash lammps/build_lammps.sh <LAMMPS source> <build dir> [mpi|serial]`. Tests: `runs/diag/lammps_multi.py`,
`runs/diag/lammps_weak.py`, `runs/diag/staged_check.py`.

## Installation

```bash
git clone https://github.com/myung-group/BAM_V2.git
cd BAM_V2 && git switch racev2
pip install -e ".[jax]"
pip install cuequivariance-jax cuequivariance-ops-jax-cu12   # for tp_backend="cue"
```

Tested with Python 3.12, jax 0.12, flax 0.12, optax 0.2.8, e3nn-jax 0.21, cuequivariance(-jax) 0.13, ase 3.29.

## Usage

```bash
# training (e.g. the 3BPA final recipe)
python -m bam_omat24.train_3bpa input/input_3bpa_omat24_abl3_full_s42_cue.json
# tests
pytest tests
# LiPS MD (Fu et al. 2023 protocol) and observables
python -m bam_omat24.md_run --help
python -m bam_omat24.md_lips_analysis <md_output_dir>
# regenerate the methods PDF and figures (needs NanumGothic fonts in docs/fonts/, not in the repository)
python docs/make_methods_pdf.py
```

On GPUs always set `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`; it avoids an XLA miscompilation of float64 GEMMs.
Example Slurm scripts for KISTI HANGANG are in `runs/diag/*.sh`.

## Main files

| File | Content |
|---|---|
| `bam_omat24/race_nnx.py` | RACEv2 model (layers, hill, Chebyshev residual, UFF repulsion, cue tensor product) |
| `bam_omat24/train_3bpa.py` | training, evaluation, energy-constant refit |
| `bam_omat24/soap_optax.py`, `soap_grouped.py` | SOAP optimizer |
| `bam_omat24/md_run.py`, `md_lips_analysis.py` | restartable Nosé–Hoover MD, LiPS observables |
| `bam_omat24/export_lammps.py`, `lammps/` | StableHLO export, PJRT C++ runtime, `pair_style race` |
| `input/` | configs of all benchmarks and ablations |
| `docs/` | methods PDF, architecture and data figures with their scripts |

## License

MIT License, see [LICENSE](LICENSE).
