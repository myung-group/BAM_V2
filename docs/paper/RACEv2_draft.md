# Highly locally resolved machine learning interatomic potential: RACEv2

*Working draft, started 2026-11-06. Numbers are taken from `README.md` / `docs/methods_pdf_data.json` (3-seed means,
best-validation checkpoint, energies after the closed-form energy-constant refit unless stated). Reference values must
be checked against `docs/RACE_benchmark_audit_2026-11-03.md` before submission; the cross-check list is
`docs/paper/audit_crosscheck.md`. Markers: **[TODO]** missing content, **[CHECK]** to verify, **[PENDING]** waiting
for runs.*

## Abstract (draft)

Machine-learning interatomic potentials (MLIPs) are mostly trained near equilibrium. When they are applied at higher temperature, they meet
bond lengths, angles and contacts that the training data do not cover. We present RACEv2, an E(3)-equivariant message-passing
potential in which every layer contributes its own energy stratum, so the total energy is a sum of locally resolved
terms of increasing body order. RACEv2 adds five structural elements to RACE:
- an atom-centred multiplicative factor built from the neighbourhood summary;
- a bounded, density-aware magnitude control (hill);
- a Chebyshev-type three-term layer recurrence;
- a Gaussian spectral prior on the radial basis together with element shell-radius channels;
- a fixed-form non-bonded repulsion.

The model is trained with the SOAP optimizer in float64, followed by a closed-form energy-constant refit. Every
model is trained only on the official training split of each benchmark: no pretraining, extra data or augmentation.
- **3BPA (trained at 300 K):** errors are lower than published fair-protocol models at 300, 600 and 1200 K
  (1200 K force RMSE 45.2 meV/Å vs 54.3 TACE and 61.7 GRACE). The dihedral scan remains a limitation.
- **rMD17:** RACEv2 matches or improves the best published values for most molecules.
- **Other benchmarks:** it is competitive on acetylacetone, Si GAP-18, LiPS (including molecular-dynamics observables)
  and liquid water.

A leave-one-out ablation shows that the gains come from the model structure, not from tuning. The elements that act
on radial smoothness and feature magnitude have effects that grow with temperature, which points to an
extrapolation mechanism rather than more capacity. We also provide an exportable C++/LAMMPS runtime with per-layer
halo exchange that scales to multiple GPUs and nodes. **[TODO: OAM-scale result, uncertainty result if positive.]**

## 1. Introduction **[TODO: write]**

Points to make:
1. Data efficiency and transferability in temperature are the practical bottleneck for molecular MLIPs. 3BPA
   (train at 300 K, test at 600 K, 1200 K and on a dihedral scan) is the standard probe.
2. Equivariant MPNNs (NequIP, MACE, Allegro) and ACE-type models (ACE(g)/GRACE, PACE) differ in how they build
   body order: MACE uses in-layer symmetric products, ACE uses explicit cluster bases, and RACE builds body order
   across layers through stratum readouts.
3. Our thesis: when every layer is a separate, *locally resolved* stratum, the extrapolation behaviour of the
   expansion can be controlled layer by layer. The elements we add are ways to keep the higher strata smooth and
   bounded outside the training distribution.
4. Contributions: the model (Sec. 2), the training method with SOAP and the energy-gauge analysis (Sec. 3), results
   on seven benchmarks (Sec. 4), the mechanism analysis by ablation (Sec. 5), and the deployable runtime (Sec. 6).

## 2. Model

Notation: $h_i^{(t)}$ are the features of atom $i$ after layer $t$ ($t = 1..L$, $L = 5$). Each feature carries
irreps up to $l_{\max} = 3$ with 128 channels per $l$ (256 for rMD17 and MD22, 64 for water). $\mathbf r_{ij}$ is the
edge vector for $r_{ij} < r_c = 5$ Å, $Y(\hat r_{ij})$ the spherical harmonics, and $\langle N\rangle$ the mean
neighbour count of the training set. Lin is an equivariant linear map, $\otimes$ the Clebsch–Gordan tensor product,
and $\odot$ a channel-wise product.

### 2.1 Stratified energy

$$E = s\sum_i\Big(\sum_{t=1}^{L} E_i^{(t)} + E_i^{\mathrm{rep}}\Big) + \sum_i \big(\mu + E_0(Z_i)\big),\qquad
E_i^{(t)} = \mathrm{Lin}\big(\mathrm{Gate}(\mathrm{Lin}(h_i^{(t)}))\big)\ \ (\text{scalar}).$$

Each layer $t$ adds one stratum. The gated readout is the only non-linearity. The layer update (2.3) is linear, so
the body order of $E^{(t)}$ grows with $t$. $s$ is one learned global scale, $\mu$ a shift, and $E_0$ the
isolated-atom (or least-squares) reference energies. Forces are $-\nabla E$ (conservative); the stress comes from
the strain derivative.

### 2.2 Messages and the atom-centred factor

$$m_i^{(t)} = \frac{1}{\langle N\rangle}\sum_{j} R_t(r_{ij})\; Y(\hat r_{ij}) \otimes h_j^{(t-1)},\qquad
A_i = \frac{1}{\langle N\rangle}\sum_j R_0(r_{ij})\,Y(\hat r_{ij})\otimes \mathbf 1_{Z_j},$$

$$x_i^{(t)} = \mathrm{Lin}_t(A_i),\qquad \hat x_i^{(t)} = \frac{x_i^{(t)}}{\rho_i}\,
2\sigma\big(a_{t,Z_i}\log\rho_i + b_{t,Z_i}\big),\quad \rho_i = \mathrm{RMS}(x_i^{(t)}),$$

$$\delta_i^{(t)} = \mathrm{Lin}\big(m_i^{(t)}\otimes \hat x_i^{(t)}\big) + \mathrm{Lin}_{\mathrm{self}}(h_i^{(t-1)}).$$

In RACE V1, $x$ was a per-species constant $a(Z_i)$. In RACEv2 it is a layer-specific linear projection of the
one-shot neighbourhood summary $A_i$, so each product $m\otimes\hat x$ raises the body order by one with a
neighbourhood-dependent coefficient. The *hill* factor $2\sigma(a\log\rho+b)$ normalises the magnitude but keeps a
bounded (≤ 2) amount of the density information $\rho_i$. Here $a$ and $b$ are learned per layer and element.
$a = b = 0$ reduces to plain RMS normalisation, which is the "nohill" ablation.

### 2.3 Chebyshev-type layer recurrence

$$h_i^{(t)} = 2\,\gamma_t\odot\delta_i^{(t)} - h_i^{(t-2)}\qquad (\gamma_t\ \text{LayerScale, initialised at }0.5).$$

This is the three-term recurrence of Chebyshev polynomials, $T_{n+1} = 2xT_n - T_{n-1}$. With plain stacking the
layer-$n$ features behave like powers $x^n$, which become more and more alike and grow without bound outside
$[-1, 1]$. The recurrence instead makes successive layers behave like orthogonal polynomials, which stay bounded on
the interval and are less correlated across layers. **[CHECK: state precisely what is proven and what is analogy;
the map is not a scalar polynomial.]**

### 2.4 Radial basis: spectral prior and shell channels

The radial weights $R_t$ are MLPs of a Bessel basis $b_n(r) = \frac{2}{r_c}\frac{\sin(n\pi r/r_c)}{r}\,f_{\mathrm{cut}}(r)$
($n = 1..8$, polynomial cutoff $p = 2$). Two changes:
- **Gaussian spectral prior.** $b_n \to b_n\,e^{-\sigma^2 n^2}$ with $\sigma = 0.224$. High-frequency radial
  components are damped before learning. This is a fixed smoothness prior: in Fourier terms, a Gaussian-process prior
  on the radial functions.
- **Shell-radius channels.** For each element pair $(a, b)$:
  $S_{ab}(r) = \exp\!\big(-r/(\rho_a+\rho_b)\big)\,f_{\mathrm{cut}}(r)$, where $\rho$ are tabulated atomic shell radii
  (`bam_omat24/atomic_shells.json`). This gives an element-aware length scale that does not have to be learned.

### 2.5 Non-bonded repulsion prior

$$E^{\mathrm{rep}} = \sum_{i<j} D_{ij}\frac{6}{\zeta_{ij}-6}\,e^{\zeta_{ij}(1-r_{ij}/x_{ij})}\;
f_{\mathrm{cut}}(r_{ij})\;\sigma\!\Big(\frac{r_{ij}-c(R_i+R_j)}{w}\Big),$$

This is the repulsive part of the UFF exp-6 potential (Rappé 1992; Open Babel parameters for $Z = 1..102$).
- The sigmoid switch removes it for bonded pairs: $R$ are covalent radii, $c = 1.25$, $w = 0.3$ Å.
- It is added to the per-atom energies before the global scale $s$. The functional form is fixed, but the amplitude
  is effectively $s \approx 0.25$ (learned), so it is not parameter-free.
- It acts in the region the training data never visit: contacts more than 0.3 Å closer than the closest pair in the
  3BPA training set.

## 3. Training

- **Loss:** Huber on per-atom energies and force components (δ = 10 in internal units **[CHECK units]**); stress
  for periodic data (weight 10 on eV/Å³ for OAM).
- **Precision and averaging:** float64; EMA of the weights with decay 0.99.
- **Checkpoint:** best validation loss.
- **Optimizer:** SOAP (Vyas et al. 2024; Shampoo with Adam in the preconditioner eigenbasis), lr 1e-2, β = 0.95.
  Gradients are clipped at 1.0.
  - Grouped implementation: same mathematics, 17.7 → 6.8 s per epoch on 3BPA.
  - For many-element data (OAM), the species-indexed tables use Adam instead of SOAP: equal accuracy, 16–22 % faster.
  - SOAP diverges in float32 on OAM data; this is why we train in float64 (Sec. 7).
- **Energy-constant refit.** For a single stoichiometry, a constant energy per atom is not fixed by the forces. With
  sign-like optimizer updates, such an offset survives training (−6 meV at 500 epochs on 3BPA, gone by 1500 epochs).
  After training, the shift is refitted in closed form on the training set (one linear least-squares problem,
  Golub–Pereyra separable form). Raw energies are reported alongside.
- **Fair protocol:** official training split only; no pretraining, extra data, augmentation or ensembles.
- **Size per dataset:** chosen by a pre-registered rule on validation only (`docs/size_screening_protocol_2026-11-03.md`).

## 4. Results

### 4.1 3BPA (RMSE; E meV/molecule, F meV/Å; trained on 500 frames at 300 K)

| Model | 300 K | 600 K | 1200 K | Dihedral |
|---|---|---|---|---|
| **RACEv2** | **1.56 / 6.31** | **6.05 / 15.92** | **22.20 / 45.16** | 16.67 / 19.09 |
| RACE V1 | 3.4 / 12.1 | 11.7 / 31.8 | 37.5 / 115.3 | — |
| GRACE (PRX 2024) | 1.7 / 7.6 | 7.3 / 19.2 | 25.8 / 61.7 | **7.4 / 15.3** |
| PACE (TMLR 2024) | 2.4 / 9.1 | 7.9 / 21.4 | 29.6 / 60.7 | 7.6 / 16.0 |
| TACE | 2.69 / 8.68 | 8.10 / 19.88 | 23.42 / 54.26 | 11.0 / 16.45 |
| MACE | 3.0 / 8.8 | 9.7 / 21.8 | 29.8 / 62.0 | 7.8 / 16.5 |

**[CHECK** against the audit: TACE is single-seed; include ICTP and SpinGTP rows; the NequIP row mixes two
implementations.**]** RACEv2 is best on all six temperature metrics. Its dihedral energy (16.7) is about twice the
best value (7.4); see Sec. 5.3 and the limitations.

### 4.2 rMD17 (MAE, official split 01, 1000 training frames, L5h256)

Use the README table. RACEv2 wins 7, ties 2 and loses 1 in energy, and wins 8, ties 1 and loses 1 in forces against the
best published value per molecule. **[CHECK:** the references are tested on ~99k frames and ours on the official 1000;
our 99k evaluation is nearly identical, give both. Raw vs refitted energies: the refit changes benzene by 7×
(audit) — report raw.**]**

### 4.3 Other benchmarks

| Benchmark | RACEv2 | Reference |
|---|---|---|
| AcAc 300 K / 600 K (RMSE E/F) | 0.46/3.06 · 5.32/15.55 | PACE 0.73/4.2 · 3.8/16.9 |
| Si GAP-18 → RSS (meV/atom, meV/Å) | 15.65 / 106.7 | XPOT-ACE 20.5 / 139 |
| Si GAP-18 → compressed MD | 4.47 / 69.0 | GAP-18 15.7 / 125 (our re-evaluation) |
| LiPS 19k (MAE meV/atom, meV/Å) | 0.020 / 1.20 | FreeCG 1.2, GemNet-T 1.3 |
| LiPS learning curve 2500/1000/100/10 (F MAE) | 2.02 / 3.17 / 9.46 / 34.1 | NequIP 4.7 (2500), CarNet 5.6 (1000), CAMP 21.6 / 76.4 (100/10) |
| LiPS MD 50 ps × 5 (stability, RDF MAE, Li D error) | 50 ps, 0.044, 0.43 | NequIP 50 ps, 0.04, 0.34; GemNet-T 0.24 |
| Water split 1 / 2 (RMSE meV/H₂O, meV/Å) | 3.38/63.6 · 0.64/21.96 **[PENDING rerun with repulsion]** | MACE 64-0 retrained on the same splits 4.5/64.8 · 0.9/23.2 |

AcAc 600 K energy is worse than PACE. The Si numbers are now 3 seeds **[CHECK the audit note: earlier 1 seed and
AMSGrad flipped the ranking]**. For water, only the same-split MACE comparison is a direct comparison (the
literature values use other splits or are validation/training errors, see the audit).

### 4.4 Large-scale data (subset screening) **[PENDING full OAM]**

On 10k-structure subsets of OMat24, MPtrj and sAlex (float64, 100 epochs, 1 seed), RACEv2 beats the V1 recipe on
every metric: energy MAE 7–31 % lower, force RMSE 11–35 % lower, stress MAE 14–20 % lower. It costs 2–2.4× more per
epoch. Model size on 100k OMat24 structures is being screened under a pre-registered rule. This section stays a
pointer to future work unless full OAM training is done (the title claim is "locally resolved", not universal).

## 5. Why it extrapolates: ablation analysis

### 5.1 Leave-one-out table (3BPA, 500 epochs, 3 seeds; pre-registered protocol `docs/ablation_protocol_2026-11-05.md`)

Relative change of the force RMSE vs the full recipe at the same budget (from the README table):

| Removed element | 300 K F | 600 K F | 1200 K F | Dihedral E |
|---|---|---|---|---|
| atom-centred factor → species constant (xspecies) | +42 % | +36 % | +32 % | −8 % |
| Gaussian radial prior (noprior) | +8 % | +14 % | +30 % | **+68 %** |
| hill → plain RMS (nohill) | +10 % | +15 % | +22 % | +26 % |
| Chebyshev recurrence (nocheb) | +2 % | +4 % | +13 % | +6 % |
| LayerScale (nols) | +8 % | +10 % | +12 % | +6 % |
| message normalisation → V1 (msgnorm) | +2 % | +2 % | +6 % | +16 % |
| linear update → gate (gate) | −5 % | −2 % | +1 % | −10 % |
| shell channels / UFF / SOAP→AMSGrad | **[PENDING]** | | | |

### 5.2 Two kinds of contribution (interpretation; to be tested, see 5.4)

- **Capacity (temperature-independent).** The atom-centred factor (xspecies) gives the largest gain at every
  temperature, and its relative effect is about the same or decreasing with T (+42 → +32 %). It makes the
  coefficient of each body-order step depend on the local environment. This is more expressive at every
  temperature, not specific to extrapolation.
- **Extrapolation control (grows with temperature).** For the radial prior, hill and the Chebyshev recurrence, the
  relative effect grows steadily from 300 to 1200 K: prior +8 → +14 → +30 %, hill +10 → +15 → +22 %, Chebyshev
  +2 → +4 → +13 %. All three act on how the strata behave on inputs outside the training range:
  1. *Radial prior.* At 1200 K, bonds visit distances that the 300 K data do not cover. Bessel components with large
     $n$ fitted on a narrow distance window oscillate outside it. Damping them by $e^{-\sigma^2 n^2}$ amounts to a
     smoothness (GP-type) prior that makes the radial functions extrapolate with low curvature. The dihedral scan
     contains contacts 0.3 Å inside the training minimum distance. That is the most out-of-range input, and the prior
     helps it most (+68 % without it), consistent with the collision analysis in the methods PDF (energy bias of the
     collision frames −45 meV without a short-range prior).
  2. *Hill.* With plain RMS normalisation, $\hat x$ has unit magnitude whatever the environment, so the density
     information is lost. Without any normalisation, $\hat x$ grows with the coordination number, and the higher
     strata (products of $L$ factors) grow like $\rho^{L}$ in dense, unusual environments. Hill keeps a bounded,
     monotone dependence on $\rho$, so each stratum is Lipschitz-bounded in the density.
  3. *Chebyshev recurrence.* With $h^{(t)} = \gamma\delta^{(t)}$, higher strata are close to repeated powers of one
     operator, whose deviations amplify geometrically off-distribution. The three-term recurrence keeps successive
     strata nearly orthogonal and bounded (the $T_n$ analogy). Its effect is small in-distribution (+2 % at 300 K)
     and appears only under extrapolation, which is what this mechanism predicts.
- **Neutral.** The linear update shows no gain over the V1 gate at this budget. We keep it only for the
  interpretability of the strata (fixed body order per layer); the paper should say so plainly.

### 5.3 Dihedral limitation

The dihedral scan is the one 3BPA metric where RACEv2 is not best (16.7 vs 7.4 meV). Lowering the learning rate
improves it (lr 3e-3: 8.1/17.4) but costs 1200 K accuracy (28.0/52.5). This trade-off and the post-hoc choice of
lr 1e-2 (the pre-registered rule picks 3e-3) must be stated. The choice of UFF-nb vs ZBL-nb was also made after
seeing test metrics; report both. **[TODO: table from methods_pdf_data.json final_recipe.selection_note]**

### 5.4 Tests of the mechanism (to run when compute is available)

1. Radial-function curvature and its out-of-window growth, with and without the prior; the error vs the minimum
   interatomic distance per frame (1200 K and the dihedral scan).
2. Stratum magnitudes $|E^{(t)}|$ vs temperature and vs coordination density, for full / nohill / nocheb.
   Prediction: without hill or Chebyshev, the upper strata grow faster with T.
3. Correlation of successive-layer features (CKA) for full vs nocheb.
These turn the interpretation of 5.2 into measured statements, as suggested by `feedback-fancy-methodology` (the
user prefers principled explanations to recipe tables).

## 6. Deployment: LAMMPS with per-layer halo exchange

- **Export:** the trained JAX model is exported to StableHLO and run from C++ through the PJRT CUDA plugin; the
  cuEquivariance kernels are re-registered in the C++ process.
- **owner mode:** one GPU; matches Python to 1e-13.
- **halo mode:** the per-layer spatial decomposition known from SevenNet (Park et al., JCTC 2024), applied here to
  exported JAX functions, including the two-layer Chebyshev dependency.
  - Matches owner mode to 1e-14 eV/Å.
  - Weak scaling at 88 % from 1 to 8 GPUs on 2 nodes (42,496 atoms).
- This section supports usability; it is not a main contribution. **[TODO: after the halo speed-ups (owned rows only,
  saved residuals), replace the timings.]**

## 7. Limitations

- Dihedral scan (5.3).
- Water: no gain from SOAP; the repulsion rerun is pending.
- Large-scale data are only screened on subsets.
- SOAP in float32 is unstable on OAM-type data, so training needs float64.
- The 500-epoch ablation orderings may differ from converged ones.
- Test sets were monitored during development (`eval_test_every`); state this in the paper.

## 8. Reporting statement (to include in the paper)

- Checkpoint: best validation (EMA). Energies: after the closed-form refit, with raw values also given.
  3-seed mean ± std, and the best-validation seed for comparison with single-run literature values.
- Units per table: RMSE or MAE as in the original benchmark.
- Disclosure: test metrics were logged during development; the final recipe choices (lr, repulsion form) were made
  after seeing them, as stated in 5.3.

## Appendix A. Negative results **[TODO: one table: faithful DeNS, bond-order stratum, monotone resolvent depth,
Aitken resummation, energy weights under SOAP, in-layer symmetric products (BRACE), others]**
