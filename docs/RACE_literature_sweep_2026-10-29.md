<!-- Literature sweep (workflow race-literature-sweep, 2026-10-29): 5 domains, 37 candidates, 13 kept after adversarial citation / fit verification. Synthesis as returned; gate results D1/D3 measured afterwards are in RACE_omat24_review.md §12.6. Candidate lists: lit_kept.json / lit_dropped.json below. -->
# RACE plan: methods from the literature for 3BPA (2026-10-29)

## 0. What the given numbers already say

Here is how much the energy RMSE grows from 600 K to 1200 K for each model:

| model | E600 | E1200 | E1200/E600 |
|---|---|---|---|
| TACE_max | 8.10 | 23.42 | 2.89 |
| MACE | 9.70 | 29.80 | 3.07 |
| GRACE | 7.30 | 25.80 | 3.53 |
| v9a | 7.84 | 28.17 | 3.59 |
| tied+hill | 7.51 | 30.99 | 4.13 |

**Harmonic error model.** Take a pocket Hessian H and a model curvature error ΔH. Define the relative error B = H^-1/2 ΔH H^-1/2, with eigenvalues ε_k. Thermal displacements are q ~ N(0, k_B T H^-1), and the energy offset is absorbed at T₀ = 300 K. Then:
- bias: μ_T = (k_B/2)(T − T₀) tr B
- standard deviation: σ_T = (k_B T/√2) ‖B‖_F
- so RMSE_1200/RMSE_600 lies in [2, 3]: 2 if the error is pure variance, 3 if it is pure bias.

This is a heuristic; gate D1 checks it. What it suggests:
1. **TACE is inside the harmonic band and RACE is not.** At 600 K tied+hill already beats TACE (7.51 vs 8.10). The whole 1200 K deficit comes from growth faster than the harmonic model allows between 600 and 1200 K.
2. **H2 (saturating magnitude).** Going from v9a to tied+hill bounded the operator. That raised the ratio from 3.59 to 4.13 while 1200 K forces got better. This fits an energy response to magnitude that saturates at large distortion. It does not fit contraction, which was already ruled out.
3. **H1 (soft-mode curvature bias).** This is the mechanism behind idea 1A. It is real, but it can only remove the harmonic part of the error. A realistic ceiling is −10 to −15 % on 1200 K E.
4. **H3 (truncation).** The tied resolvent is cut at T = 5. With ρ ≤ 0.95, the remaining tail is about ρ⁵ ≈ 0.77, which is not small. Gate D4 tests this.

Tier 1 therefore has one idea for H1 (the loss), one for H2 (the architecture), and one paper-level operator method (1C). Each has a check that needs no training.

---

## 1. Merged candidate map

| Merged idea | Source candidates (domain) | Support from other fields | Tier |
|---|---|---|---|
| **1A Dual-norm (H⁻¹) force metric** | soft-mode-hinv-force-loss (mlip) + dual-norm-force-loss (ai_opt). The dropped thermal-metric-force-loss (phys) is the same object; its objections are answered in 1A | Spectral bias under Sobolev norms (Yu et al.), optimisation preconditioners (Packwood), harmonic force/energy analysis (Bigi et al.), curvature softening (Deng et al.) | 1 |
| **1B Shape/scale homogeneous bond-order energy** | bop-shape-scale-readout (phys). Absorbs intlip-logdet arm (a), the log-magnitude readout, as an ablation arm, and the Finnis–Sinclair arm of rational-readouts | Moments theorem / bond-order potentials, climate-invariant rescaling (Beucler), Finnis–Sinclair embeddings (PACE) | 1 |
| **1C Monotone variational resolvent** | mondeq-exact-resolvent (mlip) + monotone-implicit-resolvent (ai_dyn). pole-expansion step 1 (phys) becomes its pre-test | Monotone operator splitting, polarisable force fields / Thole damping, DEQ, charge-equilibration solves | 1 (paper-level) |
| **2A SOAP (Kronecker-whitened Adam)** | soap-muon-optimizer (mlip) + soap-preconditioning (ai_opt) | Shampoo/SOAP, gradient alignment in PINNs | 2 |
| **2B Per-frame spectral-norm hinge** | jacobian-reg-operator (ai_dyn) | DEQ Jacobian regularisation, spectral-norm regularisation | 2 |
| 3a Rational / Padé readouts | rational-readouts (math) | Rational approximation theory | 3 |
| 3b Scalar SPD spectral energy channels (QEq resolvent, collective log-det) | variational-spd-resolvent-energy (math) + intlip-logdet arm (b) (math) | QEq / latent Ewald, MBD log-det, graph-filter stability | 3 (cheap prototypes of the 1C family) |
| 3c Multi-pole rational resolvent | pole-expansion-resolvent (phys), beyond step 1 | Pole expansions in electronic structure, ARMA/Cayley GNN filters | 3 |
| 3d Latent tight-binding band energy | latent-tb-band-energy (kept by mlip, dropped by phys) | DFTB layer, semi-empirical HIP-NN | 3 |

---

## 2. Checks that need no training (run first, on existing checkpoints)

| Gate | What to compute (frozen weights) | Decides | Pass criterion |
|---|---|---|---|
| **D1** | Mean signed error μ_T and std σ_T of the energy error at 300/600/1200 K, for v9a and tied+hill, 3 seeds each. Also the fraction of the error shared across seeds | Relative weight of H1 vs H2; prior for 1A | H1 predicts μ_1200/μ_600 ≈ 3 and μ²_1200 ≥ 0.3·RMSE² |
| **D2** | Model Hessian H_n on 600/1200 K frames (81 forward-over-reverse Hessian-vector products, vmapped, float64). Compute d_n = ΔF_nᵀ G_n ΔF_n (G as in 1A) and the plain ‖ΔF_n‖². Rank-correlate each with (ΔE_n − μ_T)² | Go/no-go for 1A | d_n is clearly more predictive than ‖ΔF_n‖² (Spearman gap ≥ 0.1) |
| D2b | For each 1200 K frame B, take the nearest Kabsch-aligned 300 K training frame A. Trapezoid estimate: ΔE_B − ΔE_A ≈ −½(ΔF_A + ΔF_B)·(R_B − R_A). Split it onto the 15 softest eigenvectors of H_A and the rest | Mechanism of 1A (crude: paths are long) | The soft-mode share dominates |
| **D3** | Regress the signed 1200 K energy error on Σ_i log rms(x_i) (before normalisation, tied+hill), on Σ_i log Σ_j e^(−2 r_ij), and on Σ bond strain | Go/no-go for 1B | Partial R² of the scale terms beyond strain ≥ 0.1 |
| **D4** | On tied+hill, solve h* = (I − M)^-1 u₀ per frame with GMRES. Compute r_n = ‖h₅ − h*‖/‖h*‖ and E(h*) − E(h₅) (readout r₅ applied to h*). Correlate with the energy error at each T | Whether 1C can claim anything for gap (i); whether 3c is worth it | r is larger at 1200 K and corr(r_n, \|ΔE_n\|) ≥ 0.3 |
| **D5** | On v9a, per frame: σ_max(M_t(R)) by power iteration on MᵀM (Mᵀ via jax.vjp), location of the dominant eigenvalue (20-step Arnoldi), and which atoms it sits on. Do this for 3BPA 300/1200 K and the rMD17 molecules that blow up | Sets c for 2B; prior for 1C | If v9a's dominant eigenvalues are real, positive and > 1, the monotone class (Re λ(M) ≤ 1 − m) excludes v9a's regime, which lowers 1C's prior for gap (i) |
| D6 | Fraction of 1200 K stratum-readout pre-activations that fall outside the 300 K [min, max] | 3a | ≥ 1 % |
| D7 | cos(g_E, g_F) and the Wang alignment score every 100 steps (comes free with the 2A runs). This replaces the dropped ConFIG line | 2A, combination C | Logged only |
| D8 | Regress the 1200 K energy error on ring-centroid distance and ring-plane angle | Whether to revive QEq/LES as an electrostatics term | R² > 0.2 |

---

## 3. Tier 1

### 1A. Dual-norm (H⁻¹-metric) force loss: a variance-reduced estimator of the T* energy error

**Mechanism (math).** In the harmonic pocket above:
- Plain force MSE: E‖ΔF‖² = k_B T tr(B² H) = k_B T Σ_k ε_k² κ_k. It is weighted by stiffness κ, so stiff modes dominate.
- Energy error: bias (k_B T/2) tr B, variance (k_B T)²/2 · ‖B‖_F². Both are uniform across modes and grow linearly with T.
- Dual norm: E[ΔFᵀ H⁻¹ ΔF] = k_B T ‖B‖_F². This is exactly the functional that controls the energy variance at any temperature. It also bounds the bias, since (tr B)² ≤ n‖B‖_F².
- Local-cloud identity. For u ~ N(0, k_B T* H⁻¹) around a training frame, E_u[ΔE(R+u)²] = ΔE(R)² + k_B T* ΔFᵀ H⁻¹ ΔF, plus terms in ΔH that E/F labels cannot identify.

So L_E + k_B T* L_dual is the expected squared energy error over a T*-thermal cloud around every training frame, written in closed form from existing labels.

Two readings:
- **Rao–Blackwell reading.** The energy label gives one noisy sample of this error functional per frame. The dual term is its conditional expectation, built from the 3N − 6 = 75 force components. That is how energy-like information can enter through forces, whereas the round-12 energy-weight boosts destabilised training.
- **Spectral-bias reading.** Yu et al. show that a Sobolev H^s loss with s < 0 speeds up convergence of low-frequency components. Here the frequency is the vibrational ω_k = √κ_k, and s = −1 means weighting by 1/κ_k. This is an analogy: their theorem is about Fourier modes on the input domain.

Illustration. Take about 15 soft modes at λ ≈ 0.05 eV/Å² with about 1 meV/Å residuals, and about 60 stiff modes at about 30 eV/Å² with about 8 meV/Å residuals:
- In ‖ΔF‖², soft modes contribute about 0.4 %.
- In ΔFᵀGΔF, soft modes contribute about 70 %.
- k_B T* ΔFᵀGΔF ≈ 0.103 eV × 4.3·10⁻⁴ eV ≈ (6.7 meV)². That is the same order as ΔE² ≈ (2.4 meV)², so the term neither swamps the loss nor vanishes.

**Why this differs from the closed Taylor work (round 19/20).** taylor_aug fitted E_θ(R+d) to the label E − F·d. Since E_θ(R+d) = E_θ − F_θ·d + ½ dᵀH_θ d, that loss penalises the model's whole curvature ½ dᵀH_θ d. It pushes the PES to be systematically softer, which is the error class Deng et al. report for uMLIPs and a plausible reason round 19 was negative. With isotropic d, the expected loss is also just more plain L2 force weight. The dual term has no pseudo-label, no curvature bias, and a metric shaped by the Hessian. The objection in the dropped phys entry, that stiff modes get down-weighted where the 1200 K force error lives, is avoided by variant P1 below, which leaves the plain force term unchanged.

**RACE design.** The model, operator and readouts are untouched; only the loss changes.
- **Metric G1 (preferred, refreshed within the same run).** From epoch 200, and every 300 epochs after that, compute for each of the 500 training frames:
  - H_n = −∂F_θ/∂R at R_n (81 forward-over-reverse products, vmapped, float64), symmetrised, with the 6 rigid-body modes projected out by P_⊥.
  - Eigendecompose H_n = VΛVᵀ and set G_n = P_⊥ V diag(1/max(|λ_k|, λ_f)) Vᵀ P_⊥, with λ_f between 0.02 and 0.05 eV/Å² (torsion scale).
  - Store L_n = G_n^1/2 under stop_gradient (500 × 81 × 81 in float64 is about 26 MB).
  - G_n is mass-unweighted, which is correct for the configurational distribution. It transforms as R G Rᵀ, so the loss stays rotation-invariant. Before the first refresh, γ = 0.
- **Metric G2 (data only).** Generalised equipartition, ⟨x_i ∂_j U⟩ = k_B T δ_ij, gives Cov(δx) ≈ k_B T H⁻¹ in the harmonic limit. Cluster the training frames by the 3 backbone dihedrals. Kabsch-align each frame to its basin mean, project out rigid modes, and apply Ledoit–Wolf shrinkage to get Σ_b. Then G_n = R_n Σ_b R_nᵀ / (k_B T_train).
- **Metric G0 (negative control).** The Packwood "Exp" preconditioner: an isotropic bond-graph Laplacian ⊗ I₃ plus μ C_stab. Its soft eigenvectors are collective translations, not torsions. If G0 gives the same gain as G1, the soft-mode mechanism is falsified.
- **Loss, variant P1 (energy side; the stiff-bond tail is protected):**
  L = Σ_n w_E [ ΔE_n² + γ k_B T* ‖L_n ΔF_n‖² ] + w_F ‖ΔF_n‖²/(3N), with T* = 1200 K (k_B T* = 0.103 eV).
  - γ = 1 is the first-order local-cloud value.
  - γ = 2 matches the harmonic variance at T* from frames at T_train: c = (k_B T*)²/(2 k_B T_train) = 2 k_B T*.
  - Screen γ ∈ {1, 2, 4}.
- **Loss, variant P2 (force side):** w_F [(1 − α)‖ΔF‖² + α c ‖LΔF‖²], with α ∈ {0.2, 0.5} and c chosen so both terms are equal at initialisation. This is riskier because it down-weights stiff residuals by about 1/30.
- **Logging:** force RMSE projected on the 15 softest and 15 stiffest eigenvectors at 300 and 1200 K; μ_T; force bins for atoms with strain > 0.1 Å.
- **Protocol:** no new labels or frames; G is a metric built from the same run.

**3BPA test.** Base is tied+hill (2.42/8.11, 7.51/18.89, 30.99/55.58), with a v9a replica. Screen at 500 epochs with 1 seed, then 1500 epochs with 3 seeds.

Predictions:
- 1200 K E ≤ 27.5 meV (a gain of at least 3.5 meV, about 1.7 times the single-seed sd)
- |μ_1200| down by at least 30 %
- 600 K E ≤ 7.3
- 300 K F ≤ 8.4
- 1200 K F ≤ 56.5
- strained-bond force bins no more than 5 % worse

Kill if any of these holds:
- D2 shows no predictive gain for d_n.
- The soft-mode-projected 1200 K force RMSE drops by at least 20 % while 1200 K E moves by less than 1 meV. The offset would then not be an integrated soft-mode error.
- G0 matches G1.

**Cost:** S. About +10 % wall time for the refreshes, plus one batched (B, 81, 81) matmul.

**MLIP novelty:** new as a loss. I found no MLIP with an inverse-Hessian or dual-norm force metric.
- ModeAug (Kwak & Jo, arXiv 2609.05233) uses the same Hessian-shaped displacement covariance, but as augmentation with Taylor labels from DFT Hessians, which we do not have and may not use.
- Bigi et al. diagnose the problem only: in App. A, water intramolecular forces are 3.93 eV/Å against a 0.97 eV/Å total RMS.
- Deng et al. report systematic curvature softening.
- ASE uses the Packwood preconditioner for geometry optimisation only.

**Citations:**
- Yu, Yang, Townsend, ICLR 2023, https://arxiv.org/abs/2205.14300
- Packwood et al., JCP 144, 164109 (2016), https://arxiv.org/abs/1604.02570
- Bigi, Langer, Ceriotti, ICML 2025, https://arxiv.org/abs/2412.11569
- Deng et al., https://arxiv.org/abs/2405.07105
- Kwak & Jo, https://arxiv.org/abs/2609.05233
- Makri, Ortner, Kermode, JCP 150, 094109 (2019), https://arxiv.org/abs/1810.02705

### 1B. Shape/scale factorisation: a homogeneous bond-order energy on a normalised operator

**Mechanism (math).**
- **Moments theorem.** μ_n,i = ⟨i|Hⁿ|i⟩ sums closed hopping paths of length n.
- **Scaling.** If all hoppings scale as β → λβ, then μ_n → λⁿ μ_n and E_bond → λ E_bond, so the bond energy is homogeneous of degree 1.
- **BOP4S.** The energy depends on a scale b₁ = √μ₂ and a scale-free shape parameter (b₂/b₁)², which separates unimodal from bimodal spectra.
- **General form.** E_i = −b_i G(μ₃/b_i³, μ₄/b_i⁴, …) plus repulsion. The Finnis–Sinclair limit, G = const, gives E_i = −A√μ₂,i.

RACE reading. The strata of a normalised operator act as shape variables. They are bounded and should shift little between 300 and 1200 K. The magnitude rms(x_i) acts as the scale. Today:
- v9a compounds magnitude multiplicatively over 5 layers, which drives the rMD17 blow-up.
- hill passes magnitude through a bounded sigmoid in every layer, so it saturates at large distortion. This matches the ratio rising from 3.59 to 4.13 in §0.

1B lets the magnitude enter once, with a physical power law.

**RACE design.**
- **Operator:** x̂_i = x_i/rms(x_i), with x_norm on and the hill gate fixed to 1.
- **Width:** b_i = (ε + Σ_j β_{Z_iZ_j}(r_ij)²)^1/2, with β_{ZZ'}(r) = Σ_{k=1..3} softplus(c_k^{ZZ'}) e^(−α_k r) f_env(r). α_k ∈ {1, 2, 4} Å⁻¹ is fixed, and f_env is the existing shell envelope. β is positive and decays monotonically, so extrapolation into strained bonds is controlled. Bessel functions are not used here.
- **Optional degree-0 messages:** m_i(h) = (b̄/b_i) Σ_j R(r_ij) Y(r̂_ij) ⊗ h_j, replacing 1/⟨N⟩. Without this, the messages still carry radial magnitude and the strata are not truly scale-free.
- **Energy:**
  E = Σ_t Σ_i r_t(h_t,i) − Σ_i s_{Z_i} b_i^p [1 + g_θ(q_i)]
  - The first sum is the existing strata, with x_readout_norm kept (it was load-bearing in round 21).
  - q_i = ⊕_t 0e(h_t,i) are the shape invariants.
  - p = 0.5 + min(softplus(π), 1).
  - g_θ is a small gated-SiLU MLP whose last layer is initialised to zero, so the term starts as Finnis–Sinclair.
  - s_Z starts at about 0.1 of the per-species energy scale. Its sign is free; physically one expects s_Z > 0.
- **Arms:**
  - (a) hill (control)
  - (b) x_norm + b^p, with p learned
  - (c) p fixed at 1 (degree 1, as in TB)
  - (d) log arm, s log b_i. This is integral-Lipschitz with λh′ = 1, as in Gama et al. It is unbounded, unlike hill.
  - (e) arm (b) plus degree-0 messages
- No φ_rep: Born–Mayer repulsion was neutral in round 12.
- **Ablation:** replace b_i with the removed magnitude r₀,i = rms(x_i) at layer 0.

**3BPA test.** Run D3 first. Then 3 seeds × 1500 epochs.

Predictions for arm (b):
- 1200 K E ≤ 28.0 meV, at least 3 meV better than hill and at or below v9a
- E1200/E600 ≤ 3.6
- 1200 K F ≤ 56.0
- 300 K E ≤ 2.45, F ≤ 8.2

Physics checks: learned p lands in [0.7, 1.3], and β(r) comes out monotone.

Kill if any of these holds:
- p goes to the 0.5 floor, or s_Z goes to 0.
- There is no gain of at least 3 meV over hill.
- The log arm (d) matches (b). The mechanism would then be unboundedness, not homogeneity.

**Cost:** S.

**MLIP novelty:** partly explored.
- Finnis–Sinclair embeddings: PACE, Eq. 7 of Lysogorskiy et al. 2021, and the GRACE-FS foundation models. The GRACE 3BPA model does not use one; it uses a 16-unit MLP embedding.
- PINN-BOP (Purja Pun et al. 2019).
- Moments descriptors (Jenke et al. 2018).
- BOBOP reports 69.6/118 on 3BPA, so a pure bond-order functional is not competitive; that is why the strata stay.
- I found no shape/scale factorisation inside an equivariant operator recurrence.

**Citations:**
- Pettifor & Oleinik, PRB 59, 8487 (1999), https://journals.aps.org/prb/abstract/10.1103/PhysRevB.59.8487
- Drautz & Pettifor, PRB 74, 174117 (2006)
- Lysogorskiy et al., npj Comput. Mater. 7, 97 (2021), https://arxiv.org/abs/2103.00814
- Purja Pun et al., Nat. Commun. 10, 2339 (2019), https://doi.org/10.1038/s41467-019-10343-5
- Jenke et al., https://arxiv.org/abs/1810.03368
- BOBOP, https://arxiv.org/abs/2608.22933
- Beucler et al., https://arxiv.org/abs/2112.08440
- Gama et al., https://arxiv.org/abs/1905.04497

### 1C. Monotone, well-posed, variational resolvent (paper-level operator method; main target gap iii)

**Mechanism (math).**
- **Monotone parametrisation.** Winston & Kolter, Prop. 1: I − W ⪰ mI holds if and only if W = (1 − m)I − AᵀA + B − Bᵀ. So the matrices can be parametrised freely and well-posedness holds by construction. Linear RACE is covered, because the identity is the prox of 0.
- **Consequences.** For every geometry the fixed point h* is unique and ‖h*‖ ≤ ‖u₀‖/m. Forward-backward and Peaceman–Rachford splitting converge linearly, and the backward pass is again a monotone problem.
- **Monotone is not contraction.** Only Re λ(M) ≤ 1 − m is enforced. |λ| is unbounded, so large negative "screening" eigenvalues and large imaginary parts are allowed. This differs from layer_scale and spectral normalisation, which bound |λ|.
- **Variational case (no skew part).** A(R) = I − M(R) = diag(m_{Z,l}) + K(R)ᵀ D K(R) ≻ 0. Define Φ(h; R) = ½⟨h, A h⟩ − ⟨u₀, h⟩ and E_pol = s · min_h Φ = −(s/2)⟨u₀, A⁻¹ u₀⟩.
  - This is the algebraic form of an induced-dipole model, (α⁻¹ − T)μ = E. Monotonicity plays the role of Thole damping against the polarisation catastrophe, which is what the rMD17 1e30–1e50 growth looks like.
  - Hellmann–Feynman forces: ∇_R E_pol = s[½⟨h*, (∂_R A) h*⟩ − ⟨h*, ∂_R u₀⟩]. Inference forces need no adjoint solve.
  - E_pol is a rational function of A(R), whereas the truncated strata are a polynomial.

**Why it addresses the failures.** For gap (iii) the blow-up becomes impossible by construction, with no x_norm, hill or layer_scale. For gap (i) the evidence is weak: h* is uniformly bounded, the kind of constraint that has cost 1200 K E before, and the Aitken tail resummation was negative. Treat the gap-(i) claim as gated by D4/D5 and the depth sweep below.

**RACE design.**
- K(R) h := linear_K(m(h) ⊗ x_i) is one RACE linear layer; un-normalised x_i is allowed.
- Kᵀ comes from jax.linear_transpose. The adjoint of an equivariant map is equivariant because real Wigner-D matrices are orthogonal. First check that the cuequivariance primitives have transpose rules. If not, write Kᵀ by hand as the edge-reversed aggregation, using Y(−r̂) = (−1)^l Y(r̂).
- D = VᵀV per irrep channel block (positive semidefinite). m_{Z,l} = m₀ + softplus(·) with m₀ ∈ {0.1, 0.5}. u₀ = linear_u(h₁).
- **Solve:** CG via jax.scipy.sparse.linalg.cg, which goes through lax.custom_linear_solve, so second derivatives for force training exist.
  - Block-Jacobi preconditioner (m_Z + c‖x_i‖²)⁻¹, warm start u₀/m₀, float64, tolerance 1e-8.
  - Log the CG iteration count per frame as a conditioning metric. κ ≤ (m + ‖K‖²‖D‖)/m, and iterations grow like √κ.
- **Energy:** E = Σ_{t∈{0,1}} Σ_i r_t(h_t,i) + the x/f readouts + E_pol. The early per-stratum readouts stay because they were load-bearing in round 21.
- **Stage 2:** an optional gated readout r_∞(h*). This breaks the Hellmann–Feynman shortcut and needs an adjoint solve.
- **Non-variational variant:** add a skew part S − Sᵀ and solve with GMRES.
- **Training-time truncation test** (from pole-expansion step 1): retrain tied+hill unrolled to T ∈ {5, 10, 20} with remat, with readouts 0–4 plus one tied readout on h_T. Claim gap (i) only if 1200 K E improves monotonically with T.

**3BPA and rMD17 tests.**
- rMD17 molecules that blow up: max ‖h*‖/‖u₀‖ ≤ 1/m holds (a correctness check), and MAE is at or below v9a.
- 3BPA 300 K within 3 % of tied+hill (2.42/8.11).
- Gap (i) is accepted only if 1200 K E ≤ 28.0.
- Kill if the variational variant loses more than 10 % on 300 K F, or if 1200 K E is worse than 30.99 (the truncated tied model).

**Cost:** L. Forward, backward and double-backward solves in float64; cost depends on κ.

**MLIP novelty:** partly explored.
- DEQuify recasts EquiformerV2 as a nonlinear DEQ with direct (non-conservative) forces, an Anderson solver and no uniqueness guarantee.
- I-MLFF builds implicit PaiNN/SO3net, with existence from layer norms (Brouwer), a Frobenius Jacobian penalty, and forces by implicit differentiation.
- SPD solves are established for scalar charges (4G-HDNNP, Ko et al., Nat. Commun. 2021; QEq).
- I found no monotone, well-posed-by-construction equivariant feature-field resolvent with a variational energy and Hellmann–Feynman forces.

**Citations:**
- Winston & Kolter, NeurIPS 2020, https://arxiv.org/abs/2006.08591
- DEQuify, https://arxiv.org/abs/2509.08734
- I-MLFF, https://arxiv.org/abs/2607.29158
- DEQ, https://arxiv.org/abs/1909.01377

---

## 4. Tier 2

### 2A. SOAP: Kronecker-factored whitening of operator weights (gap ii; decides whether gap ii is optimisation-limited)

**Mechanism.**
- Shampoo factors: L ← β L + G Gᵀ and R ← β R + Gᵀ G. Their eigenbases Q_L, Q_R are refreshed every f steps.
- SOAP runs Adam on G̃ = Q_Lᵀ G Q_R and maps the update back with Q_L(·)Q_Rᵀ.
- For a linear layer, G = Σ δ ⊗ a (backprop signal times input), so the factors approximate the Gauss–Newton block E[aaᵀ] ⊗ E[δδᵀ].
- Wang et al., Prop. 4: A(g_t, g_{t+1}) = 1 − (η²/2)‖H P^(−s) g‖²/‖g‖² + O(η³). A preconditioner close to the Hessian keeps consecutive gradients, and the E and F gradients, aligned. Their Prop. 3 notes that SOAP does not resolve conflicts at initialisation.

**Why it fits.** RACE is a product of learned linear operators, a deep linear network. Its curvature is anisotropic and coupled by rescalings (scale W_t up by c and absorb 1/c downstream), and the stratum input covariances differ widely. Adam's diagonal scaling cannot whiten that. With 500 frames, the 300 K gap to GRACE may be an optimisation gap.

**RACE design.**
- Reshape each cuequivariance path weight (l_in → l_out) to a (mul_in × mul_out) = 128 × 128 block, for linear_2, skip, W_t and the tied W. The tied W's gradient is summed over its T applications and is still 2D. Also cover the radial-MLP matrices.
- AdamW for biases, species shift/scale, readout heads, hill (a, b) and the prior width, following Harari's Table 2 grouping.
- f = 10, β₁ = 0.95, β₂ ∈ {0.95, 0.99}, learning rate ∈ {3e-3, 1e-2, 3e-2}, eigh in float64. Write it as an optax GradientTransformation, ported from the jaxpi SOAP.
- Do not use Muon on its own: on water energy it was 1.53 against AdamW's 0.773. Equivariance is unaffected, because only channel-mixing scalars change.

**3BPA test.** v9a and tied+hill, a 500-epoch learning-rate screen, then 1500 epochs × 3 seeds.
- Key falsifier: training E/F loss at equal epochs.
- Prediction: 300 K E ≤ 2.2 and F ≤ 7.8 for tied+hill, or the same accuracy in at most 60 % of the epochs.
- Log D7 and per-stratum feature RMS.
- Reject if 1200 K regresses by more than 5 %. Memory records that LR annealing hurt extrapolation, and earlier rounds showed a 300 K vs 1200 K trade-off.
- Kill: if training loss and alignment match AdamW, close the whole second-order line, KFAC included.

**Cost:** S.

**Evidence (corrected numbers):** Harari et al., Table 1, energy + force training vs AdamW:
- NequIP on water with SOAP: E −22 %, F −19 %.
- Allegro on CsH₂PO₄ with SOAP: E −9 %, F −8 %; SOAP-Muon F −14 %.
- No out-of-distribution tests.

**MLIP novelty:** partly explored (Harari 2026; RLEKF for DeePMD). No 3BPA, rMD17 or extrapolation results exist.

**Citations:**
- Harari et al., https://arxiv.org/abs/2607.02499
- Vyas et al. (SOAP), https://arxiv.org/abs/2409.11321
- Wang, Bhartari, Li, Perdikaris, NeurIPS 2025, https://arxiv.org/abs/2502.00604
- Hu et al. (RLEKF), https://arxiv.org/abs/2212.06989

### 2B. Per-frame spectral-norm hinge on the geometry-dependent operator (gap iii on untied v9a)

**Mechanism.** Bai et al., Eq. 4: L + γ‖εᵀJ‖²/d with ε ~ N(0, I), a Hutchinson estimate of the mean squared singular value. For RACE's linear layers, J = M_t(R) exactly. The Frobenius estimate spreads the top direction over d = N × 2048 components, so instead use a per-frame estimate of σ_max, as in spectral-norm regularisation (Yoshida & Miyato). σ_max ≥ ρ for a non-normal M.

**RACE design.**
- For each graph g and layer t, run 3 steps of v ← normalise(M_tᵀ M_t v), with Mᵀ from jax.vjp and v under stop_gradient. v can be warm-started per frame (about 220 MB) or drawn fresh.
- σ̂_t,g = ‖M_t v‖/‖v‖. Penalty: λ Σ_g Σ_t max(0, σ̂_t,g − c)², with c ≈ 1.3 so v9a's operating range (1.05–1.24) stays free; set c from D5.
- Alternative: an end-to-end penalty max(0, ‖M₅⋯M₁ v‖/‖v‖ − c⁵)², which leaves per-layer gains free.
- Optional: evaluate the penalty on label-free jittered geometries (σ_R = 0.05–0.1 Å). This is a regulariser only, with no labels, so it stays within the protocol.
- This is the soft, cheap counterpart of 1C's hard guarantee. Both test whether conditioning can be had without losing 1200 K E.

**Test.** v9a + hinge with λ ∈ {1e-4, 1e-3}: 500 epochs × 2 seeds, then 1500 × 3.
- rMD17: max_t ‖h_t‖/‖h₀‖ drops from about 1e30 to below 1e3.
- 3BPA 300 K within 3 % of v9a (2.65/8.62); 1200 K E ≤ 28.2.
- Kill: conditioning improves only at the x_norm-like cost of about +10 % on 1200 K E.

**Cost:** S.

**MLIP novelty:** partly explored. I-MLFF applies a Frobenius penalty (coefficient 0.32) to implicit models only. I found nothing on explicit equivariant MLIPs or 3BPA.

**Citations:**
- Bai, Koltun, Kolter, ICML 2021, https://arxiv.org/abs/2106.14342
- Yoshida & Miyato, https://arxiv.org/abs/1705.10941
- I-MLFF, https://arxiv.org/abs/2607.29158

---

## 5. Tier 3 (worth noting)

- **3a Rational / Padé readouts** (cost S; for gap i).
  - σ(x) = P(x)/Q(x) of type (3,2), linear at infinity, or (2,2), bounded. Use the PAU-safe denominator Q = 1 + |b₁x| + |b₂x²|.
  - Initialise from the best (3,2) fit to SiLU so step 0 reproduces the baseline. Share coefficients per stratum.
  - Run only if D6 is at least 1 %; under hill the readout inputs are already range-limited. Use (2,2) only with pair_repulsion on. It fits naturally inside 1B's g_θ.
  - Prediction: −5 to −10 % on 1200 K E, with 300 K within ±3 %.
  - Citations: Boullé, Nakatsukasa, Townsend, https://arxiv.org/abs/2004.01902 ; Padé Activation Units, https://arxiv.org/abs/1907.06732 ; KAN output blocks in MLIPs, https://arxiv.org/abs/2409.03430
- **3b Scalar SPD spectral energy channels** (cost S/M; N = 27, dense float64). These are cheap prototypes of the 1C family.
  - **Variational QEq.** E_Q = min_{Σq=0} [χᵀq + ½ qᵀJq], solved through the (N+1) KKT system.
    - χ_i and J_ii = J₀,Z + softplus(·) come from the stratum 0e invariants (magnitude before normalisation is allowed). J_ij = erf(r_ij/(√2 σ_ij))/r_ij with learned per-species σ.
    - Controls: J_ij = 0 (local and concave), and LES-style direct latent charges.
    - Weak: the 5-layer receptive field already covers the molecule, and TACE and GRACE are local. Any gain must come from functional form. D8 gates the electrostatics reading.
    - Citations: Shaidu et al., npj Comput. Mater. 10, 47 (2024), https://doi.org/10.1038/s41524-024-01225-6 ; Cheng (LES), https://arxiv.org/abs/2408.15165
  - **Collective log-det.** S^(k) = diag(g^(k)) W^(k) diag(g^(k)), where W_ij = Wendland ψ₃,₁(r_ij/r_c^(k)) is positive definite in R³ and g = softplus of the stratum 0e invariants before normalisation.
    - E_LD = Σ_k c_k [log det(I + S^(k)) − Σ_i log(1 + S^(k)_ii)], computed by Cholesky.
    - Controls: (i) c tr S; (ii) Σ log(1 + S_ii); (iii) tr p(S) with a polynomial fitted on the 300 K eigenvalue range. The degree-5 Taylor control in the original proposal diverges for λ > 1 and is dropped.
    - Under relative perturbations, |δ log(1 + λ_i)| ≤ 2ε.
    - Citations: Gama, Bruna, Ribeiro, https://arxiv.org/abs/1905.04497 ; Levie et al., https://arxiv.org/abs/1907.12972 ; Tkatchenko et al., PRL 108, 236402 (2012) ; Ubaru, Chen, Saad, SIMAX 38 (2017) (SLQ, for larger systems) ; MBD-ML, https://arxiv.org/abs/2602.22086 (uses polarisability labels, so not fair-protocol)
- **3c Multi-pole rational resolvent** (cost M). Only if D4 or the depth sweep is positive.
  - The proposed α_k = 0.9 tanh(θ_k)/max(1, ρ̂(R)) with ρ̂ under stop_gradient breaks F = −∇E and locality, the same defect that got the Jacobi-basis idea dropped.
  - Safer, and my own suggestion: build the poles inside 1C as E = −Σ_k (s_k/2)⟨u₀, (A + z_k I)⁻¹ u₀⟩ with z_k ≥ 0 learned. This is a Stieltjes function of an SPD operator, well-posed for every geometry, and one multi-shift CG solves all shifts.
  - Citations: Lin, Lu, Ying, E, https://arxiv.org/abs/0906.1319 ; ARMA GNN, https://arxiv.org/abs/1901.01343 ; CayleyNets, https://arxiv.org/abs/1705.07664
- **3d Latent tight-binding band energy** (cost L).
  - Orthogonal sp tight binding: 72 × 72 for C₁₂H₁₂N₂O, 76 electrons.
  - Closed-form Slater–Koster blocks: H_sp = t_spσ r̂, H_pp = t_ppπ I + (t_ppσ − t_ppπ) r̂r̂ᵀ, with t_ppπ = −softplus(·).
  - Mermin free energy at kT_el ≈ 0.1 eV; chemical potential via jax.lax.custom_root; a Daleckii–Krein custom_jvp keeps the double backward safe at degeneracies.
  - Kill if t(r) collapses or the band energy carries less than 5 % of the energy variance.
  - Citations: Li et al., https://arxiv.org/abs/1808.04526 ; Zhou et al., PNAS 119, e2120333119 (2022) ; Seiser, Pettifor, Drautz, PRB 87, 094105 (2013) ; QHFlow2, https://arxiv.org/abs/2602.16897 (trained on Hamiltonian labels, so not direct evidence here)

---

## 6. Combinations (paper-level stories)

**A. "Soft modes and scale: energy extrapolation of force-trained equivariant potentials" (1A + 1B, with 3a inside g_θ).**
- An E/F-trained model has two separate blind spots for out-of-distribution energies:
  - the loss metric weights modes by κ, so soft-mode curvature errors that drive the thermal energy bias are invisible (H1, harmonic);
  - bounding the operator for force accuracy removes the magnitude channel that carries large-distortion energy (H2, faster than harmonic).
- 1A supplies the missing low-frequency information through 75 force constraints per frame. 1B lets magnitude re-enter once, homogeneously, on a well-conditioned shape operator.
- Design: a 2 × 2 factorial on tied+x_norm, {L2, dual} × {hill, b^p}, with 3 seeds. D1–D3 predict which cell removes which part: the bias μ_T for 1A, the growth ratio for 1B.
- Theory: the harmonic error decomposition (force MSE ∝ tr B²H, bias ∝ tr B, variance ∝ ‖B‖_F²), the variance-reduced (Rao–Blackwell) estimator, and the homogeneity argument.
- Target: 1200 K E ≤ 25.5 with E1200/E600 ≤ 3.2, while keeping 300 K F ≤ 8.3 and 1200 K F ≤ 56.

**B. "Variational spectral potentials" (1C as the backbone; 3b as cheap prototypes and the ablation axis; 3c as the multi-shift extension; trained with 2A).**
- The energy is the stationary value of a learned, well-posed quadratic functional, which reads physically as a learned polarisable response. Uniqueness and bounded features hold by construction (gap iii). The response has infinite body order and extrapolates as a rational or logarithmic function.
- The ablation axis is the spectral functional f applied to a learned SPD operator:
  - (A + z)⁻¹, single or multi-shift (Stieltjes)
  - log(I + S)
  - a polynomial (today's strata)
- Optional coupling to 1B: build A from the normalised shape operator, so it is scale-free and well-conditioned, and let u₀ = b_i^(p/2) û₀ carry the scale. E_pol then inherits degree-p homogeneity.
- Staging: scalar QEq or log-det (M cost) first, then feature-level with no skew part (L cost), then multi-shift. SOAP is a natural optimiser here, since the single shared operator's weight gradient is a sum of Kronecker terms over the solve.

**C. "Spectral-bias control in three spaces" (1A + 2A + 2B; all cost S).**
- Three norms decide what converges:
  - the residual metric in configuration space (1A, after Yu et al.);
  - the parameter metric (2A, after SOAP and Wang et al.);
  - the operator gain (2B).
- Testable coupling: the dual term estimates the energy error, so its gradient should align with g_E. Prediction: cos(g_E, g_F) rises under 1A and SOAP keeps it high (D7).
- Target: 300 K E ≤ 2.2 and 1200 K E ≤ 27.5 at the same time, and on v9a an rMD17 max ‖h_t‖/‖h₀‖ below 1e3.
- This is the lowest-risk methodology paper.

---

## 7. Weak-evidence flags

- **1A**
  - Yu et al. is an analogy (Fourier frequencies on the input domain); the harmonic algebra is ours.
  - Packwood "Exp" is an isotropic graph Laplacian whose soft modes are translations, so it is the wrong metric and serves only as the G0 control.
  - Harmonic ceiling: about −10 to −15 % (§0).
  - G1 depends on the model's own Hessian. Its relative error of a few percent is harmless for a metric, but early refreshes are unreliable, so γ = 0 before epoch 200.
- **1B**
  - The scout's claim that GRACE's 3BPA model uses a Finnis–Sinclair embedding is wrong.
  - φ_rep was already neutral in round 12.
  - Without degree-0 messages the strata are not scale-free.
  - BOBOP shows a pure bond-order functional is poor at 1200 K.
- **1C**
  - The gap-(i) argument is weak: the uniform bound resembles x_norm, and Aitken was negative.
  - The Hellmann–Feynman shortcut holds only for the pure variational energy.
  - The "~1.1 iterations per MD step" figure attributed to I-MLFF is unverified.
  - The cuequivariance transpose rules are an implementation risk.
  - D5 may show that v9a's regime lies outside the monotone class.
- **2A**
  - No out-of-distribution evidence; the scout's numbers were overstated (corrected above).
  - Muon alone is worse.
  - The claim that SOAP bounds ρ(M) is false.
  - Gap (i) is unsupported.
- **2B**
  - The Frobenius version dilutes the top direction, hence the spectral hinge instead.
  - The prior is that it reproduces the x_norm trade-off.
- **3a**
  - Possibly moot if D6 is below 1 %. A bounded (2,2) type flattens the repulsive wall.
- **3b**
  - QEq: the receptive-field argument and degeneracy with the strata.
  - log-det:
    - Gama's Def. 4 also assumes |h| ≤ 1, which log(1 + λ) violates.
    - The theorem bounds an operator-valued filter output, not a scalar trace.
    - Wendland kernels break the relative-perturbation regime near the support edge.
- **3c**
  - The scout's "bounded by construction" and efficiency claims are false.
  - DEQ potentials already exist.
- **3d**
  - The verdicts conflict: kept in mlip, dropped in phys.
  - A closed-shell molecule with a gap of several eV is far from its Fermi level, so the band-energy nonlinearity matters little.
  - A latent H has a U(N) gauge freedom and only one energy label per frame, so it is poorly identifiable.
  - Second derivatives through eigh are delicate.
- **Analogous support only**
  - Deng et al. (softening in uMLIPs, a different regime).
  - QHFlow2 (trained on Hamiltonian labels).

---

## 8. Run order (fits the 5-job limit)

1. **Week 1, no training:** D1–D8 on the v9a and tied+hill checkpoints. In parallel, check that jax.linear_transpose works through the cuequivariance kernels.
2. **Week 1–2, 500-epoch screens:**
   - 1A: γ ∈ {1, 2, 4} with G1, plus a G0 control
   - 1B: arms (a), (b), (d)
   - 2A: learning-rate screen on tied+hill
   - 2B on v9a
   - 1C depth sweep, T ∈ {5, 10, 20}
3. **Week 3:** 1500 epochs × 3 seeds for the survivors, then combination A (the 2 × 2 factorial) or combination C.
4. **Week 4 onward:** if D4/D5 or the depth sweep is positive, the 3b QEq/log-det prototype, then the feature-level 1C build (combination B).
## Appendix: dropped candidates (id | title | domain | url)
- les-latent-electrostatics | Latent Ewald summation for machine learning of long-range interactions | mlip | https://arxiv.org/abs/2408.15165
- config-ef-conflict | ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks | mlip | https://arxiv.org/abs/2408.11104
- varpro-readout | Train Like a (Var)Pro: Efficient Training of Neural Networks with Variable Projection | mlip | https://arxiv.org/abs/2007.13171
- jacobi-spectral-basis | How Powerful are Spectral Graph Neural Networks | mlip | https://arxiv.org/abs/2205.11172
- learned-pole-expansion | Equivariant Matrix Function Neural Networks | math | https://arxiv.org/abs/2310.10434
- chebyshev-semi-iteration | Chebyshev semi-iterative methods, successive overrelaxation iterative methods, and second order Richardson iterative methods (Part I) | math | https://eudml.org/doc/131485
- lanczos-pade-moment-readout | LanczosNet: Multi-Scale Deep Graph Convolutional Networks | math | https://arxiv.org/abs/1901.01484
- optbasis-lanczos-strata | Graph Neural Networks with Learnable and Optimal Polynomial Bases | math | https://arxiv.org/abs/2302.12432
- path-independent-random-depth | Path Independent Equilibrium Models Can Better Exploit Test-Time Computation | ai_dyn | https://arxiv.org/abs/2211.09961
- linoss-stable-chebyshev | Oscillatory State-Space Models | ai_dyn | https://arxiv.org/abs/2410.03943
- skew-adjoint-cayley | Anti-Symmetric DGN: a stable architecture for Deep Graph Networks | ai_dyn | https://arxiv.org/abs/2210.09789
- lru-polar-ring-gamma | Resurrecting Recurrent Neural Networks for Long Sequences | ai_dyn | https://arxiv.org/abs/2303.06349
- linear-rnn-universality-joint-readout | Universality of Linear Recurrences Followed by Non-linear Projections: Finite-Width Guarantees and Benefits of Complex Eigenvalues | ai_dyn | https://arxiv.org/abs/2307.11888
- eps-constraint-primal-dual | From Energy-Force Weighting to Primal-Dual Optimization of Machine-Learned Interatomic Potentials | ai_opt | https://arxiv.org/abs/2610.00876
- risk-extrapolation-distortion-strata | Out-of-Distribution Generalization via Risk Extrapolation (REx) | ai_opt | https://arxiv.org/abs/2003.00688
- config-conflict-free-EF | ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks | ai_opt | https://arxiv.org/abs/2408.11104
- rational-resolvent-arma-deq | Graph Neural Networks with Convolutional ARMA Filters | ai_opt | https://arxiv.org/abs/1901.01343
- kfac-energy-natural-gradient | Kronecker-Factored Approximate Curvature for Physics-Informed Neural Networks | ai_opt | https://arxiv.org/abs/2405.15603
- taip-test-time-adaptation | Online Test-time Adaptation for Interatomic Potentials | ai_opt | https://arxiv.org/abs/2405.08308
- thermal-metric-force-loss | The dark side of the forces: assessing non-conservative force models for atomistic machine learning (Appendix A: harmonic analysis of force vs energy targets) | phys | https://arxiv.org/abs/2412.11569
- latent-coulomb-les | Latent Ewald summation for machine learning of long-range interactions | phys | https://arxiv.org/abs/2408.15165
- latent-tb-band-energy | Tight-binding models and density-functional theory | phys | https://journals.aps.org/prb/abstract/10.1103/PhysRevB.39.12520
- lanczos-haydock-recursion | Electronic structure based on the local atomic environment for tight-binding bands | phys | https://iopscience.iop.org/article/10.1088/0022-3719/5/20/004
- risk-extrapolation-hotness | Out-of-Distribution Generalization via Risk Extrapolation (REx) | phys | https://arxiv.org/abs/2003.00688
