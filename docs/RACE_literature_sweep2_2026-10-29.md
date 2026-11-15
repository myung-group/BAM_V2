# RACE literature sweep 2 (2026-10-29): cross-field methods

**Scope.** Fourteen ideas passed triage and verification. They come from numerical analysis, optimisation, multi-task learning, deep-learning theory, ACE/spectral graph theory and statistical physics. Eleven more were dropped at triage. Every paper cited here was checked to exist, and its claim was checked against the text. Numbers marked **[measured]** come from training-free gates run on existing checkpoints in this session (Slurm jobs 1177737, 1178133, 1178180, plus numpy runs on `research/diag/err25p`). Numbers marked **[inferred]** come from log statistics, such as an MAE/RMSE decomposition under a Gaussian assumption. Statements marked *(inference)* are our reasoning, not results from a paper.

**Headline.** Five independent verifications reached the same finding. SOAP's energy penalty, failure mode (iv), is almost entirely one constant energy offset, and forces cannot see it. A closed-form refit of that constant on training data only gives the following (lr 1e-2, seed 42, 500 epochs, forces unchanged):

| T | E before → after (meV) | F (meV/Å) |
|---|---|---|
| 300 K | 6.41 → **2.02** | 6.73 |
| 600 K | 9.41 → **6.73** | 16.80 |
| 1200 K | 30.11 → **27.17** | 49.27 |

These are **[measured]**. The bias decomposition behind them (offset of −6.1 meV, error spread of about 2.0 meV at 300 K, and the AMSGrad comparison) is in Tier 1-A.

None of the ideas has a verified mechanism for the 1200 K energy deficit, failure mode (i). Four gates falsified the hypothesis that the deficit sits in individual bonds.

---

## 0. Executive summary

**1. Remove the force-null energy offset in closed form (variable projection, applied post hoc).** Targets (iv) and the 300 K half of (ii). Tier 1; the gate is largely passed.

- **The mechanism.** 3BPA has one stoichiometry, so the per-frame energy has exactly one direction that is invisible to forces. It is carried by the scalar `shift` and the readout biases.
  - The network itself produces the whole binding energy: about −131 eV per frame, with a frame-to-frame std of 0.165 eV.
  - In `soap_optax.py`, leaves with ndim ≤ 1 get plain Adam with b1 = b2 = 0.95. Adam's step is sign-like (Balles & Hennig, 1705.07774; Kunstner et al., 1905.12558 §5). One step on `scale` moves the frame energy by about 0.65 eV·u. AMSGrad's v_max freezes these leaves; SOAP does not, so the offset drifts and never quite converges.
- **The evidence [measured].**

  | Run | Signed energy bias (pred − ref) | Centred σ, 300 K |
  |---|---|---|
  | SOAP lr 1e-2 | train −5.95, test 300 K −6.09 meV | 2.01 meV |
  | AMSGrad twin | 300 K +0.85 meV | 3.72 meV |

  - The offset carries 90 % of the 300 K MSE.
  - Refitting the constant on the training split gives the numbers in the headline.
  - Golub & Pereyra (SIAM J. Numer. Anal. 10:413, 1973; review in Inverse Problems 19 R1, 2003, doi 10.1088/0266-5611/19/2/201) is the principle behind the fix. Eliminating linearly entering parameters exactly leaves the set of stationary points unchanged and removes the ill-conditioning that the slaved linear parameters cause. Applying this to an energy gauge under sign-like optimisers is our own derivation; no paper treats it.
- **How it bears on the SOAP finding.**
  - SOAP is not trading energies for forces. After the gauge refit it is better than AMSGrad on energy at every temperature. Centred σ: 2.01 vs 3.72 (300 K), 6.69 vs 8.39 (600 K), 24.77 vs 28.11 (1200 K).
  - This matches Harari et al. (2607.02499, Table 1): SOAP improves energy as well as forces in every row, so an energy penalty is not intrinsic to SOAP.
  - The running r27 w_E screen is confounded by the random offset. Raising w_E shrinks the offset: train offset −5.95 / −3.16 / −1.12 meV at w_E = 1 / 10 / 30. But it worsens 600 K centred σ (6.69 / 6.82 / 7.90) and 1200 K centred σ (24.77 / 26.80 / 27.66) **[measured, mid-run checkpoints]**.
  - So w_E > 1 under SOAP is not recommended.

**2. Eliminate the gauge during training, and test the mechanism by moving the 1-D leaves to another optimiser.** Targets (iv). Tier 1, needs a 500-epoch screen.

- **S1, VarPro-centred energy residual.** In `_loss_terms`, centre the per-atom energy residual over the valid graphs in the batch. Take `shift` out of the optimiser, and set it in closed form before every validation pass and checkpoint save.
  - Expected effect: the gradient SOAP sees no longer carries the common-mode term c(t)·Σ_n J_n, which inflates the second moment and swamps the energy-shape signal.
- **S2, mechanism control.** Keep SOAP on leaves with ndim ≥ 2 and move the 0-D/1-D leaves to AMSGrad, with no centring.
- **Why run it.** The post-hoc refit cannot fix checkpoint selection: right now the best checkpoint is chosen on an offset-contaminated validation loss. Only the in-training version can also improve the frame-varying part of the error.
- **Expected gain on 600/1200 K:** small *(inference)*. The main value is that SOAP's force gains become usable without the energy penalty.

**3. A damped Gauss–Newton step on the energy rows after the gauge refit.** Targets (ii). Tier 1 as a training-free gate only; the prior is low.

- **The step.** Build J_E for the 450 training energies (24.7 GB float64) and the 450×450 Gram matrix K. Take one damped step d = −J_Eᵀ(K + μI)⁻¹ r_c on the centred residuals.
- **Rationale.** ENGD (2302.13163) and NNCG (Rathore et al., 2402.01868) both make the point that the exact Gauss–Newton step is the projected function-space Newton step.
- **Why the prior is low.** Two first-order energy pushes already failed: the r18 stage-two fine-tune, and SOAP at w_E = 30. Arbel et al. (2302.02904) report a speed-versus-generalisation trade-off for Gauss–Newton.
- **Cost:** about 1 GPU-hour, no training. It is worth running only because it is cheap and decisive for the whole energy-block GN family.

**4. Data efficiency (iii) is limited by generalisation, not by optimisation, so optimiser-side ideas do not target it.** Pruning result.

- **Train vs test force RMSE at 300 K [measured]:**

  | Run | Train F (meV/Å) | Test F (meV/Å) |
  |---|---|---|
  | SOAP-500 | 1.22 | 6.73 |
  | AMSGrad-1500 | 1.32 | 8.18 |
  | v9a | 1.57 | 8.63 |

  - Along the AMSGrad trajectory, d log(test F)/d log(train F) is about 0.15.
- **rMD17 ethanol:** the training force RMSE bound is ≤ 0.75 meV/Å, already below GRACE's test MAE of 1.4, while our test RMSE is 4.35 and MAE 2.26.
- **Consequences.**
  - Exact Gauss–Newton, ENGD, NNCG, a lazy/rich γ knob and readout refits cannot close the GRACE rMD17 gap.
  - Read-out refits were checked directly: refitting 5 stratum gains plus the offset gives coefficients of 0.997–1.008 and moves test 300 K E from 2.43 to 2.41 meV.
  - GRACE's advantage most plausibly comes from basis and inductive bias, not from L-BFGS-B *(inference)*.
- **One cheap attribution arm is worth adding:** plain Adam with b1 = b2 = 0.95 and no eigenbasis rotation. It tells us whether SOAP's force gain comes from the Kronecker eigenbasis at all. It is borrowed from the dropped gn-curvature-soap item.

**5. The 1200 K energy deficit (i) is a smooth, collective softening, not a bond-local defect.** Pruning result and gate for future ideas.

- **Four gates failed** (tail occupation, warped prior, bond routing, shape constraints).
  - Tail moments split by sign and bond class explain at most 4–7 % of the variance.
  - Frames with no bond beyond 0.12 Å carry 58–76 % of μ₁₂₀₀.
  - The best per-bond-type oracle correction, fitted on the 1200 K labels themselves, gives −8 to −12 % on E.
- **What the bias looks like [measured].**
  - It is a smooth convex function of E_ref. The slope is −1.0 % (v9a) and −1.4 % (tied+hill).
  - The force softening scale is 0.991–0.995, which matches the phenomenology of Deng et al. (npj Comput. Mater. 2025, doi 10.1038/s41524-024-01500-6) at very small magnitude.
  - An oracle that removes the mean plus a cubic trend in E_ref takes v9a from 27.5 to only 25.2 meV.
- **Gate for any future collective-term idea:** before training, it must reach a cross-validated partial R² ≥ 0.15 on the seed-mean 1200 K error beyond a cubic in E_ref.

---

## 1. Map of all verified ideas

| # | Idea | Field | Failure mode | Mechanism (one line) | Tier | Cost | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | varpro-hermite-readout-gauge | Numerical analysis (separable NLS); Bayesian linear readouts | (iv), (ii) | Profile out the force-null energy gauge in closed form (VarPro) instead of letting sign-like Adam drive it | 1 | S | **Keep (gauge part only)**. The Hermite-BLR, slimTrain and RFF/GDML readout parts are dropped: no headroom (gains 0.997–1.008, 300 K E 2.43 → 2.41) |
| 2 | soap-ef-geometry-audit | Multi-objective optimisation; optimiser geometry (MAdam, AdaTask, PINN NTK) | (iv) | Audit SOAP's E/F geometry; the surviving branch is the gauge/offset fit plus the S2 leaf-partition control | 1 | S | **Keep (gauge branch + S2)**. Items 2–5 (preconditioned NTK traces, K_EE spectrum, per-leaf B_noise, Hessians) dropped |
| 3 | function-space-gauss-newton-refinement | Scientific-ML optimisation (ENGD, SPRING, MinSR, NNCG) | (iv), (ii), (iii) | Exact GN restricted to null(J_F) is the scalar refit; a residual energy-block GN step is the optional extension | 1 | S (gate) | **Keep: arm (c) + Gate 2**. Full sketched ENGD/SPRING/NNCG arms dropped: the float64 Jacobian is about 2 TB, and (iii) is limited by generalisation |
| 4 | posthoc-power-ema | Generative-model training (EDM2), averaging theory, central flows | (iv) | Post-hoc power-function EMA / schedule-free averaging to remove a SOAP slow mode | 3 | M | **Drop**. The deficit is a slowly decaying offset of fixed sign; EMA lag makes it worse |
| 5 | gn-curvature-soap | Second-order optimisation (empirical Fisher vs GGN, Sophia, KFAC) | (iv) | Replace SOAP's empirical-Fisher statistics with Gauss–Newton statistics | 3 | M–L | **Drop**. The useful part (gauge direction) is subsumed by #1. Morwani 2406.17748: the EF is nearly optimal at batch 5. Keep only the plain-Adam attribution control |
| 6 | dynamic-barrier-lexico-energy | Constrained / lexicographic MTL (dynamic barrier, CAGrad, MoCo) | (iv) | Optimise energy subject to no first-order increase in force loss | 3 | M | **Drop**. 0 of 90 minibatches show E/F conflict; λ_t = 0 in 90 of 90 |
| 7 | optimizer-invariant-explicit-prior | Implicit-bias theory (Amari et al., Gunasekar et al.) | (iv), (i) | Make the radial prior explicit so it is independent of the optimiser | 3 | S | **Drop**. The radial spectrum does not track 1200 K E; the cause of (iv) is the offset |
| 8 | tail-occupation-power-counting-gate | Statistical physics / OOD diagnostics | (i) | Test whether the 1200 K bias lives in deep-tail bond strains | 3 (diagnostic) | S | **Drop as a method; result kept**. It closes the bond-tail energy line |
| 9 | warped-coordinate-regularity-prior | ACE regularity priors (Agnesi transform) | (i) | Measure radial smoothness in a warped coordinate y(r) | 3 | M | **Drop**. The prior makes bonds stiffer, not softer; the oracle bound is −8 to −12 % E |
| 10 | exclusive-coordinate-bond-routing | Heterogeneous distribution-shift theory (Simchowitz et al.) | (i) | Route the shifted bond coordinate to a simple additive class | 3 | M | **Drop**. Additive bond terms are bounded by the oracle (≤ 6 % E); the theory's assumptions fail |
| 11 | bond-shape-sign-constraints | Shape-constrained regression | (i) | Hinge penalties on Morse sign laws along unlabelled bond scans | 3 | M | **Drop**. Partial R² of 0.006–0.029; hinge expected to be inactive |
| 12 | lazy-rich-entk-diagnostic | Deep-learning theory (lazy vs rich training, eNTK) | (i), (iii) | Diagnose the training regime; tune an output-scale knob γ | 3 | M | **Drop**. Predictable "rich" outcome, mis-specified kernel for Adam, no theoretical sign for γ |
| 13 | purified-products-newton-girard | ACE self-interaction purification | (ii), (i) | Remove the j = k diagonal from symmetric products via Newton–Girard identities | 3 | M | **Drop**. Wrong regime (J/N ≈ 6–8); the best extrapolators keep self-interaction; reopens closed products |
| 14 | nb-exact-recursion | Spectral graph theory / non-backtracking GNNs | (i), (ii) | Exact non-backtracking walk recursion | 3 | L | **Drop** (marked paper_verified = false: the GRACE premise is misread). The Chebyshev gain came from x_norm; backtracking share is flat in T |

---

## 2. Tier 1 ideas in depth

The three kept ideas share one mechanism, so this section treats them as one programme in three parts: (A) post-hoc gauge elimination, (B) gauge elimination during training plus the mechanism control, (C) a residual energy-block Gauss–Newton step.

### 2.A Closed-form gauge elimination after training (VarPro)

**Mechanism (maths).**

- The total energy is E_θ(x) = Σ_i [scale · ε_θ(i) + shift] + E0.
- For a single stoichiometry with N atoms, the composition design matrix has rank 1. The training log says the same: "composition rank 1/4 ... only the per-frame offset is identifiable".
- The loss is L(θ, c) = w_E Σ_n ρ((E_θ(x_n) + Nc − E_n)/N) + w_F Σ_n ‖F_θ(x_n) − F_n‖², with F = −∇E.
- The force term does not depend on c, and ρ is quadratic in this regime (Huber with δ = 10). So the exact minimiser is c*(θ) = mean_train(E_n − E_θ(x_n))/N.
- The reduced (VarPro) functional L(θ, c*(θ)) has the same stationary points as the full one (Golub & Pereyra 1973, 2003). Applied after training, this is the exact Gauss–Newton step restricted to null(J_F), and here the GN curvature is known in closed form *(inference, from #3)*.

**Why the offset appears under SOAP but not AMSGrad.**

- The constant direction mostly sits in `shift` (a 0-D leaf). Its restoring curvature is tiny, about w_E/N² ≈ 1/729.
- Under SOAP's 1-D path (Adam with b2 = 0.95) the step is O(lr) whatever the residual, so the offset follows a noisy, slowly decaying trajectory.
- AMSGrad's monotone v_max is frozen at large early gradients, so its step is roughly proportional to the residual.
- This explanation is consistent with the measurements below but has not been tested causally; that is what arm S2 is for *(inference)*.

**Evidence [measured unless marked].**

- SOAP lr 1e-2: train mean −5.95 ± 1.45 meV; test 300 K −6.09 ± 2.01 meV. The train and test means differ by 0.14 meV, so a constant fitted on training data transfers.
- After the refit, at other learning rates:
  - lr 3e-3: 1.96/7.20, 7.48/17.80, 29.05/52.10.
  - lr 3e-2: 300 K E 2.41.
- AMSGrad controls: offset < 1 meV (+0.85, or +0.68 / −0.15 depending on the run). AMSGrad 1500-epoch checkpoints are within 0.33 meV.
- Validation MAE/RMSE ratio: 0.93–0.99 under SOAP against 0.64–0.67 under AMSGrad **[from logs]**.
- Live-parameter training energy RMS: about 93–103 meV per frame under SOAP against about 56 meV under AMSGrad **[inferred from E_LOSS]**.

**Standing against GRACE, TACE and MACE after the refit (one seed, 500 epochs).** This corrects a "5 of 6" claim made in one verification.

| Comparison | Metrics beaten (of 6) | Lost on |
|---|---|---|
| MACE | 6 | none |
| TACE_max | 5 | 1200 K E (27.17 vs 23.42) |
| GRACE | 4 | 300 K E (2.02 vs 1.7) and 1200 K E (27.17 vs 25.8) |

- Wins against the best of all three references: 300 K F, 600 K E, 600 K F, 1200 K F.
- The dihedral scan has not yet been rescored after the refit.

**How it differs from what was tried.**

- It is not E/F re-weighting: it removes an exact symmetry direction.
- L-BFGS refinement and the stage-two fine-tune were run on AMSGrad models, which have no offset. Neither was ever run on SOAP checkpoints.
- The earlier E0 fix (doc §11.16b) handled identifiability at initialisation, not drift during training.
- `use_dataset_stats=true` (r2) also decouples the scale, but it changed avg_n_neighbors as well and hurt 1200 K, so it was not the same test.
- It stays within the fair protocol: it uses training labels only and one model, exactly like the E0 least-squares fit in MACE. It is not test-time adaptation.

**RACE design.** No code change to the model.

- Add a post-processing step to `analyze_errors.py` / `eval_sets.py`: shift += mean_train(E_ref − E_pred)/N, computed from the EMA parameters over the 450 training frames.
- For datasets with several stoichiometries (water, Si/C, HEA), replace the mean with a least-squares fit on composition counts.
- Apply it to every SOAP result: 3BPA, the rMD17 molecules and AcAc. Each rMD17 molecule is a single stoichiometry, so its gauge is also one-dimensional.

**Training-free gate. Status: largely passed.**

| Pass criterion | Measured |
|---|---|
| \|δ*\| ≥ 3–4 meV | 5.95 ✓ |
| \|μ_valid − μ_train\| ≤ 1 and \|μ_300K − μ_train\| ≤ 1 | 0.14 to test 300 K ✓; valid −6.12 ✓ |
| Corrected 300 K E ≤ 3.0 meV | 2.02 ✓ |
| AMSGrad \|δ*\| < 1 meV | ✓ |
| δ* stable within 0.3 meV across 450 vs 500 frames and two random halves | **not yet checked** |
| Live (`--raw`) vs EMA offset (EMA lag vs dynamics bias) | **not yet checked** |
| Seed 43 SOAP | **missing** |
| Dihedral rescore with shift ← shift + 5.95e-3/27 eV/atom | **not yet done** |

**Screen.** None needed: the arm costs nothing. Adopt it as the reporting protocol for all SOAP runs, and use it as the reference arm S0 in 2.B.

**Risks.**

- One seed per learning rate, and the offset differs between runs. The refit must therefore be applied to every run, including before reading the r27 screen.
- The refit improved 1200 K here only because the offset's sign (negative) matched the hot-frame bias. With the opposite sign, 1200 K would get worse, but the training mean is still the correct estimator.
- The 2.0 meV residual is a lower bound under the Gaussian assumption (it cannot be below 1.97); heavy tails may push it to about 2.5 meV. The measured refit value of 2.02 supersedes the inference.
- As a contribution it is an artefact fix framed as a principle: exact elimination of force-null gauges in E+F training under adaptive optimisers. The user should judge whether it counts as "methodology".

**Citations.**

- Golub & Pereyra, SIAM J. Numer. Anal. 10:413–432 (1973).
- Golub & Pereyra, Inverse Problems 19 R1 (2003), doi 10.1088/0266-5611/19/2/201.
- Newman, Ruthotto, Hart, van Bloemen Waanders, arXiv 2007.13171 (GNvpro).
- Newman et al., slimTrain, arXiv 2109.14002.
- Balles & Hennig, arXiv 1705.07774.
- Kunstner, Balles, Hennig, arXiv 1905.12558.
- Harari et al., arXiv 2607.02499.
- Vyas et al., SOAP, arXiv 2409.11321.

### 2.B Gauge elimination during training, and the optimiser-partition mechanism test

**Mechanism.**

- Per batch of B valid graphs, replace the per-atom residual r_n with r_n − mean_batch(r). This is the exact reduced functional for the one-dimensional gauge in a quadratic loss.
- The gradient sent to SOAP then no longer contains c(t)·Σ_n J_n. That common-mode term inflates the second moment v and its Kronecker statistics, and so hides the energy-shape signal.
- This is AdaTask-type dominance (2211.15055) in the rotated basis *(inference)*.
- MAdam (2606.03904, Prop. 1–2) explains why simply raising w_E cannot fix it: Adam-type metrics absorb loss weights, and a sign-like step is invariant to scale.

**How it differs from what was tried.** No closed item removes null(J_F) from the optimisation. The energy-weight boosts, ConFIG and primal-dual weighting all re-balance E against F; this removes a direction that forces cannot see.

**RACE design** (new config key `energy_gauge: "varpro"`):

1. In `_loss_terms`, use the graph-mask-weighted, batch-centred per-atom energy residual, and multiply w_E by B/(B−1) = 1.25 to compensate for the lost degree of freedom.
2. Mask `shift` out of the optimiser (optax.masked / set_to_zero).
3. At the end of every epoch, and before every validation pass and checkpoint save, set `shift` in both the live and the EMA parameters to its closed-form value. This takes one energy-only pass over the training frames, about 1–2 s.
4. Log the uncentred training-mode energy RMS for the mechanism check.

**Mechanism control S2.** Use `optax.partition` (or `multi_transform`): SOAP on leaves with ndim ≥ 2, AMSGrad on the 0-D/1-D leaves (shift, scale, biases, gains), no centring. An optional variant D runs the 1-D leaves at b2 = 0.999 instead.

**Training-free gate:** the same as 2.A. Run the screen only if the half-split stability check holds.

**500-epoch screen.** A+hill, SOAP lr 1e-2, b1 = b2 = 0.95, preconditioning frequency 10, w_E 1 / w_F 10, EMA 0.99. Every arm, references included, is scored **after** the 2.A post-hoc refit, so no arm gets credit for offset removal alone.

| Arm | Content | Seeds |
|---|---|---|
| S0 | Existing r26_hill_soap_lr1e2 plus refit; new s43 reference run | 42 (exists), 43 |
| S1 | SOAP + `energy_gauge: varpro` | 42, 43 |
| S2 | SOAP with ≤ 1-D leaves on AMSGrad, no centring | 42 (43 if close to S1) |
| (opt) H | Plain Adam, b1 = b2 = 0.95, no rotation: attribution control for SOAP's force gain | 42 |
| (skip) B | AMSGrad + varpro: predicted null, since the AMSGrad offset is < 1 meV | — |

**Decision rules.**

- **Adopt S1 or S2 as the SOAP default** if, on both seeds, all of the following hold:
  - 300 K E ≤ 2.2 meV, or at least 5 % below S0 after refit;
  - forces within +2 % of 6.73/16.80/49.27;
  - 1200 K E no more than 3 % (about 0.8 meV) worse than the S0 refit value;
  - dihedral E not worse than AMSGrad, after the refit.
- If S2 matches S1, the cause is confirmed as the 1-D Adam path. Prefer S2 as the simpler fix and report the mechanism.
- If S1 and S2 are equal to S0 within noise, keep only the post-hoc refit (2.A) and take plain SOAP to 1500 epochs × 3 seeds.
- **Mechanism check for S1:**
  - training-mode energy RMS falls from about 90–100 meV to a few meV;
  - the epoch-to-epoch std of validation E falls from 0.6–3 meV to about 0.2 meV.
  - If neither happens, the gauge is not the noise source.
- **Kill the line** if S1 or S2 loses more than 5 % on forces.
- **Arm H:** if H reproduces SOAP's force gain, the eigenbasis rationale is void, and that should be reported in any SOAP write-up.

**Risks.**

- Centring a batch of 5 discards 1/5 of the energy information. The 1.25 factor compensates only in expectation.
- The common mode drifts freely during training. This is harmless as long as `shift` is re-solved before every evaluation and save; if it is not, checkpoint selection stays contaminated.
- Slow, nearly force-null modes (for example dihedral-scale energy errors) are not removed by a one-dimensional gauge.
- All SOAP numbers are single-seed and 500 epochs. The deep-supervision lesson applies: a 500-epoch lead can be a convergence-speed effect, so confirmation at 1500 epochs × 3 seeds is mandatory.
- SOAP costs about 3.5–4× AMSGrad per epoch (17–20 s vs 5 s).

**Citations.** Golub & Pereyra (1973, 2003); Newman et al., arXiv 2007.13171; AdaTask, arXiv 2211.15055 (AAAI 2023); MAdam, arXiv 2606.03904; Harari et al., arXiv 2607.02499 (App. A.3: β1 = β2 = 0.95, preconditioning frequency 10; Table 2 keeps per-type energy scale/shift in an Adam group).

### 2.C Damped Gauss–Newton step on the energy rows after the gauge refit (training-free)

**Mechanism.**

- After the gauge refit, the centred training energy residuals r_c (σ_train about 1.45 meV under SOAP) are still under-fitted: the energy elasticity along the AMSGrad run is about 0.9 *(inference)*.
- The exact GN step on the energy rows is d = −J_Eᵀ(K + μI)⁻¹ r_c, with K = J_E J_Eᵀ (450 × 450). This is the push-through (Woodbury) form used in ENGD/SPRING-type methods (Guzman-Cordero et al., 2505.12149). It is the function-space Newton direction projected onto the tangent space (ENGD, 2302.13163, Thm 2).
- Unlike a first-order energy push, it moves only within the span of the energy Jacobian, by a controlled amount.

**How it differs from what was tried.** L-BFGS refinement was quasi-Newton on the full E+F loss of AMSGrad models. The stage-two fine-tune and the w_E boosts were first-order pushes, and both cost 1200 K accuracy. This step is exact and one-shot, and validation-selected damping limits the force cost. It is still the "energy-block balancing" family, so the prior is low.

**Gate** (about 1 GPU-hour on a GH200, normal partition, Triton GEMM disabled):

1. Start from the r26_hill_soap_lr1e2 EMA parameters with the refit applied.
2. Compute J_E with one VJP per frame and stream it to K.
3. Grid μ ∈ {1e-3, 1e-2, 1e-1, 1} × tr(K)/450 and η ∈ {0.25, 0.5, 1}.
4. Select on the 50 validation frames, subject to validation F ≤ +1 %.

**Decision.**

- **Pass:** test 300 K E ≤ 1.8 meV (at least 10 % better), 600 K E not worse, every force within +2 %, and 1200 K E within +3 %.
- **On a pass,** add arm S3 = S1 followed by this step at epoch 500, on both seeds.
- **On a fail,** close the energy-block GN family together with the stage-two fine-tune.

**Risks.**

- The Arbel et al. (2302.02904) trade-off: faster convergence but worse generalisation.
- Validation selection on 50 frames is noisy.
- Fort et al. (2010.15110): a kernel solve matches full training but does not exceed it.

**Citations.** ENGD, arXiv 2302.13163 (ICML 2023); Guzman-Cordero, Dangel, Goldshlager, Zeinhofer, arXiv 2505.12149; Rathore et al., arXiv 2402.01868 (ICML 2024); Arbel et al., arXiv 2302.02904 (NeurIPS 2023); MinSR, arXiv 2302.01941.

---

## 3. Tier 2 / Tier 3 ideas (brief)

There are no surviving Tier 2 ideas. Tier 3 entries are kept for the record and for what they ruled out.

- **Post-hoc power EMA / schedule-free** (Karras et al., 2312.02696; Cohen et al., 2410.24206; Morales-Brotons et al., 2411.18704; Defazio et al., 2405.15682).
  - The SOAP slow mode is a decaying offset of fixed sign, not an oscillation. Detrended jitter is 0.88 meV, so removing all of it would only take E from 6.51 to 6.45.
  - Power-EMA lag costs +0.6 meV at σ_rel = 0.05 and +1.4 meV at 0.10 **[inferred from logs]**.
  - Re-test only if the gauge refit unexpectedly leaves 300 K E above 4 meV.
- **GN curvature for SOAP** (Kunstner et al., 1905.12558; Sophia, 2305.14342; Morwani et al., 2406.17748; Amari et al., 2006.10732).
  - The correct principle for the gauge direction, but there the exact Newton step is the closed-form refit.
  - At batch size 5 the empirical Fisher nearly matches the optimal Kronecker approximation of H_GN (Morwani §4.2).
  - Keep only the plain-Adam attribution arm H.
- **Optimiser-invariant explicit prior** (Amari et al., 2006.10732; Gunasekar et al., 1802.08246; Darby et al., 2601.15072).
  - The theory covers fixed preconditioners only.
  - SOAP doubles the share of first-layer radial weight at n ≥ 5, but no radial measure tracks 1200 K E. For example, the smoothest run (SOAP 3e-3) gives 31.6 meV.
  - The proposed λ is 200–20,000× the data loss.
- **Lazy/rich eNTK diagnostic** (Chizat et al., 1812.07956; Fort et al., 2010.15110; Atanasov et al., 2410.04642; Malladi et al., 2210.05643).
  - The outcome ("rich") is predictable.
  - The Adam-consistent kernel is the SignGD kernel (Malladi), not the NTK.
  - There is no theoretical sign for γ in the offline small-data regime.
- **Purified products via Newton–Girard** (Ho, Gutleb, Ortner, 2401.01550 / J. Comput. Phys. 515, 113271; Chong et al., 2509.14146; Tan et al., 2502.16125).
  - The gains vanish for J ≫ N; here J/N ≈ 6–8.
  - Backtracking across layers keeps repeated indices, so features are never canonical.
  - The diagonal term carries no angular information.
  - TACE and GRACE use self-interacting products and still extrapolate best.
- **Exact non-backtracking recursion** (Alon et al., math/0610550; Kempton, 1603.05553; NBA-GNN, 2310.07430; GRACE, 2311.16326).
  - No exact node-space recursion exists for operator-valued weights beyond t = 2.
  - The Chebyshev gain was attributed to x_norm (doc §12.4, round 25b).
  - The backtracking share of walk mass is flat in temperature: 0.451 → 0.454 for bond-peaked weights.
  - GRACE's practical model keeps self-interactions, which is why this entry is marked paper_verified = false.
- **Bond-local family for (i)** (#8–#11). This is the evidence base for Executive summary point 5 and the collective-term gate. Partial R² of tail and Morse proxies is 0.006–0.07; the oracle additive bond corrections reach ≤ 6–12 % E even when fitted on the test labels.

---

## 4. Dropped ideas (one-line reasons)

**Dropped at verification**

- posthoc-power-ema: the premise is refuted by the logs; the deficit is a constant offset of fixed sign, and EMA lag only adds error.
- gn-curvature-soap: its one relevant arm duplicates the closed-form gauge refit; the eigenbasis barely changes at batch size 5; the GGN step law confounds the learning rate.
- dynamic-barrier-lexico-energy: no E/F conflict (0 of 90 batches); the barrier degenerates (λ_t = 0 in 90 of 90); the theory is continuous-time and full-batch only.
- optimizer-invariant-explicit-prior: the theory does not cover adaptive preconditioners; no link to 1200 K; does not touch the offset.
- tail-occupation-power-counting-gate: the bond-tail hypothesis for (i) is falsified (partial R² 0.04–0.07; zero-tail frames carry 58–76 % of the bias).
- warped-coordinate-regularity-prior: the prior stiffens rather than softens bonds; the oracle ceiling is −8 to −12 % E; it compresses 2.5–5 Å, where 88 % of edges sit.
- exclusive-coordinate-bond-routing: the clamped variable stays measurable through the 1-3 distances; the shift is not heterogeneous; the oracle additive term gives ≤ 6 % E.
- bond-shape-sign-constraints: the defect is in magnitude, not sign, so the hinge is inactive; Kronberger et al. (2103.15624) report worse accuracy under shape constraints.
- lazy-rich-entk-diagnostic: predictable gate, mis-specified kernel for Adam, and a hyper-parameter knob at the end.
- purified-products-newton-girard: wrong J/N regime, reopens the closed product line, and leaves (i) untouched.
- nb-exact-recursion: inexact for weighted operators; the Chebyshev gain came from x_norm; flat in temperature; misreads GRACE.

**Dropped at triage**

- kfac-reduce-expand-forces: large cost and needs jvp through cuequivariance; dominated by the GN items.
- ekf-online-natural-gradient-split-measurements: recursive least squares on the readout, which is covered by the readout check (no headroom), plus a large block-diagonal natural-gradient implementation.
- famo-alpha-fair-ef: an adaptive E/F weight schedule, which is closed in spirit and interacts badly with SOAP (MAdam Prop. 1).
- upgrad-frame-jacobian-descent: per-frame conflict weights border on closed re-weighting, and the gate found no conflicts.
- boltzmann-density-identities-ruled-out: a negative result, not a fair method; it would import ANI-1x sampling information.
- rotational-equilibrium-forced-wn: weak, untested link; overlaps closed LayerScale, spectral normalisation and AdamW.
- spectral-geometry-implicit-bias: the theorems cover separable classification only; Muon doubled water energy MAE in Harari et al.
- separable-isotypic-xnorm: incremental x-norm variant; gate D3 showed x magnitudes are in range.
- two-centre-torsion-energy: flexibility of the kind that previously traded 300 K for 1200 K. Backup only, if a gate shows a Fourier series explains at least 30 % of dihedral error variance.
- laplace-evidence-hyperlearning: the readout-level part has no headroom; full-network evidence is ill-defined with RACE's normalisations.
- degree-limited-extrapolant-readout: overlaps closed readout variants and the Padé idea; weak evidence.

---

## 5. Suggested order of experiments (5 job slots, each up to 4 GPUs; SOAP r27, DeNS and rMD17 in flight)

**Step 0: one short job now (training-free, about 30–45 min, 1 GPU, normal partition, `--xla_gpu_enable_triton_gemm=false`; never on the login node).**

- Run `analyze_errors` with EMA and with `--raw` on r26_hill_soap_lr{3e3,1e2,3e2}_s42, the finished or best-so-far r27_soap_ew{3,10,30,100}, r25_cheb_hill_s42 (AMSGrad control), and any SOAP rMD17 or AcAc checkpoints.
- Compute signed means per split, the corrected RMSE sqrt(RMSE² − 2μ_train μ_set + μ_train²), and δ* stability across 450 vs 500 frames and two random halves.
- Rescore the dihedral scan with `eval_sets.py` after the shift.
- **Re-read the r27 w_E screen after the refit.** If the result confirms point 1 of the Executive summary, consider stopping r27 early to free its slot; this is the user's decision.

**Step 1: same or a second short job.**

- Run Gate 2.C (energy-block GN step, about 1 GPU-hour).

**Step 2: 500-epoch screen in the slots freed by r27 (and by DeNS when it finishes).**

| Priority | Run |
|---|---|
| 1 | S1 s42 |
| 2 | S0 seed-43 reference (needed anyway) |
| 3 | S2 s42 |
| 4 | S1 s43 |
| 5 (opt.) | Arm H (plain Adam, β = 0.95, no rotation) |

- About 2.5–2.75 h per run.
- Leave the rMD17 runs alone; apply the refit to them afterwards.

**Step 3: promotion.**

- Run the winner (S1, S2, or plain SOAP + post-hoc refit) at 1500 epochs × 3 seeds (3 slots), then 3000 epochs on the best seed config.
- Report 300/600/1200 K, the dihedral scan (refit applied) and rMD17 ethanol/aspirin with the refit.
- Run S3 (S1 + GN step) only if Gate 2.C passed.

**Step 4: failure mode (i).**

- No verified candidate. Do not spend slots on bond-local terms.
- Any new collective-term proposal must first pass the training-free gate: cross-validated partial R² ≥ 0.15 on the seed-mean 1200 K error beyond a cubic in E_ref. It must also be able to beat the oracle ceiling of 27.5 → 25.2 meV (v9a), since TACE sits at 23.4.

**Housekeeping noted by the verifiers.** Running JAX on the login node crashed twice and overwrote the untracked `/scratch/t589a02/BAM_V2/core`, a 4.7 GB core dump. One verifier reports deleting it; another says it is still there and safe to delete. Check before relying on either.

**Gate scripts and outputs from this sweep**

- `/scratch/t589a02/claude_tmp/lexigate/gate_lexi.py`, `gate.json`, `gate_1178133.out`
- `/scratch/t589a02/BAM_V2/runs/diag/gn_gate/`
- `/scratch/t589a02/BAM_V2/runs/diag/oiep_gate/gate_radial_spectrum.py`, `gate_out.json`
- `/tmp/claude-500030602/-scratch-t589a02-BAM-V2/5ef853ce-88a8-4dac-9612-afaeb18bb68a/scratchpad/offset_gate.py`
- `/tmp/claude-500030602/-scratch-t589a02-BAM-V2/5ef853ce-88a8-4dac-9612-afaeb18bb68a/scratchpad/vp/refit.py`
- `/tmp/claude-500030602/-scratch-t589a02-BAM-V2/5ef853ce-88a8-4dac-9612-afaeb18bb68a/scratchpad/tailgate/` (tailgate.py, tailgate2.py, tailgate3.py, out.json)
- `/tmp/claude-500030602/-scratch-t589a02-BAM-V2/5ef853ce-88a8-4dac-9612-afaeb18bb68a/scratchpad/gate/` (bond_stiffness_gate.py, pair_bound.py, bond_anh_gate.py, ratio.py, side.py)
- `/tmp/claude-500030602/-scratch-t589a02-BAM-V2/5ef853ce-88a8-4dac-9612-afaeb18bb68a/scratchpad/gate_route/gate2.py`