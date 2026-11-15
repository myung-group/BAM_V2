# DeNS audit: does the round-22 implementation match the published method?

Sources are the paper (arXiv:2403.09549v3, TMLR 12/2024) and the official code: atomicarchitects/equiformer @64cb786 for MD17 with gradient forces, and atomicarchitects/DeNS @e98b747 for OC20/OC22. Code tags used below: [E] `equiformer/main_md17_dens.py`, [EM] `equiformer/nets/equiformer_md17_dens.py`, [EL] `equiformer/main_md17.py`, [T] `DeNS/trainers/denoising_forces_trainer.py`. The clones are under `/tmp/claude-500030602/-scratch-t589a02-BAM-V2/5ef853ce-88a8-4dac-9612-afaeb18bb68a/scratchpad/dens_src/`. Nothing in the repo was modified and nothing was committed.

---

## 1. Verdict

**The round-22 runs did not implement DeNS faithfully, so they do not test the published method.** They are closest to the paper's own "all atoms corrupted" ablation (Table 6 Index 5), made more extreme in four ways:
- every structure is noised (p_DeNS = 1);
- the DeNS term is added on top of the clean pass instead of replacing it;
- the weight is constant instead of decaying;
- the effective weight is 10–1000× the published regime.

**The developer's claim "DeNS cannot make things worse" is still wrong.** The paper itself reports cases where DeNS is neutral or worse (section 3). Both statements hold: our test was unfaithful, and the claim is false.

### Confirmed deviations that could explain the result (both verifiers: real and could matter)

| ID | Deviation | Main effect on the r22 outcome |
|---|---|---|
| **D1** | DeNS weight far above the reference regime | **Primary cause of the w=1 harm.** See the numbers below. |
| **D2** | No linear decay of λ_DeNS to 0 | The auxiliary pull persists at the end of training. |
| **D3** | Every atom is corrupted (r_DeNS = 1) | Real but secondary. |
| **D5** | Base-loss shape and units differ from the reference | Explains most of the w=1 *energy* error. |
| **D7** | Force encoding is concatenated, which re-parameterises the energy model | Confounds the "neutral at w=0.1" verdict. |
| **D8** | Noise head is one linear map | Secondary, low impact. |
| **D10** | No force loss on uncorrupted atoms at the noisy geometry | The data-augmentation half of DeNS was never tested. |
| **D12** | Protocol confounds (plateau LR schedule, best-val selection) | Inflated the reported w=1 energies and probably cost w=0.1 a few percent on forces. |

Details for each:

- **D1, weight regime.**
  - Reference MD17 recipe: λ_E:λ_F:λ_DeNS = 1:80:5 (Table 9). Every term uses unsquared L2MAE, written `criterion = L2MAELoss()` at [E]:199 and `dists = torch.norm(input - target, p=2, dim=-1)` at [EL]:127. Forces are divided by the **energy** std ([E]:159-163, 389-393). λ_DeNS decays to 0.
  - Measured as a ratio of output-space gradient magnitudes, with targets scaled to unit variance, the reference DeNS-to-force ratio is λ_DeNS·σ_E/(λ_F·σ_F). That is ≈ 0.011 for λ_F = 80 and ≈ 0.044 for λ_F = 20, using 3BPA's σ_E = 0.165 eV and σ_F = 0.948 eV/Å, and it falls to 0 by the end.
  - Ours, from logged residuals:
    - w = 0.1: 0.14 at epoch 10, 0.54 at epoch 112, 1.47 at epoch 500.
    - w = 1: 1.2 rising to 7.8.
  - So w = 0.1 sits roughly 10–100× above the reference and w = 1 roughly 100–1000× above, and the gap **grows** during training. It grows because our force loss is MSE, so its gradient shrinks as the fit improves, while L_dn levels off at 0.10–0.14.
  - The implementation report's "13–53× / 1.3–5×" figures depend on the convention chosen and understate the gap. Its suggested fix of w ≈ 0.05–0.1 is essentially the w = 0.1 run that was already done and came out neutral.
- **D2, no decay.**
  - The paper says (App. D.2): "linearly decaying DeNS coefficient λDeNS to 0 thoughout the training can result in better performance".
  - The code ([E]:372-377) is `denoising_pos_weight = args.denoising_pos_weight * (1 - min(1.0, ((epoch + 0.0) / args.epochs)))`. All 16 MD17 scripts pass `--use-denoising-pos-weight-linear-decay`: at line 28 in the two aspirin scripts and line 32 in the other 14.
  - Ours is a Python float baked into the jit-compiled step (training/train.py:719; training/utils.py:449-457, 608).
  - The decay is specific to the MD17 recipe; OC20/OC22 keep the weight constant ([T]:311-313, 521). MD17 is still the right analogue for 3BPA.
  - The paper never ablates the decay, so its effect size is unknown.
- **D3, all atoms corrupted.**
  - The paper (§3.2.3) says "adding noise to all atoms … makes denoising too difficult and potentially not well-defined".
  - The code draws a per-atom mask: `corrupt_mask = (corrupt_mask < corrupt_ratio)` ([E]:533-534). MD17 uses r_DeNS = 0.25 for every molecule.
  - Ours noises everything (training/utils.py:636-639).
  - Table 6 Index 2 vs Index 5 shows the size of the effect: partial corruption improves forces by up to about 20% over all-atom corruption (aspirin 5.7 vs 6.9).
  - It is real but secondary. Its penalty in the paper is a few percent, far smaller than our w=1 loss.
- **D5, base-loss form.**
  - Reference: L1-type loss on σ_E-normalised total energy and forces.
  - Ours: 0.5·(ΔE/N)² plus 10 × 0.5·ΔF² in eV and eV/Å.
  - The real difference is the loss **shape**, not the energy share: the reference is also about 1–2% energy. An L1 loss pulls an energy offset back with constant force. Our quadratic per-atom loss pulls with a gradient that shrinks with the offset, about 26× weaker relative to the force gradients at a 14 meV offset.
  - This explains most of the w=1 energy error (offset² / MSE ≈ 0.75 at 300 K). It explains neither the +46% force error nor the w=0.1 result.
- **D7, concatenated force encoding.**
  - The concatenation is at models/race.py:1576.
  - It grows layer-0 tp_irreps from 512 to 1280. Under e3nn's default path normalisation (path_weight = 1/√fan_in), the existing energy paths into layer-0 `linear_m` and all five `x_projs` are rescaled by 1/√2 or 1/√3. That changes both their initial output and their effective Adam step.
  - It also adds 786,560 parameters and shifts the random initialisation of every later module, because `force_enc` is built at :1218, before them.
  - No force_encoding=true, w=0 control exists. The "neutral at w=0.1" differences (−10%/+7% at 300 K) are within reach of this confound.
  - It cannot explain the w=1 harm: the Noisy-Nodes run without encoding is just as bad.
- **D8, noise head.**
  - Ours is a single e3nn Linear 128x1o → 1x1o (models/race.py:1411-1415), about 128 weights, with no nonlinearity.
  - The reference MD17 head is a full `GraphAttention` block ([EM]:160-173; 3.50M → 4.00M parameters).
  - eSCN+DeNS uses a per-node MLP and still gains, so message passing is not strictly required, but a head with real capacity is.
  - The paper does not ablate head capacity. Both verifiers rate this low impact.
- **D10, no force loss at the noisy geometry.**
  - The code computes `loss_f = criterion(pred_dy[(~data.noise_mask)], data.dy[(~data.noise_mask)] / task_std)` ([E]:390-393), where pred_dy = −∂E/∂pos at the noisy positions with force encoding active ([EM]:244, 322-329, 345).
  - Our denoise pass computes no forces and no energy (models/race.py:1777-1785).
  - This is the "data augmentation" half of DeNS (§3.2.4), and r22 never tested it. Its sign on 3BPA is unknown.
- **D12, protocol confounds.**
  - The plateau scheduler cut the learning rate only in the DeNS arms. At w=1 the first cut was at epoch 267 and the final LR was 0.0017; at w=0.1 the first cut was at epoch 282 and the final LR was 0.0051. The reference stayed at 0.01.
  - Best-validation selection, on a loss that is 99.98% forces, picked high-offset EMA epochs: 415 for w=1 and 362 for σ=0.1.
  - Gradient clipping, EMA and force-dominated selection are *not* departures from DeNS practice: OC20 uses clip 100 and EMA 0.999 (Tables 7–8), and MD17 selects on 1·E_MAE + 80·F_MAE ([EL]:335).

### Confirmed but low or no impact

| ID | Deviation | Why it matters little |
|---|---|---|
| **D4** | No per-structure p_DeNS, and DeNS is added rather than replacing the clean pass | Structurally real. The "16× dose" argument is **refuted**: the MD17 loss averages over noised atoms only ([E]:394-397), so p and r do not scale the term's weight. Replacing the clean pass alone just removes clean labels; its value comes only together with D10 and D6. One verifier: low to moderate; the other: cannot matter. |
| **D6** | No Eq. 6 energy term on the noisy, force-encoded structure | Real, but the paper's own ablation shows forces barely depend on it (MD17 Table 6 Index 4 vs Index 2: ≤ 0.1 meV/Å; OC20 Index 4 forces are *better*, 18.87 vs 19.09). It mainly matters in the reference because a picked structure there gets no clean energy at all. |
| **D9** | Only the L=1 force channel is encoded | (|F|/√3)·Y¹_component(F̂) = F exactly, so no information is lost. The claim that the task becomes "ill-posed" is refuted. |
| **D11** | Squared denoising loss instead of L2MAE | It matches the paper's Eq. 4 up to a factor of 1/3 and differs only from the code. The claimed mechanism is **backwards**: at our residuals the squared form gives 0.4–1.15× the L2MAE gradient, i.e. less, not more. |
| **D15** | Neighbour list not rebuilt after noise | Real, and the reference *does* rebuild it (`radius_graph` on the noised positions, [EM]:246-251). But the missed entering edges carry about 3.5e-7 of the total envelope mass, so the effect is negligible. |

### Refuted claims

- **D13, "DeNS only during pre-training":** not a deviation. The paper's MD17 recipe keeps DeNS on through gradient-force training (App. D.2: "Gradient methods are used to predict forces").
- **D14, noise scale and sampling:** not a deviation. Fixed σ = 0.05 Å matches MD17. Multi-scale noise exists only in v1 and is off in every released config (`fixed_noise_std: True`). The 2.5 eV/Å filter belongs to EquiformerV3, and on 3BPA it would drop 475 of 500 training frames.
- **Sub-claims:**
  - "16× dose" (D4).
  - "squared loss puts more gradient into the irreducible part" (D11).
  - "uncorrupted atoms as anchor" (D3): our clean pass already exists.
  - "balanced L1 energy" (D5).
  - "clipping shrinks the supervised step" (D12): global clipping scales both terms equally, and AMSGrad is roughly scale-invariant.
  - "about 130× by extrapolation": the logs show 6.8 at epoch 10 rising to 307 at epoch 499.
  - "the 15.71 meV is a DeNS generalisation effect" (section 2).
  - D7's fix "sum into the species embedding" is impossible, because that embedding is 0e only (:1183).
- **Citation errors:**
  - Alg. 2 line numbers: the sampling is at rendered lines 7–8 and 10–14, the force encoding at 19, the losses at 23–27.
  - The quoted comments "# per STRUCTURE" and "# per ATOM" do not exist in the source.
  - The aspirin decay flag is at line 28, not 32.

---

## 2. Forensic findings

**What the log columns mean.** `LOSS` = 1·E_LOSS + 10·F_LOSS and does **not** include AUX. `AUX` = w·L_dn.

**Loss-value ratios (AUX / supervised LOSS):**

| run | epoch 10 | epoch ~100–112 | epoch 300 | epoch 499/500 |
|---|---|---|---|---|
| w = 0.1 | 0.8 (vs 10F) | **12.9** (ep 112: AUX 0.018498 vs LOSS 0.0014348 = 1.0103e-5 + 10×1.4247e-4; `runs/r22_dens01_s42_cue/loss_r22_dens01_s42_cue.out:200`) | 50 | 80–95 |
| w = 1 | 6.8 | 43–46 | 71 | 270–307 |

- **The impl report's "13×" is a different quantity.** It compares coefficients (ρ = 0.223 or 0.022 against an assumed paper ρ of 0.004–0.017, which presumed Eq. 1's squared norm with |S| dilution). The code report's "≈13×" is the measured loss-value ratio above.
- **Output-gradient ratio, the better measure:**
  - Reference: 0.011–0.044, falling to 0.
  - Ours at w = 0.1: 0.14 → 0.54 → 1.47.
  - Ours at w = 1: 1.2 → 7.8.
- **Parameter-gradient ratio |w·g_dn| / |g_E+F|** (`audit_diag/grad.out`):
  - At initialisation: 0.89 (w=1) and 0.09 (w=0.1).
  - At the best checkpoint: 0.44 (w=1) and 0.41 (w=0.1), with cos = +0.04 and +0.05, i.e. nearly orthogonal.
  - So at w = 0.1 the relative pull grew about 4.5× during training, which is exactly what the reference decay prevents.

**The 15.71 meV coincidence.** The two w=1 runs agree at 300 K to four digits (15.715 vs 15.712 meV), but they have different bias and spread:

| run | bias | std | MAE |
|---|---|---|---|
| dens | −13.59 | 7.90 | 14.31 |
| sig01 | −13.78 | 7.55 | 13.99 |

- The match is a coincidence of √(b² + s²).
- The error is mostly a **constant offset that is also present on the training set**: −14.28 and −14.09 meV, with bias²/MSE of 0.75–0.81 on train, valid and 300 K. It is a failure to fit the mean energy, not a generalisation effect.
- **Checkpoint selection made it worse.** The valid loss is 99.98% forces (energy part 1.7e-7 against 9.6e-4), while the EMA energy offset swung between epochs 401 and 500:

  | run | valid E RMSE, epochs 401–500 (meV) |
  |---|---|
  | dens w1 | 6.87–16.82 |
  | sig01 w1 | 5.29–13.01 |
  | ref | 3.78–4.60 |

- The final-epoch EMA would have reported 9.71 meV (σ=0.05) and 6.60 meV (σ=0.1).
- The official-style selection rule (E_MAE + 80·F_MAE) would have picked epoch 436 (valid 8.26 meV) instead of 415 for σ=0.05, and the same epoch 362 for σ=0.1.

**w=1 is genuinely harmful, not only an artifact.**
- With the offset removed, energy std is 7.9 meV against 3.9, and forces are 14.2 meV/Å against 9.7.
- Training forces are underfit: 7.10 and 7.65 meV/Å against 3.52.
- Final-epoch forces are 14.22 / 30.15 / 78.50 against 9.62 / 21.93 / 61.49 meV/Å (+48 / +37 / +28%).
- The force harm was already there at epoch 250, before any LR cut: valid F 14.98 vs 10.68 meV/Å.

**w=0.1 is about neutral.**
- 3.63/10.31, 9.79/23.03, 30.44/62.57 against the v9a-500 reference 4.03/9.62, 9.54/21.93, 29.33/61.49.
- 300 K energy bias is −0.49 meV.
- Valid F tracked the reference until the LR cut at epoch 282 (10.49 vs 10.51 at epoch 267), then fell behind (9.79 vs 9.34 at epoch 500). The +5–7% force deficit may therefore be a scheduler confound.

**Run without force encoding (r22_nn).**
- L_dn ends at 0.343, against 0.114 with encoding.
- 300 K energy RMSE is 14.93 meV (bias −7.67, std 12.81).
- This agrees with the paper and with other reports of plain noisy-nodes denoising: PIWSL says NoisyNode "deteriorates the accuracy"; FRAMES reports Equiformer + Noisy Nodes on aspirin at 10.5/8 against 5.3/7.2.

**Other forensics:**
- **Paired noise draws.** The σ=0.05 and σ=0.1 runs drew identical standard normals (`PRNGKey(seed+1000)` folded with the step, training/train.py:723; training/utils.py:611-613), so their ε/σ targets are identical. The σ comparison is paired with n = 1; it is not biased.
- **Gradient clipping.** Mean pre-clip GRAD stayed above 1 until epoch 16 at w=1, against epoch 5–6 for the reference. The clip fraction is computed but never logged.
- **Released ablation code is not what Table 6 describes.** The MD17 "no force encoding" path (Table 6 Index 3) checks `denoising_mask`, which the trainer never sets ([EM]:314 vs [E]:528), and zeroes the picked outputs ([EM]:348-350). As released, it trains E(noisy) → E(clean) with no denoising gradient.
- **fairchem port.** In fairchem_core 1.3.0 and 1.9.0 the DeNS branch appends a second force loss ([FT]:595-612). Third-party ports are not the reference.
- **Paper vs code conflicts.** The code was taken as the reference in each case:

  | Item | Paper | Code |
  |---|---|---|
  | Loss form | squared L2 | L2MAE |
  | Force normaliser | σ_F | σ_E |
  | Encoder input | normalised f′ | raw dy |
  | Encoding scale | ‖f‖ | ‖f‖/√3 with component-normalised harmonics |
  | DeNS-loss averaging | 1/\|S\| | mean over displaced atoms |

- **What already matches the reference:**
  - σ = 0.05 Å;
  - target ε/σ;
  - raw training forces as the encoder input;
  - no force labels at evaluation;
  - both tasks in one gradient step (App. D.1).

---

## 3. What the evidence says about "DeNS cannot make things worse"

**MD17 main results, Table 4** (E meV / F meV/Å; 950 training frames, in-distribution test):

| | Asp | Benz | Eth | Mal | Naph | Sal | Tol | Ura |
|---|---|---|---|---|---|---|---|---|
| Equiformer L2 | 5.3/7.2 | 2.2/6.6 | 2.2/3.1 | 3.3/5.8 | 3.7/2.1 | 4.5/4.1 | 3.8/2.1 | 4.3/3.3 |
| + DeNS | 5.1/5.7 | **2.3**/6.1 | 2.2/2.6 | 3.2/4.4 | 3.7/1.7 | 4.4/3.7 | 3.5/1.9 | 4.2/**3.3** |
| Equiformer L3 | 5.3/6.6 | 2.5/8.1 | 2.2/2.9 | 3.2/5.4 | 4.4/2.0 | 4.3/3.9 | 3.7/2.1 | 4.3/3.4 |
| + DeNS | 5.0/5.2 | 2.3/6.1 | 2.2/2.4 | 3.2/4.1 | 3.7/1.6 | 4.2/3.2 | 3.4/1.8 | 4.1/2.9 |

Faithful DeNS improves forces by 0–21% (aspirin 7.2 → 5.7, "20.8%"). Energies move by ±0.1–0.3 meV, and benzene energy gets worse (2.2 → 2.3).

**MD17 design ablation, Table 6** (L_max = 2):

| Idx | Setting | Asp | Benz | Eth | Mal | Naph | Sal | Tol | Ura |
|---|---|---|---|---|---|---|---|---|---|
| 1 | no DeNS | 5.3/7.2 | 2.2/6.6 | 2.2/3.1 | 3.3/5.8 | 3.7/2.1 | 4.5/4.1 | 3.8/2.1 | 4.3/3.3 |
| 2 | full DeNS | 5.1/5.7 | 2.3/6.1 | 2.2/2.6 | 3.2/4.4 | 3.7/1.7 | 4.4/3.7 | 3.5/1.9 | 4.2/3.3 |
| 3 | no force encoding | **8.6/9.1** | 2.3/6.3 | 2.3/3.3 | 3.2/5.8 | **7.7/6.1** | **5.2/10.6** | 3.7/2.0 | **5.5/6.5** |
| 4 | λ_E = 0 in Eq. 6 | 5.1/5.8 | 2.3/6.1 | 2.3/2.6 | 3.2/4.5 | 3.7/1.7 | 4.8/3.6 | 3.7/1.9 | 4.2/3.3 |
| 5 | all atoms corrupted | 5.3/6.9 | 2.4/6.3 | 2.3/**3.2** | 3.3/5.4 | 3.7/2.0 | **4.8/4.2** | 4.0/2.0 | 4.2/**3.8** |

The paper's own words: "DeNS with noise added to all atoms (Index 5) can be worse than training without DeNS (Index 1)". Index 5 is the variant closest to our r22 setup.

**Other evidence from the paper:**
- **Table 5 (model variants):** forces always improve, but some energies get worse (salicylic acid 4.1 → 4.2, toluene 3.3 → 3.4).
- **App. D.1:** switching between DeNS and the original task per iteration "does not result in better performance on the MD17 dataset than training without DeNS".
- **OC20 Table 1(b):** baseline 20.46/285, full DeNS 19.09/269, no force encoding **21.32**/278 (forces worse than baseline), λ_E = 0 18.87/285, no partial corruption 19.54/279.
- **MD22:** absent from v1, v2, v3 and all code (`grep -ci md22` = 0).
- **Out-of-distribution:** no temperature-extrapolation or other OOD test anywhere.

**Follow-up work (verified quotes):**
- **eSEN, App. D:** "DeNS is not used during OMat24 training or conservative fine-tuning". DeNS is used only in direct-force pre-training.
- **EquiformerV3** (same first author as DeNS), App. E: "During gradient fine-tuning, we disable all regularizations and DeNS". It also caps the corrupted fraction at 0.75 "preventing denoising from degenerating into the case where noise is added to all atoms".
- **TECE** (arXiv 2607.10664), App. A: "DeNS was mainly used with the direct model to accelerate convergence, rather than to improve the final accuracy". One verifier found and verified this; the other could not locate the paper.
- **MinDScAIP:** DeNS in direct pre-training only. None of these papers ablates DeNS on the final conservative model.
- **OMat24:**
  - On MPtrj with direct-force eqV2-S, DeNS helps: validation 12.4 / 32.22 → 11.43 / 31.67 meV/atom / meV/Å.
  - At scale it is "slightly worse": 11.3 / 52.0 against 11 / 49.2.
  - "we did train models using DeNS for both pre-training and fine-tuning, but did not see an improvement in accuracy".
- **Shuang et al. (defects benchmark):** the eqV2 DeNS models trained on MPtrj "perform the worst" on energy.
- **DPA4:** reaches competitive accuracy "without auxiliary DeNS or direct-force pretraining".
- **AniDS:** reports large low-data rMD17 gains (aspirin 71.40 → 45.52), but these are confounded with PCQM4Mv2 pre-training.
- No work applies DeNS to MACE, NequIP or Allegro, or to 3BPA. This is a negative finding, so it cannot be fully verified.

**Judgement.**
- The claim is false as a general statement. The paper documents worse cells even with the full method (benzene energy; Table 5 energies), clear harm without force encoding, and harm on some molecules with all-atom corruption. OMat24 reports a penalty at scale.
- What the evidence does support is narrower: in the original MD17 in-distribution setting, with the exact recipe, faithful DeNS improved forces on every molecule and changed energies only marginally.
- No published evidence covers a 500-frame conservative model evaluated on temperature extrapolation. Even a faithful DeNS may be neutral on 3BPA.

---

## 4. Faithful DeNS specification for conservative RACE on 3BPA

Where the paper and code disagree, the code is followed. The screening base is now "A + hill" (`research/input/input_3bpa_omat24_r25_cheb_hill_s42_cue.json`). It differs from the r22 base (r20_rp05 / v9a) by `residual: chebyshev`, `layer_scale: 0.5` and `x_norm: hill`, so the r22 conclusions transfer only approximately. In the base run the plateau scheduler never fired: the LR was 0.01 for all 500 epochs, and the best epoch was 498.

### Components

1. **Corruption subset** ([E]:521-548). Draw a per-structure Bernoulli with p_DeNS = 0.25 and, inside picked structures, a per-atom Bernoulli with r_DeNS = 0.25. Set m_i = 1 only for corrupted real atoms.
   - At batch size 5, about 24% of steps have no picked structure (0.75⁵). The reference skips the term via a NaN check; use a safe denominator `max(Σm, 1)` instead.
   - The chance that a picked 27-atom molecule has no corrupted atom is about 4e-4.
2. **Noise.** Fixed σ = 0.05 Å, isotropic per component: p̃_i = p_i + m_i·ε_i. Use a noise key independent of the base RNG so that runs at different σ are not paired.
3. **Single pass with replacement** ([E]:386; Alg. 2 lines 7–8 and 23). The whole batch goes through **one** energy/force pass with corrupted positions and force input m⊙F. Unpicked structures have m = 0 and no noise, so for them this pass is exactly the clean pass. There is no extra clean copy of picked structures, and both tasks share the gradient step (App. D.1).
4. **Force encoding** (Eq. 5; [EM]:274-294).
   - Input: (‖F‖/√3)·Y^L_component(F̂) for L = 0..3 (irreps 1x0e+1x1o+1x2e+1x3o), using raw eV/Å training forces, zeroed where m = 0 and absent at evaluation.
   - Projection: an equivariant Linear to 128x(0e+1o+2e+3o), **bias-free**. A bias would only be a constant shift in every pass, and dropping it keeps the energy pass bit-identical to the base.
   - **Add** the result to the node features after layer 0, the first place where all of 0e/1o/2e/3o exist. This mirrors Equiformer adding after atom + edge-degree embedding. Do not concatenate at the layer-0 input; that is what removes the D7 fan-in confound.
   - Build the new modules from a separate `nnx.Rngs` stream so the base parameters initialise identically.
   - The paper uses L = 0..L_max. L = 1 alone carries the same information (D9), so this is for fidelity, not expected gain.
5. **Losses**, all in one total:
   - **Energy:** E(S̃, F̃) against the **clean** energy for every graph, picked or not. This is Eq. 6 with the shared energy head. Use our existing energy loss form and weight 1.
   - **Forces:** −∂E(S̃, F̃)/∂p̃ against the **clean** labels on uncorrupted atoms only (weight 1 − m_i), averaged over Σ(1 − m) rather than 3·n_atoms. Use our existing form and weight 10. This is Alg. 2 line 26 and [E]:390-393.
   - **DeNS:** L2MAE over corrupted atoms, Σ_i m_i‖ε̂_i − ε_i/σ‖ / max(Σm, 1), following [E]:394-397 and [EL]:127.
6. **Noise head.** A dedicated nonlinear head on the final node features. Its output replaces −∂E/∂r only on corrupted atoms; it is never the energy gradient.
   - Preferred: one extra RACE/NequIP-style interaction block (edges plus a tensor product, output 1x1o), mirroring the MD17 GraphAttention head.
   - Cheaper alternative: a gated equivariant MLP (Linear → gate → Linear → 1x1o), as in the eSCN ForceBlock.
7. **Coefficient regime, expressed relative to our force loss.**
   - Our force loss 10·mean_c(0.5ΔF_c²) gives an aggregate output-gradient magnitude, in unit-variance units, of about (10/3)·σ_F·⟨|ΔF_i|⟩.
   - From the base log, ⟨|ΔF_i|⟩ ≈ 1.6·√(2·F_LOSS). That is 0.092, 0.023, 0.012 and 0.007 eV/Å at epochs 10, 100, 250 and 500, so the aggregate is ≈ 0.29, 0.073, 0.038 and 0.023.
   - The L2MAE DeNS term has aggregate magnitude w(t). Matching the reference ratio G_ref = λ_DeNS·σ_E/(λ_F·σ_F) = 0.011 (λ_F = 80) to 0.044 (λ_F = 20) gives **w(t) = w₀·(1 − epoch/n_epochs)** with:
     - w₀ ≈ 0.5e-3 to 3e-3 for the λ_F = 80 equivalent, depending on which training phase is matched;
     - up to about 4× that for the λ_F = 20 equivalent.
   - Use w₀ = 1e-3 as the primary value and 4e-3 as the upper bracket. That is roughly 25–100× below r22's constant w = 0.1.
   - Keep λ_E = 1 and λ_F = 10 unchanged so the base is not perturbed.
   - Do not move the base loss to official σ_E-normalised L2MAE. That would require a new baseline.
8. **Protocol** (D12).
   - Hold the LR constant at 0.01, the base's realised schedule, by setting `lr_schedule_patience` to 100000000.
   - Keep grad clip 1.0, EMA 0.99 and 500 epochs as in the base.
   - Report **both** the best-valid checkpoint and the final-epoch EMA.
   - Log the DeNS loss, w(t) and the clip fraction.

**Risk to flag (my estimate, not measured).** Under a harmonic estimate (½σ²·tr H_ii ≈ 0.05 eV per displaced atom), Eq. 6's E(S̃,F̃) → E(S) target means the model must undo roughly 0.3–0.4 eV of strain energy per picked molecule. The clean-label forces next to displaced atoms are off by roughly 1 eV/Å. The reference loss is L1-type, so these residuals contribute bounded gradients. Under our MSE they could dominate the base loss, whose train F_LOSS at the end is about 1e-5. This is why run S3 below exists.

### Code locations to change (all read-only so far)

**`race_v2/models/race.py`**
- :1188-1190 (`input_irreps` with concatenated 1o): stop concatenating, so the layer-0 input stays `embed_irreps`.
- :1218-1222 (`force_enc`): replace with a bias-free Linear from `1x0e+1x1o+1x2e+1x3o` to `hidden_irreps`, using a separate RNG stream.
- :1568-1576: remove the concatenation. Compute the SH encoding from a node field such as `data.nodes['dens_force']` (= m·F; absent means zero), so `__call__` → `energy_forces_stress(self.node_energies, …)` at :1835-1838 serves both the training pass and evaluation. Add the projected encoding to `features` right after `layer(...)` for i == 0 in the loop at :1662-1675.
- :1411-1415 (`noise_head`): replace the Linear with an interaction block or gated MLP.
- :1777-1785 (`denoise`): retire it. The noise prediction comes from the `node_features` already returned at :1766-1768.

**`race_v2/models/blocks.py`**
- `energy_forces_stress` (:270ff; the aux dicts in both branches, around :321-328 and :347-355): pass `node_features` through `aux` and return them, so a single grad pass yields E, F and features for the noise head.

**`race_v2/training/utils.py`**
- :636-639: replace the all-atom noise with structure and atom masks (via `node_graph_idx`) and m = mask ∧ padding. Write `positions + m⊙ε`, `dens_force = m⊙F`, `dens_mask = m` and `noise` into `batch.nodes`.
- `compute_loss` :405-418: run the model once on the corrupted batch. Energy against clean energy is unchanged. For forces, use the per-node weight path at :238-240 with (1 − m), and normalise by Σ(1 − m) instead of 3·n_atoms at :411.
- :449-457: change the DeNS term to L2MAE over m, weighted by a **traced** w(t).
- `make_sharded_train_step` :556 and :600-608: add `dens_weight` to the dynamic `loss_weights` so the decay does not trigger a recompile. Log `denoise_loss` (:483) and the clip fraction (:692).

**`race_v2/training/train.py`**
- :380 and :391: model flags for the new encoding and head.
- :606 and :719: parse a new `dens` config block (p, r, σ, w₀, decay, loss form, noisy_ef on/off, head type).
- :598 and :838-842: force the dynamic-weights path whenever `dens` is set, and feed w₀·(1 − epoch/n_epochs) each epoch.
- :845-875: add log columns.
- :887-926: also save and report the final-epoch EMA test results (`eval_test_every: 100` already evaluates epoch 500).

**Test before running (replaces the w = 0 control).** With `dens_force` absent, the energies and forces of the modified model must be bit-identical to the A + hill model with the same parameters, and the parameter count on the energy path must be unchanged. Extend `tests/test_round22.py` accordingly.

---

## 5. Minimal screening plan (3 runs × 500 epochs, seed 42)

All three runs start from `research/input/input_3bpa_omat24_r25_cheb_hill_s42_cue.json` unchanged (λ_E = 1, λ_F = 10, Huber δ 10, lr 0.01, batch 5, EMA 0.99, clip 1.0, A + hill model), with two additions:
- `"lr_schedule_patience": 100000000`, which reproduces the base's realised constant LR;
- a `dens` block (new keys, to be implemented as in section 4).

Shared DeNS settings: p = 0.25, r = 0.25, σ = 0.05 Å, additive L0–3 bias-free encoding after layer 0, interaction-block noise head, L2MAE DeNS loss over corrupted atoms, linear decay to 0 over 500 epochs.

| Run | DeNS weight w₀ | E(S̃,F̃) and uncorrupted-atom F terms (Eq. 6, D10) | Clean pass for picked structures | Purpose |
|---|---|---|---|---|
| **S1 faithful** | 1e-3 | on (replacement, single pass) | no | The faithful MD17 translation (G ≈ 0.011–0.016) |
| **S2 faithful-hi** | 4e-3 | on | no | Upper bracket (λ_F = 20 equivalent, G ≈ 0.044) |
| **S3 aux-only** | 1e-3 | off: noise loss only on corrupted atoms of picked structures | yes (all structures stay clean-supervised; extra noisy pass) | Separates representation shaping from augmentation; tests the MSE-domination risk from section 4 |

Example `dens` block for S1:

```json
"dens": {"p": 0.25, "r": 0.25, "sigma": 0.05, "weight": 1e-3, "decay": "linear", "loss": "l2mae", "noisy_ef": true, "head": "interaction", "encoding": "add_l0_3"}
```

**Baseline:** test E/F at 300 / 600 / 1200 K = 3.82/9.24, 8.40/20.90, 31.12/60.05 meV, meV/Å (best epoch 498; final-epoch EMA is about the same).

**Report:** best-valid and final-epoch EMA test numbers; the DeNS loss trace; w(t); the energy bias on train and 300 K; train F RMSE; the clip fraction.

**Decision rule.** Seed spread at 300 K is about ±3% E and ±1% F (v9a s42/s43: 3.92 vs 4.13 meV, 9.69 vs 9.55 meV/Å), and about ±1.24 meV E at 1200 K.
- Call a gain only if it exceeds about 2× the seed spread at ≥ 2 temperatures.
- If so, replicate on s43 and s44 before adopting it.
- If S1 and S2 are worse and S3 is neutral, the noisy-geometry E/F supervision is the harmful component under our MSE loss. Next step: put the noisy-pass E/F terms in L1/L2MAE form.
- If all three are neutral, DeNS has no measurable effect on 3BPA under a faithful recipe. That would confirm the "no precedent" reading from section 3 and close the question independently of the developer's claim.

Diagnostic files from the audit: `/tmp/claude-500030602/-scratch-t589a02-BAM-V2/5ef853ce-88a8-4dac-9612-afaeb18bb68a/scratchpad/audit_diag/bias.out`, `/tmp/claude-500030602/-scratch-t589a02-BAM-V2/5ef853ce-88a8-4dac-9612-afaeb18bb68a/scratchpad/audit_diag/grad.out`, `/tmp/claude-500030602/-scratch-t589a02-BAM-V2/5ef853ce-88a8-4dac-9612-afaeb18bb68a/scratchpad/d1_audit/ratio.py`.