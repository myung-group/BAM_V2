# 1C — monotone variational resolvent: design and feasibility probe (2026-10-29)

Source idea: `docs/RACE_literature_sweep_2026-10-29.md` §3 "1C" (Winston & Kolter, monDEQ, arXiv 2006.08591).
Code grounding: `bam_omat24/race_nnx.py` (`RACEConvolution.__call__`, `aggregate`, residual `"neumann"`,
`tie_operator`, `_hill_normalize`, `RACE.node_energies`) and `docs/RACE_omat24_review.md` §12.4–12.6.
Probe: `runs/diag/probe_1c.py` → `runs/diag/probe_1c.txt` (job 1174479, GH200, float64, cue backend,
Triton GEMM off, 323 s). Checkpoint: `runs/r25p_ntied_hill_s42_cue/ckpt_r25p_ntied_hill_s42_cue.pkl`
(epoch 1491). Every number in §2 is transcribed from that file. The converged-series check on the depth-sweep
checkpoints (§2.1) is in `runs/diag/probe_1c_T{10,20}_1174690.txt`. Nothing in `race_nnx.py`, `train_3bpa.py`
or `training_utils.py` was changed. This document says where the changes would go.

---

## 0. Summary

**What 1C is.** Today the tied model applies one operator M(R) five times:
h_{t+1} = u₀ + M h_t, so h₅ = Σ_{k<5} M^k u₀. 1C replaces this truncated series with the exact solution of a
linear system that is well-posed for every geometry:
- A(R) h* = u₀, with A(R) = Λ + K(R)ᵀ K(R) ≻ 0 and Λ = diag(m_{Z,l}) ≥ m₀ I;
- the energy is variational, E_pol = s · min_h [½⟨h, A h⟩ − ⟨u₀, h⟩] = −(s/2)⟨u₀, A⁻¹u₀⟩;
- forces follow from the Hellmann–Feynman theorem;
- the solver is CG through `lax.custom_linear_solve(symmetric=True)`.

**What the probe established.**

*Feasible:*
1. jax.vjp gives the exact adjoint of the RACE layer map on the cue backend: |⟨Mu,v⟩ − ⟨u,Mᵀv⟩| / (‖Mu‖‖v‖) ≤ 2.2e-18.
2. jax.linear_transpose also works on the cue backend. cuequivariance-jax 0.13 registers a JVP rule and a transpose rule for `segmented_polynomial_p`, and the result agrees with vjp to 1.85e-16.
3. The exact solve is cheap:
   - (I − M)h = u₀ takes 105–114 GMRES iterations to reach 1e-8 and 135–141 to reach 1e-10;
   - CG on an SPD stand-in mI + MᵀM takes 44–45 iterations at m = 0.5 and 98–99 at m = 0.1, to 1e-8.
   - One matvec costs about 0.5 ms for 5 frames.

*What the trained operator looks like:*
4. The trained tied+hill operator already satisfies Re λ(M) < 1. The largest real part is 0.89–0.92 and is real, so min Re λ(I − M) = 0.083–0.108. The largest modulus is 0.97–0.995, a complex pair near 0.19 ± 0.96i.
5. The trained operator is **not monotone in the current (Euclidean) feature coordinates**: λ_max of its symmetric part is 2.40–2.48, while the monotone class needs ≤ 1 − m. In the same coordinates it is strongly non-normal: ‖M − Mᵀ‖_F/‖M + Mᵀ‖_F = 1.00 and σ_max = 3.9–4.0.

   These three numbers depend on the metric. Since max Re λ(M) < 1, Lyapunov's theorem gives, for each geometry, an SPD equivariant P(R) in which I − M is strongly monotone. Whether one *fixed* per-(species, l) channel basis does the same, which 1C-N could absorb into its source and readout, is untested (§3.6).

   Two statements are basis-independent:
   - The complex pair rules out 1C-V reproducing (I − M)⁻¹ in any basis, because I − A is similar to a symmetric matrix and has a real spectrum.
   - The spectral bound Re λ(W) ≤ 1 − m₀ of 1C-N excludes the trained spectrum at m₀ = 0.1, since it needs m₀ ≤ 0.083.

   Section 3.6 shows that the variational class can only *screen*: a stronger coupling K always lowers the response. The trained operator, by contrast, amplifies along its dominant modes (λ ≈ +0.9).

*The truncation tail and the energy error (gate D4):*
6. The truncation is not small. ‖h₅ − h*‖/‖h*‖ averages 0.240 on train, 0.242 / 0.274 / 0.320 on 300 / 600 / 1200 K, with a maximum of 0.58. The tail grows with temperature, so the first D4 criterion passes.
7. Within each temperature the tail does not explain the error: corr(r_n, |ΔE_n|) = 0.10 / 0.15 / 0.12 / 0.16, below the 0.3 needed, so D4 fails.
8. The energy of the missing tail, read through the trained last readout, does not track the model's error either. Its spread is 35 / 38 / 64 / 116 meV, yet corr(ΔE_tail, ΔE) is between −0.08 and +0.11.
9. Pooled over the three test sets, a fitted 4 % of the tail lowers 1200 K RMSE from 32.1 to 28.6 meV but raises 300 K from 2.4 to 5.7. An affine fit in E_ref does better (24–25.5 meV, review §12.5).

**Verdict.**
- The machinery is cheap and correct on this code base.
- The probe does not support the claim that the 1200 K energy gap (gap i) is a truncation artefact.
- 1C's case rests on well-posedness (gap iii: the rMD17 / water blow-up) and on the cheap training-time depth sweep (Stage 1).
  - With the operator *trained at T = 5*, S₂₀ is within 2–6 % of h* and S₄₀ within 1 %.
  - That does not carry over to a model retrained at T = 10 / 20. Its M is a different operator, nothing in tied+hill bounds ρ(M) (the T = 5 value is already 0.97–0.995; untied v9a reaches 1.04–1.24, §5.6), and 1a also adds 15 strata.
  - So 1a is a resolvent proxy only if the retrained series has converged: ρ(M) < 1 and ‖S_T − h*‖/‖h*‖ ≤ 5 % at T = n_layers on the probed frames. Check this with `probe_1c.py --run r26_ntied_hill_T{10,20}_s42 --items 3,4` on the final checkpoints (§2.1, §8).
  - Early snapshots pass: T10 at epoch 100 has ρ 0.89–0.90 and a 1.4–2.1 % tail; T20 at epoch 39 has ρ 0.89–0.92 and a 0.5–0.9 % tail (§2.1). The final checkpoints are still pending.
  - A negative 1a closes 1C for gap (i) only if it passes this check and 1b is also negative.
- Build the feature-level 1C-V (Stages 2–3) only if:
  - 1200 K E falls monotonically with T, or
  - the rMD17 well-posedness story is wanted on its own.

---

## 1. What the tied Neumann model computes (code grounding)

`RACEConvolution.__call__` with `update="linear"`, `message_norm="avg"`, `x_mode="basis"` (equivariant x),
`residual="neumann"`, `layer_scale=0.5`, for layer t ≥ 1 and fixed geometry R:

```
agg_i(h)  = (1/<N>) sum_{j in N(i)} w_ij (x) [ (L_1 h)_j (x) Y(r_ij^) ]     # aggregate(): cue fused TP, w_ij = radial_mlp(basis(r_ij))
m_i       = L_m agg_i(h)                                                    # linear_m: tp_irreps -> hidden
p_i       = m_i (x) x_i                                                     # channelwise TP, all CG paths into hidden irreps
M h       = gamma (.) [ L_2 p + S_{Z_i} h_i ]                               # linear_2 + species-indexed skip, LayerScale gamma
h_{t+1}   = u_0 + M h_t,           u_0 = h_1 = output of layer 0 (prev_features = feats_hist[1])
x_i       = hill( W_x A_i ),       A_i = layers[0].aggregate(embedding)     # fixed per geometry; x_projs[1] / x_hill row 1 shared by layers 1..
E         = scale * sum_t sum_i [ r_t(h_{t+1,i}) + x_t(x_i) (+ f_t) ] + shift
```

- With `tie_operator=True`, every sub-module of layer 1 except its stratum readout is shared by layers 2–4, so M is one operator.
  - The probe confirms this: ‖M_t u − M_1 u‖/‖M_1 u‖ ≤ 3.8e-16 for t = 2, 3, 4.
  - The model's last features equal the 5-term partial sum S₅ = Σ_{k<5} M^k u₀ to 5e-16.
- The map h ↦ layer(h)[1] with `prev_features=None` is M.
  - It is exactly linear: ‖M0‖ = 0 and the linearity error is 3.7e-15.
  - It is block-diagonal over the graphs of a batch: zero leak between frames and into padding.
- Hidden irreps are 128x0e+128x1o+128x2e+128x3o, which is 2048 components per node. That gives 55 296 unknowns per 27-atom frame and 294 912 for a padded batch of 5 (144 nodes).

## 2. Feasibility probe: results (verbatim numbers from `runs/diag/probe_1c.txt`, job 1174479)

Five training frames and five `test_1200K` frames, each collated as one padded batch:

| quantity | train (frames 0–4) | test_1200K (frames 0–4) |
|---|---|---|
| (1) adjoint via jax.vjp: \|⟨Mu,v⟩−⟨u,Mᵀv⟩\|/(‖Mu‖‖v‖) | 3.09e-19, 2.17e-18, 6.15e-19 | 1.24e-18, 2.17e-18, 7.70e-19 |
| (1) same, relative to \|⟨Mu,v⟩\| | 6.75e-16, 5.16e-15, 3.81e-16 | 3.90e-15, 6.09e-15, 7.92e-16 |
| (1) matvec M / Mᵀ (vjp), 5 frames, jitted | 0.48 / 0.37 ms | 0.51 / 0.40 ms |
| (2) jax.linear_transpose vs vjp | **works**, 1.85e-16, 0.41 ms | **works**, 1.85e-16, 0.41 ms |
| (3) GMRES iterations to rel. res. 1e-4 / 1e-6 / 1e-8 / 1e-10 | 45–52 / 75–85 / 105–114 / 135–141 | 50–54 / 80–84 / 108–112 / 138–141 |
| (3) jax.scipy gmres (batch) rel. res; ‖h_jax − h*‖/‖h*‖ | 5e-14 – 9e-14; 3.7e-10 – 4.8e-10 | 8e-14 – 1.4e-13; 4.0e-10 – 4.6e-10 |
| (3) ‖h*‖/‖u₀‖ ; ‖S₅‖/‖u₀‖ | 1.19–1.22 ; 1.17–1.19 | 1.25–1.36 ; 1.19–1.22 |
| (3) ‖S_T − h*‖/‖h*‖, T = 5 | 0.202 0.237 0.233 0.218 0.243 | 0.333 0.294 0.374 0.270 0.287 |
| T = 10 | 0.087 0.106 0.112 0.105 0.114 | 0.180 0.151 0.195 0.140 0.144 |
| T = 20 | 0.022 0.026 0.031 0.031 0.034 | 0.057 0.044 0.062 0.046 0.042 |
| T = 40 | 0.0036 0.0024 0.0059 0.0087 0.0109 | 0.0084 0.0086 0.0077 0.0062 0.0050 |
| (3) D4: r₄(h*) − r₄(S_T) [meV], T = 5 | +193 +181 +244 +211 +213 | +134 +355 +403 +267 +313 |
| T = 20 / T = 40 | ≤ 10.9 / ≤ 2.0 | ≤ 46.5 / ≤ 3.8 |
| (3) model error E − E_ref [meV] | +1.0 −0.7 −4.4 −1.5 +0.4 | −0.4 +10.4 −37.0 −18.6 +11.2 |
| (4) λ(M) of largest Re (ARPACK LR) | +0.914, +0.915, +0.907±0.001i, +0.915, +0.902 | +0.903, +0.917, +0.892, +0.903, +0.896±0.002i |
| (4) largest \|λ(M)\| (LM) | 0.192±0.952i (0.971) … 0.184±0.978i (0.995) | 0.188±0.952i (0.970) … 0.187±0.970i (0.988) |
| (4) λ_max / λ_min of sym(M) = (M+Mᵀ)/2 | +2.40…+2.48 / −1.90…−1.97 | +2.41…+2.44 / −1.93…−1.97 |
| (4) σ_max(M) | 3.87–4.02 | 3.90–3.97 |
| (4) skew ratio ‖M−Mᵀ‖_F/‖M+Mᵀ‖_F | 1.001–1.007 | 1.002–1.006 |
| (5) CG on mI + MᵀM to 1e-8, m = 0.5 / 0.1 (frames 0–2) | 45 / 98–99 its, ‖h*‖/‖u₀‖ 1.56 / 5.83–5.84 | 44–45 / 98–99 its, 1.56–1.57 / 5.85–5.87 |

(6) D4 on every frame. Columns:
- r_n = ‖S₅ − h*‖/‖h*‖;
- ΔE_t = r₄(h*) − r₄(S₅), the trained last-stratum readout applied to the exact resolvent;
- ΔE = model error.

The largest GMRES residual over all 6396 frames is 3.9e-12.

| set | n | r_n mean (sd) [max] | ΔE_t mean / sd [meV] | ΔE mean / RMSE [meV] | corr(r_n,\|ΔE\|) | corr(ΔE_t, ΔE) |
|---|---|---|---|---|---|---|
| train | 450 | 0.240 (0.025) [0.368] | +205.4 / 34.5 | −0.12 / 1.75 | +0.101 | +0.109 |
| test_300K | 1669 | 0.242 (0.028) [0.444] | +206.5 / 38.3 | −0.22 / 2.43 | +0.145 | −0.023 |
| test_600K | 2138 | 0.274 (0.039) [0.464] | +234.0 / 64.1 | −1.33 / 7.54 | +0.120 | −0.043 |
| test_1200K | 2139 | 0.320 (0.051) [0.582] | +280.3 / 115.5 | −16.95 / 32.06 | +0.158 | −0.084 |

Pooled over the three test sets:
- corr(r_n, |ΔE|) = +0.392 and corr(ΔE_t, ΔE) = −0.199.
- An in-sample fit ΔE ≈ a + b ΔE_t gives b = −0.0426.
- After the fit, RMSE is 2.43 → 5.66 at 300 K, 7.54 → 9.22 at 600 K and 32.06 → 28.61 at 1200 K.

**Reading.**

1. **Machinery.** Every primitive 1C needs works on the cue backend in float64:
   - the exact adjoint via vjp or linear_transpose;
   - Krylov solves inside jit;
   - implicit differentiation via custom_linear_solve, whose JVP rule differentiates closed-over matvec constants. This is in jax 0.12 `solves.py:_custom_linear_solve_jvp`.

   The cue primitive is closed under JVP and transpose (`polynomial.jvp` / `polynomial.transpose`), so higher derivatives of K and Kᵀ exist. The transpose rule turns the gather-by-sender / scatter-to-receiver kernel into the edge-reversed kernel automatically, so no hand-written Kᵀ is needed.

2. **Re λ(M) < 1 holds already.** The resolvent of the trained operator exists (min Re λ(I − M) = 0.083–0.108).
   - **In the current coordinates** the *monotone* condition does not hold: sym(I − M) has eigenvalues down to 1 − 2.48 = −1.48, and the operator is as antisymmetric as it is symmetric.
   - These are properties of M in the Euclidean metric of today's feature basis, not of the operator itself. With Re λ(I − M) > 0, the Lyapunov solution P(R) = ∫₀^∞ e^{−(I−M)ᵀt} e^{−(I−M)t} dt is SPD, equivariant (ρ(g) is orthogonal), and makes sym(P(I − M)) ≻ 0.
     - That P is geometry-dependent and dense.
     - 1C-N could use only a *fixed* equivariant change of basis T, per (species, l) and absorbed into `resolvent_source: "linear"` and the readout, with P = TᵀT. The probe does not test whether one such P exists.
     - So the probe does not show that 1C-N cannot represent the trained operator, and its numerical range of 2.4 does not by itself mean that 1C-N "removes transient-growth directions".
   - **Basis-independent:**
     - The variational A = Λ + KᵀK cannot reproduce the trained resolvent in any fixed basis, because I − A is diagonalisable with a real spectrum while M has the pair 0.19 ± 0.96i. A 1C-V model has to learn a different function.
     - 1C-N has Re λ(W) ≤ 1 − m₀ in every basis. The trained max Re λ(M) of 0.892–0.917 then needs m₀ ≤ 0.083, so use m₀ = 0.05 for 1C-N arms.
   - Before ranking 1C-N or Stage 4 on this argument, test whether one fixed block-diagonal equivariant P (per-l 128 × 128 blocks, or a per-(l, channel) diagonal) makes sym(P(I − M)) ≻ 0 on the probed frames, e.g. a small SDP over P with Lanczos on the P-weighted symmetric parts (§8 Stage 4 prerequisite).

3. **Truncation (H3).** The tail is large and grows with temperature. The trained readout, however, is fitted to S₅: putting h* in its place shifts the energy by +205 meV on average, and that shift varies from frame to frame. Within each temperature the variation is uncorrelated with the error, so D4 fails.

   The temperature trend of the tail mean (+206 → +280 meV) follows the bias trend (−0.2 → −17 meV). That is a proxy for distance from equilibrium, and E_ref is a better one (§12.5: an affine fit in E_ref leaves 23.9–25.5 meV).

   So truncation does not explain the 1200 K error *of this trained model*. Whether a model *trained* with h* (or with T = 20) generalises better is a separate question.
   - The depth sweep answers it only if the retrained series converges (§2.1): the 2–6 % tail at T = 20 was measured on the T = 5-trained operator.
   - A T = 20-trained M need not have ρ(M) < 1, and then S₂₀ approximates no resolvent.
   - 1a also changes the number of strata (5 → 20), which 1b separates from depth.

### 2.1 Converged-series check on the depth-sweep checkpoints (snapshot, job 1174690)

`probe_1c.py --run r26_ntied_hill_T{10,20}_s42 --items 3,4` was run on the checkpoints of the running Stage-1a jobs.
- T10 was at epoch 100 of 500 and T20 at epoch 39 of 500, so both are **early snapshots**.
- Outputs: `runs/diag/probe_1c_T10_1174690.txt` and `runs/diag/probe_1c_T20_1174690.txt`.
- The same job re-ran items 1–5 on r25p (`probe_1c_r25p_regress_1174690.txt`) and reproduces `probe_1c.txt`. Items 3–5 match to the printed digits. The adjoint residuals vary run to run at the 1e-18 level (6.2e-19–1.9e-18), from non-deterministic GPU scatter order.

| quantity (5 frames each) | T10 train | T10 test_1200K | T20 train | T20 test_1200K |
|---|---|---|---|---|
| ρ(M) (ARPACK LM) | 0.893–0.898 | 0.895–0.901 | 0.907–0.918 | 0.8945–0.912 |
| dominant pair | ≈ 0.61 ± 0.65i | ≈ 0.62 ± 0.65i | ≈ 0.61 ± 0.67i | ≈ 0.61 ± 0.67i |
| max Re λ(M) | +0.706…+0.716 | +0.704…+0.719 | +0.668…+0.674 | +0.670…+0.676 |
| ‖S_T − h*‖/‖h*‖ at T = n_layers | 0.014–0.021 | 0.015–0.020 | 0.0050–0.0094 | 0.0047–0.0074 |
| GMRES its to 1e-10 | 58–59 | 59–60 | 52–54 | 52–53 |
| λ_max sym(M) / σ_max(M) | 1.43–1.46 / 2.35–2.40 | 1.43–1.44 / 2.35–2.38 | 1.36–1.39 / 2.48–2.51 | 1.34–1.37 / 2.44–2.52 |
| r_{L−1}(h*) − r_{L−1}(S_L) [meV] | −2.6…−0.5 | −3.4…−0.9 | −0.3…+0.6 | +0.2…+0.6 |
| converged-series check | PASS | PASS | PASS | PASS |

**Reading.**
- At these snapshots the retrained series converges, and faster than the T = 5 model's: ρ ≈ 0.89–0.92 against 0.97–0.995.
  - S_L is within 2.1 % of h* at T = 10 and within 0.94 % at T = 20.
  - The last readout changes by ≤ 3.4 meV when h* replaces S_L.
  - So far a trained T = 20 model *is* a close proxy for its own exact resolvent.
- This does not settle the question:
  - The T = 5 operator reached ρ = 0.97–0.995 by epoch 1491, and nothing stops ρ from growing with training. Its early-epoch ρ is not known.
  - The check must be repeated on the final (epoch 500) checkpoints before 1a is used as resolvent evidence, with the same command and output named `probe_1c_T{10,20}_<jobid>.txt`.
  - The energy errors of these snapshots (up to −149 meV at 1200 K for T20) are early-training values, not accuracy evidence.

---

## 3. Parametrisation in this code base

### 3.1 Spaces and inner product
- Solve space H = ⊕_i ⊕_l (ℝ^128 ⊗ ℝ^{2l+1}): the hidden irreps of every node, laid out mul_ir as in e3nn (2048 per node).
- Inner product: the Euclidean ⟨a, b⟩ = Σ_i a_i·b_i over real nodes of the flat array. e3nn components are an orthonormal real basis and real Wigner-D matrices are orthogonal, so this inner product is O(3)-invariant.
- The cue ir_mul ↔ mul_ir conversions (`_mul_ir_to_ir_mul`, `_ir_mul_to_mul_ir`) are permutations, hence orthogonal. AD transposes them correctly.

### 3.2 K and Kᵀ
- K(R): H → H′ is one RACE message path without the skip and without γ:
  - K h := mask ⊙ L_K( (L_m agg(L_1 (mask ⊙ h))) ⊗ x̂_i ).
  - Here L_1, `radial_mlp`, L_m and L_K have the same shapes as `linear_1`, `radial_mlp`, `linear_m` and `linear_2` of `RACEConvolution`.
  - H′ is the hidden irreps by default; `resolvent_rank` can reduce it, e.g. to 64 channels per l.
- Precompute once per forward, not inside the CG loop:
  - the radial messages w_ij = radial_mlp(radial_scale ⊙ basis(r_ij)), (n_edges × n_paths);
  - Y(r̂_ij);
  - x̂_i, the layer-1 x factor after `_hill_normalize` (or rms / none);
  - the masks.

  `RACEConvolution.aggregate` recomputes the radial MLP on every call. The resolvent therefore calls `cue_edge_tensor_product` directly with the precomputed `radial_message`.
- Kᵀ: `KT = jax.linear_transpose(partial(K_apply, consts), h_struct)`, built once per forward.
  - This is exact because K is linear in h at fixed consts.
  - It works with the cue kernels because `segmented_polynomial_p` has a registered transpose rule. Probe: linear_transpose agrees with vjp to 1.85e-16 at the same cost, 0.41 ms per 5-frame batch.
  - `jax.vjp(K_apply, 0)` is the equivalent fallback (0.37–0.40 ms). It traces a dead primal that XLA removes.
  - A hand-written edge-reversed aggregation using Y(−r̂) = (−1)^l Y(r̂) is not needed.

### 3.3 Where D and Λ live
- **D** lives on H′. By Schur's lemma, the most general symmetric PSD equivariant map on 128 × l is D = ⊕_l (V_lᵀV_l ⊗ I_{2l+1}), with V_l a 128 × 128 channel mixer per irrep block.
  - Since L_K already ends in a free per-path linear map, V_l L_K is another such map, and KᵀDK = (VK)ᵀ(VK). D is therefore absorbed: **D = I in code**.
  - Option `resolvent_D: "diag"` gives a per-(l, channel) softplus diagonal. Use it when K is shared with layer 1 (§6.4), where the absorption is not free.
- **Λ** lives on H, per node, as Λ_{i,l,c} = m₀ + softplus(θ_{Z_i,l,c}).
  - Default `resolvent_lambda: "species"`: one scalar per (species, l, channel), 4 × 4 × 128 values. It acts as a scalar on the 2l+1 components of each channel.
  - `"local"`: Λ_{i,l,c} = m₀ + softplus(θ_{Z,l,c} + g_{l,c}(s_i)), with s_i the 0e invariants of A_i passed through a small MLP. This is the analogue of an environment-dependent inverse polarisability α_i(R)⁻¹ (see §3.6 for why it matters).
  - m₀ is a fixed floor, default 0.1. The initial value is m_init = 1.0, so at initialisation A ≈ I + KᵀK.
- The skip S_Z of M is not carried over. It is non-symmetric local channel mixing. Its symmetric counterpart is a per-species SPD block Λ_{Z,l} = m₀I + B_{Z,l}ᵀB_{Z,l}, the `"block"` option for Stage 4.

### 3.4 A(R) is SPD and E(3)-equivariant
- **SPD.** For any h, ⟨h, A h⟩ = Σ_{i,l,c} Λ_{i,l,c}‖h_{i,l,c}‖² + ‖K h‖² ≥ m₀‖h‖².
  - So λ_min(A) ≥ m₀ for every geometry, every parameter value and every training state.
  - λ_max(A) ≤ max Λ + ‖K(R)‖². This is finite for any finite geometry but is **not** uniform in the geometry, even with hill.
    - Hill bounds only the per-node RMS of x̂ (≤ 2).
    - The aggregation is Σ_{j∈N(i)} / ⟨N⟩ with ⟨N⟩ fixed at 16.723, so ‖K(R)‖ grows with the actual coordination inside r_c (§5.5).
- **Equivariance.**
  - Every block is an equivariant linear map: L_1, L_m, L_K, the CG products, and the Y-weighted aggregation. Writing ρ(g) = ⊕_i ⊕_l D^l(g) (orthogonal, parity included), K(gR) = ρ(g) K(R) ρ(g)ᵀ.
  - The adjoint inherits this: K(gR)ᵀ = ρ(g) K(R)ᵀ ρ(g)ᵀ, because ρᵀ = ρ⁻¹.
  - Λ is a scalar per (node, l, channel), so it commutes with ρ.
  - Hence A(gR) = ρ A(R) ρᵀ. With u₀(gR) = ρ u₀(R), this gives h*(gR) = ρ h*(R) and E_pol(gR) = E_pol(R).
  - Permutations: A(PR) = P A Pᵀ, since Λ and the masks are node-indexed. Translations enter only through r_ij.
- **Locality.** KᵀK couples nodes up to 2 hops apart (no self edges), so A is sparse. A⁻¹ is dense within a connected graph and decays with graph distance at a rate set by κ (Demko–Moss–Smith). This is infinite body order with a finite, learnable decay.

### 3.5 u₀
- `resolvent_source: "h1"` (default) uses u₀ = h₁, the layer-0 output. This is what `feats_hist[1]` is today.
- `"linear"` uses u₀ = L_u(Z_i, h₁), a species-indexed Linear from hidden to hidden. It decouples the source from the r₀ stratum.
- `"h2"` uses u₀ = h₂ = u₀ + M u₀, the explicit layer-1 output.
- The energy scale s is one learnable scalar, initialised at `resolvent_s_init` = 0.0 for the additive arm and 0.1 otherwise.
  - Its sign is free; HF needs stationarity, not minimality.
  - This holds only because s is a *differentiable primal input* of the HF rule (§5.3). If s were closed over or put in `nondiff_argnums`, JAX would stop its gradient: s would stay at its initial value, E_pol would stay ≤ 0 in "replace" mode, and the V+ arm would stay identical to tied+hill.

### 3.6 What the variational class can and cannot represent (from the probe)
- In 1C-V the geometry enters A only through +K(R)ᵀK(R) ⪰ 0 (and through Λ(R) with the "local" option).
- A⁻¹ is operator-monotone decreasing, so for fixed u₀, a stronger coupling always lowers ⟨u₀, A⁻¹u₀⟩. 1C-V is a **screening** model:
  - its effective Neumann operator M_var = I − A has real eigenvalues ≤ 1 − m₀;
  - its response along a mode is 1/(λ_Λ + σ²_K), which falls as the coupling grows.
- The trained truncated model instead uses **anti-screening** directions:
  - a dominant real eigenvalue of +0.90–0.92, i.e. amplification 1/(1 − λ) ≈ 10 in the resolvent (basis-independent);
  - a numerical range up to 2.4 *in the current Euclidean coordinates* (metric-dependent, see §2 Reading 2);
  - a complex pair 0.19 ± 0.96i (basis-independent).
- In 1C-V, geometry-dependent amplification is possible only through a geometry-dependent Λ(R), bounded by 1/m₀. That is why `resolvent_lambda: "local"` is part of the Stage-2 screen, not an afterthought.
- The complex part needs the skew term of 1C-N (§3.8). Whether 1C-N can also hold the transient-growth / non-normal part depends on whether one fixed equivariant metric P makes I − M monotone (§2 Reading 2). Lyapunov guarantees only a geometry-dependent P(R).

### 3.7 Padding
- K is masked on input and output (mask ⊙ K(mask ⊙ h)), so padded nodes and padded edges contribute nothing to K.
- Λ is **not** masked. On padded nodes, A = Λ_pad ≥ m₀ I, so A stays SPD on the whole padded space, and the diagonal preconditioner stays finite.
- Masking Λ as well would make A singular on padding. That gives 0/0 in the preconditioner and a NaN breakdown risk.
- **Unmasked Λ does not make graph-wise CG NaN-safe.**
  - `default_collate_fn` → `jraph.pad_with_graphs` always puts the padded nodes (≥ 1, since pad_nodes = bucket(n + 1)) into one extra padding graph, plus empty graphs in a partial last batch (`pad_graphs = batch_size + 1`).
  - With u₀ = 0 on padding and x₀ = P⁻¹u₀ = 0, that graph has r_g = p_g = 0, so α_g = ⟨r_g, r_g⟩/⟨p_g, A p_g⟩ = 0/0 = NaN on *every* batch. An empty graph gives 0/0 from `segment_sum` as well.
  - mask ⊙ NaN is still NaN. Node energies are masked by multiplication (`utils.py` `ne*node_mask`) and the loss by `* graph_mask` (`training_utils.py`), so the NaN would reach the loss and the gradients.
  - The transposed solve has the same problem, because its cotangent on padding is zero.
- **Rule for `graph_pcg`:**
  - α_g = where(den_g > 0, num_g/den_g, 0) and β_g = where(⟨r,r⟩_old,g > 0, ⟨r,r⟩_new,g/⟨r,r⟩_old,g, 0).
  - Freeze converged and empty graphs with `jnp.where(active_g, new, old)`, never by multiplying with a 0/1 flag.
  - The solve is never differentiated through its iterations (custom_linear_solve), so the double-where gradient trick is not needed inside the loop.
  - With this rule the padding solution is exactly 0, for the forward and the transposed solve alike.
- The rule copies `strata_energies_forces`: mask = `jraph.get_node_padding_mask(data)` if n_graphs > 1, otherwise all ones.
- Padded edges carry r = 0. The radial MLP and `_bessel_safe` already keep them finite.

### 3.8 Variants
- **1C-V (variational).** Energy from E_pol, HF forces. This is the core of this document.
- **1C-V+ (additive, rule-safe).** All five tied strata are kept unchanged, and E_pol(h*) is added with s initialised to 0, so at initialisation the model *is* tied+hill. This keeps n_layers = 5, per the "no layer reduction" rule.
- **1C-N (monotone, non-variational).** I − W = Λ + KᵀK + (B − Bᵀ), with B h a second message path L_B(m′(h) ⊗ x̂).
  - The skew part has zero quadratic form, so ⟨h, (I − W)h⟩ ≥ m₀‖h‖² and the bound ‖h*‖ ≤ ‖u₀‖/m₀ still holds.
  - Eigenvalues may be complex with unbounded imaginary part (Re λ(W) ≤ 1 − m₀ in every basis). The trained max Re λ(M) = 0.892–0.917 needs m₀ ≤ 0.083, so 1C-N arms use m₀ = 0.05.
  - Solved by GMRES. Energy from a gated readout r_∞(h*). Adjoint solve required (§5.3).
- **Multi-shift (Stieltjes).**
  - E = −Σ_k (s_k/2)⟨u₀, (A + z_k Λ)⁻¹u₀⟩, with z₀ = 0 and z_k = softplus(ζ_k) for k = 1..3. The shifts are along **Λ, not I**.
  - Multi-shift CG relies on K_k(A + zI, b) = K_k(A, b). That identity fails under a non-scalar preconditioner: P⁻¹(A + zI) is not a shift of P⁻¹A for the Jacobi P = Λ + d of §5.2.
  - With the shift along Λ and P = Λ (`resolvent_precond: "lambda"`), Λ^{-1/2}(A + zΛ)Λ^{-1/2} = Ã + zI with Ã = Λ^{-1/2}AΛ^{-1/2} SPD. So one preconditioned Krylov basis serves all shifts (shift invariance; Jegerlehner 1996, Frommer 2003), at the cost of O(n_shift · n) vector updates per iteration.
  - ⟨u₀, (A + zΛ)⁻¹u₀⟩ = ⟨ũ, (Ã + zI)⁻¹ũ⟩ with ũ = Λ^{-1/2}u₀ is still a Stieltjes function of an SPD operator.
  - The alternative is shifts along I with no preconditioner. Jacobi plus shifts is rejected by the validator (§6.1).
  - HF holds per shift, since each term is s_k min_h[½⟨h, (A + z_kΛ)h⟩ − ⟨u₀, h⟩].
  - The training backward has a different right-hand side per shift, so it costs n_shift solves (block CG).
- **Readout r_∞(h*).** A gated MLP on the 0e part of h*, like the stratum readouts. It breaks the HF shortcut: the forces then need λ = A⁻¹∇_h r_∞, one extra solve. custom_linear_solve handles this automatically (§5.3).

## 4. Energy assembly, Hellmann–Feynman forces, tolerance error

**Energy (variant V, replacing layers 2–4):**

```
E = scale * [ sum_i ( r_0(h_1,i) + x_0(x_i) )  +  sum_i ( r_1(h_2,i) + x_1(x_i) )  +  sum_i e_pol,i ] + shift
e_pol,i = s * [ 1/2 h*_i . (Lambda h*)_i + 1/2 |(K h*)_i|^2 - u0_i . h*_i ]      (sums to s*Phi(h*) = -(s/2)<u0, h*>)
```

The f-readouts (if on) stay where they are. The per-node partition is a convention. Only the sum per graph
enters the loss and the forces, and the HF identity holds per graph because A is block-diagonal over graphs.

**Variant V+:** E = E_tied+hill (5 strata, unchanged) + scale · Σ_i e_pol,i.

**Hellmann–Feynman.** Write Φ(h; R) = ½⟨h, A(R) h⟩ − ⟨u₀(R), h⟩ and E_pol(R) = s Φ(h*(R); R), where ∇_h Φ(h*) = A h* − u₀ = 0. Then

```
dE_pol/dR = s [ dPhi/dR |_{h fixed} + < grad_h Phi(h*), dh*/dR > ]
          = s [ 1/2 <h*, (dA/dR) h*> - <du0/dR, h*> ]
          = s [ d/dR ( 1/2 <h*, Lambda h*> + 1/2 ||K(R) h*||^2 ) |_{h* fixed}  -  <du0/dR, h*> ].
```

The last form needs only *forward* applications of K: ⟨h, KᵀK h⟩ = ‖K h‖². The forces are therefore one
reverse pass through a single application of K (and u₀) at fixed h*, with no adjoint solve. This equals
−dE/dR exactly at the converged h*, because the omitted term multiplies ∇_h Φ(h*) = 0.

**Finite tolerance.** Let h̃ be the CG output, r = u₀ − A h̃ with ‖r‖ ≤ ε‖u₀‖, and e = h̃ − h* = −A⁻¹r, so ‖e‖ ≤ ‖r‖/m₀.
- **Energy.** Evaluate it in the Φ form, s[½⟨h̃, A h̃⟩ − ⟨u₀, h̃⟩] = −(s/2)(⟨u₀, h̃⟩ + ⟨h̃, r⟩); CG carries r, so this costs nothing extra. Then

  E(h̃) − E* = (s/2)⟨r, A⁻¹r⟩, so |ΔE| ≤ |s|‖r‖²/(2m₀): second order in ε.

  The naive −(s/2)⟨u₀, h̃⟩ has the first-order error (s/2)⟨h*, r⟩.
- **Forces.**

  ‖F_HF(h̃) − F*‖ ≤ |s| · ‖(∂_R A) h* − ∂_R u₀‖ · ‖r‖/m₀ + O(‖r‖²): first order in ε.

  The HF force at h̃ is the exact gradient of R ↦ sΦ(h̃; R) with h̃ frozen. Compared with the gradient of the computed energy E(h̃(R)), it differs by −s⟨r, dh̃/dR⟩.
- **Size.** At ε = 1e-10 in float64 both errors are about 1e-10 relative, against force errors of 8 meV/Å. Negligible.

## 5. Solver and derivatives

### 5.1 CG through `lax.custom_linear_solve`
- `jax.scipy.sparse.linalg.cg` calls `lax.custom_linear_solve(A, b, solve=cg, transpose_solve=cg, symmetric=True)` for real b (`_isolve(..., check_symmetric=True)`).
- So x = A⁻¹b is differentiated implicitly: dx = A⁻¹(db − dA x). The JVP rule `_custom_linear_solve_jvp` handles tangents of every constant closed over by the matvec — geometry, parameters, x̂, w_ij — through `_tangent_linear_map`. The transpose reuses `solve` because the problem is symmetric.
- The CG iterations themselves (the `lax.while_loop`) are never differentiated, so the backward pass stores only a few vectors, not T layers of activations.
- jax's cg returns `info=None`, with no iteration count. For diagnostics we pass our own solve with `has_aux=True` (supported in jax 0.12, `solves.py:292`), returning per-graph iteration count and residual.

### 5.2 Graph-wise preconditioned CG, tolerance, warm start
- **Graph-wise CG.** A is block-diagonal over the graphs of the padded batch. CG uses per-graph inner products, α_g and β_g from `jax.ops.segment_sum` over `node_graph_idx`, and stops when all graphs satisfy ‖r_g‖ ≤ tol‖u₀,g‖.
  - α_g and β_g use the safe division and `where`-freezing of §3.7. The padding graph and the empty graphs have zero right-hand side, so they are inactive from iteration 0. Their 0/0 is discarded by the `where` and never reaches x, r or p.
  - This is mathematically the same as independent CGs per frame, and each frame converges at its own rate.
  - A single global CG would couple the frames' Krylov spaces. The probe's batch GMRES needed the same order of iterations, but the per-graph form is the correct one.
- **Preconditioner.** Diagonal / block-Jacobi P_{i,l} = Λ_{i,l} + d_{i,l}, where d_{i,l} estimates the mean diagonal of (KᵀK) on node i, block l.
  - The deterministic invariant proxy is d_{i,l} = c_l · Σ_{j: i∈N(j)} ‖w_ji‖² ‖x̂_j‖² / ⟨N⟩². Here c_l is calibrated once per batch by a 4-probe Hutchinson estimate with fixed seeds, under `stop_gradient`.
  - P is positive and a scalar per (node, l), so it is SPD and equivariant.
  - It enters only the `solve` closure, never the matvec, so it changes the iteration path, not the solution or its derivatives.
  - The multi-shift variant cannot use it (§3.8). It needs P = Λ (`"lambda"`) or no preconditioner.
- **Tolerance.** Relative residual 1e-10 per graph for training and evaluation, 1e-8 for screens (`resolvent_tol`).
  - The residual-form bound of §5.5 gives about +22 % iterations from 1e-8 to 1e-10; probe (5): 44–45 → about 55 at m = 0.5.
  - `resolvent_maxiter` = 500. Hitting it sets an aux flag that is logged. The solve stays bounded but becomes inexact.
- **Warm start.** x₀ = P⁻¹u₀ by default, which equals u₀/Λ when K = 0. Two later options:
  - in training, the previous epoch's h* for the same frame (an optional per-frame cache, 199 MB for 450 frames);
  - in MD, the previous time step.

  The gradients do not depend on x₀ (implicit differentiation).
- **Precision.** Everything in float64 (the model runs with x64), with `--xla_gpu_enable_triton_gemm=false` as always.

### 5.3 Derivative accounting for force training
The training loss is L(θ) = w_E ℓ(E − E_ref) + w_F ℓ(F − F_ref), with F = −∂_R E computed by `energy_forces_stress`
(jax.grad over positions). Training needs dL/dθ, i.e. second mixed derivatives ∂_θ∂_R E.

**Route HF (default, `resolvent_grad: "hf"`).** `e_pol` is a `jax.custom_jvp` of (consts, u₀, s). All three are differentiable primal inputs; s is **not** a nondiff argument (see the implementation notes).
- **Rule primal.** h* = custom_linear_solve(A_c, u₀) (solve #1, differentiable) and e = s Φ(h*).
- **Rule tangent.** de = ds · Φ(h*) + s[jvp of Φ_q(c; h*) along dc − ⟨du₀, h*⟩], where Φ_q = ½⟨h, Λ_c h⟩ + ½‖K_c h‖². This is linear in (dc, du₀, ds) and uses only forward K.
  - The ds term is essential: the rest of the tangent is ∝ s.
  - At the V+ initialisation s₀ = 0, the only non-zero gradient into the resolvent is dL/ds = Φ(h*) · ∂L/∂E plus the force term −⟨∂_RΦ(h*), F̄⟩. K and Λ start receiving gradients once s ≠ 0, as in a zero-initialised residual branch (ReZero).
  - Without the ds term, s would get zero gradient with no error raised, and arm (a) would stay bit-identical to tied+hill.
- **Forces.** The inner jax.grad transposes the tangent: one VJP of K and of u₀ at fixed h*. **No solve.**
- **Training.** The outer reverse pass differentiates the rule's code.
  - The force contribution reaches h* with cotangent h̄ = s[(∂_R A)·F̄] h* − s(∂_R u₀)·F̄. This is "∂A along the force cotangent" applied to h*: one JVP of K along F̄.
  - Under the outer trace, the rule body is differentiated as ordinary code, so the energy term also reaches h*. Because the rule's primal is evaluated in the Φ form, its cotangent on h* is s ∇_hΦ(h*) = −s r ≈ 0, and the explicit part it leaves is exactly the HF term.
  - JAX sums the cotangents on h* and transposes `linear_solve_p` once: λ = A⁻¹h̄ (solve #2, symmetric).
  - Then dL/dθ = explicit terms + ⟨λ, ∂_θ u₀ − (∂_θ A) h*⟩. This is the classic adjoint for the mixed derivative.
  - For s, the outer pass differentiates the s multiplier inside the rule, giving ∂F/∂s = −∂_RΦ(h*). This needs s to be a primal of the rule, as above.
- **Total:** 2 solves per training step, 1 per inference with forces.
- **Implementation notes.**
  - The custom_jvp takes only array pytrees as arguments: `consts` (the `nnx.split` states and geometry arrays), u₀ and s.
  - Everything static is closed over by a factory, built once per model and batch shape: `make_e_pol_hf(graphdef, n_graphs, tol, maxiter, precond)` returns `e_pol(consts, u0, s)`. The static set is the graphdef, n_graphs, tol, maxiter and the precond mode.
    - None of these may sit inside the traced pytree. Under jit, custom_jvp traces every flattened leaf of its arguments to an abstract value (`partial_eval.py` `DynamicJaxprTrace.process_custom_jvp_call`: `tracers = map(to_jaxpr_tracer, tracers)`, `trace_to_jaxpr_dynamic(fun, in_avals)`).
    - A Python int `n_graphs` inside `ResolventConsts` would therefore become a tracer, and `segment_sum(num_segments=n_graphs)` would raise a concretization error.
    - `nondiff_argnums` is an acceptable alternative for these Python scalars only.
  - **Never** put s or any other parameter in `nondiff_argnums`. JAX applies `stop_gradient` to those arguments (`custom_derivatives.py:273`), which silently zeroes dL/ds.
  - Integer arrays (senders, receivers, species, graph_idx) are ordinary arguments with float0 tangents, which the rule ignores. Do not use `nondiff_argnums` for arrays.

**Route autodiff (`resolvent_grad: "autodiff"`, the reference for tests).** E_pol = s Φ(h*) with h* from custom_linear_solve, and no custom_jvp.
- **Forces:** forward solve plus one transposed solve on the cotangent of h*.
  - With the Φ form that cotangent is s(A h* − u₀) ≈ 0. A purely relative tolerance still spends a full solve on it, since the iteration count does not depend on the norm of the right-hand side. An absolute floor, atol = tol·‖u₀‖, makes it terminate at once.
  - With E = −(s/2)⟨u₀, h*⟩ it is a full solve on −(s/2)u₀.
- **Training:** the outer pass transposes both solves, so 4 solves per step.
- **Required** for r_∞ and for 1C-N, where there is no variational principle. jax `gmres` already passes `transpose_solve`, which uses linear_transpose of the matvec, and that works on cue.

### 5.4 Cost per step, in matvecs
- **Unit.** One A matvec = 1 K + 1 Kᵀ, plus a diagonal, ≈ 0.9 ms per 5-frame batch (probe: 0.48–0.51 + 0.41 ms with Python dispatch). Inside a `while_loop` the per-iteration overhead is lower, so this is an upper estimate.
- **Iteration counts.**
  - Probe (5), unpreconditioned CG on the stand-in mI + MᵀM with the trained σ_max ≈ 3.9 operator: n_it = 44–45 at m = 0.5 and 98–99 at m = 0.1, to 1e-8.
  - GMRES on I − M (the 1C-N analogue with the trained operator) needs 105–114 to 1e-8 and 135–141 to 1e-10.
- **Training step (HF)** ≈ 4 n_it + ~15 K-equivalents:
  - forward CG: 2 n_it;
  - adjoint CG: 2 n_it;
  - force VJP, its second-order differentiation, u₀ and ∂A-along-F̄ terms: ~15.
- **Wall time.**
  - n_it = 45: about 200 K-equivalents ≈ 60–90 ms.
  - n_it = 100: ≈ 130–190 ms.
  - Today's tied+hill step is about 33 ms (90 steps in 3.0 s per epoch; 1500 epochs in 1.5 h), so 1C-V is about 2.5–6× → 1500 epochs in about 4–9 h.
  - The autodiff route doubles the solve part.
- **Memory.** O(10) vectors of 295k float64 per solve, versus the T stored activations of the unrolled sweep.

### 5.5 CG iteration count and conditioning
- **Bound.** κ = λ_max(A)/λ_min(A) ≤ (max Λ + ‖K‖²)/m₀.
  - k ≤ ⌈½√κ ln(2/ε)⌉ bounds the *A-norm* error ‖e_k‖_A/‖e_0‖_A ≤ ε.
  - The solver and the probe stop on the relative *residual*. Since ‖r_k‖/‖r_0‖ ≤ √κ ‖e_k‖_A/‖e_0‖_A, that needs k ≤ ⌈½√κ ln(2√κ/ε)⌉. This residual form is the one used for maxiter and test 7.
- **Check on the probe's stand-in** (κ = (m + σ_max²)/m with σ_max = 3.87–4.02, x₀ = 0):
  - m = 0.5: κ = 31–33, residual bound 58–61 at ε = 1e-8 (A-norm 54–56); measured 44–45.
  - m = 0.1: κ = 151–163, residual bound 133–139 (A-norm 118–122); measured 98–99.
- **What bounds what.** The floor m₀ is enforced by construction, so κ, and with it the iteration count, is controlled only through ‖K(R)‖². That is bounded **per geometry, not uniformly**.
  - Hill bounds the per-node RMS of x̂ (≤ 2), and the radial MLP is bounded on [0, r_c].
  - The aggregation is Σ_{j∈N(i)}/⟨N⟩ with ⟨N⟩ fixed at 16.723. By the Schur test, ‖K(R)‖ ≲ 2 · max_i |N(i)| · sup|w| / ⟨N⟩ · ‖L_1‖‖L_m‖‖L_K‖, up to CG-coefficient constants. So it grows with the actual coordination inside r_c.
  - On 3BPA the coordination is capped by the molecule (27 atoms):
    - at equilibrium (train / test_1200K) the mean |N(i)| is 16.7 / 16.1 and the max is 26;
    - after test 5's ×0.75 compression the mean is 22.4 and the max is still 26.
  - In a water box or a bulk cell it grows with density without a cap.
- **Monitoring.** The CG α/β coefficients form the Lanczos tridiagonal T_k. Its extreme Ritz values estimate λ_max and λ_min (Golub–Meurant), so κ̂ per graph comes for free. Log mean and max of n_it and κ̂ every epoch: they are the conditioning metric.
- **Interpretation.** The iteration cap does not bound κ: a frame that hits maxiter is ill-conditioned, not ill-posed. The solution remains bounded by ‖u₀‖/m₀.

### 5.6 rMD17 molecules where v9a features blow up (1e25–1e53)
- **Cause in v9a.** h_t = M_t⋯M_1 u₀ is a degree-t product with ρ(M_t) = 1.04–1.24 and σ_max up to 8.5, plus non-normal transient growth. That gives the 1e25 (ethanol) to 1e53 (water) layer-4 norms of review §12.4.
- **1C has no operator powers.**
  - The output satisfies ‖h*‖ ≤ ‖u₀‖/m₀ for every geometry and every parameter value: m₀‖h*‖² ≤ ⟨h*, A h*⟩ = ⟨h*, u₀⟩ ≤ ‖h*‖‖u₀‖.
  - The energy satisfies |E_pol| ≤ |s|‖u₀‖²/(2m₀).
  - u₀ = h₁ is a single layer-0 output, linear in the density A_i times a bounded x̂.
- **Where the large v9a gains go.** σ_max = 8.5 in v9a terms gives ‖K‖² ≈ 72. At m₀ = 0.1 that is κ ≈ 720. The residual-form bound of §5.5 then allows up to about 362 CG iterations to relative residual 1e-10 (318 in the A-norm form), inside maxiter = 500: slow but bounded.
- **Without x_norm** (`x_norm: null`, allowed for 1C), ‖K‖ ∝ |x_i| grows polynomially with local density, not exponentially with depth. A stays SPD and h* stays bounded; only κ grows. This is the arm that gets well-posedness *without* bounding the x factor (H2, review §12.5: every bounded-x variant cost 1200 K E).
- **Failure mode.** Hitting maxiter, logged per frame. The "exploding features" failure mode is gone by construction.

---

## 6. Code plan

No edits now to `race_nnx.py`, `train_3bpa.py` or `training_utils.py` (another engineer is editing them). Everything
new goes into **`bam_omat24/resolvent.py`**. The hooks into the three files are listed here for later.

### 6.1 Config keys (`train_3bpa.py`, read with `_get`, passed to `RACE(...)`)
| key | default | meaning |
|---|---|---|
| `resolvent` | `null` | `null` (off, current behaviour) \| `"variational"` (1C-V) \| `"monotone"` (1C-N) |
| `resolvent_mode` | `"replace"` | `"replace"`: layers 0, 1 explicit (strata r₀, r₁), solve instead of layers 2..n−1 \| `"add"`: all strata kept + E_pol (V+) |
| `resolvent_K` | `"own"` | `"own"` (new message path) \| `"layer1"` (share linear_1 / radial_mlp / linear_m / linear_2 of layer 1) |
| `resolvent_source` | `"h1"` | `"h1"` \| `"linear"` \| `"h2"` (§3.5) |
| `resolvent_lambda` | `"species"` | `"species"` \| `"local"` \| `"block"` (§3.3) |
| `resolvent_m0`, `resolvent_m_init` | 0.1, 1.0 | floor and initial value of Λ |
| `resolvent_D` | `"identity"` | `"identity"` \| `"diag"` |
| `resolvent_rank` | `null` | channels per l of H′ (null = hidden) |
| `resolvent_s_init` | 0.1 (0.0 for `"add"`) | initial energy scale s (learnable: a differentiable primal of the HF rule, §5.3) |
| `resolvent_shifts` | 0 | number of extra Stieltjes shifts z_k (along Λ, §3.8) |
| `resolvent_readout` | `false` | gated r_∞(h*) (forces need an adjoint solve) |
| `resolvent_grad` | `"hf"` | `"hf"` \| `"autodiff"` (forced for readout / monotone) |
| `resolvent_tol`, `resolvent_maxiter` | 1e-10, 500 | per-graph relative residual, iteration cap |
| `resolvent_precond` | `"jacobi"` | `"jacobi"` (§5.2, single shift only) \| `"lambda"` (P = Λ) \| `null`; multi-shift requires `"lambda"` or `null` |
| `neumann_steps` | `null` | depth-sweep variant: extra tied applications after the last layer with one readout on h_T (Stage 1b) |

Validation (raise `ValueError`):
- `resolvent` requires:
  - `update="linear"`;
  - `message_norm="avg"` (the layer norm makes K nonlinear);
  - `x_mode="basis"` (x̂ must be fixed per geometry; `node_irreps` makes x depend on h);
  - `feature_norm=null`;
  - `in_layer_order=1` and no `sym_contraction` (both nonlinear in h);
  - `tp_backend` in {cue, e3nn} (both have transpose rules);
  - the `"replace"` mode requires n_layers ≥ 2.
- `resolvent_grad="hf"` with `resolvent_readout` or `"monotone"`: error.
- `resolvent_shifts > 0` with `resolvent_precond="jacobi"`: error (no shift invariance, §3.8).

### 6.2 `bam_omat24/resolvent.py` (new)
```python
class ResolventConsts(NamedTuple):          # pytree of ARRAYS ONLY: every leaf is traced by custom_jvp (§5.3)
    k_state: nnx.State                      # params of the K path (split from ResolventOperator.K)
    radial_message: jax.Array               # (n_edges, n_paths)  precomputed w_ij (edge_scale applied)
    sh: jax.Array                           # (n_edges, sh_dim)
    x_hat: jax.Array                        # (n_nodes, hidden_dim)  normalised x factor of layer 1
    lam: jax.Array                          # (n_nodes, hidden_dim)  Lambda, broadcast per component, unmasked
    mask: jax.Array                         # (n_nodes,)  real-node mask
    senders: jax.Array; receivers: jax.Array; species: jax.Array; graph_idx: jax.Array
    # n_graphs, tol, maxiter, precond mode are static: closed over by the make_* factories below, never stored here

class ResolventOperator(nnx.Module):
    def __init__(self, hidden_irreps, sh_irreps, n_species, radial_basis_size, radial_mlp_size,
                 radial_mlp_layers, mlp_init_scale, avg_n_neighbors, tp_backend, *, kind="variational",
                 lambda_mode="species", m0=0.1, m_init=1.0, D_mode="identity", rank=None, s_init=0.1,
                 n_shifts=0, readout=False, shared_conv: Optional[RACEConvolution] = None, rngs: nnx.Rngs)
    def prepare(self, sh, radial_basis, x_hat, species, senders, receivers, node_mask, graph_idx,
                atomic_basis=None, edge_scale=None) -> ResolventConsts
    def __call__(self, consts: ResolventConsts, u0: jax.Array, n_graphs: int) -> Tuple[jax.Array, Dict]
        # n_graphs = data.n_node.shape[0] (static); builds/caches make_e_pol_*(graphdef, n_graphs, ...) and
        # calls e_pol(consts, u0, self.s[...]) with s as a differentiable argument
        # -> per-node energies (n_nodes,) and aux {iters, rel_res, kappa_est, hstar_norm_ratio}

def K_apply(graphdef, consts: ResolventConsts, h: jax.Array) -> jax.Array          # mask ⊙ L_K((L_m agg(L_1 mask⊙h)) ⊗ x̂)
def KT_apply(graphdef, consts, y) -> jax.Array                                   # jax.linear_transpose of K_apply
def A_apply(graphdef, consts, h) -> jax.Array                                     # lam*h + KT(K h)   (+ skew for "monotone")
def graph_pcg(matvec, b, graph_idx, n_graphs, precond, tol, maxiter, x0=None) -> Tuple[jax.Array, Dict]
    # n_graphs/tol/maxiter Python scalars (static); safe alpha_g/beta_g and where-freezing (§3.7)
def graph_gmres(matvec, b, graph_idx, n_graphs, tol, restart, maxiter) -> Tuple[jax.Array, Dict]
def solve_spd(graphdef, consts, b, *, n_graphs, tol, maxiter, precond) -> Tuple[jax.Array, Dict]
    # lax.custom_linear_solve(partial(A_apply, graphdef, consts), b, solve=graph_pcg(...),
    #                         symmetric=True, has_aux=True)
def jacobi_precond(graphdef, consts, n_probe=4, seed=0) -> jax.Array             # stop_gradient, §5.2
def make_e_pol_hf(graphdef, n_graphs, *, tol, maxiter, precond) -> Callable        # returns e_pol(consts, u0, s):
    # jax.custom_jvp with ALL of (consts, u0, s) differentiable; tangent ds*Phi(h*) + s[...]  (§5.3)
def make_e_pol_autodiff(graphdef, n_graphs, *, tol, maxiter, precond) -> Callable  # e_pol(consts, u0, s)
def make_e_pol_multishift(graphdef, n_graphs, *, tol, maxiter, precond) -> Callable  # e_pol(consts, u0, s_k, z_k); precond in {"lambda", None}
```

### 6.3 Changes to `RACE.__init__` / `RACE.node_energies` (later, when `race_nnx.py` is free)
- `__init__`: if `resolvent`, build `self.resolvent = ResolventOperator(..., shared_conv=self.layers[1] if resolvent_K == "layer1" else None)`.
- `node_energies`:
  1. Keep the x̂ of layer 1 (the `x_feats` after `_hill_normalize` with row `min(i,1)`) as `x_hat_1`.
  2. In the `"replace"` mode, break out of the layer loop after layer 1 (strata r₀, r₁ and the x / f readouts of layers 0, 1 appended as now). In `"add"`, run all layers.
  3. `consts = self.resolvent.prepare(sh, radial_basis, x_hat_1, species, senders, receivers, node_mask, node_graph_idx(data), atomic_basis, edge_scale_1)`, then u₀ = `feats_hist[1]` (or `"linear"` / `"h2"`).
  4. `e_pol, aux = self.resolvent(consts, u0, n_graphs)`; `outputs.append(e_pol)`; `layer_parts["node"].append(e_pol)` (zeros for x / f), so `strata` has one more row.
  5. Optional `r_∞(h*)` stratum.
  6. Return `aux` under `out["resolvent"]` for logging.
- `node_features` (used by the variance heads) becomes h* in `"replace"` and stays h₅ in `"add"`.
- `training_utils`: log the mean and max of `out["resolvent"]["iters"]` and `kappa_est` per epoch. Nothing else changes.
  - The solve runs per device inside `jax.shard_map`. Graphs never span devices, and there are no collectives inside the while_loop.
  - `reduce_on_plateau` and the optax chain are unchanged. The new parameters (`resolvent/…`) are ordinary `nnx.Param` leaves. With `stratum_weight_decay`, `_layer_of` returns None for them, so they are not decayed.
- `train_3bpa.build_model`: read the keys of §6.1 and print them in the model line.

### 6.4 Interaction with the existing options
- **x_norm / hill.**
  - K uses the same x̂ as layers 1.. (x_projs[1] and x_hill row 1 with `tie_operator`). Hill bounds the per-node RMS of x̂ by 2. That bounds ‖K‖, and therefore κ, per geometry, scaling with the coordination (§5.5), not uniformly.
  - `x_norm: null` is allowed: A stays SPD, h* stays bounded, and only κ grows. It is a designated Stage-2 arm, the H2 test.
  - `"spectral"` works too, but it was negative in round 25c.
- **layer_scale.** γ multiplies M's increment. In A, the analogous scale is absorbed by L_K and Λ (A = Λ + γ²KᵀK), so the resolvent has **no** layer_scale. γ stays on the explicit layer 1.
- **tie_operator.**
  - In `"replace"` mode, layers 2..n−1 are never executed, so tying is irrelevant beyond layer 1.
  - With `resolvent_K: "layer1"`, the resolvent reuses the tied operator's message path but not its skip or γ. Parameter count then stays close to tied+hill (2.83 M), minus readouts r₂–r₄, plus Λ (2 k) and s.
  - With `"own"`, a new message path adds about 1.3 M, roughly one untied layer: (6.86 − 2.83)/3.
- **residual.** `"neumann"` for layers 0, 1 as today (h₂ = u₀ + M u₀). The resolvent does not use `prev_features`.
- **stratum_gain.** E_pol has gain 1. `strata_resum="aitken"` must be off; the validator rejects it.

## 7. Tests (`tests/test_resolvent.py`, run through `runs/sbatch_py.sh`, float64)
Small model for speed (hidden 8, lmax 2: 72 components per node, 1944 per 27-atom frame), plus one full-size check
on a 3BPA frame:
1. **E(3) equivariance.** Random rotation, reflection and translation of a 3BPA frame and of a random 12-atom molecule: E invariant to 1e-10 (relative), forces rotate to 1e-9, and h* transforms as ρ(g) h*. cue and e3nn backends.
2. **Finite-difference forces.** Central differences, δ = 1e-4 Å, on 9 random coordinates. HF route and autodiff route each against FD to 1e-7 relative. HF vs autodiff to 1e-9 at tol 1e-12.
3. **Second derivatives (training).**
   - `jax.test_util.check_grads(loss_of_params, order=1, modes=["rev"])` on a loss containing ‖F‖², and a directional FD in θ of ‖F‖² against jvp/grad to 1e-6. This checks the custom_jvp, the custom_linear_solve JVP and the cue transpose / JVP chain together.
   - HF vs autodiff dL/dθ to 1e-8. **s is one of the checked parameters** (check_grads and HF-vs-autodiff), alongside the K, Λ and u₀ parameters.
   - **dL/ds at s = 0** (V+ initialisation): non-zero and equal to the autodiff route to 1e-8, for an energy-only and a force-only loss. One optimizer step from s = 0 must change s. This guards against s being closed over or put in `nondiff_argnums`, which test 10 cannot detect.
4. **SPD check.**
   - Symmetry: ⟨A u, v⟩ = ⟨u, A v⟩ to 1e-12.
   - min over 20 random h of ⟨h, A h⟩/‖h‖² ≥ m₀, and Lanczos λ_min(A) ≥ m₀ − 1e-10, also with the K weights scaled ×10.
   - Padding: A on padded nodes equals Λ_pad, CG converges with zero padded right-hand side, h* = 0 there.
   - Two batch shapes, each for the forward solve and the transposed solve, with no NaN in E, F, dL/dθ:
     - a full batch, i.e. one padding graph holding ≥ 1 padded node;
     - a partial last batch with extra empty graphs (`pad_graphs = batch_size + 1` and fewer real frames).
   - Also assert that α_g, β_g are finite for the padding and empty graphs (§3.7).
5. **Monotone bound.** ‖h*‖ ≤ ‖u₀‖/m₀ and |E_pol| ≤ |s|‖u₀‖²/(2m₀) on 3BPA frames compressed ×0.75 and stretched ×1.3, with random untrained K scaled ×10, on both backends. CG converges below maxiter.
6. **Agreement with the truncated Neumann series when the coupling is small.**
   - With Λ = I exactly and K scaled by 1e-2, M_var = −KᵀK and h* = Σ_k M_var^k u₀.
   - ‖S_T − h*‖ ≤ ‖K‖^{2T}‖u₀‖/(1 − ‖K‖²) for T = 1..5, with the observed rate ‖K‖².
   - For 1C-N with K = 0 and skew B: compare against GMRES on (Λ − (B − Bᵀ)).
7. **CG vs dense on a 27-atom frame.**
   - Small model: assemble A densely with `jax.jacfwd(A_apply)` (1944²), then `jnp.linalg.solve` and `eigvalsh`. h* agrees to 1e-10 and E to 1e-12.
   - The CG iteration count obeys the residual-form bound ⌈½√κ ln(2√κ/ε)⌉ of §5.5, with κ from `eigvalsh` (in the preconditioned coordinates when P ≠ I).
   - Full size (55 296 unknowns, dense 24 GB, too large): CG vs GMRES vs 200-term preconditioned Richardson to 1e-9.
8. **Batch consistency.** Per-frame energies and forces from a batch of 5 vs one frame at a time, within tol, using graph-wise CG.
9. **Multi-shift.**
   - n_shifts = 0 reproduces the single solve exactly.
   - With shifts along Λ and P = Λ, and with shifts along I and P = I, each shifted solution agrees with an independent CG on A + z_kΛ (resp. A + z_kI), and FD forces pass.
   - `resolvent_shifts > 0` with `"jacobi"` raises.
10. **Regression guard.** `resolvent: null` reproduces the current model bit-for-bit (`tests/test_round25.py` unchanged), and `"add"` with s = 0 reproduces tied+hill to 1e-14. This test says nothing about whether s can move; that is test 3.

## 8. Staged experiment plan, numbers to beat, kill criteria
At most 2 concurrent jobs of ours (5-job account limit, one long user job, the engineer's jobs). Each row is
one GPU job. "500 ep" screens use seed 42; promotions use seeds 42–44.

**Reference numbers** (test E meV / F meV/Å RMSE at 300 / 600 / 1200 K):
- 500 ep, s42:
  - tied+hill `r25_ntied_hill`: 3.44/8.99, 8.46/20.71, 33.85/61.25 (valid 3.11/8.58);
  - v9a `r25_v9a_nt`: 3.88/9.68, 9.86/22.13, 29.95/61.06;
  - A+hill: 3.82/9.24, 8.40/20.90, 31.12/60.05.
- 1500 ep × 3 seeds:
  - **tied+hill 2.42/8.11, 7.51/18.89, 30.99/55.58**;
  - **A+hill 2.67/8.18, 7.40/18.87, 28.47/54.46**;
  - **v9a 1500 2.65/8.62, 7.84/20.12, 28.17/57.95**.
- Literature: TACE_max 23.42 at 1200 K E; GRACE 25.80; MACE 29.80.
- The 1200 K E seed sd is about 1–2 meV. One-seed 500-ep differences below 2 meV are not resolved (§12.5).

| stage | arms | code | decision |
|---|---|---|---|
| **0** (done) | probe + D4 on 6396 frames | `runs/diag/probe_1c.py` | Machinery OK. D4 fails within temperature (corr 0.10–0.16 < 0.3). The trained operator is not monotone in the current coordinates (§2 Reading 2). |
| **1a** depth sweep | tied+hill with `n_layers` ∈ {10, 20}, every stratum read out (existing keys; running as `r26_ntied_hill_T{10,20}_s42`) | none, plus `probe_1c.py --run r26_ntied_hill_T{10,20}_s42 --items 3,4` on the final checkpoints | **Series check first:** 1a counts as a resolvent proxy only if ρ(M) < 1 and ‖S_T − h*‖/‖h*‖ ≤ 5 % at T = n_layers on every probed frame (snapshots at epochs 100 / 39 pass, §2.1). **Positive:** 1200 K E falls monotonically with T, with T=20 ≤ 31.8 (≥ 2 meV below 33.85) and 300 K F ≤ 9.4 → Stage 2. **Negative with the check passed:** evidence against gap (i). Close 1C for gap (i) only if 1b is also negative. **Check failed:** 1a says nothing about an exact resolvent; decide on 1b and the Stage-2 screen. |
| **1b** (required before closing gap (i)) | tied+hill, readouts 0–4 + one on h_T, T ∈ {10, 20} (`neumann_steps`) | race_nnx hook | Same energy/force criterion and the same series check. 1a adds 15 strata, so 1b is what separates "more strata" from "more depth". |
| **2** 1C-V screen | (a) V+ add, K=layer1, s₀=0 (s learnable through the ds term of the HF rule, §5.3; test 3 must pass); (b) V replace, K own, hill; (c) V replace, `x_norm: null`; (d) V replace, `resolvent_lambda: "local"` | resolvent.py + hooks + tests | Promote any arm with 1200 K E ≤ 31.8 and 300 K F ≤ 9.4 (500 ep). Kill an arm with 300 K F > 9.9 (+10 % vs tied+hill 500), or mean CG iterations > 300, or step > 6× the baseline. |
| **3** promotion | best 1–2 Stage-2 arms, 1500 ep × 3 seeds | — | **Accept gap (i)** only if 1200 K E ≤ 28.0 (below v9a 28.17 and A+hill 28.47) with 300 K E/F within 3 % of tied+hill (≤ 2.49 / ≤ 8.35) and 1200 K F ≤ 56. **Kill** if 300 K F > 8.92 (+10 %) or 1200 K E > 30.99 (worse than the truncated tied model). |
| 4 | 1C-N (skew B, GMRES, r_∞, m₀ = 0.05); multi-shift (3 shifts along Λ, P = Λ) | resolvent.py | Only if Stage 3 keeps forces but misses energy, i.e. the screening restriction of §3.6 is the suspect. Prerequisite for ranking 1C-N: the fixed-metric test of §2 Reading 2 (does one block-diagonal equivariant P make sym(P(I − M)) ≻ 0 on the probed frames?). Same acceptance and kill numbers. |
| **5** gap (iii) | rMD17 ethanol, malonaldehyde, uracil and water, 1500 ep, best 1C-V config | — | Correctness: max over frames of ‖h*‖/‖u₀‖ ≤ 1/m₀, no maxiter hits, finite energies everywhere. Accuracy: F MAE ≤ v9a (ethanol v9a 2.26 meV/Å MAE; GRACE 1.4). Kill if F MAE is more than 10 % worse than v9a. |

Per-run cost: Stage 1a T = 20 is about 3–4× a tied+hill step, so 500 ep takes about 1.5–2 h. A Stage-2 500-ep
arm takes about 1.5–3 h. A Stage-3 1500-ep run takes about 4–9 h, which fits the 12 h limit; restart is supported.

## 9. Risks
1. **The variational class is far from the learned operator** (probe §2).
   - In the current coordinates, λ_max(sym M) is 2.4 against the ≤ 1 − m₀ required and the skew ratio is 1.0. Both are metric-dependent (§2 Reading 2).
   - Basis-independent: the dominant modes amplify (+0.9), and the complex pair 0.19 ± 0.96i is outside the real spectrum of any 1C-V operator.
   - 1C-V can only screen (§3.6), so the model must find a different solution. The main accuracy risk is the 300 K force guard.
   - Mitigations: Λ(R) "local", the V+ additive arm, and 1C-N.
2. **The case for gap (i) is weak.**
   - D4 failed within temperature. Earlier, uniform bounds (x_norm, every bounded-x variant) cost 1200 K E, and Aitken tail resummation was negative.
   - The probe could not test whether a model *trained* with the exact resolvent extrapolates better.
   - Stage 1a tests that cheaply, but only if the retrained series converges (§2.1, §8). Otherwise 1b and the Stage-2 screen decide.
3. **Energy form.**
   - E_pol is one quadratic form in u₀ per frame, so its growth with source magnitude is quadratic, not saturating. This is consistent with the "too soft at large amplitude" 1200 K bias (§12.5), but it is unproven.
   - A single scalar s may be too few degrees of freedom: multi-shift adds s_k, and `"linear"` u₀ adds channels.
4. **Cost.** 2 solves × 45–140 CG iterations per step means about 2.5–6× step time. If κ grows during training (‖K‖ growing), iteration counts grow too. This is monitored through n_it and κ̂, with maxiter as the guard.
5. **AD plumbing.**
   - custom_jvp combined with nnx (explicit state pytrees), shard_map and while_loop is new in this code base. Tests 2, 3 and 10 must pass before any training run.
   - cue's JVP-of-transpose is exercised only by the second derivatives (test 3).
6. **Inexact solves.** Force error is first order in tol (§4). At 1e-10 it is negligible, but screens at 1e-8 should be checked once against 1e-10 on valid metrics.
7. **Rule interplay.** The `"replace"` mode removes explicit strata 2–4, and the user's "no layer reduction" rule may count that as a reduction even though the message-passing depth becomes infinite. The `"add"` (V+) arm is the rule-safe primary arm in Stage 2. The "fair 3BPA-only protocol" is respected: no extra data or labels.
8. **Probe limits.**
   - One seed (s42) of tied+hill; per-frame spectra on 10 frames only; D4 on all 6396 frames.
   - The T10 / T20 series check (§2.1) is from early snapshots (epochs 100 / 39 of 500) and must be repeated on the final checkpoints.
   - The stand-in mI + MᵀM includes the skip and γ, and is not the trained K of a 1C model. Its iteration counts are cost estimates, not predictions.
