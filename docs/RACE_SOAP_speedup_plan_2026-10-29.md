# SOAP speed-up plan for RACE 3BPA (r26 config, one GH200, float64)

Sources for the numbers:
- **[probe]** is `runs/diag/soap_fix/probe64.txt`, measured.
- **[micro]** is `/scratch/t589a02/BAM_V2_mgpu/soap_speed/micro.txt`. It was measured on a GH200 in float64 using synthetic factors at the real r26 factor sizes, taking the median of 7 runs. The run was cut off before the eigh timings.
- **[numpy]** is `numpy_orth_check.txt`, a CPU accuracy check.
- **est.** marks a projection that has not been measured.

Job 1178813 ended at 19:57:56. The end-to-end grouped benchmark (`bench_soap.py`, `run_smoke.sh`, `run_full1.sh`, `run_full2.sh`) and the draft (`soap_grouped_draft.py`, `test_soap_grouped_draft.py`) have therefore **never run**. No repo file was edited.

## 1. Why SOAP is slow here

**Measured cost per optimizer update [probe], with the r26 config at its default `max_precond_dim` (full two-sided, state 1040 MiB):**

| | optax Adam | current `scale_by_soap` |
|---|---|---|
| regular step | 5.36 ms | 31.0 ms (40.6 ms with mpd=1024) |
| refresh step (every 10th: power step + QR) | 5.3 ms | 887–900 ms (808–831 ms with mpd=1024) |
| first step (eigh) | 9.8 ms | 803 ms (589 ms with mpd=1024), once per run |
| amortized over f=10 | 5.4 ms | 116.7 ms |
| optimizer time per 90-step epoch (81 regular + 9 refresh) | ~0.5 s | ~10.5 s |

**Epoch accounting.**
- An AMSGrad epoch takes about 5 s, so the non-optimizer work is about 4.5 s per epoch.
- 4.5 s + 10.5 s ≈ 15 s, but SOAP epochs measure 17–20 s. That leaves a residual **R ≈ 2–5 s per epoch that nobody has explained**. Candidates are the host syncs from the two `lax.cond` per step, memory pressure, and logging; none has been verified.
- About 76% of the explained overhead is refresh (9 × ~0.89 s ≈ 8.0 s) and about 24% is regular steps (81 × 31 ms ≈ 2.5 s).

**Root cause: the work is bound by launch latency, not by FLOPs.**
- r26 has 196 leaves. 160 of them are matrix leaves in 14–15 distinct shapes. Species-indexed 3-D leaves are flattened per slice, so the 128×128 group alone has batch 153.
- These leaves hold 496 Shampoo factors in 11 sizes: 2944:4, 896:9, 768:18, 512:10, 128:361, 64:60, 24:5, 6:2, 4:6, 2:5, 1:16.
- **Regular step:** about 960 GEMMs plus about 500 fusions, plus two `lax.cond`s that sync with the host. That is about 1,450–1,650 kernels, roughly 20–25 µs each at the measured 31 ms. The compute is only 35.9 GFLOP; the cost-model floor is about 0.8 ms.
- **Refresh:** about 22 XLA ops per leaf, so about 3,500 ops, plus 640 cuSOLVER geqrf/orgqr calls.
  - Looping Householder QR over all 496 factors costs **417 ms [micro]** by itself.
  - F@Q is computed twice (once for the eigenvalue estimate, then again after the permutation), which wastes 238 GFLOP per refresh.
- The literature describes this regime. Shi et al. (Distributed Shampoo) say blocks below 1024 do not get cheaper "due to CPU-GPU latency and additional kernel launches". The SOAP paper says the overhead floor at large f is the per-step rotation matmuls.
- The wide leaves are not the main cost. The four (64, 2944) leaves account for about 75–80 ms of each refresh [probe: mpd=1024 saves about 75 ms per refresh] and 81 of 417 ms of the looped QR [micro].

**QR cost by factor size, in ms [micro]:**

| k × B | per-matrix Householder | batched `jnp.linalg.qr` | CholQR2 | shifted CholQR3 | F@Q |
|---|---|---|---|---|---|
| 2944×4 | 81.3 | 81.4 | 39.8 | 60.1 | 3.7 |
| 896×9 | 35.7 | **1717** | 5.6 | 8.4 | 0.4 |
| 768×18 | 58.7 | **1317** | 5.6 | 8.4 | 0.5 |
| 512×10 | 19.7 | **258** | 2.5 | 3.7 | 0.2 |
| 128×361 | 204 | 18.1 | 1.4 | 2.1 | 0.2 |
| 64×60 | 14.0 | 2.2 | 0.36 | 0.47 | 0.15 |
| all 11 groups | 417 | 3395 | 56.6 | 84.8 | 5.8 |

- **Batched QR trap:** on GPU in jax 0.12, batched Householder QR is 13–48× *slower* than looping for k ≥ 512, and 6–11× faster for k ≤ 128.
- **float32 does not help:** CholQR2 is 53.3 ms in float32 against 56.6 ms in float64, and the full looped Householder path in float32 is no faster (the 'hh32' column at 2315 ms is the batched path). FP64 throughput on GH200 is about the same as FP32.

## 2. Ranked options

Epoch time is estimated as 4.5 s + 90 × (amortized optimizer ms) + R, with R between 0 and 5 s. R = 0 assumes the residual comes from the per-leaf dispatch and the conds.

| Rank | Change | Same math? | Optimizer ms/step | Epoch time | Quality risk / test | Effort | Files / functions |
|---|---|---|---|---|---|---|---|
| **1** | **Shape-grouped stacking.** (a) A static plan keyed `(m, n, left, right, dtype)`, with 3-D species leaves becoming `count` slices. (b) The regular step runs as about 6 batched GEMMs plus 3 fusions per group, about 150 kernels in total instead of about 1,500. (c) The refresh is grouped by factor size: one Y = F@Q, the estimate taken from Y, Y's columns permuted (removes the duplicate GEMM), and m re-projected in batches. (d) Householder QR is dispatched by size: batched for k ≤ 128–256, Python-unrolled per matrix for k ≥ 512, and not `lax.map`. (e) The k=1 sides are dropped (the basis is ±1 and the sign cancels). | **Yes**, up to round-off. Reusing F@Q is exact because F(QP) = (FQ)P. | regular ~3–5 (est.; report 1 said 5–10). Refresh ~230 (est.): 217 [micro] Householder + ~6 F@Q + ~5 sort/re-projection. Amortized at f=10: 26.2 (est.). | ~6.9 s + R (est.), against 17–20 s today | No risk if the test passes. Equivalence test against the current `scale_by_soap`: step 0 on the old code, convert the state, then 35 steps (3 refreshes) on both, comparing updates at rtol 1e-9. Also compare the rotation-invariant state: the factors, `QL·mu·QRᵀ`, and nu. Use full-rank or seeded factors, because null-space bases are arbitrary in both versions. Include a mid-run pickle round trip. Then run the smoke test on the real r26 checkpoint state. | Medium–high. The draft exists but is untested. | New `bam_omat24/soap_grouped.py`, or a new function next to `scale_by_soap` in `soap_optax.py` (keep the old one as the reference). `train_3bpa.py` `build_model` 'soap' branch: new config key. New test next to `tests/test_soap.py`. Draft: `/scratch/t589a02/BAM_V2_mgpu/soap_speed/soap_grouped_draft.py` and `test_soap_grouped_draft.py`. |
| **2** | **Regularised shifted CholeskyQR3 in place of Householder**, applied to (F/trF + δI)·Q_perm with δ ≈ 1e-12 (6.7e-13 at k=128, 1.5e-11 at k=2944). The eigenvectors of F are unchanged. If the result is not finite or trF = 0, keep the old Q; optionally fall back to Householder. Built entirely from batched GEMM + potrf + trsm. | **Near-identical.** Q equals the Householder Q up to column signs, and SOAP's update does not depend on those signs. Only directions with eigenvalue below about δ·trF differ: they keep the old basis instead of Householder's arbitrary completion, and they converge very slightly more slowly. [numpy] gives orthogonality ~5e-15 and an eigenvalue-weighted difference from Householder of 5e-10 at spread 1e4 and 1.3e-7 at spread 1e8, with no failures. | refresh ~95–102 (est.), of which 84.8 [micro] is QR. Amortized at f=10: 12.9 (est.). | ~5.7 s + R (est.) | Very low. Test against grouped Householder on the real checkpoint state and on an early state (10 random steps): ‖QᵀQ−I‖ < 1e-12, weighted basis difference ≤ 1e-5, no NaN, log the fallback count. Optional short 3BPA screen (decision rule in section 3). | Low once rank 1 exists. | The `orth()` dispatch in the grouped module. |
| **3** | **Replace the two per-step `lax.cond`s.** Either use one `lax.switch` (one sync instead of two), or compile static first/regular/refresh step variants chosen on the host from the step counter. Google's `distributed_shampoo` avoids `lax.cond` for the same reason (`efficient_cond`). | **Yes.** | Not measured. It may account for part of the unexplained residual R (2–5 s per epoch). | Unknown; measure it | None. Bitwise-equal updates against rank 1. | `lax.switch` is low effort. Static variants are medium, because the train-step and optax API must expose a static refresh flag. | Grouped module. `train_3bpa.py` train loop and step compilation. |
| 4 | **Larger `precondition_frequency`** | **No**, it is a hyperparameter change. With β_s = 0.95 the factor half-life is about 13.5 steps, so the basis drifts on the f ≈ 10 time scale. | After ranks 1+2, amortized: 12.9 / 7.0 / 5.0 / 4.0 at f = 10/25/50/100 (est.). | Saves ≤ ~0.8 s per epoch beyond rank 2 (est.) | Medium. The 18–27% force gain was measured at f=10. In the literature SOAP is robust from f=1 to 100 at constant batch×f, but NVIDIA (arXiv 2607.20548) needed QR every step. Needs a 3BPA screen. Once the refresh is cheap, f=5 is also worth screening for quality. | Config only | `soap_precondition_frequency` in the input JSON |
| 5 | **One-sided (64, 2944) leaves** (`max_precond_dim=1024`), or block the 2944 side into 23 × 128 blocks that join the 128 group | **No.** It affects 0.75M parameters in the radial-MLP output weights. | Saves ~75 ms per refresh today [probe]; after rank 2, ~40–60 ms (est.) of the remaining ~100. Today's regular step is *slower* with mpd=1024 (40.6 vs 31.0 ms [probe]). | Saves ~0.4–0.5 s per epoch after rank 2 (est.) | Medium. The SOAP paper reports a larger loss increase for one-sided; Anil et al. report no quality drop from blocking. Needs a 3BPA screen. Saves 0.53 GB of state. | Config (mpd) or medium (blocking) | Config, or the grouped plan builder |
| 6 | Plain CholeskyQR2 with δ ≈ 1e-7 | **Small change.** Weighted basis difference 5e-3 to 7e-3 against Householder [numpy]. Unregularised CQR2 fails on our rank-deficient factors: (64,1), (128,1), (128,6), (4,128), and the 2944 side while t < 46. | ~28 ms per refresh cheaper than rank 2 (56.6 vs 84.8 [micro]) | Saves ~0.25 s per epoch | Low–medium. Not worth the deviation. | Low | `orth()` |
| – | Not worth it: float32 or bf16 (≤ 10% gain, worse accuracy); Newton–Schulz or polar in place of QR (a mathematical no-op, since polar(FQ) = Q for SPD F [numpy: 5.6e-14]); CPU or async refresh; adaptive τ-skipping (no saving with static batched shapes); multi-GPU split (one GPU per run); switching to PSGD, SPlus or KL-SOAP (different optimizers); grouping the eigh for the first step (once per run). | | | | | | |

The state layout has two options:
- **Per-leaf (drop-in):** keep the current `ScaleBySoapState` and stack and split it on every step. This costs about 1 ms per step (est.) and needs no checkpoint change.
- **Grouped:** a new `ScaleBySoapGroupedState` plus an exact `convert_opt_state(..., to='grouped'|'per_leaf')` called after `ckpt['opt_state']` loads.

Start with the drop-in. Switch to grouped only if the profile shows the stack and split matter.

## 3. Recommended implementation order and acceptance criteria

Implement everything as a **new transform beside the existing `scale_by_soap`**, selected by a config key with the old path as the default. The queued jobs (soapR1–R3, r28soap) import the current code and must not change.

1. **Rank 1: grouped Householder, mathematically identical.**
   - Run `test_soap_grouped_draft.py`, then `run_smoke.sh` (real r26 state), then `run_full1.sh`.
   - Accept if:
     - updates match `scale_by_soap` at rtol ≤ 1e-9 (atol 1e-12 · max|u|) over 35 steps with 3 refreshes;
     - the rotation-invariant state matches at rtol ≤ 1e-9;
     - per-leaf ↔ grouped conversion and pickle round trips are bit-exact;
     - regular step ≤ 8 ms, refresh ≤ 300 ms, measured SOAP epoch ≤ 10 s.
   - If that epoch target is missed, profile the residual R before going further.
2. **Rank 3: one `lax.switch` or static step variants.**
   - Accept if updates are bitwise equal to step 1 and the epoch time does not regress. Record how much of R it removes.
3. **Rank 2: regularised shifted CholeskyQR3 with δ ≈ 1e-12·trF and a finite-check fallback.** Run `run_full2.sh` for the accuracy table on the checkpoint state and on an early state.
   - Accept if ‖QᵀQ−I‖ ≤ 1e-12, the eigenvalue-weighted basis difference against grouped Householder is ≤ 1e-5, there are 0 NaNs, and the fallback rate is < 1%.
   - Also accept on updates matching step 1 to rtol ≤ 1e-6 over 35 steps with full-rank factors.
   - Speed targets: refresh ≤ 130 ms, amortized ≤ 15 ms per step, epoch ≤ 8 s.
   - **3BPA screen (optional but cheap):** one seed (s42), r26 config, lr 1e-2, the same number of epochs as a current SOAP reference run. Keep it if the 300/600/1200 K energy and force MAEs are within the seed-to-seed spread of the existing SOAP replicates. If no replicate spread exists, use ≤ 2% relative.
4. **Ranks 4 and 5: changes to the math, only after 1–3.** Screen f ∈ {5, 20} and mpd=1024 (or 128-blocking of the 2944 side) with the same decision rule.
   - Adopt only if the force MAE is within noise and the epoch saving is worthwhile, which will be ≤ ~1 s per epoch at that point.

**Overall target, estimated:** SOAP epoch time from 17–20 s down to about 6–8 s, with identical math (steps 1–3). That is about 1.2–1.6× the AMSGrad epoch (~5 s), down from 3.4–4×.

The supporting files are all in /scratch/t589a02/BAM_V2_mgpu/soap_speed/: `micro.txt`, `numpy_orth_check.txt`, `cost_model.txt`, `r26_leaves.json`, `soap_grouped_draft.py`, `test_soap_grouped_draft.py`, `bench_soap.py`, `run_smoke.sh`, `run_full1.sh` and `run_full2.sh`. Nothing in the repo was edited.