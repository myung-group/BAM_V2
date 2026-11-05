# Paper outline: RACEv2 (draft, 2026-10-30)

**Title (user's).** Highly locally resolved machine learning interatomic potential: RACEv2

**Scope (user, 2026-10-30).** Large-scale data (MPtrj / OMat24 / Matbench Discovery) is not validated yet,
so the paper claims a locally resolved, data-efficient potential — not a universal / foundation model.
Large-scale training is future work (multi-node data parallel and the build_pkl_v2 PBC fix are ready).

**Target venues.** JCTC / npj Computational Materials / Digital Discovery.

**Claim.** RACEv2 (per-layer stratum energies, Gaussian radial regularity prior, Chebyshev residual,
hill-normalised x factor) trained with SOAP plus a closed-form energy-gauge refit reaches the best
published 3BPA numbers on 5–6 of 6 temperature metrics, and improves rMD17 / AcAc, in the
small-data, out-of-distribution regime. Contributions are separated in an ablation table (model vs
optimizer). Limitation stated: no SOAP gain on liquid water; large-scale data untested.

**Positioning of SOAP.** SOAP: Vyas et al. (arXiv 2409.11321); for MLIPs: Harari … Kozinsky
(arXiv 2607.02499, large / direct-force settings). Here it is part of the training method of RACEv2 for
conservative, small-data models, with the energy-gauge analysis and a fast grouped implementation.

## Sections

1. Introduction — data efficiency and extrapolation; optimizer as an inductive bias.
2. Model and training — RACE (A + hill: Chebyshev residual, LayerScale, hill-normalised x factor,
   Gaussian radial regularity prior); losses; datasets; fair protocol (no pretraining / extra data /
   ensembles).
3. SOAP for conservative MLIPs — algorithm; grouped implementation (same maths, 17.7 → 6.8 s / epoch);
   reproducibility note (run-to-run variance from rank-deficient early factors).
4. The energy gauge — single-stoichiometry force-null direction; why sign-like updates of 0-D/1-D leaves
   leave an offset (−6 meV at 500 ep) while AMSGrad does not; closed-form refit (Golub–Pereyra);
   it vanishes by 1500 ep.
5. Results — 3BPA (T and dihedral), AcAc, rMD17 (both test protocols), water, Si GAP-18 (stratified and
   structure-extrapolation splits).
6. Mechanism / ablations — plain Adam β = 0.95 (no rotation) vs SOAP; v9a vs A + hill under SOAP;
   learning rate (1e-2 vs 3e-3: temperature vs dihedral trade-off); where SOAP does not help (water:
   steps per epoch, atoms per frame).
7. Limitations and outlook — water (no SOAP gain), large-scale data untested (MPtrj / Matbench
   Discovery as future work).
8. Negative results (appendix) — faithful DeNS, bond-order term, monotone resolvent / depth, energy
   weights under SOAP, others (one table).

## Figures

| # | Content | Data | Status |
|---|---|---|---|
| F1 | Learning curves valid F vs epoch, AMSGrad vs SOAP, 3BPA / rMD17 / AcAc / water | loss logs | have (seeds pending) |
| F2 | Energy gauge: signed train/test energy offset vs epoch (SOAP vs AMSGrad); RMSE before/after refit | gauge_refit, logs | have (500 ep) |
| F3 | 3BPA error vs temperature with MACE / TACE / GRACE | r28, literature | have 2 seeds; s44 queued |
| F4 | Dihedral scan error by β slice | eval_sets | **pending** (1500-ep checkpoints) |
| F5 | Speed: ms/step breakdown, reference vs grouped SOAP vs Adam | probe / micro / r29_soapg | have |
| F6 | Si GAP-18 error by config type (stratified; extrapolation categories) | si_* runs | queued |

## Tables

| # | Content | Status |
|---|---|---|
| T1 | 3BPA 300/600/1200 K + dihedral, ours (3 seeds) vs MACE / TACE / GRACE / NequIP | 2 seeds; s44 + dihedral pending |
| T2 | rMD17 10 molecules, official split 01, test-1000 and rest-99k, 3 seeds | 1 seed; rest-99k + seeds queued |
| T3 | AcAc 300 / 600 K, 3 seeds | 1 seed; seeds queued |
| T4 | Water E / F (with outlier frame noted) | runs interrupted by node failure; restart queued |
| T5 | Si GAP-18 | queued |
| T6 | Ablations: Adam β = 0.95, v9a + SOAP, lr, energy weight, gauge | Adam / v9a queued; rest have |
| T7 (appendix) | Negative results | have |

## Experiments still needed (queue `runs/queue_1030.txt`, auto-submitted after the cluster outage)

1. Dihedral scan of r28 checkpoints; rMD17 SOAP rest-99k (posteval2).
2. 3BPA SOAP seed 44 (lr 1e-2, 3e-3).
3. Attribution: plain Adam β = 0.95 (lr 1e-2, 3e-3); v9a + SOAP.
4. rMD17 + AcAc SOAP seeds 43, 44.
5. Water restart to 300 ep; Si GAP-18 (stratified / extrapolation × SOAP / AMSGrad).
6. Literature check: how MACE / NequIP / ACE report Si GAP-18 and water (splits, units).

## Open issues to resolve before submission

- Restart is not bit-exact (AMSGrad diverges after resume; cause open) — affects the water runs only.
- Exact literature protocols for water and Si.
- Uncommitted code (multi-node, DeNS, grouped SOAP, gauge, restart): commit only on the user's request.
