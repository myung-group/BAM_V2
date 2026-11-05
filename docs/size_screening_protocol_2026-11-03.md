# Per-dataset size screening protocol (pre-registered 2026-11-03)

Agreed with the user: the methodology is fixed (final RACEv2 recipe); only width/depth may differ per dataset.

**Screening runs (150 ep, 1 seed, unified prior UFF-nb w0.3 unless the prior was shown neutral):**
- Si GAP-18 (train_full, valid 5 %): L5h128, L3h128, L5h64, L3h64 (`siS_*`)
- LiPS 2500-frame learning-curve setting (valid lc_valid): same four (`lipsS_*`)
- rMD17 (aspirin, ethanol, uracil as representatives; one size for all 10 molecules): L5h128, L5h256, L3h128, L5h64 (`rmdS_*`)
- Water split 1 (train_only / valid, lr 1e-3, E:F 100:1): L5h64, L3h64, L4h64, L5h32 (`wS_*`)

**Selection rule (fixed before results; validation only, never test):**
1. candidate set = configs whose final-epoch validation energy RMSE is within +10 % of the best one;
2. choose the lowest final-epoch validation force RMSE among candidates;
3. ties (< 2 %) -> the smaller model.
Caveat recorded: short-run winners may differ from long-run winners (water: 2-layer models led early on energy and lost on force); the rule is applied as stated anyway and reported.

**Then:** the selected size is trained to convergence (Si 300+ ep, LiPS 150-500 ep, water 400 ep) with 3 seeds.
rMD17 (10 molecules) and AcAc: final recipe, 1500 ep x 3 seeds (configs `rmd17f_*`, `acacf_*`, both prior variants prepared; queued once the recipe is decided by the pre-registered 3BPA rule).
