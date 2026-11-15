#!/bin/bash
# Copies the small result files of the OMol25 4M v3 run into the package (docs/results_omol4m_v3), 2026-11-16.
# Small files only (cp of text / json / png): no checkpoints, no size tables, no extxyz.  Re-run after new results.
# usage: bash collect_results.sh [label ...]      labels of md_water/eval/<label> to copy (default v3_e20 v3_e70)
set -euo pipefail
MD=/scratch/t510a13/md_water
RUN=/scratch/t510a13/BAM_V2_run_v3/runs/omol4m/omol4m_v3_polarqeq_c6_cs_ct_L5x128_r12p6_spec_hub_spinE_nbo_csloc_decL_disp_xrn3_lr1e-2_soap_g48x12_s42
SP=/tmp/claude-500022713/-scratch-t510a13-race-lr/09325077-957d-4b0c-93ee-81439d057b0f/scratchpad
OUT=/scratch/t510a13/BAM_V2_wt_thole/docs/results_omol4m_v3
LABELS=("$@"); [ ${#LABELS[@]} -gt 0 ] || LABELS=(v3_e20 v3_e70)
mkdir -p "$OUT"/{training,eval,official_val,prep,figures}

# training: the loss log (config + per-epoch table + neutral_val tests); "*.out" is in .gitignore, hence .txt
cp "$RUN/loss.out" "$OUT/training/loss_log.txt"

for L in "${LABELS[@]}"; do
    if [ -d "$MD/eval/$L" ]; then
        mkdir -p "$OUT/eval/$L/iw"
        for f in bench.json bench_report.txt bench_report.png groups.json spin_eval.json spin_eval.txt; do
            [ -f "$MD/eval/$L/$f" ] && cp "$MD/eval/$L/$f" "$OUT/eval/$L/"
        done
        for f in compare.png compare.txt ion_water.json ion_water.png ion_water.txt; do
            [ -f "$MD/eval/$L/iw/$f" ] && cp "$MD/eval/$L/iw/$f" "$OUT/eval/$L/iw/"
        done
    fi
    [ -f "$MD/eval/debug_iw_step_$L/iw_step.json" ] && cp "$MD/eval/debug_iw_step_$L/iw_step.json" "$OUT/eval/$L/iw_step_0p05A.json"
    [ -f "$MD/eval/debug_decomp_$L/decomp.json" ] && cp "$MD/eval/debug_decomp_$L/decomp.json" "$OUT/eval/$L/energy_decomposition.json"
    for e in json txt; do
        [ -f "$MD/eval/size_consistency/$L.$e" ] && cp "$MD/eval/size_consistency/$L.$e" "$OUT/eval/$L/size_consistency.$e"
    done
    # official val (32k sample, per-structure errors): gzip, 5.5 MB -> about 1.5 MB
    [ -f "$MD/val_omol25/$L.json" ] && gzip -9 -c "$MD/val_omol25/$L.json" > "$OUT/official_val/$L.json.gz"
    # group table of the full official-val run, from the newest val_eval log that scored 32000 structures of $L
    VL=$(grep -l "^$L: 32000 scored" /scratch/t510a13/logs/val_eval_*.out 2>/dev/null | tail -n 1 || true)
    [ -n "$VL" ] && grep -E "^\s+\S.*n=\s*[0-9]+\s+E_MAE|^== |scored" "$VL" | cut -c1-200 > "$OUT/official_val/${L}_summary.txt"
    for t in test200 noauto_test200; do
        [ -f "$MD/val_omol25/${L}_$t.json" ] && cp "$MD/val_omol25/${L}_$t.json" "$OUT/official_val/"
    done
done
[ -f "$MD/eval/formal_assign/check_300.json" ] && cp "$MD/eval/formal_assign/check_300.json" "$OUT/eval/formal_assign_check_train300.json"

# preprocessing: per-element E0 + meta + the size-table summaries (the tables themselves stay in /appsdata)
prep() {   # <name> <dir>
    [ -d "$2" ] || return 0
    mkdir -p "$OUT/prep/$1"
    for f in e0.json meta.json; do [ -f "$2/$f" ] && cp "$2/$f" "$OUT/prep/$1/"; done
    for s in train valid neutral_val; do
        [ -f "$2/sizes_$s/sizes_meta.json" ] && cp "$2/sizes_$s/sizes_meta.json" "$OUT/prep/$1/sizes_${s}_meta.json"
    done
    return 0
}
prep omol4m_c5 /appsdata/t510a13/omol25/prep_4M
prep omol4m_c6 /appsdata/t510a13/omol25/prep_4M_c6
prep opoly26_c6 /appsdata/t510a13/prep/opoly26_c6.0
prep oc25_c6 /appsdata/t510a13/prep/oc25_c6.0
prep omol_full_c6 /appsdata/t510a13/prep/omol_full_c6.0

# figures made in this session
# (only the newest per-epoch loss figure)
EP=$(ls -t "$SP"/figs/v3_epochs_*.png 2>/dev/null | head -n 1)
for f in $EP "$SP"/figs/v3_loss_now.png "$SP"/figs/v3_summary*.png "$SP"/figs/bench_vs_public*.png "$SP"/figs/iw_vs_public*.png; do
    [ -f "$f" ] && cp "$f" "$OUT/figures/"
done
ls -la "$OUT" "$OUT"/*/ | awk 'NF>=9 {s+=$5} END {printf "total %.1f MB\n", s/1e6}'
find "$OUT" -type f | wc -l
