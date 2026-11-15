#!/bin/bash
# eval_x64.sh <snapshot.pkl> <tag>: float64 evaluation of a checkpoint (BAM_PREDICT_X64=1): 3k WBM sample, kappa on the
# 103 PhononDB structures (out-of-memory materials rerun with APC=256), full WBM (2 relax jobs + score/RMSD), CPS.
cd /scratch/t510a13/race_v2/work
S=$1; T=$2; X='--exclude=node[0449-0484,0499-0576]'; P='-p exclusive06 --comment=field=sciai;appl=jax --nodes=1 --exclusive --gres=gpu:4'
W=/scratch/t510a13/probabilistic_symmetrization/bundle/scratch/t589a02/mp_data/wbm; PY=/scratch/t510a13/envs/bam_jax/bin/python
fin() { until St=$(sacct -j $1 -X -n -o State 2>/dev/null | awk '{print $1}'); echo "$St" | grep -q -E '^(COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)'; do sleep 60; done; sacct -j $1 -X -n -o JobID,JobName%11,State,Elapsed,Start,NodeList; }
J3=$(sbatch --parsable -J wbmsample3k $P -t 00:40:00 $X --export=ALL,CKPT=$S,TAG=sample3k_${T}_x64,NSH=86,SHARDS=0,BAM_PREDICT_X64=1 scripts/eval/mbd/relax_wbm.sh)
JK=$(sbatch --parsable -J kappafull $P -t 02:00:00 $X --export=ALL,CKPT=$S,TAG=kappa_${T}_x64,NSH=4,BAM_PREDICT_X64=1 scripts/eval/mbd/kappa.sh)
C=CKPT=$S,TAG=wbm_${T}_x64,NSH=8,BAM_PREDICT_X64=1
A=$(sbatch --parsable -J wbmfull_a $P -t 02:00:00 $X --export=ALL,$C,SHARDS="0 1 2 3",STAGE=relax scripts/eval/mbd/eval_wbm.sh)
B=$(sbatch --parsable -J wbmfull_b $P -t 02:00:00 $X --export=ALL,$C,SHARDS="4 5 6 7",STAGE=relax scripts/eval/mbd/eval_wbm.sh)
SC=$(sbatch --parsable -J wbmfull_s $P -t 02:00:00 $X --dependency=afterok:$A:$B --export=ALL,$C,STAGE=score scripts/eval/mbd/eval_wbm.sh)
echo "$T float64: 3k $J3, kappa $JK, full WBM relax $A $B score $SC  $(date '+%F %T')"
fin $J3; TG=sample3k_${T}_x64
OPENBLAS_NUM_THREADS=1 bash scripts/eval/mbd/score_sample.sh $TG 2>&1 | grep -E '^unique' | cut -c1-230
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $PY -I scripts/eval/mbd/rmsd_sample.py --relaxed runs/mbd/$TG/relaxed --mp2020-cse $W/2022-10-19-wbm-computed-structure-entries.jsonl.gz --summary $W/2023-12-13-wbm-summary.csv.gz --output runs/mbd/$TG/rmsd.csv.gz 2>&1 | grep -E '^unique' | cut -c1-200
fin $JK; K=runs/mbd/kappa_${T}_x64; grep -a -A12 -E '^\{' runs/slurm/kappafull_$JK.out | grep -E 'kappa_SRME"|n_scored|n_censored|RESOURCE|imaginary'
BAD=$(grep -l -E 'RESOURCE_EXHAUSTED' $K/kappa/mp-*.json 2>/dev/null)
if [ -n "$BAD" ]; then
  mkdir -p $K/oom_first_try; mv $BAD $K/oom_first_try/; cp $K/kappa/kappa_srme.json $K/oom_first_try/kappa_srme_first_try.json
  JR=$(sbatch --parsable -J kappafull $P -t 01:00:00 $X --export=ALL,CKPT=$S,TAG=kappa_${T}_x64,NSH=4,APC=256,BAM_PREDICT_X64=1 scripts/eval/mbd/kappa.sh)
  echo "kappa rerun of $(echo $BAD | wc -w) out-of-memory materials: $JR"; fin $JR
  grep -a -A12 -E '^\{' runs/slurm/kappafull_$JR.out | grep -E 'kappa_SRME"|n_scored|n_censored|RESOURCE|imaginary'
fi
until grep -a -q -E '^unique_prototypes .*MAE' runs/slurm/wbmfull_s_$SC.out 2>/dev/null; do sleep 60; sacct -j $SC -X -n -o State | grep -q -E 'FAILED|CANCELLED|TIMEOUT|NODE_FAIL' && break; done
echo "full WBM energy metrics $(date +%T)"; grep -a -E '^(all|unique_prototypes|MAE by)' runs/slurm/wbmfull_s_$SC.out | head -n 3 | cut -c1-240
fin $SC; grep -a -E 'RMSD' runs/slurm/wbmfull_s_$SC.out | head -n 2 | cut -c1-200
$PY -I - <<PYEOF
import json, re, ast
s = json.load(open('$K/kappa/kappa_srme.json')); s = s.get('summary', s); k = s['kappa_SRME']
L = open('runs/slurm/wbmfull_s_$SC.out').read()
u = [ast.literal_eval(m) for m in re.findall(r"^unique_prototypes (\{.*\})", L, re.M)]
f1 = u[0]['F1']; rm = u[1]['RMSD']
print('$T float64: F1 %.3f MAE %.4f R2 %.3f RMSD %.4f kappa SRME %.3f  CPS %.4f' % (f1, u[0]['MAE'], u[0]['R2'], rm, k, 0.5 * f1 + 0.4 * (1 - k / 2) + 0.1 * (1 - rm / 0.15)))
PYEOF
date '+%F %T'
