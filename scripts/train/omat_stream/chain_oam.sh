#!/bin/bash
# One allocation for the whole chain (2026-11-16): (1) OMat24 pretraining continued from a checkpoint, (2) MPtrj + sAlex
# fine-tuning from its end, (3) float64 evaluation of the final model: full WBM relaxation on up to 8 nodes, kappa on
# one node, scoring + RMSD, CPS line.  Each training stage resumes from its .last.pkl (restart: true in the configs).
# Checked end to end in the old layout (branch racev2, job 1285602: 2 nodes, short configs); not run in this layout yet.
# Configs of the planned run (12 nodes): configs/omat_pretrain/race_v2_omat24_pretrain_continue.json and
# configs/oam_finetune/race_v2_oam_finetune_after_continue.json (virtual epochs are set for 48 GPUs).
#   $ sbatch -J oamchain -p normal --comment="field=sciai;appl=jax" --nodes=12 --ntasks-per-node=1 --exclusive --gres=gpu:4 -t 1-12:00:00 --export=ALL,XLA_EXTRA=--xla_gpu_shard_autotuning=false scripts/train/omat_stream/chain_oam.sh <pretrain.json> <finetune.json> <tag>
#   TEST=1: the evaluation relaxes 8 structures per shard and 1 phonon material (small check of the chain)
#   STOP_AFTER=1: stop after the pretraining (a later submission of the same command goes on with stage 2: a stage
#   whose log is at its last epoch is not run again); PHONON_DIR: folder with the PhononDB files (kappa.sh)
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd ${RACE_WORK:-/scratch/t589a02/BAM_V2_more/work}
A=${1:?pretraining config}; B=${2:?fine-tuning config}; T=${3:?tag}
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/home01/t589a02/.conda/envs/bam_cue}
val() { python -I -c "import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" $1 $2; }
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST)); NN=${#NODES[@]}
echo "chain $T  nodes $NN (${NODES[*]})  A $A  B $B  test ${TEST:-0}  $(date '+%F %T')"
stage() {   # <config>: train to the last epoch of the config; 0 when the log has that epoch
  local CFG=$1 NE LOG LE TRY; NE=$(val $CFG n_epochs); LOG=$(val $CFG log_file)
  LE=$(grep -a -E '^[0-9]+/[0-9]+/[0-9]+ ' $LOG 2>/dev/null | tail -n 1 | awk '{print $3}')
  [ "$LE" = "$NE" ] && { echo "stage $CFG: the log is at epoch $NE of $NE already, not run again"; return 0; }
  for TRY in 1 2; do
    bash scripts/train/omat_stream/pretrain.sh $CFG
    LE=$(grep -a -E '^[0-9]+/[0-9]+/[0-9]+ ' $LOG 2>/dev/null | tail -n 1 | awk '{print $3}')
    [ "$LE" = "$NE" ] && return 0
    echo "stage $CFG stopped at epoch ${LE:-none} of $NE (attempt $TRY of 2)  $(date '+%F %T')"
    [ $TRY = 2 ] && break
    # the next step starts only after the steps of this job are gone and a pause: a step launched while a stopped
    # one was still being cleaned up failed on 5 of 12 nodes with "Error configuring interconnect" (job 1288438)
    for W in $(seq 1 30); do squeue -h -s -j $SLURM_JOB_ID -o %i | grep -q -E '\.[0-9]+$' || break; sleep 60; done
    echo "steps gone after $((W * 60)) s at most, pause ${RELAUNCH_PAUSE:-180} s  $(date '+%F %T')"; sleep ${RELAUNCH_PAUSE:-180}
  done
  return 1
}
echo "==== stage 1: pretraining  $(date '+%F %T')"
stage $A || { echo "CHAIN STOPPED in stage 1"; exit 1; }
CK=$(val $A ckpt_name); INIT=$(val $B init_params_from); sleep 45
command cp -pf ${CK%.pkl}.last.pkl $INIT && ls -l $INIT | awk '{print "pretraining checkpoint", $5, $9}'
[ "${STOP_AFTER:-}" = 1 ] && { echo "CHAIN STOPPED after stage 1 (STOP_AFTER=1)  $(date '+%F %T')"; exit 0; }
echo "==== stage 2: fine-tuning  $(date '+%F %T')"
stage $B || { echo "CHAIN STOPPED in stage 2"; exit 1; }
CK=$(val $B ckpt_name); S=runs/mbd/snap/${T}_ft_final.pkl; sleep 45
command cp -pf ${CK%.pkl}.last.pkl $S && ls -l $S | awk '{print "final checkpoint", $5, $9}'
echo "==== stage 3: float64 evaluation  $(date '+%F %T')"
K=$(( NN > 1 ? NN - 1 : 1 )); [ $K -gt 8 ] && K=8; NSH=$(( 4 * K )); TG=wbm_${T}_x64; KT=kappa_${T}_x64; J=$SLURM_JOB_ID
on() { local n=$1; shift; srun --nodes=1 --ntasks=1 --cpus-per-task=288 -w $n env "$@"; }
for i in $(seq 0 $((K - 1))); do
  SH="$((4 * i)) $((4 * i + 1)) $((4 * i + 2)) $((4 * i + 3))"
  if [ "${TEST:-0}" = 1 ]; then
    on ${NODES[$i]} CKPT=$S TAG=$TG NSH=$NSH "SHARDS=$SH" LIMIT=8 BAM_PREDICT_X64=1 bash scripts/eval/mbd/relax_wbm.sh > runs/slurm/chain_${J}_relax$i.log 2>&1 &
  else
    on ${NODES[$i]} CKPT=$S TAG=$TG NSH=$NSH "SHARDS=$SH" STAGE=relax BAM_PREDICT_X64=1 bash scripts/eval/mbd/eval_wbm.sh > runs/slurm/chain_${J}_relax$i.log 2>&1 &
  fi
done
KN=${NODES[$((NN - 1))]}; KL=runs/slurm/chain_${J}_kappa.log
( on $KN CKPT=$S TAG=$KT NSH=4 ${TEST:+LIMIT=1} BAM_PREDICT_X64=1 bash scripts/eval/mbd/kappa.sh > $KL 2>&1
  BAD=$(grep -l -E 'RESOURCE_EXHAUSTED' runs/mbd/$KT/kappa/mp-*.json 2>/dev/null)
  if [ -n "$BAD" ]; then                                        # out-of-memory supercells again with smaller calls
    mkdir -p runs/mbd/$KT/oom_first_try; mv $BAD runs/mbd/$KT/oom_first_try/
    on $KN CKPT=$S TAG=$KT NSH=4 APC=256 BAM_PREDICT_X64=1 bash scripts/eval/mbd/kappa.sh >> $KL 2>&1
  fi ) &
wait
for i in $(seq 0 $((K - 1))); do tail -n 3 runs/slurm/chain_${J}_relax$i.log | cut -c1-200; done
grep -a -A12 -E '^\{' $KL | grep -E 'kappa_SRME"|n_scored|n_censored|RESOURCE|imaginary' | tail -n 6
SL=runs/slurm/chain_${J}_score.log
if [ "${TEST:-0}" = 1 ]; then
  OPENBLAS_NUM_THREADS=1 bash scripts/eval/mbd/score_sample.sh $TG > $SL 2>&1; grep -a -E '^(all|unique)' $SL | cut -c1-230
else
  env CKPT=$S TAG=$TG NSH=$NSH STAGE=score BAM_PREDICT_X64=1 bash scripts/eval/mbd/eval_wbm.sh > $SL 2>&1
  grep -a -E '^(all|unique_prototypes|MAE by)' $SL | cut -c1-240
  python -I - <<PYEOF
import json, re, ast
s = json.load(open('runs/mbd/$KT/kappa/kappa_srme.json')); s = s.get('summary', s); k = s['kappa_SRME']
u = [ast.literal_eval(m) for m in re.findall(r"^unique_prototypes (\{.*\})", open('$SL').read(), re.M)]
f1 = u[0]['F1']; rm = u[1]['RMSD']
print('$T float64: F1 %.3f MAE %.4f R2 %.3f RMSD %.4f kappa SRME %.3f  CPS %.4f' % (f1, u[0]['MAE'], u[0]['R2'], rm, k, 0.5 * f1 + 0.4 * (1 - k / 2) + 0.1 * (1 - rm / 0.15)))
PYEOF
fi
echo "CHAIN DONE $T  $(date '+%F %T')"
