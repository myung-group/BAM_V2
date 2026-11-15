#!/bin/bash
# Can a float64 SOAP run be resumed as float32 model + float64 SOAP state ("x64": false, "soap_state_dtype":
# "float64"; models/blocks.py fix a26057c, resume cast in train_3bpa)?  One node, one process, 4 GPUs.  The running job is
# not touched: its last checkpoint is copied into two test run folders, and each copy is continued for EPOCHS short
# virtual epochs (STEPS steps each, same learning rate as the source run) -- first as float32 + float64 state, then
# as float64 (control).  Compare step time (progress lines), GPU memory and the validation lines of the two logs.
# MODES="f32_norefresh f32_cholqr f32_b32 f32_amsgrad" VALMAX=2000 EPOCHS=2: where the step time goes (see the mode list below).
# usage (from /scratch/t510a13/race_v2/work):
#   $ sbatch -J f32resume -p exclusive06 --comment="field=sciai;appl=jax" --nodes=1 --exclusive --gres=gpu:4 -t 01:00:00 --export=ALL,SRC=mptrj_L5x128_s42 research/diag/omat_stream/f32_resume_test.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
SRC=${SRC:?source run name}; STEPS=${STEPS:-250}; EPOCHS=${EPOCHS:-3}; MODES=${MODES:-"f32 f64"}; VALMAX=${VALMAX:-0}
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/scratch/t510a13/envs/bam_cue}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" BAM_NEIGHBORS=ase
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 BAM_PROGRESS_STEPS=50 PYTHONUNBUFFERED=1
echo "node $SLURM_JOB_NODELIST  source $SRC  code $(git -C $(readlink -f race_v2)/.. rev-parse --short HEAD)  $(date '+%F %T')"
LAST=runs/${SRC}_cue/ckpt_${SRC}_cue.last.pkl
until [ $(( $(date +%s) - $(stat -c %Y $LAST) )) -gt 90 ]; do sleep 15; done      # not while it is being written
for MODE in $MODES; do
  NAME=${SRC}_resume_${MODE}; RUN=runs/${NAME}_cue; mkdir -p $RUN
  rm -f $RUN/*.pkl $RUN/loss_${NAME}_cue.out
  cp $LAST $RUN/ckpt_${NAME}_cue.last.pkl
  python -I - "$SRC" "$NAME" "$MODE" "$STEPS" "$VALMAX" <<'PY'
import json, sys
src, name, mode, steps, valmax = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
c = json.load(open(f'runs/omat_stream/{src}.json'))
# the learning rate follows the step counter: keep the source run's total and warm-up step counts (8 devices there)
per_epoch = None
for line in open(c['log_file']):
    if line.startswith('LR schedule: cosine'):
        w = line.replace(',', ' ').split()
        warm, total = int(w[w.index('warmup') + 1]), int(w[w.index('total') + 1])
        per_epoch = total // int(c['n_epochs'])
assert per_epoch, 'no "LR schedule" line in the source log'
c['n_epochs'] = -(-total // steps); c['lr_warmup_epochs'] = -(-warm // steps)
c['stream']['steps_per_epoch'] = steps
run = f'runs/{name}_cue'
c.update(log_file=f'{run}/loss_{name}_cue.out', ckpt_name=f'{run}/ckpt_{name}_cue.pkl', restart=True)
if mode != 'f64':
    c['x64'] = False; c['soap_state_dtype'] = 'float64'
# where the step time goes (all float32 model + float64 SOAP state unless noted)
if mode == 'f32_norefresh':                      # no eigenbasis refresh (diagnostic only: changes the optimiser)
    c['soap_precondition_frequency'] = 10 ** 6
elif mode == 'f32_cholqr':
    c['soap_orth'] = 'cholqr'
elif mode == 'f32_b32':                          # twice the structures per step: fixed cost per step vs per structure
    c['batch_size'] = 32; c['stream']['graphs_per_pack'] = 32
elif mode == 'f32_amsgrad':                      # no SOAP at all: fresh AMSGrad state on the checkpoint's parameters
    c.pop('soap_state_dtype'); c['optimizer'] = 'amsgrad'; c['restart'] = False
    c['init_params_from'] = f'{run}/ckpt_{name}_cue.last.pkl'; c['init_from_ema'] = False
elif mode not in ('f32', 'f64'):
    raise SystemExit(f'unknown mode {mode}')
if valmax:
    c['stream']['valid']['max_structures'] = valmax
json.dump(c, open(f'runs/omat_stream/{name}.json', 'w'), indent=1)
print(name, 'x64', c['x64'], 'soap_state_dtype', c.get('soap_state_dtype'), 'n_epochs', c['n_epochs'],
      'warmup epochs', c['lr_warmup_epochs'], 'source steps/epoch', per_epoch)
PY
  LOG=$RUN/loss_${NAME}_cue.out
  setsid python -m race_v2.training.train runs/omat_stream/${NAME}.json > runs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${MODE}.log 2>&1 &
  PID=$!
  until [ "$(grep -c -E '^[0-9]+/[0-9]+/[0-9]+ ' $LOG 2>/dev/null || true)" -ge $EPOCHS ] 2>/dev/null || ! kill -0 $PID 2>/dev/null; do sleep 10; done
  echo "== $MODE  $(date '+%F %T')  GPU memory used (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
  kill -TERM -- -$PID 2>/dev/null; sleep 5; kill -KILL -- -$PID 2>/dev/null; wait $PID 2>/dev/null
  grep -E 'Resume|Resumed|float64|LR schedule|stream: start' $LOG | cut -c1-200
  grep -E '^[0-9]+/[0-9]+/[0-9]+ ' $LOG | cut -c1-230
  grep -E 'first training step|^epoch [0-9]+ step' runs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${MODE}.log | tail -n 12
  grep -c -E 'Error|Traceback' runs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${MODE}.log
  rm -f $RUN/*.pkl                              # 2.4 GB copies: not kept
done
echo "done $(date '+%F %T')"
