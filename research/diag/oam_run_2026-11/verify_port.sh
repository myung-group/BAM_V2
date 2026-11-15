#!/bin/bash
# Check of the scripts ported from branch racev2 to the new layout (2026-11-16): unit tests, float64 prediction of the
# final OAM checkpoint with the new package against the reference written by the old code, memory benchmark and
# Laplace OOD script on a tiny setting.  One node, about 10 minutes.
#   $ sbatch -J portcheck -p exclusive06 --comment="field=sciai;appl=jax" --nodes=1 --exclusive --gres=gpu:4 -t 00:25:00 --exclude=node[0449-0484,0499-0576] /scratch/t510a13/BAM_V2_merge_check/research/diag/oam_run_2026-11/verify_port.sh
#SBATCH -o /scratch/t510a13/race_v2/work/runs/slurm/%x_%j.out
ulimit -c 0
REPO=${REPO:-/scratch/t510a13/BAM_V2_merge_check}; OLD=/scratch/t510a13/race_v2/work
W=/scratch/t510a13/race_v2/work_merge; mkdir -p $W && cd $W || exit 1
for l in race_v2 bam_omat24 scripts; do ln -sfn $REPO/$l $l; done; ln -sfn $OLD/runs runs       # new code, the runs of the old work folder
CKPT=runs/mbd/snap/oamft_final_e38.pkl; O=runs/bench/portcheck; mkdir -p $O runs/uq
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/scratch/t510a13/envs/bam_cue}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" BAM_NEIGHBORS=ase
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
echo "node $(hostname)  commit $(git -C $REPO rev-parse --short HEAD)  dirty files $(git -C $REPO status --short | wc -l)  $(date '+%F %T')"
( cd $REPO && JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=$REPO timeout 1200 python -m pytest -q -p no:cacheprovider --tb=short \
    tests/test_stream_parts.py tests/test_compat_aliases.py "tests/test_round13.py::test_in_layer_order_changes_energy_default_unchanged" \
    > $W/$O/pytest.log 2>&1; echo "pytest exit $?" >> $W/$O/pytest.log ) &
( CUDA_VISIBLE_DEVICES=0 BAM_PREDICT_X64=1 python scripts/eval/bench/ref_rattled.py $CKPT $O/ref_rattled_x64.npz 16 > $O/ref.log 2>&1; echo "ref exit $?" >> $O/ref.log ) &
( CUDA_VISIBLE_DEVICES=1 python scripts/eval/bench/mem_bench.py $CKPT $O/mem_float32.json 64,512 > $O/mem.log 2>&1; echo "mem exit $?" >> $O/mem.log ) &
( rm -rf runs/uq/portcheck; CUDA_VISIBLE_DEVICES=2 python scripts/eval/uq/ood_laplace.py --ckpt $CKPT --out runs/uq/portcheck --n-fit 16 --n-eval 32 --samples 4 \
    --lam 1e3 --sets mptrj_val,omol25 > $O/ood.log 2>&1; echo "ood exit $?" >> $O/ood.log ) &
wait
echo "== unit tests"; tail -n 6 $O/pytest.log | cut -c1-250
echo "== float64 prediction, new package vs old code (same checkpoint, 16 rattled PhononDB structures)"; tail -n 2 $O/ref.log | cut -c1-250
python -I - <<PY
import numpy as np
a = np.load('$OLD/runs/bench/oamft_final_e38/ref_rattled_x64.npz'); b = np.load('$O/ref_rattled_x64.npz')
print('max |dE| %.3e eV, max |dF| %.3e eV/A  (max |F| %.2f eV/A)' % (abs(a['e'] - b['e']).max(), abs(a['f'] - b['f']).max(), abs(a['f']).max()))
PY
echo "== memory benchmark (2 sizes)"; grep -a '^{' $O/mem.log | cut -c1-260; tail -n 1 $O/mem.log
echo "== Laplace OOD script (tiny)"; grep -a -E "AUROC|auroc|Spearman|Traceback|Error" $O/ood.log | tail -n 8 | cut -c1-250; tail -n 2 $O/ood.log | cut -c1-250
grep -h " exit " $O/pytest.log $O/ref.log $O/mem.log $O/ood.log | tr '\n' ' '; echo; echo "PORT CHECK DONE $(date '+%F %T')"
