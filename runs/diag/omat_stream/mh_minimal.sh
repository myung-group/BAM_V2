#!/bin/bash
# Minimal multi-process JAX test on N nodes (no model; runs/diag/omat_stream/mh_minimal.py), to separate a JAX/NCCL
# setup problem from a trainer problem when 4 or more processes hang (2026-11-11).
# usage (from /scratch/t510a13/race_v2/work):
#   $ sbatch -J mhmin -p exclusive06 --comment="field=sciai;appl=jax" --nodes=4 --ntasks-per-node=1 --exclusive --gres=gpu:4 -t 00:10:00 runs/diag/omat_stream/mh_minimal.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${BAM_ENV:-/scratch/t510a13/envs/bam_cue}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" PYTHONUNBUFFERED=1
export NCCL_PROTO=^LL128 NCCL_DEBUG=${NCCL_DEBUG:-WARN} BAM_STACK_DUMP_SECS=120
export BAM_DISTRIBUTED=1 MASTER_ADDR=$(hostname) MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
echo "nodes $SLURM_JOB_NODELIST ($SLURM_NNODES)  $(date +%T)"
for N in 2 $SLURM_NNODES; do
  echo "== $N processes  $(date +%T)"
  export MASTER_PORT=$((MASTER_PORT + 1))
  timeout 240 srun --nodes=$N --ntasks=$N --ntasks-per-node=1 --cpus-per-task=288 \
    python -X faulthandler runs/diag/omat_stream/mh_minimal.py 2>&1 | grep -E '^\[p[0-9]+ |Timeout|backend_compile|Error|error' | sort -s -k1,1 | head -n 40
  echo "   srun exit ${PIPESTATUS[0]}"
done
echo "done $(date +%T)"
