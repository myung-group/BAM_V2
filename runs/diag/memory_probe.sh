#!/bin/bash
# memory of the exported function per bucket (gather vs ragged indexed linear), largest buffers with source lines
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export TMPDIR=/scratch/t510a13/tmp_xla_dump; mkdir -p $TMPDIR
C=input/input_3bpa_omat24_lipsF_fu_s42_cue.json
K=runs/lipsF_fu_s42_cue/ckpt_lipsF_fu_s42_cue.gauge.pkl
for REMAT in 1 0; do
  for B in "4096 131072" "16384 524288"; do
    BAM_REMAT_LAYERS=$REMAT BAM_INDEXED_LINEAR=ragged python runs/diag/memory_probe.py $C $K $B cue 2>&1 | grep -a "^bucket" | sed "s/^/remat=$REMAT /"
  done
done
rm -rf $TMPDIR/xla_dump_*
