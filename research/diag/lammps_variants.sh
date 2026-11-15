#!/bin/bash
# LAMMPS runtime variants (2026-11-06): export LiPS model as e3nn/cue x float64/float32, run the C++ test of each on
# the e3nn-float64 test cases (accuracy vs float64 reference + ms per call).  One GPU.
# usage: sbatch -J rv2_var -p normal --comment="field=sciai;appl=jax" --gres=gpu:1 -t 01:00:00 research/diag/lammps_variants.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
C=research/input/input_3bpa_omat24_lipsF_fu_s42_cue.json
K=runs/lipsF_fu_s42_cue/ckpt_lipsF_fu_s42_cue.gauge.pkl
REF=runs/lammps_export/lips_e3nn
for V in "e3nn float64" "e3nn float32" "cue float64" "cue float32"; do
  set -- $V; OUT=runs/lammps_export/lips_${1}_${2}
  CASES=""; [ "$1 $2" = "e3nn float64" ] && CASES="--cases /scratch/t510a13/data/datasets_ext/lips/fu_test.xyz" && OUT=$REF
  python -m race_v2.export.lammps $C --ckpt $K --out $OUT --backend $1 --dtype $2 \
    --buckets 128:4096 1024:32768 4096:131072 $CASES 2>&1 | grep -E "exported|recorded|Error|error|Traceback" | head -5
done
export LD_LIBRARY_PATH=$(ls -d $SP/nvidia/*/lib | tr '\n' ':')$SP/cuequivariance_ops/lib:${LD_LIBRARY_PATH}
export PATH=$SP/nvidia/cuda_nvcc/bin:$PATH
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cuda_data_dir=$SP/nvidia/cuda_nvcc"
for V in e3nn_float64 e3nn_float32 cue_float64 cue_float32; do
  D=runs/lammps_export/lips_$V; [ $V = e3nn_float64 ] && D=$REF
  TOL=1e-8; [ $V != e3nn_float64 ] && TOL=1e-2
  echo "== $V"; ./lammps/test_pjrt $SP/jax_plugins/xla_cuda12/xla_cuda_plugin.so $D 3 $REF $TOL 2>&1 | grep -E "case|ms per|ALL|FAIL|racev2: (cue|PJRT)|rror"
done
