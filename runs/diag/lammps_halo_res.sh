#!/bin/bash
# pair_style race mode halo with saved residuals (2026-11-06, written while the allocation was exhausted; not run yet):
# staged check in Python (residual vs recompute), export with --staged-residuals, then LiPS 2x2x2 / 4x4x4 on 4 ranks
# with RACEV2_STAGED_RESIDUALS=1 and =0 (same export) to measure the gain.
# usage: sbatch -J rv2_halores -p normal --comment="field=sciai;appl=jax" -N 1 --ntasks-per-node=4 --gres=gpu:4 \
#          -t 01:30:00 runs/diag/lammps_halo_res.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t589a02/BAM_V2
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
module load PrgEnv-gnu cray-mpich >/dev/null 2>&1
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export BAM_INDEXED_LINEAR=ragged RACEV2_TIMING=1
C=input/input_3bpa_omat24_lipsF_fu_s42_cue.json
K=runs/lipsF_fu_s42_cue/ckpt_lipsF_fu_s42_cue.gauge.pkl
CUDA_VISIBLE_DEVICES=0 python runs/diag/staged_check.py $C $K /scratch/t589a02/datasets_ext/lips/fu_test.xyz runs/lammps_export/lips_e3nn cue 1 2>&1 | grep -a -E "staged|halo|Error|error|Traceback"
OUT=runs/lammps_export/lips_cue_halores
CUDA_VISIBLE_DEVICES=0 python -m bam_omat24.export_lammps $C --ckpt $K --out $OUT --backend cue \
  --buckets 128:4096 1024:32768 8192:262144 \
  --staged-buckets 1024:8192 1024:16384 2048:8192:256 2048:8192 2048:16384:512 2048:16384 2048:32768 4096:32768:1024 4096:32768 4096:65536:2048 4096:65536 4096:131072 8192:131072:4096 8192:131072 8192:262144 12288:262144:8192 12288:262144 \
  --staged-residuals 2>&1 | grep -a -E "exported|recorded|written|Error|error|Traceback"
export LD_LIBRARY_PATH=$(ls -d $SP/nvidia/*/lib | tr '\n' ':')$SP/cuequivariance_ops/lib:${LD_LIBRARY_PATH}
export PATH=$SP/nvidia/cuda_nvcc/bin:$PATH
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cuda_data_dir=$SP/nvidia/cuda_nvcc"
for R in 1 0; do
  echo "=== RACEV2_STAGED_RESIDUALS=$R"
  for REP in "2 2 2" "4 4 4"; do
    RACEV2_STAGED_RESIDUALS=$R python runs/diag/lammps_multi.py --serial /scratch/t589a02/src/lammps-racev2-build/lmp_racev2 \
      --mpi /scratch/t589a02/src/lammps-racev2-mpi-build/lmp_racev2_mpi \
      --plugin $SP/jax_plugins/xla_cuda12/xla_cuda_plugin.so --export $OUT --modes halo --tag halores$R \
      --xyz /scratch/t589a02/datasets_ext/lips/fu_test.xyz --rep $REP --ranks 4 --steps 200 --skip-serial-ghost
  done
done
