#!/bin/bash
#SBATCH --job-name=jax-multihost
#SBATCH --nodes=2                     # 2개 노드 사용
#SBATCH --ntasks-per-node=1           # 노드당 1개 프로세스 (JAX가 multi-GPU 관리)
#SBATCH --gpus-per-node=4             # 노드당 4개 GPU
#SBATCH --cpus-per-task=32            # 데이터 로딩용 CPU
#SBATCH --mem=256G                    # 노드당 메모리
#SBATCH --time=48:00:00               # 최대 실행 시간
#SBATCH --partition=gpu               # GPU 파티션 (클러스터에 맞게 수정)
#SBATCH --output=slurm-%j.out         # 표준 출력 파일
#SBATCH --error=slurm-%j.err          # 에러 출력 파일

# ============================================================================
# Environment Setup
# ============================================================================

# Conda 환경 활성화 (필요시 경로 수정)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate jax-env

# CUDA 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ============================================================================
# NCCL Configuration (Inter-node Communication)
# ============================================================================

# NCCL 디버그 (문제 해결 시 INFO로 설정)
export NCCL_DEBUG=WARN
# export NCCL_DEBUG=INFO

# InfiniBand 사용 (클러스터에 IB가 있는 경우)
export NCCL_IB_DISABLE=0

# GPU Direct RDMA (지원하는 경우)
export NCCL_NET_GDR_LEVEL=2

# 네트워크 인터페이스 설정 (클러스터에 맞게 수정)
# export NCCL_SOCKET_IFNAME=ib0  # InfiniBand
# export NCCL_SOCKET_IFNAME=eth0 # Ethernet

# ============================================================================
# JAX/XLA Configuration
# ============================================================================

# XLA 최적화 플래그
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true"

# 메모리 할당 설정
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# JAX 64비트 비활성화 (메모리 절약)
export JAX_ENABLE_X64=0

# ============================================================================
# Distributed Coordination
# ============================================================================

# 첫 번째 노드를 coordinator로 설정
HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export JAX_COORDINATOR_ADDRESS="${HEAD_NODE}:29500"
export JAX_NUM_PROCESSES=$SLURM_NPROCS

echo "============================================"
echo "SLURM Job Info"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NPROCS"
echo "GPUs per node: 4"
echo "Coordinator: $JAX_COORDINATOR_ADDRESS"
echo "============================================"

# ============================================================================
# Run Training
# ============================================================================

# srun이 각 노드에서 프로세스를 실행
# SLURM_PROCID가 자동으로 각 프로세스에 할당됨
srun python train_sharded_multihost.py
