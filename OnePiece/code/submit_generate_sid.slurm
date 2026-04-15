#!/bin/bash
#SBATCH --job-name=onepiece_sid
#SBATCH --account=torch_pr_74_tandon_priority
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/cy2668/TAAC2025/OnePiece/logs/slurm-%j.out
#SBATCH --error=/scratch/cy2668/TAAC2025/OnePiece/logs/slurm-%j.err

set -euo pipefail

export RUNTIME_SCRIPT_DIR="/scratch/cy2668/TAAC2025/OnePiece/code"
export USER_CACHE_PATH="/scratch/cy2668/TAAC2025/OnePiece/user_cache"
export PYTHONPATH="/scratch/cy2668/TAAC2025/pylibs:${RUNTIME_SCRIPT_DIR}:${PYTHONPATH:-}"

echo "=============================================="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "=============================================="

CKPT="/scratch/cy2668/TAAC2025/OnePiece/checkpoints/epoch7_global_step13692/model.pt"
SID_DIR="${USER_CACHE_PATH}/sid"

mkdir -p "${SID_DIR}"

python3 -u "${RUNTIME_SCRIPT_DIR}/generate_sid.py" \
    --checkpoint "${CKPT}" \
    --output_dir "${SID_DIR}" \
    --n_clusters_l1 512 \
    --n_clusters_l2 32

echo ""
echo "=============================================="
echo "Job finished at $(date)"
echo "=============================================="
ls -la "${SID_DIR}/"
