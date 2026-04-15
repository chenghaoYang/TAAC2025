#!/bin/bash
#SBATCH --job-name=onepiece_sid_train
#SBATCH --account=torch_pr_74_tandon_priority
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/cy2668/TAAC2025/OnePiece/logs/slurm-%j.out
#SBATCH --error=/scratch/cy2668/TAAC2025/OnePiece/logs/slurm-%j.err
#
# ==============================================================================
# OnePiece Phase 3: SID Fine-tuning
# ==============================================================================
# Loads epoch7 InfoNCE checkpoint, enables SID losses, trains 8 more epochs.
# ==============================================================================

set -euo pipefail

# ========================== PATH CONFIGURATION ================================

export RUNTIME_SCRIPT_DIR="/scratch/cy2668/TAAC2025/OnePiece/code"
export TRAIN_DATA_PATH="/scratch/cy2668/TAAC2025/OnePiece/data"
export USER_CACHE_PATH="/scratch/cy2668/TAAC2025/OnePiece/user_cache"
export TRAIN_LOG_PATH="/scratch/cy2668/TAAC2025/OnePiece/logs/train_sid"
export TRAIN_TF_EVENTS_PATH="/scratch/cy2668/TAAC2025/OnePiece/logs/tensorboard_sid"
export TRAIN_CKPT_PATH="/scratch/cy2668/TAAC2025/OnePiece/checkpoints_sid"

# ========================== CREATE DIRECTORIES ================================
mkdir -p "${TRAIN_LOG_PATH}"
mkdir -p "${TRAIN_TF_EVENTS_PATH}"
mkdir -p "${TRAIN_CKPT_PATH}"
mkdir -p "/scratch/cy2668/TAAC2025/OnePiece/logs"

CKPT="/scratch/cy2668/TAAC2025/OnePiece/checkpoints/epoch7_global_step13692/model.pt"

# ========================== ENVIRONMENT SETUP =================================
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURMD_NODENAME}"
echo "Start time:    $(date)"
echo "Phase:         SID Fine-tuning (Phase 3)"
echo "Checkpoint:    ${CKPT}"
echo "=============================================="

if command -v module &>/dev/null; then
    module purge 2>/dev/null || true
    module load anaconda3/2025.06 2>/dev/null || true
fi

export PYTHONPATH="/scratch/cy2668/TAAC2025/pylibs:${RUNTIME_SCRIPT_DIR}:${PYTHONPATH:-}"

# Install training-time pip deps
pip install --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.python.org \
    --target=/scratch/cy2668/TAAC2025/pylibs \
    ujson orjson pynvml 2>/dev/null || true

echo "Python: $(which python3) $(python3 --version)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available yet"

# ========================== TRAINING ==========================================
cd "${RUNTIME_SCRIPT_DIR}"

# SID fine-tuning: load Phase 1 weights, enable --sid, train 8 epochs
python3 -u main_dist.py \
    --use_my_dataparallel \
    --gpu_ids 0 \
    --num_workers 8 \
    --mm_emb_id 81 82 \
    --single_device_batch_size 128 \
    --gradient_accumulation_steps 4 \
    --num_epochs 8 \
    --sid \
    --mm_sid 81 82 \
    --state_dict_path "${CKPT}"

echo ""
echo "=============================================="
echo "Job finished at $(date)"
echo "=============================================="
echo "Checkpoints: ${TRAIN_CKPT_PATH}"
echo "Logs:        ${TRAIN_LOG_PATH}"
ls -la "${TRAIN_CKPT_PATH}/" 2>/dev/null || true
