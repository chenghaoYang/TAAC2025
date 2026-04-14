#!/bin/bash
#SBATCH --job-name=onepiece_train
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
# OnePiece Training - SLURM Submission Script
# ==============================================================================
# Submit with:  sbatch submit_train.slurm
# Monitor with: squeue -u $USER
# Cancel with:  scancel <jobid>
#
# Two-phase execution:
#   Phase 1: Data preprocessing (parquet -> OnePiece format)
#   Phase 2: Training via run.sh (original OnePiece training pipeline)
#
# To run ONLY preprocessing (skip training):
#   Set environment variable SKIP_TRAINING=1 before submitting, e.g.:
#     SKIP_TRAINING=1 sbatch submit_train.slurm
#
# To skip preprocessing if data already exists:
#   Set environment variable SKIP_PREPROCESSING=1 before submitting.
# ==============================================================================

set -euo pipefail

# ========================== PATH CONFIGURATION ================================

export RUNTIME_SCRIPT_DIR="/scratch/cy2668/TAAC2025/OnePiece/code"
export RAW_DATA_PATH="/scratch/cy2668/TAAC2025/data/TencentGR-1M"
export TRAIN_DATA_PATH="/scratch/cy2668/TAAC2025/OnePiece/data"
export USER_CACHE_PATH="/scratch/cy2668/TAAC2025/OnePiece/user_cache"
export TRAIN_LOG_PATH="/scratch/cy2668/TAAC2025/OnePiece/logs/train"
export TRAIN_TF_EVENTS_PATH="/scratch/cy2668/TAAC2025/OnePiece/logs/tensorboard"
export TRAIN_CKPT_PATH="/scratch/cy2668/TAAC2025/OnePiece/checkpoints"

# ========================== CREATE DIRECTORIES ================================
mkdir -p "${TRAIN_DATA_PATH}"
mkdir -p "${USER_CACHE_PATH}"
mkdir -p "${TRAIN_LOG_PATH}"
mkdir -p "${TRAIN_TF_EVENTS_PATH}"
mkdir -p "${TRAIN_CKPT_PATH}"
mkdir -p "/scratch/cy2668/TAAC2025/OnePiece/logs"

# ========================== ENVIRONMENT SETUP =================================
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURMD_NODENAME}"
echo "Start time:    $(date)"
echo "=============================================="

# Load anaconda for Python 3.13 (has pyarrow built-in)
if command -v module &>/dev/null; then
    module purge 2>/dev/null || true
    module load anaconda3/2025.06 2>/dev/null || true
fi

# Ensure custom Python packages are importable
export PYTHONPATH="/scratch/cy2668/TAAC2025/pylibs:${RUNTIME_SCRIPT_DIR}:${PYTHONPATH:-}"

# Install required packages to /scratch (avoids home disk quota limit)
echo "Installing Python packages ..."
pip install --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.python.org \
    --target=/scratch/cy2668/TAAC2025/pylibs \
    orjson ujson tqdm numpy 2>/dev/null || true

pip install --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.python.org \
    --target=/scratch/cy2668/TAAC2025/pylibs \
    torch scikit-learn tensorboard 2>/dev/null || true

export PYTHONPATH="/scratch/cy2668/TAAC2025/pylibs:${PYTHONPATH:-}"

echo "Python: $(which python3) $(python3 --version)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available yet"

# ========================== PHASE 1: DATA PREPROCESSING =======================
if [ "${SKIP_PREPROCESSING:-0}" = "1" ]; then
    echo ""
    echo "SKIP_PREPROCESSING=1, skipping data preprocessing."
else
    echo ""
    echo "=============================================="
    echo "PHASE 1: Data Preprocessing"
    echo "=============================================="

    cd "${RUNTIME_SCRIPT_DIR}"

    python3 -u preprocess_data.py \
        --raw_data_path "${RAW_DATA_PATH}" \
        --output_path "${TRAIN_DATA_PATH}" \
        --user_cache_path "${USER_CACHE_PATH}" \
        --skip_item_feat \
        --skip_seq

    echo "Preprocessing completed at $(date)"

    # -----------------------------------------------------------------------
    # Copy files that run.sh expects in USER_CACHE_PATH
    # run.sh does: cp $TRAIN_DATA_PATH/{item_feat_dict.json,seq.jsonl,seq_offsets.pkl} $USER_CACHE_PATH/
    # -----------------------------------------------------------------------
    echo "Copying preprocessed data to USER_CACHE_PATH ..."
    cp "${TRAIN_DATA_PATH}/item_feat_dict.json" "${USER_CACHE_PATH}/item_feat_dict.json"
    cp "${TRAIN_DATA_PATH}/seq.jsonl"           "${USER_CACHE_PATH}/seq.jsonl"
    cp "${TRAIN_DATA_PATH}/seq_offsets.pkl"      "${USER_CACHE_PATH}/seq_offsets.pkl"

    # Symlink creative_emb if not already present
    if [ ! -e "${USER_CACHE_PATH}/creative_emb" ]; then
        ln -s "${TRAIN_DATA_PATH}/creative_emb" "${USER_CACHE_PATH}/creative_emb"
    fi

    echo "Data copy completed at $(date)"
fi

# ========================== PHASE 2: TRAINING ================================
echo ""
echo "=============================================="
echo "PHASE 2: Training"
echo "=============================================="

cd "${RUNTIME_SCRIPT_DIR}"

# Install training-time pip deps
pip install --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.python.org \
    --target=/scratch/cy2668/TAAC2025/pylibs \
    ujson orjson pynvml 2>/dev/null || true

# Run auxiliary data scripts (idempotent if already done)
python3 -u item_exposure_data.py
python3 -u timestamp_buckets.py

# Train (single GPU)
python3 -u main_dist.py \
    --use_my_dataparallel \
    --gpu_ids 0 \
    --num_workers 8 \
    --mm_emb_id 81 82 \
    --single_device_batch_size 128 \
    --gradient_accumulation_steps 4 \
    --num_epochs 8

echo ""
echo "=============================================="
echo "Job finished at $(date)"
echo "=============================================="
echo "Checkpoints: ${TRAIN_CKPT_PATH}"
echo "Logs:        ${TRAIN_LOG_PATH}"
ls -la "${TRAIN_CKPT_PATH}/" 2>/dev/null || true
