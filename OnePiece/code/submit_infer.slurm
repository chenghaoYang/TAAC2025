#!/bin/bash
#SBATCH --job-name=onepiece_infer
#SBATCH --account=torch_pr_74_tandon_priority
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/cy2668/TAAC2025/OnePiece/logs/slurm-%j.out
#SBATCH --error=/scratch/cy2668/TAAC2025/OnePiece/logs/slurm-%j.err

set -euo pipefail

export RUNTIME_SCRIPT_DIR="/scratch/cy2668/TAAC2025/OnePiece/code"
export USER_CACHE_PATH="/scratch/cy2668/TAAC2025/OnePiece/user_cache"
export TRAIN_DATA_PATH="/scratch/cy2668/TAAC2025/OnePiece/data"
export EVAL_DATA_PATH="/scratch/cy2668/TAAC2025/OnePiece/data"
export MODEL_OUTPUT_PATH="/scratch/cy2668/TAAC2025/OnePiece/checkpoints_sid/epoch8_global_step15648"
export TRAIN_CKPT_PATH="/scratch/cy2668/TAAC2025/OnePiece/checkpoints_sid"
export TRAIN_LOG_PATH="/scratch/cy2668/TAAC2025/OnePiece/logs"
export TRAIN_TF_EVENTS_PATH="/scratch/cy2668/TAAC2025/OnePiece/logs/tensorboard"

CKPT="${MODEL_OUTPUT_PATH}/model.pt"
RESULT_DIR="/scratch/cy2668/TAAC2025/OnePiece/results"

mkdir -p "${RESULT_DIR}"
mkdir -p "/scratch/cy2668/TAAC2025/OnePiece/logs"

echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURMD_NODENAME}"
echo "Start time:    $(date)"
echo "Checkpoint:    ${CKPT}"
echo "=============================================="

if command -v module &>/dev/null; then
    module purge 2>/dev/null || true
    module load anaconda3/2025.06 2>/dev/null || true
fi

export PYTHONPATH="/scratch/cy2668/TAAC2025/pylibs:${RUNTIME_SCRIPT_DIR}:${PYTHONPATH:-}"

echo "Python: $(which python3) $(python3 --version)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available yet"

cd "${RUNTIME_SCRIPT_DIR}"

# Run inference using infer.py's main logic
python3 -u -c "
import sys
sys.argv = ['infer.py',
    '--mode', 'infer',
    '--sid',
    '--mm_sid', '81', '82',
    '--mm_emb_id', '81', '82',
    '--use_my_dataparallel',
    '--gpu_ids', '0',
    '--hidden_units', '128',
    '--num_blocks', '24',
    '--num_heads', '8',
    '--dnn_hidden_units', '4',
    '--feed_forward_hidden_units', '2',
    '--use_hstu',
    '--bf16',
    '--infonce',
    '--infonce_temp', '0.02',
    '--muon',
    '--norm_first',
    '--embedding_zero_init',
    '--inference_only',
    '--sid_codebook_layer', '2',
    '--sid_codebook_size', '16384',
    '--single_device_batch_size', '512',
    '--maxlen', '101',
    '--device', 'cuda',
    '--state_dict_path', '${CKPT}',
]

from infer import infer
import json

all_top10s, all_user_ids = infer()

# Save results
output_path = '${RESULT_DIR}/predictions.json'
results = {}
for uid, top10 in zip(all_user_ids, all_top10s):
    results[str(uid)] = [str(x) for x in top10]

with open(output_path, 'w') as f:
    json.dump(results, f)
print(f'Results saved to {output_path}')
print(f'Total users: {len(results)}')
"

echo ""
echo "=============================================="
echo "Job finished at $(date)"
echo "=============================================="
ls -la "${RESULT_DIR}/"
