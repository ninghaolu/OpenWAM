#!/usr/bin/env bash
#SBATCH --job-name=owam-abc-full
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=1024G
#SBATCH --time=60-00:00:00
#SBATCH --chdir=/home/alex/policy-finetuning/OpenWAM
#SBATCH --output=/home/alex/policy-finetuning/OpenWAM/logs/abc-full-%j.out
#SBATCH --error=/home/alex/policy-finetuning/OpenWAM/logs/abc-full-%j.err
set -euo pipefail

ROOT="${OPEN_WAM_REPO_ROOT:-/home/alex/policy-finetuning/OpenWAM}"
cd "${ROOT}"

# Eight H100s, matching this cluster's proven DreamZero ABC allocation
# (dreamzero/slurm/abc/finetune/full/full_finetune_h100_100k.sbatch).
export NGPU="${SLURM_GPUS_ON_NODE:-8}"
# Reuse OUTPUT_DIR to resume the latest completed checkpoint on each launch.
# The selection below is launcher behavior; the Python CLI requires --resume-from.
export OUTPUT_DIR="${OUTPUT_DIR:-/projects/data/cloud_user/checkpoints/openwam_abc130k/dual_expert_joint_${SLURM_JOB_ID:?Submit with sbatch}}"
mkdir -p "${OUTPUT_DIR}"

# Keep small logs on the home volume, separate from large checkpoints.
export RUN_LOG_DIR="${RUN_LOG_DIR:-${ROOT}/logs/abc/$(basename "${OUTPUT_DIR}")}"
mkdir -p "${RUN_LOG_DIR}"

# Resolve the newest completed checkpoint on every start, including Slurm
# requeues. Explicit initialization/resume arguments retain precedence.
resume_args=()
explicit_initialization=0
for arg in "$@"; do
  case "${arg}" in
    --resume-from|--resume-from=*|--initialize-weights-from|--initialize-weights-from=*)
      explicit_initialization=1 ;;
  esac
done
if [[ "${explicit_initialization}" == 0 ]]; then
  latest_checkpoint=$(python3 - "${OUTPUT_DIR}/checkpoints" <<'PY_CHECKPOINT'
from pathlib import Path
import sys
root = Path(sys.argv[1])
completed = [
    p for p in root.glob("checkpoint_step_*")
    if p.name.removeprefix("checkpoint_step_").isdigit()
    and (p / ".checkpoint_complete").is_file()
    and (p / "full_training_state.pt").is_file()
]
if completed:
    print(max(completed, key=lambda p: int(p.name.removeprefix("checkpoint_step_"))))
PY_CHECKPOINT
  )
  if [[ -n "${latest_checkpoint}" ]]; then
    resume_args=(--resume-from "${latest_checkpoint}")
    echo "Resuming completed checkpoint: ${latest_checkpoint}"
  fi
fi

export WANDB_MODE=online
export WANDB_ENTITY="${WANDB_ENTITY:-oneworld-ai}"
export WANDB_DIR="${RUN_LOG_DIR}"
export OPEN_WAM_WANDB_CONTIGUOUS_STEPS=1
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-${ROOT}/../yam-abc-reproduce/.secrets/wandb_alex_api_key}"
if [[ -z "${WANDB_API_KEY:-}" && -r "${WANDB_API_KEY_FILE}" ]]; then
  IFS= read -r WANDB_API_KEY < "${WANDB_API_KEY_FILE}"
  export WANDB_API_KEY
fi

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv --loop=5 > "${RUN_LOG_DIR}/gpu_utilization_${SLURM_JOB_ID}.csv" 2> "${RUN_LOG_DIR}/gpu_utilization_${SLURM_JOB_ID}.err" &
gpu_sampler_pid=$!
cleanup_sampler() {
  kill "${gpu_sampler_pid}" 2>/dev/null || true
  wait "${gpu_sampler_pid}" 2>/dev/null || true
}
trap cleanup_sampler EXIT

bash scripts/run_dual_expert_posttrain_abc.sh \
  --save-root "${OUTPUT_DIR}" \
  "${resume_args[@]}" \
  --set trainer.metrics_filename="${RUN_LOG_DIR}/metrics_${SLURM_JOB_ID}.jsonl" \
  --enable-wandb --wandb-mode online --wandb-entity "${WANDB_ENTITY}" \
  --set training.num_steps="${MAX_STEPS:-100000}" \
  --set trainer.save_interval="${SAVE_STEPS:-1000}" \
  --set trainer.max_checkpoints_to_keep="${KEEP_CHECKPOINTS:-100}" \
  "$@"
