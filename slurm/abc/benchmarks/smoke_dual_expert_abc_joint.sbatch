#!/usr/bin/env bash
#SBATCH --job-name=owam-abc-smoke
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/alex/policy-finetuning/OpenWAM
#SBATCH --output=/home/alex/policy-finetuning/OpenWAM/logs/abc-smoke-%j.out
#SBATCH --error=/home/alex/policy-finetuning/OpenWAM/logs/abc-smoke-%j.err
set -euo pipefail

ROOT="${OPEN_WAM_REPO_ROOT:-/home/alex/policy-finetuning/OpenWAM}"
cd "${ROOT}"

# Two-GPU, few-step correctness check before committing an eight-GPU
# allocation: confirms the abc_v3 adapter, canonical view compositing, VAE
# frontend, and dual-expert joint step all run together on real hardware.
export NGPU="${SLURM_GPUS_ON_NODE:-2}"
export OUTPUT_DIR="${OUTPUT_DIR:-/projects/data/cloud_user/checkpoints/openwam_abc130k/smoke_${SLURM_JOB_ID:?Submit with sbatch}}"
mkdir -p "${OUTPUT_DIR}"

bash scripts/run_dual_expert_posttrain_abc.sh \
  --save-root "${OUTPUT_DIR}" \
  --disable-wandb \
  --set training.num_steps="${MAX_STEPS:-3}" \
  --set trainer.save_interval="${SAVE_STEPS:-3}" \
  "$@"
