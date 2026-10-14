#!/usr/bin/env bash
set -euo pipefail
if [[ "${TRACE:-0}" == "1" ]]; then
  set -x
fi

umask 007

# Use the prepared training environment without re-resolving it on GPU nodes.
export UV_NO_SYNC=${UV_NO_SYNC:-1}
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/training_launcher_common.sh"

# One launcher for the ABC-130k dual-expert joint recipe. The experiment YAML
# owns the training recipe; this launcher only selects the config, the
# out-of-tree abc_v3 dataset adapter, and process topology. Run from the
# OpenWAM repo root (or with --chdir set there) so `abc_open_wam/` resolves on
# `python -m`'s implicit cwd sys.path entry -- no PYTHONPATH juggling needed.
CONFIG_NAME=${CONFIG_NAME:-"dual_expert_abc_joint"}

export WANDB_PROJECT=${WANDB_PROJECT:-"openwam-abc130k-finetune"}
# Same dual-expert packed-coupling memory profile as the LIBERO joint launcher
# (~7.66B trainable parameters across video DiT + action expert). Override
# with OPEN_WAM_FSDP_CPU_OFFLOAD=0 on hardware with enough VRAM to skip it.
export OPEN_WAM_FSDP_CPU_OFFLOAD=${OPEN_WAM_FSDP_CPU_OFFLOAD:-1}

open_wam_launch_training "${CONFIG_NAME}" --extension abc_open_wam "$@"
