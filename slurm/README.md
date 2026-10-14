# Slurm jobs

Run submission commands from the OpenWAM repo root. Create `logs/` before
submitting (`mkdir -p logs`), since Slurm opens log files before the script runs. Jobs are grouped by
purpose, following the same layout as `dreamzero/slurm/`:

| Directory | Jobs |
| --- | --- |
| `abc/benchmarks/` | Two-GPU, few-step correctness smoke test |
| `abc/finetune/full/` | Eight-GPU ABC-130k dual-expert joint full fine-tune |

Both scripts run `configs/experiments/dual_expert_abc_joint.yaml` through
`scripts/run_dual_expert_posttrain_abc.sh`, which loads the `abc_v3` dataset
adapter (`abc_open_wam/`, an out-of-tree extension — see its module docstring
and `docs/cookbooks/new_dataset.md`) via `--extension abc_open_wam`.

## Prerequisites

```bash
cd /home/alex/policy-finetuning/OpenWAM
uv sync --frozen --group dev --extra train --extra eval --extra tracking
uv pip install decord2   # random-access MP4 reads for abc_v3; not in any uv extra
cp configs/local_paths.sample.yaml configs/local_paths.yaml
# populate paths.datasets.abc_root / abc_val_root and paths.models.lingbot_va_base
```

`lingbot_va_base` is the same public video-pretraining checkpoint the
validated LIBERO joint recipe uses:

```bash
uv run python scripts/pretraining/download_hf.py \
  --repo robbyant/lingbot-va-base --repo-type model --revision main \
  --include 'transformer/**' 'vae/**' 'text_encoder/**' 'tokenizer/**' \
  --out /projects/data/cloud_user/checkpoints/lingbot-va-base
```

`abc_root` / `abc_val_root` point directly at the original
`abc_130k_v3_train` / `abc_130k_v3_val` LeRobot v3 roots — no offline
conversion or manifest-preparation pass is needed; `abc_v3` reads native
`meta/`, `data/`, and `videos/` directly.

The ABC launcher uses the already prepared `.venv` (`UV_NO_SYNC=1`) so
compute nodes do not change dependencies at job startup. Re-run the setup
commands explicitly when dependencies change.

The full job forces online W&B logging to `oneworld-ai/openwam-abc130k-finetune`.
It uses `WANDB_API_KEY`, or reads `WANDB_API_KEY_FILE` (default:
`../yam-abc-reproduce/.secrets/wandb_alex_api_key`). Existing W&B login is also
supported. Override `WANDB_ENTITY` / `WANDB_PROJECT` as needed. No credential
is written into the repository or logs. W&B, JSONL metrics, and GPU telemetry live in `RUN_LOG_DIR` (default:
`logs/abc/<output-directory-name>` on the home volume), separate from large
checkpoints. JSONL disk-full/quota errors issue a warning and disable that
secondary sink; console and W&B logging continue. Checkpoint write failures
remain fatal so incomplete saves are never treated as successful.

Full-state checkpoints are saved every 1,000 optimizer steps; the latest 100
are retained. Override `MAX_STEPS`, `SAVE_STEPS`, and `KEEP_CHECKPOINTS` as needed.
The full job requests 60 days to accommodate the 100,000-step budget at the
measured initial throughput (~35 seconds/update). It exits when training finishes.
One checkpoint occupies approximately 124 GiB; retaining 100 requires roughly
12.1 TiB of storage, plus temporary headroom during checkpoint writes.

## Common commands

```bash
# Two H100s, 3 optimizer steps, W&B disabled.
sbatch slurm/abc/benchmarks/smoke_dual_expert_abc_joint.sbatch

# Eight H100s, full run (100,000 steps by default).
sbatch slurm/abc/finetune/full/finetune_dual_expert_abc_joint.sbatch

# Override step/save cadence or resume an existing run's OUTPUT_DIR.
MAX_STEPS=200 SAVE_STEPS=50 sbatch slurm/abc/finetune/full/finetune_dual_expert_abc_joint.sbatch
OUTPUT_DIR=/projects/data/cloud_user/checkpoints/openwam_abc130k/dual_expert_joint_12345 \
  sbatch slurm/abc/finetune/full/finetune_dual_expert_abc_joint.sbatch

# Inside an existing GPU allocation, without Slurm:
NGPU=2 bash scripts/run_dual_expert_posttrain_abc.sh --save-root outputs/abc_smoke --disable-wandb \
  --set training.num_steps=3
```

Run the smoke job first. It exercises the full path end to end (`abc_v3`
adapter -> canonical view compositing -> VAE frontend -> dual-expert joint
step -> checkpoint save) on real hardware before committing an eight-GPU
allocation on this shared, `defq`-only cluster.

The full Slurm launcher now selects the newest checkpoint containing both
`.checkpoint_complete` and `full_training_state.pt` under `OUTPUT_DIR/checkpoints`
on every start, including scheduler requeues. Explicit `--resume-from` or
`--initialize-weights-from` takes precedence. The underlying Python CLI still
requires an explicit resume path. To continue a W&B run, export its
`WANDB_RUN_ID` and `WANDB_RESUME=must`.

## v1 scope

The `abc_v3` adapter (`abc_open_wam/dataset.py`) uses a **fixed** K=4-chunk
sampling window (128 supervised actions / 65 RGB frames per camera at 15 fps / 4 state
anchors), not the design doc's randomized K in [1,4] — see the module
docstring in `abc_open_wam/dataset.py` and
`/home/alex/policy-finetuning/openwam_abc_finetuning.md`. Episodes shorter
than the fixed window are skipped rather than padded. Randomized geometry,
padding/masking for short episodes, and a held-out evaluation recipe (see
DreamZero's `docs/abc/FULL_FINETUNING.md` for the shape of one) remain
follow-up work.

The online Wan VAE maps 65 RGB frames to 17 latent frames. The adapter emits
136 action slots: eight masked t0 slots, followed by all 128 supervised actions.
It also emits 17 states at source rows `s + [0, 8, ..., 128]` through the raw
sample/batch `proprio_context_frames` fields. The runtime selects the preceding
chunk-boundary state, including when chunk size varies from one to four.
Future chunks start at latent frame 1; the initial frame has no action or video
loss. The original `state` field alone does not provide this hidden context.

Focused checks (use a writable `TMPDIR` with sufficient space):

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q \
  tests/test_abc_training_contract.py tests/test_training_runtime.py
```
