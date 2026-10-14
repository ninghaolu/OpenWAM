"""ABC-130k raw dataset adapter: `dataset_type: abc_v3`.

Returns raw per-camera RGB + 14-D action/state `WAMSample`s straight from the
original LeRobot v3 Parquet/MP4 files (no offline VAE latent pass). This is
deliberately the "views" contract (see `open_wam.training.step_executor.
ViewBatchAdapter`, which explicitly documents RGB batches "before online VAE
encoding") — pair with `trainer.batch_adapter: views` in the experiment
config, not `latents`; `latents` selects a different loader path that expects
a dataset already emitting pre-encoded `LatentWAMSample`s, which this adapter
does not produce.

v1 sampling geometry is fixed (K=4 chunks; the design doc's randomized
K in [1,4] is left for a follow-up), matching the doc's own K=4 worked
example and the LIBERO joint recipe's `training.chunk_size=4` /
`window_size=64` upper bounds:

    source start s -> actions  masked t0 + s .. s+127 (136, 14)
                    -> states  s+[0,32,64,96]         (4, 14)
                    -> proprio s+[0,8,...,128]        (17, 14)
                    -> frames  s, s+2, ..., s+128      (65,)  per camera, stride 2

Episodes shorter than the fixed 129-row window are skipped rather than
padded/masked (the doc allows either; padding is a follow-up, not a
correctness requirement for a first working run).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from open_wam.sdk.config import DataConfig
from open_wam.sdk.data import (
    DatasetArtifactKind,
    DatasetArtifactRequirement,
    WAMSample,
    register_dataset_adapter,
)

from .episode_index import CAMERA_NAMES, AbcEpisodeCatalog
from .numeric_reader import AbcNumericReader
from .video_reader import RandomAccessVideoReader

ACTIONS_PER_LATENT = 8  # 30-Hz actions / (15-fps RGB / 4x VAE compression)
ACTION_ROWS = 128  # s .. s+127
FRAME_STRIDE = 2
NUM_FRAMES = ACTION_ROWS // FRAME_STRIDE + 1  # 65: s, s+2, ..., s+128
NUM_CHUNKS = 4
STATE_ANCHORS = tuple(i * (ACTION_ROWS // NUM_CHUNKS) for i in range(NUM_CHUNKS))  # (0, 32, 64, 96)
MIN_EPISODE_LENGTH = ACTION_ROWS + 1  # rows s .. s+128 must exist


class AbcV3Dataset(Dataset[WAMSample]):
    def __init__(self, config: DataConfig, *, split: str) -> None:
        self.split = split
        dataset_root = config.local_root if split == "train" else (config.val_local_root or config.local_root)
        if not dataset_root:
            raise ValueError(f"`data.{'local_root' if split == 'train' else 'val_local_root'}` is required for abc_v3.")
        self.dataset_root = dataset_root
        self.camera_names = tuple(config.camera_names) if config.camera_names else CAMERA_NAMES
        unknown = set(self.camera_names) - set(CAMERA_NAMES)
        if unknown:
            raise ValueError(f"abc_v3 does not know camera(s) {sorted(unknown)}; expected a subset of {CAMERA_NAMES}.")

        self.catalog = AbcEpisodeCatalog(dataset_root)
        self._eligible_episodes = [
            i for i in range(len(self.catalog)) if self.catalog.episode(i).length >= MIN_EPISODE_LENGTH
        ]
        if not self._eligible_episodes:
            raise ValueError(f"No episodes under {dataset_root} reach the minimum length {MIN_EPISODE_LENGTH}.")
        skipped = len(self.catalog) - len(self._eligible_episodes)
        if skipped:
            print(
                f"[abc_v3:{split}] skipping {skipped}/{len(self.catalog)} episodes shorter than "
                f"{MIN_EPISODE_LENGTH} rows (below the fixed K={NUM_CHUNKS} sampling window).",
                flush=True,
            )

        numeric_cache_bytes = int(config.adapter_options.get("numeric_cache_bytes", 128 * 1024**2))
        video_max_readers = int(config.adapter_options.get("video_max_readers", 3))
        video_frame_cache_bytes = int(config.adapter_options.get("video_frame_cache_bytes", 32 * 1024**2))
        self._numeric_reader = AbcNumericReader(numeric_cache_bytes=numeric_cache_bytes)
        self._video_reader = RandomAccessVideoReader(
            max_readers=video_max_readers, frame_cache_bytes=video_frame_cache_bytes
        )
        self._rng: random.Random | None = None
        self._rng_pid: int | None = None

    def __len__(self) -> int:
        return len(self._eligible_episodes)

    def _worker_rng(self) -> random.Random:
        # A plain `random.randint` call would share Python's global RNG state
        # across DataLoader workers forked from the same parent, correlating
        # their sampled starts. Seed a private Random per worker PROCESS from
        # OS entropy instead.
        pid = os.getpid()
        if self._rng is None or self._rng_pid != pid:
            self._rng = random.Random(os.urandom(16))
            self._rng_pid = pid
        return self._rng

    def __getitem__(self, index: int) -> WAMSample:
        episode = self.catalog.episode(self._eligible_episodes[index % len(self._eligible_episodes)])
        rows = self._numeric_reader.read_episode(episode, self.catalog.dataset_root)

        # Uniform random valid start; re-drawn every call (replacement sampling),
        # matching the doc's "bounded random windows throughout episodes" guidance.
        max_start = episode.length - MIN_EPISODE_LENGTH
        start = self._worker_rng().randint(0, max_start)

        actions = torch.from_numpy(rows["action"][start : start + ACTION_ROWS].copy())
        # The online Wan VAE retains t0 plus 16 future latent frames. Packed
        # training requires eight action slots per latent, including t0. The
        # t0 slots carry no targets and must be masked out of the loss.
        actions = torch.cat([actions.new_zeros(ACTIONS_PER_LATENT, 14), actions])
        action_mask = torch.ones_like(actions)
        action_mask[:ACTIONS_PER_LATENT] = 0
        proprio_context_frames = torch.from_numpy(
            rows["observation.state"][start : start + ACTION_ROWS + 1 : ACTIONS_PER_LATENT].copy()
        )
        state = torch.from_numpy(
            np.stack([rows["observation.state"][start + offset] for offset in STATE_ANCHORS])
        )

        sample_relative_timestamps = (
            np.arange(0, ACTION_ROWS + 1, FRAME_STRIDE, dtype=np.float64) / self.catalog.fps
        )
        source_timestamps = rows["timestamp"][start : start + ACTION_ROWS + 1 : FRAME_STRIDE].astype(np.float64)
        episode_local_start = source_timestamps[0]
        if not np.allclose(sample_relative_timestamps, source_timestamps - episode_local_start, atol=1e-4):
            raise ValueError(
                f"Episode {episode.episode_index} timestamps at rows {start}..{start + ACTION_ROWS} "
                "are not the expected fixed-rate sequence; the episode may contain a boundary anomaly."
            )

        views: dict[str, torch.Tensor] = {}
        for camera in self.camera_names:
            _, _, from_ts, _ = episode.cameras[camera]
            path = self.catalog.video_path(episode, camera)
            frames, _ = self._video_reader.read(
                str(path),
                sample_relative_timestamps + episode_local_start + from_ts,
                interval=episode.cameras[camera][2:4],
                fps=self.catalog.fps,
            )
            views[camera] = torch.from_numpy(frames.copy())

        return WAMSample(
            views=views,
            actions=actions,
            action_mask=action_mask,
            proprio_context_frames=proprio_context_frames,
            state=state,
            task_text=episode.task,
            metadata={
                "dataset_name": "abc_130k_v3",
                "split": self.split,
                "episode_index": episode.episode_index,
                "start": start,
                "chunk_origin_frame": 1,
                "action_loss_frame_start": 1,
                "latent_loss_frame_start": 1,
            },
        )


def build_train_val(config: DataConfig) -> tuple[AbcV3Dataset, AbcV3Dataset]:
    return AbcV3Dataset(config, split="train"), AbcV3Dataset(config, split="val")


def resolve_artifacts(config: DataConfig) -> tuple[DatasetArtifactRequirement, ...]:
    return (
        DatasetArtifactRequirement(
            name="ABC-130k training root",
            path=config.local_root,
            kind=DatasetArtifactKind.DIRECTORY,
            required=True,
            config_path="data.local_root",
            purpose="abc_v3 reads native LeRobot v3 meta/data/videos from this directory",
        ),
        DatasetArtifactRequirement(
            name="ABC-130k validation root",
            path=config.val_local_root,
            kind=DatasetArtifactKind.DIRECTORY,
            required=True,
            config_path="data.val_local_root",
            purpose="abc_v3 reads the validation split's native LeRobot v3 metadata from this directory",
        ),
    )


def register_abc_v3_adapter() -> None:
    register_dataset_adapter(
        "abc_v3",
        raw_builder=build_train_val,
        artifact_resolver=resolve_artifacts,
        description="ABC-130k raw LeRobot v3 (Parquet + MP4) adapter with online VAE encoding.",
    )


__all__ = ["AbcV3Dataset", "build_train_val", "resolve_artifacts", "register_abc_v3_adapter"]
