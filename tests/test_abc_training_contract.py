"""ABC timing and raw-batch proprio regression coverage without external assets."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from abc_open_wam.dataset import AbcV3Dataset, ACTIONS_PER_LATENT
from open_wam.data.contracts import collate_wam_samples, move_wam_batch_to_device
from open_wam.models.common.proprio_conditioning import (
    project_hidden_proprio_context_to_frames,
    resolve_hidden_proprio_context,
)
from open_wam.models.policy_variants.dual_expert.sequence_layout import DualExpertTrainingLayout
from open_wam.training.step_executor import ViewBatchAdapter


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4])
def test_abc_alignment_survives_raw_batch_path(chunk_size):
    dataset = object.__new__(AbcV3Dataset)
    dataset.split = "train"
    dataset.camera_names = ()
    dataset._eligible_episodes = [0]
    episode = SimpleNamespace(length=133, episode_index=0, task="test")
    dataset.catalog = SimpleNamespace(episode=lambda _: episode, dataset_root="unused", fps=30)
    rows = {
        "action": np.arange(133 * 14, dtype=np.float32).reshape(133, 14),
        "observation.state": np.repeat(np.arange(133, dtype=np.float32)[:, None], 14, axis=1),
        "timestamp": np.arange(133, dtype=np.float64) / 30,
    }
    dataset._numeric_reader = SimpleNamespace(read_episode=lambda *_: rows)
    dataset._worker_rng = lambda: SimpleNamespace(randint=lambda *_: 4)
    sample = dataset[0]
    assert sample.actions.shape == (17 * ACTIONS_PER_LATENT, 14)
    assert torch.count_nonzero(sample.action_mask[:8]) == 0
    assert torch.all(sample.action_mask[8:] == 1)
    torch.testing.assert_close(sample.actions[8:], torch.from_numpy(rows["action"][4:132]))
    batch = move_wam_batch_to_device(collate_wam_samples([sample]), "cpu")
    prepared = ViewBatchAdapter().prepare(batch).policy_batch
    assert DualExpertTrainingLayout.resolve_action_tokens_per_frame(batch=prepared, observed_num_frames=17) == 8
    context = resolve_hidden_proprio_context(prepared.extra, require_frame_aligned=True, label="test")
    projected = project_hidden_proprio_context_to_frames(
        context, num_frames=17, chunk_size=chunk_size, chunk_origin_frame=1,
    )
    # Each future chunk sees only the state at its preceding boundary.
    for frame in range(1, 17):
        expected_row = 4 + ((frame - 1) // chunk_size) * chunk_size * 8
        assert torch.all(projected[0, frame] == expected_row)
