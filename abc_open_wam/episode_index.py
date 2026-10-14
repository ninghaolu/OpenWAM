"""Native LeRobot v3 episode catalog for ABC-130k.

ABC-130k already ships a complete per-episode index at
`meta/episodes/chunk-*/*.parquet` (row/video chunk-file locations, from/to
timestamps per camera, task text). Unlike DreamZero's ABC adapter, this reads
that native index directly instead of a separately prepared manifest: no
offline `prepare_abc_yam_v3.py`-style pass is required for correctness, only
loading these few small files once per process.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMERA_NAMES = (
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    task: str
    length: int
    dataset_from_index: int
    dataset_to_index: int
    data_chunk_index: int
    data_file_index: int
    # camera -> (chunk_index, file_index, from_timestamp, to_timestamp)
    cameras: dict[str, tuple[int, int, float, float]]


class AbcEpisodeCatalog:
    """In-memory index over one ABC-130k LeRobot v3 split root."""

    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root)
        info_path = self.dataset_root / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        if info.get("codebase_version") != "v3.0":
            raise ValueError(
                f"Expected LeRobot v3.0 metadata at {info_path}, got "
                f"{info.get('codebase_version')!r}."
            )
        self.fps = float(info["fps"])
        missing_cameras = [
            camera for camera in CAMERA_NAMES if camera not in info["features"]
        ]
        if missing_cameras:
            raise ValueError(f"Dataset root {self.dataset_root} is missing cameras: {missing_cameras}")

        episode_files = sorted(
            glob.glob(str(self.dataset_root / "meta" / "episodes" / "chunk-*" / "*.parquet"))
        )
        if not episode_files:
            raise FileNotFoundError(f"No episode metadata parquet files under {self.dataset_root}")
        tables = [pq.read_table(path) for path in episode_files]
        table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)

        episode_index = table.column("episode_index").to_numpy()
        order = np.argsort(episode_index, kind="stable")
        if not np.array_equal(episode_index[order], np.arange(len(episode_index))):
            raise ValueError(
                f"Expected contiguous 0..N-1 episode_index values under {self.dataset_root}."
            )
        table = table.take(order)

        self.num_episodes = table.num_rows
        self._length = table.column("length").to_numpy()
        self._from_index = table.column("dataset_from_index").to_numpy()
        self._to_index = table.column("dataset_to_index").to_numpy()
        self._data_chunk = table.column("data/chunk_index").to_numpy()
        self._data_file = table.column("data/file_index").to_numpy()
        self._tasks = [str(tasks[0]) for tasks in table.column("tasks").to_pylist()]
        self._camera_columns = {
            camera: (
                table.column(f"videos/{camera}/chunk_index").to_numpy(),
                table.column(f"videos/{camera}/file_index").to_numpy(),
                table.column(f"videos/{camera}/from_timestamp").to_numpy(),
                table.column(f"videos/{camera}/to_timestamp").to_numpy(),
            )
            for camera in CAMERA_NAMES
        }

    def __len__(self) -> int:
        return self.num_episodes

    def episode(self, episode_index: int) -> EpisodeRecord:
        i = int(episode_index)
        cameras = {
            camera: (
                int(chunk[i]),
                int(file_[i]),
                float(from_ts[i]),
                float(to_ts[i]),
            )
            for camera, (chunk, file_, from_ts, to_ts) in self._camera_columns.items()
        }
        return EpisodeRecord(
            episode_index=i,
            task=self._tasks[i],
            length=int(self._length[i]),
            dataset_from_index=int(self._from_index[i]),
            dataset_to_index=int(self._to_index[i]),
            data_chunk_index=int(self._data_chunk[i]),
            data_file_index=int(self._data_file[i]),
            cameras=cameras,
        )

    def data_path(self, episode: EpisodeRecord) -> Path:
        return (
            self.dataset_root
            / "data"
            / f"chunk-{episode.data_chunk_index:03d}"
            / f"file-{episode.data_file_index:03d}.parquet"
        )

    def video_path(self, episode: EpisodeRecord, camera: str) -> Path:
        chunk_index, file_index, _, _ = episode.cameras[camera]
        return (
            self.dataset_root
            / "videos"
            / camera
            / f"chunk-{chunk_index:03d}"
            / f"file-{file_index:03d}.mp4"
        )


__all__ = ["CAMERA_NAMES", "EpisodeRecord", "AbcEpisodeCatalog"]
