"""Bounded, worker-local reads of one episode's state/action/timestamp rows.

Each ABC-130k data file (`data/chunk-*/file-*.parquet`) multiplexes roughly a
hundred episodes' rows behind a dataset-wide `index` column, with row groups
observed to align exactly one-to-one with episodes. Parquet row-group column
statistics (`min`/`max` on `index`) let us find the right row group for an
episode without reading any row data, so a sample only ever pays for decoding
the one row group it needs. This mirrors the caching/read shape of DreamZero's
`ABCV3Storage.read_episode` (`groot/vla/data/dataset/abc_v3_storage.py`) but
resolves file/row-group locations from ABC-130k's native episode metadata
(`episode_index.py`) instead of a separately prepared manifest.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .episode_index import EpisodeRecord

_NUMERIC_COLUMNS = ("observation.state", "action", "timestamp")


class AbcNumericReader:
    def __init__(self, numeric_cache_bytes: int = 128 * 1024**2) -> None:
        if numeric_cache_bytes < 1:
            raise ValueError("Numeric cache limit must be positive")
        self.numeric_cache_bytes = numeric_cache_bytes
        self._row_group_bounds: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
        self.reset_cache()

    def reset_cache(self) -> None:
        self._episode_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._cached_bytes = 0
        self.pid = os.getpid()

    def __getstate__(self) -> dict:
        return {"numeric_cache_bytes": self.numeric_cache_bytes}

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state)

    def _row_group_for(self, path: Path, target_index: int) -> int:
        if self.pid != os.getpid():
            self.reset_cache()
            self._row_group_bounds = {}
        bounds = self._row_group_bounds.get(path)
        if bounds is None:
            pf = pq.ParquetFile(path)
            index_column = pf.schema_arrow.get_field_index("index")
            mins = np.empty(pf.num_row_groups, dtype=np.int64)
            maxs = np.empty(pf.num_row_groups, dtype=np.int64)
            for i in range(pf.num_row_groups):
                stats = pf.metadata.row_group(i).column(index_column).statistics
                if stats is None or not stats.has_min_max:
                    raise ValueError(f"Missing 'index' column statistics in row group {i} of {path}")
                mins[i] = stats.min
                maxs[i] = stats.max
            bounds = (mins, maxs)
            self._row_group_bounds[path] = bounds
        mins, maxs = bounds
        row_group = int(np.searchsorted(mins, target_index, side="right")) - 1
        if row_group < 0 or not (mins[row_group] <= target_index <= maxs[row_group]):
            raise ValueError(f"No row group in {path} covers global index {target_index}")
        return row_group

    def read_episode(self, episode: EpisodeRecord, dataset_root: Path) -> dict[str, np.ndarray]:
        if self.pid != os.getpid():
            self.reset_cache()
        if episode.episode_index in self._episode_cache:
            self._episode_cache.move_to_end(episode.episode_index)
            return self._episode_cache[episode.episode_index]

        path = (
            dataset_root
            / "data"
            / f"chunk-{episode.data_chunk_index:03d}"
            / f"file-{episode.data_file_index:03d}.parquet"
        )
        row_group = self._row_group_for(path, episode.dataset_from_index)
        pf = pq.ParquetFile(path)
        table = pf.read_row_group(row_group, columns=["index", *_NUMERIC_COLUMNS])
        index = table.column("index").to_numpy()
        selected = (index >= episode.dataset_from_index) & (index < episode.dataset_to_index)
        if int(selected.sum()) != episode.length:
            raise ValueError(
                f"Episode {episode.episode_index} row-group selection returned "
                f"{int(selected.sum())} rows, expected {episode.length}."
            )
        table = table.filter(selected)
        rows: dict[str, np.ndarray] = {}
        for name in _NUMERIC_COLUMNS:
            column = table.column(name)
            if column.null_count:
                raise ValueError(f"Null numeric column {name!r} in episode {episode.episode_index}")
            if name in ("observation.state", "action"):
                array = np.asarray(column.to_pylist(), dtype=np.float32)
                if array.shape != (episode.length, 14):
                    raise ValueError(
                        f"Expected {name} shape {(episode.length, 14)} for episode "
                        f"{episode.episode_index}, got {array.shape}."
                    )
            else:
                array = column.to_numpy()
            rows[name] = array
        if not np.isfinite(rows["observation.state"]).all() or not np.isfinite(rows["action"]).all():
            raise ValueError(f"Non-finite state/action values in episode {episode.episode_index}")

        size = sum(a.nbytes for a in rows.values())
        if size > self.numeric_cache_bytes:
            raise ValueError(
                f"Episode {episode.episode_index} needs {size} bytes, above the numeric cache limit."
            )
        while self._episode_cache and self._cached_bytes + size > self.numeric_cache_bytes:
            _, old = self._episode_cache.popitem(last=False)
            self._cached_bytes -= sum(a.nbytes for a in old.values())
        self._episode_cache[episode.episode_index] = rows
        self._cached_bytes += size
        return rows


__all__ = ["AbcNumericReader"]
