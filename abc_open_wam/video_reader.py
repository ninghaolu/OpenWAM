"""Worker-local, bounded, random-access video decoding with timestamp checks.

Adapted from DreamZero's `V3VideoReader`
(`groot/vla/data/dataset/lerobot_v3_video.py`, no DreamZero/GPU imports there
either). ABC-130k multiplexes many episodes into each MP4 file, so a sample
needs O(1) seeking to a handful of frames inside a long shared video rather
than the sequential full-file decode OpenWAM's `mixed_video_decode_backends`
provides; decord's keyframe index is what makes that cheap.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np


class RandomAccessVideoReader:
    """Bounded LRU cache of open decord readers and decoded frames."""

    def __init__(
        self,
        max_readers: int = 3,
        frame_cache_bytes: int = 32 * 1024**2,
        decode_batch: int = 16,
    ) -> None:
        if max_readers < 1 or frame_cache_bytes < 0 or decode_batch < 1:
            raise ValueError("Invalid video cache limits")
        self.max_readers = max_readers
        self.frame_cache_bytes = frame_cache_bytes
        self.decode_batch = decode_batch
        self.close()

    def close(self) -> None:
        self.readers: OrderedDict = OrderedDict()
        self.frames: OrderedDict = OrderedDict()
        self.cached_bytes = 0
        self.pid = os.getpid()

    def __getstate__(self) -> dict:
        return {
            "max_readers": self.max_readers,
            "frame_cache_bytes": self.frame_cache_bytes,
            "decode_batch": self.decode_batch,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state)

    def read(
        self,
        path: str,
        timestamps,
        interval: tuple[float, float],
        fps: float = 30.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `(frames[N,H,W,3] uint8, actual_timestamps[N])` for `timestamps`.

        `interval` is the episode's `(from_timestamp, to_timestamp)` window
        inside this shared video file; every requested timestamp must fall
        inside it.
        """

        import decord

        if self.pid != os.getpid():
            self.close()
        path = str(path)
        if path not in self.readers:
            while len(self.readers) >= self.max_readers:
                self.readers.popitem(last=False)
            reader = decord.VideoReader(path, num_threads=1)
            pts = np.asarray(reader.get_frame_timestamp(np.arange(len(reader))))[:, 0].astype(
                np.float64
            )
            if len(pts) == 0 or not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
                raise ValueError(f"Invalid video timestamps: {path}")
            self.readers[path] = reader, pts
        self.readers.move_to_end(path)
        reader, pts = self.readers[path]
        requested = np.asarray(timestamps, dtype=np.float64)
        start, end = interval
        # Float32 decoder timestamps lose precision in long shared videos.
        rounding = max(2e-5, abs(end) * np.finfo(np.float32).eps * 2)
        if np.any(requested < start - rounding) or np.any(requested >= end + rounding):
            raise ValueError(f"Requested timestamp outside episode camera interval: {path}")
        right = np.clip(np.searchsorted(pts, requested), 0, len(pts) - 1)
        left = np.maximum(right - 1, 0)
        indices = np.where(abs(pts[left] - requested) <= abs(pts[right] - requested), left, right)
        actual = pts[indices]
        if (
            np.any(abs(actual - requested) > 0.5 / fps + rounding)
            or np.any(actual < start - rounding)
            or np.any(actual >= end - rounding)
        ):
            raise ValueError(f"Decoded frame outside timestamp tolerance/episode interval: {path}")
        unique, inverse = np.unique(indices, return_inverse=True)
        found: dict[int, np.ndarray] = {}
        missing: list[int] = []
        for index in unique:
            key = (path, int(index))
            if key in self.frames:
                self.frames.move_to_end(key)
                found[int(index)] = self.frames[key]
            else:
                missing.append(int(index))
        for offset in range(0, len(missing), self.decode_batch):
            batch_ids = missing[offset : offset + self.decode_batch]
            decoded = reader.get_batch(batch_ids).asnumpy()
            for index, frame in zip(batch_ids, decoded):
                # Own the frame; otherwise a small cached slice retains a whole decode batch.
                frame = frame.copy()
                found[index] = frame
                if frame.nbytes <= self.frame_cache_bytes:
                    while self.frames and self.cached_bytes + frame.nbytes > self.frame_cache_bytes:
                        _, old = self.frames.popitem(last=False)
                        self.cached_bytes -= old.nbytes
                    self.frames[(path, index)] = frame
                    self.cached_bytes += frame.nbytes
        return np.stack([found[int(unique[i])] for i in inverse]), actual


__all__ = ["RandomAccessVideoReader"]
