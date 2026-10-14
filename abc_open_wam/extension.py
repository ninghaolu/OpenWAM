"""Registration hook loaded via `--extension abc_open_wam`."""

from __future__ import annotations

from .dataset import register_abc_v3_adapter


def register_open_wam() -> None:
    register_abc_v3_adapter()


__all__ = ["register_open_wam"]
