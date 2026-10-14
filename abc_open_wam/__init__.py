"""OpenWAM extension package: ABC-130k raw LeRobot v3 dataset adapter.

Load with ``--extension abc_open_wam`` (see `open_wam.extensions`). Registers
`dataset_type: abc_v3` for `data.dataset_type` in experiment configs.
"""

from .extension import register_open_wam

__all__ = ["register_open_wam"]
