"""Simulation result container."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SimResult:
    t: np.ndarray                         # (T,)
    data: dict                            # name -> (N, T)
    params: dict                          # name -> (N,)
    config: dict = field(default_factory=dict)
    fast: pd.DataFrame | None = None      # optional intra-period log (sim.fast_log window, element 0)

    @property
    def N(self) -> int:
        return next(iter(self.data.values())).shape[0]

    def frame(self, idx: int = 0) -> pd.DataFrame:
        return pd.DataFrame({"t": self.t, **{k: v[idx] for k, v in self.data.items()}})

    def save(self, path: str | Path, idx: int | None = None) -> Path:
        from motorsim.core.io.result_store import save_result
        return save_result(self, path, idx)
