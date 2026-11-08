"""Saving / loading SimResult (.csv = one batch element, .npz = everything)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from motorsim.core.model.results import SimResult


def save_result(res: SimResult, path: str | Path, idx: int | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        res.frame(0 if idx is None else idx).to_csv(path, index=False)
    else:
        cfg = {k: v for k, v in res.config.items() if not k.startswith("_")}
        np.savez_compressed(
            path, t=res.t,
            **{f"data/{k}": v for k, v in res.data.items()},
            **{f"param/{k}": v for k, v in res.params.items()},
            config=yaml.safe_dump(cfg, sort_keys=False),
        )
    return path


def load_result(path: str | Path) -> SimResult:
    z = np.load(path, allow_pickle=False)
    data = {k[5:]: z[k] for k in z.files if k.startswith("data/")}
    params = {k[6:]: z[k] for k in z.files if k.startswith("param/")}
    cfg = yaml.safe_load(str(z["config"])) if "config" in z.files else {}
    return SimResult(t=z["t"], data=data, params=params, config=cfg)
