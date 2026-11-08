"""Model validation: replay a measurement through the model and compute fit metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from motorsim.core.analysis.filters import mechanical_signals
from motorsim.core.analysis.items.output_error import replay
from motorsim.core.analysis.metrics import metrics
from motorsim.core.config import load_config


def validate(cfg, data: dict, params: dict, friction_model: str | None = None, substeps: int = 2):
    """Returns (metrics DataFrame, dict of {channel: (measured, simulated)})."""
    from motorsim.domains import get_domain

    cfg = load_config(cfg)
    dom = get_domain(cfg["motor"]["type"])
    w_meas, _ = mechanical_signals(data["t"], data.get("theta"), data.get("w"))
    r = replay(cfg, data, params, substeps=substeps, friction_model=friction_model, w_meas=w_meas)
    pairs = {c: (np.asarray(data[c]), r.data[c][0]) for c in dom.CUR_KEYS}
    pairs["w"] = (w_meas, r.data["w"][0])
    rows = [{"channel": c, **metrics(y, yh)} for c, (y, yh) in pairs.items()]
    return pd.DataFrame(rows), pairs


def param_error_table(true: dict, estimates: dict[str, dict], names) -> pd.DataFrame:
    """Relative error [%] of each estimate vs. ground truth (synthetic experiments)."""
    rows = []
    for n in names:
        row = {"param": n, "true": true.get(n, np.nan)}
        for label, est in estimates.items():
            v = est.get(n, np.nan)
            row[label] = v
            row[f"{label}_err%"] = 100 * (v - row["true"]) / row["true"] if row["true"] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
