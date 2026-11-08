"""Model validation: replay a measurement through the model, compute fit metrics, plot."""
from __future__ import annotations

import numpy as np
import pandas as pd

from motorsim.config import load_config
from motorsim.data import metrics
from motorsim.ident.ls import mechanical_signals
from motorsim.ident.output_error import replay


def validate(cfg, data: dict, params: dict, friction_model: str | None = None, substeps: int = 2):
    """Returns (metrics DataFrame, dict of {channel: (measured, simulated)})."""
    cfg = load_config(cfg)
    mtype = cfg["motor"]["type"]
    w_meas, _ = mechanical_signals(data["t"], data.get("theta"), data.get("w"))
    r = replay(cfg, data, params, substeps=substeps, friction_model=friction_model, w_meas=w_meas)
    cur = ["i"] if mtype == "dc" else ["id", "iq"]
    pairs = {c: (np.asarray(data[c]), r.data[c][0]) for c in cur}
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
