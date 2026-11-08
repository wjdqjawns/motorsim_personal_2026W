"""Loading measured (or simulated) time series into the standard channel dict.

Standard keys: t, v | vd, vq, i | id, iq, theta, w, ref (all SI, mechanical angle).

Simulated CSV/NPZ files from `simulate` are recognised automatically (the *_meas
channels are used, i.e. what a real drive would see). Real data is mapped with a
`measurement` section, e.g.

    measurement:
      file: data/run1.csv
      columns: {t: time_ms, v: Vbus_cmd, i: Ia, theta: enc_cnt, ref: speed_cmd}
      scale:   {t: 1.0e-3, theta: 9.5874e-5}     # value_SI = raw * scale + offset
      offset:  {}
      resample: true                               # uniform grid at median dt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STANDARD = ("t", "v", "vd", "vq", "i", "id", "iq", "theta", "w", "ref", "ref_d")

def _from_frame(df: pd.DataFrame, spec: dict | None) -> dict:
    spec = spec or {}
    cols = spec.get("columns")
    out = {}
    if cols:
        for key, col in cols.items():
            out[key] = df[col].to_numpy(float)
    else:
        for key in STANDARD:
            if key in ("w",) and "theta_meas" in df:
                continue # speed is re-derived from the encoder
            src = f"{key}_meas" if f"{key}_meas" in df else key
            if src in df:
                out[key] = df[src].to_numpy(float)

    for key, s in (spec.get("scale") or {}).items():
        out[key] = out[key] * float(s)

    for key, o in (spec.get("offset") or {}).items():
        out[key] = out[key] + float(o)
    
    if "t" not in out:
        raise KeyError("measurement needs a time column 't'")

    out["t"] = out["t"] - out["t"][0]

    if spec.get("resample", True):
        dt = np.diff(out["t"])

        if np.ptp(dt) > 1e-3 * np.median(dt):
            Ts = float(np.median(dt))
            tn = np.arange(0, out["t"][-1], Ts)
            out = {k: (tn if k == "t" else np.interp(tn, out["t"], v)) for k, v in out.items()}

    return out

def load_data(path: str | Path | None = None, spec: dict | None = None, idx: int = 0) -> dict:
    spec = dict(spec or {})
    path = Path(path or spec.get("file"))

    if path.suffix == ".npz":
        from .simulator import load_result
        df = load_result(path).frame(idx)
    else:
        df = pd.read_csv(path, sep=spec.get("sep", ","))
    
    return _from_frame(df, spec)

def slice_data(data: dict, t0: float, t1: float) -> dict:
    m = (data["t"] >= t0) & (data["t"] < t1)
    out = {k: np.asarray(v)[m] for k, v in data.items()}
    out["t"] = out["t"] - out["t"][0]

    return out

def metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    e = y - yhat
    rmse = float(np.sqrt(np.mean(e ** 2)))
    denom = np.linalg.norm(y - y.mean()) or 1e-300

    return {
        "rmse": rmse,
        "nrmse_range": rmse / (np.ptp(y) or 1e-300),
        "fit_pct": float(100 * (1 - np.linalg.norm(e) / denom)),   # MATLAB compare() style
        "r2": float(1 - np.sum(e ** 2) / denom ** 2),
        "max_abs": float(np.abs(e).max()),
    }