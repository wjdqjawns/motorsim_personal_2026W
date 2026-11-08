"""Fit metrics between a measured and a simulated signal."""
from __future__ import annotations

import numpy as np


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
