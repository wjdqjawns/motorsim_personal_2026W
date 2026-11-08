#*************************************************************************/
# File Name: ./src/motorsim/ident/friction_fit.py
# Author: Beomjun Chung
# Updated: 2026-03-22
#
# Description:
#    Fit static friction models (optionally together with inertia J) and rank them.
#       y = [J a] + T_f(w; theta) + [T0]
#    y is the torque balance target (e.g. Kt*i in steady state, or Kt*i - nothing with accel a).
#    Models are compared with RMSE, R^2, AIC and BIC (lower AIC/BIC = better trade-off).
#    Main entrypoint for thermoviz runtime.
#*************************************************************************/

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from motorsim.models.friction import STATIC_PARAMS, static_friction

@dataclass
class FitResult:
    model: str
    params: dict
    rmse: float
    r2: float
    aic: float
    bic: float
    n: int
    k: int
    success: bool = True
    extra: dict = field(default_factory=dict)

def _info(resid, y, k):
    n = len(y)
    rss = float(np.sum(resid ** 2))
    rss_n = max(rss / n, 1e-300)
    tss = float(np.sum((y - y.mean()) ** 2)) or 1e-300

    return dict(rmse=np.sqrt(rss / n), r2=1 - rss / tss, aic=n * np.log(rss_n) + 2 * k, bic=n * np.log(rss_n) + k * np.log(n), n=n, k=k)

def fit_static(model: str, w, y, accel=None, offset: bool = False, delta: float = 2.0) -> FitResult:
    """Nonlinear LS fit of a static friction model (+ J if accel given, + offset T0)."""
    w, y = np.asarray(w, float), np.asarray(y, float)
    fmodel = "stribeck" if model == "lugre" else model
    fnames = list(STATIC_PARAMS[fmodel])
    use_J = accel is not None

    # linear Coulomb+viscous fit -> initial guess
    cols = [np.sign(w), w] + ([accel] if use_J else []) + ([np.ones_like(w)] if offset else [])
    lin, *_ = np.linalg.lstsq(np.column_stack(cols), y, rcond=None)
    Tc0 = max(lin[0], 1e-6 * (np.abs(y).max() + 1e-12))
    B0 = max(lin[1], 1e-12)
    J0 = max(lin[2], 1e-12) if use_J else None
    wabs = np.abs(w[w != 0]) if np.any(w != 0) else np.array([1.0])
    guess = {"Tc": Tc0, "B": B0, "Tst": 1.5 * Tc0, "ws": max(np.percentile(wabs, 10), 1e-3)}

    names = (["J"] if use_J else []) + fnames + (["T0"] if offset else [])
    x0 = np.array([J0 if n == "J" else (lin[-1] if n == "T0" else guess[n]) for n in names], float)
    lb = np.array([-np.inf if n == "T0" else (1e-9 if n == "ws" else 0.0) for n in names])
    ub = np.full(len(names), np.inf)
    x0 = np.clip(x0, lb + 1e-15, None) if len(x0) else x0

    def predict(x):
        p = dict(zip(names, x))
        p["delta"] = delta
        out = static_friction(fmodel, w, p) if fnames else np.zeros_like(w)

        if use_J:
            out = out + p["J"] * accel
        return out + p.get("T0", 0.0)

    if len(names) == 0:
        resid = -y

        return FitResult(model, {}, **_info(resid, y, 0))
    
    sol = least_squares(lambda x: predict(x) - y, x0, bounds=(lb, ub), x_scale=np.maximum(np.abs(x0), 1e-12))
    params = dict(zip(names, map(float, sol.x)))

    if fmodel == "stribeck":
        params["delta"] = delta
    return FitResult(model, params, **_info(sol.fun, y, len(names)), success=bool(sol.success))

def compare_models(w, y, models=("viscous", "coulomb", "coulomb_viscous", "stribeck"), accel=None, offset=False) -> tuple[pd.DataFrame, dict]:
    fits = {m: fit_static(m, w, y, accel=accel, offset=offset) for m in models}
    rows = [{"model": m, "k": f.k, "rmse": f.rmse, "r2": f.r2, "aic": f.aic, "bic": f.bic, **{p: v for p, v in f.params.items()}} for m, f in fits.items()]
    table = pd.DataFrame(rows).sort_values("bic").reset_index(drop=True)

    return table, fits

def steady_state_points(t, ref, w, torque, settle: float = 0.5, min_len: int = 10):
    """Average speed & torque over the settled part of each constant-reference segment.

    Speed is better taken from position: pass w as None and theta via `w=('theta', theta)`.
    """
    ref = np.asarray(ref)
    change = np.flatnonzero(np.abs(np.diff(ref)) > 1e-12 * (np.abs(ref).max() + 1)) + 1
    bounds = np.r_[0, change, len(ref)]
    ws, ts, rs = [], [], []

    for a, b in zip(bounds[:-1], bounds[1:]):
        s = a + int(settle * (b - a))

        if b - s < min_len:
            continue

        if isinstance(w, tuple) and w[0] == "theta":
            th = w[1]
            wv = (th[b - 1] - th[s]) / (t[b - 1] - t[s])
        else:
            wv = float(np.mean(w[s:b]))
        
        ws.append(wv)
        ts.append(float(np.mean(torque[s:b])))
        rs.append(float(ref[a]))
    
    return np.array(ws), np.array(ts), np.array(rs)