"""Linear-in-parameters (equation-error) identification from measured signals.

Electrical part (domain-specific regression, see domains/<type>/*_ident.py): sampled ZOH
voltage over [k, k+1] compared with the interval mid-point. All signals pass through the
same zero-phase low-pass filter first (keeps the equation consistent, suppresses noise).
Mechanical part: angle and current pass the SAME zero-phase low-pass filter; speed and
acceleration are central differences of the filtered angle -> consistent equation
    Te(i) = J dw/dt + T_f(w) [+ T0]          for each candidate friction model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from motorsim.core.analysis.filters import lowpass, mechanical_signals
from motorsim.core.analysis.items.friction_fit import compare_models

DEFAULT_MODELS = ("viscous", "coulomb", "coulomb_viscous", "stribeck")


@dataclass
class IdentResult:
    motor_type: str
    params: dict                    # flat params with the best friction model
    best_model: str
    electrical: dict
    table: pd.DataFrame             # friction-model comparison (mechanical equation)
    fits: dict
    diagnostics: dict = field(default_factory=dict)

    def params_for(self, model: str) -> dict:
        p = dict(self.electrical)
        p.update(self.fits[model].params)
        return p

def _lstsq(A, b):
    th, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = b - A @ th
    r2 = 1 - np.sum(r ** 2) / max(np.sum((b - b.mean()) ** 2), 1e-300)
    
    return th, float(r2)

def identify_ls(data: dict, motor_type: str, friction_models=DEFAULT_MODELS, pole_pairs: float | None = None,
                lp_hz: float | None = None, mech_lp_hz: float | None = None,
                w_min: float | None = None, delay: int = 0, offset: bool = False, trim: int = 100) -> IdentResult:
    """data: dict of 1-D arrays with keys t, theta (or w) and v,i (DC) or vd,vq,id,iq (PMSM)."""
    from motorsim.domains import get_domain

    dom = get_domain(motor_type)
    t = np.asarray(data["t"], float)
    Ts = t[1] - t[0]
    fs = 1.0 / Ts
    T = len(t)
    lp_hz = 0.2 * fs if lp_hz is None else lp_hz
    mech_lp_hz = 0.04 * fs if mech_lp_hz is None else mech_lp_hz
    w_s, a_s = mechanical_signals(t, data.get("theta"), data.get("w"), mech_lp_hz)

    sl = slice(trim, T - 1 - delay - trim)
    elec, r2e = dom.electrical_ls(data, Ts, lambda x: lowpass(x, fs, lp_hz), lowpass(w_s, fs, lp_hz), delay, sl, _lstsq, pole_pairs)
    elec = {k: float(v) for k, v in elec.items()}
    te = dom.electromagnetic_torque(elec, data, lambda x: lowpass(x, fs, mech_lp_hz))

    m = np.zeros(T, bool)
    m[trim:T - trim] = True
    w_min = 0.02 * np.abs(w_s[m]).max() if w_min is None else w_min
    m &= np.abs(w_s) > w_min
    table, fits = compare_models(w_s[m], te[m], models=friction_models, accel=a_s[m], offset=offset)
    best = table.iloc[0]["model"]
    params = dict(elec)
    params.update(fits[best].params)
    diag = {"r2_electrical": r2e, "n_mech": int(m.sum()), "w_min": float(w_min), "lp_hz": float(lp_hz), "mech_lp_hz": float(mech_lp_hz)}
    
    return IdentResult(motor_type, params, best, elec, table, fits, diag)