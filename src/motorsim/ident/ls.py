"""Linear-in-parameters (equation-error) identification from measured signals.

Electrical part (sampled, ZOH voltage over [k, k+1] -> compare with interval mid-point):
    DC   : v = R i + L di/dt + Ke w                          -> [R, L, Ke]
    PMSM : vd = R id + Ld did/dt - we Lq iq
           vq = R iq + Lq diq/dt + we Ld id + we psi          -> [R, Ld, Lq, psi]
    All signals pass through the same zero-phase low-pass filter first (keeps the
    equation consistent while suppressing derivative noise).
Mechanical part (angle, current pass the SAME zero-phase low-pass filter; speed and
acceleration are central differences of the filtered angle -> consistent equation):
    Te(i) = J dw/dt + T_f(w) [+ T0]          for each candidate friction model
    with Kt = Ke (DC, SI) or 1.5 p psi (PMSM) taken from the electrical fit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .friction_fit import compare_models

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
        p.update({k: v for k, v in self.fits[model].params.items()})
        return p


def mechanical_signals(t, theta=None, w=None, lp_hz=None):
    """Speed and acceleration from the encoder angle (preferred) or a speed signal."""
    Ts = t[1] - t[0]
    fs = 1.0 / Ts
    lp_hz = 0.04 * fs if lp_hz is None else lp_hz
    if theta is not None:
        w_s = np.gradient(_lp(theta, fs, lp_hz), Ts)
    else:
        w_s = _lp(w, fs, lp_hz)
    return w_s, np.gradient(w_s, Ts)


def _lp(x, fs, fc, order=2):
    if fc is None or fc <= 0 or fc >= fs / 2:
        return np.asarray(x, float)
    b, a = butter(order, fc / (fs / 2))
    return filtfilt(b, a, x)


def _lstsq(A, b):
    th, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = b - A @ th
    r2 = 1 - np.sum(r ** 2) / max(np.sum((b - b.mean()) ** 2), 1e-300)
    return th, float(r2)


def identify_ls(data: dict, motor_type: str, friction_models=DEFAULT_MODELS, pole_pairs: float | None = None,
                lp_hz: float | None = None, mech_lp_hz: float | None = None,
                w_min: float | None = None, delay: int = 0, offset: bool = False, trim: int = 100) -> IdentResult:
    """data: dict of 1-D arrays with keys t, theta (or w) and v,i (DC) or vd,vq,id,iq (PMSM)."""
    t = np.asarray(data["t"], float)
    Ts = t[1] - t[0]
    fs = 1.0 / Ts
    lp_hz = 0.2 * fs if lp_hz is None else lp_hz
    theta = data.get("theta")
    mech_lp_hz = 0.04 * fs if mech_lp_hz is None else mech_lp_hz
    w_s, a_s = mechanical_signals(t, theta, data.get("w"), mech_lp_hz)
    T = len(t)
    d = delay
    sl = slice(trim, T - 1 - d - trim)          # interval index range after delay shift
    mid = lambda x: 0.5 * (x[:-1] + x[1:])

    if motor_type == "dc":
        v = _lp(data["v"], fs, lp_hz)
        i = _lp(data["i"], fs, lp_hz)
        wf = _lp(w_s, fs, lp_hz)
        ib, di, wb = mid(i)[d:], np.diff(i)[d:] / Ts, mid(wf)[d:]
        vk = v[: T - 1 - d]
        A = np.column_stack([ib, di, wb])[sl]
        th, r2e = _lstsq(A, vk[sl])
        elec = {"R": th[0], "L": th[1], "Ke": th[2], "Kt": th[2]}
        i_s = _lp(data["i"], fs, mech_lp_hz)
        te = elec["Kt"] * i_s
    elif motor_type == "pmsm":
        if pole_pairs is None:
            raise ValueError("PMSM identification needs pole_pairs")
        p = float(pole_pairs)
        vd, vq = _lp(data["vd"], fs, lp_hz), _lp(data["vq"], fs, lp_hz)
        id_, iq = _lp(data["id"], fs, lp_hz), _lp(data["iq"], fs, lp_hz)
        we = p * _lp(w_s, fs, lp_hz)
        idb, iqb, web = mid(id_)[d:], mid(iq)[d:], mid(we)[d:]
        did, diq = np.diff(id_)[d:] / Ts, np.diff(iq)[d:] / Ts
        z = np.zeros_like(idb)
        # params: [R, Ld, Lq, psi]
        Ad = np.column_stack([idb, did, -web * iqb, z])[sl]
        Aq = np.column_stack([iqb, web * idb, diq, web])[sl]
        A = np.vstack([Ad, Aq])
        b = np.r_[vd[: T - 1 - d][sl], vq[: T - 1 - d][sl]]
        th, r2e = _lstsq(A, b)
        elec = {"R": th[0], "Ld": th[1], "Lq": th[2], "psi": th[3], "p": p}
        id_s = _lp(data["id"], fs, mech_lp_hz)
        iq_s = _lp(data["iq"], fs, mech_lp_hz)
        te = 1.5 * p * (elec["psi"] * iq_s + (elec["Ld"] - elec["Lq"]) * id_s * iq_s)
    else:
        raise ValueError(motor_type)
    elec = {k: float(v) for k, v in elec.items()}

    edge = trim
    m = np.zeros(T, bool)
    m[edge:T - edge] = True
    w_min = 0.02 * np.abs(w_s[m]).max() if w_min is None else w_min
    m &= np.abs(w_s) > w_min
    table, fits = compare_models(w_s[m], te[m], models=friction_models, accel=a_s[m], offset=offset)
    best = table.iloc[0]["model"]
    params = dict(elec)
    params.update(fits[best].params)
    diag = {"r2_electrical": r2e, "n_mech": int(m.sum()), "w_min": float(w_min), "lp_hz": lp_hz,
            "mech_lp_hz": mech_lp_hz}
    return IdentResult(motor_type, params, best, elec, table, fits, diag)
