"""DC motor: electrical equation-error regression and replay inputs.

    v[k] = R i_mid + L (i[k+1] - i[k]) / Ts + Ke w_mid      (ZOH voltage over [k, k+1])
"""
from __future__ import annotations

import numpy as np

def electrical_ls(data: dict, Ts: float, lp, w_f, delay: int, sl: slice, lstsq, pole_pairs=None):
    """lp: zero-phase filter applied to every signal; w_f: filtered speed. Returns (elec, r2)."""
    T = len(data["t"])
    d = delay
    mid = lambda x: 0.5 * (x[:-1] + x[1:])
    v, i = lp(data["v"]), lp(data["i"])
    A = np.column_stack([mid(i)[d:], np.diff(i)[d:] / Ts, mid(w_f)[d:]])[sl]
    th, r2 = lstsq(A, v[: T - 1 - d][sl])
    return {"R": th[0], "L": th[1], "Ke": th[2], "Kt": th[2]}, r2

def electromagnetic_torque(elec: dict, data: dict, filt):
    return elec["Kt"] * filt(data["i"])

def replay_inputs(data: dict) -> dict:
    return {"ref": data["v"], "ref_d": None, "init": {"i": [data["i"][0]]}}