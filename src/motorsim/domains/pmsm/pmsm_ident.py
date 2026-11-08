"""PMSM: dq electrical equation-error regression and replay inputs.

    vd = R id + Ld did/dt - we Lq iq
    vq = R iq + Lq diq/dt + we Ld id + we psi        -> [R, Ld, Lq, psi]
Uses dq signals (vd, vq commanded; id, iq measured). For abc-frame recordings these are the
drive's own Clarke/Park outputs (id_meas, iq_meas), exactly as a real drive would log them.
"""
from __future__ import annotations

import numpy as np


def electrical_ls(data: dict, Ts: float, lp, w_f, delay: int, sl: slice, lstsq, pole_pairs=None):
    if pole_pairs is None:
        raise ValueError("PMSM identification needs pole_pairs")
    
    p = float(pole_pairs)
    T = len(data["t"])
    d = delay
    mid = lambda x: 0.5 * (x[:-1] + x[1:])
    vd, vq = lp(data["vd"]), lp(data["vq"])
    id_, iq = lp(data["id"]), lp(data["iq"])
    we = p * w_f
    idb, iqb, web = mid(id_)[d:], mid(iq)[d:], mid(we)[d:]
    did, diq = np.diff(id_)[d:] / Ts, np.diff(iq)[d:] / Ts
    z = np.zeros_like(idb)
    Ad = np.column_stack([idb, did, -web * iqb, z])[sl]
    Aq = np.column_stack([iqb, web * idb, diq, web])[sl]
    b = np.r_[vd[: T - 1 - d][sl], vq[: T - 1 - d][sl]]
    th, r2 = lstsq(np.vstack([Ad, Aq]), b)
    return {"R": th[0], "Ld": th[1], "Lq": th[2], "psi": th[3], "p": p}, r2

def electromagnetic_torque(elec: dict, data: dict, filt):
    id_s, iq_s = filt(data["id"]), filt(data["iq"])
    return 1.5 * elec["p"] * (elec["psi"] * iq_s + (elec["Ld"] - elec["Lq"]) * id_s * iq_s)

def replay_inputs(data: dict) -> dict:
    return {"ref": data["vq"], "ref_d": data["vd"], "init": {"i": [data["id"][0], data["iq"][0]]}}