"""Batched closed-loop motor simulation (motor-type independent).

Timing convention (matches a typical drive ISR): at sample k the sensors are read,
the controller computes u[k], and the drive applies it over [k Ts, (k+1) Ts] as one or
more segments (ZOH sub-steps, or PWM switching intervals). Each segment is integrated
with RK4; friction is evaluated once per segment (Karnopp stick/slip for discontinuous
models, implicit bristle update for LuGre).

The motor type plugs in through motorsim.domains.<type>: plant, controller, drive.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from motorsim.core.config import flat_params, load_config, nominal_params
from motorsim.core.model.friction import Friction
from motorsim.core.model.integrators import rk4
from motorsim.core.model.results import SimResult
from motorsim.core.sensors.measurement import Sensors
from motorsim.core.signals import build


def as_batch(P: dict, N: int) -> dict:
    return {k: np.broadcast_to(np.asarray(v, float), (N,)).copy() for k, v in P.items()}


class _Recorder:
    def __init__(self, N, T):
        self.N, self.T, self.log = N, T, {}

    def add(self, k, row):
        for name, v in row.items():
            arr = self.log.get(name)
            if arr is None:
                arr = self.log[name] = np.empty((self.N, self.T))
            arr[:, k] = v


def simulate(cfg, params: dict | None = None, N: int | None = None, ref=None, ref_d=None,
             init: dict | None = None, rng: np.random.Generator | None = None) -> SimResult:
    """Run a (batched) simulation.

    cfg     : config dict or YAML path
    params  : flat plant parameters (scalars or (N,) arrays); default from cfg
    ref     : optional explicit reference array (T,) or (N, T) -> overrides cfg['reference']
    init    : optional initial state {'i': currents, 'w': speed, 'th': angle}
    """
    from motorsim.domains import get_domain

    cfg = load_config(cfg)
    dom = get_domain(cfg["motor"]["type"])
    sc, cc = cfg["sim"], cfg["control"]
    Ts = float(cc["Ts"])
    n_sub = int(sc["substeps"])
    rng = rng if rng is not None else np.random.default_rng(sc["seed"])

    P = flat_params(cfg)
    if params:
        P.update(params)
    if N is None:
        sizes = [np.size(v) for v in P.values()] + ([np.shape(ref)[0]] if ref is not None and np.ndim(ref) == 2 else [])
        N = max(max(sizes), int(sc.get("batch", 1)))
    P = as_batch(P, N)

    if ref is not None:
        ref = np.asarray(ref, float)
        T = ref.shape[-1]
    else:
        T = int(round(sc["duration"] / Ts))
    t = np.arange(T) * Ts
    refs = {"ref": np.broadcast_to(ref, (N, T)) if ref is not None else build(cfg["reference"], t, N, rng)}
    refs["ref_d"] = (np.broadcast_to(np.asarray(ref_d, float), (N, T)) if ref_d is not None
                     else build(cfg.get("reference_d"), t, N, rng))
    if cfg.get("reference_abc"):
        for ph, spec in zip("abc", cfg["reference_abc"]):
            refs[f"ref_{ph}"] = build(spec, t, N, rng)

    plant = dom.make_plant(P, cfg)
    ctrl = dom.make_controller(cfg, nominal_params(cfg), N)
    drive = dom.make_drive(cfg, P, N)
    fric = Friction(cfg["friction"]["model"], P, N, cfg["friction"].get("v_eps", 1e-3))
    sens = Sensors(cfg["sensors"], Ts, N, rng)

    x = plant.initial_state(N, init)
    iw, ith = plant.iw, plant.ith
    sens.reset(x[iw], x[ith])
    rec = _Recorder(N, T)
    TL, cogA, cogN, cogphi = P["TL"], P["cog_A"], P["cog_N"], P["cog_phi"]
    has_cog = bool(np.any(cogA != 0))
    cur_names = plant.cur_names
    fast_win = sc.get("fast_log")              # [t0, t1]: log every integration segment (element 0)
    fast = []

    for k in range(T):
        cur = plant.currents(x)
        m = sens.measure(cur, x[iw], x[ith])
        rk = {name: r[:, k] for name, r in refs.items()}
        u = ctrl.step(rk, m)
        segs = drive.segments(u)

        row = {name: v for name, v in rk.items() if name != "ref_d" or plant.nc > 1}
        row.update(plant.log_signals(x))
        for j, c in enumerate(cur_names):
            row[f"{c}_meas"] = m["i"][j]
        row.update(w=x[iw], w_meas=m["w"], theta=x[ith], theta_meas=m["th"])
        row.update(ctrl.log_signals())
        row.update(drive.log_signals())

        log_fast = bool(fast_win) and fast_win[0] <= t[k] <= fast_win[1]
        tt = t[k]
        if log_fast:
            fast.append({"t": tt, **{c: v for c, v in zip(cur_names, plant.currents(x)[:, 0])},
                         **drive.segment_signals(0), "sample": 1.0})
        for s, (up, h) in enumerate(segs):
            tl = TL + cogA * np.sin(cogN * x[ith] + cogphi) if has_cog else TL
            te = plant.torque(x)
            w0 = x[iw]
            tf = fric.torque(w0, te - tl, h)
            if s == 0:
                row["Te"], row["Tf"] = te, tf
            xn = rk4(plant.deriv, x, up, tf, tl, h)
            if fric.karnopp:
                xn[iw] = fric.stick(w0, xn[iw], te - tl)
            x = xn
            if log_fast:
                tt = tt + float(np.broadcast_to(h, (N,))[0])
                nxt = drive.segment_signals(s + 1) if s + 1 < len(segs) else drive.segment_signals(s)
                fast.append({"t": tt, **{c: v for c, v in zip(cur_names, plant.currents(x)[:, 0])},
                             **nxt, "sample": 0.0})
        rec.add(k, row)

    return SimResult(t=t, data=rec.log, params=P, config=copy.deepcopy(cfg),
                     fast=pd.DataFrame(fast) if fast else None)
