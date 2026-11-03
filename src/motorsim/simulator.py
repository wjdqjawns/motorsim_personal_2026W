"""Batched closed-loop motor simulation.

Timing convention (matches a typical drive ISR): at sample k the sensors are read,
the controller computes u[k], and u[k] is held (ZOH) over [k Ts, (k+1) Ts]. The plant
is integrated with RK4 on `substeps` sub-intervals; friction is evaluated once per
sub-step (Karnopp stick/slip for discontinuous models, implicit bristle update for LuGre).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from motorsim import signals
from motorsim.config import flat_params, load_config, nominal_params
from motorsim.control import Controller
from motorsim.models import Friction, make_plant, rk4
from motorsim.sensors import Sensors

@dataclass
class SimResult:
    t: np.ndarray                         # (T,)
    data: dict                            # name -> (N, T)
    params: dict                          # name -> (N,)
    config: dict = field(default_factory=dict)

    @property
    def N(self):
        return next(iter(self.data.values())).shape[0]

    def frame(self, idx: int = 0) -> pd.DataFrame:
        return pd.DataFrame({"t": self.t, **{k: v[idx] for k, v in self.data.items()}})

    def save(self, path: str | Path, idx: int | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".csv":
            self.frame(0 if idx is None else idx).to_csv(path, index=False)
        else:
            cfg = {k: v for k, v in self.config.items() if not k.startswith("_")}
            np.savez_compressed(
                path, t=self.t,
                **{f"data/{k}": v for k, v in self.data.items()},
                **{f"param/{k}": v for k, v in self.params.items()},
                config=yaml.safe_dump(cfg, sort_keys=False),
            )


def _as_batch(P: dict, N: int) -> dict:
    return {k: np.broadcast_to(np.asarray(v, float), (N,)).copy() for k, v in P.items()}


def simulate(cfg, params: dict | None = None, N: int | None = None, ref=None, ref_d=None,
             init: dict | None = None, rng: np.random.Generator | None = None) -> SimResult:
    """Run a (batched) simulation.

    cfg     : config dict or YAML path
    params  : flat plant parameters (scalars or (N,) arrays); default from cfg
    ref     : optional explicit reference array (T,) or (N, T) -> overrides cfg['reference']
    init    : optional initial state {'i': (nc,) or (nc,N), 'w': ..., 'th': ...}
    """
    cfg = load_config(cfg)
    mtype = cfg["motor"]["type"]
    sc, cc = cfg["sim"], cfg["control"]
    Ts = float(cc["Ts"])
    n_sub = int(sc["substeps"])
    h = Ts / n_sub
    rng = rng if rng is not None else np.random.default_rng(sc["seed"])

    P = flat_params(cfg)
    if params:
        P.update(params)
    if N is None:
        sizes = [np.size(v) for v in P.values()] + ([np.shape(ref)[0]] if ref is not None and np.ndim(ref) == 2 else [])
        N = max(max(sizes), int(sc.get("batch", 1)))
    P = _as_batch(P, N)

    if ref is not None:
        ref = np.asarray(ref, float)
        T = ref.shape[-1]
    else:
        T = int(round(sc["duration"] / Ts))
    t = np.arange(T) * Ts
    ref = np.broadcast_to(ref, (N, T)) if ref is not None else signals.build(cfg["reference"], t, N, rng)
    if ref_d is not None:
        ref_d = np.broadcast_to(np.asarray(ref_d, float), (N, T))
    else:
        ref_d = signals.build(cfg.get("reference_d"), t, N, rng)

    plant = make_plant(mtype, P)
    nc = plant.nc
    fric = Friction(cfg["friction"]["model"], P, N, cfg["friction"].get("v_eps", 1e-3))
    ctrl = Controller(cfg, nominal_params(cfg), N)
    sens = Sensors(cfg["sensors"], Ts, N, nc, rng)

    x = np.zeros((nc + 2, N))
    if init:
        if "i" in init:
            x[:nc] = np.asarray(init["i"], float).reshape(nc, -1)
        x[nc] = init.get("w", 0.0)
        x[nc + 1] = init.get("th", 0.0)
    sens.reset(x)

    names = (["ref"] + (["ref_d"] if nc == 2 else []) + list(plant.volt_names) + list(plant.cur_names)
             + [f"{c}_meas" for c in plant.cur_names]
             + ["w", "w_meas", "theta", "theta_meas", "Te", "Tf", "i_ref", "w_ref"])
    log = {k: np.empty((N, T)) for k in names}
    TL, cogA, cogN, cogphi = P["TL"], P["cog_A"], P["cog_N"], P["cog_phi"]
    has_cog = bool(np.any(cogA != 0))
    iw, ith = nc, nc + 1

    for k in range(T):
        m = sens.measure(x)
        u = ctrl.step(ref[:, k], ref_d[:, k], m)
        log["ref"][:, k] = ref[:, k]
        if nc == 2:
            log["ref_d"][:, k] = ref_d[:, k]
        for j in range(nc):
            log[plant.volt_names[j]][:, k] = u[j]
            log[plant.cur_names[j]][:, k] = x[j]
            log[plant.cur_names[j] + "_meas"][:, k] = m["i"][j]
        log["w"][:, k] = x[iw]
        log["w_meas"][:, k] = m["w"]
        log["theta"][:, k] = x[ith]
        log["theta_meas"][:, k] = m["th"]
        log["i_ref"][:, k] = ctrl.i_ref
        log["w_ref"][:, k] = ctrl.w_ref

        for s in range(n_sub):
            tl = TL + cogA * np.sin(cogN * x[ith] + cogphi) if has_cog else TL
            te = plant.torque(x)
            w0 = x[iw]
            tf = fric.torque(w0, te - tl, h)
            if s == 0:
                log["Te"][:, k] = te
                log["Tf"][:, k] = tf
            xn = rk4(plant.deriv, x, u, tf, tl, h)
            if fric.karnopp:
                xn[iw] = fric.stick(w0, xn[iw], te - tl)
            x = xn

    return SimResult(t=t, data=log, params=P, config=copy.deepcopy(cfg))


def load_result(path: str | Path) -> SimResult:
    z = np.load(path, allow_pickle=False)
    data = {k[5:]: z[k] for k in z.files if k.startswith("data/")}
    params = {k[6:]: z[k] for k in z.files if k.startswith("param/")}
    cfg = yaml.safe_load(str(z["config"])) if "config" in z.files else {}
    return SimResult(t=z["t"], data=data, params=params, config=cfg)
