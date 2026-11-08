"""Output-error (simulation-error) identification.

The measured terminal voltage is replayed into the open-loop plant model; the
parameters (in log-space, so they stay positive) are adjusted by nonlinear least
squares so that the simulated currents and speed match the measurements. The
finite-difference Jacobian is computed in ONE batched simulation (P+1 parameter sets).
This is the step that can also identify dynamic friction (LuGre sigma0, sigma1).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from motorsim.core.analysis.filters import mechanical_signals
from motorsim.core.config import flat_params, load_config
from motorsim.core.sim.simulator import simulate


@dataclass
class OEResult:
    params: dict
    fit_names: list
    cost0: float
    cost: float
    nfev: int
    success: bool
    message: str


def replay_config(cfg, friction_model: str | None = None, substeps: int = 2) -> dict:
    """Open-loop, ideal-sensor version of cfg that replays a measured voltage."""
    from motorsim.domains import get_domain

    cfg = copy.deepcopy(load_config(cfg))
    if friction_model:
        cfg["friction"]["model"] = friction_model
    cfg["control"]["mode"] = "voltage"
    cfg["sensors"] = {"current_noise": 0.0, "encoder_cpr": 0, "speed_source": "true"}
    cfg["sim"]["substeps"] = substeps
    return get_domain(cfg["motor"]["type"]).replay_overrides(cfg)


def replay(cfg, data: dict, params: dict | None = None, N: int | None = None, substeps: int = 2,
           friction_model: str | None = None, w_meas=None):
    """Simulate the model with the measured voltage(s). params may hold (N,) arrays."""
    from motorsim.domains import get_domain

    rc = replay_config(cfg, friction_model, substeps)
    base = flat_params(rc)
    if params:
        base.update(params)
    rp = get_domain(rc["motor"]["type"]).replay_inputs(data)
    if w_meas is None:
        w_meas, _ = mechanical_signals(data["t"], data.get("theta"), data.get("w"))
    init = dict(rp["init"], w=w_meas[0], th=data["theta"][0] if "theta" in data else 0.0)
    ref = rp["ref"] if N is None else np.broadcast_to(rp["ref"], (N, len(rp["ref"])))
    return simulate(rc, params=base, N=N, ref=ref, ref_d=rp["ref_d"], init=init)


def fit_output_error(cfg, data: dict, init_params: dict, fit_names, friction_model: str,
                     substeps: int = 2, decimate: int = 1, max_nfev: int = 30, fd_step: float = 1e-3,
                     w_weight: float = 1.0, verbose: int = 0) -> OEResult:
    from motorsim.domains import get_domain

    cfg = load_config(cfg)
    dom = get_domain(cfg["motor"]["type"])
    fit_names = list(fit_names)
    base = flat_params(cfg)
    base.update(init_params)
    w_meas, _ = mechanical_signals(data["t"], data.get("theta"), data.get("w"))
    cur = dom.CUR_KEYS
    meas = [np.asarray(data[c], float) for c in cur]
    s_cur = [max(np.std(y), 1e-9) for y in meas]
    s_w = max(np.std(w_meas), 1e-9)

    def residuals(X):
        M = X.shape[0]
        P = {k: np.full(M, v) for k, v in base.items()}
        for j, n in enumerate(fit_names):
            P[n] = np.exp(X[:, j])
        P = dom.tie_fit(P, base, fit_names)
        r = replay(cfg, data, P, N=M, substeps=substeps, friction_model=friction_model, w_meas=w_meas)
        parts = [(r.data[c] - y) / s for c, y, s in zip(cur, meas, s_cur)]
        parts.append(w_weight * (r.data["w"] - w_meas) / s_w)
        return np.concatenate(parts, axis=1)[:, ::decimate] / np.sqrt(len(parts))

    cache = {}

    def fun(x):
        key = x.tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = residuals(x[None])[0]
        return cache[key]

    def jac(x):
        X = np.vstack([x, x + fd_step * np.eye(len(x))])
        R = residuals(X)
        cache.clear()
        cache[x.tobytes()] = R[0]
        return ((R[1:] - R[0]) / fd_step).T

    x0 = np.log([base[n] for n in fit_names])
    r0 = fun(x0)
    sol = least_squares(fun, x0, jac=jac, method="trf", max_nfev=max_nfev, verbose=verbose)
    params = dict(base)
    params.update({n: float(np.exp(v)) for n, v in zip(fit_names, sol.x)})
    params = {k: float(np.asarray(v).reshape(-1)[0]) for k, v in dom.tie_fit(params, base, fit_names).items()}
    return OEResult(params, fit_names, float(0.5 * r0 @ r0), float(sol.cost), int(sol.nfev),
                    bool(sol.success), str(sol.message))
