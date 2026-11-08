"""Standard experiments: experiment overrides, constant-speed friction map."""
from __future__ import annotations

import copy

import numpy as np

from motorsim.core.analysis.items.friction_fit import steady_state_points
from motorsim.core.config import load_config, nominal_params, torque_constant
from motorsim.core.io.data_reader import from_frame
from motorsim.core.sim.simulator import simulate


def speed_levels(spec: dict) -> np.ndarray:
    """{min, max, n, both_directions} -> log-spaced speed set-points."""
    lv = np.geomspace(spec.get("min", 0.5), spec.get("max", 200.0), int(spec.get("n", 12)))
    return np.r_[lv, -lv] if spec.get("both_directions", True) else lv


def experiment_config(cfg, exp: dict) -> dict:
    """Copy of cfg with an experiment's overrides (control mode, reference, duration, ...)."""
    c = copy.deepcopy(load_config(cfg))
    for key in ("reference", "reference_d", "reference_abc"):
        if key in exp:
            c[key] = exp[key]
    if "mode" in exp:
        c["control"]["mode"] = exp["mode"]
    if "duration" in exp:
        c["sim"]["duration"] = exp["duration"]
    if "seed" in exp:
        c["sim"]["seed"] = exp["seed"]
    return c


def run_friction_map(cfg, speeds=None, hold: float = 0.3, settle: float = 0.5):
    """Constant-speed test: average torque estimate (nominal Kt * current) per set-point."""
    cfg = copy.deepcopy(load_config(cfg))
    speeds = speed_levels({}) if speeds is None else np.asarray(speeds, float)
    cfg["control"]["mode"] = "speed"
    cfg["reference"] = {"type": "steps", "values": speeds.tolist(), "duration": hold}
    cfg["reference_d"] = None
    cfg["sim"]["duration"] = hold * len(speeds)
    res = simulate(cfg)
    data = from_frame(res.frame(0), None)
    w, T = friction_map_points(data, cfg, settle)
    return res, w, T


def friction_map_points(data: dict, cfg, settle: float = 0.5):
    """Steady-state (speed, torque) pairs from a staircase speed test (needs a 'ref' channel)."""
    cfg = load_config(cfg)
    mtype = cfg["motor"]["type"]
    nom = nominal_params(cfg)
    kt = torque_constant(mtype, nom)
    torque = kt * (data["i"] if mtype == "dc" else data["iq"])
    src = ("theta", data["theta"]) if "theta" in data else data["w"]
    w, T, _ = steady_state_points(data["t"], data["ref"], src, torque, settle)
    keep = np.abs(w) > 1e-6       # drop set-points where the rotor never moved (stuck)
    return w[keep], T[keep]
