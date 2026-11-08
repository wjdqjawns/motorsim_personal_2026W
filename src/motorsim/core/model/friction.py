"""Friction models (batched over N parameter sets).

Static (memoryless) models, written as T_f(w) for sliding motion:
    none             0
    viscous          B w
    coulomb          Tc sgn(w)
    coulomb_viscous  Tc sgn(w) + B w
    stribeck         [Tc + (Tst - Tc) exp(-|w/ws|^delta)] sgn(w) + B w
Dynamic:
    lugre            dz/dt = w - sigma0 |w| z / g(w),  g(w) = Stribeck curve without B
                     T_f = sigma0 z + sigma1 dz/dt + B w

Static models with a discontinuity at w=0 use a Karnopp stick/slip treatment in the
simulator so that the motor really sticks (|T_drive| <= breakaway torque -> w = 0).
The integration step h may be a scalar or an (N,) array (PWM segments).
"""
from __future__ import annotations

import numpy as np

STATIC_PARAMS = {
    "none": [],
    "viscous": ["B"],
    "coulomb": ["Tc"],
    "coulomb_viscous": ["Tc", "B"],
    "stribeck": ["Tc", "Tst", "ws", "B"],
    "lugre": ["Tc", "Tst", "ws", "B"],   # steady-state (sliding) part of LuGre
}
DYNAMIC_PARAMS = {"lugre": ["sigma0", "sigma1"]}
MODELS = list(STATIC_PARAMS)
KARNOPP_MODELS = {"coulomb", "coulomb_viscous", "stribeck"}


def model_params(model: str) -> list[str]:
    return STATIC_PARAMS[model] + DYNAMIC_PARAMS.get(model, [])


def stribeck_g(w, p):
    delta = p.get("delta", 2.0)
    return p["Tc"] + (p["Tst"] - p["Tc"]) * np.exp(-np.abs(w / p["ws"]) ** delta)


def static_friction(model: str, w, p: dict):
    """Steady sliding friction torque T_f(w) for any model (LuGre -> its steady state)."""
    if model == "none":
        return np.zeros_like(w)
    if model == "viscous":
        return p["B"] * w
    if model == "coulomb":
        return p["Tc"] * np.sign(w)
    if model == "coulomb_viscous":
        return p["Tc"] * np.sign(w) + p["B"] * w
    if model in ("stribeck", "lugre"):
        return stribeck_g(w, p) * np.sign(w) + p["B"] * w
    raise ValueError(f"unknown friction model {model!r}; choose from {MODELS}")


def breakaway_torque(model: str, p: dict):
    if model in ("stribeck", "lugre"):
        return p["Tst"]
    if model in ("coulomb", "coulomb_viscous"):
        return p["Tc"]
    return 0.0


class Friction:
    """Stateful friction element used inside the simulator."""

    def __init__(self, model: str, P: dict, N: int, v_eps: float = 1e-3):
        if model not in STATIC_PARAMS:
            raise ValueError(f"unknown friction model {model!r}; choose from {MODELS}")
        self.model = model
        self.p = {k: P[k] for k in ("B", "Tc", "Tst", "ws", "delta", "sigma0", "sigma1") if k in P}
        self.karnopp = model in KARNOPP_MODELS
        self.fs = breakaway_torque(model, self.p)
        self.v_eps = v_eps
        self.z = np.zeros(N)

    def torque(self, w, t_drive, h):
        """Friction torque held constant over the next integration step of length h."""
        if self.model == "lugre":
            p = self.p
            g = stribeck_g(w, p)
            a = p["sigma0"] * np.abs(w) / g
            z = (self.z + h * w) / (1.0 + h * a)       # implicit Euler: stiff bristle state
            zd = w - a * z
            self.z = z
            return p["sigma0"] * z + p["sigma1"] * zd + p["B"] * w
        tf = static_friction(self.model, w, self.p)
        if self.karnopp:
            stick = np.abs(w) < self.v_eps
            if stick.any():
                tf = np.where(stick, np.clip(t_drive, -self.fs, self.fs), tf)
        return tf

    def stick(self, w_old, w_new, t_drive):
        """Karnopp: stop the rotor at a zero crossing if the drive cannot break it away."""
        hold = ((w_old * w_new < 0) | (np.abs(w_new) < self.v_eps)) & (np.abs(t_drive) <= self.fs)
        return np.where(hold, 0.0, w_new)
