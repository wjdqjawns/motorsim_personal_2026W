"""DC motor plant.   x = [i, w, theta]   (shape (3, N))

    L di/dt = v - R i - Ke w
    J dw/dt = Kt i - T_f - T_L
"""
from __future__ import annotations

import numpy as np

class DCMotor:
    nc = 1
    cur_names = ("i",)
    input_names = ("v",)
    iw, ith = 1, 2

    def __init__(self, P: dict):
        self.R, self.L, self.Kt, self.Ke, self.J = (P[k] for k in ("R", "L", "Kt", "Ke", "J"))

    def initial_state(self, N, init=None):
        x = np.zeros((3, N))

        if init:
            if "i" in init:
                x[0] = np.asarray(init["i"], float).reshape(-1)

            x[1] = init.get("w", 0.0)
            x[2] = init.get("th", 0.0)

        return x

    def plant_input(self, u):
        return u

    def currents(self, x):
        return x[:1]

    def log_signals(self, x) -> dict:
        return {"i": x[0]}

    def torque(self, x):
        return self.Kt * x[0]

    def deriv(self, x, u, tf, tl):
        i, w = x[0], x[1]
        
        return np.array([
            (u[0] - self.R * i - self.Ke * w) / self.L,
            (self.Kt * i - tf - tl) / self.J,
            w,
        ])