"""Motor plant models. State arrays have shape (nx, N); N = batch of parameter sets.

DC motor          x = [i, w, theta]
    L di/dt = v - R i - Ke w
    J dw/dt = Kt i - T_f - T_L

PMSM (dq, rotor-flux frame, amplitude-invariant Park)    x = [id, iq, w, theta]
    Ld did/dt = vd - R id + we Lq iq
    Lq diq/dt = vq - R iq - we (Ld id + psi)
    Te = 1.5 p (psi iq + (Ld - Lq) id iq),   we = p w
    J dw/dt = Te - T_f - T_L
"""
from __future__ import annotations

import numpy as np

class DCMotor:
    nc = 1
    cur_names = ("i",)
    volt_names = ("v",)

    def __init__(self, P: dict):
        self.R, self.L, self.Kt, self.Ke, self.J = (P[k] for k in ("R", "L", "Kt", "Ke", "J"))

    def torque(self, x):
        return self.Kt * x[0]

    def deriv(self, x, u, tf, tl):
        i, w = x[0], x[1]
        return np.array([
            (u[0] - self.R * i - self.Ke * w) / self.L,
            (self.Kt * i - tf - tl) / self.J,
            w,
        ])

class PMSM:
    nc = 2
    cur_names = ("id", "iq")
    volt_names = ("vd", "vq")

    def __init__(self, P: dict):
        self.R, self.Ld, self.Lq, self.psi, self.p, self.J = (P[k] for k in ("R", "Ld", "Lq", "psi", "p", "J"))
        self.k = 1.5 * self.p

    def torque(self, x):
        return self.k * (self.psi * x[1] + (self.Ld - self.Lq) * x[0] * x[1])

    def deriv(self, x, u, tf, tl):
        id_, iq, w = x[0], x[1], x[2]
        we = self.p * w
        te = self.k * (self.psi * iq + (self.Ld - self.Lq) * id_ * iq)
        return np.array([
            (u[0] - self.R * id_ + we * self.Lq * iq) / self.Ld,
            (u[1] - self.R * iq - we * (self.Ld * id_ + self.psi)) / self.Lq,
            (te - tf - tl) / self.J,
            w,
        ])

def make_plant(mtype: str, P: dict):
    if mtype == "dc":
        return DCMotor(P)
    if mtype == "pmsm":
        return PMSM(P)
    raise ValueError(f"unknown motor type {mtype!r}")

def rk4(f, x, u, tf, tl, h):
    k1 = f(x, u, tf, tl)
    k2 = f(x + 0.5 * h * k1, u, tf, tl)
    k3 = f(x + 0.5 * h * k2, u, tf, tl)
    k4 = f(x + h * k3, u, tf, tl)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)