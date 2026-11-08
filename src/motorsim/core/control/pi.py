"""Discrete PI controller with output clamp and back-calculation anti-windup (batched)."""
from __future__ import annotations

import numpy as np


class PI:
    def __init__(self, kp, ki, Ts, lim, N):
        self.kp, self.ki, self.Ts, self.lim = float(kp), float(ki), Ts, lim
        self.kb = min(ki / kp, 0.5 / Ts) if kp > 0 else 0.0
        self.xi = np.zeros(N)

    def step(self, e, ff=0.0):
        u = self.kp * e + self.xi + ff
        us = np.clip(u, -self.lim, self.lim)
        self.xi += self.Ts * (self.ki * e + self.kb * (us - u))
        return us

    def windup_correct(self, delta):
        """Extra saturation applied downstream (e.g. PMSM voltage circle)."""
        self.xi += self.Ts * self.kb * delta
