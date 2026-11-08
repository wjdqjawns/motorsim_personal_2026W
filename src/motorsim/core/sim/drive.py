"""Power stage between controller output and plant input.

A drive turns the controller output u[k] into a list of (plant_input, h) segments that
cover one control period Ts. h may be a scalar or an (N,) array.
"""
from __future__ import annotations

class ZOHDrive:
    """Ideal amplifier: the commanded voltage is applied unchanged (zero-order hold)."""

    def __init__(self, Ts: float, n_sub: int):
        self.h = Ts / n_sub
        self.n_sub = n_sub

    def segments(self, u):
        return [(u, self.h)] * self.n_sub

    def log_signals(self) -> dict:
        return {}

    def segment_signals(self, s: int, idx: int = 0) -> dict:
        return {}