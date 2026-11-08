"""Two-level three-phase inverter models (input: phase duty cycles (3, N) in [0, 1]).

average    : leg voltage = d * Vdc over the whole period (state-space averaged).
switching  : center-aligned PWM, one carrier period per control period Ts. Leg x is high
             during [(1 - d_x) Ts / 2, (1 + d_x) Ts / 2]; the 6 switching instants split the
             period into 7 intervals that are integrated exactly as they are (per-batch
             interval lengths), so current ripple and switching harmonics appear.
             Currents are sampled at the period start (centre of the zero vector), as in a
             real drive.
Phase-to-neutral voltages are leg voltages minus their mean (floating star point).
"""
from __future__ import annotations

import numpy as np

from motorsim.core.control.transforms import clarke


class AverageInverter:
    def __init__(self, vdc, Ts: float, n_sub: int):
        self.vdc, self.h, self.n_sub = vdc, Ts / n_sub, n_sub
        self.vph = None

    def segments(self, duty):
        legs = duty * self.vdc
        self.vph = legs - legs.mean(axis=0)
        u = np.array(clarke(*legs))
        return [(u, self.h)] * self.n_sub

    def log_signals(self) -> dict:
        return {"va": self.vph[0], "vb": self.vph[1], "vc": self.vph[2]}

    def segment_signals(self, s: int, idx: int = 0) -> dict:
        return {}


class SwitchingInverter(AverageInverter):
    def __init__(self, vdc, Ts: float, split: int = 1):
        super().__init__(vdc, Ts, 1)
        self.Ts, self.split = Ts, max(1, int(split))

    def segments(self, duty):
        legs_avg = duty * self.vdc
        self.vph = legs_avg - legs_avg.mean(axis=0)
        ton = 0.5 * (1.0 - duty) * self.Ts
        toff = 0.5 * (1.0 + duty) * self.Ts
        N = duty.shape[1]
        b = np.vstack([np.zeros((1, N)), np.sort(np.vstack([ton, toff]), axis=0), np.full((1, N), self.Ts)])
        segs, self.seg_legs = [], []
        for s in range(7):
            h = b[s + 1] - b[s]
            tm = 0.5 * (b[s] + b[s + 1])
            legs = ((tm >= ton) & (tm < toff)) * self.vdc
            u = np.array(clarke(*legs))
            segs += [(u, h / self.split)] * self.split
            self.seg_legs += [legs > 0] * self.split
        return segs

    def segment_signals(self, s: int, idx: int = 0) -> dict:
        """Leg switch states (1 = high side on) during segment s, for the fast log."""
        sa, sb, sc = self.seg_legs[s][:, idx]
        return {"Sa": float(sa), "Sb": float(sb), "Sc": float(sc)}


def make_inverter(cfg: dict, P: dict, N: int):
    inv = cfg["motor"].get("inverter") or {}
    vdc = cfg["motor"]["limits"]["v_dc"]
    Ts = cfg["control"]["Ts"]
    kind = inv.get("type", "average")
    if kind == "average":
        return AverageInverter(vdc, Ts, int(cfg["sim"]["substeps"]))
    if kind == "switching":
        return SwitchingInverter(vdc, Ts, inv.get("substeps", 1))
    raise ValueError(f"unknown inverter type {kind!r} (average | switching)")
