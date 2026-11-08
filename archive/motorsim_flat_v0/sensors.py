"""Measurement model: noisy current sensors, quantized encoder, speed estimation."""
from __future__ import annotations

import numpy as np

class Sensors:
    def __init__(self, cfg: dict, Ts: float, N: int, nc: int, rng: np.random.Generator):
        self.noise = float(cfg.get("current_noise", 0.0))
        self.offset = float(cfg.get("current_offset", 0.0))
        cpr = int(cfg.get("encoder_cpr", 0))
        self.res = 2 * np.pi / cpr if cpr > 0 else 0.0
        fc = float(cfg.get("speed_lpf_hz", 0.0))
        self.alpha = 1.0 - np.exp(-2 * np.pi * fc * Ts) if fc > 0 else 1.0
        self.source = cfg.get("speed_source", "encoder")
        self.Ts, self.N, self.nc, self.rng = Ts, N, nc, rng

    def _quant(self, th):
        return np.floor(th / self.res) * self.res if self.res else th.copy()

    def reset(self, x):
        self.th_prev = self._quant(x[-1])
        self.wf = x[-2].copy()

    def measure(self, x):
        cur = x[: self.nc]
        if self.noise > 0:
            cur = cur + self.noise * self.rng.standard_normal(cur.shape)
        if self.offset:
            cur = cur + self.offset
        th = self._quant(x[-1])
        if self.source == "true":
            w = x[-2].copy()
        else:
            raw = (th - self.th_prev) / self.Ts
            self.th_prev = th
            self.wf = self.wf + self.alpha * (raw - self.wf)
            w = self.wf
        return {"i": cur, "w": w, "th": th}
