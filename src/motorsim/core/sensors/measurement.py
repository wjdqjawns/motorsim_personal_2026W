"""Measurement model: noisy current sensors, quantized encoder, speed estimation."""
from __future__ import annotations

import numpy as np


class Sensors:
    def __init__(self, cfg: dict, Ts: float, N: int, rng: np.random.Generator):
        self.noise = float(cfg.get("current_noise", 0.0))
        self.offset = float(cfg.get("current_offset", 0.0))
        cpr = int(cfg.get("encoder_cpr", 0))
        self.res = 2 * np.pi / cpr if cpr > 0 else 0.0
        fc = float(cfg.get("speed_lpf_hz", 0.0))
        self.alpha = 1.0 - np.exp(-2 * np.pi * fc * Ts) if fc > 0 else 1.0
        self.source = cfg.get("speed_source", "encoder")
        self.Ts, self.N, self.rng = Ts, N, rng

    def _quant(self, th):
        return np.floor(th / self.res) * self.res if self.res else th.copy()

    def reset(self, w, th):
        self.th_prev = self._quant(th)
        self.wf = np.array(w, float).copy()

    def measure(self, cur, w, th):
        """cur: (nc, N) currents in the sensor frame (i | id,iq | ia,ib,ic); w, th: true speed/angle."""
        if self.noise > 0:
            cur = cur + self.noise * self.rng.standard_normal(cur.shape)
        if self.offset:
            cur = cur + self.offset
        thq = self._quant(th)
        if self.source == "true":
            wm = w.copy()
        else:
            raw = (thq - self.th_prev) / self.Ts
            self.th_prev = thq
            self.wf = self.wf + self.alpha * (raw - self.wf)
            wm = self.wf
        return {"i": cur, "w": wm, "th": thq}
