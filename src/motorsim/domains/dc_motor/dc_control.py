"""DC motor cascade control (batched).

Modes
    voltage   : reference = terminal voltage
    current   : reference = armature current
    torque    : reference = torque -> current via the *nominal* Kt
    speed     : speed PI -> current PI
    position  : position P -> speed PI -> current PI
Default gains: current kp = L wc, ki = R wc; speed kp = J ws / Kt, ki = kp ws / 4; position kp = wp.
"""
from __future__ import annotations

import numpy as np

from motorsim.core.control.pi import PI
from motorsim.domains.dc_motor.dc_schema import MODES

class DCController:
    def __init__(self, cfg: dict, nom: dict, N: int):
        cc = cfg["control"]
        self.mode = cc["mode"]
        if self.mode not in MODES:
            raise ValueError(f"DC motor: unknown control mode {self.mode!r}; choose from {MODES}")
        Ts = cc["Ts"]
        lim = cfg["motor"]["limits"]
        self.vmax, self.imax, self.wmax = lim["v_max"], lim["i_max"], cc["speed_max"]
        self.backemf_ff = cc.get("backemf_ff", True)
        self.nom = nom
        bw = {k: 2 * np.pi * v for k, v in cc["bandwidth_hz"].items()}
        g = cc.get("gains") or {}
        self.kt = nom["Kt"]
        gc = g.get("current", {})
        self.pi_i = PI(gc.get("kp", nom["L"] * bw["current"]), gc.get("ki", nom["R"] * bw["current"]), Ts, self.vmax, N)
        gs = g.get("speed", {})
        kps = gs.get("kp", nom["J"] * bw["speed"] / self.kt)
        self.pi_w = PI(kps, gs.get("ki", kps * bw["speed"] / 4.0), Ts, self.imax, N)
        self.kp_pos = g.get("position", {}).get("kp", bw["position"])
        z = np.zeros(N)
        self.i_ref, self.w_ref, self.v = z.copy(), z.copy(), z.copy()

    def _current_ref(self, ref, m):
        if self.mode in ("speed", "position"):
            w_ref = ref if self.mode == "speed" else np.clip(self.kp_pos * (ref - m["th"]), -self.wmax, self.wmax)
            self.w_ref = w_ref
            return self.pi_w.step(w_ref - m["w"])
        if self.mode == "torque":
            return np.clip(ref / self.kt, -self.imax, self.imax)
        return np.clip(ref, -self.imax, self.imax)

    def step(self, r: dict, m: dict):
        ref = r["ref"]
        if self.mode == "voltage":
            self.v = np.clip(ref, -self.vmax, self.vmax)
        else:
            self.i_ref = self._current_ref(ref, m)
            ff = self.nom["Ke"] * m["w"] if self.backemf_ff else 0.0
            self.v = self.pi_i.step(self.i_ref - m["i"][0], ff)
        return self.v[None]

    def log_signals(self) -> dict:
        return {"v": self.v, "i_ref": self.i_ref, "w_ref": self.w_ref}