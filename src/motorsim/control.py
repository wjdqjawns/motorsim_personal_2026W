"""Discrete cascade controllers for DC motors and PMSM (FOC), batched over N.

Modes
    voltage   : reference is the terminal voltage (DC) / vq (PMSM, vd from reference_d)
    current   : reference is the (q-axis) current
    torque    : reference is torque  -> current via the *nominal* torque constant
    speed     : speed PI  -> current loop
    position  : position P -> speed PI -> current loop
Default gains come from the nominal model and the requested bandwidths:
    current PI  kp = L wc,  ki = R wc           (pole-zero cancellation)
    speed PI    kp = J ws / Kt, ki = kp ws / 4
    position P  kp = wp
"""
from __future__ import annotations

import numpy as np

from motorsim.config import torque_constant

MODES = ("voltage", "current", "torque", "speed", "position")

class PI:
    """PI with output clamp and back-calculation anti-windup."""

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

class Controller:
    def __init__(self, cfg: dict, nom: dict, N: int):
        cc = cfg["control"]
        self.mtype = cfg["motor"]["type"]
        self.mode = cc["mode"]
        if self.mode not in MODES:
            raise ValueError(f"unknown control mode {self.mode!r}; choose from {MODES}")
        self.Ts = Ts = cc["Ts"]
        lim = cfg["motor"]["limits"]
        self.vmax = lim["v_max"] if self.mtype == "dc" else lim["v_dc"] / np.sqrt(3.0)
        self.imax = lim["i_max"]
        self.wmax = cc["speed_max"]
        self.backemf_ff = cc.get("backemf_ff", True)
        self.decoupling = cc.get("decoupling", True)
        self.nom = nom
        bw = {k: 2 * np.pi * v for k, v in cc["bandwidth_hz"].items()}
        g = cc.get("gains") or {}
        self.kt = torque_constant(self.mtype, nom)

        gc = g.get("current", {})
        if self.mtype == "dc":
            self.pi_i = PI(gc.get("kp", nom["L"] * bw["current"]), gc.get("ki", nom["R"] * bw["current"]), Ts, self.vmax, N)
        else:
            self.pi_d = PI(gc.get("kp", nom["Ld"] * bw["current"]), gc.get("ki", nom["R"] * bw["current"]), Ts, self.vmax, N)
            self.pi_q = PI(gc.get("kp", nom["Lq"] * bw["current"]), gc.get("ki", nom["R"] * bw["current"]), Ts, self.vmax, N)
        gs = g.get("speed", {})
        kps = gs.get("kp", nom["J"] * bw["speed"] / self.kt)
        self.pi_w = PI(kps, gs.get("ki", kps * bw["speed"] / 4.0), Ts, self.imax, N)
        self.kp_pos = g.get("position", {}).get("kp", bw["position"])
        z = np.zeros(N)
        self.i_ref, self.w_ref = z.copy(), z.copy()

    def _current_ref(self, ref, m):
        mode = self.mode
        if mode in ("speed", "position"):
            w_ref = ref if mode == "speed" else np.clip(self.kp_pos * (ref - m["th"]), -self.wmax, self.wmax)
            self.w_ref = w_ref
            return self.pi_w.step(w_ref - m["w"])
        if mode == "torque":
            return np.clip(ref / self.kt, -self.imax, self.imax)
        return np.clip(ref, -self.imax, self.imax)          # current

    def step(self, ref, ref_d, m):
        """ref, ref_d: (N,) references; m: measurements {'i': (nc,N), 'w', 'th'}. Returns u (nc,N)."""
        if self.mtype == "dc":
            if self.mode == "voltage":
                return np.clip(ref, -self.vmax, self.vmax)[None]
            i_ref = self._current_ref(ref, m)
            self.i_ref = i_ref
            ff = self.nom["Ke"] * m["w"] if self.backemf_ff else 0.0
            return self.pi_i.step(i_ref - m["i"][0], ff)[None]

        # PMSM
        if self.mode == "voltage":
            vd, vq = ref_d, ref
        else:
            n = self.nom
            id_, iq = m["i"]
            we = n["p"] * m["w"]
            iq_ref = self._current_ref(ref, m)
            id_ref = np.clip(ref_d, -self.imax, self.imax)
            self.i_ref = iq_ref
            ffd = -we * n["Lq"] * iq if self.decoupling else 0.0
            ffq = we * (n["Ld"] * id_ + n["psi"]) if self.decoupling else 0.0
            vd = self.pi_d.step(id_ref - id_, ffd)
            vq = self.pi_q.step(iq_ref - iq, ffq)
        mag = np.hypot(vd, vq)
        scale = np.minimum(1.0, self.vmax / np.maximum(mag, 1e-12))
        vd2, vq2 = vd * scale, vq * scale
        if self.mode != "voltage":
            self.pi_d.windup_correct(vd2 - vd)
            self.pi_q.windup_correct(vq2 - vq)
        return np.array([vd2, vq2])
