"""PMSM control (batched).

PMSMController  (frame: dq)   controller output = (vd, vq) applied directly to the dq plant.
FOCController   (frame: abc)  full field-oriented control chain of a real drive:

    ia, ib, ic --Clarke--> i_alpha,beta --Park(theta_e_meas)--> id, iq
        --> PI (+ decoupling, voltage circle, anti-windup) --> vd, vq
        --invPark(theta_e_meas + 0.5 we Ts)--> v_alpha,beta --SVPWM--> duty a, b, c --> inverter

    theta_e_meas = p * theta_encoder + angle_offset   (misalignment can be injected)

Modes: voltage | current | torque | speed | position, and for abc also
    vf   : open-loop V/f, reference = electrical frequency [Hz]  (no sensors used)
    abc  : direct phase voltages from reference_abc (sinusoidal PWM)
"""
from __future__ import annotations

import numpy as np

from motorsim.core.control.pi import PI
from motorsim.core.control.transforms import clarke, inv_park, park, spwm, svpwm, wrap_angle
from motorsim.domains.pmsm.pmsm_schema import MODES, MODES_ABC, torque_constant

class PMSMController:
    modes = MODES

    def __init__(self, cfg: dict, nom: dict, N: int):
        cc = cfg["control"]
        self.mode = cc["mode"]
        if self.mode not in self.modes:
            raise ValueError(f"PMSM ({cfg['motor'].get('frame', 'dq')}): unknown control mode {self.mode!r}; "
                             f"choose from {self.modes}")
        self.Ts = Ts = cc["Ts"]
        lim = cfg["motor"]["limits"]
        self.vdc = lim["v_dc"]
        self.vmax = self.vdc / np.sqrt(3.0)
        self.imax, self.wmax = lim["i_max"], cc["speed_max"]
        self.decoupling = cc.get("decoupling", True)
        self.nom = nom
        bw = {k: 2 * np.pi * v for k, v in cc["bandwidth_hz"].items()}
        g = cc.get("gains") or {}
        self.kt = torque_constant(nom)
        gc = g.get("current", {})
        self.pi_d = PI(gc.get("kp", nom["Ld"] * bw["current"]), gc.get("ki", nom["R"] * bw["current"]), Ts, self.vmax, N)
        self.pi_q = PI(gc.get("kp", nom["Lq"] * bw["current"]), gc.get("ki", nom["R"] * bw["current"]), Ts, self.vmax, N)
        gs = g.get("speed", {})
        kps = gs.get("kp", nom["J"] * bw["speed"] / self.kt)
        self.pi_w = PI(kps, gs.get("ki", kps * bw["speed"] / 4.0), Ts, self.imax, N)
        self.kp_pos = g.get("position", {}).get("kp", bw["position"])
        z = np.zeros(N)
        self.i_ref, self.w_ref, self.vd, self.vq = z.copy(), z.copy(), z.copy(), z.copy()

    def _current_ref(self, ref, m):
        if self.mode in ("speed", "position"):
            w_ref = ref if self.mode == "speed" else np.clip(self.kp_pos * (ref - m["th"]), -self.wmax, self.wmax)
            self.w_ref = w_ref
            return self.pi_w.step(w_ref - m["w"])
        if self.mode == "torque":
            return np.clip(ref / self.kt, -self.imax, self.imax)
        return np.clip(ref, -self.imax, self.imax)

    def dq_voltage(self, r: dict, id_, iq, m: dict):
        """Current loops (or voltage reference) -> limited (vd, vq)."""
        if self.mode == "voltage":
            vd, vq = r["ref_d"], r["ref"]
        else:
            n = self.nom
            we = n["p"] * m["w"]
            iq_ref = self._current_ref(r["ref"], m)
            id_ref = np.clip(r["ref_d"], -self.imax, self.imax)
            self.i_ref = iq_ref
            ffd = -we * n["Lq"] * iq if self.decoupling else 0.0
            ffq = we * (n["Ld"] * id_ + n["psi"]) if self.decoupling else 0.0
            vd = self.pi_d.step(id_ref - id_, ffd)
            vq = self.pi_q.step(iq_ref - iq, ffq)
        scale = np.minimum(1.0, self.vmax / np.maximum(np.hypot(vd, vq), 1e-12))
        vd2, vq2 = vd * scale, vq * scale
        if self.mode != "voltage":
            self.pi_d.windup_correct(vd2 - vd)
            self.pi_q.windup_correct(vq2 - vq)
        return vd2, vq2

    def step(self, r: dict, m: dict):
        id_, iq = m["i"]
        self.vd, self.vq = self.dq_voltage(r, id_, iq, m)
        return np.array([self.vd, self.vq])

    def log_signals(self) -> dict:
        return {"vd": self.vd, "vq": self.vq, "i_ref": self.i_ref, "w_ref": self.w_ref}


class FOCController(PMSMController):
    modes = MODES_ABC

    def __init__(self, cfg: dict, nom: dict, N: int):
        super().__init__(cfg, nom, N)
        cc = cfg["control"]
        self.p = nom["p"]
        self.offset = np.deg2rad(cc.get("angle_offset_deg", 0.0))
        self.delay_comp = cc.get("delay_comp", True)
        self.two_sensors = int(cc.get("current_sensors", 3)) == 2
        vf = cc.get("vf") or {}
        self.v_per_hz, self.boost = vf.get("v_per_hz", 0.05), vf.get("boost", 0.5)
        z = np.zeros(N)
        self.th_vf, self.id_m, self.iq_m, self.th_e = z.copy(), z.copy(), z.copy(), z.copy()
        self.duty = np.full((3, N), 0.5)

    def step(self, r: dict, m: dict):
        ia, ib, ic = m["i"]
        if self.two_sensors:
            ic = -ia - ib
        th_e = self.p * m["th"] + self.offset
        id_, iq = park(*clarke(ia, ib, ic), th_e)
        we = self.p * m["w"]

        if self.mode == "vf":
            f = r["ref"]
            V = np.minimum(self.v_per_hz * np.abs(f) + self.boost, self.vmax)
            th_v = self.th_vf + (np.pi * f * self.Ts if self.delay_comp else 0.0)
            self.th_vf = np.mod(self.th_vf + 2 * np.pi * f * self.Ts, 2 * np.pi)
            va_, vb_ = V * np.cos(th_v), V * np.sin(th_v)
            duty = svpwm(va_, vb_, self.vdc)
            self.vd, self.vq = park(va_, vb_, th_e)
        elif self.mode == "abc":
            duty = spwm(r["ref_a"], r["ref_b"], r["ref_c"], self.vdc)
            self.vd, self.vq = park(*clarke(r["ref_a"], r["ref_b"], r["ref_c"]), th_e)
        else:
            self.vd, self.vq = self.dq_voltage(r, id_, iq, m)
            th_v = th_e + (0.5 * we * self.Ts if self.delay_comp else 0.0)
            duty = svpwm(*inv_park(self.vd, self.vq, th_v), self.vdc)

        self.id_m, self.iq_m, self.th_e, self.duty = id_, iq, wrap_angle(th_e), duty
        return duty

    def log_signals(self) -> dict:
        out = super().log_signals()
        out.update(id_meas=self.id_m, iq_meas=self.iq_m, theta_e_meas=self.th_e, da=self.duty[0], db=self.duty[1], dc=self.duty[2])
        
        return out