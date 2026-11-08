"""PMSM plants (amplitude-invariant transforms, mechanical angle theta, we = p w).

dq model (rotor frame)                       x = [id, iq, w, theta]
    Ld did/dt = vd - R id + we Lq iq
    Lq diq/dt = vq - R iq - we (Ld id + psi)

abc model (stationary frame, star connection, floating neutral)
                                             x = [psi_alpha, psi_beta, w, theta]
    d psi_ab / dt = v_ab - R i_ab            v_ab = Clarke(va, vb, vc)
    currents from flux through the rotor-position-dependent inductance:
        [psi_d, psi_q] = Park(psi_ab, p theta),  id = (psi_d - psi)/Ld,  iq = psi_q / Lq
        i_ab = invPark(id, iq),  (ia, ib, ic) = invClarke(i_ab)
    This model never uses the controller's transforms or angle: it is the "real" motor.

Both:  Te = 1.5 p (psi iq + (Ld - Lq) id iq) = 1.5 p (psi_a i_b - psi_b i_a),  J dw/dt = Te - T_f - T_L
"""
from __future__ import annotations

import numpy as np

from motorsim.core.control.transforms import clarke, inv_clarke, inv_park, park, wrap_angle


class PMSMdq:
    nc = 2
    cur_names = ("id", "iq")
    input_names = ("vd", "vq")
    iw, ith = 2, 3

    def __init__(self, P: dict):
        self.R, self.Ld, self.Lq, self.psi, self.p, self.J = (P[k] for k in ("R", "Ld", "Lq", "psi", "p", "J"))
        self.k = 1.5 * self.p

    def initial_state(self, N, init=None):
        x = np.zeros((4, N))
        if init:
            if "i" in init:
                x[:2] = np.asarray(init["i"], float).reshape(2, -1)
            x[2] = init.get("w", 0.0)
            x[3] = init.get("th", 0.0)
        return x

    def plant_input(self, u):
        return u

    def currents(self, x):
        return x[:2]

    def log_signals(self, x) -> dict:
        return {"id": x[0], "iq": x[1]}

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


class PMSMabc:
    nc = 3
    cur_names = ("ia", "ib", "ic")
    input_names = ("va", "vb", "vc")
    iw, ith = 2, 3

    def __init__(self, P: dict):
        self.R, self.Ld, self.Lq, self.psi, self.p, self.J = (P[k] for k in ("R", "Ld", "Lq", "psi", "p", "J"))
        self.k = 1.5 * self.p

    def _idq(self, x):
        th_e = self.p * x[3]
        psid, psiq = park(x[0], x[1], th_e)
        return (psid - self.psi) / self.Ld, psiq / self.Lq, th_e

    def _iab(self, x):
        id_, iq, th_e = self._idq(x)
        return inv_park(id_, iq, th_e)

    def initial_state(self, N, init=None):
        """init: {'w', 'th', and 'i' = (ia, ib, ic) or 'idq' = (id, iq)}; default zero current."""
        init = init or {}
        x = np.zeros((4, N))
        x[2] = init.get("w", 0.0)
        x[3] = init.get("th", 0.0)
        th_e = self.p * x[3]
        if "idq" in init:
            id_, iq = (np.asarray(v, float) for v in init["idq"])
        elif "i" in init and len(init["i"]) == 3:
            id_, iq = park(*clarke(*(np.asarray(v, float) for v in init["i"])), th_e)
        else:
            id_, iq = 0.0, 0.0
        x[0], x[1] = inv_park(self.Ld * id_ + self.psi, self.Lq * iq, th_e)
        return x

    def plant_input(self, u):
        """(va, vb, vc) phase-to-neutral (or leg) voltages -> v_alpha_beta; common mode drops out."""
        return np.array(clarke(u[0], u[1], u[2]))

    def currents(self, x):
        return np.array(inv_clarke(*self._iab(x)))

    def log_signals(self, x) -> dict:
        id_, iq, th_e = self._idq(x)
        ia, ib, ic = inv_clarke(*inv_park(id_, iq, th_e))
        return {"ia": ia, "ib": ib, "ic": ic, "id": id_, "iq": iq, "theta_e": wrap_angle(th_e)}

    def torque(self, x):
        ia, ib = self._iab(x)
        return self.k * (x[0] * ib - x[1] * ia)

    def deriv(self, x, u, tf, tl):
        ia, ib = self._iab(x)
        te = self.k * (x[0] * ib - x[1] * ia)
        return np.array([
            u[0] - self.R * ia,
            u[1] - self.R * ib,
            (te - tf - tl) / self.J,
            x[2],
        ])
