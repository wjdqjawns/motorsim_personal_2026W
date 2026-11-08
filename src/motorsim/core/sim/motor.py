"""A stand-alone motor you drive with voltages, one step at a time.

    from motorsim import Motor
    m = Motor("config/pmsm_abc.yaml")          # or Motor({"motor": {"type": "pmsm", "frame": "abc"}})
    print(m.inputs)                             # ('va', 'vb', 'vc')
    for k in range(20000):
        t = k * 1e-5
        out = m.step(va, vb, vc, dt=1e-5)       # phase-to-neutral voltages [V]
        out["ia"], out["w"], out["theta"] ...

Inputs per model:  DC ('v',)   PMSM dq ('vd', 'vq')   PMSM abc ('va', 'vb', 'vc')
No controller, no sensors: this is the bare plant (+ friction + load). Use `run()` to
feed whole signal arrays or a function u(t, state) -> inputs (e.g. your own controller).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from motorsim.core.config import flat_params, load_config
from motorsim.core.model.friction import Friction
from motorsim.core.model.integrators import rk4
from motorsim.core.sim.simulator import as_batch

class Motor:
    def __init__(self, cfg=None, params: dict | None = None, init: dict | None = None):
        from motorsim.domains import get_domain

        self.cfg = load_config(cfg)
        P = flat_params(self.cfg)
        if params:
            P.update(params)
        self.P = as_batch(P, 1)
        self.plant = get_domain(self.cfg["motor"]["type"]).make_plant(self.P, self.cfg)
        self.fric = Friction(self.cfg["friction"]["model"], self.P, 1, self.cfg["friction"].get("v_eps", 1e-3))
        self.inputs = self.plant.input_names
        self.x = self.plant.initial_state(1, init)
        self.t = 0.0
        self._te = self._tf = 0.0

    def step(self, *u, dt: float, load_torque: float = 0.0, substeps: int = 1) -> dict:
        """Apply the input voltages (held constant) for dt seconds and return the new state."""
        if len(u) != len(self.inputs):
            raise ValueError(f"expected inputs {self.inputs}, got {len(u)} values")
        up = self.plant.plant_input(np.asarray(u, float).reshape(len(u), 1))
        h = dt / substeps
        P, iw, ith = self.P, self.plant.iw, self.plant.ith
        for _ in range(substeps):
            x = self.x
            tl = load_torque + P["TL"] + P["cog_A"] * np.sin(P["cog_N"] * x[ith] + P["cog_phi"])
            te = self.plant.torque(x)
            w0 = x[iw]
            tf = self.fric.torque(w0, te - tl, h)
            xn = rk4(self.plant.deriv, x, up, tf, tl, h)
            if self.fric.karnopp:
                xn[iw] = self.fric.stick(w0, xn[iw], te - tl)
            self.x = xn
            self._te, self._tf = float(te[0]), float(tf[0])
        self.t += dt
        return self.state()

    def state(self) -> dict:
        out = {"t": self.t}
        out.update({k: float(np.asarray(v)[0]) for k, v in self.plant.log_signals(self.x).items()})
        out.update(w=float(self.x[self.plant.iw][0]), theta=float(self.x[self.plant.ith][0]),
                   Te=self._te, Tf=self._tf)
        return out

    def run(self, u, dt: float, n: int | None = None, load_torque: float = 0.0, substeps: int = 1) -> pd.DataFrame:
        """u: array (n_inputs, T)  or  callable u(t, state) -> sequence of inputs."""
        rows = [self.state()]
        if callable(u):
            if n is None:
                raise ValueError("n (number of steps) is required with a callable input")
            for _ in range(n):
                v = u(self.t, rows[-1])
                rows.append(self.step(*v, dt=dt, load_torque=load_torque, substeps=substeps))
        else:
            u = np.atleast_2d(np.asarray(u, float))
            for k in range(u.shape[1]):
                rows.append(self.step(*u[:, k], dt=dt, load_torque=load_torque, substeps=substeps))
        return pd.DataFrame(rows)