"""Drive a bare 3-phase PMSM with your own signals, step by step (no built-in controller).

    py -3.13 examples/drive_motor.py

1) open loop: balanced 3-phase voltages with a frequency ramp -> the rotor follows (synchronous)
2) your own FOC loop written here with the library transforms (Clarke / Park / inverse Park)
Writes export/examples/drive_motor.png
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motorsim import Motor  # noqa: E402
from motorsim.core.control.transforms import clarke, inv_clarke, inv_park, park  # noqa: E402
from motorsim.core.plot.common import C, plt, save_figure  # noqa: E402

CFG = {"motor": {"type": "pmsm", "frame": "abc",
                 "params": {"R": 0.5, "Ld": 1.0e-3, "Lq": 1.0e-3, "psi": 0.01, "p": 4, "J": 3.0e-5}},
       "friction": {"model": "viscous", "params": {"B": 2.0e-5}}}
dt = 2e-5

# 1) open loop ---------------------------------------------------------------------------
m = Motor(CFG)
print("inputs:", m.inputs)
th = 0.0
rows = []
for k in range(int(0.5 / dt)):
    t = k * dt
    f = min(40.0, 100.0 * t)                  # electrical frequency ramp, 100 Hz/s up to 40 Hz
    V = 0.5 + 0.3 * f                         # V/f with boost
    th += 2 * np.pi * f * dt
    va, vb, vc = (V * np.cos(th - s) for s in (0.0, 2 * np.pi / 3, -2 * np.pi / 3))
    rows.append(m.step(va, vb, vc, dt=dt))
ol = {k: np.array([r[k] for r in rows]) for k in rows[0]}
print(f"open loop: final speed {ol['w'][-1]:.2f} rad/s (synchronous = {2 * np.pi * 40 / 4:.2f})")

# 2) your own FOC (current loop only, iq* = 2 A) -------------------------------------------
m = Motor(CFG)
p, R, L, psi, vdc = 4, 0.5, 1.0e-3, 0.01, 24.0
wc = 2 * np.pi * 500
kp, ki = L * wc, R * wc
xi_d = xi_q = 0.0
Ts = 1e-4                                       # controller period; plant stepped with dt inside
st = m.state()
rows = []
for k in range(int(0.1 / Ts)):
    th_e = p * st["theta"]
    i_d, i_q = park(*clarke(st["ia"], st["ib"], st["ic"]), th_e)
    e_d, e_q = 0.0 - i_d, 2.0 - i_q
    xi_d += ki * Ts * e_d
    xi_q += ki * Ts * e_q
    we = p * st["w"]
    vd = kp * e_d + xi_d - we * L * i_q
    vq = kp * e_q + xi_q + we * (L * i_d + psi)
    va, vb, vc = inv_clarke(*inv_park(vd, vq, th_e + 0.5 * we * Ts))
    for _ in range(int(Ts / dt)):
        st = m.step(va, vb, vc, dt=dt)
    rows.append(dict(st, id=i_d, iq=i_q))
cl = {k: np.array([r[k] for r in rows]) for k in rows[0]}
print(f"own FOC: iq = {cl['iq'][-1]:.3f} A, id = {cl['id'][-1]:.3f} A, speed {cl['w'][-1]:.1f} rad/s")

fig, axs = plt.subplots(3, 2, figsize=(11, 7))
axs[0, 0].plot(ol["t"], ol["ia"], color=C[0], lw=0.7, label="ia")
axs[0, 0].plot(ol["t"], ol["ib"], color=C[1], lw=0.7, label="ib")
axs[0, 0].plot(ol["t"], ol["ic"], color=C[2], lw=0.7, label="ic")
axs[0, 0].set_title("1) open loop V/f: phase currents")
axs[1, 0].plot(ol["t"], ol["w"], color=C[0], label="rotor speed")
axs[1, 0].plot(ol["t"], np.minimum(40, 100 * ol["t"]) * 2 * np.pi / 4, color="#555550", ls="--", lw=0.8,
               label="synchronous speed")
axs[2, 0].plot(ol["t"], ol["Te"], color=C[0], label="Te")
axs[0, 1].plot(cl["t"], cl["ia"], color=C[0], lw=0.7, label="ia")
axs[0, 1].plot(cl["t"], cl["ib"], color=C[1], lw=0.7, label="ib")
axs[0, 1].plot(cl["t"], cl["ic"], color=C[2], lw=0.7, label="ic")
axs[0, 1].set_title("2) own FOC loop: phase currents")
axs[1, 1].plot(cl["t"], cl["iq"], color=C[0], label="iq (Park)")
axs[1, 1].plot(cl["t"], cl["id"], color=C[1], label="id (Park)")
axs[2, 1].plot(cl["t"], cl["w"], color=C[0], label="rotor speed")
for ax, yl in zip(axs.ravel(), ["current [A]", "current [A]", "speed [rad/s]", "dq current [A]",
                                "torque [Nm]", "speed [rad/s]"]):
    ax.set_ylabel(yl)
    ax.legend(loc="upper left", fontsize=8)
for ax in axs[-1]:
    ax.set_xlabel("time [s]")
out = save_figure(fig, ROOT / "export" / "examples" / "drive_motor.png")
print(f"saved -> {out}")
