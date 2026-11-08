"""Field-oriented-control view of a 3-phase (abc-frame) simulation."""
from __future__ import annotations

import numpy as np

from motorsim.core.plot.common import C, plt, save_figure

INK = "#555550"


def _window(t, d, idx, window):
    """Explicit window, else 10 ms starting just before the largest torque transient."""
    if window is not None:
        return (t >= window[0]) & (t <= window[1])
    k = int(np.argmax(np.abs(np.diff(d["Te"][idx]))))
    t0 = max(t[0], t[k] - 0.001)
    return (t >= t0) & (t <= t0 + 0.01)


def plot_foc(res, path, idx: int = 0):
    """Phase currents -> Clarke/Park dq currents -> dq voltages -> angle, over the whole run."""
    d, t = res.data, res.t
    fig, axs = plt.subplots(5, 1, figsize=(9, 10), sharex=True)
    for j, ph in enumerate("abc"):
        axs[0].plot(t, d[f"i{ph}"][idx], color=C[j], lw=0.8, label=f"i{ph}")
    axs[0].set_ylabel("phase current [A]")
    axs[0].legend(loc="upper right", ncol=3)

    axs[1].plot(t, d["iq"][idx], color=C[0], label="iq (true)")
    axs[1].plot(t, d["id"][idx], color=C[1], label="id (true)")
    axs[1].plot(t, d["i_ref"][idx], color=INK, lw=0.8, ls="--", label="iq ref")
    axs[1].set_ylabel("dq current [A]")
    axs[1].legend(loc="upper right", ncol=3)

    axs[2].plot(t, d["vq"][idx], color=C[0], label="vq cmd")
    axs[2].plot(t, d["vd"][idx], color=C[1], label="vd cmd")
    axs[2].set_ylabel("dq voltage [V]")
    axs[2].legend(loc="upper right", ncol=2)

    axs[3].plot(t, d["w"][idx], color=C[0], label="w")
    if res.config.get("control", {}).get("mode") == "speed":
        axs[3].plot(t, d["ref"][idx], color=INK, lw=0.8, ls="--", label="reference")
    axs[3].set_ylabel("speed [rad/s]")
    axs[3].legend(loc="upper right", ncol=2)

    axs[4].plot(t, d["theta_e"][idx], color=C[0], lw=0.8, label="theta_e (true)")
    axs[4].plot(t, d["theta_e_meas"][idx], color=C[1], lw=0.8, ls="--", label="theta_e (controller)")
    axs[4].set_ylabel("elec. angle [rad]")
    axs[4].legend(loc="upper right", ncol=2)
    axs[-1].set_xlabel("time [s]")
    m = res.config.get("motor", {})
    axs[0].set_title(f"{res.config.get('name', '')} | PMSM abc | inverter={m.get('inverter', {}).get('type')} "
                     f"| mode={res.config.get('control', {}).get('mode')}")
    return save_figure(fig, path)


def plot_foc_zoom(res, path, idx: int = 0, window=None):
    """Short window. With a fast log (sim.fast_log, switching inverter) the currents are shown between
    samples (PWM ripple) together with the leg switch states; otherwise per-sample values."""
    d, t = res.data, res.t
    fast = res.fast if (res.fast is not None and idx == 0) else None
    if fast is not None and window is None:
        window = (fast["t"].iloc[0], fast["t"].iloc[-1])
    sel = _window(t, d, idx, window)
    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(3, 3)
    ax_i = fig.add_subplot(gs[0, :2])
    ax_v = fig.add_subplot(gs[1, :2], sharex=ax_i)
    ax_d = fig.add_subplot(gs[2, :2], sharex=ax_i)
    ax_ab = fig.add_subplot(gs[:, 2])
    for j, ph in enumerate("abc"):
        if fast is not None:
            ax_i.plot(fast["t"], fast[f"i{ph}"], color=C[j], lw=0.8, label=f"i{ph} (continuous)")
            smp = fast[fast["sample"] > 0]
            ax_i.plot(smp["t"], smp[f"i{ph}"], "o", ms=2.5, color=C[j])
        else:
            ax_i.plot(t[sel], d[f"i{ph}"][idx, sel], color=C[j], lw=0.9, label=f"i{ph}")
        ax_v.step(t[sel], d[f"v{ph}"][idx, sel], where="post", color=C[j], lw=0.8, label=f"v{ph}n (period avg)")
    if fast is not None and "Sa" in fast:
        for j, ph in enumerate("abc"):
            ax_d.step(fast["t"], fast[f"S{ph}"] * 0.8 + 2 - j, where="post", color=C[j], lw=0.8,
                      label=f"S{ph} (high side)")
        ax_d.set_yticks([0.4, 1.4, 2.4], ["Sc", "Sb", "Sa"])
        ax_d.set_ylabel("switch state")
        ax_i.set_title("zoom: currents between samples (dots = ADC samples at period start)")
    else:
        for j, ph in enumerate("abc"):
            ax_d.step(t[sel], d[f"d{ph}"][idx, sel], where="post", color=C[j], lw=0.8, label=f"d{ph}")
        ax_d.set_ylabel("duty [-]")
        ax_i.set_title("zoom: phase quantities (one value per control period)")
    ax_i.set_ylabel("phase current [A]")
    ax_v.set_ylabel("phase voltage [V]")
    ax_d.set_xlabel("time [s]")
    for ax in (ax_i, ax_v, ax_d):
        ax.legend(loc="upper right", ncol=3, fontsize=8)
    src = fast if fast is not None else {ph: d[ph][idx, sel] for ph in ("ia", "ib", "ic")}
    ia, ib, ic = (np.asarray(src[ph]) for ph in ("ia", "ib", "ic"))
    ax_ab.plot((2 * ia - ib - ic) / 3, (ib - ic) / np.sqrt(3), color=C[0], lw=0.8)
    ax_ab.set_aspect("equal", "datalim")
    ax_ab.set_xlabel("i_alpha [A]")
    ax_ab.set_ylabel("i_beta [A]")
    ax_ab.set_title("current vector (Clarke)")
    return save_figure(fig, path)
