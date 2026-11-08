"""Friction map with fitted models."""
from __future__ import annotations

import numpy as np

from motorsim.core.plot.common import C, plt, save_figure
from motorsim.core.model.friction import static_friction

def plot_friction_fits(w, T, fits: dict, path, title="friction map"):
    """Left: signed map. Right: |T| vs |w| on a log speed axis, each direction separately."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    wf = np.linspace(w.min(), w.max(), 1600)
    wl = np.geomspace(max(np.abs(w).min() * 0.7, 1e-4), np.abs(w).max() * 1.1, 400)

    def curve(name, f, x):
        p = dict(f.params)
        fm = "stribeck" if name == "lugre" else name
        return (static_friction(fm, x, p) if name != "none" else 0 * x) + p.get("T0", 0.0)

    axs[0].plot(w, T, "o", ms=4, color="#555550", label="measured", zorder=3)
    pos, neg = w > 0, w < 0
    axs[1].plot(w[pos], T[pos], "o", ms=5, color="#555550", label="measured, w > 0", zorder=3)
    axs[1].plot(-w[neg], -T[neg], "o", ms=5, mfc="white", color="#555550", label="measured, w < 0 (mirrored)", zorder=3)
    for j, (name, f) in enumerate(fits.items()):
        axs[0].plot(wf, curve(name, f, wf), color=C[j % len(C)], label=f"{name} (BIC {f.bic:.0f})")
        axs[1].plot(wl, curve(name, f, wl), color=C[j % len(C)])
    axs[0].set_xlabel("speed [rad/s]")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("|speed| [rad/s] (log)")
    for ax in axs:
        ax.set_ylabel("friction torque [Nm]")
    axs[0].legend(loc="upper left", fontsize=8)
    axs[1].legend(loc="upper left", fontsize=8)
    axs[0].set_title(title)
    return save_figure(fig, path)
