"""Matplotlib figures (saved to files). One measure per axis; shared time axis."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from motorsim.models.friction import static_friction  # noqa: E402

# categorical slots in fixed order (validated palette)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GRID = dict(color="#d9d8d3", linewidth=0.6)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID["color"], "grid.linewidth": GRID["linewidth"], "lines.linewidth": 1.2, "font.size": 9, "legend.frameon": False})

def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path

def plot_sim(res, path, idx: int = 0):
    d, t = res.data, res.t
    pmsm = "iq" in d
    mode = res.config.get("control", {}).get("mode", "")
    rows = [("speed [rad/s]", [("w", "w"), ("w_meas", "w measured")] + ([("ref", "reference")] if mode == "speed" else [])),
            ("current [A]", [("iq", "iq"), ("id", "id")] if pmsm else [("i", "i"), ("i_ref", "i ref")]),
            ("voltage [V]", [("vq", "vq"), ("vd", "vd")] if pmsm else [("v", "v")]),
            ("torque [Nm]", [("Te", "electromagnetic"), ("Tf", "friction")]),
            ("position [rad]", [("theta", "theta")] + ([("ref", "reference")] if mode == "position" else []))]
    fig, axs = plt.subplots(len(rows), 1, figsize=(9, 9), sharex=True)

    for ax, (ylabel, series) in zip(axs, rows):
        for j, (k, lab) in enumerate(series):
            if k in d:
                ax.plot(t, d[k][idx], color=C[j], label=lab, lw=1.0 if j == 0 else 0.9)
        ax.set_ylabel(ylabel)
        if len(series) > 1:
            ax.legend(loc="upper right", ncol=len(series))

    axs[-1].set_xlabel("time [s]")
    axs[0].set_title(f"{res.config.get('name', '')} | {res.config.get('motor', {}).get('type', '')} | mode={mode}")
    return _save(fig, path)

def plot_overlay(t, pairs: dict, path, title="", labels=("measured", "model")):
    fig, axs = plt.subplots(len(pairs), 1, figsize=(9, 2.3 * len(pairs)), sharex=True, squeeze=False)
    units = {"i": "current [A]", "id": "id [A]", "iq": "iq [A]", "w": "speed [rad/s]"}

    for ax, (k, (y, yh)) in zip(axs[:, 0], pairs.items()):
        ax.plot(t, y, color=C[1], label=labels[0], lw=1.0)
        ax.plot(t, yh, color=C[0], label=labels[1], lw=1.0)
        ax.set_ylabel(units.get(k, k))
        ax.legend(loc="upper right", ncol=2)
    
    axs[-1, 0].set_xlabel("time [s]")
    axs[0, 0].set_title(title)
    return _save(fig, path)


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
    return _save(fig, path)


def plot_learning(y_true, y_pred, names, path):
    n = len(names)
    fig, axs = plt.subplots(1, n, figsize=(2.6 * n, 2.8), squeeze=False)
    for j, ax in enumerate(axs[0]):
        ax.loglog(y_true[:, j], y_pred[:, j], ".", ms=3, color=C[0])
        lo, hi = y_true[:, j].min(), y_true[:, j].max()
        ax.plot([lo, hi], [lo, hi], color="#888780", lw=0.8)
        ax.set_title(names[j])
        ax.set_xlabel("true")
    axs[0, 0].set_ylabel("predicted")
    return _save(fig, path)
