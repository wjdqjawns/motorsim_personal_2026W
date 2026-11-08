"""Measured vs model overlay."""
from __future__ import annotations

import numpy as np

from motorsim.core.plot.common import C, plt, save_figure


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
    return save_figure(fig, path)
