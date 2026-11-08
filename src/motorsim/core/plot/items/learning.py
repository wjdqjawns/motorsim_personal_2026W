"""Learned-estimator accuracy (predicted vs true)."""
from __future__ import annotations

import numpy as np

from motorsim.core.plot.common import C, plt, save_figure


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
    return save_figure(fig, path)
