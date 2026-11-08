"""Overview of one simulated experiment."""
from __future__ import annotations

import numpy as np

from motorsim.core.plot.common import C, plt, save_figure


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
    return save_figure(fig, path)
