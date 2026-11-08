"""Shared figure style (validated categorical palette, recessive grid) and saving."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# categorical slots in fixed order (validated palette)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GRID = dict(color="#d9d8d3", linewidth=0.6)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID["color"], "grid.linewidth": GRID["linewidth"],
                     "lines.linewidth": 1.2, "font.size": 9, "legend.frameon": False})


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
