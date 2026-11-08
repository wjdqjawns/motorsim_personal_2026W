"""Report / file exporters."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def md_table(df: pd.DataFrame) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    fmt = lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else ("" if v is None else str(v))
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_report(path: str | Path, lines: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
