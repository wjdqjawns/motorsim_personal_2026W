"""Identified-parameter files (YAML)."""
from __future__ import annotations

from pathlib import Path

import yaml


def save_params(path: str | Path, mtype: str, friction_model: str, params: dict, extra: dict | None = None) -> Path:
    doc = {"motor_type": mtype, "friction_model": friction_model,
           "params": {k: float(v) for k, v in params.items()}}
    if extra:
        doc.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    return path


def load_params(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
