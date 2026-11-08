"""YAML loading with defaults (common + selected motor domain)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from motorsim.core.config.schema import COMMON_DEFAULTS

def deep_merge(base: dict, over: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def load_config(cfg: str | Path | dict | None = None) -> dict:
    """Load a YAML file (or dict) and fill in defaults for the selected motor type."""
    from motorsim.domains import get_domain

    if cfg is None:
        cfg = {}
    elif isinstance(cfg, (str, Path)):
        with open(cfg, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    if cfg.get("_resolved"):
        return cfg
    dom = get_domain((cfg.get("motor") or {}).get("type", "dc"))
    out = deep_merge(COMMON_DEFAULTS, {"motor": dom.DEFAULTS})
    out = deep_merge(out, cfg)
    out["_resolved"] = True
    return out