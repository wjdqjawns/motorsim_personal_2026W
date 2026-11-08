"""Project-wide logger (plain messages to stdout)."""
from __future__ import annotations

import logging
import sys

def get_logger(name: str = "motorsim") -> logging.Logger:
    root = logging.getLogger("motorsim")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    return logging.getLogger(name if name.startswith("motorsim") else f"motorsim.{name}")