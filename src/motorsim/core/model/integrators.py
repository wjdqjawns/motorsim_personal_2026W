"""Fixed-step integrators. h may be a scalar or an (N,) array (per-batch step length)."""
from __future__ import annotations


def rk4(f, x, u, tf, tl, h):
    k1 = f(x, u, tf, tl)
    k2 = f(x + 0.5 * h * k1, u, tf, tl)
    k3 = f(x + 0.5 * h * k2, u, tf, tl)
    k4 = f(x + h * k3, u, tf, tl)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
