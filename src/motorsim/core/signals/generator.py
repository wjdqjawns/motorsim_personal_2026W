"""Reference / excitation signal generator.

A spec is a dict with a 'type' (or a list of specs, which are summed). Any numeric
field may be given as [lo, hi] to draw it uniformly per batch element.

    constant      value
    step          value, t0, initial
    steps         values: [...], duration: s or [...]        (holds last value)
    sine          amplitude, freq, offset, phase
    square        amplitude, freq, offset
    triangle      amplitude, freq, offset
    chirp         amplitude, f0, f1, method (linear|logarithmic|quadratic), offset
    multisine     amplitude (peak), f_min, f_max, n, offset     (random phases)
    prbs          amplitude, dwell, offset                     (random +/-1 levels)
    random_steps  low, high, dwell (s or [min, max])           (random levels)
    array         values (array broadcastable to (N, T))
Common optional keys: start (signal is 0 before), stop (0 after).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import chirp as _chirp


def _rv(v, N, rng):
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return rng.uniform(float(v[0]), float(v[1]), size=(N, 1))
    return np.full((N, 1), float(v))


def _piecewise(edges, levels, t):
    idx = np.searchsorted(edges, t, side="right")
    return levels[np.minimum(idx, len(levels) - 1)]


def build(spec, t: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
    T = len(t)
    if spec is None:
        return np.zeros((N, T))
    if isinstance(spec, (list, tuple)):
        return sum(build(s, t, N, rng) for s in spec)
    typ = spec["type"]
    g = lambda key, d=0.0: _rv(spec.get(key, d), N, rng)
    tt = t[None, :]

    if typ == "constant":
        y = g("value") * np.ones((1, T))
    elif typ == "step":
        y0 = g("initial")
        y = y0 + (g("value", 1.0) - y0) * (tt >= spec.get("t0", 0.0))
    elif typ == "steps":
        values = np.asarray(spec["values"], float)
        dur = np.broadcast_to(np.asarray(spec.get("duration", 0.1), float), values.shape)
        y = _piecewise(np.cumsum(dur), values, t)[None, :]
    elif typ == "sine":
        y = g("offset") + g("amplitude", 1.0) * np.sin(2 * np.pi * g("freq", 1.0) * tt + g("phase"))
    elif typ == "square":
        y = g("offset") + g("amplitude", 1.0) * np.sign(np.sin(2 * np.pi * g("freq", 1.0) * tt + 1e-12))
    elif typ == "triangle":
        y = g("offset") + g("amplitude", 1.0) * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * g("freq", 1.0) * tt))
    elif typ == "chirp":
        base = _chirp(t, spec.get("f0", 1.0), t[-1], spec.get("f1", 100.0),
                      method=spec.get("method", "logarithmic"), phi=-90)
        y = g("offset") + g("amplitude", 1.0) * base[None, :]
    elif typ == "multisine":
        n = int(spec.get("n", 20))
        f = np.geomspace(spec.get("f_min", 1.0), spec.get("f_max", 100.0), n)
        ph = rng.uniform(0, 2 * np.pi, (N, n, 1))
        s = np.sin(2 * np.pi * f[None, :, None] * t[None, None, :] + ph).sum(axis=1)
        s /= np.abs(s).max(axis=1, keepdims=True)
        y = g("offset") + g("amplitude", 1.0) * s
    elif typ == "prbs":
        dwell = float(spec.get("dwell", 0.01))
        nseg = int(np.ceil((t[-1] + 1e-12) / dwell)) + 1
        lv = rng.choice([-1.0, 1.0], size=(N, nseg))
        idx = np.minimum((t / dwell).astype(int), nseg - 1)
        y = g("offset") + g("amplitude", 1.0) * lv[:, idx]
    elif typ == "random_steps":
        lo, hi = float(spec.get("low", -1.0)), float(spec.get("high", 1.0))
        dw = spec.get("dwell", 0.05)
        dmin, dmax = (dw, dw) if np.isscalar(dw) else map(float, dw)
        y = np.empty((N, T))
        for n in range(N):
            edges, tc = [], 0.0
            while tc <= t[-1]:
                tc += rng.uniform(dmin, dmax) if dmax > dmin else dmin
                edges.append(tc)
            y[n] = _piecewise(np.asarray(edges), rng.uniform(lo, hi, len(edges)), t)
    elif typ == "array":
        y = np.asarray(spec["values"], float)
    else:
        raise ValueError(f"unknown signal type {typ!r}")

    y = np.broadcast_to(y, (N, T)).astype(float).copy()
    if "start" in spec:
        y[:, t < spec["start"]] = 0.0
    if "stop" in spec:
        y[:, t >= spec["stop"]] = 0.0
    return y
