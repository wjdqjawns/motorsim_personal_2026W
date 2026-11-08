"""Zero-phase filtering and encoder-based speed/acceleration."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def lowpass(x, fs: float, fc: float | None, order: int = 2):
    """Zero-phase Butterworth (filtfilt). fc <= 0, None or >= fs/2 -> unfiltered."""
    if fc is None or fc <= 0 or fc >= fs / 2:
        return np.asarray(x, float)
    b, a = butter(order, fc / (fs / 2))
    return filtfilt(b, a, x)


def mechanical_signals(t, theta=None, w=None, lp_hz: float | None = None):
    """Speed and acceleration from the encoder angle (preferred) or a speed signal."""
    Ts = t[1] - t[0]
    fs = 1.0 / Ts
    lp_hz = 0.04 * fs if lp_hz is None else lp_hz
    if theta is not None:
        w_s = np.gradient(lowpass(theta, fs, lp_hz), Ts)
    else:
        w_s = lowpass(w, fs, lp_hz)
    return w_s, np.gradient(w_s, Ts)
