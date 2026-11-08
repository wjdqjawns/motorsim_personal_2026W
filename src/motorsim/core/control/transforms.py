"""Reference-frame transforms and space-vector modulation (amplitude-invariant, batched).

    Clarke      abc -> alpha-beta   x_alpha = (2a - b - c)/3,  x_beta = (b - c)/sqrt(3)
    Park        alpha-beta -> dq    d =  cos(th) alpha + sin(th) beta
                                    q = -sin(th) alpha + cos(th) beta
    SVPWM       min-max (common-mode) injection; linear up to |v| = Vdc / sqrt(3)
Amplitude-invariant: a balanced phase amplitude A gives |x_alpha-beta| = A, and the
torque is Te = 1.5 p (psi iq + (Ld - Lq) id iq).
"""
from __future__ import annotations

import numpy as np

SQRT3 = np.sqrt(3.0)


def clarke(a, b, c):
    return (2.0 * a - b - c) / 3.0, (b - c) / SQRT3


def inv_clarke(alpha, beta):
    return alpha, -0.5 * alpha + 0.5 * SQRT3 * beta, -0.5 * alpha - 0.5 * SQRT3 * beta


def park(alpha, beta, theta):
    c, s = np.cos(theta), np.sin(theta)
    return c * alpha + s * beta, -s * alpha + c * beta


def inv_park(d, q, theta):
    c, s = np.cos(theta), np.sin(theta)
    return c * d - s * q, s * d + c * q


def svpwm(v_alpha, v_beta, vdc):
    """Phase duty cycles (3, N) in [0, 1] for a 2-level inverter (legs referred to the - rail)."""
    va, vb, vc = inv_clarke(v_alpha, v_beta)
    vcm = -0.5 * (np.maximum(np.maximum(va, vb), vc) + np.minimum(np.minimum(va, vb), vc))
    return np.clip(0.5 + (np.array([va, vb, vc]) + vcm) / vdc, 0.0, 1.0)

def spwm(va, vb, vc, vdc):
    """Plain sinusoidal PWM (no common-mode injection): linear up to phase amplitude Vdc/2."""
    return np.clip(0.5 + np.array([va, vb, vc]) / vdc, 0.0, 1.0)

def wrap_angle(theta):
    """Wrap to (-pi, pi]."""
    return np.pi - np.mod(np.pi - theta, 2 * np.pi)