"""3-phase PMSM / FOC tests:  py -3.13 -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motorsim import Motor, simulate  # noqa: E402
from motorsim.core.control.transforms import clarke, inv_clarke, inv_park, park, svpwm  # noqa: E402

PM = {"R": 0.5, "Ld": 1.0e-3, "Lq": 1.0e-3, "psi": 0.01, "p": 4, "J": 3e-5}


def pmsm(frame="abc", inverter="average", **over):
    c = {"motor": {"type": "pmsm", "frame": frame, "params": dict(PM), "inverter": {"type": inverter}},
         "friction": {"model": "viscous", "params": {"B": 1e-5}}, "sim": {"duration": 0.1}}
    c.update(over)
    return c


def test_clarke_park_roundtrip_and_amplitude_invariance():
    th = np.linspace(0, 4 * np.pi, 200)
    a, b, c = np.cos(th), np.cos(th - 2 * np.pi / 3), np.cos(th + 2 * np.pi / 3)
    al, be = clarke(a, b, c)
    np.testing.assert_allclose(np.hypot(al, be), 1.0, atol=1e-12)          # amplitude invariant
    d, q = park(al, be, th)
    np.testing.assert_allclose(d, 1.0, atol=1e-12)                          # aligned vector -> pure d
    np.testing.assert_allclose(q, 0.0, atol=1e-12)
    np.testing.assert_allclose(np.array(inv_clarke(*inv_park(d, q, th))), np.array([a, b, c]), atol=1e-12)

def test_svpwm_reproduces_voltage_vector_up_to_linear_limit():
    vdc = 24.0
    ang = np.linspace(0, 2 * np.pi, 37)
    mag = 0.999 * vdc / np.sqrt(3)
    va, vb = mag * np.cos(ang), mag * np.sin(ang)
    duty = svpwm(va, vb, vdc)
    assert duty.min() >= 0 and duty.max() <= 1
    np.testing.assert_allclose(np.array(clarke(*(duty * vdc))), np.array([va, vb]), atol=1e-9)


@pytest.mark.parametrize("inverter", ["average", "switching"])
def test_abc_foc_matches_dq_model(inverter):
    """Independent chain (abc plant + Clarke/Park + SVPWM + inverter) vs. the dq model: < 0.1 rad/s
    over a +/-150 rad/s profile (measured ~0.02 rad/s)."""
    ref = {"type": "steps", "values": [0, 150, -150], "duration": 0.035}
    w_dq = simulate(pmsm("dq", reference=ref)).data["w"][0]
    w_abc = simulate(pmsm("abc", inverter, reference=ref)).data["w"][0]
    assert np.abs(w_abc - w_dq).max() < 0.1


def test_rotor_aligns_with_phase_a_under_dc_voltage():
    m = Motor(pmsm("abc", friction={"model": "viscous", "params": {"B": 2e-4}}), init={"th": 0.3})
    V = 1.0
    out = None
    for _ in range(30000):
        out = m.step(V, -V / 2, -V / 2, dt=2e-5)
    th_e = np.angle(np.exp(1j * out["theta_e"]))
    assert abs(th_e) < 1e-3                     # d-axis pulled onto the alpha (phase-a) axis
    assert out["ia"] == pytest.approx(V / PM["R"], rel=1e-3)


def test_angle_offset_reduces_torque_by_cosine():
    def te(offset):
        c = pmsm("abc", control={"mode": "current", "angle_offset_deg": offset},
                 reference={"type": "constant", "value": 2.0}, sim={"duration": 0.004})
        c["motor"]["params"]["J"] = 10.0          # locked-ish rotor: isolate the current loop
        return simulate(c).data["Te"][0, -1]
    assert te(30.0) / te(0.0) == pytest.approx(np.cos(np.deg2rad(30.0)), rel=2e-3)


def test_open_loop_vf_reaches_synchronous_speed():
    f = 20.0
    c = pmsm("abc", control={"mode": "vf", "vf": {"v_per_hz": 0.3, "boost": 0.5}},
             reference={"type": "constant", "value": f}, sim={"duration": 0.5})
    w = simulate(c).data["w"][0]
    assert w[-500:].mean() == pytest.approx(2 * np.pi * f / PM["p"], rel=1e-2)
