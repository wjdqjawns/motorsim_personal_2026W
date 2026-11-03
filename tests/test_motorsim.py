"""Physics and identification sanity tests:  py -3.13 -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motorsim import simulate  # noqa: E402
from motorsim.data import _from_frame  # noqa: E402
from motorsim.ident import fit_static, identify_ls  # noqa: E402
from motorsim.models.friction import static_friction  # noqa: E402
from motorsim.signals import build  # noqa: E402

DC = {"motor": {"type": "dc", "params": {"R": 1.0, "L": 0.5e-3, "Kt": 0.05, "J": 2e-5}}}


def cfg(**over):
    c = {**DC, "sim": {"duration": 0.3, "substeps": 4}}
    c.update(over)
    return c


def test_dc_open_loop_steady_state_matches_analytic():
    B, V = 1e-4, 6.0
    r = simulate(cfg(friction={"model": "viscous", "params": {"B": B}},
                     control={"mode": "voltage"}, reference={"type": "constant", "value": V}))
    w_ss = 0.05 * V / (1.0 * B + 0.05 * 0.05)          # Kt V / (R B + Kt Ke)
    assert r.data["w"][0, -1] == pytest.approx(w_ss, rel=1e-3)


def test_karnopp_sticks_below_breakaway_and_moves_above():
    fr = {"model": "stribeck", "params": {"Tc": 5e-3, "Tst": 8e-3, "B": 1e-5, "ws": 5.0}}
    kt = 0.05
    below = simulate(cfg(friction=fr, control={"mode": "torque"}, reference={"type": "constant", "value": 7e-3}))
    above = simulate(cfg(friction=fr, control={"mode": "torque"}, reference={"type": "constant", "value": 9e-3}))
    assert np.abs(below.data["w"]).max() == 0.0
    assert above.data["w"][0, -1] > 1.0
    assert kt * below.data["i"][0, -1] == pytest.approx(7e-3, rel=0.02)


def test_lugre_presliding_is_small_and_breaks_away():
    fr = {"model": "lugre", "params": {"Tc": 5e-3, "Tst": 8e-3, "B": 1e-5, "ws": 5.0, "sigma0": 50, "sigma1": 0.05}}
    below = simulate(cfg(friction=fr, control={"mode": "torque"}, reference={"type": "constant", "value": 4e-3}))
    above = simulate(cfg(friction=fr, control={"mode": "torque"}, reference={"type": "constant", "value": 1e-2}))
    assert np.abs(below.data["theta"]).max() < 1e-3          # only bristle deflection
    assert above.data["w"][0, -1] > 10.0


@pytest.mark.parametrize("mode,ref,key,target", [
    ("current", 1.0, "i", 1.0),
    ("speed", 100.0, "w", 100.0),
    ("position", 2.0, "theta", 2.0),
])
def test_dc_closed_loop_tracks_reference(mode, ref, key, target):
    r = simulate(cfg(control={"mode": mode}, reference={"type": "constant", "value": ref},
                     sim={"duration": {"current": 0.01, "speed": 0.3, "position": 1.0}[mode]}))
    assert r.data[key][0, -1] == pytest.approx(target, rel=1e-2, abs=1e-3)


def test_pmsm_speed_loop_and_torque_equation():
    r = simulate({"motor": {"type": "pmsm"}, "control": {"mode": "speed"},
                  "reference": {"type": "constant", "value": 100.0}, "sim": {"duration": 0.3}})
    d = r.data
    assert d["w"][0, -1] == pytest.approx(100.0, rel=1e-2)
    assert abs(d["id"][0, -1]) < 1e-3
    assert d["Te"][0, -1] == pytest.approx(d["Tf"][0, -1], rel=1e-2)     # steady state: Te = Tf


def test_batch_equals_individual_runs():
    P = {"J": np.array([1e-5, 4e-5])}
    rb = simulate(cfg(reference={"type": "step", "value": 50.0}), params=P)
    r1 = simulate(cfg(reference={"type": "step", "value": 50.0}), params={"J": 4e-5})
    np.testing.assert_allclose(rb.data["w"][1], r1.data["w"][0], rtol=1e-9, atol=1e-9)


def test_signals_shapes_and_random_ranges():
    t = np.arange(1000) * 1e-3
    rng = np.random.default_rng(0)
    y = build([{"type": "sine", "amplitude": [1, 2], "freq": 3},
               {"type": "prbs", "amplitude": 0.5, "dwell": 0.01}], t, 4, rng)
    assert y.shape == (4, 1000)
    assert len({round(v, 6) for v in np.abs(y).max(1)}) > 1


def test_friction_fit_recovers_stribeck():
    p = {"Tc": 5e-3, "Tst": 8e-3, "ws": 5.0, "B": 1e-5, "delta": 2.0}
    w = np.r_[np.geomspace(0.3, 200, 20), -np.geomspace(0.3, 200, 20)]
    f = fit_static("stribeck", w, static_friction("stribeck", w, p))
    for k in ("Tc", "Tst", "ws", "B"):
        assert f.params[k] == pytest.approx(p[k], rel=1e-3)


def test_ls_identification_dc_noise_free():
    true = {"R": 1.2, "L": 0.6e-3, "Kt": 0.045, "J": 2.5e-5}
    c = {"motor": {"type": "dc", "params": true},
         "friction": {"model": "coulomb_viscous", "params": {"B": 1.5e-5, "Tc": 6e-3}},
         "control": {"mode": "speed", "nominal": {"R": 1.0, "L": 0.5e-3, "Kt": 0.05, "J": 2e-5}},
         "reference": [{"type": "random_steps", "low": -150, "high": 150, "dwell": [0.05, 0.15]},
                       {"type": "multisine", "amplitude": 30, "f_min": 1, "f_max": 200, "n": 20}],
         "sim": {"duration": 0.8, "seed": 3}}
    d = _from_frame(simulate(c).frame(0), None)
    res = identify_ls(d, "dc", friction_models=["coulomb_viscous"])
    p = res.params
    for k, tol in (("R", 0.01), ("L", 0.03), ("Kt", 0.01), ("J", 0.01)):
        assert p[k] == pytest.approx(true[k], rel=tol), k
    assert p["Tc"] == pytest.approx(6e-3, rel=0.05)
