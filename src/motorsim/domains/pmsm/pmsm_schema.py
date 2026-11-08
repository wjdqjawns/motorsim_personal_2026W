"""PMSM: defaults, parameter names, parameter tying rules."""
from __future__ import annotations

TYPE = "pmsm"
DEFAULTS = {
    "type": "pmsm",
    "frame": "dq",                  # dq: rotor-frame model | abc: 3-phase model + Clarke/Park FOC + inverter
    "params": {"R": 0.5, "Ld": 1.0e-3, "Lq": 1.0e-3, "psi": 0.01, "p": 4, "J": 3.0e-5},
    "limits": {"v_dc": 24.0, "i_max": 10.0},
    "inverter": {"type": "average", "substeps": 1},   # abc frame only: average | switching
}
PARAM_NAMES = ["R", "Ld", "Lq", "psi", "p", "J"]
ELEC_PARAMS = ["R", "Ld", "Lq", "psi"]
CUR_KEYS = ["id", "iq"]
DATASET_CHANNELS = ["vd", "vq", "id_meas", "iq_meas", "w_enc"]
MODES = ("voltage", "current", "torque", "speed", "position")
MODES_ABC = MODES + ("vf", "abc")


def torque_constant(p: dict):
    return 1.5 * p["p"] * p["psi"]


def finalize_params(p: dict) -> dict:
    return p


def tie_nominal(p: dict, nom_given: dict) -> dict:
    return p


def tie_randomized(out: dict, base: dict, randomize: dict) -> dict:
    return out


def tie_fit(P: dict, base: dict, fit_names) -> dict:
    return P
