"""DC motor: defaults, parameter names, parameter tying rules."""
from __future__ import annotations

TYPE = "dc"
DEFAULTS = {
    "type": "dc",
    "params": {"R": 1.0, "L": 0.5e-3, "Kt": 0.05, "J": 2.0e-5},   # Ke defaults to Kt (SI)
    "limits": {"v_max": 24.0, "i_max": 5.0},
}
PARAM_NAMES = ["R", "L", "Kt", "Ke", "J"]
ELEC_PARAMS = ["R", "L", "Kt"]          # identified electrical set (Ke tied to Kt)
CUR_KEYS = ["i"]                        # currents compared in replay / OE
DATASET_CHANNELS = ["v", "i_meas", "w_enc"]
MODES = ("voltage", "current", "torque", "speed", "position")


def torque_constant(p: dict):
    return p["Kt"]


def finalize_params(p: dict) -> dict:
    p.setdefault("Ke", p["Kt"])
    return p


def tie_nominal(p: dict, nom_given: dict) -> dict:
    if "Kt" in nom_given and "Ke" not in nom_given:
        p["Ke"] = p["Kt"]
    return p


def tie_randomized(out: dict, base: dict, randomize: dict) -> dict:
    """Keep Ke = Kt (SI) unless Ke itself is randomized."""
    if "Kt" in randomize and "Ke" not in randomize:
        out["Ke"] = out["Kt"] * base["Ke"] / base["Kt"]
    return out


def tie_fit(P: dict, base: dict, fit_names) -> dict:
    """Output-error fit: Ke follows Kt when only Kt is fitted."""
    if "Kt" in fit_names and "Ke" not in fit_names:
        P["Ke"] = P["Kt"] * base["Ke"] / base["Kt"]
    return P
