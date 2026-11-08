"""PMSM domain: dq / abc plants, dq control and full FOC (Clarke/Park/SVPWM), inverter, identification."""
from motorsim.core.sim.drive import ZOHDrive
from motorsim.domains.pmsm.pmsm_control import FOCController, PMSMController
from motorsim.domains.pmsm.pmsm_ident import electrical_ls, electromagnetic_torque, replay_inputs
from motorsim.domains.pmsm.pmsm_inverter import AverageInverter, SwitchingInverter, make_inverter
from motorsim.domains.pmsm.pmsm_model import PMSMabc, PMSMdq
from motorsim.domains.pmsm.pmsm_schema import (CUR_KEYS, DATASET_CHANNELS, DEFAULTS, ELEC_PARAMS, MODES, MODES_ABC,
                                               PARAM_NAMES, TYPE, finalize_params, tie_fit, tie_nominal,
                                               tie_randomized, torque_constant)


def _frame(cfg):
    frame = cfg["motor"].get("frame", "dq")
    if frame not in ("dq", "abc"):
        raise ValueError(f"PMSM frame must be 'dq' or 'abc', got {frame!r}")
    return frame


def make_plant(P: dict, cfg: dict):
    return PMSMabc(P) if _frame(cfg) == "abc" else PMSMdq(P)


def make_controller(cfg: dict, nom: dict, N: int):
    return FOCController(cfg, nom, N) if _frame(cfg) == "abc" else PMSMController(cfg, nom, N)


def make_drive(cfg: dict, P: dict, N: int):
    if _frame(cfg) == "abc":
        return make_inverter(cfg, P, N)
    return ZOHDrive(cfg["control"]["Ts"], int(cfg["sim"]["substeps"]))


def replay_overrides(cfg: dict) -> dict:
    """Open-loop dq voltage replay (identification works on dq signals for both frames)."""
    cfg["motor"]["limits"]["v_dc"] = 1e9
    cfg["motor"]["frame"] = "dq"
    return cfg
