"""DC motor domain: plant, cascade controller, identification hooks."""
from motorsim.core.sim.drive import ZOHDrive
from motorsim.domains.dc_motor.dc_control import DCController
from motorsim.domains.dc_motor.dc_ident import electrical_ls, electromagnetic_torque, replay_inputs
from motorsim.domains.dc_motor.dc_model import DCMotor
from motorsim.domains.dc_motor.dc_schema import (CUR_KEYS, DATASET_CHANNELS, DEFAULTS, ELEC_PARAMS, MODES,
                                                 PARAM_NAMES, TYPE, finalize_params, tie_fit, tie_nominal,
                                                 tie_randomized, torque_constant)


def make_plant(P: dict, cfg: dict):
    return DCMotor(P)


def make_controller(cfg: dict, nom: dict, N: int):
    return DCController(cfg, nom, N)


def make_drive(cfg: dict, P: dict, N: int):
    return ZOHDrive(cfg["control"]["Ts"], int(cfg["sim"]["substeps"]))


def replay_overrides(cfg: dict) -> dict:
    """Changes that turn cfg into an open-loop voltage-replay model."""
    cfg["motor"]["limits"]["v_max"] = 1e9
    return cfg
