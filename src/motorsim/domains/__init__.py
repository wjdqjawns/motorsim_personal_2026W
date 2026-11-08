"""Motor-type domains. Each domain package provides the same hooks:

    DEFAULTS, PARAM_NAMES, ELEC_PARAMS, CUR_KEYS, DATASET_CHANNELS, torque_constant,
    finalize_params, tie_nominal, tie_randomized, tie_fit,
    make_plant, make_controller, make_drive, replay_overrides,
    electrical_ls, electromagnetic_torque, replay_inputs
"""
from __future__ import annotations

from motorsim.domains import dc_motor, pmsm

DOMAINS = {dc_motor.TYPE: dc_motor, pmsm.TYPE: pmsm}


def get_domain(mtype: str):
    try:
        return DOMAINS[mtype]
    except KeyError:
        raise ValueError(f"unknown motor type {mtype!r}; choose from {list(DOMAINS)}") from None
