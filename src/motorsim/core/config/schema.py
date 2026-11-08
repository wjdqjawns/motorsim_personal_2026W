"""Motor-independent configuration defaults. Motor defaults live in each domain's *_schema.py."""
from __future__ import annotations

# Superset of friction parameters; each model uses only the ones it needs.
FRICTION_DEFAULTS = {
    "B": 1.0e-5,       # viscous [Nm s/rad]
    "Tc": 5.0e-3,      # Coulomb [Nm]
    "Tst": 8.0e-3,     # static / breakaway [Nm]
    "ws": 5.0,         # Stribeck velocity [rad/s]
    "delta": 2.0,      # Stribeck exponent [-]
    "sigma0": 50.0,    # LuGre bristle stiffness [Nm/rad]
    "sigma1": 0.05,    # LuGre bristle damping [Nm s/rad]
}

COMMON_DEFAULTS = {
    "name": "sim",
    "friction": {"model": "coulomb_viscous", "params": {}, "v_eps": 1.0e-3},
    "load": {"torque": 0.0, "cogging": {"amplitude": 0.0, "periods": 0, "phase": 0.0}},
    "control": {
        "mode": "speed",            # see the domain controller for the available modes
        "Ts": 1.0e-4,               # controller / sampling period [s]
        "bandwidth_hz": {"current": 500.0, "speed": 30.0, "position": 5.0},
        "gains": {},                # optional overrides: {current: {kp, ki}, speed: {kp, ki}, position: {kp}}
        "nominal": {},              # controller's (possibly wrong) model of the motor
        "speed_max": 300.0,         # position-loop speed limit [rad/s]
        "backemf_ff": True,
        "decoupling": True,         # PMSM dq decoupling feed-forward
        "angle_offset_deg": 0.0,    # PMSM abc: encoder / rotor-flux misalignment (electrical deg)
        "delay_comp": True,         # PMSM abc: rotate the output voltage by 0.5 we Ts
        "current_sensors": 3,       # PMSM abc: 2 -> ic = -ia - ib
        "vf": {"v_per_hz": 0.05, "boost": 0.5},   # PMSM abc open-loop V/f mode
    },
    "reference": {"type": "step", "value": 100.0, "t0": 0.01},
    "reference_d": None,            # PMSM: id reference (or vd in voltage mode)
    "reference_abc": None,          # PMSM abc mode: [spec_a, spec_b, spec_c] phase voltages
    "sensors": {
        "current_noise": 0.0,       # std [A]
        "current_offset": 0.0,      # [A]
        "encoder_cpr": 0,           # counts per rev after quadrature; 0 = ideal
        "speed_lpf_hz": 0.0,        # 1st-order LPF on differentiated encoder; 0 = off
        "speed_source": "encoder",  # encoder | true
    },
    "sim": {"duration": 0.5, "substeps": 4, "seed": 0, "batch": 1},
    "randomize": {},
}
