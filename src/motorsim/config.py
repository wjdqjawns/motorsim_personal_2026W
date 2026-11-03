"""Configuration handling: defaults, YAML loading, flat physical-parameter dicts.

All physical quantities are SI. Angles/speeds are *mechanical* (rad, rad/s).
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

MOTOR_DEFAULTS = {
    "dc": {
        "type": "dc",
        # Ke defaults to Kt (SI units) when omitted
        "params": {"R": 1.0, "L": 0.5e-3, "Kt": 0.05, "J": 2.0e-5},
        "limits": {"v_max": 24.0, "i_max": 5.0},
    },
    "pmsm": {
        "type": "pmsm",
        "params": {"R": 0.5, "Ld": 1.0e-3, "Lq": 1.0e-3, "psi": 0.01, "p": 4, "J": 3.0e-5},
        "limits": {"v_dc": 24.0, "i_max": 10.0},
    },
}

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
        "mode": "speed",            # voltage | current | torque | speed | position
        "Ts": 1.0e-4,               # controller / sampling period [s]
        "bandwidth_hz": {"current": 500.0, "speed": 30.0, "position": 5.0},
        "gains": {},                # optional overrides: {current: {kp, ki}, speed: {kp, ki}, position: {kp}}
        "nominal": {},              # controller's (possibly wrong) model of the motor
        "speed_max": 300.0,         # position-loop speed limit [rad/s]
        "backemf_ff": True,
        "decoupling": True,         # PMSM dq decoupling feed-forward
    },
    "reference": {"type": "step", "value": 100.0, "t0": 0.01},
    "reference_d": None,            # PMSM: id reference (or vd in voltage mode)
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

MOTOR_PARAM_NAMES = {
    "dc": ["R", "L", "Kt", "Ke", "J"],
    "pmsm": ["R", "Ld", "Lq", "psi", "p", "J"],
}


def deep_merge(base: dict, over: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(cfg: str | Path | dict | None = None) -> dict:
    """Load a YAML file (or dict) and fill in defaults for the selected motor type."""
    if cfg is None:
        cfg = {}
    elif isinstance(cfg, (str, Path)):
        with open(cfg, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    if cfg.get("_resolved"):
        return cfg
    mtype = (cfg.get("motor") or {}).get("type", "dc")
    if mtype not in MOTOR_DEFAULTS:
        raise ValueError(f"unknown motor type {mtype!r} (use dc | pmsm)")
    out = deep_merge(COMMON_DEFAULTS, {"motor": MOTOR_DEFAULTS[mtype]})
    out = deep_merge(out, cfg)
    out["_resolved"] = True
    return out


def flat_params(cfg: dict) -> dict:
    """All physical plant parameters (motor + friction + load) as one flat dict of floats."""
    cfg = load_config(cfg)
    p = {k: float(v) for k, v in cfg["motor"]["params"].items()}
    if cfg["motor"]["type"] == "dc":
        p.setdefault("Ke", p["Kt"])
    fp = dict(FRICTION_DEFAULTS)
    fp.update({k: float(v) for k, v in (cfg["friction"].get("params") or {}).items()})
    p.update(fp)
    load = cfg["load"]
    cog = load.get("cogging") or {}
    p["TL"] = float(load.get("torque", 0.0))
    p["cog_A"] = float(cog.get("amplitude", 0.0))
    p["cog_N"] = float(cog.get("periods", 0))
    p["cog_phi"] = float(cog.get("phase", 0.0))
    return p


def nominal_params(cfg: dict) -> dict:
    """Parameters the *controller* believes in (defaults to the plant's)."""
    cfg = load_config(cfg)
    p = flat_params(cfg)
    nom = {k: float(v) for k, v in (cfg["control"].get("nominal") or {}).items()}
    p.update(nom)
    if cfg["motor"]["type"] == "dc" and "Kt" in nom and "Ke" not in nom:
        p["Ke"] = p["Kt"]
    return p


def torque_constant(mtype: str, p: dict):
    """Torque per unit (q-axis) current."""
    if mtype == "dc":
        return p["Kt"]
    return 1.5 * p["p"] * p["psi"]


def with_params(cfg: dict, params: dict, friction_model: str | None = None) -> dict:
    """Return a copy of cfg whose plant uses `params` (flat dict) and optionally another friction model."""
    cfg = copy.deepcopy(load_config(cfg))
    mtype = cfg["motor"]["type"]
    for k, v in params.items():
        if k in MOTOR_PARAM_NAMES[mtype]:
            cfg["motor"]["params"][k] = float(v)
        elif k in FRICTION_DEFAULTS:
            cfg["friction"]["params"][k] = float(v)
        elif k == "TL":
            cfg["load"]["torque"] = float(v)
    if friction_model:
        cfg["friction"]["model"] = friction_model
    return cfg


def sample_params(cfg: dict, n: int, rng: np.random.Generator) -> dict:
    """Draw n random plant-parameter sets according to cfg['randomize'].

    randomize:
      R: [0.5, 2.0]                         # factor on nominal, log-uniform
      Tc: {range: [0.0, 0.02], absolute: true, dist: uniform}
    """
    cfg = load_config(cfg)
    base = flat_params(cfg)
    out = {k: np.full(n, v) for k, v in base.items()}
    for name, spec in (cfg.get("randomize") or {}).items():
        if name not in base:
            raise KeyError(f"randomize: unknown parameter {name!r}")
        if isinstance(spec, (list, tuple)):
            spec = {"range": spec}
        lo, hi = map(float, spec["range"])
        absolute = spec.get("absolute", False)
        dist = spec.get("dist", "uniform" if absolute else "log")
        if dist == "log":
            if lo <= 0:
                raise ValueError(f"log-uniform range for {name} must be positive")
            val = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        else:
            val = rng.uniform(lo, hi, n)
        out[name] = val if absolute else base[name] * val
    # keep Ke = Kt (SI) unless Ke itself is randomized
    if cfg["motor"]["type"] == "dc" and "Kt" in cfg["randomize"] and "Ke" not in cfg["randomize"]:
        out["Ke"] = out["Kt"] * base["Ke"] / base["Kt"]
    return out


def save_params(path: str | Path, mtype: str, friction_model: str, params: dict, extra: dict | None = None):
    doc = {
        "motor_type": mtype,
        "friction_model": friction_model,
        "params": {k: float(v) for k, v in params.items()},
    }
    if extra:
        doc.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)


def load_params(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
