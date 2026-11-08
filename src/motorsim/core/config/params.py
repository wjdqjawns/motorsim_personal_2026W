"""Flat physical-parameter dicts (motor + friction + load), nominal values, randomization."""
from __future__ import annotations

import copy

import numpy as np

from motorsim.core.config.loader import load_config
from motorsim.core.config.schema import FRICTION_DEFAULTS


def _domain(cfg):
    from motorsim.domains import get_domain
    return get_domain(cfg["motor"]["type"])


def flat_params(cfg) -> dict:
    """All physical plant parameters as one flat dict of floats."""
    cfg = load_config(cfg)
    p = {k: float(v) for k, v in cfg["motor"]["params"].items()}
    p = _domain(cfg).finalize_params(p)
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


def nominal_params(cfg) -> dict:
    """Parameters the *controller* believes in (defaults to the plant's)."""
    cfg = load_config(cfg)
    p = flat_params(cfg)
    nom = {k: float(v) for k, v in (cfg["control"].get("nominal") or {}).items()}
    p.update(nom)
    return _domain(cfg).tie_nominal(p, nom)


def torque_constant(mtype: str, p: dict):
    from motorsim.domains import get_domain
    return get_domain(mtype).torque_constant(p)


def with_params(cfg, params: dict, friction_model: str | None = None) -> dict:
    """Copy of cfg whose plant uses `params` (flat dict) and optionally another friction model."""
    cfg = copy.deepcopy(load_config(cfg))
    names = _domain(cfg).PARAM_NAMES
    for k, v in params.items():
        if k in names:
            cfg["motor"]["params"][k] = float(v)
        elif k in FRICTION_DEFAULTS:
            cfg["friction"]["params"][k] = float(v)
        elif k == "TL":
            cfg["load"]["torque"] = float(v)
    if friction_model:
        cfg["friction"]["model"] = friction_model
    return cfg


def sample_params(cfg, n: int, rng: np.random.Generator) -> dict:
    """Draw n random plant-parameter sets according to cfg['randomize'].

    randomize:
      R: [0.5, 2.0]                         # factor on nominal, log-uniform
      Tc: {range: [0.0, 0.02], absolute: true, dist: uniform}
    """
    cfg = load_config(cfg)
    base = flat_params(cfg)
    out = {k: np.full(n, v) for k, v in base.items()}
    rand = cfg.get("randomize") or {}
    for name, spec in rand.items():
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
    return _domain(cfg).tie_randomized(out, base, rand)
