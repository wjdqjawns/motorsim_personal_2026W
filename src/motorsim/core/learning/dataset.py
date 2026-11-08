"""Synthetic training-data generation with randomized plant parameters.

Every sample = one closed-loop experiment of a motor whose physical parameters are
drawn from cfg['randomize'] (the controller keeps using the nominal model, like a real
drive commissioning on an unknown motor). Stored as a compressed .npz:

    X        (N, C, T) float32   measured channels (what a real drive records)
    y        (N, P)    float64   true values of the randomized parameters
    channels, param_names, nominal (P,), Ts, config (YAML)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import yaml

from motorsim.core.analysis.filters import mechanical_signals
from motorsim.core.config import flat_params, load_config, sample_params
from motorsim.utils.logger import get_logger

_log = get_logger(__name__).info


def default_channels(mtype: str) -> list[str]:
    from motorsim.domains import get_domain
    return list(get_domain(mtype).DATASET_CHANNELS)

def generate_dataset(cfg, n: int, out: str | Path, batch: int = 256, channels=None, decimate: int = 1, seed: int | None = None, log=_log) -> Path:
    from motorsim.core.sim.simulator import simulate

    cfg = load_config(cfg)
    mtype = cfg["motor"]["type"]
    channels = channels or cfg.get("dataset", {}).get("channels") or default_channels(mtype)
    names = list(cfg["randomize"])
    
    if not names:
        raise ValueError("dataset config needs a non-empty 'randomize' section")
    
    rng = np.random.default_rng(cfg["sim"]["seed"] if seed is None else seed)
    nominal = flat_params(cfg)
    Ts = cfg["control"]["Ts"]
    X, Y = [], []
    t0 = time.time()
    
    for b0 in range(0, n, batch):
        nb = min(batch, n - b0)
        P = sample_params(cfg, nb, rng)
        r = simulate(cfg, params=P, N=nb, rng=rng)
        cols = []

        for c in channels:
            if c == "w_enc":            # speed from the encoder angle (zero-phase, offline)
                cols.append(np.stack([mechanical_signals(r.t, th)[0] for th in r.data["theta_meas"]]))
            else:
                cols.append(r.data[c])

        X.append(np.stack(cols, axis=1)[:, :, ::decimate].astype(np.float32))
        Y.append(np.stack([P[k] for k in names], axis=1))
        log(f"  dataset: {b0 + nb}/{n} samples ({time.time() - t0:.1f} s)")
    
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    np.savez_compressed(out, X=np.concatenate(X), y=np.concatenate(Y), channels=np.array(channels), param_names=np.array(names), nominal=np.array([nominal[k] for k in names]), Ts=Ts * decimate, config=yaml.safe_dump(clean, sort_keys=False))
    
    return out

def load_dataset(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=False)
    
    return {k: z[k] for k in z.files}