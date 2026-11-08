"""Learning-based (amortized) parameter identification with PyTorch.

A 1-D CNN maps a recorded experiment (channels x time) directly to the motor
parameters. Targets are log(p / p_nominal), standardized; input channels are scaled
by FIXED per-channel constants (not per-sample normalization, which would erase the
amplitude information that carries R, Kt, ...).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from motorsim.core.learning.dataset import load_dataset
from motorsim.core.learning.network import ParamNet
from motorsim.utils.logger import get_logger

_log = get_logger(__name__).info

def _metrics(y_true, y_pred, names) -> pd.DataFrame:
    rel = (y_pred - y_true) / y_true
    lt, lp = np.log(y_true), np.log(y_pred)
    r2 = 1 - ((lt - lp) ** 2).sum(0) / np.maximum(((lt - lt.mean(0)) ** 2).sum(0), 1e-300)
    return pd.DataFrame({"param": names, "median_abs_rel_err_%": 100 * np.median(np.abs(rel), 0), "p90_abs_rel_err_%": 100 * np.percentile(np.abs(rel), 90, 0), "r2_log": r2})

def train(dataset_path, out_path, epochs: int = 40, batch: int = 64, lr: float = 2e-3, val_frac: float = 0.15, seed: int = 0, width: int = 64, log=_log) -> tuple[Path, pd.DataFrame]:
    torch.manual_seed(seed)
    ds = load_dataset(dataset_path)
    X, y, nominal = ds["X"].astype(np.float32), ds["y"], ds["nominal"]
    names = [str(s) for s in ds["param_names"]]
    x_scale = np.abs(X).reshape(X.shape[0], X.shape[1], -1).max(-1).mean(0) + 1e-9   # per-channel constant
    tgt = np.log(y / nominal)
    t_mu, t_sd = tgt.mean(0), tgt.std(0) + 1e-9
    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(val_frac * n))
    iv, it = perm[:n_val], perm[n_val:]
    Xt = torch.from_numpy(X / x_scale[None, :, None])
    Yt = torch.from_numpy(((tgt - t_mu) / t_sd).astype(np.float32))
    model = ParamNet(X.shape[1], len(names), width)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(it) / batch)))
    lossf = nn.MSELoss()
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        rng.shuffle(it)
        tot = 0.0
        for b in range(0, len(it), batch):
            idx = torch.from_numpy(it[b:b + batch])
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), Yt[idx])
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xt[iv]), Yt[iv]).item()
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            log(f"  epoch {ep + 1:3d}/{epochs}  train {tot / len(it):.4f}  val {vl:.4f}  ({time.time() - t0:.0f} s)")
    meta = {"names": names, "nominal": nominal, "x_scale": x_scale, "t_mu": t_mu, "t_sd": t_sd,
            "channels": [str(c) for c in ds["channels"]], "Ts": float(ds["Ts"]), "T": X.shape[-1],
            "c_in": X.shape[1], "width": width}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "meta": meta}, out_path)
    y_pred = _predict(model, meta, X[iv])
    return out_path, _metrics(y[iv], y_pred, names)

def _predict(model, meta, X):
    model.eval()
    with torch.no_grad():
        z = model(torch.from_numpy((X / meta["x_scale"][None, :, None]).astype(np.float32))).numpy()
    return meta["nominal"] * np.exp(z * meta["t_sd"] + meta["t_mu"])

def load_model(path):
    ck = torch.load(path, weights_only=False)
    meta = ck["meta"]
    model = ParamNet(meta["c_in"], len(meta["names"]), meta["width"])
    model.load_state_dict(ck["state"])
    return model, meta

def predict(model_path, X) -> list[dict]:
    """X: (C, T) or (N, C, T) recorded with the same experiment/channels as the training data."""
    model, meta = load_model(model_path)
    X = np.asarray(X, np.float32)
    if X.ndim == 2:
        X = X[None]
    if X.shape[-1] != meta["T"]:
        raise ValueError(f"expected {meta['T']} samples (Ts={meta['Ts']}), got {X.shape[-1]}")
    y = _predict(model, meta, X)
    return [dict(zip(meta["names"], map(float, row))) for row in y]