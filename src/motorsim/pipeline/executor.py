"""CLI command implementations (registered in pipeline.registry)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from motorsim.pipeline.registry import arg, command
from motorsim.utils.logger import get_logger

log = get_logger(__name__).info
ROOT = Path(__file__).resolve().parents[3]
EXPORT = ROOT / "export"
MODES = ["voltage", "current", "torque", "speed", "position", "vf", "abc"]

def _out(args, tag, cfg=None) -> Path:
    if args.out:
        return Path(args.out)
    return EXPORT / f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}_{(cfg or {}).get('name', 'run')}"

def _load(args) -> dict:
    from motorsim.core.config import load_config
    return load_config(args.config)

def _data(cfg, path) -> dict:
    from motorsim.core.io import load_data
    spec = dict(cfg.get("measurement") or {})
    if path is None and "file" not in spec:
        raise SystemExit("no data file: pass --data or set measurement.file in the config")
    if path is not None and Path(path).suffix == ".csv" and "columns" not in spec:
        spec = {}                       # a simulator CSV: auto-detected columns
    return load_data(path, spec)

def _table(df) -> str:
    return df.to_string(index=False, float_format=lambda v: f"{v:.4g}")

@command("simulate", "run one (batched) closed-loop simulation", args=[
    arg("--mode", choices=MODES),
    arg("--frame", choices=["dq", "abc"], help="PMSM model frame"),
    arg("--inverter", choices=["average", "switching"], help="PMSM abc inverter model"),
])
def cmd_simulate(args):
    from motorsim.core.plot import plot_foc, plot_foc_zoom, plot_sim
    from motorsim.core.sim import simulate
    cfg = _load(args)
    if args.mode:
        cfg["control"]["mode"] = args.mode
    if args.frame:
        cfg["motor"]["frame"] = args.frame
    if args.inverter:
        cfg["motor"].setdefault("inverter", {})["type"] = args.inverter
    out = _out(args, "sim", cfg)
    t0 = time.time()
    res = simulate(cfg)
    log(f"simulated {res.N} x {len(res.t)} steps in {time.time() - t0:.1f} s")
    res.save(out / "sim.npz")
    res.save(out / "sim.csv")
    plot_sim(res, out / "sim.png")
    if res.fast is not None:
        res.fast.to_csv(out / "sim_fast.csv", index=False)
    if "ia" in res.data:
        plot_foc(res, out / "foc.png")
        plot_foc_zoom(res, out / "foc_zoom.png")
    log(f"saved -> {out}")


@command("dataset", "generate a randomized-parameter training dataset", args=[arg("-n", type=int, default=None)])
def cmd_dataset(args):
    from motorsim.core.learning import generate_dataset
    cfg = _load(args)
    dcfg = cfg.get("dataset", {})
    out = Path(args.out) if args.out else EXPORT / f"dataset_{cfg.get('name', 'run')}.npz"
    generate_dataset(cfg, args.n or dcfg.get("n", 1000), out, batch=dcfg.get("batch", 256),
                     decimate=dcfg.get("decimate", 1))
    log(f"saved -> {out}")


@command("train", "train the CNN parameter estimator", needs_config=False, args=[
    arg("-d", "--dataset", required=True),
    arg("--epochs", type=int, default=40),
    arg("--batch", type=int, default=64),
    arg("--lr", type=float, default=2e-3),
])
def cmd_train(args):
    from motorsim.core.learning.dataset import load_dataset
    from motorsim.core.learning.trainer import _predict, load_model, train
    from motorsim.core.plot import plot_learning
    out = Path(args.out) if args.out else Path(args.dataset).with_suffix(".pt")
    path, table = train(args.dataset, out, epochs=args.epochs, batch=args.batch, lr=args.lr)
    log(_table(table))
    table.to_csv(path.with_suffix(".metrics.csv"), index=False)
    model, meta = load_model(path)
    ds = load_dataset(args.dataset)
    k = min(len(ds["X"]), 1000)
    plot_learning(ds["y"][-k:], _predict(model, meta, ds["X"][-k:]), meta["names"], path.with_suffix(".png"))
    log(f"saved -> {path}")

def channels_from_data(data: dict, channels):
    from motorsim.core.analysis import mechanical_signals
    rows = []
    for c in channels:
        if c == "w_enc":
            rows.append(mechanical_signals(data["t"], data.get("theta"), data.get("w"))[0])
        else:
            rows.append(data[c.replace("_meas", "")])
    return np.stack(rows)

@command("predict", "estimate parameters of a recording with a trained network", args=[
    arg("-m", "--model", required=True), arg("--data", default=None)])
def cmd_predict(args):
    from motorsim.core.learning.trainer import load_model, predict
    cfg = _load(args)
    data = _data(cfg, args.data)
    _, meta = load_model(args.model)
    X = channels_from_data(data, meta["channels"])[:, : meta["T"]]
    for k, v in predict(args.model, X)[0].items():
        log(f"{k:8s} {v:.6g}")

@command("identify", "LS + output-error identification from a recording", args=[
    arg("--data", default=None),
    arg("--val", default=None, help="independent validation recording"),
    arg("--fmap", default=None, help="staircase constant-speed recording (needs ref column)"),
])
def cmd_identify(args):
    from motorsim.pipeline.experiments import friction_map_points
    from motorsim.pipeline.runner import identify
    cfg = _load(args)
    out = _out(args, "ident", cfg)
    data = _data(cfg, args.data)
    val = _data(cfg, args.val) if args.val else None
    fmap = friction_map_points(_data(cfg, args.fmap), cfg) if args.fmap else None
    res = identify(cfg, data, out, data_val=val, fmap=fmap)
    log(_table(res["ls"].table))
    log(f"report -> {out / 'report.md'}")


@command("friction-map", "constant-speed friction map + model comparison", args=[arg("--data", default=None)])
def cmd_friction_map(args):
    from motorsim.core.analysis import compare_models
    from motorsim.core.plot import plot_friction_fits
    from motorsim.pipeline.experiments import friction_map_points, run_friction_map, speed_levels
    cfg = _load(args)
    out = _out(args, "fmap", cfg)
    if args.data:
        w, T = friction_map_points(_data(cfg, args.data), cfg)
    else:
        fm = cfg.get("experiments", {}).get("friction_map", {})
        res, w, T = run_friction_map(cfg, speed_levels(fm.get("speeds", {})), fm.get("hold", 0.3))
        res.save(out / "exp_friction_map.csv")
    models = cfg.get("identification", {}).get("friction_models",
                                               ["viscous", "coulomb", "coulomb_viscous", "stribeck"])
    table, fits = compare_models(w, T, models=models)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "friction_map_models.csv", index=False)
    plot_friction_fits(w, T, fits, out / "friction_map.png")
    log(_table(table))
    log(f"saved -> {out}")


@command("validate", "replay a recording through identified parameters", args=[
    arg("-p", "--params", required=True), arg("--data", default=None)])
def cmd_validate(args):
    from motorsim.core.analysis import validate
    from motorsim.core.io import load_params
    from motorsim.core.plot import plot_overlay
    cfg = _load(args)
    out = _out(args, "val", cfg)
    doc = load_params(args.params)
    data = _data(cfg, args.data)
    m, pairs = validate(cfg, data, doc["params"], doc.get("friction_model"))
    out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "validation_metrics.csv", index=False)
    plot_overlay(data["t"], pairs, out / "validation_overlay.png", title=f"measured vs model ({args.params})")
    log(_table(m))
    log(f"saved -> {out}")


@command("demo", "synthetic end-to-end: hidden true motor -> identify -> validate")
def cmd_demo(args):
    from motorsim.pipeline.runner import run_demo
    cfg = _load(args)
    out = _out(args, "demo", cfg)
    res = run_demo(cfg, out)
    log(f"report -> {res['out'] / 'report.md'}")
