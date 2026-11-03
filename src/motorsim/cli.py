"""Command-line interface:  python run.py <command> -c config/xxx.yaml [...]"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / "export"

def _out(args, tag, cfg=None):
    if args.out:
        return Path(args.out)
    name = (cfg or {}).get("name", "run")
    return EXPORT / f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}_{name}"

def _load(args):
    from .config import load_config
    return load_config(args.config)

def _data(cfg, path):
    from .data import load_data
    spec = dict(cfg.get("measurement") or {})
    if path is None and "file" not in spec:
        raise SystemExit("no data file: pass --data or set measurement.file in the config")
    if path is not None and Path(path).suffix == ".csv" and "columns" not in spec:
        spec = {}                       # a simulator CSV: auto-detected columns
    return load_data(path, spec)

def cmd_simulate(args):
    from .plotting import plot_sim
    from .simulator import simulate
    cfg = _load(args)
    if args.mode:
        cfg["control"]["mode"] = args.mode
    out = _out(args, "sim", cfg)
    t0 = time.time()
    res = simulate(cfg)
    print(f"simulated {res.N} x {len(res.t)} steps in {time.time() - t0:.1f} s")
    res.save(out / "sim.npz")
    res.save(out / "sim.csv")
    plot_sim(res, out / "sim.png")
    print(f"saved -> {out}")

def cmd_dataset(args):
    from .dataset import generate_dataset
    cfg = _load(args)
    dcfg = cfg.get("dataset", {})
    n = args.n or dcfg.get("n", 1000)
    out = Path(args.out) if args.out else EXPORT / f"dataset_{cfg.get('name', 'run')}.npz"
    generate_dataset(cfg, n, out, batch=dcfg.get("batch", 256), decimate=dcfg.get("decimate", 1))
    print(f"saved -> {out}")

def cmd_train(args):
    from .dataset import load_dataset
    from .learn import load_model, _predict, train
    from .plotting import plot_learning
    out = Path(args.out) if args.out else Path(args.dataset).with_suffix(".pt")
    path, table = train(args.dataset, out, epochs=args.epochs, batch=args.batch, lr=args.lr)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    table.to_csv(path.with_suffix(".metrics.csv"), index=False)
    model, meta = load_model(path)
    ds = load_dataset(args.dataset)
    k = min(len(ds["X"]), 1000)
    plot_learning(ds["y"][-k:], _predict(model, meta, ds["X"][-k:]), meta["names"], path.with_suffix(".png"))
    print(f"saved -> {path}")

def channels_from_data(data: dict, channels):
    from .ident import mechanical_signals
    rows = []
    for c in channels:
        if c == "w_enc":
            rows.append(mechanical_signals(data["t"], data.get("theta"), data.get("w"))[0])
        else:
            rows.append(data[c.replace("_meas", "")])
    return np.stack(rows)

def cmd_predict(args):
    from .learn import load_model, predict
    cfg = _load(args)
    data = _data(cfg, args.data)
    _, meta = load_model(args.model)
    X = channels_from_data(data, meta["channels"])[:, : meta["T"]]
    for k, v in predict(args.model, X)[0].items():
        print(f"{k:8s} {v:.6g}")

def cmd_identify(args):
    from .pipeline import friction_map_points, identify
    cfg = _load(args)
    out = _out(args, "ident", cfg)
    data = _data(cfg, args.data)
    val = _data(cfg, args.val) if args.val else None
    fmap = friction_map_points(_data(cfg, args.fmap), cfg) if args.fmap else None
    res = identify(cfg, data, out, data_val=val, fmap=fmap)
    print(res["ls"].table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"report -> {out / 'report.md'}")

def cmd_friction_map(args):
    from .ident import compare_models
    from .pipeline import friction_map_points, run_friction_map, speed_levels
    from .plotting import plot_friction_fits
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
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"saved -> {out}")

def cmd_validate(args):
    from .config import load_params
    from .plotting import plot_overlay
    from .validate import validate
    cfg = _load(args)
    out = _out(args, "val", cfg)
    doc = load_params(args.params)
    data = _data(cfg, args.data)
    m, pairs = validate(cfg, data, doc["params"], doc.get("friction_model"))
    out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "validation_metrics.csv", index=False)
    plot_overlay(data["t"], pairs, out / "validation_overlay.png", title=f"measured vs model ({args.params})")
    print(m.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"saved -> {out}")

def cmd_demo(args):
    from .pipeline import run_demo
    cfg = _load(args)
    out = _out(args, "demo", cfg)
    res = run_demo(cfg, out)
    print(f"report -> {res['out'] / 'report.md'}")

def main(argv=None):
    ap = argparse.ArgumentParser(prog="motorsim", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_, cfg=True):
        p = sub.add_parser(name, help=help_)
        if cfg:
            p.add_argument("-c", "--config", required=True)
        p.add_argument("-o", "--out", default=None)
        p.set_defaults(fn=fn)
        return p

    p = add("simulate", cmd_simulate, "run one (batched) closed-loop simulation")
    p.add_argument("--mode", choices=["voltage", "current", "torque", "speed", "position"])
    p = add("dataset", cmd_dataset, "generate a randomized-parameter training dataset")
    p.add_argument("-n", type=int, default=None)
    p = add("train", cmd_train, "train the CNN parameter estimator", cfg=False)
    p.add_argument("-d", "--dataset", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p = add("predict", cmd_predict, "estimate parameters of a recording with a trained network")
    p.add_argument("-m", "--model", required=True)
    p.add_argument("--data", default=None)
    p = add("identify", cmd_identify, "LS + output-error identification from a recording")
    p.add_argument("--data", default=None)
    p.add_argument("--val", default=None, help="independent validation recording")
    p.add_argument("--fmap", default=None, help="staircase constant-speed recording (needs ref column)")
    p = add("friction-map", cmd_friction_map, "constant-speed friction map + model comparison")
    p.add_argument("--data", default=None)
    p = add("validate", cmd_validate, "replay a recording through identified parameters")
    p.add_argument("-p", "--params", required=True)
    p.add_argument("--data", default=None)
    add("demo", cmd_demo, "synthetic end-to-end: hidden true motor -> identify -> validate")

    args = ap.parse_args(argv)
    args.fn(args)

if __name__ == "__main__":
    main()