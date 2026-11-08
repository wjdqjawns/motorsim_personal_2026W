"""End-to-end workflows: identification pipeline and the synthetic demo."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from motorsim.core.analysis import compare_models, fit_output_error, identify_ls, param_error_table, validate
from motorsim.core.config import flat_params, load_config, nominal_params, torque_constant
from motorsim.core.io import from_frame, save_params, slice_data
from motorsim.core.model.friction import model_params
from motorsim.core.plot import plot_foc, plot_foc_zoom, plot_friction_fits, plot_overlay, plot_sim
from motorsim.core.sim.simulator import simulate
from motorsim.domains import get_domain
from motorsim.pipeline.experiments import experiment_config, run_friction_map, speed_levels
from motorsim.pipeline.exporters import md_table, write_report
from motorsim.utils.logger import get_logger

_log = get_logger(__name__).info


def identify(cfg, data_id: dict, out_dir: str | Path, data_val: dict | None = None,
             fmap: tuple | None = None, truth: dict | None = None, log=_log) -> dict:
    """LS -> friction-model selection -> output-error refinement -> validation -> report."""
    cfg = load_config(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mtype = cfg["motor"]["type"]
    dom = get_domain(mtype)
    icfg = cfg.get("identification", {})
    models = icfg.get("friction_models", ["viscous", "coulomb", "coulomb_viscous", "stribeck"])
    nom = nominal_params(cfg)
    report = [f"# Identification report: {cfg.get('name', '')}", "",
              f"- motor: **{mtype}**, samples: {len(data_id['t'])}, Ts = {data_id['t'][1] - data_id['t'][0]:.3g} s",
              f"- created: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]

    # 1) equation-error LS
    log("  LS identification ...")
    ls = identify_ls(data_id, mtype, friction_models=models, pole_pairs=nom.get("p"),
                     **icfg.get("ls", {}))
    ls.table.to_csv(out / "ls_mechanical_models.csv", index=False)
    report += ["## 1. Least-squares (equation error)", "",
               "Electrical: " + ", ".join(f"{k} = {v:.5g}" for k, v in ls.electrical.items()),
               f" (R^2 = {ls.diagnostics['r2_electrical']:.5f})", "",
               "Mechanical equation  Te = J dw/dt + T_f(w), candidate friction models:", "",
               ls.table.pipe(md_table), ""]

    # 2) friction map (optional). Torque = Kt * mean current, with Kt from step 1 (not nominal).
    fmap_table = None
    if fmap is not None:
        w, T = fmap                                           # T computed with the nominal Kt
        kscale = torque_constant(mtype, ls.electrical) / torque_constant(mtype, nom)
        T = T * kscale
        fmap_table, fmap_fits = compare_models(w, T, models=models, offset=icfg.get("offset", False))
        plot_friction_fits(w, T, fmap_fits, out / "friction_map.png")
        fmap_table.to_csv(out / "friction_map_models.csv", index=False)
        report += ["## 2. Friction map (constant-speed test)", "",
                   f"Torque = identified Kt x mean current at each settled speed set-point "
                   f"(Kt_identified / Kt_nominal = {kscale:.4f}).", "",
                   fmap_table.pipe(md_table), "", "![](friction_map.png)", ""]

    # model choice: friction-map BIC if available (dedicated low-speed data), else LS BIC
    chosen = icfg.get("friction_model") or (fmap_table.iloc[0]["model"] if fmap_table is not None else ls.best_model)
    p_ls = ls.params_for(chosen if chosen in ls.fits else ls.best_model)
    note = ""
    if fmap is not None and chosen in ("stribeck", "lugre") and chosen in fmap_fits:
        for k in ("Tst", "ws"):            # Stribeck shape lives at low speed -> trust the map
            p_ls[k] = fmap_fits[chosen].params[k]
        note = " Stribeck shape (Tst, ws) taken from the friction map."
    report += [f"**Selected friction model: `{chosen}`** "
               f"({'given in config' if icfg.get('friction_model') else 'lowest BIC'}).{note}", ""]

    est = {"LS": p_ls}
    oe_cfg = icfg.get("oe", {})
    oe_model = oe_cfg.get("friction_model", chosen)
    if oe_cfg.get("enabled", True):
        d_oe = slice_data(data_id, 0.0, oe_cfg.get("duration", 0.6))
        # Stribeck shape (Tst, ws) is poorly excited in a fast-moving record -> leave it to LS /
        # the friction map unless the config explicitly asks OE to fit it.
        fit = oe_cfg.get("fit") or (dom.ELEC_PARAMS + ["J"] + [n for n in model_params(oe_model)
                                                          if n not in ("Tst", "ws")])
        init = dict(p_ls)
        if oe_model == "lugre":
            init.setdefault("sigma0", nom["sigma0"])
            init.setdefault("sigma1", nom["sigma1"])
        log(f"  output-error refinement ({oe_model}: {', '.join(fit)}) ...")
        t0 = time.time()
        oe = fit_output_error(cfg, d_oe, init, fit, oe_model, decimate=oe_cfg.get("decimate", 2),
                              max_nfev=oe_cfg.get("max_nfev", 15), substeps=oe_cfg.get("substeps", 2))
        est["OE"] = {k: oe.params[k] for k in set(fit) | set(p_ls)}
        report += ["## 3. Output-error refinement", "",
                   f"Replay measured voltage in the open-loop model; fit {', '.join(fit)} "
                   f"(friction `{oe_model}`). cost {oe.cost0:.4g} -> {oe.cost:.4g}, "
                   f"{oe.nfev} evaluations, {time.time() - t0:.0f} s.", ""]
    # 4) validation on a separate data set -> also decides which estimate is kept
    models_used = {"LS": chosen, "OE": oe_model, "truth": cfg["friction"]["model"]}
    final_key = "OE" if "OE" in est else "LS"
    if data_val is not None:
        rows, overlays = [], {}
        est_v = {"truth": truth, **est} if truth is not None else est
        for label, p in est_v.items():
            m, overlays[label] = validate(cfg, data_val, p, models_used[label])
            for _, r in m.iterrows():
                rows.append({"params": label, **r.to_dict()})
        vt = pd.DataFrame(rows)
        vt.to_csv(out / "validation_metrics.csv", index=False)
        # id is ~0 under id*=0 control (fit% = noise) -> score on torque-producing current + speed
        sel = vt.params.isin(list(est)) & vt.channel.isin(["i", "iq", "w"])
        score = vt[sel].groupby("params")["fit_pct"].mean()
        final_key = str(score.idxmax())
        plot_overlay(data_val["t"], overlays[final_key], out / "validation_overlay.png",
                     title=f"validation: measured vs model ({final_key}, friction {models_used[final_key]})")
        report += ["## 4. Validation (independent experiment, voltage replay)", "",
                   "fit_pct = 100 (1 - |y - y_model| / |y - mean(y)|).  `truth` = replay with the true "
                   "(hidden) model: the best achievable given sensor noise.", "",
                   vt.pipe(md_table), "",
                   f"**Kept estimate: {final_key}** (highest mean validation fit of i/iq and w among {', '.join(est)})", "",
                   "![](validation_overlay.png)", ""]
    save_params(out / "identified_params.yaml", mtype, models_used[final_key], est[final_key],
                {"method": final_key, "source": cfg.get("name", "")})

    # 5) ground-truth comparison (synthetic data only)
    if truth is not None:
        names = dom.ELEC_PARAMS + ["J"] + model_params(cfg["friction"]["model"])
        et = param_error_table(truth, est, names)
        et.to_csv(out / "param_errors.csv", index=False)
        report += ["## 5. Parameter error vs. ground truth", "",
                   f"True friction model: `{cfg['friction']['model']}`", "",
                   et.pipe(md_table), ""]

    write_report(out / "report.md", report)
    return {"ls": ls, "estimates": est, "chosen": chosen, "final": final_key, "out": out}


# --------------------------------------------------------------------------- synthetic demo
def run_demo(cfg, out_dir: str | Path, log=_log) -> dict:
    """Pseudo-real experiment on a hidden 'true' motor, then identify & validate it."""
    cfg = load_config(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    exps = cfg.get("experiments", {})
    truth = flat_params(cfg)

    log("  simulating identification experiment ...")
    r_id = simulate(experiment_config(cfg, exps.get("identification", {})))
    r_id.save(out / "exp_identification.csv")
    plot_sim(r_id, out / "exp_identification.png")
    if "ia" in r_id.data:                       # 3-phase model: also show the FOC chain
        plot_foc(r_id, out / "exp_identification_foc.png")
        plot_foc_zoom(r_id, out / "exp_identification_foc_zoom.png")
    log("  simulating validation experiment ...")
    r_val = simulate(experiment_config(cfg, exps.get("validation", {"seed": 123})))
    r_val.save(out / "exp_validation.csv")
    fmap = None
    if "friction_map" in exps:
        log("  simulating friction-map experiment ...")
        fm = exps["friction_map"]
        r_fm, w, T = run_friction_map(cfg, speed_levels(fm.get("speeds", {})), fm.get("hold", 0.3))
        r_fm.save(out / "exp_friction_map.csv")
        fmap = (w, T)
    data_id = from_frame(r_id.frame(0), None)
    data_val = from_frame(r_val.frame(0), None)
    return identify(cfg, data_id, out, data_val=data_val, fmap=fmap, truth=truth, log=log)
