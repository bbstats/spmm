"""Loop the windows: two half-fits -> beta, plug-in u fit, robustness runs -> coefs.parquet.

Runs (column `run` in outputs/coefs.parquet):
  base        the ten non-overlapping 3-season windows at the fixed lambda
  lam_x3      same design, lambda x 3 (set_lam on the stored eigen path, no refit)
  lam_div3    same design, lambda / 3
  rolling     3-season windows stepping 1 year
  gt_half     gt_weight 0.5
  gt_zero     gt_weight 0
  margin_bins 7 margin bins x 2 time halves instead of the linear rubber band
Also outputs/feature_corr.parquet (possession-weighted correlation of the 13 rates per window)
and outputs/variance.parquet (tau2 per side, zero-prior vs plug-in, per window).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .boxtable import season_box
from .checks import beta_table
from .config import resolve
from .cv import crossfit_beta, lambda_ratio_grid, plugin_fit
from .design import FEATURES, build_design
from .stints import build_season

_STINT_CACHE: dict = {}


def load_stints(season: int, phase: str, cfg) -> pd.DataFrame:
    key = (season, phase)
    if key not in _STINT_CACHE:
        _STINT_CACHE[key] = build_season(season, phase, cfg)[0]
    return _STINT_CACHE[key]


def window_seasons(cfg, rolling=False):
    """[(first, last), ...] for the ten non-overlapping windows, or the rolling ones."""
    if not rolling:
        return [tuple(w) for w in cfg["windows"]]
    first, last = cfg["first_season"], cfg["last_season"]
    step = int(cfg.get("rolling_step", 1))
    return [(s, s + 2) for s in range(first, last - 1, step)]


def build_window(seasons, cfg, phases=("RS",), gt_weight=None, margin_bins=False):
    stints = pd.concat([load_stints(s, p, cfg) for s in seasons for p in phases], ignore_index=True)
    box = season_box(seasons, list(phases), cfg)
    return build_design(stints, box, cfg["features"], cfg, gt_weight=gt_weight, margin_bins=margin_bins)


def window_label(seasons) -> str:
    return f"{seasons[0]}-{seasons[-1]}"


def window_mid(seasons) -> float:
    return float(np.mean(seasons))


def _rows(est, run, seasons, extra=None) -> pd.DataFrame:
    tb = est.coef_table(flip_defense=True)[["side", "feature", "beta", "se"]].copy()
    tb["run"] = run
    tb["window"] = window_label(seasons)
    tb["window_mid"] = window_mid(seasons)
    tb["first_season"] = seasons[0]
    tb["last_season"] = seasons[-1]
    for k, v in (extra or {}).items():
        tb[k] = v
    return tb


def fit_beta(wd, cfg, lam=None, lam_ratio=None, lam_scale=(1.0,), lam_buckets=None, features=None,
             include_po=False, lam_delta=None, mask=None):
    """Cross-fitted beta at one or more lambda multiples (one moment build, set_lam per multiple)."""
    lam = float(cfg["lam_beta"]) if lam is None else float(lam)
    ratio = float(cfg["lam_ratio_beta"]) if lam_ratio is None else float(lam_ratio)
    lams = sorted({lam * s for s in lam_scale})
    cf = crossfit_beta(wd, lams=lams, cv=2, lam_ratio=ratio, lam_buckets=lam_buckets, features=features,
                       include_po=include_po, lam_delta=lam_delta, mask=mask, pad_target=cfg["pad_target"])
    out = {}
    for s in lam_scale:
        for h in ("A", "B"):
            cf.fits[h]["mm"].set_lam(lam * s)
        a, b = cf.fits["A"]["mm"], cf.fits["B"]["mm"]
        r = type(cf)(beta_=0.5 * (a.beta_ + b.beta_), beta_se_=None, cov_beta_=0.25 * (a.cov_beta_ + b.cov_beta_),
                     delta_=0.5 * (a.delta_ + b.delta_) if len(a.delta_) else np.zeros(0),
                     delta_se_=0.5 * np.sqrt(a.delta_se_ ** 2 + b.delta_se_ ** 2) if len(a.delta_) else np.zeros(0),
                     n_feat_=a.n_feat_, spec=cf.spec, fits=cf.fits, lam={"A": a.lam_, "B": b.lam_})
        r.beta_se_ = np.sqrt(np.maximum(np.diag(r.cov_beta_), 0.0))
        out[s] = r
    return out


def feature_correlations(wd, cfg, seasons) -> pd.DataFrame:
    """Possession-weighted correlation of the 13 padded season rates across player-seasons."""
    from .cv import make_exposure
    exp = make_exposure(wd, mode="full", pad_target=cfg["pad_target"]).fit(wd.X, sample_weight=wd.w)
    R, w = exp.season_rates_, exp.season_poss_off_
    ok = w > 0
    R, w = R[ok], w[ok]
    mu = np.average(R, axis=0, weights=w)
    C = ((R - mu) * w[:, None]).T @ (R - mu) / w.sum()
    d = np.sqrt(np.diag(C))
    corr = C / np.outer(d, d)
    feats = cfg["features"]
    df = pd.DataFrame([{"window": window_label(seasons), "window_mid": window_mid(seasons),
                        "feature_i": feats[i], "feature_j": feats[j], "corr": corr[i, j], "sd_i": d[i]}
                       for i in range(len(feats)) for j in range(len(feats))])
    return df


def share_explained(wd, beta, cfg, lams, ratios_zero=(0.5, 0.75, 1.0, 1.5, 2.0), ratios_plug=(0.12, 0.18, 0.25, 0.33, 0.5),
                    seasons=None) -> pd.DataFrame:
    """tau2 per side for the zero-prior and plug-in fits (REML per window) and 1 - tau2_1/tau2_0."""
    from .cv import make_pipeline
    out = {}
    for name, ratios in (("zero_prior", ratios_zero), ("plugin", ratios_plug)):
        fits = {}

        def fit_fn(r, name=name, fits=fits):
            if name == "zero_prior":
                p = make_pipeline(wd, lams=lams, cv=2, lam_ratio=r, features=[], mode="full", pad_target=cfg["pad_target"])
                p.fit(wd.X, wd.y, sample_weight=wd.w, groups=wd.groups)
            else:
                p = plugin_fit(wd, beta, lams=lams, cv=2, lam_ratio=r, pad_target=cfg["pad_target"])
            fits[r] = p
            return p["mm"]

        tab, lam, ratio, how = lambda_ratio_grid(fit_fn, list(ratios), lams, selection="reml")
        mm = fits[ratio]["mm"]
        mm.set_lam(lam)
        out[name] = dict(lam=lam, ratio=ratio, sigma2=float(mm.sigma2_), tau2_O=mm.sigma2_ / lam,
                         tau2_D=mm.sigma2_ / (lam * ratio))
    z, p = out["zero_prior"], out["plugin"]
    return pd.DataFrame([dict(window=window_label(seasons), window_mid=window_mid(seasons),
                              tau2_O_zero=z["tau2_O"], tau2_D_zero=z["tau2_D"], lam_zero=z["lam"], ratio_zero=z["ratio"],
                              tau2_O_plugin=p["tau2_O"], tau2_D_plugin=p["tau2_D"], lam_plugin=p["lam"], ratio_plugin=p["ratio"],
                              share_O=1 - p["tau2_O"] / z["tau2_O"], share_D=1 - p["tau2_D"] / z["tau2_D"])])


def run_all(cfg, runs=None, lams_var=None, verbose=True) -> dict:
    """Every run in `runs` (default: all seven) -> coefs / feature_corr / variance frames."""
    runs = runs or ["base", "lam_x3", "lam_div3", "rolling", "gt_half", "gt_zero", "margin_bins"]
    lams_var = np.geomspace(500, 60000, 20) if lams_var is None else lams_var
    coefs, corrs, var = [], [], []
    t0 = time.time()

    base_windows = window_seasons(cfg)
    need_base = {"base", "lam_x3", "lam_div3"} & set(runs)
    for w in base_windows:
        seasons = list(range(w[0], w[1] + 1))
        lab = window_label(seasons)
        if need_base or "gt_half" in runs or "gt_zero" in runs or "margin_bins" in runs:
            wd = build_window(seasons, cfg)
        if need_base:
            scales = [s for s, r in ((1.0, "base"), (3.0, "lam_x3"), (1 / 3, "lam_div3")) if r in runs]
            fits = fit_beta(wd, cfg, lam_scale=tuple(scales))
            for s, r in ((1.0, "base"), (3.0, "lam_x3"), (1 / 3, "lam_div3")):
                if r in runs:
                    coefs.append(_rows(fits[s], r, seasons, {"lam": float(cfg["lam_beta"]) * s}))
            corrs.append(feature_correlations(wd, cfg, seasons))
            var.append(share_explained(wd, fits[1.0].beta_ if 1.0 in fits else list(fits.values())[0].beta_,
                                       cfg, lams_var, seasons=seasons))
            if verbose:
                print(f"  {lab} base done ({time.time() - t0:.0f}s)", flush=True)
        for r, kw in (("gt_half", dict(gt_weight=0.5)), ("gt_zero", dict(gt_weight=0.0)), ("margin_bins", dict(margin_bins=True))):
            if r not in runs:
                continue
            wd2 = build_window(seasons, cfg, **kw)
            coefs.append(_rows(fit_beta(wd2, cfg)[1.0], r, seasons, {"lam": float(cfg["lam_beta"])}))
            if verbose:
                print(f"  {lab} {r} done ({time.time() - t0:.0f}s)", flush=True)

    if "rolling" in runs:
        for w in window_seasons(cfg, rolling=True):
            seasons = list(range(w[0], w[1] + 1))
            wd = build_window(seasons, cfg)
            coefs.append(_rows(fit_beta(wd, cfg)[1.0], "rolling", seasons, {"lam": float(cfg["lam_beta"])}))
            if verbose:
                print(f"  rolling {window_label(seasons)} done ({time.time() - t0:.0f}s)", flush=True)

    return dict(coefs=pd.concat(coefs, ignore_index=True) if coefs else pd.DataFrame(),
                feature_corr=pd.concat(corrs, ignore_index=True) if corrs else pd.DataFrame(),
                variance=pd.concat(var, ignore_index=True) if var else pd.DataFrame())


def write_outputs(res: dict, cfg, tag=""):
    out = resolve(cfg, "outputs")
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in res.items():
        if df is None or not len(df):
            continue
        p = out / (f"{name}{tag}.parquet" if name != "coefs" else f"coefs{tag}.parquet")
        if p.exists():
            old = pd.read_parquet(p)
            if "run" in df.columns and "run" in old.columns:
                old = old[~old["run"].isin(df["run"].unique())]
                df = pd.concat([old, df], ignore_index=True)
        df.to_parquet(p, index=False)
        paths[name] = p
    return paths
