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


def build_window(seasons, cfg, phases=("RS",), gt_weight=None, margin_bins=False, target="pts"):
    stints = pd.concat([load_stints(s, p, cfg) for s in seasons for p in phases], ignore_index=True)
    box = season_box(seasons, list(phases), cfg)
    return build_design(stints, box, cfg["features"], cfg, gt_weight=gt_weight, margin_bins=margin_bins,
                        target=target)


def window_label(seasons) -> str:
    return f"{seasons[0]}-{seasons[-1]}"


def window_mid(seasons) -> float:
    return float(np.mean(seasons))


def total_rows(est) -> pd.DataFrame:
    """Combined coefficient: what one unit per 100 is worth to a player overall, offense plus defense.

    In flipped space (positive = good on both sides) that is beta_O - beta_D_raw, so the standard
    error needs the cross-term: Var = Var(O) + Var(D) - 2 Cov(O, D).
    """
    nf = est.n_feat_
    cov = est.cov_beta_
    rows = []
    for j, f in enumerate(est.spec.features):
        c = np.zeros(2 * nf)
        c[j], c[nf + j] = 1.0, -1.0
        rows.append(dict(side="Total", feature=f, beta=est.beta_[j] - est.beta_[nf + j],
                         se=float(np.sqrt(max(c @ cov @ c, 0.0)))))
    return pd.DataFrame(rows)


def _rows(est, run, seasons, extra=None) -> pd.DataFrame:
    tb = est.coef_table(flip_defense=True)[["side", "feature", "beta", "se"]].copy()
    tb = pd.concat([tb, total_rows(est)], ignore_index=True)
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


def player_priced_beta(panel, features, exclude=(), min_poss=2000, pen=5.0) -> np.ndarray:
    """The box prior priced to predict a PLAYER, not a team.

    The published beta is fit by regressing stint margin on LINEUP SUMS, which prices the box score
    for the team.  This one regresses a player's own padded rates on his own NEXT-window pure
    on-court impact (`u / a`, the de-shrunk residual, weighted by the shrinkage `a`), so it prices
    the box score for the player.  Returns the model's raw 26-vector, offense then defense.

    `panel` is outputs/xrapm_panel.parquet from scripts/27_xrapm_prior.py: one row per player per
    window per side, with the 13 rates, `poss`, `u` and `a`.  `exclude` drops windows from the FIT
    (not the panel), which is how a window's prior is kept off its own data.

    No consensus is involved: the target is the model's own next-window on-court estimate.
    """
    windows = sorted(panel["window"].unique())
    nxt = {windows[i]: windows[i + 1] for i in range(len(windows) - 1)}
    exclude = set(exclude)
    out = []
    for side in ("O", "D"):
        a = panel[(panel.side == side) & (panel.poss >= min_poss)].copy()
        b = a[["window", "player_id", "u", "a"]].rename(
            columns={"window": "next", "u": "u_next", "a": "a_next"})
        a["next"] = a.window.map(nxt)
        d = a.merge(b, on=["next", "player_id"], how="inner")
        d = d[~d["next"].isin(exclude) & ~d.window.isin(exclude)]
        X = np.column_stack([np.ones(len(d)), d[features].to_numpy()])
        y = (d.u_next / np.maximum(d.a_next, 1e-9)).to_numpy()
        w = d.a_next.to_numpy()
        p = pen * np.eye(X.shape[1])
        p[0, 0] = 0.0                                     # the intercept is not penalised
        out.append(np.linalg.solve((X * w[:, None]).T @ X + p, (X * w[:, None]).T @ y)[1:])
    return np.concatenate(out)


def hybrid_beta(panel, features, window_label_, min_poss=2000, pen=5.0) -> np.ndarray:
    """The shipped prior: player-priced on offense, ZERO on defense, leave-one-window-out.

    Defense gets no box prior at all because the box score has no defensive vocabulary -- the
    weighted R^2 of a player's 13 rates on his own defensive on-court impact is 0.26, and pure RAPM
    beats the box-primed defensive rating 0.877 to 0.755 (FINDINGS.md, scripts/30_ladder.py).  The
    box term enters the model as an offset `Xbox @ beta` with separate offensive and defensive
    columns, so zeroing the defensive half IS "no defensive prior", in one fit at one penalty.

    `window_label_` is excluded from the beta fit.  The fit's rows are (window X rates -> window Y
    impact) pairs, and `player_priced_beta` drops a pair if EITHER end is excluded, so one label is
    enough to keep the window off both sides of its own prior.
    """
    beta = player_priced_beta(panel, features, exclude=[window_label_], min_poss=min_poss, pen=pen)
    nf = len(features)
    return np.concatenate([beta[:nf], np.zeros(nf)])


def player_ratings_table(wd, beta, cfg, seasons, beta_po=None, names=None,
                         prior_offset=None) -> pd.DataFrame:
    """Per player-season ratings from the plug-in fit at the fixed lambda.

    prior  = the player's padded full-season rates times the window's coefficients, centred on the
             average player (the exposure columns are centred on the sum of five, so divide by five)
    u      = what the ridge finds beyond the box score, and beyond `prior_offset` when one is given
    rapm_mm = prior + the residual fit with NO offset: what the model says without the boosted
             correction, kept so the with and without versions stay comparable
    Defense is flipped so positive = good; total is offense plus defense.

    `prior_offset` is the per-Z-column boosted correction in raw sign.  Passing it makes `u` the
    residual refit *given* the correction, which is what scripts/10_boost.py ships.  Adding a
    correction on top of a residual that was fit without it double-counts whatever it explains.
    """
    kw = dict(lam=float(cfg["lam_plugin"]), lam_ratio=float(cfg["lam_ratio_plugin"]),
              pad_target=cfg["pad_target"])
    plain = plugin_fit(wd, beta, **kw)
    pipe = plain if prior_offset is None else plugin_fit(wd, beta, prior_offset=prior_offset, **kw)
    exp, mm = pipe["exposure"], pipe["mm"]
    u_plain = plain["mm"].u_
    nf = mm.n_feat_
    m = wd.spec.n_ps
    ro = exp.season_rates_ - exp.means_o_ / 5.0
    rd = exp.season_rates_d_ - exp.means_d_ / 5.0
    df = wd.spec.ps_table.copy()
    df["window"] = window_label(seasons)
    df["window_mid"] = window_mid(seasons)
    df["poss_off"] = exp.season_poss_off_
    df["poss_def"] = exp.season_poss_def_
    df["prior_off"] = ro @ beta[:nf]
    df["prior_def"] = -(rd @ beta[nf:])
    df["u_off"] = mm.u_[:m]
    df["u_def"] = -mm.u_[m:]
    df["u_plain_off"] = u_plain[:m]
    df["u_plain_def"] = -u_plain[m:]
    if beta_po is not None:
        df["prior_off_po"] = ro @ beta_po[:nf]
        df["prior_def_po"] = -(rd @ beta_po[nf:])
    for part in ("prior", "u", "u_plain"):
        df[f"{part}_total"] = df[f"{part}_off"] + df[f"{part}_def"]
    for side in ("off", "def", "total"):
        df[f"rapm_mm_{side}"] = df[f"prior_{side}"] + df[f"u_plain_{side}"]
    if beta_po is not None:
        df["prior_total_po"] = df.prior_off_po + df.prior_def_po
        for side in ("off", "def", "total"):
            df[f"rapm_mm_{side}_po"] = df[f"prior_{side}_po"] + df[f"u_plain_{side}"]
    if names is not None:
        if "season" in df.columns and "season" in names.columns:
            df = df.merge(names, on=["player_id", "season"], how="left")
        else:
            # player-window units: take the name from the season he played most
            gp = wd.game_poss.merge(wd.spec.psx_table[["psx_idx", "player_id", "season"]], on="psx_idx", how="left")
            poss = gp.groupby(["player_id", "season"], as_index=False)["poss_off"].sum()
            pick = poss.sort_values("poss_off").drop_duplicates("player_id", keep="last")
            nm = pick.merge(names, on=["player_id", "season"], how="left")[["player_id", "player_name"]]
            df = df.merge(nm, on="player_id", how="left")
    df["shrinkage"] = np.where(df.poss_off > 0, df.poss_off / (df.poss_off + float(cfg["lam_plugin"])), 0.0)
    return df.drop(columns=["ps_idx", "psx_idx"], errors="ignore")


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


# ---------------------------------------------------------------- LRBoost
def boost_panels(cfg, windows=None, lam=None, lam_ratio=None, lam_plugin=None, ratio_plugin=None,
                 verbose=True):
    """Per-window linear fit plus the player panel the booster trains on.

    Returns (panels, per_window) where per_window maps the window label to the pieces needed to
    rebuild its prior: beta, the fitted exposure, the plug-in fit and the design.
    """
    from .boost import dominant_team, player_panel
    windows = window_seasons(cfg) if windows is None else windows
    lam = float(cfg["lam_beta"]) if lam is None else float(lam)
    lam_ratio = float(cfg["lam_ratio_beta"]) if lam_ratio is None else float(lam_ratio)
    lam_plugin = float(cfg["lam_plugin"]) if lam_plugin is None else float(lam_plugin)
    ratio_plugin = float(cfg["lam_ratio_plugin"]) if ratio_plugin is None else float(ratio_plugin)
    panels, per_window = [], {}
    t0 = time.time()
    for w in windows:
        seasons = list(range(w[0], w[1] + 1))
        lab = window_label(seasons)
        wd = build_window(seasons, cfg)
        cf = crossfit_beta(wd, lam=lam, lam_ratio=lam_ratio, pad_target=cfg["pad_target"])
        pipe = plugin_fit(wd, cf.beta_, lams=[lam_plugin], cv=2, lam_ratio=ratio_plugin,
                          pad_target=cfg["pad_target"])
        grp = dominant_team(wd, season_box(seasons, ["RS"], cfg))
        panels.append(player_panel(wd, cf.beta_, cfg, window=lab, groups=grp, pipe=pipe,
                                   lam=lam_plugin, lam_ratio=ratio_plugin))
        per_window[lab] = dict(seasons=seasons, wd=wd, beta=cf.beta_, cov_beta=cf.cov_beta_,
                               pipe=pipe, groups=grp)
        if verbose:
            print(f"  panel {lab}: {wd.spec.n_ps} players ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(panels, ignore_index=True), per_window


def fit_pooled_boost(panel, cfg, group_by="player", hold_out=None, **kw):
    """One booster pooled over every window, cross-fitted so a player never scores himself.

    `group_by="player"` folds on the player id, so a player recurring across windows never lands on
    both sides of a split; that is the channel that would inflate the shrinkage factor spuriously.
    `group_by="team"` folds on team-window instead, which also separates teammates but lets a player
    train on his own later self.  The next-window test in the study harness is the honest guard.
    """
    from .boost import fit_boost
    p = panel.copy()
    if hold_out is not None:                      # leave-one-window-out, for an honest gate
        p = p[p.window != hold_out].copy()
    if group_by == "player":
        p["group"] = p["player_id"].to_numpy()
    elif group_by == "team":
        p["group"] = p["window"].astype(str) + ":" + p["group"].astype(str)
    kw.setdefault("min_poss", float(cfg.get("boost_min_poss", 4500)))
    kw.setdefault("quality", cfg.get("boost_quality"))
    nui = cfg.get("boost_nuisance")
    if nui is not None:
        kw.setdefault("nuisance", {k: tuple(v or ()) for k, v in nui.items()})
    return fit_boost(p, cfg["features"], **kw)
