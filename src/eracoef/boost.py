"""LRBoost: a gradient-boosted correction on top of the linear box-score prior.

The prior enters every row as a sum over the ten players on the floor, so any per-player function
slots into the mixed model as an offset (`Z @ prior_offset`).  That lets us keep the linear part
exactly as validated -- beta is fit first and frozen, so the coefficient-drift figure is untouched
-- and let a booster explain only what beta leaves behind.

Per window:
    1. cross-fitted beta                                   (unchanged, see cv.crossfit_beta)
    2. plug-in ridge fit with the linear prior as offset -> residual u, shrinkage a = diag(G(G+lam I)^-1)
    3. ChimeraBoost on the player panel, cross-fitted by team so a player's own residual and his
       teammates' never train the model that scores him
    4. refit u with (linear + boost) as the offset

Target and weights.  Var(u) = tau^2 G(G+lam I)^-1 exactly, so Var(u_i) = tau^2 a_i and the
de-shrunk target u_i/a_i has variance tau^2/a_i.  The weight is therefore a_i, which is also the
Fay-Herriot weight and is collinearity-aware: players whose effects are poorly identified by their
lineup patterns are downweighted automatically.  G(G+lam I)^-1 is not diagonal, so u_i/a_i also
carries teammate contamination amplified by 1/a_i; that, and the fact that the split-half
reliability of the target is near zero below about 1500 possessions, is why the booster trains only
on high-possession players.  It still *scores* everyone -- extrapolating a role-appropriate value to
a low-minute player is the entire point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .cv import plugin_fit

SIDES = ("O", "D")


def _wls_slope(x, y, w):
    """Weighted slope of y on x, through a weighted-centred fit."""
    mx = np.average(x, weights=w)
    my = np.average(y, weights=w)
    vx = np.average((x - mx) ** 2, weights=w)
    if vx <= 0:
        return 0.0
    return float(np.average((x - mx) * (y - my), weights=w) / vx)


def _winsorize(v, w, n_sd=4.0):
    mu = np.average(v, weights=w)
    sd = np.sqrt(np.average((v - mu) ** 2, weights=w))
    return np.clip(v, mu - n_sd * sd, mu + n_sd * sd)


def _strip_linear(g, X, w):
    """Residual of g after a possession-weighted linear fit on the rates (intercept included)."""
    A = np.column_stack([np.ones(len(g)), X])
    sw = np.sqrt(w)[:, None]
    b, *_ = np.linalg.lstsq(A * sw, g * np.sqrt(w), rcond=None)
    return g - A @ b


def player_panel(wd, beta, cfg, window="", lam=None, lam_ratio=None, groups=None,
                 pipe=None) -> pd.DataFrame:
    """One row per player per side: the 13 padded rates, the de-shrunk residual, and its weight.

    `groups` is a per-Z-unit label used for cross-fitting; pass the player's dominant team so
    teammates never straddle a fold.  Falls back to the player id.
    """
    lam = float(cfg["lam_plugin"]) if lam is None else float(lam)
    lam_ratio = float(cfg["lam_ratio_plugin"]) if lam_ratio is None else float(lam_ratio)
    if pipe is None:
        pipe = plugin_fit(wd, beta, lams=[lam], cv=2, lam_ratio=lam_ratio, pad_target=cfg["pad_target"])
    exp, mm = pipe["exposure"], pipe["mm"]
    a = mm.shrinkage_diag()
    m = wd.spec.n_ps
    feats = list(wd.spec.features)
    pid = wd.spec.ps_table["player_id"].to_numpy()
    grp = pid if groups is None else np.asarray(groups)
    out = []
    for side, sl, rates, poss in (("O", slice(0, m), exp.rates_, exp.season_poss_off_),
                                  ("D", slice(m, 2 * m), exp.rates_d_, exp.season_poss_def_)):
        ai = a[sl]
        u = mm.u_[sl]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(ai > 1e-9, u / np.where(ai > 1e-9, ai, 1.0), 0.0)
        df = pd.DataFrame(rates, columns=feats)
        df.insert(0, "window", window)
        df.insert(1, "ps_idx", np.arange(m))
        df.insert(2, "side", side)
        df.insert(3, "group", grp)
        df.insert(4, "player_id", pid)
        df["poss"] = poss
        df["poss_rank"] = pd.Series(poss).rank(pct=True).to_numpy()
        df["a"] = ai
        df["u"] = u
        df["target"] = t
        out.append(df)
    return pd.concat(out, ignore_index=True)


def score_panel(exp, spec, window="") -> pd.DataFrame:
    """The scoring half of a panel: rates, possessions and playing-time rank, no target.

    Built from a fitted exposure so a cross-validation fold scores players on the rates that
    fold actually saw, rather than on full-data rates that would leak the held-out games.
    """
    m = spec.n_ps
    feats = list(spec.features)
    out = []
    for side, rates, poss in (("O", exp.rates_, exp.season_poss_off_),
                              ("D", exp.rates_d_, exp.season_poss_def_)):
        df = pd.DataFrame(rates, columns=feats)
        df.insert(0, "window", window)
        df.insert(1, "ps_idx", np.arange(m))
        df.insert(2, "side", side)
        df["poss"] = poss
        df["poss_rank"] = pd.Series(poss).rank(pct=True).to_numpy()
        out.append(df)
    return pd.concat(out, ignore_index=True)


def fit_boost(panel: pd.DataFrame, features, min_poss=1500.0, n_folds=5, quality=None, seed=0,
              extra_features=(), nuisance=("poss_rank",), verbose=True) -> dict:
    """Cross-fitted ChimeraBoost per side, with a self-calibrating shrinkage factor.

    Returns models, out-of-fold predictions on the training rows, and `s`, the weighted slope of the
    target on the out-of-fold prediction.  Shipping `s * g` means a booster that fits noise gives
    `s ~ 0` and degrades gracefully to the linear prior.
    """
    from chimeraboost import ChimeraBoostRegressor

    base = list(features) + list(extra_features)
    nui_of = nuisance if isinstance(nuisance, dict) else {s: nuisance for s in SIDES}
    res = {"linear": base, "features": {}, "nuisance": {},
           "models": {}, "s": {}, "oof": {}, "n_train": {}, "ref": {}}
    for side in SIDES:
        nuisance_s = list(nui_of.get(side, ()))
        cols = base + nuisance_s
        res["features"][side], res["nuisance"][side] = cols, nuisance_s
        p = panel[panel.side == side]
        tr = p[(p.poss >= min_poss) & (p.a > 1e-9)].copy()
        if len(tr) < 200:
            res["models"][side], res["s"][side], res["n_train"][side] = None, 0.0, len(tr)
            continue
        w = tr["a"].to_numpy()
        y = _winsorize(tr["target"].to_numpy(), w)
        X = tr[cols].to_numpy(dtype=float)
        g = np.asarray(tr["group"])
        nui = [cols.index(c) for c in nuisance_s]
        oof = np.zeros(len(tr))
        nf = min(n_folds, len(np.unique(g)))
        for k, (i_tr, i_te) in enumerate(GroupKFold(nf).split(X, y, g)):
            mdl = ChimeraBoostRegressor(quality=quality, random_state=seed + k)
            mdl.fit(X[i_tr], y[i_tr], sample_weight=w[i_tr])
            # score the held-out fold the way boost_vector will: nuisance frozen at the training
            # reference.  Otherwise `s` credits the booster for a playing-time effect we then throw
            # away, and the shipped correction is scaled by a slope it did not earn.
            Xte = X[i_te].copy()
            for j in nui:
                Xte[:, j] = np.average(X[i_tr][:, j], weights=w[i_tr])
            oof[i_te] = mdl.predict(Xte)
        s = float(np.clip(_wls_slope(oof, y, w), 0.0, 1.0))
        full = ChimeraBoostRegressor(quality=quality, random_state=seed)
        full.fit(X, y, sample_weight=w)
        res["models"][side] = full
        res["ref"][side] = {c: float(np.average(tr[c].to_numpy(), weights=w)) for c in nuisance_s}
        res["s"][side] = s
        res["oof"][side] = pd.DataFrame({"window": tr.window.to_numpy(), "ps_idx": tr.ps_idx.to_numpy(),
                                         "target": y, "oof": oof, "w": w})
        res["n_train"][side] = len(tr)
        if verbose:
            print(f"  boost {side}: n_train {len(tr)}, shrinkage s = {s:.3f}, "
                  f"sd(g_oof) = {oof.std():.3f}, sd(target) = {y.std():.3f}", flush=True)
    return res


def boost_vector(res: dict, panel: pd.DataFrame, window=None, apply_shrinkage=True,
                 residualize=True, freeze_nuisance=True) -> np.ndarray:
    """The per-Z-column boost g (raw sign, possession-centred), ready to pass as `prior_offset`.

    Two corrections, both needed to keep g doing only the job it is meant to do.

    `freeze_nuisance` scores every player at the training populations playing time rather than his
    own.  Playing time is a feature on the defensive side so that the booster can soak up the part
    of the residual that tracks role rather than skill -- a backup centre faces backups, and the
    ridge under-credits opponents whose own effects are shrunk, so the residual pays him for it.
    Scoring everyone at a starter reference keeps the shape of the box profile and drops that
    premium; `fit_boost` measures its shrinkage the same way, so `s` is not credited for an effect
    that is then thrown away.  Against five stratified permutation nulls, defence keeps a margin of
    0.57 over a null of 0.015 with playing time in, against 0.48 over 0.097 without; offence is the
    other way, 0.80 over 0.069 without it against 0.61 over 0.037 with, so the two sides differ.

    `residualize` strips the weighted linear span of the 13 rates out of g.  The plug-in fit holds
    beta fixed while penalising u, so the residual keeps about 8 percent of the linear prior and a
    free booster re-emits it -- silently rescaling the coefficients we froze on purpose, and
    amplifying the very extrapolation this stage exists to damp.  After this, g is curvature only.
    """
    p = panel if window is None else panel[panel.window == window]
    m = int(p.ps_idx.max()) + 1
    out = np.zeros(2 * m)
    for side, off in (("O", 0), ("D", m)):
        mdl = res["models"].get(side)
        q = p[p.side == side].sort_values("ps_idx")
        if mdl is None:
            continue
        Xq = q[res["features"][side]].copy()
        if freeze_nuisance:
            for c, v in res.get("ref", {}).get(side, {}).items():
                Xq[c] = v
        g = mdl.predict(Xq.to_numpy(dtype=float))
        if apply_shrinkage:
            g = g * res["s"][side]
        wq = np.maximum(q["poss"].to_numpy(), 1e-9)
        if residualize:
            g = _strip_linear(g, q[res["linear"]].to_numpy(dtype=float), wq)
        if wq.sum() > 0:
            g = g - np.average(g, weights=wq)
        out[off + q.ps_idx.to_numpy()] = g
    return out


def dominant_team(wd, box: pd.DataFrame) -> np.ndarray:
    """Per Z unit, the team the player logged the most box-score minutes for in the window."""
    psx = wd.spec.psx_table[["psx_idx", "player_id", "season", "ps_idx"]]
    b = box[box.phase == "RS"].groupby(["player_id", "season", "team_id"], as_index=False)["minutes"].sum()
    b = b.merge(psx, on=["player_id", "season"], how="inner")
    b = b.groupby(["ps_idx", "team_id"], as_index=False)["minutes"].sum()
    pick = b.sort_values("minutes").drop_duplicates("ps_idx", keep="last")
    out = np.full(wd.spec.n_ps, -1, dtype=np.int64)
    out[pick.ps_idx.to_numpy()] = pick.team_id.to_numpy()
    miss = out < 0
    if miss.any():                       # players with no box rows: their own id keeps them separable
        out[miss] = -wd.spec.ps_table["player_id"].to_numpy()[miss]
    return out
