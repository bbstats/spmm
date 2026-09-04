"""Sanity checks, OOS comparisons, calibration slopes and simulation tests."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .cv import crossfit_beta, lambda_ratio_grid, make_pipeline, plugin_fit
from .design import WindowData, build_design
from .simulate import MECH, simulate, truth_aligned


# ----------------------------------------------------------------------------- recovery
def beta_table(mm, truth=None, flip_defense=False) -> pd.DataFrame:
    """Estimated beta with SE (raw signs), optionally beside the simulation truth.

    `mm` is anything exposing beta_, beta_se_, n_feat_, spec (MixedModelRAPM or CrossfitResult).
    """
    nf = mm.n_feat_
    feats = list(mm.spec.features)
    rows = []
    for side, off in (("O", 0), ("D", nf)):
        for j, f in enumerate(feats):
            b = mm.beta_[off + j]
            se = mm.beta_se_[off + j]
            r = dict(side=side, feature=f, beta=b, se=se)
            if truth is not None:
                t = (truth["beta_O"] if side == "O" else truth["beta_D"])[j]
                r.update(true=t, z=(b - t) / se)
            if flip_defense and side == "D":
                r["beta"] = -b
                if truth is not None:
                    r["true"] = -r["true"]
            rows.append(r)
    return pd.DataFrame(rows)


def u_recovery(mm, spec, truth) -> pd.DataFrame:
    """Correlation of the fitted u with the true eta and with the true total impact."""
    t = truth_aligned(truth, spec)
    m = spec.n_ps
    out = []
    for side, sl, eta, imp in (("O", slice(0, m), "u_O", "impact_O"), ("D", slice(m, 2 * m), "u_D", "impact_D")):
        u = mm.u_[sl]
        ok = t[eta].notna().to_numpy() & (u != 0)
        out.append(dict(side=side, n=int(ok.sum()),
                        corr_u_eta=np.corrcoef(u[ok], t[eta].to_numpy()[ok])[0, 1],
                        sd_u=u[ok].std(), sd_eta=t[eta].to_numpy()[ok].std()))
    return pd.DataFrame(out)


def player_ratings(pipe, spec) -> pd.DataFrame:
    """Per player-season prior (season_rates x beta), u, and total, O and D (D flipped)."""
    exp, mm = pipe["exposure"], pipe["mm"]
    nf = mm.n_feat_
    m = spec.n_ps
    prior_O = exp.season_rates_ @ mm.beta_[:nf] - exp.means_o_ @ mm.beta_[:nf] if nf else np.zeros(m)
    prior_D = exp.season_rates_d_ @ mm.beta_[nf:2 * nf] - exp.means_d_ @ mm.beta_[nf:2 * nf] if nf else np.zeros(m)
    df = spec.ps_table.copy()
    df["poss"] = exp.season_poss_off_
    df["prior_O"] = prior_O
    df["u_O"] = mm.u_[:m]
    df["O"] = prior_O + mm.u_[:m]
    df["prior_D"] = -prior_D
    df["u_D"] = -mm.u_[m:]
    df["D"] = -(prior_D + mm.u_[m:])
    df["total"] = df["O"] + df["D"]
    return df


# ----------------------------------------------------------------------------- SE calibration
def se_calibration(n_reps=30, lam=4000.0, lam_ratio=1.0, sim_kwargs=None, cfg=None, pad_k="auto", mode="crossfit",
                   verbose=True):
    """Simulate n_reps datasets, fit at fixed lambda, return z = (beta_hat - beta)/se (n_reps x 26)."""
    sim_kwargs = dict(sim_kwargs or {})
    cfg = cfg or {}
    zs, ses, ests = [], [], []
    t0 = time.time()
    for r in range(n_reps):
        sim = simulate(seed=1000 + r, **sim_kwargs)
        wd = build_design(sim["stints"], sim["box"], sim["truth"]["features"], cfg)
        if mode == "crossfit":
            est = crossfit_beta(wd, lam=lam, lam_ratio=lam_ratio, pad_k=pad_k)
        else:
            est = make_pipeline(wd, lam=lam, lam_ratio=lam_ratio, pad_k=pad_k, mode=mode).fit(wd.X, wd.y, sample_weight=wd.w)["mm"]
        tb = beta_table(est, sim["truth"])
        zs.append(tb["z"].to_numpy()); ses.append(tb["se"].to_numpy()); ests.append(tb["beta"].to_numpy())
        if verbose and (r + 1) % 10 == 0:
            print(f"  rep {r + 1}/{n_reps}  ({time.time() - t0:.0f}s)")
    z = np.asarray(zs)
    labels = tb["side"] + ":" + tb["feature"]
    summary = pd.DataFrame({"coef": labels, "mean_z": z.mean(0), "sd_z": z.std(0, ddof=1),
                            "cover95": (np.abs(z) < 1.96).mean(0), "mean_se": np.mean(ses, 0),
                            "sd_est": np.std(ests, 0, ddof=1)})
    return z, summary


# ----------------------------------------------------------------------------- leakage bias table
def _theta(lam, d):
    return 1.0 - np.sqrt(lam / (lam + d))


def leakage_inputs(wd: WindowData, truth: dict, mask=None) -> pd.DataFrame:
    """Per feature: v/n (sampling variance of a player's season-mean rate, possession-weighted) and tau2_x.

    Poisson counts: Var(100 C/P) = 100 mu / P, so v/n = 100 mu_i / P_i, averaged over players with weight P_i.
    tau2_x = possession-weighted between-player variance of the true rates.
    """
    t = truth_aligned(truth, wd.spec)
    gp = wd.game_poss
    if mask is not None:
        games = np.unique(wd.rows["game_idx"].to_numpy()[mask])
        gp = gp[np.isin(gp["game_idx"], games)]
    P = np.zeros(wd.spec.n_ps)
    np.add.at(P, wd.spec.ps_of_psx[gp["psx_idx"].to_numpy()], gp["poss_off"].to_numpy())
    rows = []
    for f in truth["features"]:
        mu = t[f"rate_{f}"].to_numpy()
        ok = (P > 0) & np.isfinite(mu)
        w = P[ok]
        vn = np.sum(w * (100.0 * mu[ok] / P[ok])) / w.sum()
        mbar = np.sum(w * mu[ok]) / w.sum()
        tau2 = np.sum(w * (mu[ok] - mbar) ** 2) / w.sum()
        rows.append(dict(feature=f, v_over_n=vn, tau2_x=tau2, m=MECH.get(f, 0.0)))
    return pd.DataFrame(rows).set_index("feature")


def bias_table(n_reps=40, lams=(800.0, 8000.0, 80000.0), modes=("full", "loo", "crossfit"), sim_kwargs=None, cfg=None,
               features=None, lam_ratio=1.0, verbose=True) -> pd.DataFrame:
    """Mean bias of beta_hat for the mechanical O features by lambda and exposure mode, with the predicted biases.

    predicted full bias  = +(m - beta) (v/n) / tau2_x               (lambda independent)
    predicted loo bias   = -m (v/n) theta(2-theta) / ((1-theta)^2 tau2_x),  theta at the median starter
    predicted crossfit   = 0
    """
    sim_kwargs = dict(sim_kwargs or {})
    cfg = cfg or {}
    feats = [f for f in MECH] if features is None else list(features)
    acc = {}          # (lam, mode) -> list of beta_hat arrays for feats (O side)
    ses = {}
    thetas = {}
    inputs = None
    t0 = time.time()
    for r in range(n_reps):
        sim = simulate(seed=3000 + r, **sim_kwargs)
        wd = build_design(sim["stints"], sim["box"], sim["truth"]["features"], cfg)
        jf = [sim["truth"]["features"].index(f) for f in feats]
        if inputs is None:
            inputs = leakage_inputs(wd, sim["truth"])
            d_all = np.asarray(wd.X[:, wd.spec.zo].T @ wd.w).ravel()
            d_all = d_all[d_all > 0]
            d_med = float(np.median(d_all[d_all >= np.quantile(d_all, 0.6)]))
            half_mask = wd.half_mask("A")
            d_half = np.asarray(wd.X[half_mask][:, wd.spec.zo].T @ wd.w[half_mask]).ravel()
            d_half = d_half[d_half > 0]
            d_med_half = float(np.median(d_half[d_half >= np.quantile(d_half, 0.6)]))
        for lam in lams:
            for mode in modes:
                if mode == "crossfit":
                    est = crossfit_beta(wd, lam=lam, lam_ratio=lam_ratio)
                else:
                    est = make_pipeline(wd, lam=lam, lam_ratio=lam_ratio, mode=mode).fit(wd.X, wd.y, sample_weight=wd.w)["mm"]
                acc.setdefault((lam, mode), []).append(est.beta_[jf])
                ses.setdefault((lam, mode), []).append(est.beta_se_[jf])
        if verbose and (r + 1) % 10 == 0:
            print(f"  rep {r + 1}/{n_reps}  ({time.time() - t0:.0f}s)")
    rows = []
    beta_true = sim["truth"]["beta_O"][jf]
    for lam in lams:
        th = _theta(lam, d_med)
        th_half = _theta(lam, d_med_half)
        for mode in modes:
            B = np.asarray(acc[(lam, mode)])
            S = np.asarray(ses[(lam, mode)])
            for i, f in enumerate(feats):
                m = inputs.loc[f, "m"]; vn = inputs.loc[f, "v_over_n"]; t2 = inputs.loc[f, "tau2_x"]
                if mode == "full":
                    pred = (m - beta_true[i]) * vn / t2
                elif mode == "loo":
                    pred = -m * vn * th * (2 - th) / ((1 - th) ** 2 * t2)
                else:
                    pred = 0.0
                bias = B[:, i].mean() - beta_true[i]
                rows.append(dict(lam=lam, theta_starter=th if mode != "crossfit" else th_half, mode=mode, feature=f,
                                 true=beta_true[i], mean_beta=B[:, i].mean(), bias=bias,
                                 bias_se=B[:, i].std(ddof=1) / np.sqrt(len(B)), bias_in_se=bias / S[:, i].mean(),
                                 mean_z=((B[:, i] - beta_true[i]) / S[:, i]).mean(), sd_z=((B[:, i] - beta_true[i]) / S[:, i]).std(ddof=1),
                                 pred_bias=pred, m=m, v_over_n=vn, tau2_x=t2))
    out = pd.DataFrame(rows)
    out.attrs["d_med_starter"] = d_med
    out.attrs["d_med_starter_half"] = d_med_half
    return out


# ----------------------------------------------------------------------------- calibration slopes
def _wls(Xm, y, w):
    Xw = Xm * w[:, None]
    return np.linalg.solve(Xm.T @ Xw, Xw.T @ y)


def calibration_slopes(wd: WindowData, lam, lam_ratio=1.0, lam_buckets=None, n_folds=5, buckets=(0, 500, 1500, np.inf),
                       features=None, pad_k="auto", pad_scale=1.0, lam_delta=None, mode="full", verbose=False,
                       lam_half=None, eval_rows=None, prior_offset=None, controls=True,
                       **exposure_kw) -> pd.DataFrame:
    """OOS calibration: regress y - F gamma on the O/D prediction components, weighted by possessions.

    mode="crossfit": beta from the two half-fits on the training games, then u from a plug-in fit on all
    training rows with full-season rates.  mode="full"/"loo": one joint fit on the training rows.
    Slopes are reported for the prior alone (rates x beta), the full EBLUP (prior + u), and jointly for
    [prior, u] (the u slope isolates the ridge), by side and by the possession bucket of the players involved.

    `controls` adds `Z @ in_bucket` headcount columns to every regression.  Without them the low
    bucket regressor is driven as much by HOW MANY bench players are on the floor as by which ones,
    and that headcount tracks the outcome through blowouts, injuries and tanking; it drags the low
    bucket slope down by about a third.

    `prior_offset` is the LRBoost correction, one entry per Z column in raw sign.  It joins the prior,
    not u, so the slopes score the boosted prior rather than the linear one.  Pass a vector fit
    WITHOUT this window in it, or the held-out games leak in through the boosters training target.
    """
    spec = wd.spec
    m = spec.n_ps
    nb = len(buckets) - 1
    splits = list(GroupKFold(n_folds).split(wd.X, wd.y, wd.groups))
    parts = []
    for k, (tr, te) in enumerate(splits):
        tr_mask = np.zeros(len(wd.y), bool); tr_mask[tr] = True
        if mode == "crossfit":
            cf = crossfit_beta(wd, lam=lam if lam_half is None else lam_half, lam_ratio=lam_ratio, lam_buckets=lam_buckets,
                               features=features, pad_k=pad_k, pad_scale=pad_scale, lam_delta=lam_delta, mask=tr_mask,
                               **exposure_kw)
            pipe = plugin_fit(wd, cf.beta_, lam=lam, lam_ratio=lam_ratio, lam_buckets=lam_buckets, features=features,
                              pad_k=pad_k, pad_scale=pad_scale, lam_delta=lam_delta, mask=tr_mask,
                              prior_offset=prior_offset, **exposure_kw)
            cov_beta = cf.cov_beta_
        else:
            pipe = make_pipeline(wd, lam=lam, lam_ratio=lam_ratio, lam_buckets=lam_buckets, features=features,
                                 pad_k=pad_k, pad_scale=pad_scale, lam_delta=lam_delta, mode=mode, **exposure_kw)
            pipe.fit(wd.X[tr], wd.y[tr], sample_weight=wd.w[tr])
            cov_beta = pipe["mm"].cov_beta_
        if eval_rows is not None:
            te = te[np.asarray(eval_rows, dtype=bool)[te]]
        exp, mm = pipe["exposure"], pipe["mm"]
        nf = mm.n_feat_
        rates_o, rates_d = exp.rates_, exp.rates_d_          # what transform uses for held-out rows
        pri_O = rates_o @ mm.beta_[:nf] if nf else np.zeros(m)
        pri_D = rates_d @ mm.beta_[nf:] if nf else np.zeros(m)
        if prior_offset is not None:
            po = np.asarray(prior_offset, dtype=float)
            pri_O = pri_O + po[:m]
            pri_D = pri_D + po[m:]
        u_O, u_D = mm.u_[:m], mm.u_[m:]
        poss = exp.poss_off_
        Xte = wd.X[te]
        ZO, ZD = Xte[:, spec.zo], Xte[:, spec.zd]
        fixed = Xte[:, spec.f] @ mm.gamma_
        resid = wd.y[te] - fixed
        cols = {"fold": np.full(len(te), k), "y": resid, "w": wd.w[te]}
        for b in range(nb):
            mask = ((poss >= buckets[b]) & (poss < buckets[b + 1])).astype(float)
            cols[f"prior_O_{b}"] = ZO @ (pri_O * mask)
            cols[f"prior_D_{b}"] = ZD @ (pri_D * mask)
            cols[f"u_O_{b}"] = ZO @ (u_O * mask)
            cols[f"u_D_{b}"] = ZD @ (u_D * mask)
            cols[f"eblup_O_{b}"] = cols[f"prior_O_{b}"] + cols[f"u_O_{b}"]
            cols[f"eblup_D_{b}"] = cols[f"prior_D_{b}"] + cols[f"u_D_{b}"]
            cols[f"n_O_{b}"] = ZO @ mask          # headcount of this bucket on the floor
            cols[f"n_D_{b}"] = ZD @ mask
        # expected attenuation of the prior's slope from beta estimation noise: x' Cov(beta) x per row
        if nf:
            XO = ZO @ rates_o; XD = ZD @ rates_d
            wte = wd.w[te][:, None]
            XO = XO - (wte * XO).sum(0) / wte.sum()     # centered: the calibration intercept absorbs the mean direction
            XD = XD - (wte * XD).sum(0) / wte.sum()
            cols["bnoise_O"] = np.einsum("ij,jk,ik->i", XO, cov_beta[:nf, :nf], XO)
            cols["bnoise_D"] = np.einsum("ij,jk,ik->i", XD, cov_beta[nf:, nf:], XD)
            Xb = np.hstack([XO, XD])
            cols["bnoise_OD"] = np.einsum("ij,jk,ik->i", Xb, cov_beta, Xb)
        else:
            cols["bnoise_O"] = cols["bnoise_D"] = cols["bnoise_OD"] = np.zeros(len(te))
        parts.append(pd.DataFrame(cols))
        if verbose:
            print(f"  fold {k + 1}/{n_folds} done")
    df = pd.concat(parts, ignore_index=True)
    # fold fixed effects: each fold has its own beta-hat, gamma and league rates, so the components and the
    # residual shift in level between folds; remove the possession-weighted fold means before pooling
    comp_cols = [c for c in df.columns if c not in ("fold", "w") and not c.startswith("bnoise")]
    for k in range(n_folds):
        f = df["fold"].to_numpy() == k
        wf = df.loc[f, "w"].to_numpy()
        for c in comp_cols:
            v = df.loc[f, c].to_numpy()
            df.loc[f, c] = v - np.sum(wf * v) / wf.sum()
    wv = df["w"].to_numpy()
    fold_of = df["fold"].to_numpy()
    yv = df["y"].to_numpy()

    def _wvar(x):
        mu = np.sum(wv * x) / wv.sum()
        return np.sum(wv * (x - mu) ** 2) / wv.sum()

    ctrl = (np.column_stack([df[f"n_{s}_{b}"].to_numpy() for s in ("O", "D") for b in range(nb)])
            if controls else np.zeros((len(df), 0)))

    def _fold_slopes(Xm):
        return np.asarray([_wls(Xm[fold_of == k], yv[fold_of == k], wv[fold_of == k]) for k in range(n_folds)])

    out = []
    for kind in ("prior", "eblup", "joint"):
        if kind == "joint":
            names = [f"{c}_{s}_{b}" for c in ("prior", "u") for s in ("O", "D") for b in range(nb)]
            Xm = np.column_stack([df[n].to_numpy() for n in names] + [ctrl, np.ones(len(df))])
            sl = _wls(Xm, yv, wv)
            fold_sl = _fold_slopes(Xm)
            for i, n in enumerate(names):
                c, side, b = n.split("_")
                b = int(b)
                out.append(dict(kind=f"joint_{c}", side=side, bucket=f"[{buckets[b]},{buckets[b + 1]})", slope=sl[i],
                                se=fold_sl[:, i].std(ddof=1) / np.sqrt(n_folds)))
            agg = {f"{c}_{s}": sum(df[f"{c}_{s}_{b}"].to_numpy() for b in range(nb)) for c in ("prior", "u") for s in ("O", "D")}
            Xa = np.column_stack([agg["prior_O"], agg["prior_D"], agg["u_O"], agg["u_D"], ctrl, np.ones(len(df))])
            sl = _wls(Xa, yv, wv)
            fold_sl = _fold_slopes(Xa)
            for i, (c, s) in enumerate((("prior", "O"), ("prior", "D"), ("u", "O"), ("u", "D"))):
                out.append(dict(kind=f"joint_{c}", side=s, bucket="all", slope=sl[i], se=fold_sl[:, i].std(ddof=1) / np.sqrt(n_folds)))
            Xt = np.column_stack([agg["prior_O"] + agg["prior_D"], agg["u_O"] + agg["u_D"], ctrl, np.ones(len(df))])
            sl = _wls(Xt, yv, wv)
            out.append(dict(kind="joint_prior", side="O+D", bucket="all", slope=sl[0], se=np.nan))
            out.append(dict(kind="joint_u", side="O+D", bucket="all", slope=sl[1], se=np.nan))
            continue
        names = [f"{kind}_{s}_{b}" for s in ("O", "D") for b in range(nb)]
        Xm = np.column_stack([df[n].to_numpy() for n in names] + [ctrl, np.ones(len(df))])
        sl = _wls(Xm, yv, wv)
        fold_sl = _fold_slopes(Xm)
        for i, n in enumerate(names):
            _, side, b = n.split("_")
            b = int(b)
            out.append(dict(kind=kind, side=side, bucket=f"[{buckets[b]},{buckets[b + 1]})", slope=sl[i],
                            se=fold_sl[:, i].std(ddof=1) / np.sqrt(n_folds)))
        Xa = np.column_stack([sum(df[f"{kind}_{s}_{b}"].to_numpy() for b in range(nb)) for s in ("O", "D")]
                             + [ctrl, np.ones(len(df))])
        sl = _wls(Xa, yv, wv)
        fold_sl = _fold_slopes(Xa)
        for i, s in enumerate(("O", "D")):
            r = dict(kind=kind, side=s, bucket="all", slope=sl[i], se=fold_sl[:, i].std(ddof=1) / np.sqrt(n_folds))
            if kind == "prior":
                bn = np.sum(wv * df[f"bnoise_{s}"].to_numpy()) / wv.sum()
                r["beta_noise_var"] = bn
                r["expected_slope"] = 1.0 - bn / max(_wvar(Xa[:, i]), 1e-12)
            out.append(r)
        Xt = np.column_stack([Xa[:, 0] + Xa[:, 1], ctrl, np.ones(len(df))])
        sl = _wls(Xt, yv, wv)
        r = dict(kind=kind, side="O+D", bucket="all", slope=sl[0], se=np.nan)
        if kind == "prior":
            bn = np.sum(wv * df["bnoise_OD"].to_numpy()) / wv.sum()
            r["beta_noise_var"] = bn
            r["expected_slope"] = 1.0 - bn / max(_wvar(Xt[:, 0]), 1e-12)
        out.append(r)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- OOS comparisons
def _predict_crossfit(wd, tr_mask, te, lam, lam_ratio, pad_k, features):
    cf = crossfit_beta(wd, lam=lam, lam_ratio=lam_ratio, features=features, pad_k=pad_k, mask=tr_mask)
    pipe = plugin_fit(wd, cf.beta_, lam=lam, lam_ratio=lam_ratio, features=features, pad_k=pad_k, mask=tr_mask)
    return pipe.predict(wd.X[te])


def oos_compare(wd: WindowData, lam, lam_ratio=1.0, n_folds=5, pad_k="auto", features=None, mode="crossfit",
                lam_zero=None) -> pd.DataFrame:
    """Held-out weighted MSE per fold: zero-prior RAPM (features=[]) vs joint mixed-model RAPM."""
    splits = list(GroupKFold(n_folds).split(wd.X, wd.y, wd.groups))
    rows = []
    for k, (tr, te) in enumerate(splits):
        tr_mask = np.zeros(len(wd.y), bool); tr_mask[tr] = True
        res = dict(fold=k)
        zp = make_pipeline(wd, lam=lam if lam_zero is None else lam_zero, lam_ratio=lam_ratio, features=[], pad_k=pad_k, mode="full")
        zp.fit(wd.X[tr], wd.y[tr], sample_weight=wd.w[tr])
        yhat0 = zp.predict(wd.X[te])
        if mode == "crossfit":
            yhat1 = _predict_crossfit(wd, tr_mask, te, lam, lam_ratio, pad_k, features)
        else:
            jp = make_pipeline(wd, lam=lam, lam_ratio=lam_ratio, features=features, pad_k=pad_k, mode=mode)
            jp.fit(wd.X[tr], wd.y[tr], sample_weight=wd.w[tr])
            yhat1 = jp.predict(wd.X[te])
        res["zero_prior"] = float(np.sum(wd.w[te] * (wd.y[te] - yhat0) ** 2) / np.sum(wd.w[te]))
        res["joint"] = float(np.sum(wd.w[te] * (wd.y[te] - yhat1) ** 2) / np.sum(wd.w[te]))
        res["diff"] = res["joint"] - res["zero_prior"]
        rows.append(res)
    return pd.DataFrame(rows)


def mode_comparison(wd: WindowData, lam, lam_ratio=1.0, pad_k="auto", mech=None, truth=None) -> pd.DataFrame:
    """beta by exposure mode (crossfit / full / loo); full should sit above crossfit on makes, loo below."""
    out = None
    for mode in ("crossfit", "full", "loo"):
        if mode == "crossfit":
            est = crossfit_beta(wd, lam=lam, lam_ratio=lam_ratio, pad_k=pad_k)
        else:
            est = make_pipeline(wd, lam=lam, lam_ratio=lam_ratio, pad_k=pad_k, mode=mode).fit(wd.X, wd.y, sample_weight=wd.w)["mm"]
        tb = beta_table(est, truth)[["side", "feature", "beta", "se"] + (["true"] if truth else [])]
        tb = tb.rename(columns={"beta": f"beta_{mode}", "se": f"se_{mode}"})
        out = tb if out is None else out.merge(tb, on=["side", "feature"] + (["true"] if truth else []))
    out["full_minus_cf_in_se"] = (out["beta_full"] - out["beta_crossfit"]) / out["se_crossfit"]
    out["loo_minus_cf_in_se"] = (out["beta_loo"] - out["beta_crossfit"]) / out["se_crossfit"]
    mech = MECH if mech is None else mech
    out["mechanical"] = [mech.get(f, 0.0) if s == "O" else np.nan for s, f in zip(out["side"], out["feature"])]
    return out


def bootstrap_beta(wd: WindowData, lam, lam_ratio=1.0, n_boot=200, seed=0, features=None, pad_k="auto",
                   lam_buckets=None, scheme="bayesian", fix_padding=True, verbose=True, **exposure_kw) -> dict:
    """Game-level bootstrap of the cross-fitted beta.

    scheme="bayesian": Exp(1) weight per game (normalized to mean 1 within each half), applied to the
    row weights and to the covariate-side counts, with padding on n_eff = (sum w n)^2 / sum w^2 n.
    scheme="multinomial": resample games with replacement (duplicates under-pad unless n_eff is used).
    fix_padding: freeze k_j, league means and the padding target at their full-sample values.
    The A/B assignment is fixed throughout.
    Returns dict(beta=(n_boot x 2nf), se_boot, se_formula, ratio, corr_boot, corr_formula).
    """
    rng = np.random.default_rng(seed)
    games = wd.games
    rs = games["phase"].to_numpy() == "RS"
    n_games = int(games["game_idx"].max()) + 1
    row_game = wd.rows["game_idx"].to_numpy()
    base = crossfit_beta(wd, lam=lam, lam_ratio=lam_ratio, features=features, pad_k=pad_k, lam_buckets=lam_buckets,
                         **exposure_kw)
    by_half = None
    if fix_padding:
        by_half = {h: {"fixed_padding": base.fits[h]["exposure"].padding_state()} for h in ("A", "B")}
    betas = []
    t0 = time.time()
    for b in range(n_boot):
        mult = np.zeros(n_games)
        for h in ("A", "B"):
            idx = games.loc[rs & (games["half"].to_numpy() == h), "game_idx"].to_numpy()
            if scheme == "bayesian":
                w = rng.exponential(size=len(idx))
                mult[idx] = w / w.mean()
            else:
                draw = rng.choice(idx, size=len(idx), replace=True)
                np.add.at(mult, draw, 1.0)
        cf = crossfit_beta(wd, lam=lam, lam_ratio=lam_ratio, features=features, pad_k=pad_k, lam_buckets=lam_buckets,
                           w_mult=mult[row_game], game_mult=mult, exposure_kw_by_half=by_half, **exposure_kw)
        betas.append(cf.beta_)
        if verbose and (b + 1) % 25 == 0:
            print(f"  bootstrap {b + 1}/{n_boot}  ({time.time() - t0:.0f}s)", flush=True)
    B = np.asarray(betas)
    se_boot = B.std(0, ddof=1)
    cov_boot = np.cov(B.T)
    d = np.sqrt(np.diag(cov_boot)); corr_boot = cov_boot / np.outer(d, d)
    df = np.sqrt(np.diag(base.cov_beta_)); corr_formula = base.cov_beta_ / np.outer(df, df)
    return dict(beta=B, beta_hat=base.beta_, se_boot=se_boot, se_formula=base.beta_se_, ratio=se_boot / base.beta_se_,
                corr_boot=corr_boot, corr_formula=corr_formula,
                boot_mean_minus_hat_in_se=(B.mean(0) - base.beta_) / base.beta_se_)


def variance_components(wd: WindowData, beta, lams, ratios, n_folds=5, features=None, pad_k="auto",
                        verbose=True, **exposure_kw) -> pd.DataFrame:
    """tau2 per side for a zero-prior fit and the plug-in fit; 1 - tau2_1/tau2_0 = share the prior explains.

    tau2 = sigma2 / lambda_side, with lambda_O = lam and lambda_D = lam * lam_ratio; (lam, ratio) are
    chosen for each model on its own joint grid by the REML-in-CV-1SE-band rule.
    """
    out = []
    for name, kw in (("zero_prior", dict(features=[], mode="full")), ("plugin", dict(beta_fixed=np.asarray(beta, float), mode="full", features=features))):
        fits = {}

        def fit_fn(r, kw=kw, fits=fits):
            if name == "zero_prior":
                p = make_pipeline(wd, lams=lams, cv=n_folds, lam_ratio=r, pad_k=pad_k, **kw, **exposure_kw)
                p.fit(wd.X, wd.y, sample_weight=wd.w, groups=wd.groups)
            else:
                p = plugin_fit(wd, beta, lams=lams, cv=n_folds, lam_ratio=r, pad_k=pad_k, features=features, **exposure_kw)
            fits[r] = p
            return p["mm"]

        tab, lam, ratio, how = lambda_ratio_grid(fit_fn, ratios, lams)
        mm = fits[ratio]["mm"]
        mm.set_lam(lam)
        s2 = mm.sigma2_
        out.append(dict(model=name, lam=lam, lam_ratio=ratio, chosen_by=how, sigma2=s2,
                        tau2_O=s2 / lam, tau2_D=s2 / (lam * ratio), edf=mm.edf_))
        if verbose:
            print(f"  {name}: lambda {lam:.0f}, ratio {ratio}, chosen by {how}, sigma2 {s2:.0f}", flush=True)
    df = pd.DataFrame(out)
    z, p = df.iloc[0], df.iloc[1]
    df.loc[len(df)] = dict(model="share_explained", lam=np.nan, lam_ratio=np.nan, chosen_by="",
                           sigma2=np.nan, tau2_O=1 - p.tau2_O / z.tau2_O, tau2_D=1 - p.tau2_D / z.tau2_D, edf=np.nan)
    return df


def grid_fold_se(gs) -> pd.DataFrame:
    """GridSearchCV results as per-candidate OOS-MSE difference vs the best candidate, fold-paired mean and SE."""
    res = pd.DataFrame(gs.cv_results_)
    split_cols = sorted([c for c in res.columns if c.startswith("split") and c.endswith("_test_score")])
    S = -res[split_cols].to_numpy()                        # MSE per fold (scorer is negative MSE)
    best = int(np.argmin(S.mean(1)))
    diff = S - S[[best]]
    out = pd.DataFrame({c.replace("param_", ""): res[c] for c in res.columns if c.startswith("param_")})
    out["mean_mse"] = S.mean(1)
    out["diff_vs_best"] = diff.mean(1)
    out["diff_se"] = diff.std(1, ddof=1) / np.sqrt(S.shape[1])
    out["in_1se_band"] = out["diff_vs_best"] <= out["diff_se"]
    return out


def rs_to_po_check(wd: WindowData, lam, lam_ratio=1.0, pad_k="auto", features=None) -> pd.DataFrame:
    """RS-only fit, predict PO rows: zero-prior RAPM vs joint (weighted OOS MSE on playoff rows)."""
    rs = wd.rs_mask()
    po = ~rs
    rows = []
    zp = make_pipeline(wd, lam=lam, lam_ratio=lam_ratio, features=[], pad_k=pad_k, mode="full")
    zp.fit(wd.X[rs], wd.y[rs], sample_weight=wd.w[rs])
    yhat0 = zp.predict(wd.X[po])
    yhat1 = _predict_crossfit(wd, rs, np.flatnonzero(po), lam, lam_ratio, pad_k, features)
    for name, yhat in (("zero_prior", yhat0), ("joint", yhat1)):
        rows.append(dict(model=name, po_mse=float(np.sum(wd.w[po] * (wd.y[po] - yhat) ** 2) / np.sum(wd.w[po])),
                         n_po_rows=int(po.sum())))
    return pd.DataFrame(rows)
