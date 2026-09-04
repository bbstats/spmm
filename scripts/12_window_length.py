"""Phase 4: three-season windows against five-season windows, decided on out-of-sample evidence.

Both designs partition the same thirty seasons, so within each design every game is held out exactly
once by a game-grouped five-fold split and the pooled held-out mean squared error is comparable.

Four things, in this order:
  1. held-out weighted MSE, the joint fit against a zero-prior RAPM on the same folds
  2. coefficient standard errors, and how much of the apparent drift survives
  3. calibration slopes by possession bucket, prior and EBLUP
  4. rating stability: the correlation of a player between adjacent windows

usage: python scripts/12_window_length.py [lengths=3,5]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.checks import calibration_slopes  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.cv import crossfit_beta, plugin_fit  # noqa: E402
from eracoef.windows import build_window, window_label  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
LENGTHS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "3,5").split(",")]
FIRST, LAST = cfg["first_season"], cfg["last_season"]
LAM_B, RAT_B = float(cfg["lam_beta"]), float(cfg["lam_ratio_beta"])
LAM_P, RAT_P = float(cfg["lam_plugin"]), float(cfg["lam_ratio_plugin"])
BUCKETS = tuple(cfg["cv"]["poss_buckets"][:-1]) + (np.inf,)
FEATS = cfg["features"]
nf = len(FEATS)
OUT = Path(cfg["_root"]) / "outputs"


def windows_of(length):
    """Non-overlapping windows of `length` seasons, dropping any short tail."""
    n = (LAST - FIRST + 1) // length
    start = LAST + 1 - n * length                 # keep the most recent seasons whole
    return [list(range(start + i * length, start + (i + 1) * length)) for i in range(n)]


t0 = time.time()
mse_rows, coef_rows, cal_rows, rat_rows = [], [], [], []
for L in LENGTHS:
    wins = windows_of(L)
    print(f"\n=== {L}-season windows: {[window_label(w) for w in wins]}", flush=True)
    for seasons in wins:
        lab = window_label(seasons)
        wd = build_window(seasons, cfg)

        # --- 1. held-out MSE, joint fit vs zero-prior RAPM, same folds
        for k, (tr, te) in enumerate(GroupKFold(5).split(wd.X, wd.y, wd.groups)):
            tr_mask = np.zeros(len(wd.y), bool)
            tr_mask[tr] = True
            cf = crossfit_beta(wd, lams=[LAM_B], cv=2, lam_ratio=RAT_B, mask=tr_mask,
                               pad_target=cfg["pad_target"])
            joint = plugin_fit(wd, cf.beta_, lams=[LAM_P], cv=2, lam_ratio=RAT_P, mask=tr_mask,
                               pad_target=cfg["pad_target"])
            zero = plugin_fit(wd, np.zeros(2 * nf), lams=[LAM_P], cv=2, lam_ratio=RAT_P, mask=tr_mask,
                              pad_target=cfg["pad_target"])
            for name, pipe in (("joint", joint), ("zero_prior", zero)):
                p = pipe.predict(wd.X[te])
                mse_rows.append(dict(length=L, window=lab, fold=k, model=name,
                                     sse=float(np.sum(wd.w[te] * (wd.y[te] - p) ** 2)),
                                     w=float(np.sum(wd.w[te])), n=len(te)))

        # --- 2. coefficients and their standard errors on the full window
        cf = crossfit_beta(wd, lams=[LAM_B], cv=2, lam_ratio=RAT_B, pad_target=cfg["pad_target"])
        se = np.sqrt(np.maximum(np.diag(cf.cov_beta_), 0.0))
        for j, f in enumerate(FEATS):
            c = np.zeros(2 * nf)
            c[j], c[nf + j] = 1.0, -1.0
            coef_rows.append(dict(length=L, window=lab, feature=f,
                                  total=cf.beta_[j] - cf.beta_[nf + j],
                                  se=float(np.sqrt(max(c @ cf.cov_beta_ @ c, 0.0))),
                                  se_o=se[j], se_d=se[nf + j]))

        # --- 3. calibration, and 4. ratings for the stability check
        t = calibration_slopes(wd, lam=LAM_P, lam_ratio=RAT_P, buckets=BUCKETS, mode="crossfit",
                               lam_half=LAM_B, pad_target=cfg["pad_target"], controls=True)
        t["length"], t["window"] = L, lab
        cal_rows.append(t)

        pipe = plugin_fit(wd, cf.beta_, lams=[LAM_P], cv=2, lam_ratio=RAT_P, pad_target=cfg["pad_target"])
        exp, mm = pipe["exposure"], pipe["mm"]
        m = wd.spec.n_ps
        ro = exp.season_rates_ - exp.means_o_ / 5.0
        rd = exp.season_rates_d_ - exp.means_d_ / 5.0
        rat_rows.append(pd.DataFrame({
            "length": L, "window": lab, "mid": float(np.mean(seasons)),
            "player_id": wd.spec.ps_table["player_id"].to_numpy(),
            "poss": exp.season_poss_off_,
            "rating": (ro @ cf.beta_[:nf]) - (rd @ cf.beta_[nf:]) + mm.u_[:m] - mm.u_[m:]}))
        print(f"  {lab}: {m} players, {len(wd.y):,} rows ({time.time() - t0:.0f}s)", flush=True)

mse = pd.DataFrame(mse_rows)
coef = pd.DataFrame(coef_rows)
cal = pd.concat(cal_rows, ignore_index=True)
rat = pd.concat(rat_rows, ignore_index=True)
for name, d in (("window_length_mse", mse), ("window_length_coefs", coef),
                ("window_length_calibration", cal), ("window_length_ratings", rat)):
    d.to_parquet(OUT / f"{name}.parquet", index=False)

print("\n\n=== 1. held-out weighted MSE, pooled over every window (lower is better)")
p = mse.groupby(["length", "model"]).apply(lambda d: d.sse.sum() / d.w.sum(), include_groups=False).unstack()
p["prior_gain_bp"] = 1e4 * (p.zero_prior - p.joint) / p.zero_prior
print(p.round(4).to_string())
print("\n  fold-paired difference between the two lengths is not defined (different windows);")
print("  the comparison is the pooled MSE above, over the same games.")

print("\n=== 2. coefficient standard errors, mean over features and windows")
print(coef.groupby("length")[["se", "se_o", "se_d"]].mean().round(4).to_string())
print("\n  drift that survives: range across windows divided by the mean standard error")
dr = coef.groupby(["length", "feature"]).apply(
    lambda d: pd.Series({"range": d.total.max() - d.total.min(), "mean_se": d.se.mean(),
                         "range_over_se": (d.total.max() - d.total.min()) / d.se.mean()}),
    include_groups=False).reset_index()
print(dr.pivot(index="feature", columns="length", values="range_over_se").round(2).to_string())
print("\n  mean over features:")
print(dr.groupby("length")["range_over_se"].mean().round(2).to_string())

print("\n=== 3. calibration slopes, mean over windows (1.0 = calibrated)")
c = cal[cal.kind.isin(["prior", "eblup"])]
print(c.pivot_table(index=["kind", "bucket"], columns=["length", "side"], values="slope").round(3).to_string())

print("\n=== 4. rating stability between adjacent windows (possession-weighted correlation)")
for L in LENGTHS:
    r = rat[rat.length == L]
    mids = sorted(r.mid.unique())
    cors = []
    for a, b in zip(mids, mids[1:]):
        j = (r[r.mid == a][["player_id", "poss", "rating"]]
             .merge(r[r.mid == b][["player_id", "poss", "rating"]], on="player_id", suffixes=("_a", "_b")))
        j = j[(j.poss_a >= 1500) & (j.poss_b >= 1500)]
        if len(j) > 30:
            w = np.minimum(j.poss_a, j.poss_b)
            ma, mb = np.average(j.rating_a, weights=w), np.average(j.rating_b, weights=w)
            cov = np.average((j.rating_a - ma) * (j.rating_b - mb), weights=w)
            cors.append(cov / np.sqrt(np.average((j.rating_a - ma) ** 2, weights=w)
                                      * np.average((j.rating_b - mb) ** 2, weights=w)))
    print(f"  {L}-season: {len(cors)} adjacent pairs, mean r = {np.mean(cors):.3f} "
          f"(min {np.min(cors):.3f}, max {np.max(cors):.3f})")

print(f"\nwrote outputs/window_length_*.parquet  ({time.time() - t0:.0f}s)")
