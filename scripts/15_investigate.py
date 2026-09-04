"""Why do bigs sweep the board?  Four diagnostics that do not assume the answer.

1. CONSERVATION.  A stint's box term is the SUM of five players' per-100 rates.  For a stat that the
   team produces exactly once (a defensive rebound goes to one of the five), that sum is close to a
   team constant, so almost all of the individual variation cancels inside the lineup.  Measure, per
   feature, how much individual spread survives into the lineup sum.  Where little survives, beta is
   identified off a thin slice of between-lineup variation and is then applied to fat individual
   variation to build a rating.

2. WHAT THE CALIBRATION TEST CANNOT SEE.  `Z @ prior` is a lineup sum.  Moving credit from a player
   to his teammate leaves every lineup sum they share unchanged, so a calibration slope of 1.0 says
   the lineup-level prior is right and says NOTHING about whether the split within a lineup is.
   Quantify how much cancellation there is: sd of the lineup prior against sqrt(5) x sd of the
   player prior.

3. THE HONEST PLAYER-LEVEL TEST.  Fit a zero-prior RAPM (no box score at all) in window w+1 and ask
   how well window w's components predict it.  Different teammates, different games, so within-lineup
   misattribution cannot hide.  Then add a bigness term: if the box prior over-credits bigs, bigness
   carries a negative coefficient after the prior is already in the regression.

4. PARSIMONY.  How much of the prior's spread do the four rebound and defensive-event features carry,
   and does the 13-feature prior beat a much smaller one out of sample?

usage: python scripts/15_investigate.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.cv import crossfit_beta, plugin_fit  # noqa: E402
from eracoef.windows import build_window, window_label, window_seasons  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
FE = cfg["features"]
nf = len(FE)
LAM_B, RAT_B = float(cfg["lam_beta"]), float(cfg["lam_ratio_beta"])
LAM_P, RAT_P = float(cfg["lam_plugin"]), float(cfg["lam_ratio_plugin"])
OUT = Path(cfg["_root"]) / "outputs"
t0 = time.time()

cons, comp, rap = [], [], []
for w in window_seasons(cfg):
    seasons = list(range(w[0], w[1] + 1))
    lab = window_label(seasons)
    wd = build_window(seasons, cfg)
    spec = wd.spec
    m = spec.n_ps
    cf = crossfit_beta(wd, lams=[LAM_B], cv=2, lam_ratio=RAT_B, pad_target=cfg["pad_target"])
    pipe = plugin_fit(wd, cf.beta_, lams=[LAM_P], cv=2, lam_ratio=RAT_P, pad_target=cfg["pad_target"])
    zero = plugin_fit(wd, np.zeros(2 * nf), lams=[LAM_P], cv=2, lam_ratio=RAT_P,
                      pad_target=cfg["pad_target"])
    exp, mm = pipe["exposure"], pipe["mm"]
    ro = exp.season_rates_ - exp.means_o_ / 5.0
    rd = exp.season_rates_d_ - exp.means_d_ / 5.0
    ZO, ZD = wd.X[:, spec.zo], wd.X[:, spec.zd]
    wt = wd.w
    poss = exp.season_poss_off_

    # --- 1. conservation: individual spread vs what survives into the lineup sum
    for j, f in enumerate(FE):
        for side, R, Z in (("O", exp.season_rates_, ZO), ("D", exp.season_rates_d_, ZD)):
            x = R[:, j]
            sd_i = np.sqrt(np.average((x - np.average(x, weights=poss)) ** 2, weights=poss))
            L = Z @ x
            sd_L = np.sqrt(np.average((L - np.average(L, weights=wt)) ** 2, weights=wt))
            cons.append(dict(window=lab, side=side, feature=f, sd_player=sd_i, sd_lineup=sd_L,
                             survives=sd_L / max(np.sqrt(5) * sd_i, 1e-12)))

    # --- 2. the same for the assembled prior, plus per-feature variance shares
    pri_o, pri_d = ro @ cf.beta_[:nf], rd @ cf.beta_[nf:]
    for side, p, Z in (("O", pri_o, ZO), ("D", pri_d, ZD)):
        sd_i = np.sqrt(np.average((p - np.average(p, weights=poss)) ** 2, weights=poss))
        L = Z @ p
        sd_L = np.sqrt(np.average((L - np.average(L, weights=wt)) ** 2, weights=wt))
        comp.append(dict(window=lab, side=side, sd_player_prior=sd_i, sd_lineup_prior=sd_L,
                         survives=sd_L / max(np.sqrt(5) * sd_i, 1e-12)))

    # --- 3. components now, pure on-court RAPM now, for the next-window test
    rap.append(pd.DataFrame({
        "window": lab, "mid": float(np.mean(seasons)),
        "player_id": spec.ps_table["player_id"].to_numpy(), "poss": poss,
        "prior_off": pri_o, "prior_def": -pri_d,
        "u_off": mm.u_[:m], "u_def": -mm.u_[m:],
        "rapm0_off": zero["mm"].u_[:m], "rapm0_def": -zero["mm"].u_[m:],
        **{f"r_{f}": exp.season_rates_[:, j] for j, f in enumerate(FE)}}))
    print(f"  {lab} ({time.time() - t0:.0f}s)", flush=True)
    del wd

cons = pd.DataFrame(cons)
comp = pd.DataFrame(comp)
rap = pd.concat(rap, ignore_index=True)
for c in ("prior", "u", "rapm0"):
    rap[f"{c}_total"] = rap[f"{c}_off"] + rap[f"{c}_def"]
rap["bigness"] = rap.r_orb + rap.r_blk + 0.3 * rap.r_drb - 0.5 * rap.r_ast - 0.4 * rap.r_fg3m
for name, d in (("investigate_conservation", cons), ("investigate_prior", comp),
                ("investigate_rapm", rap)):
    d.to_parquet(OUT / f"{name}.parquet", index=False)

print("\n\n=== 1. how much individual spread survives into the lineup sum?")
print("   1.00 = five independent players.  Near 0 = the team produces a fixed amount and the")
print("   individual rates only say who collected it.")
t = cons.groupby(["feature", "side"])["survives"].mean().unstack().reindex(FE)
print(t.round(3).to_string())

print("\n=== 2. the assembled prior: does a player's prior survive into his lineup?")
print(comp.groupby("side")[["sd_player_prior", "sd_lineup_prior", "survives"]].mean().round(3).to_string())

print("\n=== 3. does window w predict a PURE on-court RAPM in window w+1?")
mids = sorted(rap.mid.unique())
j = rap[rap.mid.isin(mids[:-1])].merge(
    rap[rap.mid.isin(mids[1:])][["player_id", "mid", "poss", "rapm0_total", "rapm0_off", "rapm0_def"]]
    .assign(mid=lambda d: d.mid - 3),
    on=["player_id", "mid"], suffixes=("", "_next"))
j = j[(j.poss >= 3000) & (j.poss_next >= 3000)]
print(f"   {len(j)} players seen in two consecutive windows with 3000+ possessions in both")


def wls(X, y, w, names):
    """Weighted least squares with honest standard errors.

    The weights must be normalised to mean 1 first: with raw possessions (order 1e4) the
    (X' W X)^-1 term is 1e4 times too small and every z score comes out in the thousands.
    """
    w = np.asarray(w, dtype=float)
    w = w / w.mean()
    A = np.column_stack([np.ones(len(y))] + list(X))
    sw = np.sqrt(w)[:, None]
    b, *_ = np.linalg.lstsq(A * sw, y * np.sqrt(w), rcond=None)
    r = y - A @ b
    s2 = np.sum(w * r ** 2) / (len(y) - A.shape[1])
    se = np.sqrt(np.diag(np.linalg.pinv((A * w[:, None]).T @ A)) * s2)
    return pd.DataFrame({"term": ["const"] + names, "coef": b, "se": se, "z": b / se})


wj = np.minimum(j.poss, j.poss_next).to_numpy()
for side in ("total", "off", "def"):
    y = j[f"rapm0_{side}_next"].to_numpy()
    print(f"\n   predicting next-window zero-prior RAPM, {side}:")
    print("     " + wls([j[f"prior_{side}"].to_numpy(), j[f"u_{side}"].to_numpy()], y, wj,
                        ["prior", "u"]).round(3).to_string().replace("\n", "\n     "))
    print("     with a bigness term added (negative = the box prior over-credits bigs):")
    print("     " + wls([j[f"prior_{side}"].to_numpy(), j[f"u_{side}"].to_numpy(),
                         j.bigness.to_numpy()], y, wj,
                        ["prior", "u", "bigness"]).round(3).to_string().replace("\n", "\n     "))

print("\n=== 4. parsimony: where does the prior's spread come from?")
last = rap[rap.window == window_label(list(range(cfg['windows'][-1][0], cfg['windows'][-1][1] + 1)))]
wd = build_window(list(range(cfg["windows"][-1][0], cfg["windows"][-1][1] + 1)), cfg)
cf = crossfit_beta(wd, lams=[LAM_B], cv=2, lam_ratio=RAT_B, pad_target=cfg["pad_target"])
pipe = plugin_fit(wd, cf.beta_, lams=[LAM_P], cv=2, lam_ratio=RAT_P, pad_target=cfg["pad_target"])
exp = pipe["exposure"]
ro = exp.season_rates_ - exp.means_o_ / 5.0
rd = exp.season_rates_d_ - exp.means_d_ / 5.0
p = exp.season_poss_off_
rows = []
for j_, f in enumerate(FE):
    co = ro[:, j_] * cf.beta_[j_]
    cd = -rd[:, j_] * cf.beta_[nf + j_]
    rows.append(dict(feature=f, sd_contrib_off=np.sqrt(np.average((co - np.average(co, weights=p)) ** 2, weights=p)),
                     sd_contrib_def=np.sqrt(np.average((cd - np.average(cd, weights=p)) ** 2, weights=p))))
v = pd.DataFrame(rows)
v["sd_total"] = np.sqrt(v.sd_contrib_off ** 2 + v.sd_contrib_def ** 2)
v = v.sort_values("sd_total", ascending=False)
print(v.round(3).to_string(index=False))
print(f"\nwrote outputs/investigate_*.parquet  ({time.time() - t0:.0f}s)")
