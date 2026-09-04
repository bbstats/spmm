"""Choose the ratings shrinkage against a PLAYER-level target, not held-out stint MSE.

`lam_plugin` and its O/D ratio were selected by minimising held-out stint mean squared error.  In a
stint regression the box prior is a low-variance, well-measured regressor and `u` is a high-variance
noisy one, so heavy shrinkage on `u` is right for predicting stints.  It is not right for ranking
players, and nothing in the pipeline ever selected these constants against a player-level target.
This script does.

Criterion.  For each pair of consecutive windows, rank players in window w by the candidate rating
`prior + u(lam, ratio)` and score it by the possession-weighted correlation with the same player's
window w+1 PURE ON-COURT RAPM: a zero-prior fit with no box score in it at all, measured on
different games with different teammates.  Offense scores against next-window offense, defense
against defense, total against total.

Beta is held fixed throughout at the same cross-fitted `lam_beta`, so nothing here touches the
coefficient study.

Outputs (to outputs/):
  tune_players.parquet   one row per player-window: prior, possessions, team, the target
  tune_u.parquet         one row per player-window per (lam, ratio): u_off, u_def
  csv/tune_plugin_grid.csv  the scored grid
usage: python scripts/16_tune_plugin.py [--rescore]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.boost import dominant_team  # noqa: E402
from eracoef.boxtable import season_box  # noqa: E402
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
RESCORE = "--rescore" in sys.argv

# The candidate grid.  lam is the offensive penalty; ratio is lambda_D / lambda_O.
LAMS = np.geomspace(200.0, 120000.0, 25)
RATIOS = [0.1, 0.18, 0.2872, 0.5, 1.0]
# The target is a zero-prior fit.  Two reference penalties, so the choice can be checked against a
# tighter and a looser target.
TARGET_LAMS = {"t18k": LAM_P, "t3k": 3000.0}


def build_panel():
    stat, longs = [], []
    t0 = time.time()
    for w in window_seasons(cfg):
        seasons = list(range(w[0], w[1] + 1))
        lab = window_label(seasons)
        wd = build_window(seasons, cfg)
        spec = wd.spec
        m = spec.n_ps
        cf = crossfit_beta(wd, lams=[LAM_B], cv=2, lam_ratio=RAT_B, pad_target=cfg["pad_target"])

        # the prior, exactly as scripts/08_ratings.py assembles it
        ref = plugin_fit(wd, cf.beta_, lam=LAM_P, lam_ratio=RAT_P, pad_target=cfg["pad_target"])
        exp = ref["exposure"]
        ro = exp.season_rates_ - exp.means_o_ / 5.0
        rd = exp.season_rates_d_ - exp.means_d_ / 5.0
        s = pd.DataFrame({
            "window": lab, "mid": float(np.mean(seasons)),
            "player_id": spec.ps_table["player_id"].to_numpy(),
            "poss_off": exp.season_poss_off_, "poss_def": exp.season_poss_def_,
            "team": dominant_team(wd, season_box(seasons, ["RS"], cfg)),
            "prior_off": ro @ cf.beta_[:nf], "prior_def": -(rd @ cf.beta_[nf:])})

        # the target: a pure on-court RAPM, no box score anywhere in it
        zp = plugin_fit(wd, np.zeros(2 * nf), lams=sorted(TARGET_LAMS.values()), cv=2,
                        lam_ratio=RAT_P, pad_target=cfg["pad_target"])
        for name, tl in TARGET_LAMS.items():
            zp["mm"].set_lam(tl)
            s[f"rapm0_off_{name}"] = zp["mm"].u_[:m]
            s[f"rapm0_def_{name}"] = -zp["mm"].u_[m:]
        stat.append(s)

        # the candidates: one eigen path per ratio, then set_lam is free
        for r in RATIOS:
            pipe = plugin_fit(wd, cf.beta_, lams=list(LAMS), cv=2, lam_ratio=r,
                              pad_target=cfg["pad_target"])
            for lam in LAMS:
                pipe["mm"].set_lam(lam)
                longs.append(pd.DataFrame({
                    "window": lab, "player_id": s.player_id.to_numpy(), "lam": lam, "ratio": r,
                    "u_off": pipe["mm"].u_[:m], "u_def": -pipe["mm"].u_[m:]}))
        print(f"  {lab}: {m} players ({time.time() - t0:.0f}s)", flush=True)
        del wd
    stat = pd.concat(stat, ignore_index=True)
    longs = pd.concat(longs, ignore_index=True)
    stat.to_parquet(OUT / "tune_players.parquet", index=False)
    longs.to_parquet(OUT / "tune_u.parquet", index=False)
    return stat, longs


if RESCORE and (OUT / "tune_players.parquet").exists():
    stat = pd.read_parquet(OUT / "tune_players.parquet")
    longs = pd.read_parquet(OUT / "tune_u.parquet")
else:
    stat, longs = build_panel()


# ------------------------------------------------------------------------------------- scoring
def wmean(x, w):
    return float(np.average(np.asarray(x, dtype=float), weights=np.asarray(w, dtype=float)))


def wcorr(x, y, w):
    """Possession-weighted Pearson correlation."""
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float) - wmean(x, w)
    y = np.asarray(y, dtype=float) - wmean(y, w)
    den = np.sqrt(wmean(x ** 2, w) * wmean(y ** 2, w))
    return float(wmean(x * y, w) / den) if den > 0 else np.nan


def prior_share(p, u, w):
    """Share of the rating's variance carried by the box prior (the handoff's alpha)."""
    w = np.asarray(w, dtype=float)
    p = np.asarray(p, dtype=float) - wmean(p, w)
    u = np.asarray(u, dtype=float) - wmean(u, w)
    tot = wmean((p + u) ** 2, w)
    return float(wmean(p ** 2, w) / tot) if tot > 0 else np.nan


def pairs(stat, target="t18k", min_poss=3000):
    """Window w rows joined to the same player's window w+1 target."""
    mids = sorted(stat.mid.unique())
    nxt = (stat[stat.mid.isin(mids[1:])][["player_id", "mid", "poss_off", "team",
                                          f"rapm0_off_{target}", f"rapm0_def_{target}"]]
           .assign(mid=lambda d: d.mid - 3)
           .rename(columns={"poss_off": "poss_next", "team": "team_next",
                            f"rapm0_off_{target}": "y_off", f"rapm0_def_{target}": "y_def"}))
    j = stat[stat.mid.isin(mids[:-1])].merge(nxt, on=["player_id", "mid"], how="inner")
    j = j[(j.poss_off >= min_poss) & (j.poss_next >= min_poss)].copy()
    j["y_total"] = j.y_off + j.y_def
    j["wt"] = np.minimum(j.poss_off, j.poss_next)
    return j


def demean_by_window(d, cols):
    """Weighted demeaning within each window, so a window-level level shift in either the rating
    or the target cannot masquerade as player-level agreement."""
    for c in cols:
        mu = d.groupby("window").apply(lambda g, c=c: wmean(g[c], g.wt), include_groups=False)
        d[c] = d[c] - d.window.map(mu)
    return d


def score_grid(stat, longs, target="t18k", min_poss=3000, mask=None):
    """Weighted correlation of prior + u(lam, ratio) with the next window's pure on-court RAPM."""
    j = pairs(stat, target, min_poss)
    if mask is not None:
        j = j[mask(j)].copy()
    keep = ["window", "player_id", "prior_off", "prior_def", "y_off", "y_def", "y_total", "wt"]
    j = demean_by_window(j[keep].copy(), ["y_off", "y_def", "y_total"])
    d = j.merge(longs, on=["window", "player_id"], how="inner")
    d["r_off"] = d.prior_off + d.u_off
    d["r_def"] = d.prior_def + d.u_def
    d["r_total"] = d.r_off + d.r_def
    d["p_total"] = d.prior_off + d.prior_def
    d["u_total"] = d.u_off + d.u_def
    rows = []
    for (lam, ratio), g in d.groupby(["lam", "ratio"]):
        g = demean_by_window(g.copy(), ["r_off", "r_def", "r_total"])
        rows.append(dict(
            lam=lam, ratio=ratio, n=len(g),
            corr_off=wcorr(g.r_off, g.y_off, g.wt),
            corr_def=wcorr(g.r_def, g.y_def, g.wt),
            corr_total=wcorr(g.r_total, g.y_total, g.wt),
            alpha=prior_share(g.p_total, g.u_total, g.wt),
            sd_rating=float(np.sqrt(wmean(g.r_total ** 2, g.wt)))))
    return pd.DataFrame(rows).sort_values(["ratio", "lam"]).reset_index(drop=True), j


def at_shipped(tab):
    g = tab[tab.ratio == RAT_P]
    return g.iloc[int((g.lam - LAM_P).abs().to_numpy().argmin())]


print("\n=== the grid: weighted correlation with the next window's pure on-court RAPM")
print(f"    target = zero-prior RAPM at lam {LAM_P:.0f}; 3000+ possessions in both windows")
tab, j = score_grid(stat, longs)
print(f"    {len(j)} player pairs")
best = tab.loc[tab.corr_total.idxmax()]
for r in RATIOS:
    g = tab[tab.ratio == r]
    b = g.loc[g.corr_total.idxmax()]
    print(f"  ratio {r:>6}: best lam {b.lam:9.0f}  total {b.corr_total:.4f}  "
          f"off {b.corr_off:.4f}  def {b.corr_def:.4f}  prior share {b.alpha:.3f}")
cur = at_shipped(tab)
print(f"\n  argmax over the whole grid: lam {best.lam:8.0f}, ratio {best.ratio:<7} "
      f"corr_total {best.corr_total:.4f}, prior variance share {best.alpha:.3f}")
print(f"  what we ship now:           lam {cur.lam:8.0f}, ratio {cur.ratio:<7} "
      f"corr_total {cur.corr_total:.4f}, prior variance share {cur.alpha:.3f}")

print("\n  per-side argmax:")
for side in ("off", "def"):
    b = tab.loc[tab[f"corr_{side}"].idxmax()]
    print(f"    {side}: lam {b.lam:9.0f} ratio {b.ratio:<7} corr {b[f'corr_{side}']:.4f}")

print("\n=== the lam profile at the best ratio")
print(tab[tab.ratio == best.ratio][["lam", "corr_off", "corr_def", "corr_total", "alpha", "sd_rating"]]
      .round(4).to_string(index=False))

print("\n=== sensitivity")
for label, kw in (("looser target (zero-prior lam 3000)", dict(target="t3k")),
                  ("possession floor 1500", dict(min_poss=1500)),
                  ("possession floor 6000", dict(min_poss=6000))):
    t, jj = score_grid(stat, longs, **kw)
    b = t.loc[t.corr_total.idxmax()]
    c = at_shipped(t)
    print(f"  {label:38s} n={len(jj):5d}  best lam {b.lam:8.0f} ratio {b.ratio:<7} "
          f"corr {b.corr_total:.4f} (shipped {c.corr_total:.4f})")

# Teammate carry-over: a player's u in window w is contaminated by his teammates, and if the same
# teammates recur in w+1 the same contamination sits in the target too.  That would bias the
# criterion toward too little shrinkage.  Split on whether the dominant team changed.
for label, mask in (("changed team between windows", lambda d: d.team != d.team_next),
                    ("stayed on the same team", lambda d: d.team == d.team_next)):
    t, jj = score_grid(stat, longs, mask=mask)
    b = t.loc[t.corr_total.idxmax()]
    c = at_shipped(t)
    print(f"  {label:38s} n={len(jj):5d}  best lam {b.lam:8.0f} ratio {b.ratio:<7} "
          f"corr {b.corr_total:.4f} (shipped {c.corr_total:.4f})")

(OUT / "csv").mkdir(parents=True, exist_ok=True)
tab.to_csv(OUT / "csv" / "tune_plugin_grid.csv", index=False)
print("\nwrote outputs/csv/tune_plugin_grid.csv, outputs/tune_players.parquet, outputs/tune_u.parquet")
