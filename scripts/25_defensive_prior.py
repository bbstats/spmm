"""The defensive prior is wrong in the one direction the stint data cannot see. Fix it structurally.

Why the obvious internal fix fails (scripts/24).  Letting the stint regression estimate how far to
trust each side's box prior returns 1.12 on offense and 1.06 on defense: trust it MORE.  Scored
against the external consensus that is slightly worse, and it moves Robert Williams from 25th to
12th.  The reason is an identification limit, not a bug:

  the stint likelihood depends on players only through LINEUP SUMS, and the defensive prior's
  error is largely a reallocation of credit BETWEEN teammates, which leaves the lineup sum intact.

The conservation numbers say it exactly.  Of a player's own rate difference, the fraction that
survives into the lineup sum is 0.59 for defensive rebounds, 0.60 for offensive rebounds and 0.69
for blocks - the three stats the defensive prior is built from - against 0.90 for steals and 0.88
for missed twos.  A defensive rebound goes to exactly one of five players, so the team total is
nearly fixed and the individual rate mostly records who collected it.  Assembled, the offensive
prior survives at 1.006 and the defensive prior at 0.787.

So the defensive coefficients are right at the level they were estimated (lineups) and wrong at the
level they are applied (players).  A team that blocks more does allow fewer points; that does not
mean the man who blocked it is the man responsible.  No internal check can price this, because the
data is nearly invariant to it.  Something external has to set the weight.

This script does the structural version: scale the defensive half of the plug-in offset by `c`, so
the residual is refit knowing the prior is being trusted less, and sweep the offense/defense penalty
ratio at the same time.  Beta itself is untouched, so the published coefficient study does not move.

usage: python scripts/25_defensive_prior.py [--all-windows]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.cv import crossfit_beta, plugin_fit  # noqa: E402
from eracoef.windows import build_window, window_label, window_seasons  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
FE = cfg["features"]
nf = len(FE)
LAM_B, RAT_B = float(cfg["lam_beta"]), float(cfg["lam_ratio_beta"])
LAM_P, RAT_P = float(cfg["lam_plugin"]), float(cfg["lam_ratio_plugin"])
C_DEF = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0]   # 0.0 = drop the defensive box prior entirely
RATIOS = [RAT_P, 0.5, 1.0]
WIN = [(2024, 2026)] if "--all-windows" not in sys.argv else window_seasons(cfg)

con = pd.read_parquet(OUT / "vs_consensus.parquet")[
    ["player_id", "player_name", "adj_offense", "adj_defense", "adj_overall", "bigness"]]

out = []
t0 = time.time()
for w in WIN:
    seasons = list(range(w[0], w[1] + 1))
    lab = window_label(seasons)
    wd = build_window(seasons, cfg)
    m_ps = wd.spec.n_ps
    cf = crossfit_beta(wd, lams=[LAM_B], cv=2, lam_ratio=RAT_B, pad_target=cfg["pad_target"])
    beta = cf.beta_
    ids = wd.spec.ps_table["player_id"].to_numpy()
    for c in C_DEF:
        b = beta.copy()
        b[nf:] = beta[nf:] * c              # trust the defensive box score less; beta itself is safe
        for ratio in RATIOS:
            fit = plugin_fit(wd, b, lam=LAM_P, lam_ratio=ratio, pad_target=cfg["pad_target"])
            mm, exp = fit["mm"], fit["exposure"]
            ro = exp.season_rates_ - exp.means_o_ / 5.0
            rd = exp.season_rates_d_ - exp.means_d_ / 5.0
            out.append(pd.DataFrame({
                "window": lab, "c_def": c, "ratio": ratio, "player_id": ids,
                "poss_off": exp.season_poss_off_,
                "rating_off": ro @ b[:nf] + mm.u_[:m_ps],
                "rating_def": -(rd @ b[nf:]) - mm.u_[m_ps:]}))
        print(f"  {lab} c={c} done ({time.time() - t0:.0f}s)", flush=True)
    del wd

d = pd.concat(out, ignore_index=True)
d["rating_total"] = d.rating_off + d.rating_def
d.to_parquet(OUT / "defensive_prior_grid.parquet", index=False)

j = d[d.window == "2024-2026"].merge(con, on="player_id", how="inner")
rows = []
for (c, ratio), g in j.groupby(["c_def", "ratio"]):
    z = lambda s: (s - s.mean()) / s.std()  # noqa: E731
    rows.append(dict(
        c_def=c, ratio=ratio,
        tot=spearmanr(g.rating_total, g.adj_overall).statistic,
        dfn=spearmanr(g.rating_def, g.adj_defense).statistic,
        off=spearmanr(g.rating_off, g.adj_offense).statistic,
        sd_def=g.rating_def.std() / g.adj_defense.std(),
        bias=(z(g.rating_total) - z(g.adj_overall)).corr(g.bigness)))
r = pd.DataFrame(rows)
print(f"\n=== scored against the external consensus, {j.player_id.nunique()} players, 2024-26")
print("   c_def = how far the defensive box prior is trusted (1.0 is what ships)")
print("   ratio = defensive penalty / offensive penalty (0.287 ships; 1.0 is equal shrinkage)")
print(r.sort_values("tot", ascending=False).round(3).to_string(index=False))

b = r.loc[r.tot.idxmax()]
base = r[(r.c_def == 1.0) & (r.ratio == RAT_P)].iloc[0]
print(f"\n   ships:  c_def 1.00 ratio {RAT_P:.4f}  total {base.tot:.3f}  defense {base.dfn:.3f}  "
      f"defensive spread {base.sd_def:.2f}x  bigness bias {base.bias:+.3f}")
print(f"   best:   c_def {b.c_def:.2f} ratio {b.ratio:.4f}  total {b.tot:.3f}  defense {b.dfn:.3f}  "
      f"defensive spread {b.sd_def:.2f}x  bigness bias {b.bias:+.3f}")

best = j[(j.c_def == b.c_def) & (j.ratio == b.ratio)].copy()
old = j[(j.c_def == 1.0) & (j.ratio == RAT_P)][["player_id", "rating_total"]].rename(
    columns={"rating_total": "old"})
best = best.merge(old, on="player_id")
best["rk_new"] = best.rating_total.rank(ascending=False)
best["rk_old"] = best.old.rank(ascending=False)
best["rk_con"] = best.adj_overall.rank(ascending=False)
print("\n=== the anchors at the best setting")
t = best[best.player_name.str.contains(
    "Robert Williams|Nurki|Jonathan Isaac|Kornet|Sharpe|Diabat|Mitchell Robinson|Gobert|"
    "Trae Young|LaMelo|Booker|Curry|Joki|Wembanyama", case=False, na=False)]
t = t[["player_name", "rk_old", "rk_new", "rk_con"]].sort_values("rk_con")
t.columns = ["player", "ships", "fixed", "consensus"]
print(t.to_string(index=False))

print("\n=== the top 20 at the best setting")
t = best.nsmallest(20, "rk_new")[["player_name", "rating_total", "rk_new", "rk_con", "rk_old"]]
t.columns = ["player", "rating", "new rank", "consensus", "old rank"]
print(t.to_string(index=False))
r.to_csv(OUT / "csv" / "defensive_prior_grid.csv", index=False)
print("\nwrote outputs/csv/defensive_prior_grid.csv")
