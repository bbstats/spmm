"""Checkpoint 2: one real season, zero-prior RAPM (features=[], F only).

Prints top/bottom 15 by O, D, total in per-100 units, the shrinkage table, the lambda path,
and the stint-building diagnostics for the season.
usage: python scripts/02b_checkpoint2.py [season=2014] [min_poss=1000]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.boxtable import player_names, season_box  # noqa: E402
from eracoef.checks import player_ratings  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.cv import lam_grid, make_pipeline  # noqa: E402
from eracoef.design import build_design  # noqa: E402
from eracoef.stints import build_season, season_diagnostics  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 2)
cfg = load_config()
season = int(sys.argv[1]) if len(sys.argv) > 1 else 2014
min_poss = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

print(f"=== stints {season} RS ===")
t0 = time.time()
stints, diag = build_season(season, "RS", cfg)
print(f"build_season wall time (cached parse or fresh): {time.time() - t0:.1f}s; parse time recorded: {diag['seconds_total'].iloc[0]:.0f}s")
print(pd.Series(season_diagnostics(season, "RS", cfg)).to_string())

box = season_box([season], ["RS"], cfg)
wd = build_design(stints, box, cfg["features"], cfg)
print(f"\nrows={len(wd.y)}  player-seasons={wd.spec.n_ps}  games={len(wd.games)}  X={wd.X.shape}")
print(f"possessions per team-game: {wd.rows.poss.sum() / (2 * len(wd.games)):.1f}   stints per game: {len(stints) / len(wd.games):.1f}")

print("\n=== zero-prior mixed-model RAPM: features=[], lambda path (GroupKFold(5) by game) ===")
lams = lam_grid(cfg)
t0 = time.time()
pipe = make_pipeline(wd, lams=lams, cv=cfg["cv"]["n_folds"], features=[], mode="full")
pipe.fit(wd.X, wd.y, sample_weight=wd.w, groups=wd.groups)
mm = pipe["mm"]
print(f"fit {time.time() - t0:.1f}s")
print(mm.summary())
print(mm.lambda_report())
res = mm.cv_results_[["lam", "mean_mse", "diff_vs_best", "diff_se", "in_1se_band", "reml_neg2ll"]].copy()
res["reml_neg2ll"] -= res["reml_neg2ll"].min()
print(res.round(3).to_string(index=False))
th, d_med = mm.theta_median_starter()
print(f"theta at the median starter (d = {d_med:.0f} possessions): {th:.3f}")
print("\nF coefficients (gamma):")
print(pd.DataFrame({"name": wd.spec.f_names, "gamma": mm.gamma_, "se": mm.gamma_se_}).round(3).to_string(index=False))

names = player_names(box).set_index(["player_id", "season"])["player_name"]
rt = player_ratings(pipe, wd.spec)
rt["name"] = [names.get((p, s), str(p)) for p, s in zip(rt.player_id, rt.season)]
rt = rt[["name", "poss", "O", "D", "total"]]
big = rt[rt.poss >= min_poss]
for col in ("O", "D", "total"):
    print(f"\n--- top 15 by {col} (>= {min_poss} possessions; n = {len(big)}) ---")
    print(big.sort_values(col, ascending=False).head(15).to_string(index=False))
    print(f"--- bottom 15 by {col} ---")
    print(big.sort_values(col).head(15).to_string(index=False))
print("\nrating spread (sd) by possession bucket:")
rt["bucket"] = pd.cut(rt.poss, [0, 500, 1500, 1e9], labels=["<500", "500-1500", ">1500"])
print(rt.groupby("bucket", observed=True)[["O", "D", "total"]].agg(["std", "count"]).round(2).to_string())

print("\n=== shrinkage table a_i = diag(G (G+lam I)^-1) ===")
print(mm.shrinkage_table().round(3).to_string(index=False))
