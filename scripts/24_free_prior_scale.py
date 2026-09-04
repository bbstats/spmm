"""Let the stint data say how far to trust each side's box prior, instead of assuming it fully.

Two principles drive this:

  1. RAPM on possessions is the basis, informed by a prior.  That part is already right: the
     plug-in fit carries the box prior as the offset `Z @ prior`, so a player's effect is shrunk
     toward his box-score expectation rather than toward zero.  This script does not change that.

  2. Adjust for sample size and luck as smartly as you can.  This is where the model was cheating.
     The prior enters at weight exactly 1.0 on both sides, by construction, which asserts that the
     box score measures defense exactly as reliably as it measures offense.  It does not.  Against
     an external consensus the offensive prior correlates +0.87 and the defensive one +0.53, so
     pricing both at 1.0 hands the defensive board to whoever collects rebounds and blocks.

The fix is to stop asserting it.  The lineup-sum prior becomes two free unpenalized columns, one
per side, and the stint regression estimates how far each is worth trusting.  A coefficient of 1
reproduces today's behaviour; below 1 says the prior carries noise that should regress toward the
mean.  Nothing here touches beta, so the coefficient study is unaffected.

This is a pure sample-size-and-luck correction, estimated from possessions, with no reference to
the external consensus.  The consensus is only used afterwards, to score it.

usage: python scripts/24_free_prior_scale.py
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

rows, scales = [], []
t0 = time.time()
for w in window_seasons(cfg):
    seasons = list(range(w[0], w[1] + 1))
    lab = window_label(seasons)
    wd = build_window(seasons, cfg)
    m_ps = wd.spec.n_ps
    cf = crossfit_beta(wd, lams=[LAM_B], cv=2, lam_ratio=RAT_B, pad_target=cfg["pad_target"])
    fit = plugin_fit(wd, cf.beta_, lam=LAM_P, lam_ratio=RAT_P, pad_target=cfg["pad_target"],
                     free_prior_scale=True)
    mm, exp = fit["mm"], fit["exposure"]
    c = mm.prior_scale_
    se = mm.prior_scale_se_
    scales.append(dict(window=lab, c_off=c[0], c_off_se=se[0], c_def=c[1], c_def_se=se[1]))
    ro = exp.season_rates_ - exp.means_o_ / 5.0
    rd = exp.season_rates_d_ - exp.means_d_ / 5.0
    rows.append(pd.DataFrame({
        "window": lab, "player_id": wd.spec.ps_table["player_id"].to_numpy(),
        "poss_off": exp.season_poss_off_,
        "prior_off": ro @ cf.beta_[:nf], "prior_def": -(rd @ cf.beta_[nf:]),
        "u_off": mm.u_[:m_ps], "u_def": -mm.u_[m_ps:],
        "c_off": c[0], "c_def": c[1]}))
    print(f"  {lab}: trust on offense {c[0]:.3f} +/- {se[0]:.3f}, "
          f"defense {c[1]:.3f} +/- {se[1]:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    del wd

sc = pd.DataFrame(scales)
d = pd.concat(rows, ignore_index=True)
d["rating_off"] = d.c_off * d.prior_off + d.u_off
d["rating_def"] = d.c_def * d.prior_def + d.u_def
d["rating_total"] = d.rating_off + d.rating_def
d.to_parquet(OUT / "free_scale_ratings.parquet", index=False)

print("\n=== how far the stint data says to trust the box prior, by era")
print(sc.round(3).to_string(index=False))
print(f"\n   pooled mean: offense {sc.c_off.mean():.3f}, defense {sc.c_def.mean():.3f}")
print("   1.0 is what the model asserts today. The data does not agree, and it disagrees")
print("   far more about defense than about offense.")

# ------------------------------------------------------------------ score against the consensus
con = OUT / "vs_consensus.parquet"
if not con.exists():
    print("\nrun scripts/22_vs_consensus.py first to score this")
    raise SystemExit
c = pd.read_parquet(con)[["player_id", "player_name", "team", "adj_offense", "adj_defense",
                          "adj_overall", "bigness", "rating_off", "rating_def", "rating_total"]]
c = c.rename(columns={"rating_off": "old_off", "rating_def": "old_def", "rating_total": "old_total"})
j = d[d.window == "2024-2026"].merge(c, on="player_id", how="inner")
print(f"\n=== scored against the external consensus, {len(j)} players")
print("   (the scales above were estimated from possessions only; the consensus is the held-out")
print("    check, not an input)")
for side, ours, old, target in (("offense", "rating_off", "old_off", "adj_offense"),
                                ("defense", "rating_def", "old_def", "adj_defense"),
                                ("total", "rating_total", "old_total", "adj_overall")):
    a = spearmanr(j[old], j[target]).statistic
    b = spearmanr(j[ours], j[target]).statistic
    ra = j[old].std() / j[target].std()
    rb = j[ours].std() / j[target].std()
    print(f"  {side:8s} rank {a:.3f} -> {b:.3f} ({b - a:+.3f})   spread {ra:.2f}x -> {rb:.2f}x")

for name, ours, old, target in (("total", "rating_total", "old_total", "adj_overall"),
                                ("defense", "rating_def", "old_def", "adj_defense")):
    za = (j[old] - j[old].mean()) / j[old].std()
    zb = (j[ours] - j[ours].mean()) / j[ours].std()
    zc = (j[target] - j[target].mean()) / j[target].std()
    print(f"  {name:8s} gap-vs-bigness {(za - zc).corr(j.bigness):+.3f} -> {(zb - zc).corr(j.bigness):+.3f}")

j["rk_new"] = j.rating_total.rank(ascending=False)
j["rk_old"] = j.old_total.rank(ascending=False)
j["rk_con"] = j.adj_overall.rank(ascending=False)
print("\n=== the anchors")
t = j[j.player_name.str.contains("Robert Williams|Nurki|Jonathan Isaac|Kornet|Sharpe|Diabat|"
                                 "Mitchell Robinson|Gobert|Trae Young|LaMelo|Booker|Curry|Joki",
                                 case=False, na=False)]
t = t[["player_name", "rk_old", "rk_new", "rk_con"]].sort_values("rk_con")
t.columns = ["player", "old rank", "free-scale rank", "consensus"]
print(t.to_string(index=False))

print("\n=== the free-scale top 20")
t = j.nsmallest(20, "rk_new")[["player_name", "rating_total", "rk_new", "rk_con", "rk_old"]]
t.columns = ["player", "rating", "new rank", "consensus", "old rank"]
print(t.to_string(index=False))
sc.to_csv(OUT / "csv" / "free_prior_scale.csv", index=False)
print("\nwrote outputs/free_scale_ratings.parquet and outputs/csv/free_prior_scale.csv")
