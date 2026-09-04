"""Fit the box-score prior the xRAPM way: predict a player's RAPM, not the team's margin.

The diagnosis in one line: our prior is built like Wins Produced and needs to be built like xRAPM.

Today `beta` is estimated by regressing stint margin on the LINEUP SUM of box rates.  That answers a
team-economics question - what is a team's rebounding worth to its margin - and the answer is real.
We then apply it to individuals, which silently assumes the man who collected the rebound is the man
who created it.  For stats a team produces a nearly fixed amount of, that assumption is mostly
false, and it is exactly the Wins Produced failure mode: rebounders float to the top.

xRAPM never asks the box score what a rebound is worth.  It asks what a player's box line predicts
about HIS OWN RAPM, by regressing player-level RAPM on player-level box rates.  For a conserved stat
that regression discounts itself automatically, because collecting more rebounds than your teammates
does not by itself predict a better individual on-court impact.

So: fit the same 13 rates per side against a pure on-court RAPM at the PLAYER level, leave-one-
window-out so nothing scores itself, and compare the coefficients to the ones we publish.

The published coefficient study is a lineup-level question and stays exactly as it is.  This is a
second, player-level beta whose only job is to be a ratings prior.

usage: python scripts/27_xrapm_prior.py
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
LABEL = {"fg3m": "3PM", "fg3_miss": "3P miss", "fg2m": "2PM", "fg2_miss": "2P miss", "ftm": "FTM",
         "ft_miss": "FT miss", "orb": "ORB", "drb": "DRB", "ast": "AST", "tov": "TOV",
         "stl": "STL", "blk": "BLK", "pf": "PF"}

# ------------------------------------------------------------------ 1. the player-level panel
panels, betas = [], {}
t0 = time.time()
for w in window_seasons(cfg):
    seasons = list(range(w[0], w[1] + 1))
    lab = window_label(seasons)
    wd = build_window(seasons, cfg)
    m = wd.spec.n_ps
    cf = crossfit_beta(wd, lams=[LAM_B], cv=2, lam_ratio=RAT_B, pad_target=cfg["pad_target"])
    betas[lab] = cf.beta_
    # a pure on-court RAPM: no box score anywhere in it, and the shrinkage each player got
    zero = plugin_fit(wd, np.zeros(2 * nf), lams=[LAM_P], cv=2, lam_ratio=RAT_P,
                      pad_target=cfg["pad_target"])
    mm, exp = zero["mm"], zero["exposure"]
    a = mm.shrinkage_diag()
    ro = exp.season_rates_ - exp.means_o_ / 5.0
    rd = exp.season_rates_d_ - exp.means_d_ / 5.0
    for side, R, u, aa, poss in (("O", ro, mm.u_[:m], a[:m], exp.season_poss_off_),
                                 ("D", rd, mm.u_[m:], a[m:], exp.season_poss_def_)):
        d = pd.DataFrame(R, columns=FE)
        d["window"], d["side"] = lab, side
        d["player_id"] = wd.spec.ps_table["player_id"].to_numpy()
        d["poss"], d["a"], d["u"] = poss, aa, u
        # de-shrunk target with the Fay-Herriot weight: Var(u_i) = tau2 * a_i, so u_i/a_i has
        # variance tau2/a_i and the right weight is a_i
        d["target"] = np.where(aa > 1e-9, u / np.maximum(aa, 1e-9), 0.0)
        panels.append(d)
    print(f"  {lab} ({time.time() - t0:.0f}s)", flush=True)
    del wd
P = pd.concat(panels, ignore_index=True)
P.to_parquet(OUT / "xrapm_panel.parquet", index=False)


def fit_side(d, min_poss=2000):
    """Possession-weighted ridge of the de-shrunk player RAPM on his own 13 rates."""
    d = d[(d.poss >= min_poss) & (d.a > 1e-9)]
    X = np.column_stack([np.ones(len(d)), d[FE].to_numpy()])
    y = d.target.to_numpy()
    w = d.a.to_numpy()
    sw = np.sqrt(w)[:, None]
    A = X * sw
    pen = np.eye(X.shape[1]) * 1e-6
    pen[0, 0] = 0.0
    b = np.linalg.solve(A.T @ A + pen, (X * w[:, None]).T @ y)
    return b[1:]


# ------------------------------------------------------------------ 2. the two betas side by side
lin_o = np.mean([betas[k][:nf] for k in betas], axis=0)
lin_d = np.mean([betas[k][nf:] for k in betas], axis=0)
ply_o = fit_side(P[P.side == "O"])
ply_d = fit_side(P[P.side == "D"])
cmp = pd.DataFrame({
    "stat": [LABEL[f] for f in FE],
    "team_O": lin_o, "player_O": ply_o, "ratio_O": ply_o / np.where(np.abs(lin_o) > 1e-9, lin_o, np.nan),
    "team_D": -lin_d, "player_D": -ply_d, "ratio_D": ply_d / np.where(np.abs(lin_d) > 1e-9, lin_d, np.nan),
})
print("\n=== the same 13 stats, priced two ways")
print("   team_*   : regress team margin on the LINEUP SUM of rates (what we publish)")
print("   player_* : regress a player's own RAPM on his own rates (the xRAPM way)")
print("   ratio    : how much of the team-level price survives at the player level")
print("   defense is shown flipped, so positive = good on both sides")
print(cmp.round(3).to_string(index=False))
print("\n   Conserved stats are the ones that collapse. A defensive rebound is worth a lot to a")
print("   team and much less to the individual who happens to collect it.")

# ------------------------------------------------------------------ 3. leave-one-window-out ratings
con = pd.read_parquet(OUT / "vs_consensus.parquet")[
    ["player_id", "player_name", "adj_offense", "adj_defense", "adj_overall", "bigness"]]
LAB = "2024-2026"
b_o = fit_side(P[(P.side == "O") & (P.window != LAB)])
b_d = fit_side(P[(P.side == "D") & (P.window != LAB)])
print(f"\n=== rebuilt on {LAB} with a prior fit on the other nine windows only")
wd = build_window([2024, 2025, 2026], cfg)
m = wd.spec.n_ps
ids = wd.spec.ps_table["player_id"].to_numpy()
cf = crossfit_beta(wd, lams=[LAM_B], cv=2, lam_ratio=RAT_B, pad_target=cfg["pad_target"])
rows = []
for name, beta in (("team-level prior (ships)", cf.beta_),
                   ("player-level prior (xRAPM)", np.concatenate([b_o, b_d]))):
    for ratio in (RAT_P, 0.5):
        fit = plugin_fit(wd, beta, lam=LAM_P, lam_ratio=ratio, pad_target=cfg["pad_target"])
        mm, exp = fit["mm"], fit["exposure"]
        ro = exp.season_rates_ - exp.means_o_ / 5.0
        rd = exp.season_rates_d_ - exp.means_d_ / 5.0
        g = pd.DataFrame({"player_id": ids, "poss": exp.season_poss_off_,
                          "off": ro @ beta[:nf] + mm.u_[:m],
                          "dfn": -(rd @ beta[nf:]) - mm.u_[m:]}).merge(con, on="player_id")
        g["tot"] = g.off + g.dfn
        z = lambda s: (s - s.mean()) / s.std()  # noqa: E731
        rows.append(dict(prior=name, ratio=ratio,
                         total=spearmanr(g.tot, g.adj_overall).statistic,
                         defense=spearmanr(g.dfn, g.adj_defense).statistic,
                         offense=spearmanr(g.off, g.adj_offense).statistic,
                         sd_def=g.dfn.std() / g.adj_defense.std(),
                         bias=(z(g.tot) - z(g.adj_overall)).corr(g.bigness)))
        if name.startswith("player") and ratio == 0.5:
            best = g.copy()
print(pd.DataFrame(rows).round(3).to_string(index=False))

best["rk_new"] = best.tot.rank(ascending=False)
best["rk_con"] = best.adj_overall.rank(ascending=False)
print("\n=== the xRAPM-prior top 20")
t = best.nsmallest(20, "rk_new")[["player_name", "tot", "rk_new", "rk_con"]]
t.columns = ["player", "rating", "rank", "consensus"]
print(t.to_string(index=False))
print("\n=== anchors")
t = best[best.player_name.str.contains(
    "Robert Williams|Nurki|Jonathan Isaac|Kornet|Sharpe|Diabat|Mitchell Robinson|Gobert|"
    "Trae Young|LaMelo|Booker|Curry|Joki|Wembanyama", case=False, na=False)]
t = t[["player_name", "rk_new", "rk_con"]].sort_values("rk_con")
t.columns = ["player", "xRAPM-prior rank", "consensus"]
print(t.to_string(index=False))
cmp.to_csv(OUT / "csv" / "xrapm_prior_coefs.csv", index=False)
print("\nwrote outputs/csv/xrapm_prior_coefs.csv and outputs/xrapm_panel.parquet")
