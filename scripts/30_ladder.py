"""Does any of our machinery beat plain prior-informed RAPM?

The consensus CSV is VALIDATION ONLY here.  Nothing below is tuned on it, looked at against it, or
selected by it.  Every hyper-parameter is chosen by an internal criterion computed on windows that
end before 2024-26; the consensus is read once, at the end, to score the finished candidates.

The ladder, named for what each thing actually is:

  1. RAPM, no prior           ridge on player effects with the box score removed entirely
  2. box score, team-priced   the player's rates times the published beta, which was fit by
                              regressing team margin on LINEUP SUMS.  No on-court term at all.
  3. box score, player-priced the same rates times a beta fit by regressing a player's own rates on
                              his own NEXT-window on-court impact.  No on-court term at all.
  4. prior-informed RAPM,     ridge on player effects shrunk toward candidate 2 instead of toward
     team-priced prior        zero.  This is the textbook construction.
  5. prior-informed RAPM,     the same, shrunk toward candidate 3
     player-priced prior
  6. + boosted correction     candidate 4 with the gradient-boosted per-player correction added,
                              which is what the project currently ships

If 4 does not beat 1 and 2, the prior is not earning its place.  If 6 does not beat 4, the booster
is not earning its place.  Those are the two questions.

Internal selection criterion: for each candidate, the penalty and the offense/defense ratio are the
pair that best predicts a player's own NEXT-window pure on-court impact, pooled over the nine
earlier window pairs, possession-weighted.  No consensus anywhere in that.

usage: python scripts/30_ladder.py
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
from eracoef.windows import (build_window, player_priced_beta, window_label,  # noqa: E402
                             window_seasons)

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
FE = cfg["features"]
nf = len(FE)
LAM_B, RAT_B = float(cfg["lam_beta"]), float(cfg["lam_ratio_beta"])
LAMS = [3000.0, 6000.0, 12000.0, 18351.8, 30000.0, 50000.0]
RATIOS = [0.2872, 0.5, 1.0]
LAB = "2024-2026"
MIN_POSS = 2000

# ---------------------------------------------------------------- the player-priced beta
P = pd.read_parquet(OUT / "xrapm_panel.parquet")
WINDOWS = sorted(P.window.unique())
NXT = {WINDOWS[i]: WINDOWS[i + 1] for i in range(len(WINDOWS) - 1)}


def player_beta(exclude=()):
    """Regress a player's own rates on his own next-window on-court impact. No consensus involved."""
    return player_priced_beta(P, FE, exclude=exclude, min_poss=MIN_POSS)


B_PLAYER = player_beta(exclude=(LAB,))

# ---------------------------------------------------------------- build every window, every knob
boost = pd.read_parquet(OUT / "ratings_boosted.parquet")[["window", "player_id", "boost_off", "boost_def"]]
rows = []
t0 = time.time()
for w in window_seasons(cfg):
    seasons = list(range(w[0], w[1] + 1))
    lab = window_label(seasons)
    wd = build_window(seasons, cfg)
    m = wd.spec.n_ps
    ids = wd.spec.ps_table["player_id"].to_numpy()
    cf = crossfit_beta(wd, lams=[LAM_B], cv=2, lam_ratio=RAT_B, pad_target=cfg["pad_target"])
    bt = boost[boost.window == lab].set_index("player_id").reindex(ids).fillna(0.0)
    g_off = bt.boost_off.to_numpy()
    g_def = bt.boost_def.to_numpy()
    offset = np.concatenate([g_off, -g_def])
    PRIORS = {"none": np.zeros(2 * nf), "team": cf.beta_, "player": B_PLAYER}
    for pname, beta in PRIORS.items():
        for ratio in RATIOS:
            for use_boost in ((False, True) if pname == "team" else (False,)):
                pipe = plugin_fit(wd, beta, lams=LAMS, cv=2, lam_ratio=ratio,
                                  pad_target=cfg["pad_target"],
                                  prior_offset=offset if use_boost else None)
                exp = pipe["exposure"]
                ro = exp.season_rates_ - exp.means_o_ / 5.0
                rd = exp.season_rates_d_ - exp.means_d_ / 5.0
                pri_o, pri_d = ro @ beta[:nf], -(rd @ beta[nf:])
                for lam in LAMS:
                    pipe["mm"].set_lam(lam)
                    o = pri_o + pipe["mm"].u_[:m] + (g_off if use_boost else 0.0)
                    d_ = pri_d - pipe["mm"].u_[m:] + (g_def if use_boost else 0.0)
                    rows.append(pd.DataFrame({
                        "window": lab, "prior": pname, "boost": use_boost, "lam": lam,
                        "ratio": ratio, "player_id": ids, "poss": exp.season_poss_off_,
                        "off": o, "def": d_,
                        # the box-only candidates, which have no on-court term at all
                        "pri_off": pri_o, "pri_def": pri_d}))
    print(f"  {lab} ({time.time() - t0:.0f}s)", flush=True)
    del wd
D = pd.concat(rows, ignore_index=True)
D["total"] = D["off"] + D["def"]
D.to_parquet(OUT / "ladder.parquet", index=False)

# ---------------------------------------------------------------- internal selection
mid = {lab: i for i, lab in enumerate(WINDOWS)}
z0 = pd.read_parquet(OUT / "tune_players.parquet")[
    ["window", "player_id", "poss_off", "rapm0_off_t18k", "rapm0_def_t18k"]]
z0["i"] = z0.window.map(mid)
nxt = z0.assign(i=lambda x: x.i - 1).rename(
    columns={"rapm0_off_t18k": "y_off", "rapm0_def_t18k": "y_def", "poss_off": "poss_next"})[
    ["i", "player_id", "y_off", "y_def", "poss_next"]]


def internal(dd):
    """Possession-weighted rank agreement with the player's own next-window on-court impact."""
    dd = dd.assign(i=dd.window.map(mid)).merge(nxt, on=["i", "player_id"], how="inner")
    dd = dd[(dd.poss >= 3000) & (dd.poss_next >= 3000)]
    if len(dd) < 200:
        return np.nan
    return spearmanr(dd["off"] + dd["def"], dd.y_off + dd.y_def).statistic


sel = []
for (p, b, lam, r), g in D[D.window != LAB].groupby(["prior", "boost", "lam", "ratio"]):
    sel.append(dict(prior=p, boost=b, lam=lam, ratio=r, internal=internal(g)))
S = pd.DataFrame(sel)
best = S.loc[S.groupby(["prior", "boost"]).internal.idxmax()]
print("\n=== knobs chosen by the internal criterion only (no consensus)")
print(best.to_string(index=False))

# ---------------------------------------------------------------- validation, read once
con = pd.read_parquet(OUT / "vs_consensus.parquet")[
    ["player_id", "adj_offense", "adj_defense", "adj_overall", "bigness"]]
NAME = {("none", False): "1. RAPM, no prior",
        ("team", False): "4. prior-informed RAPM, team-priced prior",
        ("player", False): "5. prior-informed RAPM, player-priced prior",
        ("team", True): "6. prior-informed RAPM + boosted correction (ships today)"}
out = []
for _, r in best.iterrows():
    g = D[(D.window == LAB) & (D.prior == r.prior) & (D.boost == r.boost) &
          (D.lam == r.lam) & (D.ratio == r.ratio)].merge(con, on="player_id")
    g = g[g.poss >= 1000]
    out.append(dict(candidate=NAME[(r.prior, r.boost)], n=len(g),
                    total=spearmanr(g.total, g.adj_overall).statistic,
                    offense=spearmanr(g["off"], g.adj_offense).statistic,
                    defense=spearmanr(g["def"], g.adj_defense).statistic))
# the two box-only candidates: no on-court term, so no knobs to choose
for pname, label in (("team", "2. box score, team-priced (no on-court term)"),
                     ("player", "3. box score, player-priced (no on-court term)")):
    g = D[(D.window == LAB) & (D.prior == pname) & (~D.boost)].drop_duplicates("player_id")
    g = g.merge(con, on="player_id")
    g = g[g.poss >= 1000]
    out.append(dict(candidate=label, n=len(g),
                    total=spearmanr(g.pri_off + g.pri_def, g.adj_overall).statistic,
                    offense=spearmanr(g.pri_off, g.adj_offense).statistic,
                    defense=spearmanr(g.pri_def, g.adj_defense).statistic))
R = pd.DataFrame(out).sort_values("candidate")
print(f"\n=== validation against the consensus, {LAB}, read once, nothing tuned on it")
print("   Spearman rank correlation")
print(R.to_string(index=False))
R.to_csv(OUT / "csv" / "ladder.csv", index=False)
print("\nwrote outputs/csv/ladder.csv and outputs/ladder.parquet")
