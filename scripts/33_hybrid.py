"""Which internal criterion can choose the defensive shrinkage?  Neither of the obvious ones.

The hybrid -- box prior priced to predict a PLAYER on offense, no box prior at all on defense -- was
reported as row 7 of the ladder (total 0.853) but never existed in code; the number was read off
outputs/ladder.parquet by hand.  Reproducing it turned up something more useful than the number.

The defensive columns are penalised by `lam * lam_ratio` (estimator._scale divides them by
sqrt(lam_ratio), so a scalar ridge `lam` on the scaled design is `lam * lam_ratio` on the raw one).
Sweeping that single quantity with the offense held fixed:

  * the LADDER'S INTERNAL CRITERION -- rank agreement with the player's own next-window pure
    on-court impact -- is FLAT across the four weakest penalties (0.4797 to 0.4813, well inside
    noise) and then declines monotonically.  It has no interior optimum, so it cannot pick a value;
    it just picks whichever weak-end grid point wins a coin flip.  That is what you expect when the
    benchmark is built from the same margin data as the estimate and shares its blind spot -- the
    same failure as scripts/26_which_loss.py, where held-out stint MSE preferred the wrong end of
    the defensive prior weight at 3.4 standard errors while moving 0.065%.
  * the CONSENSUS has a clear interior optimum near an effective penalty of 6000.

So HANDOFF's row 7 knobs (effective penalty 1723) are a criterion hitting its grid boundary, not a
choice.  We do not take the consensus argmax either -- that would be fitting to the validation set.
What ships is `lam_plugin * lam_ratio_plugin` = 5271, the constant the project already used, chosen
in an earlier session by scripts/03_cv.py and scripts/16_tune_plugin.py before any of this existed.
Not moving it is the only option here that is not hindsight.

usage: python scripts/33_hybrid.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 4)
cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
LAB = "2024-2026"
MIN_POSS = 1000          # the scoring population, same as the ladder and the tests
SEL_POSS = 3000          # the internal-selection population, same as the ladder
SHIPPED = float(cfg["lam_plugin"]) * float(cfg["lam_ratio_plugin"])

D = pd.read_parquet(OUT / "ladder.parquet")
WINDOWS = sorted(D.window.unique())
mid = {lab: i for i, lab in enumerate(WINDOWS)}

# the internal target: the same player's next-window pure on-court impact
z0 = pd.read_parquet(OUT / "tune_players.parquet")[
    ["window", "player_id", "poss_off", "rapm0_off_t18k", "rapm0_def_t18k"]]
z0["i"] = z0.window.map(mid)
nxt = z0.assign(i=lambda x: x.i - 1).rename(
    columns={"rapm0_off_t18k": "y_off", "rapm0_def_t18k": "y_def", "poss_off": "poss_next"})[
    ["i", "player_id", "y_off", "y_def", "poss_next"]]
con = pd.read_parquet(OUT / "vs_consensus.parquet")[
    ["player_id", "adj_offense", "adj_defense", "adj_overall", "bigness"]]

# ------------------------------------------------- the defensive penalty path, both criteria
# The hybrid's defense is the zero-prior fit's defense, so sweep that candidate's penalty.
rows = []
for (lam, ratio), g in D[(D.prior == "none") & (~D.boost)].groupby(["lam", "ratio"]):
    a = g[g.window != LAB].assign(i=lambda x: x.window.map(mid)).merge(nxt, on=["i", "player_id"])
    a = a[(a.poss >= SEL_POSS) & (a.poss_next >= SEL_POSS)]
    b = g[g.window == LAB].merge(con, on="player_id")
    b = b[b.poss >= MIN_POSS]
    rows.append(dict(lam=lam, ratio=ratio, eff_def=lam * ratio,
                     internal=spearmanr(a["def"], a.y_def).statistic,
                     consensus=spearmanr(b["def"], b.adj_defense).statistic,
                     spread=b["def"].std() / b.adj_defense.std()))
R = pd.DataFrame(rows).sort_values("eff_def")
R["mark"] = np.where(np.isclose(R.eff_def, SHIPPED), "  <-- ships", "")
print("=== the defensive penalty path (effective penalty = lam * lam_ratio), offense not involved")
print(R.to_string(index=False))
w = R[R.eff_def <= 3000].internal
print(f"\ninternal criterion: flat over the weakest four ({w.min():.4f} to {w.max():.4f}), then "
      f"monotone down -- no interior optimum, so it cannot choose")
print(f"consensus argmax  at eff_def = {R.loc[R.consensus.idxmax(), 'eff_def']:.0f}")
print(f"what ships        at eff_def = {SHIPPED:.0f}, chosen before any of this")
R.drop(columns="mark").to_csv(OUT / "csv" / "defensive_penalty_path.csv", index=False)

# ------------------------------------------------- the shipped board, scored once
rp = OUT / "player_ratings.parquet"
if not rp.exists():
    print(f"\n{rp} not built; skipping the scorecard")
    raise SystemExit(0)
S = pd.read_parquet(rp)
S = S[S.window == LAB][["player_id", "poss_off", "rating_off", "rating_def", "rating_total"]]
g = S.merge(con, on="player_id")
g = g[g.poss_off >= MIN_POSS]


def z(s):
    return (s - s.mean()) / s.std()


print(f"\n=== the shipped hybrid board vs the consensus, {LAB}, n = {len(g)}")
print(pd.Series(dict(
    total=spearmanr(g.rating_total, g.adj_overall).statistic,
    offense=spearmanr(g.rating_off, g.adj_offense).statistic,
    defense=spearmanr(g.rating_def, g.adj_defense).statistic,
    spread_off=g.rating_off.std() / g.adj_offense.std(),
    spread_def=g.rating_def.std() / g.adj_defense.std(),
    bias_total=(z(g.rating_total) - z(g.adj_overall)).corr(g.bigness),
)).round(4).to_string())
print("\nwrote outputs/csv/defensive_penalty_path.csv")
