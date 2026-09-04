"""One scorecard, with every definition stated. What is scored, against what, by which metric.

Everything reported in this conversation has been Spearman rank correlation against an external
consensus, on one window, with varying amounts of in-sample selection. This script puts all of it in
one place with the population fixed, the provenance of each candidate stated, both weighted and
unweighted metrics, and a paired bootstrap so it is clear which gaps are real.

POPULATION
  Our window            2024-2026, i.e. the 2023-24, 2024-25 and 2025-26 regular seasons pooled
                        into one three-year player unit.
  Benchmark             data/external/consensus.csv, a published blend of the modern all-in-one
                        metrics: player_name, team, adj_offense, adj_defense, adj_overall, all in
                        points per 100 possessions. 582 rows, one snapshot.
  Join                  by accent- and suffix-stripped lower-case name. No player ids exist on the
                        benchmark side. 581 of 582 match.
  Filter                >= 1000 offensive possessions in our window, leaving 475 players.

METRIC
  Spearman rank correlation is the headline, because the deliverable is a leaderboard and the two
  sides are on different scales (our total spread 3.42 against their 2.44). Pearson and a
  possession-weighted Spearman are reported beside it so the choice is visible rather than assumed.
  Differences carry a paired bootstrap over players, resampled together so the comparison is on the
  same resample.

A CAVEAT THAT APPLIES TO EVERY ROW
  The benchmark is a single snapshot; our window pools three seasons. A player who aged or changed
  role across those seasons is a different player to each side. That depresses every correlation
  here, roughly equally, so it is a level effect rather than something that favours one candidate.

usage: python scripts/29_scorecard.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
rng = np.random.default_rng(0)

m = pd.read_parquet(OUT / "vs_consensus.parquet")
st = pd.read_parquet(OUT / "tune_players.parquet")
st = st[st.window == "2024-2026"][["player_id", "rapm0_off_t18k", "rapm0_def_t18k"]]
d = m.merge(st, on="player_id", how="inner")
print(f"population: {len(d)} players\n")


def z(s):
    s = np.asarray(s, dtype=float)
    return (s - s.mean()) / s.std()


def wspearman(x, y, w):
    """Possession-weighted correlation of the ranks."""
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    w = np.asarray(w, dtype=float)
    mx, my = np.average(rx, weights=w), np.average(ry, weights=w)
    cov = np.average((rx - mx) * (ry - my), weights=w)
    return float(cov / np.sqrt(np.average((rx - mx) ** 2, weights=w) *
                              np.average((ry - my) ** 2, weights=w)))


# --------------------------------------------------------------- the candidates, with provenance
# leave-one-team-out reliability blend of the box prior and the pure on-court RAPM
blend = {}
for side, pri, rap, tgt in (("off", "prior_off", "rapm0_off_t18k", "adj_offense"),
                            ("def", "prior_def", "rapm0_def_t18k", "adj_defense")):
    oof = np.zeros(len(d))
    zp, zr = z(d[pri]), z(d[rap])
    for t in d.team.unique():
        te = (d.team == t).to_numpy()
        tr = ~te
        A = np.column_stack([np.ones(tr.sum()), zp[tr], zr[tr]])
        b, *_ = np.linalg.lstsq(A, d[tgt].to_numpy()[tr], rcond=None)
        oof[te] = b[0] + b[1] * zp[te] + b[2] * zr[te]
    blend[side] = oof
d["blend_off"], d["blend_def"] = blend["off"], blend["def"]
d["blend_total"] = d.blend_off + d.blend_def

CAND = [
    ("what ships", "rating", "prior + boosted correction + on-court residual, all at weight 1",
     "no selection"),
    ("box prior alone", "prior", "padded 3-year rates times the published lineup-level beta",
     "no selection"),
    ("pure on-court RAPM", "rapm0", "plug-in fit with beta set to zero, penalty 18352, ratio 0.2872",
     "penalty inherited, chosen on stint MSE"),
    ("reliability blend", "blend", "per-side least-squares blend of the two rows above",
     "leave-one-team-out"),
]
TGT = {"off": "adj_offense", "def": "adj_defense", "total": "adj_overall"}


def col(cand, side):
    if cand == "rapm0":
        return "rapm0_off_t18k" if side == "off" else "rapm0_def_t18k"
    return f"{cand}_{side}"


rows = []
for label, cand, what, sel in CAND:
    for side in ("off", "def", "total"):
        if cand == "rapm0" and side == "total":
            x = d.rapm0_off_t18k + d.rapm0_def_t18k
        elif cand == "prior" and side == "total":
            x = d.prior_total
        else:
            x = d[col(cand, side)]
        t = d[TGT[side]]
        rows.append(dict(candidate=label, side=side,
                         spearman=spearmanr(x, t).statistic,
                         pearson=np.corrcoef(z(x), z(t))[0, 1],
                         w_spearman=wspearman(x, t, d.poss_off),
                         spread_ratio=np.std(x) / np.std(t),
                         selection=sel))
R = pd.DataFrame(rows)
print("=== every candidate, every side")
print("   spread_ratio is our spread over the benchmark's; 1.0 means we disagree about who,")
print("   not about how far apart players are.")
for side in ("total", "off", "def"):
    print(f"\n  --- {side} (target: {TGT[side]})")
    print(R[R.side == side][["candidate", "spearman", "pearson", "w_spearman", "spread_ratio",
                             "selection"]].to_string(index=False))

# --------------------------------------------------------------- paired bootstrap on the gaps
print("\n=== are the gaps real? paired bootstrap over players, 2000 resamples, 95% interval")
print("   both candidates are scored on the SAME resample, so this is a paired comparison")
pairs = [("what ships", "reliability blend"), ("what ships", "pure on-court RAPM"),
         ("pure on-court RAPM", "reliability blend"), ("what ships", "box prior alone")]


def series(label, side):
    cand = dict((c[0], c[1]) for c in CAND)[label]
    if cand == "rapm0" and side == "total":
        return (d.rapm0_off_t18k + d.rapm0_def_t18k).to_numpy()
    if cand == "prior" and side == "total":
        return d.prior_total.to_numpy()
    return d[col(cand, side)].to_numpy()


idx = np.arange(len(d))
for side in ("total", "def", "off"):
    t = d[TGT[side]].to_numpy()
    print(f"\n  --- {side}")
    for a, b in pairs:
        xa, xb = series(a, side), series(b, side)
        diffs = np.empty(2000)
        for i in range(2000):
            s = rng.choice(idx, size=len(idx), replace=True)
            diffs[i] = (spearmanr(xb[s], t[s]).statistic - spearmanr(xa[s], t[s]).statistic)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        obs = spearmanr(xb, t).statistic - spearmanr(xa, t).statistic
        flag = "  significant" if lo > 0 or hi < 0 else "  NOT distinguishable"
        print(f"    {b:22s} minus {a:22s} {obs:+.3f}   95% [{lo:+.3f}, {hi:+.3f}]{flag}")

print("\n=== what selection was applied to each number")
print("   'no selection'        nothing about this candidate was chosen using the benchmark")
print("   'leave-one-team-out'  weights fit on 29 teams, scored on the held-out one")
print("   Numbers quoted earlier for a swept c_def or penalty ratio were chosen ON these same")
print("   475 players and are therefore optimistic; they are excluded here for that reason.")
R.to_csv(OUT / "csv" / "scorecard.csv", index=False)
print("\nwrote outputs/csv/scorecard.csv")
