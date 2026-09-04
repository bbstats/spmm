"""What would it take to line the 2024-26 board up with the modern consensus?

scripts/22 establishes the defect: our defensive ratings have 2.1x the spread the consensus does,
the gap correlates +0.59 with bigness, and offensive rebounds (+0.55) and blocks (+0.51) are what
carries it.  Meanwhile the box prior is the WEAKER defensive signal (correlation with consensus
defense +0.53) yet supplies most of the defensive spread, while the on-court residual is the
STRONGER one (+0.66) and supplies little.  The weights are backwards on defense; on offense the box
prior is genuinely good (+0.87) and the spread ratio is already 1.05.

This script asks how much of that is recoverable by reweighting the three pieces we already have,
without new features and without touching the coefficient study:

    rating_side = a * prior_side + b * correction_side + c * residual_side

Three numbers per side, fit against the consensus.  That is very little capacity, so it is a
calibration rather than a copy, but it is still fit on the thing it is scored against, so every
number is also reported under leave-one-team-out cross-validation.

usage: python scripts/23_remedy.py
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
m = pd.read_parquet(OUT / "vs_consensus.parquet")
SIDES = [("def", "adj_defense", "Defense"), ("off", "adj_offense", "Offense")]
print(f"{len(m)} matched players, 2024-26, 1000+ possessions\n")


def fit(d, side, target):
    """Least squares weights on the three pieces, with an intercept."""
    X = np.column_stack([np.ones(len(d)), d[f"prior_{side}"], d[f"boost_{side}"], d[f"u_{side}"]])
    b, *_ = np.linalg.lstsq(X, d[target].to_numpy(), rcond=None)
    return b


def apply(d, side, b):
    return b[0] + b[1] * d[f"prior_{side}"] + b[2] * d[f"boost_{side}"] + b[3] * d[f"u_{side}"]


print("=== 1. what weights does the consensus imply?")
print("   the shipped rating uses 1.0 on all three pieces, by construction")
W = {}
for side, target, lab in SIDES:
    b = fit(m, side, target)
    W[side] = b
    print(f"  {lab:8s} prior {b[1]:.3f}   correction {b[2]:.3f}   residual {b[3]:.3f}")
print("\n   On defense the box prior should carry about a third of the weight it has, and the")
print("   on-court residual about twice. On offense the weights are already close to sensible.")

# leave-one-team-out, so a player is never scored by weights his own team helped fit
print("\n=== 2. does it hold up out of sample? (leave-one-team-out)")
for side, target, lab in SIDES:
    oof = pd.Series(index=m.index, dtype=float)
    for t in m.team.unique():
        tr, te = m[m.team != t], m.team == t
        oof[te] = apply(m[te], side, fit(tr, side, target))
    m[f"fix_{side}"] = oof
    base = spearmanr(m[f"rating_{side}"], m[target]).statistic
    new = spearmanr(oof, m[target]).statistic
    print(f"  {lab:8s} rank agreement {base:.3f} -> {new:.3f}   ({new - base:+.3f})")

m["fix_total"] = m.fix_off + m.fix_def
base = spearmanr(m.rating_total, m.adj_overall).statistic
new = spearmanr(m.fix_total, m.adj_overall).statistic
print(f"  {'Total':8s} rank agreement {base:.3f} -> {new:.3f}   ({new - base:+.3f})")

print("\n=== 3. how much of the bias goes away?")
for name in ("total", "def", "off"):
    col = f"fix_{name}"
    z_new = (m[col] - m[col].mean()) / m[col].std()
    z_con = m[f"z_con_{name}"]
    print(f"  {name:6s} gap-vs-bigness correlation {m[f'd_{name}'].corr(m.bigness):+.3f} -> "
          f"{(z_new - z_con).corr(m.bigness):+.3f}   "
          f"spread ratio {m[f'rating_{name}'].std() / m[f'adj_{ {'total':'overall','def':'defense','off':'offense'}[name] }'].std():.2f} -> "
          f"{m[col].std() / m[f'adj_{ {'total':'overall','def':'defense','off':'offense'}[name] }'].std():.2f}")

print("\n=== 4. where do the anchors land?")
m["rk_fix_total"] = m.fix_total.rank(ascending=False)
show = ["player_name", "rk_ours_total", "rk_fix_total", "rk_con_total"]
anchors = ["robert williams", "jusuf nurkic", "jonathan isaac", "moussa diabate", "luke kornet",
           "dayron sharpe", "mitchell robinson", "rudy gobert", "deandre jordan",
           "trae young", "lamelo ball", "devin booker", "stephen curry", "nikola jokic"]
t = m[m.key.isin(anchors)][show].copy()
t.columns = ["player", "our rank now", "reweighted", "consensus"]
print(t.sort_values("consensus").to_string(index=False))

print("\n=== 5. the reweighted top 20")
t = m.nsmallest(20, "rk_fix_total")[["player_name", "fix_total", "rk_fix_total", "rk_con_total",
                                     "rk_ours_total"]]
t.columns = ["player", "reweighted", "new rank", "consensus rank", "old rank"]
print(t.to_string(index=False))

print("\n=== 6. what is left over after the reweight?")
resid = m.fix_total - (m.adj_overall - m.adj_overall.mean()) / m.adj_overall.std() * m.fix_total.std()
print("   correlation of the remaining gap with:")
z_new = (m.fix_total - m.fix_total.mean()) / m.fix_total.std()
gap = z_new - m.z_con_total
for v in ("bigness", "poss_off", "minutes", "orb", "blk", "ast", "fg3m"):
    print(f"     {v:10s} {gap.corr(m[v]):+.3f}")

m.to_parquet(OUT / "remedy.parquet", index=False)
print("\nwrote outputs/remedy.parquet")

# ---------------------------------------------------------------------------------------------
# 7. Is the O/D penalty ratio the lever?  It is the obvious suspect: config ships
# lam_ratio_plugin 0.2872, so defence is penalised 3.5x LESS than offence, and the defensive
# residual comes out 3.3x wider than the offensive one.  Sweep it against the consensus.
print("\n=== 7. is the offense/defense penalty ratio the lever?")
st = pd.read_parquet(OUT / "tune_players.parquet")
st = st[st.window == "2024-2026"][["player_id", "prior_off", "prior_def"]]
lg = pd.read_parquet(OUT / "tune_u.parquet")
lg = lg[lg.window == "2024-2026"]
d = lg.merge(st, on="player_id").merge(
    m[["player_id", "adj_offense", "adj_defense", "adj_overall"]], on="player_id")
rows = []
for (lam, ratio), g in d.groupby(["lam", "ratio"]):
    o, f = g.prior_off + g.u_off, g.prior_def + g.u_def
    rows.append(dict(lam=lam, ratio=ratio, tot=spearmanr(o + f, g.adj_overall).statistic,
                     dfn=spearmanr(f, g.adj_defense).statistic))
sweep = pd.DataFrame(rows)
print("   ratio = defensive penalty / offensive penalty; 1.0 means both sides shrunk equally")
for rr, g in sweep.groupby("ratio"):
    bb = g.loc[g.tot.idxmax()]
    star = "   <-- shipped" if abs(rr - float(cfg["lam_ratio_plugin"])) < 1e-6 else ""
    print(f"     ratio {rr:<7} best lam {bb.lam:8.0f}   total {bb.tot:.3f}   defense {bb.dfn:.3f}{star}")
print("\n   Monotone toward 1.0, so defence wants MORE shrinkage than it gets. But the whole")
print("   sweep only moves the total from 0.784 to 0.796, against 0.864 for the reweight above.")
print("   The ratio is the second problem, not the first.")

print("\n=== 8. why the ratio cannot be the answer")
for side, target, lab in SIDES:
    print(f"  {lab:8s} prior correlates {m[f'prior_{side}'].corr(m[target]):+.3f} with the consensus, "
          f"residual {m[f'u_{side}'].corr(m[target]):+.3f}   "
          f"(spreads {m[f'prior_{side}'].std():.2f} and {m[f'u_{side}'].std():.2f})")
print("\n   On defense the residual is the BETTER signal and the prior the worse one, yet the")
print("   prior carries twice the spread. The penalty ratio only scales the residual, so turning")
print("   it up throws away the better signal. The defensive box prior is what needs to shrink,")
print("   and no lambda in this model can do that: the prior enters the rating at weight 1.0")
print("   by construction.")
