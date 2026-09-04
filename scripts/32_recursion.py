"""Is the joint fit the fixed point of a recursive prior-informed RAPM? Test it directly.

The original argument for the joint fit was that it lands where you would land if you iterated
    prior -> RAPM shrunk toward it -> refit the prior on that RAPM -> repeat
and that it gets there in closed form instead of by looping.  This checks whether that is true.

The theory says it should NOT be, and says exactly why.  Because the window-level rates make
X_box = Z R, the joint fit's first-order condition for beta is

    R' Z' W (y - Z(R beta + u)) = 0        residual orthogonal to the box directions in STINT space,
                                           weighted by possessions and by who plays with whom

while the recursion refits beta by regressing the player vector theta on R, so its fixed point is

    R' P (theta - R beta) = 0              residual orthogonal to the box directions in PLAYER space

Those are different orthogonality conditions.  They agree only if Z'WZ is proportional to the player
weight matrix, i.e. only if co-occurrence carries no structure - which is exactly what lineups are
made of.  So the joint answer should differ from the recursive answer, and the gap should be biggest
where co-occurrence matters most.

usage: python scripts/32_recursion.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.cv import crossfit_beta, plugin_fit  # noqa: E402
from eracoef.windows import build_window  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
FE = cfg["features"]
nf = len(FE)
LAM_B, RAT_B = float(cfg["lam_beta"]), float(cfg["lam_ratio_beta"])
LAM, RATIO = 12000.0, 0.2872
N_ITER = 12
LABEL = {"fg3m": "3PM", "fg3_miss": "3P miss", "fg2m": "2PM", "fg2_miss": "2P miss", "ftm": "FTM",
         "ft_miss": "FT miss", "orb": "ORB", "drb": "DRB", "ast": "AST", "tov": "TOV",
         "stl": "STL", "blk": "BLK", "pf": "PF"}

wd = build_window([2024, 2025, 2026], cfg)
m = wd.spec.n_ps
ids = wd.spec.ps_table["player_id"].to_numpy()
cf = crossfit_beta(wd, lams=[LAM_B], cv=2, lam_ratio=RAT_B, pad_target=cfg["pad_target"])
B_JOINT = cf.beta_
con = pd.read_parquet(OUT / "vs_consensus.parquet")[
    ["player_id", "adj_offense", "adj_defense", "adj_overall", "bigness"]]


def step(beta):
    """One turn of the loop: fit RAPM shrunk toward this prior, then refit the prior on the result."""
    f = plugin_fit(wd, beta, lam=LAM, lam_ratio=RATIO, pad_target=cfg["pad_target"])
    mm, exp = f["mm"], f["exposure"]
    Ro = exp.season_rates_ - exp.means_o_ / 5.0
    Rd = exp.season_rates_d_ - exp.means_d_ / 5.0
    po, pd_ = exp.season_poss_off_, exp.season_poss_def_
    # the player vector implied by this iterate, in the model's raw sign on both sides
    th_o = Ro @ beta[:nf] + mm.u_[:m]
    th_d = Rd @ beta[nf:] + mm.u_[m:]
    out = []
    for R, th, w in ((Ro, th_o, po), (Rd, th_d, pd_)):
        A = np.column_stack([np.ones(len(th)), R])
        b = np.linalg.solve((A * w[:, None]).T @ A + 1e-6 * np.eye(A.shape[1]),
                            (A * w[:, None]).T @ th)
        out.append(b[1:])
    return np.concatenate(out), f, (Ro, Rd, po)


def score(beta, f, geom, label):
    mm, exp = f["mm"], f["exposure"]
    Ro, Rd, po = geom
    g = pd.DataFrame({"player_id": ids, "poss": po,
                      "off": Ro @ beta[:nf] + mm.u_[:m],
                      "def": -(Rd @ beta[nf:]) - mm.u_[m:]}).merge(con, on="player_id")
    g = g[g.poss >= 1000]
    g["tot"] = g["off"] + g["def"]
    z = lambda s: (s - s.mean()) / s.std()  # noqa: E731
    return dict(iterate=label,
                total=spearmanr(g.tot, g.adj_overall).statistic,
                offense=spearmanr(g["off"], g.adj_offense).statistic,
                defense=spearmanr(g["def"], g.adj_defense).statistic,
                sd_prior_def=float((-(Rd @ beta[nf:])).std()),
                bias=(z(g.tot) - z(g.adj_overall)).corr(g.bigness))


print("=== running the recursion from the joint solution")
print("   if the joint fit were the fixed point, iteration 1 would not move")
beta = B_JOINT.copy()
rows, moves = [], []
for k in range(N_ITER):
    new, f, geom = step(beta)
    rows.append(score(beta, f, geom, f"iterate {k}" if k else "iterate 0 = the joint fit"))
    mv = float(np.linalg.norm(new - beta) / max(np.linalg.norm(beta), 1e-12))
    moves.append(mv)
    print(f"   step {k:2d}: relative move {mv:.4f}   "
          f"defensive prior spread {rows[-1]['sd_prior_def']:.3f}   "
          f"total {rows[-1]['total']:.3f}")
    beta = new
_, f, geom = step(beta)
rows.append(score(beta, f, geom, "converged"))
R = pd.DataFrame(rows)

print("\n=== the joint fit is not the fixed point")
print(f"   the very first step moves beta by {moves[0] * 100:.1f}% of its own size")
print(f"   after {N_ITER} steps the move is down to {moves[-1] * 100:.3f}%, so the loop does converge")

print("\n=== where the recursion lands, scored against the consensus (validation only)")
print(R.iloc[[0, 1, 2, len(R) - 1]].to_string(index=False))

print("\n=== the two answers, coefficient by coefficient")
T = pd.DataFrame({"stat": [LABEL[x] for x in FE],
                  "joint_O": B_JOINT[:nf], "recursive_O": beta[:nf],
                  "joint_D": -B_JOINT[nf:], "recursive_D": -beta[nf:]})
T["ratio_D"] = beta[nf:] / np.where(np.abs(B_JOINT[nf:]) > 0.05, B_JOINT[nf:], np.nan)
print(T.round(3).to_string(index=False))
print(f"\n   defensive prior spread: joint {R.sd_prior_def.iloc[0]:.3f} -> "
      f"recursive {R.sd_prior_def.iloc[-1]:.3f}")
R.to_csv(OUT / "csv" / "recursion.csv", index=False)
np.save(OUT / "recursive_beta.npy", beta)
print("\nwrote outputs/csv/recursion.csv and outputs/recursive_beta.npy")
