"""A flat shrink on the defensive prior is too blunt. Is the per-feature correction the right one?

Scaling the whole defensive prior by one number treats every box column as equally untrustworthy.
It is not.  Steals survive into the lineup sum at 0.90, meaning a player's steal rate is genuinely
his own; defensive rebounds survive at 0.59, meaning the team collects about the same number either
way and the box score is mostly recording who picked the ball up.  A flat scalar suppresses both.

The conservation ratio is already measured per feature per side (outputs/investigate_conservation
.parquet, built by scripts/15).  If the lineup-to-player amplitude problem really is conservation,
then two things should hold:

  1. the ratio between the two-stage (player-priced) beta and the joint (team-priced) beta should
     track the conservation ratio across the 13 features, and
  2. correcting the joint beta per feature by conservation should recover what the two-stage fit
     found, without ever fitting a prior at all.

Candidates, all built from the JOINT beta, all scored against the consensus as validation only:

  A  joint beta, untouched
  B  joint beta, defence times a flat scalar          (pure scale, the blunt version)
  C  joint beta, every coefficient times its own conservation ratio   (scale and shape)
  D  same as C but renormalised to keep the overall defensive spread of B (pure shape)

If D beats B, the per-feature shape matters and the flat shrink really was throwing information
away.  If D and B tie, the correction is genuinely one-dimensional and the flat scalar is fine.

usage: python scripts/31_per_feature.py
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
LABEL = {"fg3m": "3PM", "fg3_miss": "3P miss", "fg2m": "2PM", "fg2_miss": "2P miss", "ftm": "FTM",
         "ft_miss": "FT miss", "orb": "ORB", "drb": "DRB", "ast": "AST", "tov": "TOV",
         "stl": "STL", "blk": "BLK", "pf": "PF"}

# conservation: how much of a player's own rate difference survives into the lineup sum
cons = pd.read_parquet(OUT / "investigate_conservation.parquet")
S = cons.groupby(["feature", "side"])["survives"].mean().unstack().reindex(FE)
s_o, s_d = S["O"].to_numpy(), S["D"].to_numpy()

# the two betas
P = pd.read_parquet(OUT / "xrapm_panel.parquet")
W = sorted(P.window.unique())
NXT = {W[i]: W[i + 1] for i in range(len(W) - 1)}


def player_beta(exclude=()):
    out = []
    for side in ("O", "D"):
        a = P[(P.side == side) & (P.poss >= 2000)].copy()
        b = a[["window", "player_id", "u", "a"]].rename(
            columns={"window": "next", "u": "u_next", "a": "a_next"})
        a["next"] = a.window.map(NXT)
        d = a.merge(b, on=["next", "player_id"], how="inner")
        d = d[~d["next"].isin(exclude) & ~d.window.isin(exclude)]
        X = np.column_stack([np.ones(len(d)), d[FE].to_numpy()])
        y = (d.u_next / np.maximum(d.a_next, 1e-9)).to_numpy()
        w = d.a_next.to_numpy()
        pen = 5.0 * np.eye(X.shape[1])
        pen[0, 0] = 0.0
        out.append(np.linalg.solve((X * w[:, None]).T @ X + pen, (X * w[:, None]).T @ y)[1:])
    return np.concatenate(out)


B_PLAY = player_beta(exclude=("2024-2026",))
wd = build_window([2024, 2025, 2026], cfg)
m = wd.spec.n_ps
ids = wd.spec.ps_table["player_id"].to_numpy()
cf = crossfit_beta(wd, lams=[LAM_B], cv=2, lam_ratio=RAT_B, pad_target=cfg["pad_target"])
B_JOINT = cf.beta_

# --------------------------------------------------------- 1. does conservation explain the ratio?
ratio_o = B_PLAY[:nf] / np.where(np.abs(B_JOINT[:nf]) > 0.05, B_JOINT[:nf], np.nan)
ratio_d = B_PLAY[nf:] / np.where(np.abs(B_JOINT[nf:]) > 0.05, B_JOINT[nf:], np.nan)
T = pd.DataFrame({"stat": [LABEL[f] for f in FE], "survives_O": s_o, "ratio_O": ratio_o,
                  "survives_D": s_d, "ratio_D": ratio_d})
print("=== 1. two-stage beta divided by joint beta, against the conservation ratio")
print("   (coefficients smaller than 0.05 in size are blanked; the ratio is meaningless there)")
print(T.round(3).to_string(index=False))
for side, sv, rt in (("offense", s_o, ratio_o), ("defense", s_d, ratio_d)):
    ok = np.isfinite(rt)
    r = spearmanr(sv[ok], rt[ok]).statistic
    print(f"   {side}: rank correlation between conservation and the ratio = {r:+.3f} "
          f"(n = {ok.sum()} usable features)")

# --------------------------------------------------------- 2. score the candidates
con = pd.read_parquet(OUT / "vs_consensus.parquet")[
    ["player_id", "adj_offense", "adj_defense", "adj_overall", "bigness"]]


def score(beta, label):
    f = plugin_fit(wd, beta, lam=LAM, lam_ratio=RATIO, pad_target=cfg["pad_target"])
    mm, exp = f["mm"], f["exposure"]
    ro = exp.season_rates_ - exp.means_o_ / 5.0
    rd = exp.season_rates_d_ - exp.means_d_ / 5.0
    pri_d = -(rd @ beta[nf:])
    g = pd.DataFrame({"player_id": ids, "poss": exp.season_poss_off_,
                      "off": ro @ beta[:nf] + mm.u_[:m], "def": pri_d - mm.u_[m:],
                      "sd_prior_def": pri_d.std()}).merge(con, on="player_id")
    g = g[g.poss >= 1000]
    g["tot"] = g["off"] + g["def"]
    z = lambda s: (s - s.mean()) / s.std()  # noqa: E731
    return dict(candidate=label, total=spearmanr(g.tot, g.adj_overall).statistic,
                offense=spearmanr(g["off"], g.adj_offense).statistic,
                defense=spearmanr(g["def"], g.adj_defense).statistic,
                sd_prior_def=g.sd_prior_def.iloc[0],
                bias=(z(g.tot) - z(g.adj_overall)).corr(g.bigness))


# the flat scalar that matches the two-stage prior's defensive spread, measured earlier
FLAT = 0.55
rows = [score(B_JOINT, "A. joint beta, untouched")]
b = B_JOINT.copy()
b[nf:] *= FLAT
rows.append(score(b, f"B. joint beta, defence times a flat {FLAT}"))
b = B_JOINT.copy()
b[nf:] *= s_d
rows.append(score(b, "C. joint beta, defence times its own conservation ratio"))
# renormalise C so its defensive prior has the same spread as B: pure shape, no scale advantage
b_c = B_JOINT.copy()
b_c[nf:] *= s_d
sd_b = rows[1]["sd_prior_def"]
sd_c = rows[2]["sd_prior_def"]
b_d = B_JOINT.copy()
b_d[nf:] *= s_d * (sd_b / sd_c)
rows.append(score(b_d, "D. same as C, renormalised to B's spread (pure shape)"))
rows.append(score(np.concatenate([B_PLAY[:nf], B_PLAY[nf:]]), "E. two-stage (player-priced) beta"))
R = pd.DataFrame(rows)
print("\n=== 2. validation against the consensus, penalty and ratio held fixed for all")
print(R[["candidate", "total", "offense", "defense", "sd_prior_def", "bias"]].to_string(index=False))
print("\n   B against D is the question: same defensive spread, different shape.")
d_ = R.set_index("candidate")
print(f"   flat {FLAT} scalar        total {d_.total.iloc[1]:.3f}")
print(f"   per-feature, same spread total {d_.total.iloc[3]:.3f}   "
      f"difference {d_.total.iloc[3] - d_.total.iloc[1]:+.3f}")
R.to_csv(OUT / "csv" / "per_feature.csv", index=False)
print("\nwrote outputs/csv/per_feature.csv")
