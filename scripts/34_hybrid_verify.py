"""Does a SINGLE fit with a half-zeroed beta reproduce the spliced hybrid?

The hybrid is "player-priced box prior on offense, no box prior on defense".  Two ways to build it:

  spliced   two plugin fits -- one with the player-priced beta (keep its offense column), one with
            beta = 0 (keep its defense column).  This is what the ladder did, and it only makes
            sense if the two fits use different penalties.
  single    ONE plugin fit with beta = [beta_player_offense, 0].  The box term enters the model as
            an offset Xbox @ beta, and the offensive and defensive box columns are separate, so
            zeroing the defensive half IS "no defensive prior" -- in one fit, at one penalty.

The single form is what we want to ship.  This checks it against the spliced form on 2024-26 at the
shipped penalty, and reports both against the consensus.

usage: python scripts/33c_verify.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.cv import plugin_fit  # noqa: E402
from eracoef.windows import build_window  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 4)
cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
FE = cfg["features"]
nf = len(FE)
LAB = "2024-2026"
SEASONS = [2024, 2025, 2026]
LAM = float(cfg["lam_plugin"])
RATIO = float(cfg["lam_ratio_plugin"])
MIN_POSS = 2000

# ---------------------------------------------------------------- the player-priced beta
P = pd.read_parquet(OUT / "xrapm_panel.parquet")
WINDOWS = sorted(P.window.unique())
NXT = {WINDOWS[i]: WINDOWS[i + 1] for i in range(len(WINDOWS) - 1)}


def player_beta(exclude=()):
    """Regress a player's own rates on his own next-window on-court impact.  No consensus."""
    out = []
    for side in ("O", "D"):
        a = P[(P.side == side) & (P.poss >= MIN_POSS)].copy()
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


B_PLAYER = player_beta(exclude=(LAB,))
B_HYBRID = np.concatenate([B_PLAYER[:nf], np.zeros(nf)])
B_ZERO = np.zeros(2 * nf)
print(f"player-priced beta, offense half: {np.round(B_PLAYER[:nf], 3)}")
print(f"                    defense half: {np.round(B_PLAYER[nf:], 3)}  (zeroed in the hybrid)")

t0 = time.time()
wd = build_window(SEASONS, cfg)
m = wd.spec.n_ps
ids = wd.spec.ps_table["player_id"].to_numpy()
print(f"window built ({time.time() - t0:.0f}s), {m} players")


def rate(beta, label):
    pipe = plugin_fit(wd, beta, lam=LAM, lam_ratio=RATIO, pad_target=cfg["pad_target"])
    exp, mm = pipe["exposure"], pipe["mm"]
    ro = exp.season_rates_ - exp.means_o_ / 5.0
    rd = exp.season_rates_d_ - exp.means_d_ / 5.0
    df = pd.DataFrame(dict(player_id=ids, poss=exp.season_poss_off_))
    df["off"] = ro @ beta[:nf] + mm.u_[:m]
    df["def"] = -(rd @ beta[nf:]) - mm.u_[m:]
    df["total"] = df["off"] + df["def"]
    print(f"  {label}: fit at lam={LAM} ratio={RATIO} ({time.time() - t0:.0f}s)")
    return df


single = rate(B_HYBRID, "single fit, beta = [player offense, 0]")
zero = rate(B_ZERO, "zero-prior fit (its defense is the spliced hybrid's defense)")
player = rate(B_PLAYER, "player-priced both sides (its offense is the spliced hybrid's offense)")

spliced = player[["player_id", "poss", "off"]].merge(zero[["player_id", "def"]], on="player_id")
spliced["total"] = spliced["off"] + spliced["def"]

print("\n=== single fit vs spliced, over all players")
for c in ("off", "def", "total"):
    a, b = single.set_index("player_id")[c], spliced.set_index("player_id")[c]
    b = b.reindex(a.index)
    print(f"  {c:6s}  max|diff| {np.abs(a - b).max():.6f}   spearman {spearmanr(a, b).statistic:.6f}")

# ---------------------------------------------------------------- validation, read once
con = pd.read_parquet(OUT / "vs_consensus.parquet")[
    ["player_id", "player_name", "adj_offense", "adj_defense", "adj_overall", "bigness"]]


def z(s):
    return (s - s.mean()) / s.std()


def score(df, label):
    g = df.merge(con, on="player_id")
    g = g[g.poss >= 1000]
    return dict(candidate=label, n=len(g),
                total=spearmanr(g.total, g.adj_overall).statistic,
                offense=spearmanr(g["off"], g.adj_offense).statistic,
                defense=spearmanr(g["def"], g.adj_defense).statistic,
                spread_off=g["off"].std() / g.adj_offense.std(),
                spread_def=g["def"].std() / g.adj_defense.std(),
                bias=(z(g.total) - z(g.adj_overall)).corr(g.bigness))


R = pd.DataFrame([score(single, "hybrid, single fit"), score(spliced, "hybrid, spliced"),
                  score(zero, "RAPM, no prior"), score(player, "player-priced both sides")])
print(f"\n=== vs the consensus, {LAB}, at the SHIPPED penalty (lam_plugin, lam_ratio_plugin)")
print(R.to_string(index=False))

ship = pd.read_parquet(OUT / "player_ratings.parquet")
ship = ship[ship.window == LAB][["player_id", "rating_off", "rating_def", "rating_total"]]
ship = ship.rename(columns={"rating_off": "off", "rating_def": "def", "rating_total": "total"})
ship = ship.merge(single[["player_id", "poss"]], on="player_id")
print("\n=== for reference, what ships today")
print(pd.DataFrame([score(ship, "ships today")]).to_string(index=False))

single.to_parquet(OUT / "hybrid_single.parquet", index=False)
print("\nwrote outputs/hybrid_single.parquet")
