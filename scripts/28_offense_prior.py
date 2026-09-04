"""Do the offensive prior properly: train it to predict a player's NEXT-window on-court impact.

Two things this fixes about scripts/27.

1. The target.  There the box rates were regressed on the SAME window's RAPM, de-shrunk by 1/a.
   That amplifies noise badly and shares sampling error between features and target.  Here the
   target is the player's on-court RAPM in window w+1 from his rates in window w: leak-free, and it
   is how the xRAPM/BPM family actually trains a prior.

2. The regularization.  There it was effectively unpenalized least squares on 13 correlated rates
   against a noisy target, so the coefficients were high-variance and the comparison was unfair to
   the player-level approach.  Here the ridge penalty is chosen by cross-validation over windows.

It also tests the claim that scoring is worth less in some lineups than others - that usage is
partly zero-sum, so a shot taken is partly a shot a teammate did not take.  The project previously
rejected diminishing returns on the basis of held-out stint mean squared error.  Section 9 shows
that loss cannot see player-level attribution, so the rejection does not stand and the question is
re-opened here against a player-level target: usage, usage squared, and squared scoring rates.

usage: python scripts/28_offense_prior.py
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
FE = cfg["features"]
P = pd.read_parquet(OUT / "xrapm_panel.parquet")
WINDOWS = sorted(P.window.unique())
NXT = {WINDOWS[i]: WINDOWS[i + 1] for i in range(len(WINDOWS) - 1)}
MIN_POSS = 2000


def panel(side):
    """A player's rates in window w against his own on-court RAPM in window w+1."""
    a = P[(P.side == side) & (P.poss >= MIN_POSS)].copy()
    b = a[["window", "player_id", "u", "a", "poss"]].rename(
        columns={"window": "next", "u": "u_next", "a": "a_next", "poss": "poss_next"})
    a["next"] = a.window.map(NXT)
    d = a.merge(b, on=["next", "player_id"], how="inner")
    d["y"] = d.u_next / np.maximum(d.a_next, 1e-9)      # de-shrunk next-window impact
    d["w"] = d.a_next                                    # Fay-Herriot weight for that target
    return d


def design(d, mode):
    """Feature blocks. `usage` is shot attempts plus turnovers per 100, the zero-sum quantity."""
    X = [d[FE].to_numpy()]
    names = list(FE)
    if mode != "linear":
        usage = (d.fg2m + d.fg2_miss + d.fg3m + d.fg3_miss + 0.44 * (d.ftm + d.ft_miss) + d.tov).to_numpy()
        u = (usage - usage.mean()) / usage.std()
        X.append(np.column_stack([u, u ** 2]))
        names += ["usage", "usage^2"]
    if mode == "full":
        sc = ["fg3m", "fg2m", "ftm", "ast"]
        Z = d[sc].to_numpy()
        Z = (Z - Z.mean(0)) / Z.std(0)
        X.append(Z ** 2)
        names += [f"{c}^2" for c in sc]
    return np.column_stack(X), names


def ridge_cv(d, mode, lams=np.geomspace(1e-4, 1e3, 25)):
    """Leave-one-window-out ridge; returns (coefficients, chosen penalty, out-of-fold rank score)."""
    X, names = design(d, mode)
    mu, sd = X.mean(0), X.std(0)
    Xs = (X - mu) / sd
    y, w = d.y.to_numpy(), d.w.to_numpy()
    best, best_lam = -np.inf, None
    for lam in lams:
        oof = np.zeros(len(d))
        for win in d.window.unique():
            te = (d.window == win).to_numpy()
            tr = ~te
            A = np.column_stack([np.ones(tr.sum()), Xs[tr]])
            pen = lam * np.eye(A.shape[1])
            pen[0, 0] = 0.0
            b = np.linalg.solve((A * w[tr, None]).T @ A + pen, (A * w[tr, None]).T @ y[tr])
            oof[te] = b[0] + Xs[te] @ b[1:]
        s = spearmanr(oof, y).statistic
        if s > best:
            best, best_lam = s, lam
    A = np.column_stack([np.ones(len(d)), Xs])
    pen = best_lam * np.eye(A.shape[1])
    pen[0, 0] = 0.0
    b = np.linalg.solve((A * w[:, None]).T @ A + pen, (A * w[:, None]).T @ y)
    return b, best_lam, best, names, mu, sd


print(f"training a prior on what a box line says about the NEXT window's on-court impact")
res = {}
for side, lab in (("O", "offense"), ("D", "defense")):
    d = panel(side)
    print(f"\n=== {lab}: {len(d)} player-window pairs, {MIN_POSS}+ possessions in both")
    for mode in ("linear", "usage", "full"):
        b, lam, s, names, mu, sd = ridge_cv(d, mode)
        print(f"  {mode:7s} penalty {lam:8.4f}   out-of-fold rank agreement with next-window "
              f"impact  {s:.4f}")
        res[(side, mode)] = (b, names, mu, sd, s)
    b, names, mu, sd, _ = res[(side, "usage")]
    k = {n: v for n, v in zip(names, b[1:])}
    print(f"  the two zero-sum terms:  usage {k['usage']:+.3f}   usage squared {k['usage^2']:+.3f}")
    print("  (a negative squared term is diminishing returns: the next unit of usage is worth less)")

print("\n=== do the zero-sum terms earn their place?")
for side, lab in (("O", "offense"), ("D", "defense")):
    a, b, c = (res[(side, m)][4] for m in ("linear", "usage", "full"))
    print(f"  {lab:8s} linear {a:.4f}   + usage {b:.4f} ({b - a:+.4f})   "
          f"+ squared scoring {c:.4f} ({c - a:+.4f})")

# ---------------------------------------------------------------- score on the consensus board
con = pd.read_parquet(OUT / "vs_consensus.parquet")
LAB = "2024-2026"
print(f"\n=== applied to the {LAB} board, scored against the external consensus")
print("   the prior below is trained only on windows that end before this one")
rows = []
for mode in ("linear", "usage", "full"):
    parts = {}
    for side in ("O", "D"):
        d = panel(side)
        d_tr = d[d["next"] != LAB]
        b, lam, s, names, mu, sd = ridge_cv(d_tr, mode)
        cur = P[(P.window == LAB) & (P.side == side)].copy()
        X, _ = design(cur, mode)
        parts[side] = pd.DataFrame({"player_id": cur.player_id.to_numpy(),
                                    f"pri_{side}": b[0] + ((X - mu) / sd) @ b[1:]})
    g = parts["O"].merge(parts["D"], on="player_id").merge(con, on="player_id")
    rows.append(dict(prior=mode,
                     offense=spearmanr(g.pri_O, g.adj_offense).statistic,
                     defense=spearmanr(-g.pri_D, g.adj_defense).statistic))
print("   (the box prior ALONE, before any on-court residual is added)")
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(f"\n   for comparison, the shipped team-level box prior alone:")
print(f"     offense {spearmanr(con.prior_off, con.adj_offense).statistic:.3f}   "
      f"defense {spearmanr(con.prior_def, con.adj_defense).statistic:.3f}")
