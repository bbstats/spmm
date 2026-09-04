"""Does a rating travel to a new team?  The test that separates prediction from attribution.

Two criteria disagree about the hybrid prior.  The external consensus rates it far above
prior-informed RAPM (0.896 against 0.784); out-of-season prediction of actual stints rates it below
(scripts/38_yoy.py: the hybrid wins 5 of 28 seasons).  scripts/39_why.py ruled out the obvious
explanation -- the deficit is not concentrated in lightly-used players, it is worst among
well-established ones.

The remaining explanation is that predicting a lineup's points does not require splitting credit
correctly.  Teammates persist: only about half of a returning player's teammate-possessions are with
someone new, so a rating that systematically hands a big man his teammates' defensive credit can
still predict his team's lineups well, because those teammates are largely still there.

MOVERS break that.  When a player changes teams, his rating has to stand on its own alongside a
roster that shares none of the credit it was estimated from.  So:

    split held-out rows by whether any player on the floor changed team since the training block
    compare the methods within each group

If the hybrid's deficit shrinks or reverses on mover rows, its allocation is better and the
team-priced prior's edge is a lineup-sum artifact that does not transfer.  If the deficit is
unchanged or larger on mover rows, the team-priced prior genuinely attributes better too, and the
consensus agreement is the thing that needs explaining.

A player's team in a season is the one he played the most possessions for, so mid-season trades
count as a move only when the bulk of his minutes shifted.

usage: python scripts/40_movers.py [first=1998] [last=2025] [--k=2]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.boxtable import season_box  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.cv import plugin_fit  # noqa: E402
from eracoef.windows import build_window, hybrid_beta, window_label, window_seasons  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 4)
cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
FE = cfg["features"]
nf = len(FE)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
FIRST = int(args[0]) if args else 1998
LAST = int(args[1]) if len(args) > 1 else 2025
kf = [a for a in sys.argv[1:] if a.startswith("--k=")]
K = int(kf[0].split("=")[1]) if kf else 2
LAM = float(cfg["lam_plugin"])
RATIO = float(cfg["lam_ratio_plugin"])
S0, S1 = int(cfg["first_season"]), int(cfg["last_season"])

panel = pd.read_parquet(OUT / "xrapm_panel.parquet")
coefs = pd.read_parquet(OUT / "coefs.parquet")
base = coefs[coefs.run == "base"]
WIN_OF = {s: window_label(list(range(w[0], w[1] + 1)))
          for w in window_seasons(cfg) for s in range(w[0], w[1] + 1)}
_team: dict = {}


def main_team(season):
    """player_id -> the team he played the most minutes for that season."""
    if season not in _team:
        b = season_box([season], ["RS"], cfg).groupby(["player_id", "team_id"], as_index=False)["minutes"].sum()
        b = b.sort_values("minutes").drop_duplicates("player_id", keep="last")
        _team[season] = dict(zip(b.player_id.astype(int), b.team_id.astype(int)))
    return _team[season]


def team_beta(labs):
    d = base[~base.window.isin(labs)]
    o = d[d.side == "O"].groupby("feature")["beta"].mean().reindex(FE).to_numpy()
    dd = d[d.side == "D"].groupby("feature")["beta"].mean().reindex(FE).to_numpy()
    return np.concatenate([o, -dd])


_cache: dict = {}


def design(seasons):
    key = tuple(seasons)
    if key not in _cache:
        if len(_cache) > 3:
            _cache.clear()
        _cache[key] = build_window(list(seasons), cfg)
    return _cache[key]


def neighbourhood(h, k):
    out, d = [], 1
    while len(out) < k and d <= (S1 - S0):
        for s in (h - d, h + d):
            if S0 <= s <= S1 and len(out) < k:
                out.append(s)
        d += 1
    return sorted(out)


def ratings(seasons, beta):
    wd = design(seasons)
    pipe = plugin_fit(wd, beta, lam=LAM, lam_ratio=RATIO, pad_target=cfg["pad_target"])
    exp, mm = pipe["exposure"], pipe["mm"]
    m = wd.spec.n_ps
    ro = exp.season_rates_ - exp.means_o_ / 5.0
    rd = exp.season_rates_d_ - exp.means_d_ / 5.0
    return pd.DataFrame({"player_id": wd.spec.ps_table["player_id"].to_numpy(),
                         "o": ro @ beta[:nf] + mm.u_[:m], "d": rd @ beta[nf:] + mm.u_[m:],
                         "poss": exp.season_poss_off_})


rows = []
t0 = time.time()
for h in range(FIRST, LAST + 1):
    tr = neighbourhood(h, K)
    labs = {WIN_OF[s] for s in tr + [h]}
    wd = design([h])
    m = wd.spec.n_ps
    ids = wd.spec.ps_table["player_id"].to_numpy()
    y, w = wd.y, wd.w
    home = np.asarray(wd.X[:, wd.spec.f_col("home")].todense()).ravel()
    one = np.ones(len(y))

    # a player MOVED if his main team in the held-out season differs from his main team in the
    # training block; players absent from either are not movers, they are simply unknown
    now = main_team(h)
    before = {}
    for s in tr:                                   # the nearest training season he appears in wins
        for pid, t in main_team(s).items():
            before.setdefault(pid, t)
    moved = np.array([1.0 if (p in now and p in before and now[p] != before[p]) else 0.0
                      for p in ids])
    n_movers = wd.X[:, :2 * m] @ np.tile(moved, 2)   # how many of the ten on the floor moved
    grp = np.where(n_movers == 0, "no movers", np.where(n_movers <= 2, "1-2 movers", "3+ movers"))

    for name, beta in (("2 PI-RAPM, team-priced", team_beta(labs)),
                       ("3 hybrid", hybrid_beta(panel, FE, labs))):
        r = pd.DataFrame({"player_id": ids}).merge(ratings(tr, beta), on="player_id", how="left").fillna(0.0)
        contrib = wd.X[:, :m] @ r["o"].to_numpy() + wd.X[:, m:2 * m] @ r["d"].to_numpy()
        A = np.column_stack([one, home])
        c = np.linalg.solve((A * w[:, None]).T @ A, (A * w[:, None]).T @ (y - contrib))
        err2 = (y - (A @ c + contrib)) ** 2
        for g in ("no movers", "1-2 movers", "3+ movers"):
            msk = grp == g
            if msk.sum():
                rows.append(dict(held_out=h, method=name, grp=g,
                                 sse=float((w[msk] * err2[msk]).sum()), n=float(w[msk].sum())))
    print(f"  {h} ({time.time() - t0:.0f}s)", flush=True)

D = pd.DataFrame(rows)
ORDER = ["no movers", "1-2 movers", "3+ movers"]
piv = D.groupby(["grp", "method"]).apply(lambda d: d.sse.sum() / d.n.sum(), include_groups=False).unstack()
piv = piv.reindex(ORDER)
piv["hybrid - PI"] = piv["3 hybrid"] - piv["2 PI-RAPM, team-priced"]
piv["share of possessions"] = (D.groupby("grp").n.sum() / D.groupby("grp").n.sum().sum()).reindex(ORDER)
print(f"\n=== held-out MSE by how many of the ten on the floor changed team (K={K})")
print("    positive `hybrid - PI` means the hybrid is worse there")
print("    if the hybrid attributes credit better, its deficit should SHRINK as movers increase")
print(piv.to_string())

print("\n=== paired by held-out season")
for g in ORDER:
    d = D[D.grp == g]
    p = d.pivot_table(index="held_out", columns="method", values="sse") \
        .div(d.pivot_table(index="held_out", columns="method", values="n"))
    diff = (p["3 hybrid"] - p["2 PI-RAPM, team-priced"]).dropna()
    n = len(diff)
    if n < 3:
        continue
    z = diff.mean() / (diff.std(ddof=1) / np.sqrt(n))
    print(f"  {g:12s}  mean {diff.mean():+8.4f}  z {z:+6.2f}  hybrid better in {int((diff < 0).sum())}/{n}")

D.to_parquet(OUT / "movers.parquet", index=False)
print(f"\nwrote outputs/movers.parquet  ({time.time() - t0:.0f}s)")
