"""Why does the team-priced prior out-predict the hybrid out of season?

scripts/38_yoy.py found the hybrid loses to prior-informed RAPM on held-out stint prediction by
about 1.2 points of MSE, winning only 5 of 28 seasons -- while the external consensus rates the
hybrid far higher (0.896 against 0.784).  Both cannot be dismissed, so this asks WHERE the hybrid
loses rather than whether it does.

The hypothesis: the hybrid drops the defensive box prior entirely, so a player with few training
possessions gets a defensive rating shrunk to nearly zero -- the model has nothing to say about him.
Prior-informed RAPM always has something to say, because the box score scores everyone.  If that is
the mechanism, the hybrid's deficit should be concentrated in lineups full of lightly-used players
and should vanish among established ones.

If instead the deficit is flat across exposure, the hybrid is simply worse at allocating credit and
the consensus agreement is the thing that needs explaining.

Rows are binned by the SMALLEST training exposure among the ten players on the floor, because one
unknown player is enough to spoil a lineup's prediction.

usage: python scripts/39_why.py [first=1998] [last=2025] [--k=2]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
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
EDGES = [0, 1, 500, 1500, 4000, 10 ** 9]        # smallest training possessions on the floor
LABELS = ["none (0)", "1-499", "500-1499", "1500-3999", "4000+"]

panel = pd.read_parquet(OUT / "xrapm_panel.parquet")
coefs = pd.read_parquet(OUT / "coefs.parquet")
base = coefs[coefs.run == "base"]
WIN_OF = {s: window_label(list(range(w[0], w[1] + 1)))
          for w in window_seasons(cfg) for s in range(w[0], w[1] + 1)}


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
    Z = wd.X[:, :2 * m]

    # the smallest training exposure among the ten players on a row.  Z has exactly five ones in
    # each block, so a min over the on-court players is a max over (large constant - poss).
    BIG = 1e9
    for name, beta in (("2 PI-RAPM, team-priced", team_beta(labs)),
                       ("3 hybrid", hybrid_beta(panel, FE, labs))):
        rat = ratings(tr, beta)
        r = pd.DataFrame({"player_id": ids}).merge(rat, on="player_id", how="left").fillna(0.0)
        if name.startswith("2"):
            p = r["poss"].to_numpy()
            minposs = BIG - (Z.multiply(np.tile(BIG - p, 2))).max(axis=1).toarray().ravel()
            bin_i = np.digitize(minposs, EDGES) - 1
        contrib = wd.X[:, :m] @ r["o"].to_numpy() + wd.X[:, m:2 * m] @ r["d"].to_numpy()
        A = np.column_stack([one, home])
        c = np.linalg.solve((A * w[:, None]).T @ A, (A * w[:, None]).T @ (y - contrib))
        err2 = (y - (A @ c + contrib)) ** 2
        for b in range(len(LABELS)):
            msk = bin_i == b
            if msk.sum():
                rows.append(dict(held_out=h, method=name, bin=LABELS[b],
                                 sse=float((w[msk] * err2[msk]).sum()), n=float(w[msk].sum())))
    print(f"  {h} ({time.time() - t0:.0f}s)", flush=True)

D = pd.DataFrame(rows)
piv = D.groupby(["bin", "method"]).apply(lambda d: (d.sse.sum() / d.n.sum()), include_groups=False).unstack()
share = D.groupby("bin").n.sum() / D.groupby("bin").n.sum().sum()   # n counts both methods, so this is already the possession share
piv = piv.reindex(LABELS)
piv["hybrid - PI"] = piv["3 hybrid"] - piv["2 PI-RAPM, team-priced"]
piv["share of possessions"] = share.reindex(LABELS)
print(f"\n=== held-out MSE by the SMALLEST training exposure among the ten on the floor (K={K})")
print("    positive `hybrid - PI` means the hybrid is worse in that bin")
print(piv.to_string())

print("\n=== the same, paired by held-out season")
for b in LABELS:
    d = D[D.bin == b].pivot_table(index="held_out", columns="method", values="sse") \
        .div(D[D.bin == b].pivot_table(index="held_out", columns="method", values="n"))
    if len(d) < 3 or d.shape[1] < 2:
        continue
    diff = (d["3 hybrid"] - d["2 PI-RAPM, team-priced"]).dropna()
    n = len(diff)
    z = diff.mean() / (diff.std(ddof=1) / np.sqrt(n))
    print(f"  {b:12s}  mean {diff.mean():+8.4f}  z {z:+6.2f}  hybrid better in {int((diff < 0).sum())}/{n}")

D.to_parquet(OUT / "why.parquet", index=False)
print(f"\nwrote outputs/why.parquet  ({time.time() - t0:.0f}s)")
