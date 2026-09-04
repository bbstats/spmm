"""How much defensive box prior should the ratings carry?  Sweep it against held-out points.

FINDINGS.md section 13 recorded that no internal criterion this project had could select the
defensive shrinkage: held-out stint MSE within a season prefers the wrong end at 3.4 standard errors
(scripts/26_which_loss.py), and the next-window on-court benchmark has no interior optimum at all --
it is flat then declines, so it just runs to the weakest penalty.  Both fail for the same reason,
that they are built from the same margin data as the estimate and share its blind spot.

Out-of-season prediction is not.  It is scored on actual points in a season the ratings never saw,
so it can be SELECTED on without circularity, and the consensus stays untouched as the external
check.  That makes this the first honest way to answer the question the hybrid guessed at.

The sweep is one scalar `c_def` multiplying the defensive half of the team-priced beta, with the
player-priced offensive prior held fixed:

    c_def = 0    the shipped hybrid: no defensive box prior at all
    c_def = 1    the defensive prior at the weight prior-informed RAPM gives it

If the criterion has an interior optimum, it answers the question.  If it runs to an end of the
grid, it is as blind as the others and we have learned that instead.

usage: python scripts/41_defweight.py [first=1998] [last=2025] [--k=2] [--c=0,0.25,0.5,0.75,1]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.cv import plugin_fit  # noqa: E402
from eracoef.windows import build_window, hybrid_beta, player_priced_beta, window_label, window_seasons  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 4)
cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
FE = cfg["features"]
nf = len(FE)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
FIRST = int(args[0]) if args else 1998
LAST = int(args[1]) if len(args) > 1 else 2025


def _flag(n, d):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{n}=")]
    return [float(x) for x in hit[0].split("=")[1].split(",")] if hit else d


K = int(_flag("k", [2])[0])
CS = _flag("c", [0.0, 0.25, 0.5, 0.75, 1.0])
LAM = float(cfg["lam_plugin"])
RATIO = float(cfg["lam_ratio_plugin"])
S0, S1 = int(cfg["first_season"]), int(cfg["last_season"])

panel = pd.read_parquet(OUT / "xrapm_panel.parquet")
coefs = pd.read_parquet(OUT / "coefs.parquet")
base = coefs[coefs.run == "base"]
WIN_OF = {s: window_label(list(range(w[0], w[1] + 1)))
          for w in window_seasons(cfg) for s in range(w[0], w[1] + 1)}


def betas(labs, c):
    """Player-priced offense (as the hybrid), team-priced defense scaled by c."""
    d = base[~base.window.isin(labs)]
    dd = d[d.side == "D"].groupby("feature")["beta"].mean().reindex(FE).to_numpy()
    off = hybrid_beta(panel, FE, labs)[:nf]
    return np.concatenate([off, -c * dd])


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


rows = []
t0 = time.time()
for h in range(FIRST, LAST + 1):
    tr = neighbourhood(h, K)
    labs = {WIN_OF[s] for s in tr + [h]}
    wd_t, wd = design(tr), design([h])
    mt, m = wd_t.spec.n_ps, wd.spec.n_ps
    ids = wd.spec.ps_table["player_id"].to_numpy()
    y, w = wd.y, wd.w
    home = np.asarray(wd.X[:, wd.spec.f_col("home")].todense()).ravel()
    A = np.column_stack([np.ones(len(y)), home])
    for c in CS:
        beta = betas(labs, c)
        pipe = plugin_fit(wd_t, beta, lam=LAM, lam_ratio=RATIO, pad_target=cfg["pad_target"])
        exp, mm = pipe["exposure"], pipe["mm"]
        ro = exp.season_rates_ - exp.means_o_ / 5.0
        rd = exp.season_rates_d_ - exp.means_d_ / 5.0
        rat = pd.DataFrame({"player_id": wd_t.spec.ps_table["player_id"].to_numpy(),
                            "o": ro @ beta[:nf] + mm.u_[:mt], "d": rd @ beta[nf:] + mm.u_[mt:]})
        r = pd.DataFrame({"player_id": ids}).merge(rat, on="player_id", how="left").fillna(0.0)
        contrib = wd.X[:, :m] @ r["o"].to_numpy() + wd.X[:, m:2 * m] @ r["d"].to_numpy()
        cc = np.linalg.solve((A * w[:, None]).T @ A, (A * w[:, None]).T @ (y - contrib))
        pred = A @ cc + contrib
        poss = wd.rows["poss"].to_numpy()
        gg = pd.DataFrame({"g": wd.rows["game_idx"].to_numpy(), "hh": wd.rows["is_home_off"].to_numpy(),
                           "poss": poss, "act": y * poss / 100.0, "prd": pred * poss / 100.0}
                          ).groupby(["g", "hh"]).sum()
        rows.append(dict(held_out=h, c_def=c, mse=float(np.average((y - pred) ** 2, weights=w)),
                         n=float(w.sum()), sd_def=float(r["d"].std()),
                         tg=float(np.average(((gg.act - gg.prd) / gg.poss * 100) ** 2, weights=gg.poss)),
                         tg_n=float(gg.poss.sum())))
    print(f"  {h} ({time.time() - t0:.0f}s)", flush=True)

D = pd.DataFrame(rows)
D.to_parquet(OUT / "defweight.parquet", index=False)
g = D.groupby("c_def").apply(lambda d: pd.Series({
    "mse": float((d.mse * d.n).sum() / d.n.sum()),
    "team_game": float((d.tg * d.tg_n).sum() / d.tg_n.sum()),
    "sd_def": d.sd_def.mean()}), include_groups=False)
print("\n=== held-out MSE by defensive prior weight (0 = shipped hybrid, 1 = prior-informed RAPM)")
print(g.to_string())
best = g.mse.idxmin()
edge = best in (min(CS), max(CS))
print(f"\nargmin at c_def = {best}   {'AT A GRID EDGE -- this criterion cannot choose either' if edge else 'INTERIOR -- the criterion selects it'}")

p = D.pivot_table(index="held_out", columns="c_def", values="mse")
print("\n=== paired against c_def = 0 (the shipped hybrid); negative = better than shipping")
for c in CS[1:]:
    diff = (p[c] - p[CS[0]]).dropna()
    n = len(diff)
    z = diff.mean() / (diff.std(ddof=1) / np.sqrt(n))
    print(f"  c_def={c:<5} mean {diff.mean():+8.4f}  z {z:+6.2f}  better in {int((diff < 0).sum())}/{n}")
print(f"\nwrote outputs/defweight.parquet  ({time.time() - t0:.0f}s)")
