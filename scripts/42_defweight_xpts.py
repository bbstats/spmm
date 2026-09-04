"""The untested combination: the defensive-prior weight sweep, run on the luck-adjusted target.

scripts/41_defweight.py swept `c_def` (the weight on the defensive half of the team-priced beta,
player-priced offense fixed) and found an interior optimum -- but on the raw points target, and only
reported stint-level error in its table.  scripts/38_yoy.py ranked the four systems, and selected
hybrid + xPTS(ft) on team-game error -- but only at c_def = 0.  So the chosen system was never
measured at the c_def the criterion actually wants.  This script crosses them.

For every held-out season H, every K, every target and every c_def: fit ratings on the symmetric
neighbourhood of H, predict H's stints, score against ACTUAL points at stint and team-game level.
Then, once, fit the same betas on the 2024-26 window and read the consensus.  The consensus is
validation only: nothing here is selected on it, and it is printed beside the criterion so the
trade can be seen in one table rather than argued about.

usage: python scripts/42_defweight_xpts.py [first=1998] [last=2025] [--k=2,4]
                                            [--c=0,0.25,0.5,0.75,1] [--targets=pts,xpts_ft]
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
from eracoef.windows import build_window, hybrid_beta, window_label, window_seasons  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 4)
cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
FE = cfg["features"]
nf = len(FE)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
FIRST = int(args[0]) if args else 1998
LAST = int(args[1]) if len(args) > 1 else 2025


def _flag(n, d, conv=float):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{n}=")]
    return [conv(x) for x in hit[0].split("=")[1].split(",")] if hit else d


KS = _flag("k", [2, 4], int)
CS = _flag("c", [0.0, 0.25, 0.5, 0.75, 1.0])
TARGETS = _flag("targets", ["pts", "xpts_ft"], str)
LAM = float(cfg["lam_plugin"])
RATIO = float(cfg["lam_ratio_plugin"])
S0, S1 = int(cfg["first_season"]), int(cfg["last_season"])
CON_LAB, MIN_POSS = "2024-2026", 1000

panel = pd.read_parquet(OUT / "xrapm_panel.parquet")
coefs = pd.read_parquet(OUT / "coefs.parquet")
base = coefs[coefs.run == "base"]
WIN_OF = {s: window_label(list(range(w[0], w[1] + 1)))
          for w in window_seasons(cfg) for s in range(w[0], w[1] + 1)}


def betas(labs, c):
    """Player-priced offense (as the hybrid), team-priced defense scaled by c, leave-labs-out."""
    d = base[~base.window.isin(labs)]
    dd = d[d.side == "D"].groupby("feature")["beta"].mean().reindex(FE).to_numpy()
    off = hybrid_beta(panel, FE, labs)[:nf]
    return np.concatenate([off, -c * dd])


_cache: dict = {}


def design(seasons, target):
    key = (tuple(seasons), target)
    if key not in _cache:
        if len(_cache) > 4:
            _cache.clear()
        _cache[key] = build_window(list(seasons), cfg, target=target)
    return _cache[key]


def neighbourhood(h, k):
    out, d = [], 1
    while len(out) < k and d <= (S1 - S0):
        for s in (h - d, h + d):
            if S0 <= s <= S1 and len(out) < k:
                out.append(s)
        d += 1
    return sorted(out)


def ratings(wd_t, beta):
    """Per-player offense and defense from a fitted training block, raw sign."""
    pipe = plugin_fit(wd_t, beta, lam=LAM, lam_ratio=RATIO, pad_target=cfg["pad_target"])
    exp, mm = pipe["exposure"], pipe["mm"]
    mt = wd_t.spec.n_ps
    ro = exp.season_rates_ - exp.means_o_ / 5.0
    rd = exp.season_rates_d_ - exp.means_d_ / 5.0
    return pd.DataFrame({"player_id": wd_t.spec.ps_table["player_id"].to_numpy(),
                         "poss": exp.season_poss_off_,
                         "o": ro @ beta[:nf] + mm.u_[:mt], "d": rd @ beta[nf:] + mm.u_[mt:]})


# ------------------------------------------------------------------ out-of-season criterion
rows = []
t0 = time.time()
held = [h for h in range(FIRST, LAST + 1) if S0 <= h <= S1]
print(f"held-out {held[0]}-{held[-1]} ({len(held)}), K {KS}, targets {TARGETS}, c_def {CS}\n")
for h in held:
    wd = design([h], "pts")                       # ALWAYS scored against actual points
    m = wd.spec.n_ps
    ids = wd.spec.ps_table["player_id"].to_numpy()
    y, w = wd.y, wd.w
    home = np.asarray(wd.X[:, wd.spec.f_col("home")].todense()).ravel()
    A = np.column_stack([np.ones(len(y)), home])
    AtW = (A * w[:, None]).T
    poss = wd.rows["poss"].to_numpy()
    gkey = [wd.rows["game_idx"].to_numpy(), wd.rows["is_home_off"].to_numpy()]
    for k in KS:
        tr = neighbourhood(h, k)
        labs = {WIN_OF[s] for s in tr + [h]}
        for target in TARGETS:
            wd_t = design(tr, target)
            for c in CS:
                beta = betas(labs, c)
                rat = ratings(wd_t, beta)
                r = pd.DataFrame({"player_id": ids}).merge(rat, on="player_id", how="left").fillna(0.0)
                contrib = wd.X[:, :m] @ r["o"].to_numpy() + wd.X[:, m:2 * m] @ r["d"].to_numpy()
                cc = np.linalg.solve(AtW @ A, AtW @ (y - contrib))
                pred = A @ cc + contrib
                gg = pd.DataFrame({"g": gkey[0], "hh": gkey[1], "poss": poss,
                                   "act": y * poss / 100.0, "prd": pred * poss / 100.0}
                                  ).groupby(["g", "hh"]).sum()
                rows.append(dict(held_out=h, k=k, target=target, c_def=c,
                                 mse=float(np.average((y - pred) ** 2, weights=w)), n=float(w.sum()),
                                 tg=float(np.average(((gg.act - gg.prd) / gg.poss * 100) ** 2, weights=gg.poss)),
                                 tg_n=float(gg.poss.sum()), sd_def=float(r["d"].std())))
    print(f"  {h} ({time.time() - t0:.0f}s)", flush=True)

D = pd.DataFrame(rows)
D.to_parquet(OUT / "defweight_xpts.parquet", index=False)

# ------------------------------------------------------------------ the consensus, read once
con = pd.read_parquet(OUT / "vs_consensus.parquet")[
    ["player_id", "adj_offense", "adj_defense", "adj_overall", "bigness"]]
_cache.clear()
seasons = [int(s) for s in CON_LAB.split("-")]
seasons = list(range(seasons[0], seasons[1] + 1))


def z(s):
    return (s - s.mean()) / s.std()


crow = []
for target in TARGETS:
    wd_c = design(seasons, target)
    for c in CS:
        beta = betas({CON_LAB}, c)
        rat = ratings(wd_c, beta)
        rat["d"] = -rat["d"]                                   # positive = good, as the board
        rat["t"] = rat.o + rat.d
        g = rat.merge(con, on="player_id")
        g = g[g.poss >= MIN_POSS]
        crow.append(dict(target=target, c_def=c, n=len(g),
                         con_total=spearmanr(g.t, g.adj_overall).statistic,
                         con_off=spearmanr(g.o, g.adj_offense).statistic,
                         con_def=spearmanr(g.d, g.adj_defense).statistic,
                         spread_def=g.d.std() / g.adj_defense.std(),
                         bias_total=(z(g.t) - z(g.adj_overall)).corr(g.bigness)))
C = pd.DataFrame(crow)
C.to_parquet(OUT / "defweight_xpts_consensus.parquet", index=False)

# ------------------------------------------------------------------ report
summ = D.groupby(["k", "target", "c_def"]).apply(lambda d: pd.Series({
    "stint": float((d.mse * d.n).sum() / d.n.sum()),
    "game": float((d.tg * d.tg_n).sum() / d.tg_n.sum()),
    "sd_def": d.sd_def.mean()}), include_groups=False).reset_index()
summ = summ.merge(C.drop(columns="n"), on=["target", "c_def"])
print("\n=== held-out error (LOWER IS BETTER) beside the consensus (validation only, read once)")
print("    c_def 0 = the hybrid's defense (no box prior); 1 = the team-priced defensive prior")
for k in KS:
    print(f"\n-- K = {k}")
    print(summ[summ.k == k].drop(columns="k").to_string(index=False))
    for lvl in ("stint", "game"):
        s = summ[summ.k == k]
        b = s.loc[s[lvl].idxmin()]
        edge = b.c_def in (min(CS), max(CS))
        print(f"   argmin {lvl:5s}: target={b.target:8s} c_def={b.c_def}  "
              f"{'AT A GRID EDGE' if edge else 'interior'}")

for ref_t, ref_c, what in (("pts", 0.0, "the shipped hybrid"), (TARGETS[-1], 0.0, "hybrid + xPTS(ft), the chosen system")):
    if ref_t not in TARGETS or ref_c not in CS:
        continue
    print(f"\n=== paired by held-out season against {what} (target={ref_t}, c_def={ref_c}); negative = better")
    for k in KS:
        d = D[D.k == k]
        ref = d[(d.target == ref_t) & (d.c_def == ref_c)].set_index("held_out")
        out = []
        for (t, c), g in d.groupby(["target", "c_def"]):
            g = g.set_index("held_out")
            for lvl, col in (("stint", "mse"), ("game", "tg")):
                diff = (g[col] - ref[col]).dropna()
                n = len(diff)
                sd = diff.std(ddof=1)
                out.append(dict(target=t, c_def=c, level=lvl, mean_diff=diff.mean(),
                                z=diff.mean() / (sd / np.sqrt(n)) if sd > 0 else np.nan,
                                wins=int((diff < 0).sum()), n=n))
        p = pd.DataFrame(out).pivot_table(index=["target", "c_def"], columns="level",
                                          values=["mean_diff", "z", "wins"])
        p = p.reindex(columns=[(a, b) for a in ("mean_diff", "z", "wins") for b in ("stint", "game")])
        print(f"\n-- K = {k}")
        print(p.to_string())

summ.round(5).to_csv(OUT / "csv" / "defweight_xpts.csv", index=False)
print(f"\nwrote outputs/defweight_xpts.parquet, outputs/defweight_xpts_consensus.parquet, "
      f"outputs/csv/defweight_xpts.csv  ({time.time() - t0:.0f}s)")
