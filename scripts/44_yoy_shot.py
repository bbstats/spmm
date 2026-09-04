"""Out-of-season test of the stage 4 target: does xPTS(shot) predict a season it never saw?

The same criterion as scripts/38_yoy.py and scripts/42_defweight_xpts.py -- hold out H, train on the
symmetric neighbourhood, predict every stint of H, score against ACTUAL points at stint and
team-game level -- with the targets the ratings are fit to as the thing under test:

    pts         actual points                      (the original)
    xpts_ft     free throws replaced by expectation (stage 3, the chosen system)
    xshot_mult  stage 4: first attempt conditioned on, everything downstream marginalised with the
                lineup's four-factor rates; lineup eFG% scales the make probability multiplicatively
    xshot_add   the same with the eFG% shift applied additively

The four-factor rates for a training block are fit ON that block and cross-fitted A/B within it.
That is legal here because the held-out season is never in any training block, so nothing about H
reaches the target.  (It would NOT be legal for a within-season split: half-A rows scored by the fit
trained on half B leak B's outcomes into A's target.)

Each target is run at c_def = 0 (the hybrid's defense) and at the c_def the criterion selected in
scripts/42_defweight_xpts.py, so the frontier is measured for the new target too.

usage: python scripts/44_yoy_shot.py [first=1998] [last=2025] [--k=2,4] [--c=0,0.5]
                                     [--targets=pts,xpts_ft,xshot_mult,xshot_add]
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
from eracoef.xpts import xpts_design  # noqa: E402

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
CS = _flag("c", [0.0, 0.5])
TARGETS = _flag("targets", ["pts", "xpts_ft", "xshot_mult", "xshot_add"], str)
LAM = float(cfg["lam_plugin"])
RATIO = float(cfg["lam_ratio_plugin"])
S0, S1 = int(cfg["first_season"]), int(cfg["last_season"])

panel = pd.read_parquet(OUT / "xrapm_panel.parquet")
coefs = pd.read_parquet(OUT / "coefs.parquet")
base = coefs[coefs.run == "base"]
WIN_OF = {s: window_label(list(range(w[0], w[1] + 1)))
          for w in window_seasons(cfg) for s in range(w[0], w[1] + 1)}


def betas(labs, c):
    d = base[~base.window.isin(labs)]
    dd = d[d.side == "D"].groupby("feature")["beta"].mean().reindex(FE).to_numpy()
    off = hybrid_beta(panel, FE, labs)[:nf]
    return np.concatenate([off, -c * dd])


_cache: dict = {}
gate_rows = []


def design(seasons, target):
    key = (tuple(seasons), target)
    if key not in _cache:
        if len(_cache) > 6:
            _cache.clear()
        if target.startswith("xshot_"):
            wd_pts = design(seasons, "pts")
            wd, rep = xpts_design(list(seasons), cfg, variant=target.split("_")[1], wd_pts=wd_pts)
            g = rep["gates"].assign(block="+".join(map(str, seasons)), variant=target,
                                    p_clip=rep["clips"]["p_clip_rate"], mult_clip=rep["clips"]["mult_clip_rate"],
                                    calibrated=bool(len(rep["calibration"]) and rep["calibration"].applied.any()))
            gate_rows.append(g)
            _cache[key] = wd
        else:
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
    pipe = plugin_fit(wd_t, beta, lam=LAM, lam_ratio=RATIO, pad_target=cfg["pad_target"])
    exp, mm = pipe["exposure"], pipe["mm"]
    mt = wd_t.spec.n_ps
    ro = exp.season_rates_ - exp.means_o_ / 5.0
    rd = exp.season_rates_d_ - exp.means_d_ / 5.0
    return pd.DataFrame({"player_id": wd_t.spec.ps_table["player_id"].to_numpy(),
                         "o": ro @ beta[:nf] + mm.u_[:mt], "d": rd @ beta[nf:] + mm.u_[mt:]})


rows = []
t0 = time.time()
held = [h for h in range(FIRST, LAST + 1) if S0 <= h <= S1]
print(f"held-out {held[0]}-{held[-1]} ({len(held)}), K {KS}, targets {TARGETS}, c_def {CS}\n")
for h in held:
    wd = design([h], "pts")
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
                                 tg_n=float(gg.poss.sum()), sd_off=float(r["o"].std()), sd_def=float(r["d"].std())))
    print(f"  {h} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_parquet(OUT / "yoy_shot.parquet", index=False)      # checkpoint

D = pd.DataFrame(rows)
D.to_parquet(OUT / "yoy_shot.parquet", index=False)
if gate_rows:
    G = pd.concat(gate_rows, ignore_index=True)
    G.to_parquet(OUT / "yoy_shot_gates.parquet", index=False)
    print("\n=== the closure on every training block (gates must hold out of sample too)")
    print(f"    attempt closure exp/obs: min {(G.exp_mult / G.obs_mult).min():.4f}  max {(G.exp_mult / G.obs_mult).max():.4f}")
    print(f"    points ratio: min {G.ratio.min():.4f}  max {G.ratio.max():.4f}   "
          f"blocks calibrated {int(G.groupby(['block', 'variant']).calibrated.first().sum())} of {G.groupby(['block', 'variant']).ngroups}")
    print(f"    clip rates: make-prob max {100 * G.p_clip.max():.3f}%   multiplier max {100 * G.mult_clip.max():.3f}%")
    print(G.groupby("season")[["obs_mult", "exp_mult", "oreb", "ratio"]].mean().round(4).T.to_string())

summ = D.groupby(["k", "c_def", "target"]).apply(lambda d: pd.Series({
    "stint": float((d.mse * d.n).sum() / d.n.sum()),
    "game": float((d.tg * d.tg_n).sum() / d.tg_n.sum()),
    "sd_off": d.sd_off.mean(), "sd_def": d.sd_def.mean()}), include_groups=False).reset_index()
print("\n=== held-out error by target (LOWER IS BETTER)")
for k in KS:
    print(f"\n-- K = {k}")
    print(summ[summ.k == k].drop(columns="k").to_string(index=False))

print("\n=== paired by held-out season, against pts at the same c_def and K; negative = better")
for k in KS:
    for c in CS:
        d = D[(D.k == k) & (D.c_def == c)]
        ref = d[d.target == "pts"].set_index("held_out")
        out = []
        for t, g in d.groupby("target"):
            if t == "pts":
                continue
            g = g.set_index("held_out")
            for lvl, col in (("stint", "mse"), ("game", "tg")):
                diff = (g[col] - ref[col]).dropna()
                n = len(diff)
                sd = diff.std(ddof=1)
                out.append(dict(target=t, level=lvl, mean_diff=diff.mean(),
                                z=diff.mean() / (sd / np.sqrt(n)) if sd > 0 else np.nan,
                                wins=int((diff < 0).sum()), n=n))
        p = pd.DataFrame(out).pivot_table(index="target", columns="level", values=["mean_diff", "z", "wins"])
        p = p.reindex(columns=[(a, b) for a in ("mean_diff", "z", "wins") for b in ("stint", "game")])
        print(f"\n-- K = {k}, c_def = {c}")
        print(p.to_string())

if "xpts_ft" in TARGETS:
    print("\n=== the shot term's increment over the free-throw term alone (xshot - xpts_ft), game level")
    for k in KS:
        for c in CS:
            d = D[(D.k == k) & (D.c_def == c)]
            ref = d[d.target == "xpts_ft"].set_index("held_out")
            for t in [t for t in TARGETS if t.startswith("xshot")]:
                diff = (d[d.target == t].set_index("held_out").tg - ref.tg).dropna()
                n = len(diff)
                z = diff.mean() / (diff.std(ddof=1) / np.sqrt(n))
                print(f"  K={k} c_def={c:<4} {t:11s} mean {diff.mean():+8.4f}  z {z:+6.2f}  better in {int((diff < 0).sum())}/{n}")

summ.round(5).to_csv(OUT / "csv" / "yoy_shot.csv", index=False)
print(f"\nwrote outputs/yoy_shot.parquet, outputs/yoy_shot_gates.parquet, outputs/csv/yoy_shot.csv  ({time.time() - t0:.0f}s)")
