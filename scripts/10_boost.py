"""LRBoost: one gradient-boosted correction pooled over every window, on top of the frozen linear prior.

Pass 1 fits the linear prior per window, takes the de-shrunk residual u/a as the target, and fits one
ChimeraBoost over all ten windows at once (about 800 players per window is far too few for 13
features on its own).  Pass 2 refits u with the boost carried as an offset and refits the boost on
the new residual; the two boosts should agree above 0.95 if the loop has converged.

usage: python scripts/10_boost.py [--group player|team] [--passes 2] [--windows 2021-2023,2024-2026]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.boost import boost_vector, dominant_team, player_panel  # noqa: E402
from eracoef.boxtable import player_names, season_box  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.cv import crossfit_beta, plugin_fit  # noqa: E402
from eracoef.stints import season_names  # noqa: E402
from eracoef.windows import build_window, fit_pooled_boost, window_label, window_seasons  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)

ap = argparse.ArgumentParser()
ap.add_argument("--group", default="player", choices=["player", "team"])
ap.add_argument("--passes", type=int, default=2)
ap.add_argument("--windows", default="")
ap.add_argument("--min-poss", type=float, default=None)
ap.add_argument("--quality", type=int, default=None)
args = ap.parse_args()

cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
CACHE = Path(cfg["_root"]) / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
LAM_B, RAT_B = float(cfg["lam_beta"]), float(cfg["lam_ratio_beta"])
LAM_P, RAT_P = float(cfg["lam_plugin"]), float(cfg["lam_ratio_plugin"])
FEATS = cfg["features"]

wins = window_seasons(cfg)
if args.windows:
    keep = set(args.windows.split(","))
    wins = [w for w in wins if window_label(list(range(w[0], w[1] + 1))) in keep]

t0 = time.time()
W = {}
panels = []
print(f"stage 1: linear fit + panel for {len(wins)} windows", flush=True)
for w in wins:
    seasons = list(range(w[0], w[1] + 1))
    lab = window_label(seasons)
    wd = build_window(seasons, cfg)
    cf = crossfit_beta(wd, lams=[LAM_B], cv=2, lam_ratio=RAT_B, pad_target=cfg["pad_target"])
    pipe = plugin_fit(wd, cf.beta_, lams=[LAM_P], cv=2, lam_ratio=RAT_P, pad_target=cfg["pad_target"])
    grp = dominant_team(wd, season_box(seasons, ["RS"], cfg))
    pan = player_panel(wd, cf.beta_, cfg, window=lab, groups=grp, pipe=pipe, lam=LAM_P, lam_ratio=RAT_P)
    W[lab] = dict(seasons=seasons, wd=wd, beta=cf.beta_, pipe=pipe, groups=grp)
    panels.append(pan)
    a = pipe["mm"].shrinkage_diag()
    print(f"  {lab}: {wd.spec.n_ps:5d} players  mean a {a.mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)

panel = pd.concat(panels, ignore_index=True)
panel.to_parquet(CACHE / "boost_panel.parquet", index=False)
print(f"panel {panel.shape} -> data/cache/boost_panel.parquet ({time.time() - t0:.0f}s)\n", flush=True)

kw = {}
if args.min_poss is not None:
    kw["min_poss"] = args.min_poss
if args.quality is not None:
    kw["quality"] = args.quality

g_prev = {}
for p in range(1, args.passes + 1):
    print(f"pass {p}: pooled boost, folds by {args.group}", flush=True)
    res = fit_pooled_boost(panel, cfg, group_by=args.group, **kw)
    g_now = {lab: boost_vector(res, panel, window=lab) for lab in W}
    if g_prev:
        cc = [np.corrcoef(g_prev[l], g_now[l])[0, 1] for l in W if np.std(g_prev[l]) > 0 and np.std(g_now[l]) > 0]
        print(f"  corr(g_prev, g_now) per window: min {min(cc):.3f}  median {np.median(cc):.3f}", flush=True)
    g_prev = g_now
    if p == args.passes:
        break
    new = []
    for lab, d in W.items():                      # refit u with the boost carried as an offset
        pipe = plugin_fit(d["wd"], d["beta"], lams=[LAM_P], cv=2, lam_ratio=RAT_P,
                          pad_target=cfg["pad_target"], prior_offset=g_now[lab])
        d["pipe_boost"] = pipe
        new.append(player_panel(d["wd"], d["beta"], cfg, window=lab, groups=d["groups"], pipe=pipe,
                                lam=LAM_P, lam_ratio=RAT_P))
    panel = pd.concat(new, ignore_index=True)
    print(f"  refit u with offset ({time.time() - t0:.0f}s)", flush=True)

# ---- ship: g per player per window, plus the parts of the rating
rows = []
for lab, d in W.items():
    wd, g = d["wd"], g_prev[lab]
    m = wd.spec.n_ps
    exp = d["pipe"]["exposure"]
    pipe = plugin_fit(wd, d["beta"], lams=[LAM_P], cv=2, lam_ratio=RAT_P,
                      pad_target=cfg["pad_target"], prior_offset=g)
    mm = pipe["mm"]
    nf = len(FEATS)
    ro = exp.season_rates_ - exp.means_o_ / 5.0
    rd = exp.season_rates_d_ - exp.means_d_ / 5.0
    t = wd.spec.ps_table.copy()
    t["window"] = lab
    t["poss_off"], t["poss_def"] = exp.season_poss_off_, exp.season_poss_def_
    t["prior_off"], t["prior_def"] = ro @ d["beta"][:nf], -(rd @ d["beta"][nf:])
    t["boost_off"], t["boost_def"] = g[:m], -g[m:]
    t["u_off"], t["u_def"] = mm.u_[:m], -mm.u_[m:]
    rows.append(t)
    print(f"  {lab}: sd(boost) O {g[:m].std():.3f} D {g[m:].std():.3f}  "
          f"sd(u) O {mm.u_[:m].std():.3f} D {mm.u_[m:].std():.3f}", flush=True)

rat = pd.concat(rows, ignore_index=True)
for part in ("prior", "boost", "u"):
    rat[f"{part}_total"] = rat[f"{part}_off"] + rat[f"{part}_def"]
for s in ("off", "def", "total"):
    rat[f"rating_{s}"] = rat[f"prior_{s}"] + rat[f"boost_{s}"] + rat[f"u_{s}"]

allseas = sorted({s for d in W.values() for s in d["seasons"]})
names = player_names(season_box(allseas, ["RS"], cfg),
                     pd.concat([season_names(s, "RS", cfg) for s in allseas], ignore_index=True))
gp = pd.concat([W[l]["wd"].game_poss.merge(W[l]["wd"].spec.psx_table[["psx_idx", "player_id", "season"]],
                                           on="psx_idx", how="left").assign(window=l) for l in W])
pick = gp.groupby(["window", "player_id", "season"], as_index=False)["poss_off"].sum() \
         .sort_values("poss_off").drop_duplicates(["window", "player_id"], keep="last")
nm = pick.merge(names, on=["player_id", "season"], how="left")[["window", "player_id", "player_name"]]
rat = rat.merge(nm, on=["window", "player_id"], how="left").drop(columns=["ps_idx", "psx_idx"], errors="ignore")
rat.to_parquet(OUT / "ratings_boosted.parquet", index=False)
print(f"\nwrote outputs/ratings_boosted.parquet {rat.shape} ({time.time() - t0:.0f}s)")

last = window_label(list(range(wins[-1][0], wins[-1][1] + 1)))
top = rat[(rat.window == last) & (rat.poss_off >= 1000)].nlargest(25, "rating_total")
print(f"\n{last} top 25 (min 1000 poss)")
print(top[["player_name", "poss_off", "prior_total", "boost_total", "u_total", "rating_total"]].to_string(index=False))
