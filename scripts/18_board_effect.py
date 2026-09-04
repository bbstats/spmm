"""Does re-tuning the ratings shrinkage actually change the leaderboard?

scripts/16 and 17 locate the best (lam, ratio, prior weight) against the next-window criterion.
The gain is small.  This script asks the two questions that decide whether it is worth shipping:

  1. Is the gain distinguishable from noise?  Paired bootstrap over player pairs and over window
     pairs on the difference in weighted correlation, shipped against re-tuned.
  2. Does the board change?  The 2024-26 top 20 under both settings, the rank correlation between
     them, and the archetype mix of the top 20 (the actual complaint was "bigs, bigs, bigs").

Reads the panel cached by scripts/16_tune_plugin.py.
usage: python scripts/18_board_effect.py [lam=10898.7] [ratio=0.18] [c=1.0]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.boxtable import player_names, season_box  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.stints import season_names  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
LAM_P, RAT_P = float(cfg["lam_plugin"]), float(cfg["lam_ratio_plugin"])
OUT = Path(cfg["_root"]) / "outputs"

stat = pd.read_parquet(OUT / "tune_players.parquet")
longs = pd.read_parquet(OUT / "tune_u.parquet")
GRID = np.sort(longs.lam.unique())


def nearest(v, grid):
    return float(grid[int(np.abs(grid - v).argmin())])


NEW_LAM = nearest(float(sys.argv[1]) if len(sys.argv) > 1 else 10898.7, GRID)
NEW_RAT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.18
NEW_C = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
OLD_LAM = nearest(LAM_P, GRID)


def wmean(x, w):
    return float(np.average(np.asarray(x, dtype=float), weights=np.asarray(w, dtype=float)))


def wcorr(x, y, w):
    x = np.asarray(x, dtype=float) - wmean(x, w)
    y = np.asarray(y, dtype=float) - wmean(y, w)
    den = np.sqrt(wmean(x ** 2, w) * wmean(y ** 2, w))
    return float(wmean(x * y, w) / den) if den > 0 else np.nan


def rating(lam, ratio, c=1.0):
    """prior * c + u(lam, ratio), one row per player-window."""
    u = longs[(longs.lam == lam) & (longs.ratio == ratio)][["window", "player_id", "u_off", "u_def"]]
    d = stat.merge(u, on=["window", "player_id"], how="inner")
    d["off"] = c * d.prior_off + d.u_off
    d["def"] = c * d.prior_def + d.u_def
    d["total"] = d["off"] + d["def"]
    return d


# ------------------------------------------------------------------ 1. is the gain real?
def eval_pairs(min_poss=3000, target="t18k"):
    mids = sorted(stat.mid.unique())
    nxt = (stat[stat.mid.isin(mids[1:])][["player_id", "mid", "poss_off",
                                          f"rapm0_off_{target}", f"rapm0_def_{target}"]]
           .assign(mid=lambda d: d.mid - 3)
           .rename(columns={"poss_off": "poss_next",
                            f"rapm0_off_{target}": "y_off", f"rapm0_def_{target}": "y_def"}))
    base = stat[stat.mid.isin(mids[:-1])][["window", "mid", "player_id", "poss_off"]].merge(
        nxt, on=["player_id", "mid"], how="inner")
    base = base[(base.poss_off >= min_poss) & (base.poss_next >= min_poss)].copy()
    base["y"] = base.y_off + base.y_def
    base["wt"] = np.minimum(base.poss_off, base.poss_next)
    old = rating(OLD_LAM, RAT_P)[["window", "player_id", "total"]].rename(columns={"total": "old"})
    new = rating(NEW_LAM, NEW_RAT, NEW_C)[["window", "player_id", "total"]].rename(columns={"total": "new"})
    d = base.merge(old, on=["window", "player_id"]).merge(new, on=["window", "player_id"])
    for c in ("old", "new", "y"):                     # demean within window
        mu = d.groupby("window").apply(lambda g, c=c: wmean(g[c], g.wt), include_groups=False)
        d[c] = d[c] - d.window.map(mu)
    return d


d = eval_pairs()
c_old, c_new = wcorr(d.old, d.y, d.wt), wcorr(d.new, d.y, d.wt)
print(f"=== 1. shipped (lam {OLD_LAM:.0f}, ratio {RAT_P}) vs re-tuned "
      f"(lam {NEW_LAM:.0f}, ratio {NEW_RAT}, c {NEW_C})")
print(f"    weighted correlation with the next window's pure on-court RAPM, n = {len(d)}")
print(f"      shipped  {c_old:.4f}")
print(f"      re-tuned {c_new:.4f}")
print(f"      gain     {c_new - c_old:+.4f}")

rng = np.random.default_rng(0)
for label, unit in (("player pairs", None), ("window pairs (cluster)", "window")):
    diffs = []
    if unit is None:
        idx = np.arange(len(d))
        for _ in range(2000):
            s = rng.choice(idx, size=len(idx), replace=True)
            g = d.iloc[s]
            diffs.append(wcorr(g.new, g.y, g.wt) - wcorr(g.old, g.y, g.wt))
    else:
        keys = d[unit].unique()
        for _ in range(2000):
            s = rng.choice(keys, size=len(keys), replace=True)
            g = pd.concat([d[d[unit] == k] for k in s], ignore_index=True)
            diffs.append(wcorr(g.new, g.y, g.wt) - wcorr(g.old, g.y, g.wt))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"      bootstrap over {label:22s}: mean {diffs.mean():+.4f}  "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  P(gain > 0) = {(diffs > 0).mean():.3f}")

# ------------------------------------------------------------------ 2. does the board change?
LAB = "2024-2026"
seasons = [2024, 2025, 2026]
box = season_box(seasons, ["RS"], cfg)
names = player_names(box, pd.concat([season_names(s, "RS", cfg) for s in seasons], ignore_index=True))
mins = box[box.phase == "RS"].groupby("player_id", as_index=False)["minutes"].sum()
nm = names.sort_values("season").drop_duplicates("player_id", keep="last")[["player_id", "player_name"]]

# a crude archetype index from the box score itself: rebounds and blocks up, assists and threes down
per = box[box.phase == "RS"].groupby("player_id", as_index=False)[
    ["minutes", "orb", "drb", "blk", "ast", "fg3m"]].sum()
for c in ("orb", "drb", "blk", "ast", "fg3m"):
    per[c] = per[c] / per.minutes.clip(lower=1) * 36
per["bigness"] = per.orb + per.blk + 0.3 * per.drb - 0.5 * per.ast - 0.4 * per.fg3m

old = rating(OLD_LAM, RAT_P)
new = rating(NEW_LAM, NEW_RAT, NEW_C)
b = (old[old.window == LAB][["player_id", "poss_off", "prior_off", "prior_def", "total"]]
     .rename(columns={"total": "old"})
     .merge(new[new.window == LAB][["player_id", "total"]].rename(columns={"total": "new"}), on="player_id")
     .merge(nm, on="player_id", how="left").merge(per[["player_id", "bigness"]], on="player_id", how="left"))
b = b[b.poss_off >= 1000].copy()
b["prior_total"] = b.prior_off + b.prior_def
b["rank_old"] = b.old.rank(ascending=False)
b["rank_new"] = b.new.rank(ascending=False)

print(f"\n=== 2. the {LAB} board, 1000+ possessions, n = {len(b)}")
print(f"    Spearman rank correlation, shipped vs re-tuned: "
      f"{b[['old', 'new']].corr(method='spearman').iloc[0, 1]:.4f}")
print(f"    top-20 overlap: {len(set(b.nlargest(20, 'old').player_id) & set(b.nlargest(20, 'new').player_id))}/20")
print("\n    shipped top 20:")
print(b.nlargest(20, "old")[["player_name", "poss_off", "prior_total", "old", "rank_new", "bigness"]]
      .round(2).to_string(index=False))
print("\n    re-tuned top 20:")
print(b.nlargest(20, "new")[["player_name", "poss_off", "prior_total", "new", "rank_old", "bigness"]]
      .round(2).to_string(index=False))

qs = b.bigness.quantile([1 / 3, 2 / 3]).to_numpy()
b["archetype"] = np.where(b.bigness >= qs[1], "big", np.where(b.bigness >= qs[0], "wing", "guard"))
print("\n    archetype mix (terciles of a bigness index over the same player pool):")
mix = pd.DataFrame({
    "all players": b.archetype.value_counts(normalize=True),
    "shipped top 20": b.nlargest(20, "old").archetype.value_counts(normalize=True),
    "re-tuned top 20": b.nlargest(20, "new").archetype.value_counts(normalize=True),
    "shipped top 50": b.nlargest(50, "old").archetype.value_counts(normalize=True),
    "re-tuned top 50": b.nlargest(50, "new").archetype.value_counts(normalize=True),
}).reindex(["guard", "wing", "big"]).fillna(0.0)
print(mix.round(3).to_string())
