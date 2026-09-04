"""Is the board's tilt toward bigs a BIAS, or is it what the on-court evidence says?

Re-tuning the blend (scripts/16, 17) moves the next-window correlation by +0.007 and leaves the
archetype mix of the top 20 alone, so the blend is not where the problem is.  This script asks the
question that decides whether there is a problem at all:

  With the rating already in the regression, does a bigness term carry a negative coefficient when
  predicting the same player's next-window PURE on-court RAPM?

If it does, the rating over-credits bigs beyond what the on-court margin supports, and the size of
the coefficient says how much.  If it does not, the board is reporting what thirty seasons of
on-court margin actually say, and no amount of re-blending will change it.

Three framings, all weighted by possessions and demeaned within window:
  1. y ~ rating                    (+ bigness)
  2. y ~ prior + u                 (+ bigness)
  3. the archetype means of the rating, its pieces, and the target side by side

Reads the panel cached by scripts/16_tune_plugin.py.
usage: python scripts/19_archetype_bias.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.boxtable import season_box  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.windows import window_label, window_seasons  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
LAM_P, RAT_P = float(cfg["lam_plugin"]), float(cfg["lam_ratio_plugin"])
OUT = Path(cfg["_root"]) / "outputs"

stat = pd.read_parquet(OUT / "tune_players.parquet")
longs = pd.read_parquet(OUT / "tune_u.parquet")
GRID = np.sort(longs.lam.unique())
OLD_LAM = float(GRID[int(np.abs(GRID - LAM_P).argmin())])


def wmean(x, w):
    return float(np.average(np.asarray(x, dtype=float), weights=np.asarray(w, dtype=float)))


def wls(cols, y, w, names):
    """Weighted least squares with honest standard errors (weights normalised to mean 1)."""
    w = np.asarray(w, dtype=float)
    w = w / w.mean()
    y = np.asarray(y, dtype=float)
    A = np.column_stack([np.ones(len(y))] + [np.asarray(c, dtype=float) for c in cols])
    sw = np.sqrt(w)[:, None]
    b, *_ = np.linalg.lstsq(A * sw, y * np.sqrt(w), rcond=None)
    r = y - A @ b
    s2 = float(np.sum(w * r ** 2) / (len(y) - A.shape[1]))
    se = np.sqrt(np.diag(np.linalg.pinv((A * w[:, None]).T @ A)) * s2)
    return pd.DataFrame({"term": ["const"] + names, "coef": b, "se": se, "z": b / se})


# --------------------------------------------------------------- a bigness index per player-window
per = []
for w in window_seasons(cfg):
    seasons = list(range(w[0], w[1] + 1))
    box = season_box(seasons, ["RS"], cfg)
    g = box[box.phase == "RS"].groupby("player_id", as_index=False)[
        ["minutes", "orb", "drb", "blk", "ast", "fg3m"]].sum()
    for c in ("orb", "drb", "blk", "ast", "fg3m"):
        g[c] = g[c] / g.minutes.clip(lower=1) * 36
    g["bigness"] = g.orb + g.blk + 0.3 * g.drb - 0.5 * g.ast - 0.4 * g.fg3m
    g["window"] = window_label(seasons)
    per.append(g[["window", "player_id", "bigness"]])
per = pd.concat(per, ignore_index=True)

u = longs[(longs.lam == OLD_LAM) & (longs.ratio == RAT_P)][["window", "player_id", "u_off", "u_def"]]
d = stat.merge(u, on=["window", "player_id"]).merge(per, on=["window", "player_id"], how="left")
d["prior_total"] = d.prior_off + d.prior_def
d["u_total"] = d.u_off + d.u_def
d["rating_total"] = d.prior_total + d.u_total

mids = sorted(d.mid.unique())
nxt = (d[d.mid.isin(mids[1:])][["player_id", "mid", "poss_off", "rapm0_off_t18k", "rapm0_def_t18k"]]
       .assign(mid=lambda x: x.mid - 3)
       .rename(columns={"poss_off": "poss_next", "rapm0_off_t18k": "y_off", "rapm0_def_t18k": "y_def"}))
j = d[d.mid.isin(mids[:-1])].merge(nxt, on=["player_id", "mid"], how="inner")
j = j[(j.poss_off >= 3000) & (j.poss_next >= 3000) & j.bigness.notna()].copy()
j["y_total"] = j.y_off + j.y_def
j["wt"] = np.minimum(j.poss_off, j.poss_next)
for c in ("prior_total", "u_total", "rating_total", "y_total", "bigness",
          "prior_off", "prior_def", "u_off", "u_def", "y_off", "y_def"):
    mu = j.groupby("window").apply(lambda g, c=c: wmean(g[c], g.wt), include_groups=False)
    j[c] = j[c] - j.window.map(mu)
j["rating_off"] = j.prior_off + j.u_off
j["rating_def"] = j.prior_def + j.u_def
print(f"{len(j)} player pairs, 3000+ possessions in both windows, shipped lam {OLD_LAM:.0f}")

print("\n=== 1. y = next window's pure on-court RAPM, regressed on the shipped rating")
for side in ("total", "off", "def"):
    print(f"\n  {side}:")
    print("    " + wls([j[f"rating_{side}"]], j[f"y_{side}"], j.wt, ["rating"])
          .round(3).to_string(index=False).replace("\n", "\n    "))
    print("    with bigness added (negative = the rating over-credits bigs):")
    print("    " + wls([j[f"rating_{side}"], j.bigness], j[f"y_{side}"], j.wt, ["rating", "bigness"])
          .round(3).to_string(index=False).replace("\n", "\n    "))

print("\n=== 2. the same with the two pieces entered separately")
for side in ("total", "off", "def"):
    print(f"\n  {side}:")
    print("    " + wls([j[f"prior_{side}"], j[f"u_{side}"], j.bigness], j[f"y_{side}"], j.wt,
                       ["prior", "u", "bigness"]).round(3).to_string(index=False).replace("\n", "\n    "))

print("\n=== 3. who does the rating misprice?")
print("    The rating's slope on the target is about 0.33, not 1, because the target is a shrunk")
print("    single-window estimate.  So comparing raw means would conflate that attenuation with")
print("    bias.  The honest table is the mean RESIDUAL after the target is regressed on the")
print("    rating: positive = the next window's on-court margin likes this group MORE than the")
print("    rating does.  A rating with no archetype bias has zero in every row.")
qs = j.bigness.quantile([1 / 3, 2 / 3]).to_numpy()
j["archetype"] = np.where(j.bigness >= qs[1], "big", np.where(j.bigness >= qs[0], "wing", "guard"))
rows = []
for side in ("total", "off", "def"):
    b = wls([j[f"rating_{side}"]], j[f"y_{side}"], j.wt, ["rating"])
    resid = j[f"y_{side}"] - (b.coef.iloc[0] + b.coef.iloc[1] * j[f"rating_{side}"])
    for a, g in j.assign(res=resid).groupby("archetype"):
        rows.append(dict(side=side, archetype=a, n=len(g), mean_resid=wmean(g.res, g.wt),
                         mean_rating=wmean(g[f"rating_{side}"], g.wt),
                         mean_prior=wmean(g[f"prior_{side}"], g.wt),
                         mean_u=wmean(g[f"u_{side}"], g.wt)))
t = pd.DataFrame(rows).pivot_table(index="archetype", columns="side",
                                   values=["mean_prior", "mean_u", "mean_rating", "mean_resid"])
print(t.reindex(["guard", "wing", "big"]).round(3).to_string())

print("\n    the same as a regression, with archetype dummies instead of a linear bigness term")
print("    (guard is the reference; a linear term cannot see a U shape):")
for side in ("total", "off", "def"):
    print(f"      {side}:")
    print("        " + wls([j[f"rating_{side}"], (j.archetype == "wing").astype(float),
                            (j.archetype == "big").astype(float)], j[f"y_{side}"], j.wt,
                           ["rating", "wing", "big"]).round(3).to_string(index=False)
          .replace("\n", "\n        "))

print("\n=== 4. does the SAME tilt appear in the target itself?")
print("    If the pure on-court RAPM likes bigs as much as the rating does, the tilt is evidence,")
print("    not an artefact.  The rating and the target have different spreads, so each is")
print("    standardised first; the coefficient is then the tilt per standard deviation of the")
print("    thing itself, and the two are directly comparable.")


def std_tilt(x, label):
    x = np.asarray(x, dtype=float)
    x = (x - wmean(x, j.wt)) / np.sqrt(wmean((x - wmean(x, j.wt)) ** 2, j.wt))
    t = wls([j.bigness], x, j.wt, ["bigness"]).set_index("term")
    print(f"      {label:34s} {t.coef['bigness']:+.4f}  (se {t.se['bigness']:.4f}, "
          f"z {t.z['bigness']:+.2f})")


print("    tilt per standard deviation, total:")
std_tilt(j.y_total, "next window pure on-court RAPM")
std_tilt(j.rating_total, "the shipped rating")
std_tilt(j.prior_total, "the box prior alone")
std_tilt(j.u_total, "the on-court residual u alone")
for side in ("off", "def"):
    print(f"    tilt per standard deviation, {side}:")
    std_tilt(j[f"y_{side}"], "next window pure on-court RAPM")
    std_tilt(j[f"rating_{side}"], "the shipped rating")
    std_tilt(j[f"prior_{side}"], "the box prior alone")
