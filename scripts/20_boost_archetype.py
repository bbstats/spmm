"""Does the LRBoost correction fix the one real archetype bias?

scripts/19 finds that the TOTAL rating is not tilted toward bigs beyond what the on-court margin
independently says, but that each SIDE is biased, in opposite directions that cancel:

  offense: bigs are under-credited by about +0.23 relative to guards (z = 5.5)
  defense: bigs are over-credited  by about -0.28 relative to guards (z = -4.2)

The site shows offense and defense separately, so this is a real defect even though the headline
board is sound.  A per-side, archetype-shaped additive correction on the box prior is exactly what
a nonlinear function of the box rates can express, which is what the booster is.  So the question
that decides whether the booster earns its complexity is not "does it lower held-out MSE" but
"does it shrink these two dummies toward zero".

Compares, on the same next-window criterion and the same player pairs:
  rapm_mm = prior + u          (no correction)
  rating  = prior + boost + u  (the shipped correction)

usage: python scripts/20_boost_archetype.py
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
boost = pd.read_parquet(OUT / "ratings_boosted.parquet")[
    ["window", "player_id", "boost_off", "boost_def"]]
GRID = np.sort(longs.lam.unique())
OLD_LAM = float(GRID[int(np.abs(GRID - LAM_P).argmin())])


def wmean(x, w):
    return float(np.average(np.asarray(x, dtype=float), weights=np.asarray(w, dtype=float)))


def wcorr(x, y, w):
    x = np.asarray(x, dtype=float) - wmean(x, w)
    y = np.asarray(y, dtype=float) - wmean(y, w)
    den = np.sqrt(wmean(x ** 2, w) * wmean(y ** 2, w))
    return float(wmean(x * y, w) / den) if den > 0 else np.nan


def wls(cols, y, w, names):
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
d = (stat.merge(u, on=["window", "player_id"])
     .merge(per, on=["window", "player_id"], how="left")
     .merge(boost, on=["window", "player_id"], how="left"))
for c in ("boost_off", "boost_def"):
    d[c] = d[c].fillna(0.0)
for side in ("off", "def"):
    d[f"plain_{side}"] = d[f"prior_{side}"] + d[f"u_{side}"]
    d[f"boosted_{side}"] = d[f"prior_{side}"] + d[f"boost_{side}"] + d[f"u_{side}"]
for k in ("plain", "boosted"):
    d[f"{k}_total"] = d[f"{k}_off"] + d[f"{k}_def"]

mids = sorted(d.mid.unique())
nxt = (d[d.mid.isin(mids[1:])][["player_id", "mid", "poss_off", "rapm0_off_t18k", "rapm0_def_t18k"]]
       .assign(mid=lambda x: x.mid - 3)
       .rename(columns={"poss_off": "poss_next", "rapm0_off_t18k": "y_off", "rapm0_def_t18k": "y_def"}))
j = d[d.mid.isin(mids[:-1])].merge(nxt, on=["player_id", "mid"], how="inner")
j = j[(j.poss_off >= 3000) & (j.poss_next >= 3000) & j.bigness.notna()].copy()
j["y_total"] = j.y_off + j.y_def
j["wt"] = np.minimum(j.poss_off, j.poss_next)
cols = [f"{k}_{s}" for k in ("plain", "boosted") for s in ("off", "def", "total")] + \
       ["y_off", "y_def", "y_total", "bigness", "boost_off", "boost_def"]
for c in cols:
    mu = j.groupby("window").apply(lambda g, c=c: wmean(g[c], g.wt), include_groups=False)
    j[c] = j[c] - j.window.map(mu)
qs = j.bigness.quantile([1 / 3, 2 / 3]).to_numpy()
j["archetype"] = np.where(j.bigness >= qs[1], "big", np.where(j.bigness >= qs[0], "wing", "guard"))
print(f"{len(j)} player pairs, 3000+ possessions in both windows")

print("\n=== does the booster know about archetype at all?")
print("    mean correction by archetype tercile:")
print(j.groupby("archetype")[["boost_off", "boost_def"]].mean()
      .reindex(["guard", "wing", "big"]).round(3).to_string())

print("\n=== the archetype dummies, with and without the correction")
print("    (guard is the reference; zero in both dummy rows = no archetype bias left)")
for side in ("total", "off", "def"):
    print(f"\n  {side}:")
    for k in ("plain", "boosted"):
        t = wls([j[f"{k}_{side}"], (j.archetype == "wing").astype(float),
                 (j.archetype == "big").astype(float)], j[f"y_{side}"], j.wt,
                ["rating", "wing", "big"])
        r = t.set_index("term")
        print(f"    {k:8s} wing {r.coef['wing']:+.3f} (z {r.z['wing']:+.2f})   "
              f"big {r.coef['big']:+.3f} (z {r.z['big']:+.2f})   "
              f"corr {wcorr(j[f'{k}_{side}'], j[f'y_{side}'], j.wt):.4f}")

print("\n=== 3. and the plain next-window correlation, for the record")
for side in ("total", "off", "def"):
    a = wcorr(j[f"plain_{side}"], j[f"y_{side}"], j.wt)
    b = wcorr(j[f"boosted_{side}"], j[f"y_{side}"], j.wt)
    print(f"    {side:6s} plain {a:.4f}   boosted {b:.4f}   gain {b - a:+.4f}")

print("\n=== 4. does the shrinkage re-tune still help once the correction is in?")
print("    boosted rating = prior + boost + u(lam, ratio), scored on the same criterion")
base = j[["window", "player_id", "prior_off", "prior_def", "boost_off", "boost_def",
          "y_off", "y_def", "y_total", "wt"]]
rows = []
for (lam, ratio), g in longs.groupby(["lam", "ratio"]):
    m = base.merge(g, on=["window", "player_id"], how="inner")
    o = m.prior_off + m.boost_off + m.u_off
    dd = m.prior_def + m.boost_def + m.u_def
    rows.append(dict(lam=lam, ratio=ratio, corr_total=wcorr(o + dd, m.y_total, m.wt),
                     corr_off=wcorr(o, m.y_off, m.wt), corr_def=wcorr(dd, m.y_def, m.wt)))
g4 = pd.DataFrame(rows)
b = g4.loc[g4.corr_total.idxmax()]
c = g4[(g4.ratio == RAT_P)].iloc[int((g4[g4.ratio == RAT_P].lam - LAM_P).abs().to_numpy().argmin())]
print(f"    argmax:  lam {b.lam:8.0f} ratio {b.ratio:<7} corr_total {b.corr_total:.4f} "
      f"(off {b.corr_off:.4f}, def {b.corr_def:.4f})")
print(f"    shipped: lam {c.lam:8.0f} ratio {c.ratio:<7} corr_total {c.corr_total:.4f} "
      f"(off {c.corr_off:.4f}, def {c.corr_def:.4f})")
print(f"    gain from re-tuning on top of the correction: {b.corr_total - c.corr_total:+.4f}")
print("\n    profile at the best ratio:")
print(g4[g4.ratio == b.ratio].sort_values("lam")[["lam", "corr_off", "corr_def", "corr_total"]]
      .round(4).to_string(index=False))
