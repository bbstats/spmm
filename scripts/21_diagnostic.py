"""A diagnostic page for reading the ratings by eye: outputs/diagnostic.html.

This answers, in charts, the four questions the last round of analysis raised, and it deliberately
uses no Greek letters anywhere in what it prints.  Plain names throughout: "shrinkage constant",
"box-score coefficients", "spread", "tilt toward bigs".

  1. What a rating is actually made of.
  2. Does the shrinkage constant matter?  (No.)
  3. Does the board over-rate big men?  (Not on the total.  Yes on each side, in opposite
     directions that cancel.)
  4. Which players does it misprice, by name.
  5. The offense/defense trade for 2024-26, by name.
  6. Does the boosted correction earn its place?

The benchmark everywhere is the same and contains no box score at all: the same player's NEXT
three-year window measured by on-court margin only.  Different games, different teammates.

Reads only cached tables, so it runs in seconds:
  outputs/player_ratings.parquet   the canonical ratings (scripts/08_ratings.py)
  outputs/tune_players.parquet     the box-score-free benchmark (scripts/16_tune_plugin.py)
  outputs/tune_u.parquet           the shrinkage grid (scripts/16_tune_plugin.py)
usage: python scripts/21_diagnostic.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.boxtable import season_box  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.plots import DARK, ERA_STEPS, FONT, LIGHT  # noqa: E402
from eracoef.windows import window_label, window_seasons  # noqa: E402

cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
LAM_P, RAT_P = float(cfg["lam_plugin"]), float(cfg["lam_ratio_plugin"])
MIN_POSS = 3000          # a player needs this in BOTH windows to enter the benchmark
SIDES = [("total", "Total"), ("off", "Offense"), ("def", "Defense")]

rat = pd.read_parquet(OUT / "player_ratings.parquet")
stat = pd.read_parquet(OUT / "tune_players.parquet")
longs = pd.read_parquet(OUT / "tune_u.parquet")


# ------------------------------------------------------------------------------- small helpers
def wmean(x, w):
    return float(np.average(np.asarray(x, dtype=float), weights=np.asarray(w, dtype=float)))


def wsd(x, w):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(wmean((x - wmean(x, w)) ** 2, w)))


def wcorr(x, y, w):
    x = np.asarray(x, dtype=float) - wmean(x, w)
    y = np.asarray(y, dtype=float) - wmean(y, w)
    den = np.sqrt(wmean(x ** 2, w) * wmean(y ** 2, w))
    return float(wmean(x * y, w) / den) if den > 0 else np.nan


def wls(cols, y, w):
    """Weighted least squares; returns (coefficients including intercept, standard errors)."""
    w = np.asarray(w, dtype=float)
    w = w / w.mean()
    y = np.asarray(y, dtype=float)
    A = np.column_stack([np.ones(len(y))] + [np.asarray(c, dtype=float) for c in cols])
    b, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)
    r = y - A @ b
    s2 = float(np.sum(w * r ** 2) / (len(y) - A.shape[1]))
    se = np.sqrt(np.diag(np.linalg.pinv((A * w[:, None]).T @ A)) * s2)
    return b, se


def demean(d, cols, by="window"):
    for c in cols:
        mu = d.groupby(by).apply(lambda g, c=c: wmean(g[c], g.wt), include_groups=False)
        d[c] = d[c] - d[by].map(mu)
    return d


# ------------------------------------------------------------- a bigness index per player-window
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

# ------------------------------------------------------------- window w joined to window w+1
mids = sorted(stat.mid.unique())
nxt = (stat[stat.mid.isin(mids[1:])][["player_id", "mid", "poss_off", "rapm0_off_t18k", "rapm0_def_t18k"]]
       .assign(mid=lambda x: x.mid - 3)
       .rename(columns={"poss_off": "poss_next", "rapm0_off_t18k": "y_off", "rapm0_def_t18k": "y_def"}))
mid_of = stat[["window", "mid"]].drop_duplicates()
r = rat.merge(mid_of, on="window", how="left").merge(per, on=["window", "player_id"], how="left")
j = r[r.mid.isin(mids[:-1])].merge(nxt, on=["player_id", "mid"], how="inner")
j = j[(j.poss_off >= MIN_POSS) & (j.poss_next >= MIN_POSS) & j.bigness.notna()].copy()
j["y_total"] = j.y_off + j.y_def
j["wt"] = np.minimum(j.poss_off, j.poss_next)
cols = ([f"{p}_{s}" for p in ("prior", "boost", "u", "rating", "rapm_mm") for s in ("off", "def", "total")]
        + ["y_off", "y_def", "y_total", "bigness"])
j = demean(j, cols)
qs = j.bigness.quantile([1 / 3, 2 / 3]).to_numpy()
j["archetype"] = np.where(j.bigness >= qs[1], "big", np.where(j.bigness >= qs[0], "wing", "guard"))
N_PAIRS = len(j)
print(f"{N_PAIRS} player pairs with {MIN_POSS}+ possessions in two consecutive windows")

payload = {"n_pairs": N_PAIRS, "min_poss": MIN_POSS}

# --- 1. what a rating is made of -------------------------------------------------------------
board = rat[rat.poss_off >= 1000]
payload["build"] = [
    {"side": lab,
     "prior": round(wsd(board[f"prior_{s}"], board.poss_off), 3),
     "boost": round(wsd(board[f"boost_{s}"], board.poss_off), 3),
     "resid": round(wsd(board[f"u_{s}"], board.poss_off), 3)}
    for s, lab in SIDES]

# --- 2. does the shrinkage constant matter? ---------------------------------------------------
# the family here is box prior + on-court residual, with no correction, which is the family the
# shrinkage constant actually controls
b2 = j[["window", "player_id", "prior_off", "prior_def", "y_off", "y_def", "y_total", "wt"]]
rows = []
for (lam, ratio), g in longs.groupby(["lam", "ratio"]):
    m = b2.merge(g, on=["window", "player_id"], how="inner")
    if not len(m):
        continue
    rows.append(dict(lam=float(lam), ratio=float(ratio),
                     corr=wcorr((m.prior_off + m.u_off) + (m.prior_def + m.u_def), m.y_total, m.wt)))
knob = pd.DataFrame(rows)
best_ratio = knob.loc[knob["corr"].idxmax(), "ratio"]
kk = knob[knob.ratio == best_ratio].sort_values("lam")
shipped = knob[knob.ratio == RAT_P].iloc[int((knob[knob.ratio == RAT_P].lam - LAM_P).abs().to_numpy().argmin())]
bestrow = knob.loc[knob["corr"].idxmax()]
payload["knob"] = {
    "lam": [round(v, 1) for v in kk.lam], "corr": [round(v, 4) for v in kk["corr"]],
    "shipped": {"lam": round(float(shipped.lam), 1), "corr": round(float(shipped["corr"]), 4)},
    "best": {"lam": round(float(bestrow.lam), 1), "corr": round(float(bestrow["corr"]), 4)}}

# --- 3. tilt toward bigs, per standard deviation of the thing itself ---------------------------
def tilt(x):
    x = np.asarray(x, dtype=float)
    sd = wsd(x, j.wt)
    b, se = wls([j.bigness], (x - wmean(x, j.wt)) / sd, j.wt)
    return round(float(b[1]), 4), round(float(se[1]), 4)


payload["tilt"] = []
for s, lab in SIDES:
    ev, ev_se = tilt(j[f"y_{s}"])
    ra, ra_se = tilt(j[f"rating_{s}"])
    pr, pr_se = tilt(j[f"prior_{s}"])
    payload["tilt"].append({"side": lab, "evidence": ev, "evidence_se": ev_se,
                            "rating": ra, "rating_se": ra_se, "prior": pr, "prior_se": pr_se})

# --- 4. who gets mispriced -------------------------------------------------------------------
mis = {}
for s, lab in SIDES:
    b, _ = wls([j[f"rating_{s}"]], j[f"y_{s}"], j.wt)
    mis[s] = j[f"y_{s}"] - (b[0] + b[1] * j[f"rating_{s}"])
payload["misprice"] = [
    {"n": row.player_name, "w": row.window, "p": int(row.poss_off), "b": round(float(row.bigness), 2),
     "t": round(float(mis["total"].iloc[i]), 3), "o": round(float(mis["off"].iloc[i]), 3),
     "d": round(float(mis["def"].iloc[i]), 3)}
    for i, (_, row) in enumerate(j.iterrows()) if pd.notna(row.player_name)]

# --- 5. the offense/defense trade, 2024-26 ----------------------------------------------------
# The benchmark is a single-window estimate and is shrunk much harder than the rating, so the two
# are not on the same scale.  Comparing them raw would show attenuation, not disagreement.  Both
# are turned into standard deviations above average over the same pool before they are compared.
LAB = "2024-2026"
z = stat[stat.window == LAB][["player_id", "rapm0_off_t18k", "rapm0_def_t18k"]]
t5 = (rat[(rat.window == LAB) & (rat.poss_off >= MIN_POSS)]
      .merge(z, on="player_id", how="inner")
      .merge(per[per.window == LAB], on=["window", "player_id"], how="left"))
for src, dst in (("rating_off", "or"), ("rapm0_off_t18k", "oe"),
                 ("rating_def", "dr"), ("rapm0_def_t18k", "de")):
    t5[dst] = (t5[src] - wmean(t5[src], t5.poss_off)) / wsd(t5[src], t5.poss_off)
t5 = t5.nlargest(30, "rating_total")
payload["trade"] = [
    {"n": r_.player_name, "p": int(r_.poss_off), "b": round(float(r_.bigness), 2),
     "or": round(float(r_["or"]), 2), "oe": round(float(r_.oe), 2),
     "dr": round(float(r_.dr), 2), "de": round(float(r_.de), 2)}
    for _, r_ in t5.iterrows() if pd.notna(r_.player_name)]

# --- 6. does the correction earn its place? ---------------------------------------------------
# NOTE this is measured on the corrected convention: `rating` uses the residual refit WITH the
# correction as an offset, and `rapm_mm` uses the residual fit with no correction at all.
wing = (j.archetype == "wing").astype(float)
big = (j.archetype == "big").astype(float)
payload["correction"] = []
for s, lab in SIDES:
    row = {"side": lab}
    for key, col in (("before", f"rapm_mm_{s}"), ("after", f"rating_{s}")):
        b, se = wls([j[col], wing, big], j[f"y_{s}"], j.wt)
        row[key] = {"wing": round(float(b[2]), 3), "wing_z": round(float(b[2] / se[2]), 2),
                    "big": round(float(b[3]), 3), "big_z": round(float(b[3] / se[3]), 2),
                    "corr": round(wcorr(j[col], j[f"y_{s}"], j.wt), 4)}
    payload["correction"].append(row)

print("\n=== re-measured on the corrected convention (residual refit with the correction) ===")
for row in payload["correction"]:
    b, a = row["before"], row["after"]
    print(f"  {row['side']:8s} without {b['corr']:.4f}  with {a['corr']:.4f}  "
          f"gain {a['corr'] - b['corr']:+.4f}   "
          f"wing {b['wing']:+.3f}->{a['wing']:+.3f}  big {b['big']:+.3f}->{a['big']:+.3f}")

Path(OUT / "diagnostic_payload.json").write_text(json.dumps(payload), encoding="utf-8")
print(f"\npayload: {len(json.dumps(payload)) / 1024:.0f} KB")

# ------------------------------------------------------------------------------------ the page
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnostic_page import render  # noqa: E402

p = render(payload, LIGHT, DARK, ERA_STEPS, FONT, OUT / "diagnostic.html")
print("page:", p, f"{p.stat().st_size / 1024:.0f} KB")
