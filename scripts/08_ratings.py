"""Player rating tables per season group: the box prior, the ridge residual, and the joint rating.

For each three-season window and each player-season:
  prior_*    the player's padded full-season rates times that window's coefficients (the box score's
             opinion), centred on the average player
  u_*        what the ridge finds beyond the box score (the RAPM residual)
  rapm_mm_*  prior + u, the mixed-model RAPM rating
  *_po       the same with the playoff coefficients (beta + delta) in place of beta
Offense, defense (flipped so positive = good) and total = offense + defense.
usage: python scripts/08_ratings.py [min_poss=1000]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.boxtable import player_names, season_box  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.stints import season_names  # noqa: E402
from eracoef.windows import build_window, player_ratings_table, window_label, window_seasons  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 2)
cfg = load_config()
MIN_POSS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
OUT = Path(cfg["_root"]) / "outputs"
CSV = OUT / "csv"
CSV.mkdir(parents=True, exist_ok=True)
FEATS = cfg["features"]
nf = len(FEATS)

coefs = pd.read_parquet(OUT / "coefs.parquet")
base = coefs[coefs.run == "base"]
po_path = OUT / "coefs_playoffs.parquet"
delta_pool = None
if po_path.exists():
    po = pd.read_parquet(po_path)
    # the playoff block pooled over the whole era: per-window delta has SEs of 0.18-0.87 per
    # coefficient, which a player's rates amplify into meaningless swings, and delta is flat
    # across eras anyway, so the pooled estimate is what the playoff ratings use
    delta_pool = po[(po["pool"] == "era") & (po["lam_delta"] == po["lam_delta"].min())]


def raw_beta(df, col="beta"):
    """Back out the model's raw 26-vector from the flipped, per-side table."""
    o = df[df.side == "O"].set_index("feature")[col].reindex(FEATS).to_numpy()
    d = df[df.side == "D"].set_index("feature")[col].reindex(FEATS).to_numpy()
    return np.concatenate([o, -d])          # D is stored flipped; the model wants it raw


out = []
t0 = time.time()
for w in window_seasons(cfg):
    seasons = list(range(w[0], w[1] + 1))
    lab = window_label(seasons)
    wd = build_window(seasons, cfg)
    beta = raw_beta(base[base.window == lab])
    beta_po = beta + raw_beta(delta_pool, "delta") if delta_pool is not None else None
    names = player_names(season_box(seasons, ["RS"], cfg),
                         pd.concat([season_names(s, "RS", cfg) for s in seasons], ignore_index=True))
    r = player_ratings_table(wd, beta, cfg, seasons, beta_po=beta_po, names=names)
    out.append(r)
    print(f"  {lab}: {len(r)} player-seasons ({time.time() - t0:.0f}s)", flush=True)
    del wd

rat = pd.concat(out, ignore_index=True)
cols = ["window", "window_mid", "season", "player_id", "player_name", "poss_off", "poss_def", "shrinkage",
        "prior_off", "prior_def", "prior_total", "u_off", "u_def", "u_total",
        "rapm_mm_off", "rapm_mm_def", "rapm_mm_total"]
cols += [c for c in rat.columns if c.endswith("_po")]
rat = rat[[c for c in cols if c in rat.columns]].sort_values(["window", "rapm_mm_total"], ascending=[True, False])
rat.to_parquet(OUT / "player_ratings.parquet", index=False)
rat.round(4).to_csv(CSV / "player_ratings.csv", index=False)

# per player per window, possession-weighted over that window's seasons
num = [c for c in rat.columns if c not in ("window", "window_mid", "season", "player_id", "player_name")]
agg = (rat.assign(w=rat.poss_off.clip(lower=1e-9))
          .groupby(["window", "window_mid", "player_id"], as_index=False)
          .apply(lambda g: pd.Series({**{c: np.average(g[c], weights=g.w) for c in num if c not in ("poss_off", "poss_def")},
                                      "poss_off": g.poss_off.sum(), "poss_def": g.poss_def.sum(),
                                      "seasons": g.season.nunique(),
                                      "player_name": g.sort_values("poss_off").player_name.iloc[-1]}), include_groups=False)
          .reset_index(drop=True))
agg = agg.sort_values(["window", "rapm_mm_total"], ascending=[True, False])
agg.to_parquet(OUT / "player_ratings_by_window.parquet", index=False)
agg.round(4).to_csv(CSV / "player_ratings_by_window.csv", index=False)

print(f"\nwrote {len(rat)} player-season rows and {len(agg)} player-window rows")
show = ["player_name", "seasons", "poss_off", "prior_total", "u_total", "rapm_mm_total",
        "rapm_mm_off", "rapm_mm_def"]
if "rapm_mm_total_po" in agg.columns:
    show.append("rapm_mm_total_po")
for lab in agg.window.unique():
    g = agg[(agg.window == lab) & (agg.poss_off >= MIN_POSS)]
    print(f"\n=== {lab}: top 10 by mixed-model RAPM (>= {MIN_POSS} possessions, n = {len(g)}) ===")
    print(g.head(10)[show].to_string(index=False))
print(f"\ncorrelation of the box prior with the ridge residual (u), by window:")
print(rat.groupby("window").apply(lambda g: pd.Series({
    "corr_prior_u_total": np.corrcoef(g.prior_total, g.u_total)[0, 1],
    "sd_prior": g.prior_total.std(), "sd_u": g.u_total.std(), "sd_rating": g.rapm_mm_total.std()}),
    include_groups=False).round(3).to_string())
