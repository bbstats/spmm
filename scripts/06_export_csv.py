"""Export every result table as CSV, and rank the biggest movers.

Sign convention: a coefficient is the points per 100 possessions that one unit per 100 of that
box-score stat is worth, holding the other twelve fixed.  `offense` uses the five offensive
players' rates against points scored; `defense` uses the five defensive players' own rates against
points allowed, flipped so positive = good; `total` is offense + defense, what the stat is worth to
a player overall (its standard error carries the covariance between the two sides).
usage: python scripts/06_export_csv.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
CSV = OUT / "csv"
CSV.mkdir(parents=True, exist_ok=True)
LABEL = {"fg3m": "3PM", "fg3_miss": "3P miss", "fg2m": "2PM", "fg2_miss": "2P miss", "ftm": "FTM",
         "ft_miss": "FT miss", "orb": "ORB", "drb": "DRB", "ast": "AST", "tov": "TOV", "stl": "STL",
         "blk": "BLK", "pf": "PF"}
ORDER = ["fg3m", "fg3_miss", "fg2m", "fg2_miss", "ftm", "ft_miss", "orb", "drb", "ast", "tov", "stl", "blk", "pf"]
SIDE = {"Total": "total", "O": "offense", "D": "defense"}
SIGN = "positive = good on both sides; total = offense + defense"

coefs = pd.read_parquet(OUT / "coefs.parquet").rename(columns={"beta": "coef"})
coefs["side_name"] = coefs.side.map(SIDE)
coefs["stat"] = coefs.feature.map(LABEL)
coefs["sign_convention"] = SIGN
coefs.to_csv(CSV / "coefs_all_runs.csv", index=False)

base = coefs[coefs.run == "base"].copy()
base["lo95"] = base.coef - 1.96 * base.se
base["hi95"] = base.coef + 1.96 * base.se
tidy = base[["window", "window_mid", "stat", "feature", "side_name", "coef", "se", "lo95", "hi95"]]
tidy.sort_values(["side_name", "feature", "window_mid"]).round(4).to_csv(CSV / "coefs_base_tidy.csv", index=False)

# wide: one row per window and stat, the three sides side by side
w = base.pivot_table(index=["window", "window_mid", "feature", "stat"], columns="side_name",
                     values=["coef", "se"]).reset_index()
w.columns = [c[0] if not c[1] else f"{c[0]}_{c[1]}" for c in w.columns]
w = w.sort_values(["window_mid", "feature"])
order = ["window", "window_mid", "stat", "feature", "coef_total", "se_total",
         "coef_offense", "se_offense", "coef_defense", "se_defense"]
w[[c for c in order if c in w.columns]].round(4).to_csv(CSV / "coefs_by_window.csv", index=False)

for side in ("Total", "O", "D"):
    g = base[base.side == side].pivot_table(index="feature", columns="window", values="coef").reindex(ORDER)
    g.index = [LABEL[f] for f in g.index]
    g.round(3).to_csv(CSV / f"coefs_grid_{SIDE[side]}.csv")

# movers, first window to last
rows = []
for (side, f), g in base.groupby(["side", "feature"]):
    g = g.sort_values("window_mid")
    first, last = g.iloc[0], g.iloc[-1]
    ch = last.coef - first.coef
    se = float(np.hypot(first.se, last.se))
    mean = float(np.average(g.coef, weights=1 / g.se ** 2))
    dev = (g.coef - mean) / g.se
    i = int(dev.abs().to_numpy().argmax())
    rows.append(dict(side=SIDE[side], stat=LABEL[f], feature=f, first_window=first.window, first=first.coef,
                     last_window=last.window, last=last.coef, change=ch, change_se=se, change_z=ch / se,
                     era_mean=mean, max_dev_z=dev.abs().max(), max_dev_window=g.iloc[i].window,
                     min_coef=g.coef.min(), max_coef=g.coef.max()))
mov = pd.DataFrame(rows).sort_values(["side", "change_z"], key=lambda c: c.abs() if c.name == "change_z" else c,
                                     ascending=[True, False])
mov["sign_convention"] = SIGN
mov.round(4).to_csv(CSV / "movers.csv", index=False)

po_path = OUT / "coefs_playoffs.parquet"
if po_path.exists():
    po = pd.read_parquet(po_path).rename(columns={"beta": "coef", "po_beta": "coef_playoff", "po_se": "se_playoff"})
    po["side_name"] = po.side.map(SIDE)
    po["stat"] = po.feature.map(LABEL)
    po["delta_z"] = po.delta / po.delta_se
    po["sign_convention"] = SIGN
    po.round(4).to_csv(CSV / "playoffs_all.csv", index=False)
    light = po.lam_delta == po.lam_delta.min()
    po[light & (po["pool"] == "era")].sort_values("delta_z", key=abs, ascending=False).round(4).to_csv(
        CSV / "playoffs_pooled.csv", index=False)
    pw = po[light & (po["pool"] == "window")].pivot_table(
        index=["window", "feature", "stat"], columns="side_name",
        values=["coef", "se", "coef_playoff", "se_playoff"]).reset_index()
    pw.columns = [c[0] if not c[1] else f"{c[0]}_{c[1]}" for c in pw.columns]
    pw.round(4).to_csv(CSV / "playoffs_by_window.csv", index=False)

for name in ("variance", "feature_corr", "player_ratings", "ratings_boosted"):
    p = OUT / f"{name}.parquet"
    if p.exists():
        pd.read_parquet(p).round(4).to_csv(CSV / f"{name}.csv", index=False)
cov = OUT / "coverage_1997_2026.csv"
if cov.exists():
    pd.read_csv(cov).to_csv(CSV / "coverage_1997_2026.csv", index=False)

print(f"wrote {len(list(CSV.glob('*.csv')))} csv files to {CSV}")
for p in sorted(CSV.glob("*.csv")):
    print(f"  {p.name:34s} {p.stat().st_size / 1024:8.1f} KB")

print("\n=== combined coefficient (offense + defense), by window ===")
g = base[base.side == "Total"].pivot_table(index="feature", columns="window", values="coef").reindex(ORDER)
g.index = [LABEL[f] for f in g.index]
print(g.round(2).to_string())
print("\n=== biggest movers, combined ===")
show = mov[mov.side == "total"][["stat", "first", "last", "change", "change_se", "change_z", "era_mean", "max_dev_z"]]
print(show.round(3).to_string(index=False))
