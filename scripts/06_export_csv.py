"""Export every result table as CSV, and rank the biggest movers.

Sign convention in all of these: O:<feature> is the coefficient on the sum of the five OFFENSIVE
players' per-100 rate of <feature>, predicting points scored by that offense.  D:<feature> is the
same rate for the five DEFENSIVE players, predicting points allowed, with the sign flipped so that
positive = good on both sides.  Units are points per 100 possessions per unit of the per-100 rate.
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
SIGN = "D flipped: positive = good on both sides"


def label(side, feature):
    return f"{LABEL[feature]} ({'offense' if side == 'O' else 'defense'})"


coefs = pd.read_parquet(OUT / "coefs.parquet")
coefs["label"] = [label(s, f) for s, f in zip(coefs.side, coefs.feature)]
coefs["sign_convention"] = SIGN
coefs.to_csv(CSV / "coefs_all_runs.csv", index=False)

base = coefs[coefs.run == "base"].copy()
for col, name in (("beta", "beta"), ("se", "se")):
    for side in ("O", "D"):
        w = base[base.side == side].pivot_table(index="feature", columns="window", values=col)
        w.index = [LABEL[f] for f in w.index]
        w.to_csv(CSV / f"base_{name}_{'offense' if side == 'O' else 'defense'}.csv")

# tidy base table with 95% bounds
b = base[["window", "window_mid", "side", "feature", "label", "beta", "se"]].copy()
b["lo95"] = b.beta - 1.96 * b.se
b["hi95"] = b.beta + 1.96 * b.se
b["sign_convention"] = SIGN
b.sort_values(["side", "feature", "window_mid"]).to_csv(CSV / "base_tidy.csv", index=False)

# movers: first -> last change, and max deviation from the inverse-variance era mean
rows = []
for (s, f), g in base.groupby(["side", "feature"]):
    g = g.sort_values("window_mid")
    first, last = g.iloc[0], g.iloc[-1]
    ch = last.beta - first.beta
    se = float(np.hypot(first.se, last.se))
    mean = float(np.average(g.beta, weights=1 / g.se ** 2))
    dev = (g.beta - mean) / g.se
    i = int(dev.abs().to_numpy().argmax())
    rows.append(dict(side=s, feature=f, label=label(s, f), first_window=first.window, first=first.beta,
                     last_window=last.window, last=last.beta, change=ch, change_se=se, change_z=ch / se,
                     era_mean=mean, max_dev_z=dev.abs().max(), max_dev_window=g.iloc[i].window,
                     min_beta=g.beta.min(), max_beta=g.beta.max()))
mov = pd.DataFrame(rows).sort_values("change_z", key=abs, ascending=False)
mov["sign_convention"] = SIGN
mov.to_csv(CSV / "movers.csv", index=False)

po = pd.read_parquet(OUT / "coefs_playoffs.parquet")
po["label"] = [label(s, f) for s, f in zip(po.side, po.feature)]
po["delta_z"] = po.delta / po.delta_se
po["sign_convention"] = SIGN
po.to_csv(CSV / "playoffs_all.csv", index=False)
po[(po["pool"] == "era") & (po.lam_delta == 100.0)].sort_values("delta_z", key=abs, ascending=False).to_csv(
    CSV / "playoffs_pooled.csv", index=False)

for name in ("variance", "feature_corr"):
    p = OUT / f"{name}.parquet"
    if p.exists():
        pd.read_parquet(p).to_csv(CSV / f"{name}.csv", index=False)
for p in OUT.glob("*.csv"):
    if p.parent == OUT:
        pass
cov = OUT / "coverage_1997_2026.csv"
if cov.exists():
    pd.read_csv(cov).to_csv(CSV / "coverage_1997_2026.csv", index=False)

print(f"wrote {len(list(CSV.glob('*.csv')))} csv files to {CSV}")
for p in sorted(CSV.glob("*.csv")):
    print(f"  {p.name:40s} {p.stat().st_size / 1024:8.1f} KB")

print("\n=== biggest movers, base run, D flipped so positive = good ===")
show = mov[["label", "first", "last", "change", "change_se", "change_z", "era_mean", "max_dev_z", "max_dev_window"]]
print(show.round(3).to_string(index=False))
print("\nreminder: D:<feature> is the DEFENDER's own per-100 rate of that stat predicting his defensive impact,")
print("not the stat allowed to opponents.  So 'ORB (defense)' = a player's offensive-rebound rate predicting defense.")
