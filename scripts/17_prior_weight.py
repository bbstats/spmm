"""Can `lam_plugin` alone fix the blend?  No: it cannot down-weight the prior.

The plug-in fit holds beta fixed and prices the assembled prior at exactly 1.0; the only thing
`lam_plugin` moves is how much `u` is added on top.  So if the box prior is over-weighted, lowering
`lam_plugin` does not remove weight from the prior, it only adds weight to the residual, and the
prior's share of the rating's variance stays high.  This script measures the ceiling:

  1. At each (lam, ratio) on the cached grid, regress the next window's PURE on-court RAPM on
     (prior, u) with possession weights.  The multiple correlation is the best any linear blend of
     those two pieces can do, and the coefficient on `prior` is what the prior is actually worth
     when `u` is already in the rating.  A coefficient below 1 means the shipped rating, which uses
     1.0, over-weights it.
  2. Score the explicit family `c * prior + u(lam)` on the same criterion, so the chosen `c` is
     read off the same scale the rating ships on.
  3. Do both per side, since the offensive and defensive priors are not equally trustworthy.

Reads the panel cached by scripts/16_tune_plugin.py.
usage: python scripts/17_prior_weight.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
LAM_P, RAT_P = float(cfg["lam_plugin"]), float(cfg["lam_ratio_plugin"])
OUT = Path(cfg["_root"]) / "outputs"

stat = pd.read_parquet(OUT / "tune_players.parquet")
longs = pd.read_parquet(OUT / "tune_u.parquet")


def wmean(x, w):
    return float(np.average(np.asarray(x, dtype=float), weights=np.asarray(w, dtype=float)))


def wcorr(x, y, w):
    x = np.asarray(x, dtype=float) - wmean(x, w)
    y = np.asarray(y, dtype=float) - wmean(y, w)
    den = np.sqrt(wmean(x ** 2, w) * wmean(y ** 2, w))
    return float(wmean(x * y, w) / den) if den > 0 else np.nan


def wls(cols, y, w):
    """Weighted least squares; returns (coefficients, multiple correlation)."""
    w = np.asarray(w, dtype=float)
    w = w / w.mean()
    A = np.column_stack([np.ones(len(y))] + [np.asarray(c, dtype=float) for c in cols])
    sw = np.sqrt(w)[:, None]
    b, *_ = np.linalg.lstsq(A * sw, np.asarray(y, dtype=float) * np.sqrt(w), rcond=None)
    return b[1:], wcorr(A @ b, y, w)


def panel(target="t18k", min_poss=3000):
    mids = sorted(stat.mid.unique())
    nxt = (stat[stat.mid.isin(mids[1:])][["player_id", "mid", "poss_off",
                                          f"rapm0_off_{target}", f"rapm0_def_{target}"]]
           .assign(mid=lambda d: d.mid - 3)
           .rename(columns={"poss_off": "poss_next",
                            f"rapm0_off_{target}": "y_off", f"rapm0_def_{target}": "y_def"}))
    j = stat[stat.mid.isin(mids[:-1])].merge(nxt, on=["player_id", "mid"], how="inner")
    j = j[(j.poss_off >= min_poss) & (j.poss_next >= min_poss)].copy()
    j["y_total"] = j.y_off + j.y_def
    j["wt"] = np.minimum(j.poss_off, j.poss_next)
    d = j[["window", "player_id", "prior_off", "prior_def", "y_off", "y_def", "y_total", "wt"]].merge(
        longs, on=["window", "player_id"], how="inner")
    d["u_total"] = d.u_off + d.u_def
    d["prior_total"] = d.prior_off + d.prior_def
    # demean everything within window so a window-level shift cannot pose as agreement
    for c in ("prior_off", "prior_def", "prior_total", "u_off", "u_def", "u_total",
              "y_off", "y_def", "y_total"):
        mu = d.groupby(["window", "lam", "ratio"]).apply(
            lambda g, c=c: wmean(g[c], g.wt), include_groups=False).rename("mu")
        d = d.merge(mu, left_on=["window", "lam", "ratio"], right_index=True, how="left")
        d[c] = d[c] - d["mu"]
        d = d.drop(columns="mu")
    return d


d = panel()
LAMS = np.sort(d.lam.unique())
RATIOS = np.sort(d.ratio.unique())

print("=== 1. what is the prior worth when u is already in the rating?")
print("    weighted regression of the next window's pure on-court RAPM on (prior, u).")
print("    coef_prior below 1 means the shipped rating, which uses exactly 1.0, over-weights it.")
rows = []
for (lam, ratio), g in d.groupby(["lam", "ratio"]):
    r = dict(lam=lam, ratio=ratio)
    for side in ("total", "off", "def"):
        b, R = wls([g[f"prior_{side}"], g[f"u_{side}"]], g[f"y_{side}"], g.wt)
        r[f"cp_{side}"], r[f"cu_{side}"], r[f"R_{side}"] = b[0], b[1], R
        # the rating ships c*prior + 1*u, so the meaningful number is the ratio of the two
        r[f"c_{side}"] = b[0] / b[1] if b[1] != 0 else np.nan
    rows.append(r)
reg = pd.DataFrame(rows).sort_values(["ratio", "lam"]).reset_index(drop=True)

show = reg[reg.ratio == 0.2872][["lam", "cp_total", "cu_total", "c_total", "R_total",
                                 "c_off", "c_def", "R_off", "R_def"]]
print(show.round(3).to_string(index=False))

print("\n    the ceiling (best multiple correlation over the whole grid):")
b = reg.loc[reg.R_total.idxmax()]
print(f"      lam {b.lam:8.0f} ratio {b.ratio:<7} R_total {b.R_total:.4f}  "
      f"implied prior weight c = {b.c_total:.3f}")
cur = reg[reg.ratio == RAT_P].iloc[int((reg[reg.ratio == RAT_P].lam - LAM_P).abs().to_numpy().argmin())]
print(f"      at the shipped lam {cur.lam:.0f}: R_total {cur.R_total:.4f}, implied c = {cur.c_total:.3f} "
      f"(off {cur.c_off:.3f}, def {cur.c_def:.3f})")

print("\n=== 2. the explicit family c * prior + u(lam): weighted correlation with the target")
CS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
rows = []
for (lam, ratio), g in d.groupby(["lam", "ratio"]):
    for c in CS:
        rows.append(dict(lam=lam, ratio=ratio, c=c,
                         corr_total=wcorr(c * g.prior_total + g.u_total, g.y_total, g.wt),
                         corr_off=wcorr(c * g.prior_off + g.u_off, g.y_off, g.wt),
                         corr_def=wcorr(c * g.prior_def + g.u_def, g.y_def, g.wt)))
fam = pd.DataFrame(rows)
piv = fam.pivot_table(index="lam", columns="c", values="corr_total").loc[:, CS]
print("    corr_total by lam (rows) and prior weight c (columns), at the best ratio per cell:")
best_ratio = fam.groupby("ratio").corr_total.max().idxmax()
piv = (fam[fam.ratio == best_ratio].pivot_table(index="lam", columns="c", values="corr_total")
       .loc[:, CS])
print(f"    ratio = {best_ratio}")
print(piv.round(4).to_string())
b = fam.loc[fam.corr_total.idxmax()]
print(f"\n    argmax: c {b.c}, lam {b.lam:.0f}, ratio {b.ratio}, corr_total {b.corr_total:.4f}")
b1 = fam[fam.c == 1.0].loc[fam[fam.c == 1.0].corr_total.idxmax()]
print(f"    best with c = 1.0 (the current family): lam {b1.lam:.0f}, ratio {b1.ratio}, "
      f"corr_total {b1.corr_total:.4f}")
for side in ("off", "def"):
    bs = fam.loc[fam[f"corr_{side}"].idxmax()]
    print(f"    {side}: argmax c {bs.c}, lam {bs.lam:.0f}, ratio {bs.ratio}, "
          f"corr {bs[f'corr_{side}']:.4f}")

fam.to_csv(OUT / "csv" / "prior_weight_grid.csv", index=False)
reg.to_csv(OUT / "csv" / "prior_weight_regression.csv", index=False)
print("\nwrote outputs/csv/prior_weight_grid.csv and prior_weight_regression.csv")
