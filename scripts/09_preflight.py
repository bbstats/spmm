"""Pre-flight checks before the LRBoost build, on the current (player-season) fits.

1. Is the low-possession prior calibration slope of 0.32 real, or an artifact of the diagnostic?
   The bucket regressor is `Z_O @ (prior * in_bucket)`, whose variance is driven as much by HOW MANY
   low-minute players are on the floor as by which ones, and that headcount correlates with the
   outcome for reasons unrelated to the prior (blowouts, injuries, tanking).  Re-read the slope with
   `Z_O @ in_bucket` added as a control column.

2. How reliable is the target the booster would fit?  Fit the player residual u on half A's games
   and, separately, on half B's, de-shrink each by its own a_i, and correlate the two.  That is a
   model-free ceiling on what any function of the box score can learn from this target.
usage: python scripts/09_preflight.py [window=2012-2014]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.cv import plugin_fit  # noqa: E402
from eracoef.design import FEATURES  # noqa: E402
from eracoef.windows import build_window  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
WIN = sys.argv[1] if len(sys.argv) > 1 else "2012-2014"
first, last = (int(x) for x in WIN.split("-"))
seasons = list(range(first, last + 1))
BUCKETS = (0, 500, 1500, np.inf)
LAM_P, RATIO_P = float(cfg["lam_plugin"]), float(cfg["lam_ratio_plugin"])


def raw_beta(df):
    o = df[df.side == "O"].set_index("feature").beta.reindex(FEATURES).to_numpy()
    d = df[df.side == "D"].set_index("feature").beta.reindex(FEATURES).to_numpy()
    return np.concatenate([o, -d])


def wls(X, y, w):
    Xw = X * w[:, None]
    return np.linalg.solve(X.T @ Xw, Xw.T @ y)


def a_diag(mm, lam):
    """Per-Z-column shrinkage diag(G (G+lam I)^-1), in the unscaled space."""
    out = np.zeros(len(mm.u_))
    for cols, G in zip(mm.spec.season_cols(), mm.moments_.Gs):
        cf = sla.cho_factor(G + lam * np.eye(len(G)), lower=True, check_finite=False)
        out[cols] = np.diag(sla.cho_solve(cf, G, check_finite=False))
    return out


print(f"window {WIN}, plug-in lambda {LAM_P:.0f}, ratio {RATIO_P}")
wd = build_window(seasons, cfg)
coefs = pd.read_parquet(Path(cfg["_root"]) / "outputs" / "coefs.parquet")
beta = raw_beta(coefs[(coefs.run == "base") & (coefs.window == WIN)])
spec = wd.spec
m = spec.n_ps

# ---------------------------------------------------------------- 1. headcount controls
print("\n=== 1. prior calibration by possession bucket, with and without headcount controls ===")
splits = list(GroupKFold(5).split(wd.X, wd.y, wd.groups))
parts = []
for k, (tr, te) in enumerate(splits):
    trm = np.zeros(len(wd.y), bool); trm[tr] = True
    pipe = plugin_fit(wd, beta, lam=LAM_P, lam_ratio=RATIO_P, mask=trm, pad_target=cfg["pad_target"])
    exp, mm = pipe["exposure"], pipe["mm"]
    nf = mm.n_feat_
    pri_o = exp.rates_ @ mm.beta_[:nf]
    pri_d = exp.rates_d_ @ mm.beta_[nf:]
    poss = exp.poss_off_
    Xte = wd.X[te]
    ZO, ZD = Xte[:, spec.zo], Xte[:, spec.zd]
    cols = {"fold": np.full(len(te), k), "y": wd.y[te] - Xte[:, spec.f] @ mm.gamma_, "w": wd.w[te]}
    for b in range(len(BUCKETS) - 1):
        mask = ((poss >= BUCKETS[b]) & (poss < BUCKETS[b + 1])).astype(float)
        cols[f"prior_O_{b}"] = ZO @ (pri_o * mask)
        cols[f"prior_D_{b}"] = ZD @ (pri_d * mask)
        cols[f"count_O_{b}"] = ZO @ mask          # how many players of this bucket are on the floor
        cols[f"count_D_{b}"] = ZD @ mask
    parts.append(pd.DataFrame(cols))
    print(f"  fold {k + 1}/5", flush=True)

df = pd.concat(parts, ignore_index=True)
# fold fixed effects, as in checks.calibration_slopes
for c in [c for c in df.columns if c not in ("fold", "w")]:
    for k in range(5):
        f = df.fold == k
        df.loc[f, c] = df.loc[f, c] - np.average(df.loc[f, c], weights=df.loc[f, "w"])
yv, wv = df.y.to_numpy(), df.w.to_numpy()
nb = len(BUCKETS) - 1
names = [f"prior_{s}_{b}" for s in ("O", "D") for b in range(nb)]
counts = [f"count_{s}_{b}" for s in ("O", "D") for b in range(nb)]
rows = []
for label, extra in (("no controls", []), ("+ headcount controls", counts)):
    X = np.column_stack([df[n].to_numpy() for n in names + extra] + [np.ones(len(df))])
    sl = wls(X, yv, wv)
    fold_sl = np.asarray([wls(X[df.fold == k], yv[df.fold == k], wv[df.fold == k]) for k in range(5)])
    for i, n in enumerate(names):
        _, side, b = n.split("_")
        rows.append(dict(variant=label, side=side, bucket=f"[{BUCKETS[int(b)]:.0f},{BUCKETS[int(b) + 1]:.0f})",
                         slope=sl[i], se=fold_sl[:, i].std(ddof=1) / np.sqrt(5)))
out = pd.DataFrame(rows).pivot_table(index=["side", "bucket"], columns="variant", values=["slope", "se"])
print(out.round(3).to_string())

# ---------------------------------------------------------------- 2. split-half reliability
print("\n=== 2. split-half reliability of the booster's target (u de-shrunk by a) ===")
t0 = time.time()
halves = {}
for h in ("A", "B"):
    mask = wd.half_mask(h)
    pipe = plugin_fit(wd, beta, lams=[LAM_P], cv=2, lam_ratio=RATIO_P, mask=mask, pad_target=cfg["pad_target"])
    mm = pipe["mm"]
    a = a_diag(mm, LAM_P)
    halves[h] = dict(u=mm.u_.copy(), a=a, poss=pipe["exposure"].poss_off_.copy())
print(f"  two half fits in {time.time() - t0:.0f}s")
aA, aB = halves["A"]["a"], halves["B"]["a"]
tA = np.divide(halves["A"]["u"], aA, out=np.zeros_like(aA), where=aA > 1e-9)
tB = np.divide(halves["B"]["u"], aB, out=np.zeros_like(aB), where=aB > 1e-9)
poss_full = np.concatenate([halves["A"]["poss"] + halves["B"]["poss"]] * 2)
w = np.minimum(aA, aB)
rows = []
for side, sl in (("O", slice(0, m)), ("D", slice(m, 2 * m))):
    for lo, hi in ((0, 500), (500, 1500), (1500, np.inf)):
        sel = np.zeros(2 * m, bool)
        sel[sl] = True
        sel &= (poss_full >= lo) & (poss_full < hi) & (w > 1e-6)
        if sel.sum() < 20:
            continue
        x, y, ww = tA[sel], tB[sel], w[sel]
        mx, my = np.average(x, weights=ww), np.average(y, weights=ww)
        r = (np.average((x - mx) * (y - my), weights=ww)
             / np.sqrt(np.average((x - mx) ** 2, weights=ww) * np.average((y - my) ** 2, weights=ww)))
        rows.append(dict(side=side, bucket=f"[{lo},{hi})", n=int(sel.sum()), corr_halfA_halfB=r,
                         mean_a=float(np.mean(w[sel[sel]] if False else ww)),
                         sd_target=float(np.sqrt(np.average((x - mx) ** 2, weights=ww)))))
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\nThis correlation is the ceiling on what any box-score function can learn from this target.")
