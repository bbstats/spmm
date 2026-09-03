"""Step 7: the playoff block delta, pooled over the full era first (checkpoint).

1. build the RS+PO design for the era third(s) requested (default: all 30 seasons in three thirds)
2. choose lam_delta by series-grouped CV on the playoff rows (beta plugged in, delta swept from one
   moment build per fold via set_lam_delta)
3. final delta from the cross-fitted half-fits with the playoff rows included, at that lam_delta
4. RS -> PO out-of-sample check: zero-prior RAPM vs the joint model on playoff rows
5. delta pooled over the era, by era third, and per window -> outputs/coefs_playoffs.parquet
usage: python scripts/05_playoffs.py [step=all] [n_folds=5]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.boxtable import season_box  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.cv import crossfit_beta, make_pipeline, plugin_fit  # noqa: E402
from eracoef.design import FEATURES, build_design  # noqa: E402
from eracoef.stints import build_season  # noqa: E402
from eracoef.windows import build_window, window_label, window_seasons  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
N_FOLDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
OUT = Path(cfg["_root"]) / "outputs"
LAM, RATIO = float(cfg["lam_beta"]), float(cfg["lam_ratio_beta"])
LAM_P, RATIO_P = float(cfg["lam_plugin"]), float(cfg["lam_ratio_plugin"])
GRID = [float(x) for x in cfg["lam_delta_grid"]]
THIRDS = [(1997, 2006), (2007, 2016), (2017, 2026)]


def delta_rows(beta, se, delta, delta_se, label, pool, n_po_rows):
    nf = len(FEATURES)
    r = []
    for side, off in (("O", 0), ("D", nf)):
        s = -1.0 if side == "D" else 1.0
        for j, f in enumerate(FEATURES):
            r.append(dict(pool=pool, window=label, side=side, feature=f, beta=s * beta[off + j], se=se[off + j],
                          delta=s * delta[off + j], delta_se=delta_se[off + j],
                          po_beta=s * (beta[off + j] + delta[off + j]),
                          po_se=np.sqrt(se[off + j] ** 2 + delta_se[off + j] ** 2), n_po_rows=n_po_rows))
    return pd.DataFrame(r)


def choose_lam_delta(wd, beta, n_folds=N_FOLDS, verbose=True):
    """Series-grouped CV on the playoff rows; beta plugged in, lam_delta swept from one fit per fold."""
    po = ~wd.rs_mask()
    groups = wd.groups[po]
    idx_po = np.flatnonzero(po)
    rows = []
    t0 = time.time()
    for k, (tr_i, te_i) in enumerate(GroupKFold(n_folds).split(idx_po, groups=groups)):
        te = idx_po[te_i]
        mask = np.ones(len(wd.y), bool)
        mask[te] = False
        pipe = plugin_fit(wd, beta, lams=[LAM_P], cv=2, lam_ratio=RATIO_P, lam_delta=GRID[0],
                          mask=mask, pad_target=cfg["pad_target"])
        mm = pipe["mm"]
        Xte = pipe["exposure"].transform(wd.X[te])
        for ld in GRID:
            mm.set_lam_delta(ld)
            yhat = mm.predict(Xte)
            rows.append(dict(fold=k, lam_delta=ld,
                             mse=float(np.sum(wd.w[te] * (wd.y[te] - yhat) ** 2) / np.sum(wd.w[te]))))
        if verbose:
            print(f"  fold {k + 1}/{n_folds} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows).pivot(index="fold", columns="lam_delta", values="mse")
    mean = df.mean(0)
    best = mean.idxmin()
    diff = df - df[best].to_numpy()[:, None]
    tab = pd.DataFrame({"lam_delta": mean.index, "mean_mse": mean.to_numpy(),
                        "diff_vs_best": diff.mean(0).to_numpy(), "diff_se": diff.std(0, ddof=1).to_numpy() / np.sqrt(len(df))})
    tab["in_1se_band"] = tab.diff_vs_best <= tab.diff_se
    return tab, float(best)


LIGHT = 100.0   # reporting penalty: delta is then effectively unpenalized, so its SEs are real
                # (at a heavy lam_delta the SE collapses to sigma/sqrt(lam_delta), a penalty artifact)


def fit_delta(wd, lam_deltas, label, pool):
    """Cross-fitted beta and delta at each lam_delta (one moment build, set_lam_delta per value)."""
    lam_deltas = list(lam_deltas)
    cf = crossfit_beta(wd, lams=[LAM], cv=2, lam_ratio=RATIO, lam_delta=lam_deltas[0], include_po=True,
                       pad_target=cfg["pad_target"])
    n_po = int((~wd.rs_mask()).sum())
    out, fits = [], {}
    for ld in lam_deltas:
        for h in ("A", "B"):
            cf.fits[h]["mm"].set_lam(LAM, lam_delta=ld)
        a, b = cf.fits["A"]["mm"], cf.fits["B"]["mm"]
        beta = 0.5 * (a.beta_ + b.beta_)
        beta_se = 0.5 * np.sqrt(a.beta_se_ ** 2 + b.beta_se_ ** 2)
        delta = 0.5 * (a.delta_ + b.delta_)
        delta_se = 0.5 * np.sqrt(a.delta_se_ ** 2 + b.delta_se_ ** 2)
        r = delta_rows(beta, beta_se, delta, delta_se, label, pool, n_po)
        r["lam_delta"] = ld
        out.append(r)
        fits[ld] = dict(beta=beta, beta_se=beta_se, delta=delta, delta_se=delta_se)
    return fits, pd.concat(out, ignore_index=True)


print(f"lam {LAM:.0f} ratio {RATIO} | plug-in {LAM_P:.0f} ratio {RATIO_P} | pad_target {cfg['pad_target']}")
print(f"lam_delta grid: {GRID}")

# ---------------------------------------------------------------- era thirds (also the pooled fit)
out = []
cvs = []
third_fits = {}
for w0, w1 in THIRDS:
    seasons = list(range(w0, w1 + 1))
    lab = f"{w0}-{w1}"
    t0 = time.time()
    wd = build_window(seasons, cfg, phases=("RS", "PO"))
    n_po = int((~wd.rs_mask()).sum())
    print(f"\n=== era third {lab}: rows {len(wd.y)} (PO {n_po}), player-seasons {wd.spec.n_ps}, "
          f"series {len(np.unique(wd.groups[~wd.rs_mask()]))}  ({time.time() - t0:.0f}s) ===", flush=True)
    beta_rs = crossfit_beta(wd, lam=LAM, lam_ratio=RATIO, mask=wd.rs_mask(), pad_target=cfg["pad_target"]).beta_
    tab, ld = choose_lam_delta(wd, beta_rs)
    tab["pool"] = lab
    cvs.append(tab)
    print(tab.round(4).to_string(index=False))
    print(f"  lam_delta pick: {ld:g}; 1-SE band {tab.loc[tab.in_1se_band, 'lam_delta'].tolist()}")
    fits, rows = fit_delta(wd, [LIGHT, ld], lab, "era_third")
    out.append(rows)
    third_fits[lab] = dict(fits=fits, lam_delta=ld, n_po=n_po)
    print(f"  delta at lam_delta={LIGHT:g} (reporting) vs {ld:g} (CV pick): "
          f"max |delta| {np.abs(fits[LIGHT]['delta']).max():.3f} vs {np.abs(fits[ld]['delta']).max():.3f}")
    # RS -> PO out-of-sample check
    rs, po = wd.rs_mask(), ~wd.rs_mask()
    zp = make_pipeline(wd, lam=LAM_P, lam_ratio=RATIO_P, features=[], mode="full", pad_target=cfg["pad_target"])
    zp.fit(wd.X[rs], wd.y[rs], sample_weight=wd.w[rs])
    yhat0 = zp.predict(wd.X[po])
    pj = plugin_fit(wd, beta_rs, lam=LAM_P, lam_ratio=RATIO_P, mask=rs, pad_target=cfg["pad_target"])
    yhat1 = pj.predict(wd.X[po])
    mse = lambda yh: float(np.sum(wd.w[po] * (wd.y[po] - yh) ** 2) / np.sum(wd.w[po]))  # noqa: E731
    print(f"  RS-only fit -> PO rows: zero-prior {mse(yhat0):.1f}  joint {mse(yhat1):.1f}  "
          f"diff {mse(yhat1) - mse(yhat0):+.1f}")
    del wd

# pooled over the era: inverse-variance combination of the three thirds
lab = "1997-2026"
D = np.array([third_fits[f"{a}-{b}"]["fits"][LIGHT]["delta"] for a, b in THIRDS])
S = np.array([third_fits[f"{a}-{b}"]["fits"][LIGHT]["delta_se"] for a, b in THIRDS])
B = np.array([third_fits[f"{a}-{b}"]["fits"][LIGHT]["beta"] for a, b in THIRDS])
SB = np.array([third_fits[f"{a}-{b}"]["fits"][LIGHT]["beta_se"] for a, b in THIRDS])
wgt = 1.0 / S ** 2
delta_pool = (wgt * D).sum(0) / wgt.sum(0)
delta_se_pool = np.sqrt(1.0 / wgt.sum(0))
wb = 1.0 / SB ** 2
beta_pool = (wb * B).sum(0) / wb.sum(0)
beta_se_pool = np.sqrt(1.0 / wb.sum(0))
n_po_all = sum(v["n_po"] for v in third_fits.values())
pooled = delta_rows(beta_pool, beta_se_pool, delta_pool, delta_se_pool, lab, "era", n_po_all)
pooled["lam_delta"] = LIGHT
out.append(pooled)
print("\n=== delta pooled over the era (inverse-variance over the three thirds; D flipped, positive = good) ===")
p = pooled.copy()
p["z"] = p.delta / p.delta_se
print(p[["side", "feature", "beta", "se", "delta", "delta_se", "z", "po_beta", "po_se"]].round(3).to_string(index=False))
print(f"\n|z| > 2 on delta: {p.loc[p.z.abs() > 2, ['side', 'feature', 'delta', 'delta_se']].to_dict('records')}")

# ---------------------------------------------------------------- per window
ld_pool = float(pooled["lam_delta"].iloc[0])
print(f"\n=== per window, lam_delta fixed at the pooled value {ld_pool:g} ===")
for w in window_seasons(cfg):
    seasons = list(range(w[0], w[1] + 1))
    wd = build_window(seasons, cfg, phases=("RS", "PO"))
    _, rows = fit_delta(wd, [LIGHT], window_label(seasons), "window")
    out.append(rows)
    print(f"  {window_label(seasons)} done (PO rows {int((~wd.rs_mask()).sum())})", flush=True)
    del wd

res = pd.concat(out, ignore_index=True)
res.to_parquet(OUT / "coefs_playoffs.parquet", index=False)
pd.concat(cvs, ignore_index=True).to_csv(OUT / "lam_delta_cv.csv", index=False)
print(f"\nwrote outputs/coefs_playoffs.parquet ({len(res)} rows, pools {sorted(res['pool'].unique())})")
