"""Checkpoint 4 follow-up probes (review of 2026-09-03), all on 2012-14 RS unless noted.

A. coverage table 1997-2026 against the thresholds, and which seasons pbpstats serves
B. lam_ratio on the extended grid (0.25 ... 2.0), joint (ratio, lambda) fold-paired selection
C. tau2 per side, zero-prior vs plug-in: 1 - tau2_1/tau2_0 = share of impact the box prior explains
D. bench prior probes: (1) bucket slopes on non-garbage-time rows, (2) pad_target=poss_conditional
E. starter bucket lam_buckets at 0.5 / 0.7: CV MSE, starter u-slope, EBLUP slopes by bucket
F. bootstrap with padding constants fixed, multinomial vs Bayesian (Exp(1)) weights
usage: python scripts/04b_probes.py [n_boot=200]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.boxtable import season_box  # noqa: E402
from eracoef.checks import bootstrap_beta, calibration_slopes, variance_components  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.cv import crossfit_beta, lam_grid, lambda_ratio_grid, plugin_fit  # noqa: E402
from eracoef.design import FEATURES, build_design  # noqa: E402
from eracoef.stints import build_season  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = Path(cfg["_root"]) / "outputs"
seasons = [2012, 2013, 2014]
lams = lam_grid(cfg)
ratios = cfg["lam_ratio_grid"]
BUCKETS = (0, 500, 1500, np.inf)

print("=== A. coverage 1997-2026 (RS and PO) against the thresholds ===")
cov = pd.read_csv(OUT / "coverage_1997_2026.csv")
print(cov.to_string(index=False))
print("\nnot clean:", cov.loc[~cov.clean, ["season", "phase"]].to_dict("records"))
print("pbpstats serves possession/game data from 2000-01 onward (1996-97..1999-00 return no games); "
      "it has no lineup data, so it is only the possession-count cross-check (poss_within_2_frac is NaN before 2001).")

stints = pd.concat([build_season(s, "RS", cfg)[0] for s in seasons], ignore_index=True)
wd = build_design(stints, season_box(seasons, ["RS"], cfg), FEATURES, cfg)
non_gt = ~wd.rows["is_gt"].to_numpy()
print(f"\nwindow 2012-14 RS: rows={len(wd.y)} (non-GT {non_gt.sum()}) player-seasons={wd.spec.n_ps}")

print("\n=== B. lam_ratio on the extended grid, joint fold-paired selection over (ratio, lambda) ===")
t0 = time.time()
halffits = {}


def fit_halves(r):
    cf = crossfit_beta(wd, lams=lams, cv=cfg["cv"]["n_folds"], lam_ratio=r)
    halffits[r] = cf
    return [cf.fits["A"]["mm"], cf.fits["B"]["mm"]]


tab, lam_h, ratio, how = lambda_ratio_grid(fit_halves, ratios, lams)
print(tab.loc[tab.groupby("lam_ratio")["mean_mse"].idxmin()].round(3).to_string(index=False))
print(f"joint pick: lam_ratio {ratio}, half-fit lambda {lam_h:.0f} (by {how})  ({time.time() - t0:.0f}s)")
print("1-SE band ratios:", sorted(tab.loc[tab.in_1se_band, "lam_ratio"].unique().tolist()))
tab.to_csv(OUT / "checkpoint4b_ratio_grid.csv", index=False)
cf = halffits[ratio]
for h in ("A", "B"):
    cf.fits[h]["mm"].set_lam(lam_h)
cf.beta_ = 0.5 * (cf.fits["A"]["mm"].beta_ + cf.fits["B"]["mm"].beta_)
cf.cov_beta_ = 0.25 * (cf.fits["A"]["mm"].cov_beta_ + cf.fits["B"]["mm"].cov_beta_)
cf.beta_se_ = np.sqrt(np.maximum(np.diag(cf.cov_beta_), 0))
beta = cf.beta_

print("\n=== C. tau2 per side and the share of impact the box prior explains ===")
vc = variance_components(wd, beta, lams, ratios, n_folds=cfg["cv"]["n_folds"])
print(vc.round(4).to_string(index=False))
vc.to_csv(OUT / "checkpoint4b_variance_components.csv", index=False)
plug = vc[vc.model == "plugin"].iloc[0]
lam_plug, ratio_plug = float(plug.lam), float(plug.lam_ratio)
print(f"plug-in lambda {lam_plug:.0f}, ratio {ratio_plug}")

print("\n=== D1. bench prior: bucket slopes on non-garbage-time rows vs all rows ===")
common = dict(lam=lam_plug, lam_half=lam_h, lam_ratio=ratio, n_folds=cfg["cv"]["n_folds"], mode="crossfit", buckets=BUCKETS)
cs_all = calibration_slopes(wd, **common)
cs_ngt = calibration_slopes(wd, eval_rows=non_gt, **common)
sel = lambda d: d[d.kind.isin(["prior", "eblup"])].set_index(["kind", "side", "bucket"])["slope"]  # noqa: E731
cmp = pd.concat([sel(cs_all).rename("all_rows"), sel(cs_ngt).rename("non_gt_rows")], axis=1).reset_index()
print(cmp.round(3).to_string(index=False))

print("\n=== D2. bench prior: padding target league vs poss_conditional ===")
rows = []
for tgt in ("league", "poss_conditional"):
    c = calibration_slopes(wd, pad_target=tgt, **common)
    s = c[c.kind.isin(["prior", "eblup"])].copy()
    s["pad_target"] = tgt
    rows.append(s)
d2 = pd.concat(rows)
piv = d2.pivot_table(index=["kind", "side", "bucket"], columns="pad_target", values="slope").reset_index()
print(piv.round(3).to_string(index=False))
d2.to_csv(OUT / "checkpoint4b_pad_target.csv", index=False)
exp_c = crossfit_beta(wd, lam=lam_h, lam_ratio=ratio, pad_target="poss_conditional").fits["A"]["exposure"]
if exp_c.pad_bins_ is not None:
    print("possession-bin targets (half A covariate games, season 2013), per 100:")
    print(pd.DataFrame(exp_c.pad_target_[1], columns=FEATURES,
                       index=[f"[{a:.0f},{b:.0f})" for a, b in zip(exp_c.pad_bins_[:-1], exp_c.pad_bins_[1:])]).round(2).to_string())
    print("league target for comparison:", np.round(exp_c.league_rates_[1], 2))
cf_c = crossfit_beta(wd, lam=lam_h, lam_ratio=ratio, pad_target="poss_conditional")
print("beta shift (poss_conditional - league), in SE units:")
print(pd.DataFrame({"side": ["O"] * 13 + ["D"] * 13, "feature": FEATURES * 2,
                    "beta_league": beta, "beta_cond": cf_c.beta_,
                    "diff_in_se": (cf_c.beta_ - beta) / cf.beta_se_}).round(3).to_string(index=False))

print("\n=== E. starter bucket (>1500 poss) lam_buckets probe ===")
rows = []
for rb in (None, 0.5, 0.7):
    lb = None if rb is None else {"high_poss": rb}
    p = plugin_fit(wd, beta, lams=lams, cv=cfg["cv"]["n_folds"], lam_ratio=ratio_plug, lam_buckets=lb)["mm"]
    c = calibration_slopes(wd, lam=p.lam_, lam_half=lam_h, lam_ratio=ratio, lam_buckets=lb,
                           n_folds=cfg["cv"]["n_folds"], mode="crossfit", buckets=BUCKETS)
    g = lambda k, s, b="all": c[(c.kind == k) & (c.side == s) & (c.bucket == b)]["slope"].iloc[0]  # noqa: E731
    rows.append(dict(bucket_ratio=rb, lam=p.lam_, cv_mse=p.cv_results_["mean_mse"].min(),
                     u_O_starters=g("joint_u", "O", "[1500,inf)"), u_D_starters=g("joint_u", "D", "[1500,inf)"),
                     eblup_O_starters=g("eblup", "O", "[1500,inf)"), eblup_D_starters=g("eblup", "D", "[1500,inf)"),
                     eblup_O_bench=g("eblup", "O", "[0,500)"), eblup_D_bench=g("eblup", "D", "[0,500)"),
                     eblup_OD_all=g("eblup", "O+D"), u_OD_all=g("joint_u", "O+D")))
e = pd.DataFrame(rows)
e["cv_mse_minus_base"] = e.cv_mse - e.cv_mse.iloc[0]
print(e.round(4).to_string(index=False))
e.to_csv(OUT / "checkpoint4b_starter_bucket.csv", index=False)

print(f"\n=== F. bootstrap ({N_BOOT} reps): padding fixed, multinomial vs Bayesian ===")
res = {}
for scheme, fix in (("multinomial", False), ("multinomial", True), ("bayesian", True)):
    t0 = time.time()
    bt = bootstrap_beta(wd, lam=lam_h, lam_ratio=ratio, n_boot=N_BOOT, scheme=scheme, fix_padding=fix, verbose=False)
    key = f"{scheme}{'_fixedpad' if fix else ''}"
    res[key] = bt
    print(f"  {key}: mean|boot_mean - beta| = {np.abs(bt['boot_mean_minus_hat_in_se']).mean():.3f} SE, "
          f"max {np.abs(bt['boot_mean_minus_hat_in_se']).max():.3f} SE, "
          f"SE ratio median {np.median(bt['ratio']):.3f}  ({time.time() - t0:.0f}s)")
nf = len(FEATURES)
tb = pd.DataFrame({"side": ["O"] * nf + ["D"] * nf, "feature": FEATURES * 2, "beta": res["bayesian_fixedpad"]["beta_hat"],
                   "se_formula": res["bayesian_fixedpad"]["se_formula"]})
for key, bt in res.items():
    tb[f"shift_{key}"] = bt["boot_mean_minus_hat_in_se"]
    tb[f"seratio_{key}"] = bt["ratio"]
print(tb.round(3).to_string(index=False))
tb.to_csv(OUT / "checkpoint4b_bootstrap.csv", index=False)
b = res["bayesian_fixedpad"]
off = ~np.eye(2 * nf, dtype=bool)
print(f"bayesian+fixed padding: off-diagonal corr(formula, boot) = {np.corrcoef(b['corr_formula'][off], b['corr_boot'][off])[0, 1]:.3f}")
print(f"all |boot_mean - beta| < 0.5 SE? {bool(np.all(np.abs(b['boot_mean_minus_hat_in_se']) < 0.5))}")
