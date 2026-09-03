"""Checkpoint 4: regularization validation on 2012-14 (RS rows).

1. per-season k table (13 features) with sigma2_within / tau2_between
2. CV curves with fold-paired SEs for lambda on half A, half B and the plug-in fit; REML vs CV
3. GridSearchCV pad_scale x lam_ratio on half A (nested lambda path), fold-paired SEs, 1-SE band
4. calibration slopes on held-out folds (crossfit beta + plug-in u): prior / EBLUP / joint, O and D,
   by possession bucket (< 500, 500-1500, > 1500); lam_buckets probe on the low bucket
5. padding test: prior slope in the lowest bucket, variance-component k vs CV-preferred k
6. shrinkage table a_i = diag(G (G+lam I)^-1) per season for bench / rotation / starters
7. game-level bootstrap (200 reps) of the crossfit beta vs (Cov_A + Cov_B)/4
8. simulation: calibration by bucket and bootstrap vs formula on data with known truth
usage: python scripts/04_validation.py [n_boot=200]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.boxtable import season_box  # noqa: E402
from eracoef.checks import bootstrap_beta, calibration_slopes, grid_fold_se  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.cv import crossfit_beta, lam_grid, make_exposure, make_pipeline, plugin_fit, weighted_mse_scorer  # noqa: E402
from eracoef.design import FEATURES, build_design  # noqa: E402
from eracoef.simulate import simulate  # noqa: E402
from eracoef.stints import build_season  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = Path(cfg["_root"]) / "outputs"
seasons = [2012, 2013, 2014]
RATIO = 0.75
lams = lam_grid(cfg)

stints = pd.concat([build_season(s, "RS", cfg)[0] for s in seasons], ignore_index=True)
wd = build_design(stints, season_box(seasons, ["RS"], cfg), FEATURES, cfg)
print(f"window 2012-14 RS: rows={len(wd.y)} player-seasons={wd.spec.n_ps} games={len(wd.games)}")

print("\n=== 1. per-season k (possessions to half-weight) ===")
exp_all = make_exposure(wd, mode="full").fit(wd.X, sample_weight=wd.w)
print(exp_all.pad_k_table().round(0).to_string())
print("sigma2_within:"); print(pd.DataFrame(exp_all.k_sigma2_within_, index=wd.spec.seasons, columns=FEATURES).round(0).to_string())
print("tau2_between:"); print(pd.DataFrame(exp_all.k_tau2_between_, index=wd.spec.seasons, columns=FEATURES).round(2).to_string())
k_auto = exp_all.pad_k_table()

print("\n=== 2. lambda CV curves with fold-paired SEs, REML cross-check ===")
cf = crossfit_beta(wd, lams=lams, cv=cfg["cv"]["n_folds"], lam_ratio=RATIO)
lam_half = {}
for h in ("A", "B"):
    mm = cf.fits[h]["mm"]
    lam_half[h] = mm.lam_
    print(f"half {h}: {mm.lambda_report()}")
    r = mm.cv_results_[["lam", "mean_mse", "diff_vs_best", "diff_se", "in_1se_band", "reml_neg2ll"]].copy()
    r["reml_neg2ll"] -= r["reml_neg2ll"].min()
    print(r[(r.lam > 1000)].round(3).to_string(index=False))
pi = plugin_fit(wd, cf.beta_, lams=lams, cv=cfg["cv"]["n_folds"], lam_ratio=RATIO)
mm_pi = pi["mm"]
print(f"plug-in: {mm_pi.lambda_report()}")
r = mm_pi.cv_results_[["lam", "mean_mse", "diff_vs_best", "diff_se", "in_1se_band", "reml_neg2ll"]].copy()
r["reml_neg2ll"] -= r["reml_neg2ll"].min()
print(r[(r.lam > 1000)].round(3).to_string(index=False))
lam_plug = mm_pi.lam_
print("REML vs CV for lam_ratio (plug-in fit, beta fixed):")
rows = []
for ratio in cfg["lam_ratio_grid"]:
    p = plugin_fit(wd, cf.beta_, lams=lams, cv=cfg["cv"]["n_folds"], lam_ratio=ratio)["mm"]
    rows.append(dict(lam_ratio=ratio, lam_cv=p.lam_cv_, lam_reml=p.lam_reml_, min_mse=p.cv_results_["mean_mse"].min(),
                     reml_min=p.cv_results_["reml_neg2ll"].min()))
rr = pd.DataFrame(rows); rr["mse_minus_best"] = rr.min_mse - rr.min_mse.min(); rr["reml_minus_best"] = rr.reml_min - rr.reml_min.min()
print(rr.round(3).to_string(index=False))

print("\n=== 3. GridSearchCV pad_scale x lam_ratio on half A (nested lambda path), fold-paired SEs ===")
t0 = time.time()
subA = wd.subset(wd.half_mask("A"))
pipe = make_pipeline(wd, lams=np.geomspace(1000, 30000, 8), cv=cfg["cv"]["n_folds"], mode="crossfit", half="A")
gs = GridSearchCV(pipe, {"exposure__pad_scale": cfg["pad_scale_grid"] + [0.25, 4.0], "mm__lam_ratio": cfg["lam_ratio_grid"]},
                  cv=GroupKFold(cfg["cv"]["n_folds"]), scoring=weighted_mse_scorer(), n_jobs=-1, refit=False)
gs.fit(subA.X, subA.y, sample_weight=subA.w, groups=subA.groups)
gt = grid_fold_se(gs).sort_values(["exposure__pad_scale", "mm__lam_ratio"])
print(gt.round(3).to_string(index=False))
print(f"({time.time() - t0:.0f}s)")
gt.to_csv(OUT / "checkpoint4_grid.csv", index=False)
best = gt.loc[gt.diff_vs_best.idxmin()]
print(f"CV-preferred pad_scale {best['exposure__pad_scale']} (k = pad_scale x variance-component k), lam_ratio {best['mm__lam_ratio']}; "
      f"1-SE band covers pad_scale {sorted(gt.loc[gt.in_1se_band, 'exposure__pad_scale'].unique())} and lam_ratio {sorted(gt.loc[gt.in_1se_band, 'mm__lam_ratio'].unique())}")

print("\n=== 4. calibration slopes on held-out folds (crossfit beta + plug-in u) ===")
t0 = time.time()
lam_h = float(np.sqrt(lam_half["A"] * lam_half["B"]))
cs = calibration_slopes(wd, lam=lam_plug, lam_half=lam_h, lam_ratio=RATIO, n_folds=cfg["cv"]["n_folds"], mode="crossfit",
                        buckets=tuple(cfg["cv"]["poss_buckets"][:3]) + (np.inf,))
print(f"(lambda plug-in {lam_plug:.0f}, half-fits {lam_h:.0f}, ratio {RATIO}; {time.time() - t0:.0f}s)")
print(cs.round(3).to_string(index=False))
cs.to_csv(OUT / "checkpoint4_calibration.csv", index=False)
print("\nlam_buckets probe: low_poss (< 500 RS possessions) ratio x -> EBLUP / joint-u slopes by bucket")
for rb in (0.5, 2.0):
    c2 = calibration_slopes(wd, lam=lam_plug, lam_half=lam_h, lam_ratio=RATIO, lam_buckets={"low_poss": rb}, n_folds=cfg["cv"]["n_folds"],
                            mode="crossfit", buckets=tuple(cfg["cv"]["poss_buckets"][:3]) + (np.inf,))
    sel = c2[c2.kind.isin(["eblup", "joint_u"]) & (c2.bucket != "all")]
    print(f"  low_poss ratio {rb}:")
    print(sel.round(3).to_string(index=False))

print("\n=== 5. padding test: prior slope by bucket (lowest bucket = padding test) vs pad_scale ===")
for ps in (0.5, 1.0, 2.0):
    c3 = calibration_slopes(wd, lam=lam_plug, lam_half=lam_h, lam_ratio=RATIO, pad_scale=ps, n_folds=cfg["cv"]["n_folds"], mode="crossfit",
                            buckets=tuple(cfg["cv"]["poss_buckets"][:3]) + (np.inf,))
    sel = c3[c3.kind.isin(["prior", "eblup"]) & (c3.bucket != "all")]
    print(f"  pad_scale {ps}:")
    print(sel.round(3).to_string(index=False))

print("\n=== 6. shrinkage table (plug-in fit, lambda = %.0f) ===" % lam_plug)
print(mm_pi.shrinkage_table().round(3).to_string(index=False))

print(f"\n=== 7. game-level bootstrap of the crossfit beta ({N_BOOT} reps, lambda half-fits {lam_h:.0f}) ===")
bt = bootstrap_beta(wd, lam=lam_h, lam_ratio=RATIO, n_boot=N_BOOT)
nf = len(FEATURES)
tb = pd.DataFrame({"side": ["O"] * nf + ["D"] * nf, "feature": FEATURES * 2, "beta": bt["beta_hat"], "se_formula": bt["se_formula"],
                   "se_boot": bt["se_boot"], "ratio_boot_over_formula": bt["ratio"], "boot_mean_minus_hat_in_se": bt["boot_mean_minus_hat_in_se"]})
print(tb.round(3).to_string(index=False))
print(f"median ratio {np.median(bt['ratio']):.3f}   mean {np.mean(bt['ratio']):.3f}   range [{bt['ratio'].min():.2f}, {bt['ratio'].max():.2f}]")
off = ~np.eye(2 * nf, dtype=bool)
print(f"off-diagonal correlations: formula vs bootstrap corr = {np.corrcoef(bt['corr_formula'][off], bt['corr_boot'][off])[0, 1]:.3f}, "
      f"mean abs diff {np.mean(np.abs(bt['corr_formula'][off] - bt['corr_boot'][off])):.3f}")
tb.to_csv(OUT / "checkpoint4_bootstrap.csv", index=False)
np.save(OUT / "checkpoint4_bootstrap_betas.npy", bt["beta"])

print("\n=== 8. simulation: known truth (2 seasons x 16 teams, tau_u 1.5, sigma2 1e4, Poisson box counts, no leak) ===")
tau, eps_var = 1.5, 1.0
sim = simulate(n_seasons=2, n_teams=16, players_per_team=12, games_per_season=320, stints_per_game=(30, 50), tau_u=(tau, tau),
               eps_var=eps_var, leak=False, seed=11)
wds = build_design(sim["stints"], sim["box"], FEATURES, cfg)
lam_true = 1e4 * eps_var / tau ** 2
for f in (0.25, 1.0, 4.0):
    c = calibration_slopes(wds, lam=f * lam_true, n_folds=4, mode="crossfit")
    sel = c[c.kind.isin(["prior", "eblup", "joint_u"])]
    print(f"  lambda = {f} x true:")
    print(sel.round(3).to_string(index=False))
bts = bootstrap_beta(wds, lam=lam_true, n_boot=60, verbose=False)
print(f"  sim bootstrap/formula SE ratio: median {np.median(bts['ratio']):.3f}, range [{bts['ratio'].min():.2f}, {bts['ratio'].max():.2f}]")
