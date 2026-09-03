"""Checkpoint 1 (rev 2): the whole pipeline on simulated data, cross-fitted beta.

1. simulate (mechanical same-game channel on makes, misses, ORB, TOV)
2. crossfit beta recovery at a fixed lambda; plug-in u fit; k table
3. leakage bias table: full / loo / crossfit at lam/10, lam, 10*lam vs the predicted biases
4. SE calibration of the crossfit beta (Cov = (Cov_A + Cov_B)/4) over replications
5. MixedModelRAPMCV lambda path: CV argmin, REML, 1-SE band, true sigma2/tau2
6. GridSearchCV over exposure__pad_scale x mm__lam_ratio (routed sample_weight + groups)
7. OOS: zero-prior RAPM vs crossfit+plug-in on the same folds
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.checks import beta_table, bias_table, mode_comparison, oos_compare, se_calibration, u_recovery  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.cv import crossfit_beta, make_pipeline, plugin_fit, weighted_mse_scorer  # noqa: E402
from eracoef.design import FEATURES, build_design  # noqa: E402
from eracoef.simulate import MECH, simulate  # noqa: E402

pd.set_option("display.width", 220, "display.max_columns", 30, "display.precision", 4)
cfg = load_config()
N_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
N_BIAS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
SIM = dict(n_seasons=2, n_teams=10, players_per_team=12, games_per_season=150, stints_per_game=(30, 50))
TAU = 1.5

print("=== 1. simulate (2 seasons x 10 teams x 12 players, 150 games/season, ~40 stints/game) ===")
print(f"mechanical same-game slopes: {MECH}")
t0 = time.time()
sim = simulate(seed=0, **SIM)
wd = build_design(sim["stints"], sim["box"], FEATURES, cfg)
nA, nB = wd.half_mask("A").sum(), wd.half_mask("B").sum()
print(f"stints={len(sim['stints'])}  rows={len(wd.y)} (half A {nA}, half B {nB})  player-seasons={wd.spec.n_ps}  X={wd.X.shape}  ({time.time() - t0:.1f}s)")

print("\n=== 2. crossfit beta at lambda = 8000 (two half-fits, averaged) ===")
t0 = time.time()
cf = crossfit_beta(wd, lam=8000.0, lam_ratio=1.0)
print(f"{time.time() - t0:.2f}s")
print(cf.summary())
tb = beta_table(cf, sim["truth"])
ht = cf.half_table()
tb["beta_A"] = ht["beta_A"].to_numpy(); tb["beta_B"] = ht["beta_B"].to_numpy()
print(tb.to_string(index=False))
print(f"mean |z| = {np.abs(tb.z).mean():.2f}   max |z| = {np.abs(tb.z).max():.2f}")
print("\nper-season auto k from half B's games (possessions to half-weight):")
print(cf.fits["A"]["exposure"].pad_k_table().round(0).to_string())
pi = plugin_fit(wd, cf.beta_, lam=8000.0)
mm = pi["mm"]
print("\nplug-in u fit on all rows with full-season rates:")
print(mm.summary())
print(f"sigma2={mm.sigma2_:.0f}  implied true lambda = sigma2/tau_u^2 = {mm.sigma2_ / TAU ** 2:.0f}")
print(u_recovery(mm, wd.spec, sim["truth"]).to_string(index=False))

print("\n=== 3a. single-dataset comparison of the three modes (O side) ===")
mc = mode_comparison(wd, lam=8000.0, truth=sim["truth"])
print(mc[mc.side == "O"].to_string(index=False))

print(f"\n=== 3b. leakage bias table over {N_BIAS} replications: full / loo / crossfit at lam/10, lam, 10*lam ===")
LAM = 8000.0
bt = bias_table(n_reps=N_BIAS, lams=(LAM / 10, LAM, 10 * LAM), cfg=cfg, sim_kwargs=SIM)
print(f"median starter possessions in the fit: full-season d = {bt.attrs['d_med_starter']:.0f}, half-season d = {bt.attrs['d_med_starter_half']:.0f}")
cols = ["lam", "theta_starter", "mode", "feature", "true", "mean_beta", "bias", "bias_se", "pred_bias", "bias_in_se", "mean_z", "sd_z"]
for lam in sorted(bt.lam.unique()):
    print(f"\n-- lambda = {lam:.0f} --")
    print(bt[bt.lam == lam][cols].to_string(index=False))
bt.to_csv(Path(cfg["_root"]) / "outputs" / "checkpoint1_bias_table.csv", index=False)

print(f"\n=== 4. SE calibration of crossfit beta over {N_REPS} replications (lambda = 8000) ===")
z, summ = se_calibration(n_reps=N_REPS, lam=8000.0, cfg=cfg, sim_kwargs=SIM, mode="crossfit")
print(summ.to_string(index=False))
print(f"overall: mean z = {z.mean():.3f}   sd z = {z.std(ddof=1):.3f}   95% coverage = {(np.abs(z) < 1.96).mean():.3f}   (n = {z.size})")

print("\n=== 5. MixedModelRAPMCV lambda path (plug-in fit, all rows, GroupKFold(5) by game) ===")
lams = np.geomspace(500, 200000, 14)
t0 = time.time()
pcv = plugin_fit(wd, cf.beta_, lams=lams, cv=5)
mcv = pcv["mm"]
print(f"fit {time.time() - t0:.2f}s")
print(mcv.lambda_report())
print(f"true sigma2/tau_u^2 of the sim = {mcv.sigma2_ / TAU ** 2:.0f}")
res = mcv.cv_results_[["lam", "mean_mse", "diff_vs_best", "diff_se", "in_1se_band", "reml_neg2ll"]].copy()
res["reml_neg2ll"] -= res["reml_neg2ll"].min()
print(res.to_string(index=False))
th, d_med = mcv.theta_median_starter()
print(f"theta at the median starter (d = {d_med:.0f}): {th:.3f}")
t0 = time.time(); mcv.set_lam(3 * mcv.lam_); print(f"set_lam(3*lam): {(time.time() - t0) * 1000:.1f} ms")
print("\nlambda path on the crossfit half-fits themselves:")
cfl = crossfit_beta(wd, lams=lams, cv=5)
for h in ("A", "B"):
    print(f"  half {h}: {cfl.fits[h]['mm'].lambda_report()}")

print("\n=== 6. GridSearchCV over exposure__pad_scale x mm__lam_ratio on half A (routed, n_jobs=-1) ===")
t0 = time.time()
subA = wd.subset(wd.half_mask("A"))
pgs = make_pipeline(wd, lams=np.geomspace(2000, 50000, 6), cv=3, mode="crossfit", half="A")
gs = GridSearchCV(pgs, {"exposure__pad_scale": [0.5, 1.0, 2.0], "mm__lam_ratio": [0.5, 1.0, 2.0]},
                  cv=GroupKFold(4), scoring=weighted_mse_scorer(), n_jobs=-1)
gs.fit(subA.X, subA.y, sample_weight=subA.w, groups=subA.groups)
cvr = pd.DataFrame(gs.cv_results_)[["param_exposure__pad_scale", "param_mm__lam_ratio", "mean_test_score", "std_test_score", "rank_test_score"]]
cvr["mean_test_score"] *= -1
print(cvr.to_string(index=False))
print(f"best: {gs.best_params_}  inner lambda = {gs.best_estimator_['mm'].lam_:.0f}   ({time.time() - t0:.1f}s)")

print("\n=== 7. OOS: zero-prior RAPM vs crossfit beta + plug-in u (same folds) ===")
print(oos_compare(wd, lam=8000.0, n_folds=5, mode="crossfit").to_string(index=False))
