"""Checkpoint 3: one real window (2012-14), full cross-fitted joint fit, RS rows.

1. design + per-season stint diagnostics
2. per-season k table (13 features) from split halves
3. lam_ratio outer grid x lambda path on each half-fit (CV and REML), choose lam_ratio
4. crossfit beta with SEs (O and D, D flipped for display), sign check, half A/B table
5. plug-in u fit with its own lambda path
6. full-season and LOO beta at the same lambda beside crossfit (real-data leak sizes)
7. binned-margin variant (7 margin bins x 2 halves) beside the linear rubber band
8. OOS on the same folds: zero-prior RAPM vs crossfit+plug-in
usage: python scripts/03_checkpoint3.py [first=2012] [last=2014]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.boxtable import season_box  # noqa: E402
from eracoef.checks import beta_table, mode_comparison, oos_compare  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.cv import crossfit_beta, lam_grid, make_exposure, make_pipeline, plugin_fit  # noqa: E402
from eracoef.design import FEATURES, build_design  # noqa: E402
from eracoef.stints import build_season, season_diagnostics  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
first = int(sys.argv[1]) if len(sys.argv) > 1 else 2012
last = int(sys.argv[2]) if len(sys.argv) > 2 else 2014
seasons = list(range(first, last + 1))
OUT = Path(cfg["_root"]) / "outputs"
EXPECT_O = {"fg3m": +1, "fg2m": +1, "ftm": +1, "ast": +1, "orb": +1, "tov": -1, "fg3_miss": -1, "fg2_miss": -1, "ft_miss": -1}
EXPECT_D = {"stl": +1, "blk": +1, "drb": +1}   # flipped sign: positive = good defense

print(f"=== 1. window {first}-{last}, RS rows ===")
t0 = time.time()
stints = pd.concat([build_season(s, "RS", cfg)[0] for s in seasons], ignore_index=True)
diag = pd.DataFrame([season_diagnostics(s, "RS", cfg) for s in seasons])
print(diag[["season", "n_games", "n_stints", "stints_per_game", "poss_per_team_game", "valid_poss_frac", "unresolved_subs_per_game",
            "periods_starters_failed", "points_match_frac", "poss_abs_diff_per_game_vs_pbpstats", "poss_within_2_frac", "five_distinct_frac"]].to_string(index=False))
box = season_box(seasons, ["RS"], cfg)
wd = build_design(stints, box, FEATURES, cfg)
rs = wd.rs_mask()
print(f"rows={len(wd.y)}  player-seasons={wd.spec.n_ps}  games={len(wd.games)}  X={wd.X.shape}  "
      f"half A rows={wd.half_mask('A').sum()}  half B rows={wd.half_mask('B').sum()}  ({time.time() - t0:.0f}s)")

print("\n=== 2. per-season padding k (possessions to half-weight), from odd/even split halves of all RS games ===")
exp_all = make_exposure(wd, mode="full").fit(wd.X, sample_weight=wd.w)
kt = exp_all.pad_k_table()
print(kt.round(0).to_string())
print("\nsigma2_within (per-100 rate noise per possession):")
print(pd.DataFrame(exp_all.k_sigma2_within_, index=wd.spec.seasons, columns=FEATURES).round(1).to_string())
print("tau2_between (between-player variance of true rates):")
print(pd.DataFrame(exp_all.k_tau2_between_, index=wd.spec.seasons, columns=FEATURES).round(3).to_string())
print("league rates per 100 possessions:")
print(pd.DataFrame(exp_all.league_rates_, index=wd.spec.seasons, columns=FEATURES).round(2).to_string())
for h in ("A", "B"):
    e = make_exposure(wd, mode="crossfit", half=h).fit(wd.X[wd.half_mask(h)])
    print(f"k from the covariate half of fit {h} (games of half {'B' if h == 'A' else 'A'}):")
    print(e.pad_k_table().round(0).to_string())
kt.to_csv(OUT / "checkpoint3_pad_k.csv")

print("\n=== 3. lam_ratio outer grid x lambda path on each half-fit (RS rows, GroupKFold(5) by game) ===")
lams = lam_grid(cfg)
ratios = cfg["lam_ratio_grid"]
rows = []
fits = {}
t0 = time.time()
for r in ratios:
    cf = crossfit_beta(wd, lams=lams, cv=cfg["cv"]["n_folds"], lam_ratio=r)
    fits[r] = cf
    rec = dict(lam_ratio=r)
    for h in ("A", "B"):
        mm = cf.fits[h]["mm"]
        res = mm.cv_results_
        rec[f"lam_cv_{h}"] = mm.lam_cv_
        rec[f"lam_reml_{h}"] = mm.lam_reml_
        rec[f"lam_used_{h}"] = mm.lam_
        rec[f"min_mse_{h}"] = res["mean_mse"].min()
        rec[f"reml_min_{h}"] = res["reml_neg2ll"].min()
    rec["mse_sum"] = rec["min_mse_A"] + rec["min_mse_B"]
    rec["reml_sum"] = rec["reml_min_A"] + rec["reml_min_B"]
    rows.append(rec)
    print(f"  ratio {r}: {time.time() - t0:.0f}s")
grid = pd.DataFrame(rows)
grid["mse_sum_minus_best"] = grid["mse_sum"] - grid["mse_sum"].min()
grid["reml_sum_minus_best"] = grid["reml_sum"] - grid["reml_sum"].min()
print(grid.to_string(index=False))
ratio_cv = float(grid.loc[grid["mse_sum"].idxmin(), "lam_ratio"])
ratio_reml = float(grid.loc[grid["reml_sum"].idxmin(), "lam_ratio"])
ratio = ratio_reml
print(f"lam_ratio: CV {ratio_cv}, REML {ratio_reml} -> using {ratio}")
cf = fits[ratio]
for h in ("A", "B"):
    mm = cf.fits[h]["mm"]
    print(f"half {h}: {mm.lambda_report()}")
    res = mm.cv_results_[["lam", "mean_mse", "diff_vs_best", "diff_se", "in_1se_band", "reml_neg2ll"]].copy()
    res["reml_neg2ll"] -= res["reml_neg2ll"].min()
    print(res.round(3).to_string(index=False))
    th, d_med = mm.theta_median_starter()
    print(f"  theta at the median starter (d = {d_med:.0f}): {th:.3f}")
    print("  " + mm.summary().replace("\n", "\n  "))
grid.to_csv(OUT / "checkpoint3_ratio_grid.csv", index=False)

print("\n=== 4. crossfit beta (per-100 units; D flipped so positive = good on both sides) ===")
tb = cf.coef_table(flip_defense=True)
ht = cf.half_table()
tb["beta_A"] = np.where(tb.side == "D", -ht["beta_A"], ht["beta_A"])
tb["beta_B"] = np.where(tb.side == "D", -ht["beta_B"], ht["beta_B"])
tb["z"] = tb["beta"] / tb["se"]
def _expect(side, f):
    e = (EXPECT_O if side == "O" else EXPECT_D).get(f)
    return "" if e is None else ("+" if e > 0 else "-")
tb["expected"] = [_expect(s, f) for s, f in zip(tb.side, tb.feature)]
tb["sign_ok"] = [("" if e == "" else ("ok" if np.sign(b) == (1 if e == "+" else -1) else "WRONG")) for e, b in zip(tb.expected, tb.beta)]
print(tb.to_string(index=False))
print("sign check failures:", tb[tb.sign_ok == "WRONG"][["side", "feature", "beta", "se"]].to_dict("records"))
tb.to_csv(OUT / "checkpoint3_beta.csv", index=False)

print("\n=== 5. plug-in u fit (beta fixed, all RS rows, full-season rates, own lambda path) ===")
t0 = time.time()
pi = plugin_fit(wd, cf.beta_, lams=lams, cv=cfg["cv"]["n_folds"], lam_ratio=ratio)
mm = pi["mm"]
print(f"{time.time() - t0:.0f}s")
print(mm.summary())
print(mm.lambda_report())
th, d_med = mm.theta_median_starter()
print(f"theta at the median starter (d = {d_med:.0f}): {th:.3f}")
print("F coefficients:")
print(pd.DataFrame({"name": wd.spec.f_names, "gamma": mm.gamma_, "se": mm.gamma_se_}).round(3).to_string(index=False))
lam_plug = mm.lam_

print(f"\n=== 6. full-season and LOO beta at lambda = {lam_plug:.0f} (all RS rows) beside crossfit ===")
mc = mode_comparison(wd, lam=lam_plug, lam_ratio=ratio)
mc["beta_crossfit"] = np.where(mc.side == "D", -mc.beta_crossfit, mc.beta_crossfit)
mc["beta_full"] = np.where(mc.side == "D", -mc.beta_full, mc.beta_full)
mc["beta_loo"] = np.where(mc.side == "D", -mc.beta_loo, mc.beta_loo)
mc["full_minus_cf_in_se"] = (mc.beta_full - mc.beta_crossfit) / mc.se_crossfit
mc["loo_minus_cf_in_se"] = (mc.beta_loo - mc.beta_crossfit) / mc.se_crossfit
print("(crossfit here refits at the plug-in lambda so the three share one lambda)")
print(mc[["side", "feature", "beta_crossfit", "se_crossfit", "beta_full", "beta_loo", "full_minus_cf_in_se", "loo_minus_cf_in_se", "mechanical"]].to_string(index=False))
mc.to_csv(OUT / "checkpoint3_modes.csv", index=False)

print("\n=== 7. binned margin (7 bins x 2 halves) vs linear rubber band ===")
wdb = build_design(stints, box, FEATURES, cfg, margin_bins=True)
lam_half = {h: cf.fits[h]["mm"].lam_ for h in ("A", "B")}
cfb = {h: None for h in ("A", "B")}
from eracoef.cv import fit_pipeline  # noqa: E402
bA = make_pipeline(wdb, lam=lam_half["A"], lam_ratio=ratio, mode="crossfit", half="A"); fit_pipeline(bA, wdb, wdb.half_mask("A"))
bB = make_pipeline(wdb, lam=lam_half["B"], lam_ratio=ratio, mode="crossfit", half="B"); fit_pipeline(bB, wdb, wdb.half_mask("B"))
beta_bin = 0.5 * (bA["mm"].beta_ + bB["mm"].beta_)
se_bin = 0.5 * np.sqrt(bA["mm"].beta_se_ ** 2 + bB["mm"].beta_se_ ** 2)
nf = len(FEATURES)
cmp = tb[["side", "feature", "beta", "se"]].copy()
cmp["beta_binned"] = np.concatenate([beta_bin[:nf], -beta_bin[nf:]])
cmp["se_binned"] = np.concatenate([se_bin[:nf], se_bin[nf:]])
cmp["diff_in_se"] = (cmp.beta_binned - cmp.beta) / cmp.se
print(cmp.to_string(index=False))
print(f"max |diff| in SE units: {cmp.diff_in_se.abs().max():.2f}")
print("margin-bin coefficients (half A / half B), reference = bin 3 (roughly tied), first half:")
fn = wdb.spec.f_names
bins = [i for i, n in enumerate(fn) if n.startswith("mbin")]
print(pd.DataFrame({"name": [fn[i] for i in bins], "gamma_A": bA["mm"].gamma_[bins], "se_A": bA["mm"].gamma_se_[bins],
                    "gamma_B": bB["mm"].gamma_[bins], "se_B": bB["mm"].gamma_se_[bins]}).round(2).to_string(index=False))
print("linear rubber band in the base fits (half A / half B): margin, margin_frem =",
      [round(v, 3) for v in cf.fits["A"]["mm"].gamma_[[wd.spec.f_names.index("margin"), wd.spec.f_names.index("margin_frem")]]],
      [round(v, 3) for v in cf.fits["B"]["mm"].gamma_[[wd.spec.f_names.index("margin"), wd.spec.f_names.index("margin_frem")]]])
cmp.to_csv(OUT / "checkpoint3_margin_bins.csv", index=False)

print("\n=== 8. OOS on the same folds: zero-prior RAPM vs crossfit beta + plug-in u ===")
t0 = time.time()
zp = make_pipeline(wd, lams=lams, cv=cfg["cv"]["n_folds"], features=[], mode="full", lam_ratio=ratio)
zp.fit(wd.X, wd.y, sample_weight=wd.w, groups=wd.groups)
lam_zero = zp["mm"].lam_
print(f"zero-prior lambda: {zp['mm'].lambda_report()}")
oc = oos_compare(wd, lam=lam_plug, lam_ratio=ratio, n_folds=cfg["cv"]["n_folds"], mode="crossfit", lam_zero=lam_zero)
print(oc.to_string(index=False))
d = oc["diff"]
print(f"mean diff {d.mean():.2f}  se {d.std(ddof=1) / np.sqrt(len(d)):.2f}  joint wins {int((d < 0).sum())}/{len(d)} folds   ({time.time() - t0:.0f}s)")
oc.to_csv(OUT / "checkpoint3_oos.csv", index=False)
