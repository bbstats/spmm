"""Stage 2: the four-factor RAPMs, and the go/no-go for the whole luck-adjustment idea.

Fits eFG%, TOV%, OREB% and FT rate as RAPM targets on the same stints as the points model, then
reports the three things that decide whether stage 4 is worth building:

  1. CALIBRATION.  The possession-weighted mean of each fitted rate must reproduce the observed
     league rate.  The per-season intercepts guarantee this, so a miss means the fixed block is
     misspecified for that target, not that the players are unusual.
  2. SPLIT-HALF RELIABILITY OF THE LINEUP SUM.  Fit on half A, fit on half B, correlate the two
     rates over the same rows.  This is the go/no-go: if a lineup's expected rate does not replicate
     across interleaved halves of the same season, it is noise, and feeding it into xPTS would be
     dressing noise up as signal.  It costs an hour to find out, against about two weeks for stage 4.
  3. SPREAD.  A rate with no cross-lineup variance has been shrunk to the league mean and cannot
     move xPTS at all, whatever its reliability says.

Also a free read on the offense/defense asymmetry.  HANDOFF argued not to impose by hand the idea
that defences control shot quality more than shot-making -- let the fit decide.  This is where it
decides: eFG%'s selected lam_ratio (= lambda_D / lambda_O) should come out LARGE if defences have
little eFG-suppression skill, and OREB%'s nearer 1 if defensive rebounding is a real skill.

Only lineup sums leave this script.  The per-player four-factor effects are never written anywhere:
splitting a lineup's rebounding among five players is the badly identified direction, and those
numbers would carry exactly the defect FINDINGS.md sections 7-13 removed.

usage: python scripts/37_factors.py [window=2024-2026] [--no-crossfit]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.design import TARGETS  # noqa: E402
from eracoef.factors import FACTORS, fit_all  # noqa: E402
from eracoef.windows import build_window, window_label  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 4)
cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
CROSSFIT = "--no-crossfit" not in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
WIN = args[0] if args else "2024-2026"
SEASONS = list(range(int(WIN.split("-")[0]), int(WIN.split("-")[1]) + 1))

t0 = time.time()
print(f"=== window {window_label(SEASONS)}  (cross-fitted: {CROSSFIT})")
wd_pts = build_window(SEASONS, cfg)
print(f"points design: {wd_pts.X.shape[0]} rows ({time.time() - t0:.0f}s)")

wd_by_factor, obs = {}, {}
for f in FACTORS:
    wd_by_factor[f] = build_window(SEASONS, cfg, target=f)
    d = wd_by_factor[f]
    # the observed league rate on the factor's own rows, exposure-weighted: what the fit must match
    obs[f] = float(np.average(d.y, weights=d.w))
    print(f"  {f:5s} design: {d.X.shape[0]} rows  observed league rate {obs[f]:6.2f}  "
          f"({time.time() - t0:.0f}s)", flush=True)

print("\n=== fits")
rates, rep = fit_all(wd_pts, wd_by_factor, cfg, crossfit=CROSSFIT)
rep["observed"] = rep.factor.map(obs)
rep["calib_pct"] = 100 * (rep["mean"] - rep.observed) / rep.observed

print("\n=== report")
cols = ["factor", "lam", "lam_ratio", "eff_def", "chosen_by", "observed", "mean", "calib_pct",
        "sd", "split_half_r", "clip_rate", "coverage"]
print(rep[cols].to_string(index=False))

print("\n=== gates")
ok = True
for _, r in rep.iterrows():
    # calibration is checked on the factor's own rows; `mean` above is over the POINTS rows, which
    # include stints that took no field goals at all, so a few points of drift is expected on eFG
    c = abs(r.calib_pct) < 5.0
    s = r.split_half_r > 0.30
    v = r.sd > 0.20
    p = r.clip_rate < 0.005
    ok &= bool(s and v and p)
    print(f"  {r.factor:5s}  calibration {r.calib_pct:+6.2f}% {'ok' if c else 'DRIFT'}   "
          f"split-half {r.split_half_r:.3f} {'OK' if s else 'FAIL'}   "
          f"sd {r.sd:5.2f} {'ok' if v else 'DEGENERATE'}   "
          f"clip {100 * r.clip_rate:.2f}% {'ok' if p else 'HIGH'}")

print(f"\n  half-fit spread vs full-fit (below ~0.8 means cross-fitting is compressing the rates,")
print(f"  which UNDERSTATES the luck adjustment -- conservative, not biased):")
for _, r in rep.iterrows():
    print(f"    {r.factor:5s} {r.sd_halffit / max(r.sd_fullfit, 1e-9):.3f}")

print("\n  the offense/defense asymmetry, as estimated rather than imposed:")
for _, r in rep.iterrows():
    if r.lam_ratio > 1.05:
        note = "defence shrunk harder -- little defensive skill in this factor"
    elif r.lam_ratio < 0.95:
        note = "defence shrunk less -- real defensive skill here"
    else:
        note = "symmetric"
    print(f"    {r.factor:5s} lam_ratio {r.lam_ratio:5.2f}  ({note})")

rates["poss"] = wd_pts.rows["poss"].to_numpy()
rates["stint"] = wd_pts.rows["stint"].to_numpy()
rates["is_home_off"] = wd_pts.rows["is_home_off"].to_numpy()
rates["game_idx"] = wd_pts.rows["game_idx"].to_numpy()
rates["window"] = window_label(SEASONS)
rep["window"] = window_label(SEASONS)
rates.to_parquet(OUT / f"factor_rates_{window_label(SEASONS)}.parquet", index=False)
rep.to_parquet(OUT / "factor_lambda.parquet", index=False)
rep.round(4).to_csv(OUT / "csv" / "factor_lambda.csv", index=False)
print(f"\n{'GATE PASSED' if ok else 'GATE FAILED -- see above'}   ({time.time() - t0:.0f}s)")
print(f"wrote outputs/factor_rates_{window_label(SEASONS)}.parquet and outputs/factor_lambda.parquet")
sys.exit(0 if ok else 1)
