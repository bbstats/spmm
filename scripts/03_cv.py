"""Pick lam and lam_ratio once, on two windows (2000-02 and 2012-14), RS rows only.

Joint fold-paired selection over the (lam_ratio, lambda) grid, REML if inside the CV 1-SE band
else the CV argmin, for two fits:
  beta   : the two cross-fitted half-fits (their per-fold MSE and REML profiles summed)
  plugin : the ratings fit with beta plugged in
Writes outputs/lambda.yaml.  If the two windows disagree by more than ~2x, it says so loudly.
usage: python scripts/03_cv.py [pad_target=poss_conditional]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.boxtable import season_box  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.cv import crossfit_beta, lam_grid, lambda_ratio_grid, plugin_fit  # noqa: E402
from eracoef.design import FEATURES, build_design  # noqa: E402
from eracoef.stints import build_season  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
PAD_TARGET = sys.argv[1] if len(sys.argv) > 1 else "poss_conditional"
OUT = Path(cfg["_root"]) / "outputs"
lams = lam_grid(cfg)
ratios_beta = cfg["lam_ratio_grid"]
ratios_plug = [0.08, 0.12, 0.18, 0.25, 0.33, 0.5, 0.75, 1.0]
res = {}

for w0, w1 in cfg["cv"]["lambda_windows"]:
    seasons = list(range(w0, w1 + 1))
    name = f"{w0}-{w1}"
    print(f"\n=== window {name} (RS rows, pad_target={PAD_TARGET}) ===")
    stints = pd.concat([build_season(s, "RS", cfg)[0] for s in seasons], ignore_index=True)
    wd = build_design(stints, season_box(seasons, ["RS"], cfg), FEATURES, cfg)
    print(f"rows={len(wd.y)} player-seasons={wd.spec.n_ps} games={len(wd.games)}")

    t0 = time.time()
    half = {}

    def fit_halves(r, half=half, wd=wd):
        cf = crossfit_beta(wd, lams=lams, cv=cfg["cv"]["n_folds"], lam_ratio=r, pad_target=PAD_TARGET)
        half[r] = cf
        return [cf.fits["A"]["mm"], cf.fits["B"]["mm"]]

    tab_b, lam_b, ratio_b, how_b = lambda_ratio_grid(fit_halves, ratios_beta, lams)
    print("beta (half-fits): best lambda per ratio")
    print(tab_b.loc[tab_b.groupby("lam_ratio")["mean_mse"].idxmin()].round(3).to_string(index=False))
    print(f"  pick: ratio {ratio_b}, lambda {lam_b:.0f} (by {how_b}); 1-SE band ratios "
          f"{sorted(tab_b.loc[tab_b.in_1se_band, 'lam_ratio'].unique().tolist())}  ({time.time() - t0:.0f}s)")

    cf = half[ratio_b]
    for h in ("A", "B"):
        cf.fits[h]["mm"].set_lam(lam_b)
    beta = 0.5 * (cf.fits["A"]["mm"].beta_ + cf.fits["B"]["mm"].beta_)
    cov = 0.25 * (cf.fits["A"]["mm"].cov_beta_ + cf.fits["B"]["mm"].cov_beta_)

    plug = {}

    def fit_plug(r, plug=plug, wd=wd, beta=beta):
        p = plugin_fit(wd, beta, lams=lams, cv=cfg["cv"]["n_folds"], lam_ratio=r, pad_target=PAD_TARGET)
        plug[r] = p
        return p["mm"]

    tab_p, lam_p, ratio_p, how_p = lambda_ratio_grid(fit_plug, ratios_plug, lams)
    print("plug-in (ratings) fit: best lambda per ratio")
    print(tab_p.loc[tab_p.groupby("lam_ratio")["mean_mse"].idxmin()].round(3).to_string(index=False))
    mm = plug[ratio_p]["mm"]; mm.set_lam(lam_p)
    print(f"  pick: ratio {ratio_p}, lambda {lam_p:.0f} (by {how_p}); 1-SE band ratios "
          f"{sorted(tab_p.loc[tab_p.in_1se_band, 'lam_ratio'].unique().tolist())}")
    print(f"  sigma2 {mm.sigma2_:.0f}  tau2_O {mm.sigma2_ / lam_p:.3f}  tau2_D {mm.sigma2_ / (lam_p * ratio_p):.3f}")
    res[name] = dict(lam_beta=lam_b, ratio_beta=ratio_b, how_beta=how_b, lam_plugin=lam_p, ratio_plugin=ratio_p,
                     how_plugin=how_p, sigma2=float(mm.sigma2_), n_rows=int(len(wd.y)), n_ps=int(wd.spec.n_ps))
    tab_b.to_csv(OUT / f"cv_beta_grid_{name}.csv", index=False)
    tab_p.to_csv(OUT / f"cv_plugin_grid_{name}.csv", index=False)

df = pd.DataFrame(res).T
print("\n=== both windows ===")
print(df.to_string())
w = list(res)
r_beta = res[w[0]]["lam_beta"] / res[w[1]]["lam_beta"]
r_plug = res[w[0]]["lam_plugin"] / res[w[1]]["lam_plugin"]
print(f"lambda ratio between windows: beta {r_beta:.2f}x, plug-in {r_plug:.2f}x")
if max(r_beta, 1 / r_beta) > 2 or max(r_plug, 1 / r_plug) > 2:
    print("*** WINDOWS DISAGREE BY MORE THAN 2x -- stop and report ***")
else:
    print("windows agree within 2x")

# geometric mean of the two windows, ratios by the more common pick
final = dict(
    pad_target=PAD_TARGET,
    lam_beta=float(np.sqrt(res[w[0]]["lam_beta"] * res[w[1]]["lam_beta"])),
    lam_ratio_beta=float(np.sqrt(res[w[0]]["ratio_beta"] * res[w[1]]["ratio_beta"])),
    lam_plugin=float(np.sqrt(res[w[0]]["lam_plugin"] * res[w[1]]["lam_plugin"])),
    lam_ratio_plugin=float(np.sqrt(res[w[0]]["ratio_plugin"] * res[w[1]]["ratio_plugin"])),
    windows={k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv) for kk, vv in v.items()} for k, v in res.items()},
)
with open(OUT / "lambda.yaml", "w") as fh:
    yaml.safe_dump(final, fh, sort_keys=False)
print("\nwritten to outputs/lambda.yaml:")
print(yaml.safe_dump({k: v for k, v in final.items() if k != "windows"}, sort_keys=False))
