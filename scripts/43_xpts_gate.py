"""Stage 4 gate: does the expected-points closure reproduce the season it is built from?

Builds xPTS(shot) for one window under both lineup-scaling variants and reports, per season:

  1. the ATTEMPT closure -- expected attempts per possession reaching one, against the observed
     continuation multiplier from the counters.  This checks the geometry (miss rate x OREB% x the
     turnover leak) on its own, before any points are involved.
  2. the POINTS closure -- sum(xpts) / sum(pts), which must sit in [0.995, 1.005] or the per-season
     affine calibration is applied (and reported).
  3. clip rates -- make probabilities and per-lineup multipliers outside their bands.  Above 0.5% the
     factor shrinkage is too weak.
  4. how much variance the target gives up: xPTS per 100 should have LESS spread than points per 100
     (it is an expectation), and the part it keeps should be the part that replicates.

usage: python scripts/43_xpts_gate.py [window=2024-2026] [--variants=mult,add] [--no-cache]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.windows import build_window  # noqa: E402
from eracoef.xpts import MULT_CLIP, xpts_design  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 4)
cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
WIN = args[0] if args else "2024-2026"
SEASONS = list(range(int(WIN.split("-")[0]), int(WIN.split("-")[1]) + 1))
hit = [a for a in sys.argv[1:] if a.startswith("--variants=")]
VARIANTS = hit[0].split("=")[1].split(",") if hit else ["mult", "add"]
CACHE = "--no-cache" not in sys.argv

t0 = time.time()
wd = build_window(SEASONS, cfg)
print(f"=== window {WIN}: {wd.X.shape[0]} rows, counters {'present' if wd.counters is not None else 'MISSING'}")
w = wd.w
poss = wd.rows["poss"].to_numpy()


def wsd(x):
    mu = np.average(x, weights=w)
    return float(np.sqrt(np.average((x - mu) ** 2, weights=w)))


ok_all = True
rows = []
for variant in VARIANTS:
    wx, rep = xpts_design(SEASONS, cfg, variant=variant, calibrate=True, cache=CACHE, verbose=True, wd_pts=wd)
    g = rep["gates"]
    print(f"\n--- variant {variant}  ({time.time() - t0:.0f}s)")
    print(g[["season", "obs_mult", "exp_mult", "oreb", "t2", "ratio", "share_first", "share_ft1",
             "share_cont", "mult_ok", "calib_ok"]].to_string(index=False))
    c = rep["clips"]
    print(f"    make-probability clip rate {100 * c['p_clip_rate']:.3f}%   "
          f"multiplier clip rate {100 * c['mult_clip_rate']:.3f}%  (band {MULT_CLIP})")
    cal = rep["calibration"]
    if len(cal) and cal.applied.any():
        print("    per-season affine calibration APPLIED where the aggregate missed:")
        print(cal.to_string(index=False))
    else:
        print("    aggregate inside the band everywhere; no calibration applied")

    # what the target gives up
    y_pts, y_x = wd.y, wx.y
    r = np.corrcoef(y_pts, y_x)[0, 1]
    # the same statistic the FT stage is judged by: how much stint variance is expectation
    print(f"    sd per 100: points {wsd(y_pts):.2f}   xpts {wsd(y_x):.2f}   "
          f"corr {r:.3f}   mean points {np.average(y_pts, weights=w):.2f}  mean xpts {np.average(y_x, weights=w):.2f}")
    # after summing to team-game, the ratio of variances is the share of game scoring that is luck
    G = pd.DataFrame({"g": wd.rows["game_idx"].to_numpy(), "h": wd.rows["is_home_off"].to_numpy(),
                      "poss": poss, "pts": y_pts * poss / 100, "x": y_x * poss / 100}).groupby(["g", "h"]).sum()
    print(f"    team-game sd: points {G.pts.std():.2f}   xpts {G.x.std():.2f}   "
          f"corr {G.pts.corr(G.x):.3f}")
    ok = bool(g.mult_ok.all()) and c["p_clip_rate"] < 0.005 and c["mult_clip_rate"] < 0.005
    ok_all &= ok
    for _, s in g.iterrows():
        rows.append(dict(window=WIN, variant=variant, **s.to_dict(), p_clip=c["p_clip_rate"],
                         mult_clip=c["mult_clip_rate"], sd_pts=wsd(y_pts), sd_x=wsd(y_x)))
    print(f"    {'GATE PASSED' if ok else 'GATE FAILED'} for variant {variant}")

R = pd.DataFrame(rows)
R.to_parquet(OUT / "xpts_gate.parquet", index=False)
R.round(5).to_csv(OUT / "csv" / "xpts_gate.csv", index=False)
print(f"\n{'ALL GATES PASSED' if ok_all else 'A GATE FAILED -- see above'}   ({time.time() - t0:.0f}s)")
print("wrote outputs/xpts_gate.parquet")
sys.exit(0 if ok_all else 1)
