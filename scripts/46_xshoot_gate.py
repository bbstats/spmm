"""The shooter-level targets on one window: do they reproduce the season they are built from?

For every target in xshoot.TARGET_REGISTRY, per season of the window:
  * expected makes against realised, by kind: sum(x2pm)/sum(fg2m), threes, free throws
  * expected points against actual: inside (0.995, 1.005) or the per-season alignment is applied
  * the first-attempt pieces: expected points and rebound chances of first attempts against realised
  * what the target gives up: the stint and team-game spread of the target against actual points

This is the ladder's step 1 (FINDINGS.md section 18): if a ratio is off before alignment, the
counters or the curve are wrong and nothing downstream is run.

usage: python scripts/46_xshoot_gate.py [window=2024-2026] [--targets=xshoot,xshoot_flat,xshoot_x1,xcont,xcont_lineup]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.windows import build_window  # noqa: E402
from eracoef.xshoot import DEFENSE_TARGETS  # noqa: E402
from eracoef.xshoot import TARGET_REGISTRY as _OFF  # noqa: E402

TARGET_REGISTRY = {**_OFF, **DEFENSE_TARGETS}

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 4)
cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
WIN = args[0] if args else "2024-2026"
SEASONS = list(range(int(WIN.split("-")[0]), int(WIN.split("-")[1]) + 1))
hit = [a for a in sys.argv[1:] if a.startswith("--targets=")]
NAMES = hit[0].split("=")[1].split(",") if hit else list(TARGET_REGISTRY)

t0 = time.time()
wd = build_window(SEASONS, cfg)
print(f"=== window {WIN}: {wd.X.shape[0]} rows, slot counters {'present' if 'fg2a_s1' in wd.counters.columns else 'MISSING'}")
w, poss = wd.w, wd.rows["poss"].to_numpy()


def wsd(x):
    mu = np.average(x, weights=w)
    return float(np.sqrt(np.average((x - mu) ** 2, weights=w)))


rows, ok_all = [], True
for name in NAMES:
    wx, rep = TARGET_REGISTRY[name](SEASONS, cfg, wd)
    g = rep["gates"]
    print(f"\n--- {name}  ({time.time() - t0:.0f}s)")
    print(g.to_string(index=False))
    cal = rep["calibration"]
    if len(cal) and cal.applied.any():
        print("    alignment APPLIED:")
        print(cal[cal.applied].to_string(index=False))
    else:
        print("    inside the band everywhere; no alignment")
    G = pd.DataFrame({"g": wd.rows["game_idx"].to_numpy(), "h": wd.rows["is_home_off"].to_numpy(),
                      "pts": wd.y * poss / 100, "x": wx.y * poss / 100}).groupby(["g", "h"]).sum()
    print(f"    sd per 100: points {wsd(wd.y):.2f}  target {wsd(wx.y):.2f}  corr {np.corrcoef(wd.y, wx.y)[0, 1]:.3f}   "
          f"team-game sd: points {G.pts.std():.2f}  target {G.x.std():.2f}  corr {G.pts.corr(G.x):.3f}")
    ok = bool(g.r2_ok.all() and g.r3_ok.all() and g.rft_ok.all())
    ok_all &= ok
    print(f"    {'GATE PASSED' if ok else 'GATE FAILED'} for {name}")
    rows.append(g.assign(target=name, sd_pts=wsd(wd.y), sd_x=wsd(wx.y)))

R = pd.concat(rows, ignore_index=True)
R.to_parquet(OUT / "xshoot_gate.parquet", index=False)
R.round(5).to_csv(OUT / "csv" / "xshoot_gate.csv", index=False)
print(f"\n{'ALL GATES PASSED' if ok_all else 'A GATE FAILED -- see above'}   ({time.time() - t0:.0f}s)")
sys.exit(0 if ok_all else 1)
