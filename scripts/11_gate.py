"""The validation gate: does the boosted prior calibrate better than the linear one?

Five game-grouped folds on one window.  Per fold, beta comes from the two half-fits on the training
games and u from a plug-in fit on the same, so no regressor has seen the test games.  The booster is
fit on the other NINE windows, so the held-out games cannot reach it through its training target.
Slopes are headcount-controlled and reported by possession bucket; 1.0 is calibrated, and
`expected_slope` is what pure beta-estimation noise would produce on its own.
usage: python scripts/11_gate.py [window=2012-2014]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.boost import boost_vector, score_panel  # noqa: E402
from eracoef.checks import calibration_slopes  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.cv import make_exposure  # noqa: E402
from eracoef.windows import build_window, fit_pooled_boost  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
LAB = sys.argv[1] if len(sys.argv) > 1 else "2012-2014"
seasons = [int(LAB[:4]) + i for i in range(3)]
BUCKETS = tuple(cfg["cv"]["poss_buckets"][:-1]) + (np.inf,)
OUT = Path(cfg["_root"]) / "outputs"

t0 = time.time()
panel = pd.read_parquet(Path(cfg["_root"]) / "data" / "cache" / "boost_panel.parquet")
res = fit_pooled_boost(panel, cfg, group_by="player", hold_out=LAB)
wd = build_window(seasons, cfg)
exp = make_exposure(wd, mode="full", pad_target=cfg["pad_target"]).fit(wd.X, sample_weight=wd.w)
g = boost_vector(res, score_panel(exp, wd.spec, window=LAB), window=LAB)
print(f"booster fit on the other nine windows; sd(g) = {g.std():.3f}  ({time.time() - t0:.0f}s)\n", flush=True)

out = []
for kind, off in (("linear", None), ("boosted", g)):
    t = calibration_slopes(wd, lam=float(cfg["lam_plugin"]), lam_ratio=float(cfg["lam_ratio_plugin"]),
                           buckets=BUCKETS, mode="crossfit", lam_half=float(cfg["lam_beta"]),
                           pad_target=cfg["pad_target"], prior_offset=off, controls=True)
    t["prior_kind"] = kind
    out.append(t)
    print(f"  {kind} done ({time.time() - t0:.0f}s)", flush=True)
t = pd.concat(out, ignore_index=True)
t.to_csv(OUT / "gate_calibration.csv", index=False)

print(f"\n=== {LAB}: calibration slopes, headcount-controlled (1.0 = calibrated) ===")
for kind in ("prior", "eblup"):
    p = t[t.kind == kind].pivot_table(index="bucket", columns=["prior_kind", "side"], values="slope")
    print(f"\n{kind}:")
    print(p.round(3).to_string())
e = t[(t.kind == "prior") & (t.bucket == "all")][["prior_kind", "side", "slope", "se", "expected_slope"]]
print("\nprior, pooled over buckets, against what beta-estimation noise alone predicts:")
print(e.round(3).to_string(index=False))
print(f"\nwrote outputs/gate_calibration.csv  ({time.time() - t0:.0f}s)")
