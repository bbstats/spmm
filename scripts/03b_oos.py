"""Checkpoint 3, section 8 standalone: OOS zero-prior vs crossfit+plug-in on the same folds.
usage: python scripts/03b_oos.py lam_plug lam_ratio lam_zero [first=2012] [last=2014]
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.boxtable import season_box
from eracoef.checks import oos_compare
from eracoef.config import load_config
from eracoef.design import FEATURES, build_design
from eracoef.stints import build_season
cfg = load_config()
lam, ratio, lam_zero = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
first = int(sys.argv[4]) if len(sys.argv) > 4 else 2012
last = int(sys.argv[5]) if len(sys.argv) > 5 else 2014
seasons = list(range(first, last + 1))
stints = pd.concat([build_season(s, "RS", cfg)[0] for s in seasons], ignore_index=True)
wd = build_design(stints, season_box(seasons, ["RS"], cfg), FEATURES, cfg)
t0 = time.time()
oc = oos_compare(wd, lam=lam, lam_ratio=ratio, n_folds=cfg["cv"]["n_folds"], mode="crossfit", lam_zero=lam_zero)
print(oc.to_string(index=False))
d = oc["diff"]
print(f"mean diff {d.mean():.2f}  se {d.std(ddof=1) / np.sqrt(len(d)):.2f}  joint wins {int((d < 0).sum())}/{len(d)} folds  ({time.time() - t0:.0f}s)")
oc.to_csv(Path(cfg["_root"]) / "outputs" / "checkpoint3_oos.csv", index=False)
