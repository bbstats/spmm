"""Does the boosted prior beat the linear prior on held-out games, not just in fit?

Same five game-grouped folds, same beta, same lambda; the only difference is whether the per-player
boost rides along in the offset.  The booster is fit on the other nine windows, so no held-out game
reaches it.  Reported fold-paired, because the fold-to-fold spread of the level dwarfs the
difference we are testing.
usage: python scripts/13_boost_oos.py [window=2012-2014 [more windows...]]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.boost import boost_vector, score_panel  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.cv import crossfit_beta, make_exposure, plugin_fit  # noqa: E402
from eracoef.windows import build_window, fit_pooled_boost  # noqa: E402

pd.set_option("display.width", 250, "display.precision", 4)
cfg = load_config()
LABS = sys.argv[1:] or ["2012-2014", "2024-2026"]
LAM_B, RAT_B = float(cfg["lam_beta"]), float(cfg["lam_ratio_beta"])
LAM_P, RAT_P = float(cfg["lam_plugin"]), float(cfg["lam_ratio_plugin"])
OUT = Path(cfg["_root"]) / "outputs"

panel = pd.read_parquet(Path(cfg["_root"]) / "data" / "cache" / "boost_panel.parquet")
rows = []
t0 = time.time()
for LAB in LABS:
    seasons = [int(LAB[:4]) + i for i in range(3)]
    res = fit_pooled_boost(panel, cfg, group_by="player", hold_out=LAB, verbose=False)
    wd = build_window(seasons, cfg)
    for k, (tr, te) in enumerate(GroupKFold(5).split(wd.X, wd.y, wd.groups)):
        tr_mask = np.zeros(len(wd.y), bool)
        tr_mask[tr] = True
        cf = crossfit_beta(wd, lams=[LAM_B], cv=2, lam_ratio=RAT_B, mask=tr_mask, pad_target=cfg["pad_target"])
        lin = plugin_fit(wd, cf.beta_, lams=[LAM_P], cv=2, lam_ratio=RAT_P, mask=tr_mask,
                         pad_target=cfg["pad_target"])
        g = boost_vector(res, score_panel(lin["exposure"], wd.spec, window=LAB), window=LAB)
        bst = plugin_fit(wd, cf.beta_, lams=[LAM_P], cv=2, lam_ratio=RAT_P, mask=tr_mask,
                         pad_target=cfg["pad_target"], prior_offset=g)
        r = dict(window=LAB, fold=k, w=float(np.sum(wd.w[te])), sd_g=float(g.std()))
        for name, pipe in (("linear", lin), ("boosted", bst)):
            p = pipe.predict(wd.X[te])
            r[name] = float(np.average((wd.y[te] - p) ** 2, weights=wd.w[te]))
        rows.append(r)
        print(f"  {LAB} fold {k}: linear {r['linear']:.3f}  boosted {r['boosted']:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

d = pd.DataFrame(rows)
d["diff"] = d.linear - d.boosted            # positive = the boost helps
d.to_csv(OUT / "boost_oos.csv", index=False)
print("\n=== held-out weighted MSE, fold-paired (positive difference = the boost helps)")
for LAB, g in d.groupby("window"):
    m, se = g["diff"].mean(), g["diff"].std(ddof=1) / np.sqrt(len(g))
    print(f"  {LAB}: linear {g.linear.mean():9.3f}   boosted {g.boosted.mean():9.3f}   "
          f"diff {m:+.3f} +- {se:.3f}   ({m / se if se > 0 else 0:+.1f} SE)")
m, se = d["diff"].mean(), d["diff"].std(ddof=1) / np.sqrt(len(d))
print(f"  pooled: diff {m:+.3f} +- {se:.3f}  ({m / se if se > 0 else 0:+.1f} SE)")
print(f"\nwrote outputs/boost_oos.csv  ({time.time() - t0:.0f}s)")
