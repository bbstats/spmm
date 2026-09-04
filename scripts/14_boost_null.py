"""Null check: with no signal to find, does the booster's self-calibrating shrinkage go to zero?

The target is permuted WITHIN deciles of the shrinkage weight `a`, so the relationship between a
target and its own variance survives while the link between a target and the box-score features is
destroyed.  Permuting across all rows instead would be wrong twice over: it puts a low-possession
player's enormous de-shrunk residual on a starter's weight, and if playing time is permuted along
with the target then playing time, which is itself a feature, keeps its real association and the
booster legitimately finds it.

usage: python scripts/14_boost_null.py [n_reps=5]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.windows import fit_pooled_boost  # noqa: E402

cfg = load_config()
N_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
panel = pd.read_parquet(Path(cfg["_root"]) / "data" / "cache" / "boost_panel.parquet")

print("real target:")
real = fit_pooled_boost(panel, cfg, group_by="player")

rows = []
for rep in range(N_REPS):
    rng = np.random.default_rng(rep)
    q = panel.copy()
    for side in ("O", "D"):
        m = (q.side == side).to_numpy()
        sub = q.loc[m]
        strata = pd.qcut(sub["a"].rank(method="first"), 10, labels=False)
        t = sub["target"].to_numpy().copy()
        for s in range(10):                       # shuffle within a stratum of the weight
            i = np.flatnonzero(strata.to_numpy() == s)
            t[i] = t[i][rng.permutation(len(i))]
        q.loc[m, "target"] = t
    print(f"\nnull rep {rep}:")
    n = fit_pooled_boost(q, cfg, group_by="player")
    rows.append({"rep": rep, **{f"s_{k}": v for k, v in n["s"].items()}})

d = pd.DataFrame(rows)
print("\n=== self-calibrating shrinkage, real against the null")
print(f"  real      O {real['s']['O']:.3f}   D {real['s']['D']:.3f}")
print(f"  null mean O {d.s_O.mean():.3f}   D {d.s_D.mean():.3f}   "
      f"(max over {N_REPS} reps: O {d.s_O.max():.3f}, D {d.s_D.max():.3f})")
d.to_csv(Path(cfg["_root"]) / "outputs" / "boost_null.csv", index=False)
print("\nwrote outputs/boost_null.csv")
