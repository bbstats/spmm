"""Feature selection for the boosted box prior: BorutaShap on chimeraboost, once per side and mode.

On the pooled training rows of outputs/role_panel.parquet (every window; the target is each
player's value over his OTHER windows), with chimeraboost's exact SHAP values as the importance
(src/eracoef/gbdt_prior.py: make_boruta).  Two modes, matching GBDTPrior:
    residual   target u (RAPM_1 beyond the role prior), candidates = 13 rates + season
    full       target rapm1, candidates = 13 rates + season + share, gs_pct, age
Prints accepted / tentative / rejected, writes outputs/csv/boruta_{mode}_{side}.csv (the importance
history) and the YAML lines for config.yaml -> gbdt.features_* / features_full_*.  `season` is kept
whether or not Boruta accepts it: it is the era term the prior exists for; the verdict is recorded.

usage: python scripts/50_boruta.py [--trials=50] [--sides=O,D] [--modes=residual,full] [--threads=12]
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.gbdt_prior import DEFAULT_FEATURES, FULL_FEATURES, run_boruta, training_rows  # noqa: E402

cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
G = cfg.get("gbdt", {})


def _flag(name, default):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


trials = int(_flag("trials", G.get("boruta_trials", 50)))
sides = _flag("sides", "O,D").split(",")
modes = _flag("modes", "residual,full").split(",")
threads = int(_flag("threads", 12))
panel = pd.read_parquet(Path(cfg["_root"]) / cfg.get("paths", {}).get("role_panel", "outputs/role_panel.parquet"))
MODES = {"residual": ("u", DEFAULT_FEATURES, "features_{}"), "full": ("rapm1", FULL_FEATURES, "features_full_{}")}

t0 = time.time()
yaml_lines = []
for mode in modes:
    target, feats, key = MODES[mode]
    for side in sides:
        rows = training_rows(panel, side, exclude=(), features=feats, target_col=target)
        print(f"\n=== mode {mode} (target {target}), side {side}: {len(rows)} training rows "
              f"(players seen in 2+ windows), {trials} trials", flush=True)
        res = run_boruta(rows, feats, n_trials=trials, seed=int(G.get("seed", 0)), thread_count=threads,
                         verbose=False, **dict(G.get("params", {}) or {}))
        print(f"  accepted : {res['accepted']}")
        print(f"  tentative: {res['tentative']}")
        print(f"  rejected : {res['rejected']}   ({time.time() - t0:.0f}s)")
        if res["history"] is not None:
            res["history"].to_csv(OUT / "csv" / f"boruta_{mode}_{side}.csv", index=False)
            print("  mean importance over trials (z-scored; the shadow max is the bar):")
            print(res["history"].mean().round(3).sort_values(ascending=False).to_string())
        keep = sorted(set(res["accepted"]) | {"season"})
        yaml_lines.append(f"  {key.format(side)}: [{', '.join(keep)}]"
                          + ("" if "season" in res["accepted"] else "   # season kept by design")
                          + (f"   # tentative: {', '.join(res['tentative'])}" if res["tentative"] else ""))

print("\n=== config.yaml -> gbdt:")
print("\n".join(yaml_lines))
