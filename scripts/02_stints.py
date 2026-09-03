"""Build stints per season and phase from cached play-by-play; print per-season diagnostics.
usage: python scripts/02_stints.py [first_season] [last_season] [phases=RS,PO] [--force]
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.stints import build_season, season_diagnostics  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
args = [a for a in sys.argv[1:] if not a.startswith("--")]
force = "--force" in sys.argv
first = int(args[0]) if args else cfg["first_season"]
last = int(args[1]) if len(args) > 1 else first
phases = tuple(args[2].split(",")) if len(args) > 2 else ("RS", "PO")
rows = []
for season in range(first, last + 1):
    for phase in phases:
        t0 = time.time()
        build_season(season, phase, cfg, force=force)
        d = season_diagnostics(season, phase, cfg)
        d["wall_seconds"] = time.time() - t0
        rows.append(d)
        print(pd.Series(d).to_string())
        print()
out = pd.DataFrame(rows)
out_path = Path(cfg["_root"]) / cfg["paths"]["stints"] / "diagnostics_summary.parquet"
if out_path.exists():
    old = pd.read_parquet(out_path)
    old = old[~(old.season.astype(str) + old.phase).isin(out.season.astype(str) + out.phase)]
    out = pd.concat([old, out], ignore_index=True)
out.to_parquet(out_path, index=False)
print(out.to_string(index=False))
