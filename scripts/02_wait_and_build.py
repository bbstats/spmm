"""Wait until every game of the given seasons is cached, then build stints and print diagnostics.
usage: python scripts/02_wait_and_build.py 2012 2013 [phases=RS,PO]
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.ingest import _box_path, _pbp_path, fetch_boxscore, fetch_pbp, game_table, load_gamelog  # noqa: E402
from eracoef.stints import build_season, season_diagnostics  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
args = [a for a in sys.argv[1:] if not a.startswith("--")]
seasons = [int(a) for a in args if a.isdigit()]
phases = tuple(next((a for a in args if not a.isdigit()), "RS,PO").split(","))
rows = []
for season in seasons:
    for phase in phases:
        games = game_table(load_gamelog(season, phase, cfg))
        attempts = {}
        while True:
            missing = [g for g in games["game_id"] if not (_pbp_path(g, cfg).exists() and _box_path(g, cfg).exists())]
            if not missing:
                break
            print(f"{season} {phase}: waiting for {len(missing)} games", flush=True)
            # stragglers the bulk pull gave up on (transient timeouts): fetch them here, a few tries each
            if len(missing) <= 25:
                for g in missing:
                    if attempts.get(g, 0) >= 3:
                        continue
                    attempts[g] = attempts.get(g, 0) + 1
                    try:
                        fetch_pbp(g, cfg)
                        fetch_boxscore(g, cfg)
                        print(f"  fetched straggler {g}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"  straggler {g} failed (try {attempts[g]}): {repr(e)[:120]}", flush=True)
                if all(attempts.get(g, 0) >= 3 for g in missing):
                    print(f"{season} {phase}: giving up on {missing}; building without them", flush=True)
                    break
                continue
            time.sleep(120)
        t0 = time.time()
        build_season(season, phase, cfg)
        d = season_diagnostics(season, phase, cfg)
        d["wall_seconds"] = time.time() - t0
        rows.append(d)
        print(pd.Series(d).to_string(), flush=True)
out = pd.DataFrame(rows)
p = Path(cfg["_root"]) / cfg["paths"]["stints"] / "diagnostics_summary.parquet"
if p.exists():
    old = pd.read_parquet(p)
    old = old[~(old.season.astype(str) + old.phase).isin(out.season.astype(str) + out.phase)]
    out = pd.concat([old, out], ignore_index=True)
out.to_parquet(p, index=False)
print(out.sort_values(["season", "phase"]).to_string(index=False))
