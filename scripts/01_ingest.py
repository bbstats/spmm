"""Pull game logs, play-by-play (V3) and box scores for seasons; cached, idempotent.
usage: python scripts/01_ingest.py [first_season] [last_season] [phases=RS,PO] [workers=2]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config
from eracoef.ingest import ingest_season

cfg = load_config()
first = int(sys.argv[1]) if len(sys.argv) > 1 else cfg["first_season"]
last = int(sys.argv[2]) if len(sys.argv) > 2 else first
phases = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("RS", "PO")
workers = int(sys.argv[4]) if len(sys.argv) > 4 else 2
delay = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6
for season in range(first, last + 1):
    ingest_season(season, phases, cfg, workers=workers, delay=delay)
