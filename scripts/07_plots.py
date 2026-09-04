"""Step 8: one page for GitHub Pages -- the ratings leaderboard, the coefficient figure, the PNGs.

Reads outputs/coefs.parquet and outputs/ratings_boosted.parquet (written by scripts/10_boost.py).
Writes docs/index.html and docs/img/*.png, and mirrors both into outputs/figs.
usage: python scripts/07_plots.py
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef import plots, site  # noqa: E402
from eracoef.config import load_config  # noqa: E402

cfg = load_config()
root = Path(cfg["_root"])
out = root / cfg["paths"]["outputs"]
docs, img = root / "docs", root / "docs" / "img"
figs_dir = root / cfg["paths"]["figs"]

data = plots.load(cfg)
print("loaded:", {k: len(v) for k, v in data.items()})

print("metric pngs:", len(plots.metric_pngs(data, img)))
print("playoffs:", plots.playoff_delta_png(data, img / "playoff_delta.png"))
print("correlations:", plots.correlation_pngs(data, img))
print("movers:", plots.movers_png(data, img / "movers.png"))

FIGS = [("movers.png", "Dot-and-whisker of the ten largest coefficient changes",
         "The ten coefficients that moved most from 1997-99 to 2024-26, with 95% intervals on the change."),
        ("playoff_delta.png", "Playoff minus regular-season coefficients by era third",
         "The playoff block: how much each coefficient differs in the playoffs, by era third."),
        ("correlations.png", "Correlation heatmaps of the 13 per-100 rates",
         "Who accumulates what, together. A coefficient can move because the value changed or "
         "because the players did.")]

# the canonical table, written by scripts/08_ratings.py.  ratings_boosted.parquet is an
# intermediate that holds only the correction, and its residual is not the one the table ships.
rat = pd.read_parquet(out / "player_ratings.parquet")
p = site.build(cfg, rat, data, FIGS, docs / "index.html")
print("page:", p, f"{p.stat().st_size / 1024:.0f} KB")

stale = docs / "coefs.html"
if stale.exists():                      # the figure now lives inside index.html
    stale.unlink()
    print("removed the old second page", stale)

figs_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(docs / "index.html", figs_dir / "index.html")
for f in img.glob("*.png"):
    shutil.copy(f, figs_dir / f.name)
print("mirrored to", figs_dir)
