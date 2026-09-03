"""Step 8: the interactive figure and the static PNGs, written into docs/ for GitHub Pages.
usage: python scripts/07_plots.py
"""
import sys, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config
from eracoef import plots

cfg = load_config()
root = Path(cfg["_root"])
docs, img = root / "docs", root / "docs" / "img"
figs = root / cfg["paths"]["figs"]
data = plots.load(cfg)
print("loaded:", {k: len(v) for k, v in data.items()})

p = plots.interactive_coefs(data, docs / "coefs.html", cfg)
print("interactive:", p, f"{p.stat().st_size/1024:.0f} KB")
print("metric pngs:", len(plots.metric_pngs(data, img)))
print("playoffs:", plots.playoff_delta_png(data, img / "playoff_delta.png"))
print("correlations:", plots.correlation_pngs(data, img))
print("movers:", plots.movers_png(data, img / "movers.png"))

figs.mkdir(parents=True, exist_ok=True)
shutil.copy(docs / "coefs.html", figs / "coefs.html")
for f in img.glob("*.png"):
    shutil.copy(f, figs / f.name)
print("mirrored to", figs)

figs = [("movers.png", "Dot-and-whisker of the ten largest coefficient changes",
         "The ten coefficients that moved most from 1997-99 to 2024-26, with 95% intervals on the change."),
        ("playoff_delta.png", "Playoff minus regular-season coefficients by era third",
         "The playoff block: how much each coefficient differs in the playoffs, by era third."),
        ("correlations.png", "Correlation heatmaps of the 13 per-100 rates",
         "Who accumulates what, together. A coefficient can move because the value changed or because the players did.")]
from eracoef.plots import index_page
print("index:", index_page(docs / "index.html", figs))
