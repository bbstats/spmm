"""OpenRAPM: the one-page site.  Exports the board to docs/data/ratings.json for docs/index.html.

Reads outputs/player_ratings.parquet (scripts/08_ratings.py).  One row per player per three-season
window: window, name, offense, defense, total (positive = good, points per 100 possessions), and the
regular-season possessions the rating rests on.  docs/index.html is static and reads this file.
usage: python scripts/52_site.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402

cfg = load_config()
root = Path(cfg["_root"])
rat = pd.read_parquet(root / "outputs" / "player_ratings.parquet")
cols = {"window": "w", "player_name": "n", "rating_off": "o", "rating_def": "d", "rating_total": "t", "poss_off": "p"}
d = rat[list(cols)].rename(columns=cols).copy()
d["n"] = d["n"].fillna("").astype(str)
d = d[d.p > 0].sort_values(["w", "t"], ascending=[True, False])
rows = [dict(w=r.w, n=r.n, o=round(float(r.o), 2), d=round(float(r.d), 2), t=round(float(r.t), 2), p=int(r.p))
        for r in d.itertuples(index=False)]
out = root / "docs" / "data"
out.mkdir(parents=True, exist_ok=True)
meta = dict(windows=sorted(d.w.unique().tolist()), built=pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
            n_players=int(d.player_id.nunique()) if "player_id" in d.columns else int(rat.player_id.nunique()))
(out / "ratings.json").write_text(json.dumps(dict(meta=meta, rows=rows), separators=(",", ":")), encoding="utf-8")
print(f"wrote docs/data/ratings.json: {len(rows)} rows, {len(meta['windows'])} windows")
