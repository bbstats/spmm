"""Compare our 2024-26 ratings against a published blend of the modern all-in-one metrics.

The previous round of validation scored our ratings against our OWN next-window on-court RAPM.
That benchmark is built from the same margin data with the same blind spots, so it could never
catch a shared error, and it did not.  This script uses an external benchmark instead: a public
consensus of the modern all-in-one stats, in the same units we use (points per 100 possessions),
split the same way (offense, defense, overall).

Names are matched on a normalised form with no player ids on either side, so a few will miss.  The
point is the deltas, not a perfect join.

usage: python scripts/22_vs_consensus.py [--refresh]
"""
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.boxtable import season_box  # noqa: E402
from eracoef.config import load_config  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 2)
cfg = load_config()
ROOT = Path(cfg["_root"])
OUT = ROOT / "outputs"
CACHE = ROOT / "data" / "external" / "consensus.csv"
URL = ("https://docs.google.com/spreadsheets/d/e/2PACX-1vSHAONgEY5Dxun8Wegvtr4N_zWmRbdQl2UBq8vbugL"
       "xHCYkyQmo8CJFDcDvjaXqHc_ve4xArAEkNQNF/pub?gid=0&single=true&output=csv")
WINDOW = "2024-2026"
SEASONS = [2024, 2025, 2026]

SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm(s: str) -> str:
    """Accent-free, punctuation-free, suffix-free lower case, for joining without ids."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("&", " ").replace("-", " ").replace("'", "").replace(".", " ")
    s = SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


if "--refresh" in sys.argv or not CACHE.exists():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(urllib.request.urlopen(URL, timeout=60).read().decode("utf-8"), encoding="utf-8")

con = pd.read_csv(CACHE)[["player_name", "team", "adj_offense", "adj_defense", "adj_overall"]].dropna(
    subset=["player_name", "adj_overall"])
con["key"] = con.player_name.map(norm)
con = con.drop_duplicates("key")

rat = pd.read_parquet(OUT / "player_ratings.parquet")
ours = rat[(rat.window == WINDOW) & rat.player_name.notna()].copy()
ours["key"] = ours.player_name.map(norm)
ours = ours.sort_values("poss_off").drop_duplicates("key", keep="last")

m = ours.merge(con, on="key", how="inner", suffixes=("", "_con"))
print(f"consensus rows {len(con)}, our {WINDOW} rows {len(ours)}, matched {len(m)}")
unmatched = con[~con.key.isin(ours.key)].nlargest(10, "adj_overall")
if len(unmatched):
    print("\ntop consensus players we did not match (name join, no ids):")
    print(unmatched[["player_name", "team", "adj_overall"]].to_string(index=False))

# The consensus covers rotation players; our table goes much deeper.  Compare on the overlap, and
# keep a possession floor so we are not judging our own noise.
m = m[m.poss_off >= 1000].copy()
print(f"\ncompared on {len(m)} players with 1000+ possessions in {WINDOW}")

PAIRS = [("total", "rating_total", "adj_overall"), ("off", "rating_off", "adj_offense"),
         ("def", "rating_def", "adj_defense")]

# Put the two on the same footing before anything is compared.  `d_*` is the gap after both are
# standardised over this same pool, and `dr_*` is the gap in rank, which is what actually matters
# for a leaderboard and needs no scaling assumption at all.
for name, ours_c, con_c in PAIRS:
    m[f"z_ours_{name}"] = (m[ours_c] - m[ours_c].mean()) / m[ours_c].std()
    m[f"z_con_{name}"] = (m[con_c] - m[con_c].mean()) / m[con_c].std()
    m[f"d_{name}"] = m[f"z_ours_{name}"] - m[f"z_con_{name}"]
    m[f"rk_ours_{name}"] = m[ours_c].rank(ascending=False)
    m[f"rk_con_{name}"] = m[con_c].rank(ascending=False)
    m[f"dr_{name}"] = m[f"rk_con_{name}"] - m[f"rk_ours_{name}"]   # + = we rank him higher

print("\n=== 1. agreement with the consensus")
print("   spread ratio is ours over theirs AFTER nothing but a common pool: above 1 means we")
print("   spray players further apart than every modern metric combined does.")
for name, ours_c, con_c in PAIRS:
    sp = m[[ours_c, con_c]].corr(method="spearman").iloc[0, 1]
    ratio = m[ours_c].std() / m[con_c].std()
    flag = "  <-- far too wide" if ratio > 1.5 else ""
    print(f"  {name:6s} spearman {sp:.3f}   our sd {m[ours_c].std():.2f} vs theirs "
          f"{m[con_c].std():.2f}   spread ratio {ratio:.2f}{flag}")
print(f"\n   mean absolute rank gap: total {m.dr_total.abs().mean():.0f} places, "
      f"offense {m.dr_off.abs().mean():.0f}, defense {m.dr_def.abs().mean():.0f} "
      f"(out of {len(m)} players)")

# ------------------------------------------------------------------ 2. the named anchor
print("\n=== 2. the anchors")
for who in ("robert williams", "mitchell robinson", "dayron sharpe", "neemias queta",
            "moussa diabate", "luke kornet", "jusuf nurkic", "jonathan isaac", "rudy gobert",
            "nikola jokic", "shai gilgeous alexander", "lamelo ball", "devin booker",
            "trae young", "stephen curry"):
    g = m[m.key == who]
    if not len(g):
        print(f"  {who:26s} not matched")
        continue
    r = g.iloc[0]
    print(f"  {r.player_name:26s} ours #{int(r.rk_ours_total):3d} ({r.rating_total:+6.2f})   "
          f"consensus #{int(r.rk_con_total):3d} ({r.adj_overall:+5.1f})   "
          f"gap {int(r.rk_ours_total - r.rk_con_total):+4d} places")

# ------------------------------------------------------------------ 3. the top of each board
print("\n=== 3. top 20, side by side")
a = m.nsmallest(20, "rk_ours_total")[["player_name", "rating_total", "adj_overall", "rk_con_total"]]
a.columns = ["ours: player", "our rating", "consensus", "their rank"]
print(a.to_string(index=False))
print("")
b = m.nsmallest(20, "rk_con_total")[["player_name", "adj_overall", "rating_total", "rk_ours_total"]]
b.columns = ["consensus: player", "consensus", "our rating", "our rank"]
print(b.to_string(index=False))

# ------------------------------------------------------------------ 4. who do we most disagree on?
print("\n=== 4. the biggest disagreements, in rank places")
show = ["player_name", "poss_off", "prior_total", "boost_total", "u_total",
        "rk_ours_total", "rk_con_total", "dr_total", "rk_ours_def", "rk_con_def"]
hdr = ["player", "poss", "prior", "boost", "u", "our rk", "their rk", "gap", "our D rk", "their D rk"]
print("  we rank far higher than the consensus does:")
t = m.nlargest(15, "dr_total")[show]
t.columns = hdr
print(t.to_string(index=False))
print("\n  the consensus ranks far higher than we do:")
t = m.nsmallest(15, "dr_total")[show]
t.columns = hdr
print(t.to_string(index=False))

# ------------------------------------------------------------------ 5. what predicts the gap?
box = season_box(SEASONS, ["RS"], cfg)
g = box[box.phase == "RS"].groupby("player_id", as_index=False)[
    ["minutes", "orb", "drb", "blk", "ast", "fg3m", "fg3_miss", "fg2m", "fg2_miss", "ftm", "tov", "stl", "pf"]].sum()
for c in g.columns.drop(["player_id", "minutes"]):
    g[c] = g[c] / g.minutes.clip(lower=1) * 36
g["bigness"] = g.orb + g.blk + 0.3 * g.drb - 0.5 * g.ast - 0.4 * g.fg3m
m = m.merge(g[["player_id", "bigness", "minutes", "orb", "blk", "drb", "ast", "fg3m"]],
            on="player_id", how="left")
print("\n=== 5. what predicts the disagreement?")
print("   correlation of the gap (ours minus consensus, standard deviations) with:")
for name in ("total", "off", "def"):
    r = {v: m[f"d_{name}"].corr(m[v]) for v in ("bigness", "poss_off", "minutes", "orb", "blk", "ast", "fg3m")}
    print(f"  {name:6s} " + "  ".join(f"{k} {v:+.3f}" for k, v in r.items()))

qs = m.bigness.quantile([1 / 3, 2 / 3]).to_numpy()
m["archetype"] = np.where(m.bigness >= qs[1], "big", np.where(m.bigness >= qs[0], "wing", "guard"))
print("\n   mean gap by archetype (positive = we rate them above the consensus):")
print(m.groupby("archetype")[["d_total", "d_off", "d_def"]].agg(["mean", "count"])
      .reindex(["guard", "wing", "big"]).round(3).to_string())

print("\n   mean gap by minutes played (the consensus is a rotation-player list):")
m["mins_bucket"] = pd.cut(m.minutes, [0, 1000, 2000, 3000, 10000],
                          labels=["<1000", "1000-2000", "2000-3000", "3000+"])
print(m.groupby("mins_bucket", observed=True)[["d_total", "d_off", "d_def"]]
      .agg(["mean", "count"]).round(3).to_string())

# ------------------------------------------------------------------ 6. which piece is to blame?
print("\n=== 6. which piece of our rating carries the disagreement?")
print("   correlation of each piece with the consensus, and with the gap:")
for name, _, con_c in PAIRS:
    parts = [("prior", f"prior_{name}"), ("boost", f"boost_{name}"), ("u", f"u_{name}")]
    row = []
    for lab, c in parts:
        if c in m.columns:
            row.append(f"{lab} vs consensus {m[c].corr(m[con_c]):+.3f}")
    print(f"  {name:6s} " + "   ".join(row))

# ------------------------------------------------------------------ 7. every public metric, one at a time
# The blend above is the owner's own mix (defense: 0.5 xDRAPM + 0.4 EPM + 0.1 LA-RAPM, all raw-points
# but the last).  As sanity checks, agree with each component separately, so a luck adjustment that
# moves the board toward LA-RAPM and LEBRON and away from xRAPM and EPM reads as what it is.
IM = ROOT / "data" / "external" / "impact_metrics_2526.csv"
if IM.exists():
    from scipy.stats import spearmanr
    im = pd.read_csv(IM)
    name_col = [c for c in im.columns if "name" in c.lower()][0]
    im["key"] = im[name_col].map(norm)
    j = m.merge(im, on="key", how="inner", suffixes=("", "_im"))
    comps = {"off": ["pred_oepm", "xORAPM", "td_laorapm", "td_orapm", "OLEBRON", "ODRIP", "ODPM", "LA_ORAPM"],
             "def": ["pred_depm", "xDRAPM", "td_ladrapm", "td_drapm", "DLEBRON", "DDRIP", "DDPM", "LA_DRAPM", "xDLEBRON"]}
    print(f"\n=== 7. against each public metric separately (Spearman, {len(j)} players)")
    for side, cols in comps.items():
        ours = j[f"rating_{side}"]
        row = [f"{c} {spearmanr(ours, j[c].astype(float), nan_policy='omit').statistic:+.3f}" for c in cols if c in j.columns]
        print(f"  {side:4s} " + "  ".join(row))

m.to_parquet(OUT / "vs_consensus.parquet", index=False)
m.round(3).to_csv(OUT / "csv" / "vs_consensus.csv", index=False)
print("\nwrote outputs/vs_consensus.parquet and outputs/csv/vs_consensus.csv")
