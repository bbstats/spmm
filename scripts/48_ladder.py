"""The ladder's verdicts: every system against the reference, both K, both levels, with the stop rules.

Reads the holdout result files (outputs/holdout_<tag>.parquet, from scripts/45_holdout.py) given
on the command line, merges them on the stable schema, and for each system prints the paired test
against the reference at stint and team-game level for every K, then the verdict:

    wins   z <= -2 and at least 60% of the held-out seasons won, at BOTH K (per level)
    loses  z >= +2 at both K
    flat   anything else

usage: python scripts/48_ladder.py outputs/holdout_ladder_k2.parquet outputs/holdout_ladder_k4.parquet [--ref=hybrid_xft]
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.holdout import paired, pooled  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 4)
files = [a for a in sys.argv[1:] if not a.startswith("--")]
hit = [a for a in sys.argv[1:] if a.startswith("--ref=")]
REF = hit[0].split("=")[1] if hit else "hybrid_xft"
res = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
res = res[res.split == "all"]
ks = sorted(res.k.unique())

print(f"=== pooled over {res.held_out.nunique()} held-out seasons, systems {sorted(res.system.unique())}")
P = pooled(res)
print(P.pivot_table(index="system", columns="k", values=["mse", "game"]).round(3).to_string())

rows = []
for value, level in (("mse", "stint"), ("tg", "game")):
    t = paired(res, REF, value)
    for s, d in t.groupby("system"):
        d = d.set_index("k")
        zs = d.z.reindex(ks)
        wins = (d.wins / d.n_seasons).reindex(ks)
        if (zs <= -2).all() and (wins >= 0.6).all():
            verdict = "WINS"
        elif (zs >= 2).all():
            verdict = "loses"
        else:
            verdict = "flat"
        rows.append(dict(system=s, level=level, verdict=verdict,
                         **{f"diff_k{k}": float(d.mean_diff.get(k, float("nan"))) for k in ks},
                         **{f"z_k{k}": float(zs.get(k, float("nan"))) for k in ks},
                         **{f"wins_k{k}": f"{int(d.wins.get(k, 0))}/{int(d.n_seasons.get(k, 0))}" for k in ks}))
V = pd.DataFrame(rows)
print(f"\n=== paired against {REF}; negative = better; verdict needs both K")
for level in ("stint", "game"):
    print(f"\n-- {level} level")
    print(V[V.level == level].drop(columns="level").to_string(index=False))
V.to_csv(Path(files[0]).parent / "csv" / f"ladder_vs_{REF}.csv", index=False)
