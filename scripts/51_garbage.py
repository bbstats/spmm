"""Where the multi-stage chain loses: deep-bench players, garbage time, or both?

For each held-out season (K=4) and each system, predict the season and regress the held-out stints on
the lineup contributions split by the ROLE of the players on the floor -- deep bench (under
`deep_share` of the team's possessions in that season), bench (starts share under 0.5), starters --
per side, plus the level columns.  Each slope is the multiplier the season wants on that group's
ratings (1 = calibrated, below 1 = that group's ratings are too spread, above 1 = too timid).  The same
regression is run on the garbage-time rows alone and on the competitive rows alone.  Slopes are pooled
over seasons by inverse variance.  Also prints each group's share of the rating mass on the floor.

usage: python scripts/51_garbage.py [first] [last] [--systems=def3_p0,mspi,spm] [--k=4] [--deep=0.1]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.holdout import Context, Holdout, _wls, predict_season  # noqa: E402
from eracoef.systems import registry  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"


def _flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


GROUPS = ["deep", "bench", "starter"]


def role_flags(ctx: Context, h: int, ids, deep_share: float) -> dict:
    r = ctx.role_inputs[(ctx.role_inputs.season == h)].set_index("player_id")
    share = r["share"].reindex(ids).fillna(0.0).to_numpy()
    gs = r["gs_pct"].reindex(ids).fillna(0.0).to_numpy()
    deep = share < deep_share
    bench = (~deep) & (gs < float(cfg.get("holdout", {}).get("bench_gs_pct", 0.5)))
    starter = ~deep & ~bench
    return {"deep": deep.astype(float), "bench": bench.astype(float), "starter": starter.astype(float)}


def group_slopes(p, wd_h, flags: dict, mask=None) -> pd.DataFrame:
    """WLS of y on the level columns + the six group contributions; returns slope and se per (side, group)."""
    m = wd_h.spec.n_ps
    o, d = p.rat.o.to_numpy(dtype=float), p.rat.d.to_numpy(dtype=float)
    Zo, Zd = wd_h.X[:, :m], wd_h.X[:, m:2 * m]
    cols, names = [], []
    for g in GROUPS:
        cols.append(np.asarray(Zo @ (o * flags[g])).ravel())
        names.append(("o", g))
    for g in GROUPS:
        cols.append(np.asarray(Zd @ (d * flags[g])).ravel())
        names.append(("d", g))
    A = np.column_stack([p.A, *cols])
    idx = np.arange(len(p.y)) if mask is None else np.flatnonzero(mask)
    A, y, w = A[idx], p.y[idx], p.w[idx]
    AtW = (A * w[:, None]).T
    G = AtW @ A
    coef = np.linalg.lstsq(G, AtW @ y, rcond=None)[0]
    e = y - A @ coef
    sigma2 = float(np.average(e ** 2, weights=w))
    cov = sigma2 * np.linalg.pinv(G) * (w.sum() / len(w))
    k = p.A.shape[1]
    rows = []
    for j, (side, g) in enumerate(names):
        rows.append(dict(side=side, group=g, slope=float(coef[k + j]), se=float(np.sqrt(max(cov[k + j, k + j], 0.0))),
                         mass=float(np.average(np.abs(cols[j][idx]), weights=w))))
    return pd.DataFrame(rows)


def main():
    names = (_flag("systems", "def3_p0,mspi,spm")).split(",")
    k = int(_flag("k", 4))
    deep_share = float(_flag("deep", 0.1))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ctx = Context.load(cfg)
    reg = registry(cfg)
    systems = [reg[n] for n in names]
    ho = Holdout.from_config(cfg, first=int(args[0]) if args else None, last=int(args[1]) if len(args) > 1 else None)
    t0 = time.time()
    rows = []
    for h in ho.seasons():
        ctx.current_h = h
        ctx.current_k = k
        wd_h = ctx.design([h], "pts")
        train = ctx.neighbourhood(h, k)
        ids = wd_h.spec.ps_table["player_id"].to_numpy()
        flags = role_flags(ctx, h, ids, deep_share)
        gt = wd_h.rows["is_gt"].to_numpy().astype(bool)
        for s in systems:
            rat = s.fit(train, ctx)
            p = predict_season(rat, wd_h, level=ho.level)
            for subset, mask in (("all", None), ("garbage time", gt), ("competitive", ~gt)):
                t = group_slopes(p, wd_h, flags, mask)
                rows.append(t.assign(held_out=h, system=s.name, subset=subset))
        print(f"  {h} done ({time.time() - t0:.0f}s)", flush=True)
    ctx.current_h = None
    R = pd.concat(rows, ignore_index=True)
    R.to_parquet(OUT / "garbage_slopes.parquet", index=False)

    def pool(d):
        iv = 1.0 / np.maximum(d.se.to_numpy() ** 2, 1e-12)
        slope = float((d.slope.to_numpy() * iv).sum() / iv.sum())
        se = float(np.sqrt(1.0 / iv.sum()))
        return pd.Series({"slope": slope, "se": se, "z": (slope - 1.0) / se, "mass": float(d.mass.mean()), "seasons": len(d)})

    P = R.groupby(["subset", "system", "side", "group"]).apply(pool, include_groups=False).reset_index()
    P.to_csv(OUT / "csv" / "garbage_slopes.csv", index=False)
    print(f"\n=== what the held-out seasons want each role group's ratings multiplied by (K={k}, deep = share < {deep_share})")
    print("    1 = calibrated; below 1 = too spread; above 1 = too timid.  mass = mean |contribution| per row, points per 100")
    for subset in ("all", "garbage time", "competitive"):
        print(f"\n-- rows: {subset}")
        d = P[P.subset == subset]
        print(d.pivot_table(index=["side", "group"], columns="system", values=["slope", "z", "mass"]).round(3).to_string())
    print(f"\nwrote outputs/garbage_slopes.parquet and outputs/csv/garbage_slopes.csv  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
