"""The out-of-season criterion, from the command line.  One entry point for every comparison.

Hold out each season, train on the K nearest seasons around it, predict its stints, score against
actual points at stint and team-game level.  src/eracoef/holdout.py is the system; this registers
the systems this project has built and prints the report.

    rapm            no box prior
    pi              the team-priced prior (prior-informed RAPM)
    hybrid          the shipped prior: player-priced offense, no defensive prior
    hybrid_xft      hybrid on the free-throw-adjusted target (the chosen system)
    hybrid_c<c>     hybrid offense with the team-priced defensive prior scaled by c (0.25, 0.5, 0.75, 1)
    hybrid_xshoot*  the shooter-level expected-points targets (when src/eracoef/xshoot.py is built)

Every earlier held-out script is one invocation of this:
    38_yoy          --systems=rapm,pi,hybrid,hybrid_xft
    39_why          --systems=pi,hybrid --splits=exposure
    40_movers       --systems=pi,hybrid --splits=movers
    41/42_defweight --systems=hybrid,hybrid_c0.25,hybrid_c0.5,hybrid_c0.75,hybrid_c1 --consensus
    44_yoy_shot     --systems=hybrid,hybrid_xft,hybrid_xshoot,...

usage: python scripts/45_holdout.py [first] [last] --systems=a,b,c [--k=2,4] [--lams=18352]
           [--splits=movers,exposure] [--rank] [--consensus] [--tag=name] [--ref=hybrid_xft] [--quiet]
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.holdout import (SPLITS, Context, Holdout, PluginSystem, SplitSystem, beta_hybrid, beta_mixed,  # noqa: E402
                             beta_none, beta_team, pooled_rank, report, vs_consensus)

cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"


def _flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


def _list(name, default, conv=str):
    v = _flag(name)
    return default if v is None else [conv(x) for x in v.split(",") if x]


# ------------------------------------------------------------------ the registry
SYSTEMS = {
    "rapm": PluginSystem("rapm", beta=beta_none),
    "pi": PluginSystem("pi", beta=beta_team),
    "hybrid": PluginSystem("hybrid", beta=beta_hybrid),
    "hybrid_xft": PluginSystem("hybrid_xft", target="xpts_ft", beta=beta_hybrid),
    "pi_xft": PluginSystem("pi_xft", target="xpts_ft", beta=beta_team),
}
for c in (0.25, 0.5, 0.75, 1.0):
    SYSTEMS[f"hybrid_c{c:g}"] = PluginSystem(f"hybrid_c{c:g}", beta=beta_mixed(c))
    SYSTEMS[f"hybrid_xft_c{c:g}"] = PluginSystem(f"hybrid_xft_c{c:g}", target="xpts_ft", beta=beta_mixed(c))
try:                                                   # the shooter-level targets, once built
    from eracoef import xshoot
    for name, target in xshoot.TARGET_REGISTRY.items():
        SYSTEMS[f"hybrid_{name}"] = PluginSystem(f"hybrid_{name}", target=target, beta=beta_hybrid)
        SYSTEMS[f"split_{name}"] = SplitSystem(f"split_{name}", offense=SYSTEMS[f"hybrid_{name}"], defense=SYSTEMS["hybrid"])
except ImportError:
    pass

# --rankmap=<holdout_<tag>_rank.parquet>: for every system named in --systems that has rank rows in that
# file, also score rankmap_<system>, the rank-calibration correction mapped leave-one-season-out
rm = _flag("rankmap")
if rm:
    from eracoef.holdout import RankMappedSystem
    rank_table = pd.read_parquet(rm)
    for n in list(SYSTEMS):
        if n in set(rank_table.system):
            SYSTEMS[f"rankmap_{n}"] = RankMappedSystem(f"rankmap_{n}", SYSTEMS[n], rank_table)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
names = _list("systems", ["rapm", "pi", "hybrid", "hybrid_xft"])
unknown = [n for n in names if n not in SYSTEMS]
if unknown:
    raise SystemExit(f"unknown systems {unknown}; known: {sorted(SYSTEMS)}")
systems = [SYSTEMS[n] for n in names]
splits = {s: SPLITS[s] for s in _list("splits", [])}
tag = _flag("tag", "_".join(names)[:60])
ref = _flag("ref", names[0])
quiet = "--quiet" in sys.argv

ctx = Context.load(cfg)
ho = Holdout.from_config(cfg, first=int(args[0]) if args else None, last=int(args[1]) if len(args) > 1 else None,
                         ks=_list("k", None, int), lam=_list("lams", None, float))
t0 = time.time()
res = ho.run(systems, ctx, splits=splits, rank="--rank" in sys.argv, out=OUT / f"holdout_{tag}.parquet",
             verbose=not quiet)
print()
print(report(res, ref=ref))
(OUT / "csv").mkdir(exist_ok=True)
res.round(6).to_csv(OUT / "csv" / f"holdout_{tag}.csv", index=False)

if ho.rank_ is not None:
    ho.rank_.to_parquet(OUT / f"holdout_{tag}_rank.parquet", index=False)
    pr = pooled_rank(ho.rank_)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
    print("\n=== rank calibration: what the held-out season wants each decile of the ratings multiplied by")
    print("    (1 = calibrated; rising with decile = the top is under-rated; z = (slope - 1) / se)")
    for (k, system), d in pr[pr.decile >= 0].groupby(["k", "system"]):
        print(f"\n-- K = {k}, {system}")
        print(d.pivot_table(index="decile", columns="side", values=["slope", "z"]).round(3).to_string())

if "--consensus" in sys.argv:
    w = cfg["holdout"]["consensus"]["window"]
    seasons = list(range(int(w[0]), int(w[1]) + 1))
    ctx.current_h = None
    print(f"\n=== against the consensus, {seasons[0]}-{seasons[-1]}, VALIDATION ONLY, read once")
    rows = []
    for s in systems:
        # keep the prior off the consensus window itself, as the shipped ratings do
        ctx.current_h = seasons[0]
        rows.append(dict(system=s.name, **vs_consensus(s.fit(seasons, ctx), cfg)))
    ctx.current_h = None
    C = pd.DataFrame(rows)
    C.to_parquet(OUT / f"holdout_{tag}_consensus.parquet", index=False)
    print(C.round(4).to_string(index=False))

print(f"\nwrote outputs/holdout_{tag}.parquet and outputs/csv/holdout_{tag}.csv  ({time.time() - t0:.0f}s)")
