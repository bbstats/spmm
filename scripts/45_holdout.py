"""The out-of-season criterion, from the command line.  One entry point for every comparison.

Hold out each season, train on the K nearest seasons around it, predict its stints, score against
actual points at stint and team-game level.  src/eracoef/holdout.py is the system, src/eracoef/
systems.py names the systems this project has built, and this prints the report.

    rapm            no box prior
    pi              the team-priced prior (prior-informed RAPM)
    hybrid          the shipped prior: player-priced offense, no defensive prior
    hybrid_xft      hybrid on the free-throw-adjusted target
    hybrid_c<c>     hybrid offense with the team-priced defensive prior scaled by c (0.25, 0.5, 0.75, 1)
    hybrid_xshoot*  the shooter-level expected-points targets
    def3_p0         the board: offense from hybrid_xft, defense from the opponent-3PM-replaced target
    spm             the role prior (APM -> Simple SPM -> ridge) on both sides, no box prior
    mspi_resid        the role prior + a GBDT on the residual beyond it, both sides (mspi_resid_o: offense only)
    mspi       a GBDT on RAPM_1 with the role inputs among its features, alone (mspi_o: offense only)

Every earlier held-out script is one invocation of this:
    38_yoy          --systems=rapm,pi,hybrid,hybrid_xft
    39_why          --systems=pi,hybrid --splits=exposure
    40_movers       --systems=pi,hybrid --splits=movers
    41/42_defweight --systems=hybrid,hybrid_c0.25,hybrid_c0.5,hybrid_c0.75,hybrid_c1 --consensus
    44_yoy_shot     --systems=hybrid,hybrid_xft,hybrid_xshoot,...

usage: python scripts/45_holdout.py [first] [last] --systems=a,b,c [--k=2,4] [--lams=18352]
           [--splits=movers,exposure,bigs,bench] [--rank] [--consensus] [--tag=name] [--ref=hybrid_xft]
           [--workers=4] [--spread] [--top=1997-1999] [--pdp] [--rankmap=<rank parquet>] [--quiet]
    --workers=N  run the held-out seasons across N processes (config holdout.workers; 1 = in this process)
    --spread     per system and side, the possession-weighted sd of the prior against the sd of the residual
                 (players with 1000+ possessions), on the consensus window and on 1997-1999
    --top=A-B    fit each system on that block and print the top 15 offense with prior, residual and rating
    --pdp        the boosted prior's partial dependence of assists, steals and threes by season
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.holdout import SPLITS, Context, Holdout, pooled_rank, report, run_parallel, vs_consensus  # noqa: E402
from eracoef.systems import registry  # noqa: E402

cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"


def _flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


def _list(name, default, conv=str):
    v = _flag(name)
    return default if v is None else [conv(x) for x in v.split(",") if x]


def _wsd(v, w):
    v, w = np.asarray(v, dtype=float), np.asarray(w, dtype=float)
    if w.sum() <= 0:
        return float("nan")
    mu = np.average(v, weights=w)
    return float(np.sqrt(np.average((v - mu) ** 2, weights=w)))


def _fit_block(system, seasons, ctx):
    """Fit on a block with the prior kept off the block's own windows (what the shipped ratings do)."""
    ctx.current_h = seasons[0]
    try:
        return system.fit(seasons, ctx)
    finally:
        ctx.current_h = None


def main():
    rm = _flag("rankmap")
    SYSTEMS = registry(cfg, rankmap=rm)

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = _list("systems", ["rapm", "pi", "hybrid", "hybrid_xft"])
    unknown = [n for n in names if n not in SYSTEMS]
    if unknown:
        raise SystemExit(f"unknown systems {unknown}; known: {sorted(SYSTEMS)}")
    systems = [SYSTEMS[n] for n in names]
    split_names = _list("splits", [])
    splits = {s: SPLITS[s] for s in split_names}
    tag = _flag("tag", "_".join(names)[:60])
    ref = _flag("ref", names[0])
    quiet = "--quiet" in sys.argv
    workers = int(_flag("workers", cfg.get("holdout", {}).get("workers", 1)))
    skip_run = "--norun" in sys.argv

    ctx = Context.load(cfg)
    ho = Holdout.from_config(cfg, first=int(args[0]) if args else None, last=int(args[1]) if len(args) > 1 else None,
                             ks=_list("k", None, int), lam=_list("lams", None, float))
    t0 = time.time()
    gbdt_reports = []
    if not skip_run:
        if workers > 1:
            res, _, gbdt_reports = run_parallel(ho, names, splits=split_names, rank="--rank" in sys.argv,
                                                out=OUT / f"holdout_{tag}.parquet", verbose=not quiet, workers=workers,
                                                rankmap=rm)
        else:
            res = ho.run(systems, ctx, splits=splits, rank="--rank" in sys.argv, out=OUT / f"holdout_{tag}.parquet",
                         verbose=not quiet)
            gbdt_reports = [r for prior in (ctx.gbdt, ctx.mspi) if prior is not None
                            for r in getattr(prior, "reports", [])]
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

        if gbdt_reports:
            g = pd.DataFrame(gbdt_reports)
            g.to_csv(OUT / "csv" / f"holdout_{tag}_gbdt_drag.csv", index=False)
            print("\n=== boosted prior: distributional drag of each leave-window-out training set (points per 100)")
            print(f"    {len(g)} fits; |drag| max {g.drag_before.abs().max():.4f}, counterbalanced {int(g.applied.sum())}; "
                  f"rows {int(g.n_rows.min())}-{int(g.n_rows.max())}; best iteration median {g.best_iteration.median():.0f}")

    if "--consensus" in sys.argv:
        w = cfg["holdout"]["consensus"]["window"]
        seasons = list(range(int(w[0]), int(w[1]) + 1))
        print(f"\n=== against the consensus, {seasons[0]}-{seasons[-1]}, VALIDATION ONLY, read once")
        rows = []
        for s in systems:
            rows.append(dict(system=s.name, **vs_consensus(_fit_block(s, seasons, ctx), cfg)))
        C = pd.DataFrame(rows)
        C.to_parquet(OUT / f"holdout_{tag}_consensus.parquet", index=False)
        print(C.round(4).to_string(index=False))

    if "--spread" in sys.argv:
        w = cfg["holdout"]["consensus"]["window"]
        blocks = [list(range(int(w[0]), int(w[1]) + 1)), [1997, 1998, 1999]]
        print("\n=== prior spread against residual spread, possession-weighted sd over players with 1000+ possessions")
        print("    (HANDOFF item 6: the linear offensive prior on 1997-99 had sd 1.76 against 0.36 for the residual)")
        rows = []
        for seasons in blocks:
            for s in systems:
                r = _fit_block(s, seasons, ctx).df
                r = r[r.poss >= 1000]
                for side in ("o", "d"):
                    pc = f"prior_{side}"
                    prior = r[pc].to_numpy() if pc in r.columns else np.zeros(len(r))
                    rows.append(dict(block=f"{seasons[0]}-{seasons[-1]}", system=s.name, side=side, n=len(r),
                                     sd_prior=_wsd(prior, r.poss), sd_resid=_wsd(r[side].to_numpy() - prior, r.poss),
                                     sd_rating=_wsd(r[side].to_numpy(), r.poss),
                                     corr_prior_rating=float(np.corrcoef(prior, r[side])[0, 1]) if prior.std() > 0 else float("nan")))
        Sp = pd.DataFrame(rows)
        Sp.to_csv(OUT / "csv" / f"holdout_{tag}_spread.csv", index=False)
        print(Sp.round(3).to_string(index=False))

    top = _flag("top")
    if top:
        a, b = (int(x) for x in top.split("-"))
        seasons = list(range(a, b + 1))
        from eracoef.boxtable import player_names, season_box
        nm = player_names(season_box(seasons, ["RS"], cfg)).sort_values("season").drop_duplicates("player_id", keep="last")
        print(f"\n=== top 15 offense on {a}-{b}, per system: prior, residual, rating (raw sign; players with 2000+ possessions)")
        for s in systems:
            r = _fit_block(s, seasons, ctx).df
            r = r[r.poss >= 2000].merge(nm[["player_id", "player_name"]], on="player_id", how="left")
            r["prior"] = r["prior_o"] if "prior_o" in r.columns else 0.0
            r["resid"] = r.o - r.prior
            t = r.nlargest(15, "o")[["player_name", "poss", "prior", "resid", "o"]]
            print(f"\n-- {s.name}")
            print(t.round(2).to_string(index=False))

    if "--pdp" in sys.argv:
        if ctx.gbdt is None:
            raise SystemExit("--pdp needs outputs/role_panel.parquet (scripts/49_role_panel.py)")
        from eracoef.gbdt_prior import partial_dependence, training_rows
        seasons = [1998, 2004, 2010, 2016, 2022, 2025]
        rows = []
        for side in ("O", "D"):
            feats = ctx.gbdt.features[side]
            tr = training_rows(ctx.rpanel, side, exclude=(), features=feats)
            model, _ = ctx.gbdt.model(side, exclude=())
            for f in ("ast", "stl", "fg3m", "orb", "blk"):
                if f not in feats:
                    continue
                grid = np.quantile(tr[f], [0.1, 0.5, 0.9])
                pdp = partial_dependence(model, tr, feats, f, grid, seasons=seasons if "season" in feats else None)
                pdp["side"] = side
                rows.append(pdp)
        Pd = pd.concat(rows, ignore_index=True)
        Pd.to_csv(OUT / "csv" / "gbdt_pdp.csv", index=False)
        print("\n=== boosted prior: value of moving a rate from its 10th to its 90th percentile, by season (raw sign)")
        piv = (Pd.pivot_table(index=["side", "feature"], columns="season", values="pd", aggfunc=["max", "min"]))
        print((piv["max"] - piv["min"]).round(3).to_string())

    print(f"\nwrote outputs/holdout_{tag}.parquet and outputs/csv/holdout_{tag}.csv  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":       # the parallel runner spawns workers that re-import this file
    main()
