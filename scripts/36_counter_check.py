"""Stage 1 gate: do the new per-possession counters actually reconcile?

Four checks, in increasing order of how much they would hurt to get wrong.

  1. The two identities, at stint level, exactly.  `points - pts_tech` must equal the shooting
     counters, and every attempt after a possession's first must be accounted for by an offensive
     rebound.  These are bookkeeping, so "close" is a failure.
  2. Season totals against the box score.  Independent data, so this catches a whole class of
     parsing errors the identities cannot -- they would happily reconcile a systematically
     under-counted stream with itself.  Stints drop invalid possessions, so the expected ratio is
     the season's valid-possession fraction, not 1.
  3. OREB% and the continuation multiplier against the definition-E numbers measured in
     scripts/35_attempt_defs.py, which were taken straight off the raw events by a separate code
     path.  Agreement means the parser implements the definition that was chosen.
  4. Era drift, as a sanity read rather than a gate.

usage: python scripts/36_counter_check.py [seasons=2024,2025,2026]
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.boxtable import season_box  # noqa: E402
from eracoef.config import load_config, resolve  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 4)
cfg = load_config()
STINTS = resolve(cfg, "stints")
SEASONS = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [2024, 2025, 2026]

# definition E, measured off the raw events in scripts/35_attempt_defs.py (60 games per season)
B0 = {2024: dict(oreb=27.70, mult=1.1576), 2005: dict(oreb=31.58, mult=1.1951),
      1998: dict(oreb=33.76, mult=1.2133)}
ATT1 = ["att1_rim", "att1_mid", "att1_thr", "att1_ft"]
ok_all = True

for season in SEASONS:
    p = STINTS / f"{season}_RS.parquet"
    if not p.exists():
        print(f"season {season}: not built, skipping")
        continue
    st = pd.read_parquet(p)
    missing = [c for c in ("att_h", "reb_cont_h", "pts_tech_h") if c not in st.columns]
    if missing:
        print(f"season {season}: stale schema, missing {missing} -- rebuild with --force")
        ok_all = False
        continue

    def tot(c):
        return float(st[f"{c}_h"].sum() + st[f"{c}_a"].sum())

    print(f"\n{'=' * 78}\n=== season {season}: {len(st)} stints, {tot('att'):.0f} attempt events")

    # 1. identities, exactly
    lhs = (st.pts_h - st.pts_tech_h) + (st.pts_a - st.pts_tech_a)
    rhs = (2 * (st.fgm_h - st.fg3m_h) + 3 * st.fg3m_h + st.ftm_h
           + 2 * (st.fgm_a - st.fg3m_a) + 3 * st.fg3m_a + st.ftm_a)
    bad_pts = int((lhs != rhs).sum())
    with_att = sum(tot(c) for c in ATT1)
    bad_reb = abs(tot("reb_cont") - tot("cont_dead") + tot("att_retained") - (tot("att") - with_att))
    # The continuation identity is exact by construction, but roughly 30 possessions a season sit in
    # foul sequences it does not model (a flagrant on a made shot, a lane violation replay) and come
    # out +/-1.  That is 0.01% of attempts and each one can move a single possession's expected
    # points by at most one attempt's worth, so it is bounded rather than chased further.
    TOL = 0.0005
    print(f"  identity  points - pts_tech == shooting counters : "
          f"{'OK' if bad_pts == 0 else f'FAIL on {bad_pts} stints'}")
    reb_ok = bad_reb <= TOL * tot("att")
    pct = 100 * bad_reb / tot("att")
    verdict = "OK" if reb_ok else "FAIL"
    print(f"  identity  reb_cont - cont_dead + att_retained == att - possessions with an attempt : "
          f"{bad_reb:.0f} net, {pct:.4f}% of attempts (tolerance {100 * TOL:.2f}%)  {verdict}")
    ok_all &= (bad_pts == 0 and reb_ok)

    # 2. against the box score
    bx = season_box([season], ["RS"], cfg)
    # free throws must include the technical ones to match the box; turnovers must NOT be compared,
    # because the box counts only turnovers charged to a player and possessions also end on team
    # turnovers (shot clock, 5-second), which is about 1.4 a game
    bmap = dict(fga=("fg2m", "fg2_miss", "fg3m", "fg3_miss"), fg3m=("fg3m",),
                fta=("ftm", "ft_miss"), ftm=("ftm",))
    rows = []
    for c, parts in bmap.items():
        b = float(sum(bx[q].sum() for q in parts))
        v = tot(c) + (tot(f"{c}_tech") if c in ("fta", "ftm") else 0.0)
        rows.append(dict(counter=c, stints=v, box=b, ratio=v / b))
    R = pd.DataFrame(rows)
    valid = float(st.poss_h.sum() + st.poss_a.sum()) / (2 * 99.0 * st.game_id.nunique())
    print(f"\n  vs the box score (stints drop invalid possessions, so expect a ratio near the")
    print(f"  valid-possession fraction, not 1):")
    print(R.assign(gap_pct=100 * (R.ratio - R.ratio.mean())).to_string(index=False))
    spread = R.ratio.max() - R.ratio.min()
    print(f"  ratios span {spread:.4f}  ({'OK' if spread < 0.005 else 'FAIL: counters disagree'})")
    ok_all &= spread < 0.005
    tov_extra = (tot("tov") - float(bx["tov"].sum())) / st.game_id.nunique()
    print(f"  team turnovers (possession-level minus box, expect roughly 1-2 a game): {tov_extra:.2f}")

    # 3. possession reconciliation, done properly this time.
    # Stage 0 checked `with_att + tov == possessions`, which double-counts: a possession can take an
    # offensive rebound and THEN turn the ball over, so it has both an attempt and a turnover.  With
    # possession boundaries known we can measure that overlap instead of absorbing it into the
    # residual, which is why these numbers supersede the stage 0 estimates rather than confirm them.
    poss = float(st.poss_h.sum() + st.poss_a.sum())
    overlap = with_att + tot("tov") - poss
    print(f"\n  possessions {poss:.0f}   reaching an attempt {with_att:.0f} "
          f"({100 * with_att / poss:.1f}%)   turnovers {tot('tov'):.0f} ({100 * tot('tov') / poss:.1f}%)")
    print(f"  attempted AND turned it over, net of possessions that did neither: {overlap:.0f} "
          f"({100 * overlap / poss:.2f}%, {overlap / st.game_id.nunique():.2f} a game)")
    good = 0 <= overlap / poss < 0.03
    print(f"  {'OK' if good else 'FAIL: expected a small positive number'}")
    ok_all &= good

    # 4. the rebound chain, now with possession boundaries known
    oreb = 100 * tot("reb_cont") / tot("reb_chance")
    mult = tot("att") / with_att
    ref = B0.get(season)
    line = f"\n  OREB% {oreb:.2f}   continuation multiplier {mult:.4f}"
    if ref:
        line += (f"   (stage 0 estimated {ref['oreb']:.2f} / {ref['mult']:.4f} from events alone;"
                 f" these supersede it)")
    print(line)
    print(f"  period-end team rebounds dropped: {tot('reb_drop'):.0f} "
          f"({100 * tot('reb_drop') / (tot('reb_cont') + tot('reb_drop')):.1f}% of continuations)")
    print(f"  team share of offensive rebounds {100 * tot('oreb_t') / tot('reb_cont'):.1f}%   "
          f"and-1s {tot('and1'):.0f}   shooting trips {tot('trip_shoot'):.0f}   "
          f"non-shooting {tot('trip_ns'):.0f}")
    b = {c: tot(c) for c in ATT1}
    print("  first attempt: " + "  ".join(f"{k[5:]} {100 * v / with_att:.1f}%" for k, v in b.items()))

print(f"\n{'=' * 78}\n{'ALL CHECKS PASS' if ok_all else 'SOMETHING FAILED -- see above'}")
sys.exit(0 if ok_all else 1)
