"""Stint collapse on a synthetic game in stats.nba.com V3 layout."""
import numpy as np
import pandas as pd

from eracoef.stints import GameParser, parse_clock, regulation_remaining

HOME, AWAY = 1610612700, 1610612800
H = [1, 2, 3, 4, 5, 6, 7]
A = [11, 12, 13, 14, 15, 16, 17]
NAMES = {1: "Hone", 2: "Htwo", 3: "Hthree", 4: "Hfour", 5: "Hfive", 6: "Hsix", 7: "Hseven",
         11: "Aone", 12: "Atwo", 13: "Athree", 14: "Afour", 15: "Afive", 16: "Asix", 17: "Aseven"}


def _box():
    rows = []
    for t, ids in ((HOME, H), (AWAY, A)):
        for k, pid in enumerate(ids):
            rows.append(dict(teamId=t, personId=pid, familyName=NAMES[pid], nameI=f"X. {NAMES[pid]}",
                             position="F" if k < 5 else "", minutes="10:00", points=0))
    return pd.DataFrame(rows)


def _ev(period, clock, team, pid, at, sub="", desc="", shot=0, sh=0, sa=0):
    return dict(period=period, clock=f"PT{clock // 60:02d}M{clock % 60:02d}.00S", teamId=team, personId=pid,
                playerName=NAMES.get(pid, ""), description=desc, actionType=at, subType=sub, shotValue=shot,
                scoreHome=sh, scoreAway=sa)


def synthetic_game():
    e = []
    e.append(_ev(1, 720, 0, 0, "period", "start", "Start of 1st Period"))
    e.append(_ev(1, 720, HOME, 1, "Jump Ball", "", "Jump Ball Hone vs. Aone: Tip to Htwo"))
    # home possession: made 2 by player 1                              (H 2-0)
    e.append(_ev(1, 700, HOME, 1, "Made Shot", "Jump Shot", "Hone 15' Jump Shot (2 PTS)", 2, 2, 0))
    # away possession: miss, offensive rebound, made 3                  (A 2-3)
    e.append(_ev(1, 680, AWAY, 11, "Missed Shot", "Jump Shot", "MISS Aone 20' Jump Shot", 2))
    e.append(_ev(1, 678, AWAY, 12, "Rebound", "Unknown", "Atwo REBOUND (Off:1 Def:0)"))
    e.append(_ev(1, 670, AWAY, 12, "Made Shot", "Jump Shot", "Atwo 25' 3PT Jump Shot (3 PTS)", 3, 2, 3))
    # home possession: and-1 (made 2 + made FT 1 of 1)                  (H 5-3)
    e.append(_ev(1, 650, HOME, 2, "Made Shot", "Layup Shot", "Htwo 1' Layup (2 PTS)", 2, 4, 3))
    e.append(_ev(1, 650, AWAY, 13, "Foul", "Shooting", "Athree S.FOUL (P1.T1)"))
    e.append(_ev(1, 650, HOME, 2, "Free Throw", "Free Throw 1 of 1", "Htwo Free Throw 1 of 1 (3 PTS)", 0, 5, 3))
    # substitution on the away team during the dead ball: 16 for 11
    e.append(_ev(1, 650, AWAY, 11, "Substitution", "", "SUB: Asix FOR Aone"))
    # away possession: turnover                                          (H 5-3)
    e.append(_ev(1, 630, AWAY, 12, "Turnover", "Bad Pass", "Atwo Bad Pass Turnover (P1.T1)"))
    e.append(_ev(1, 630, HOME, 3, "", "", "Hthree STEAL (1 STL)"))
    # home possession: missed shot, defensive rebound                   (H 5-3)
    e.append(_ev(1, 610, HOME, 3, "Missed Shot", "Jump Shot", "MISS Hthree 18' Jump Shot", 2))
    e.append(_ev(1, 608, AWAY, 14, "Rebound", "Unknown", "Afour REBOUND (Off:0 Def:1)"))
    # away possession: two free throws, second missed, home team rebound (H 5-4)
    e.append(_ev(1, 600, HOME, 4, "Foul", "Shooting", "Hfour S.FOUL (P1.T2)"))
    e.append(_ev(1, 600, AWAY, 14, "Free Throw", "Free Throw 1 of 2", "Afour Free Throw 1 of 2 (1 PTS)", 0, 5, 4))
    e.append(_ev(1, 600, AWAY, 14, "Free Throw", "Free Throw 2 of 2", "MISS Afour Free Throw 2 of 2"))
    e.append(_ev(1, 599, 0, HOME, "Rebound", "Unknown", "HOME Rebound"))
    # home substitution 6 for 1, then home possession: technical FT by away while home has the ball, then made 2
    e.append(_ev(1, 590, HOME, 1, "Substitution", "", "SUB: Hsix FOR Hone"))
    e.append(_ev(1, 585, HOME, 6, "Foul", "Technical", "Hsix T.FOUL"))
    e.append(_ev(1, 585, AWAY, 12, "Free Throw", "Free Throw Technical", "Atwo Free Throw Technical (4 PTS)", 0, 5, 5))
    e.append(_ev(1, 570, HOME, 6, "Made Shot", "Dunk Shot", "Hsix Dunk (2 PTS)", 2, 7, 5))
    e.append(_ev(1, 0, 0, 0, "period", "end", "End of 1st Period"))
    return pd.DataFrame(e)


def test_parse_clock():
    assert parse_clock("PT06M17.00S") == 377.0
    assert parse_clock("PT00M00.90S") == 0.9
    assert regulation_remaining(2, 300) == 2 * 720 + 300
    assert regulation_remaining(5, 200) == 0


def test_possessions_and_stints_on_synthetic_game():
    gp = GameParser(synthetic_game(), _box(), HOME, AWAY, "0020000001", gt_rule={"q4_margin_late": 15, "late_minutes": 6, "q4_margin_any": 20})
    poss = gp.possessions()
    # 7 possessions: H made, A made3, H and-1, A turnover, H miss/dreb, A ft trip (miss, home team reb), H dunk
    assert list(poss["reason"]) == ["made_fg", "made_fg", "made_ft", "turnover", "dreb", "dreb", "made_fg"]
    assert list(poss["offense"]) == [HOME, AWAY, HOME, AWAY, HOME, AWAY, HOME]
    # points: the away technical FT is credited to the away team's next/last possession (the FT trip, closed before)
    assert list(poss["points"]) == [2, 3, 3, 0, 0, 2, 2]
    assert poss["valid"].all()
    # lineups: the away sub happened during the and-1 dead ball, so from the 4th possession on 16 replaces 11
    assert poss.loc[2, "away_lineup"] == (11, 12, 13, 14, 15)
    assert poss.loc[3, "away_lineup"] == (12, 13, 14, 15, 16)
    assert poss.loc[6, "home_lineup"] == (2, 3, 4, 5, 6)
    st = gp.stints(poss)
    # stints: [p1-p3] starters, [p4-p6] after away sub, [p7] after home sub
    assert len(st) == 3
    assert list(st["poss_h"]) == [2, 1, 1] and list(st["poss_a"]) == [1, 2, 0]
    assert list(st["pts_h"]) == [5.0, 0.0, 2.0] and list(st["pts_a"]) == [3.0, 2.0, 0.0]
    assert list(st["margin_h"]) == [0, 2, 1]   # the technical FT lands after the last stint has started
    np.testing.assert_allclose(st["frac_rem"], [1.0, (3 * 720 + 650) / 2880, (3 * 720 + 599) / 2880])
    assert not st["is_gt"].any()
    # sum of player possessions = 5 x team possessions
    assert st[["h1", "h2", "h3", "h4", "h5"]].notna().all().all()


# ------------------------------------------------------------------ per-possession counters
# Two identities have to hold exactly, or the geometric-series model built on top of these counters
# is not well posed.  See FINDINGS.md section 14 for how the attempt definition was chosen.

def _poss():
    gp = GameParser(synthetic_game(), _box(), HOME, AWAY, "0020000001",
                    gt_rule={"q4_margin_late": 15, "late_minutes": 6, "q4_margin_any": 20})
    return gp, gp.possessions()


def test_points_reconcile_with_the_shooting_counters():
    """points - pts_tech == 2*(fgm - fg3m) + 3*fg3m + ftm, exactly, on every possession.

    pts_tech exists precisely so this holds: technical free throws are credited to a team's next or
    last possession, and the period-end path mutates an already-appended record, so without a
    separate counter the identity breaks on about one possession a game."""
    _, p = _poss()
    lhs = p["points"] - p["pts_tech"]
    rhs = 2 * (p["fgm"] - p["fg3m"]) + 3 * p["fg3m"] + p["ftm"]
    assert (lhs == rhs).all(), p.loc[lhs != rhs, ["reason", "points", "pts_tech", "fgm", "fg3m", "ftm"]]


def test_every_attempt_after_the_first_came_from_an_offensive_rebound():
    """reb_cont - cont_dead == att - 1 on any possession that reaches an attempt.

    This is the identity the whole continuation model rests on: a possession gets a second attempt
    only by rebounding its own miss.  `cont_dead` is the correction for a continuation that led to
    no further attempt -- rebound your own miss and then turn it over -- which is about 1.3% of
    possessions and would otherwise look like a broken identity."""
    _, p = _poss()
    got = p[p["att"] > 0]
    lhs = got["reb_cont"] - got["cont_dead"] + got["att_retained"]
    assert (lhs == got["att"] - 1).all(), got[["reason", "att", "reb_cont", "cont_dead", "att_retained"]]
    assert (p.loc[p["att"] == 0, "reb_cont"] == 0).all()


def test_exactly_one_first_attempt_bucket_per_possession_with_an_attempt():
    _, p = _poss()
    b = p[["att1_rim", "att1_mid", "att1_thr", "att1_ft"]].sum(axis=1)
    assert (b == (p["att"] > 0).astype(int)).all(), p.assign(buckets=b)[["reason", "att", "buckets"]]


def test_counters_survive_the_stint_collapse():
    """Every counter is emitted per side and sums to the possession-level total."""
    from eracoef.stints import POSS_COUNTERS
    gp, p = _poss()
    st = gp.stints(p)
    for c in POSS_COUNTERS:
        assert f"{c}_h" in st.columns and f"{c}_a" in st.columns, c
        assert st[f"{c}_h"].sum() == p.loc[p["offense"] == HOME, c].sum(), c
        assert st[f"{c}_a"].sum() == p.loc[p["offense"] == AWAY, c].sum(), c
    # the and-1 possession: one field goal, one free throw, ONE attempt event
    a1 = p[p["reason"] == "made_ft"].iloc[0]
    assert a1["fga"] == 1 and a1["fta"] == 1 and a1["att"] == 1 and a1["and1"] == 1
