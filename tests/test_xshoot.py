"""Shooter-level expected points (src/eracoef/xshoot.py): the pricing, the padding and the alignment."""
import numpy as np
import pandas as pd

from eracoef.stints import SHOOTER_COUNTERS, SLOTS
from eracoef.xshoot import ShooterRates, align, expected_makes, rates_from_tables


def _tables():
    """Two shooters over two halves: 10 makes twos at an easy spot, 3s, and free throws."""
    rows = []
    for h, gid in (("A", "g1"), ("B", "g2")):
        rows.append(dict(game_id=gid, player_id=1, half=h, fg2a=100, fg2m=60, xl2=50.0, fg3a=50, fg3m=20, xl3=18.0))
        rows.append(dict(game_id=gid, player_id=2, half=h, fg2a=100, fg2m=40, xl2=50.0, fg3a=0, fg3m=0, xl3=0.0))
    for pid in range(3, 60):                           # a population so mom_k has something to estimate
        for h, gid in (("A", "g1"), ("B", "g2")):
            m = 40 + (pid % 7) * 3
            rows.append(dict(game_id=gid, player_id=pid, half=h, fg2a=100, fg2m=m, xl2=50.0, fg3a=60, fg3m=18 + pid % 5, xl3=21.0))
    shots = pd.DataFrame(rows)
    ft = pd.DataFrame([dict(game_id="g1", player_id=1, half="A", ftm=90, ft_miss=10, fta=100),
                       dict(game_id="g2", player_id=1, half="B", ftm=90, ft_miss=10, fta=100)] +
                      [dict(game_id=g, player_id=p, half=h, ftm=50 + (p % 10) * 5, ft_miss=50 - (p % 10) * 5, fta=100)
                       for p in range(2, 60) for g, h in (("g1", "A"), ("g2", "B"))])
    return shots, ft


def test_rates_pad_toward_own_mix():
    shots, ft = _tables()
    r = rates_from_tables(shots, ft)
    # player 1 makes 60% from spots where the league makes 50%: ratio above 1, below 1.2 (padded)
    assert 1.05 < r.ratio2["B"][1] < 1.2
    # player 2 makes 40% from 50% spots: ratio below 1
    assert 0.8 < r.ratio2["B"][2] < 0.95
    # no threes at all: the average shooter from his spots
    assert r.ratio3["B"][2] == 1.0
    assert 0.85 < r.pft["B"][1] < 0.9                  # 90% padded toward the league ~0.74
    assert r.k["fg2"] > 0 and r.league["fg2"] > 0.4
    # the RS totals pool both halves: twice the attempts, so less padding, so nearer the raw 1.2
    assert r.ratio2["A"][1] < r.ratio2["RS"][1] < 1.2


def test_extra_seasons_and_fixed_k():
    """Earlier seasons pool into every half's totals; a fixed k overrides the method of moments."""
    shots, ft = _tables()
    base = rates_from_tables(shots, ft, k_fixed={"fg3": 450.0})
    assert base.k["fg3"] == 450.0
    # player 1: 50 threes at 40% per half; an extra season of 200 more at 40% moves him nearer 0.40
    extra = pd.DataFrame([dict(game_id="g0", player_id=1, half="A", fg2a=0, fg2m=0, xl2=0.0, fg3a=200, fg3m=80, xl3=72.0)])
    more = rates_from_tables(shots, ft, extra=extra, k_fixed={"fg3": 450.0})
    assert more.p3["B"][1] > base.p3["B"][1]                    # more evidence of a 40% shooter than the league ~0.33
    assert more.p3["A"][1] > base.p3["A"][1] and more.p3["RS"][1] > base.p3["RS"][1]
    assert more.pft["B"][1] == base.pft["B"][1]                 # the extra table has no free throws: untouched


def _cnt(pids, half, **slot):
    d = {f"{c}_s{s}": [0.0] for c in SHOOTER_COUNTERS for s in SLOTS}
    for k, v in slot.items():
        d[k] = [float(v)]
    for i, p in enumerate(pids):
        d[f"pid_s{i + 1}"] = [p]
    d["half"] = [half]
    return pd.DataFrame(d)


def test_expected_makes_prices_each_slot_with_the_other_half():
    r = ShooterRates(ratio2={"A": {1: 1.5}, "B": {1: 1.2}, "RS": {1: 1.3}}, ratio3={"A": {}, "B": {}, "RS": {}},
                     p2={"A": {1: 0.6}, "B": {1: 0.55}, "RS": {1: 0.58}}, p3={"A": {}, "B": {}, "RS": {}},
                     pft={"A": {1: 0.9}, "B": {1: 0.8}, "RS": {1: 0.85}}, league={"fg2": 0.5, "fg3": 0.36, "ft": 0.75},
                     k={"fg2": 30.0, "fg3": 30.0, "ft": 30.0})
    # row in half A -> priced with the B rates; player 1 in slot 2 took two twos (league expectation 0.9),
    # one of them the first attempt (0.5), and two free throws on the first attempt; slot x had one three
    c = _cnt([7, 1, 8, 9, 10], "A", fg2a_s2=2, xl2_s2=0.9, fg2a1_s2=1, xl2_1_s2=0.5, fta_s2=2, fta1_s2=2, ftlast1_s2=1,
             fg3a_sx=1, xl3_sx=0.4, fg3a1_sx=0, xl3_1_sx=0)
    x = expected_makes(c, r, location=True).iloc[0]
    assert abs(x.x2pm - 0.9 * 1.2) < 1e-12
    assert abs(x.x3pm - 0.4) < 1e-12                   # slot x: the league's view
    assert abs(x.xftm - 2 * 0.8) < 1e-12
    assert abs(x.xpts1 - (2 * 0.5 * 1.2 + 2 * 0.8)) < 1e-12
    assert abs(x.xchance1 - ((1 - 0.5 * 1.2) + 1 * (1 - 0.8))) < 1e-12
    f = expected_makes(c, r, location=False).iloc[0]
    assert abs(f.x2pm - 2 * 0.55) < 1e-12
    assert abs(f.x3pm - 0.36) < 1e-12
    # a playoff row uses the whole regular season; an unknown shooter is the average one
    c2 = _cnt([1, 2, 3, 4, 5], "PO", fg2a_s1=1, xl2_s1=0.5, fg2a_s2=1, xl2_s2=0.5)
    x2 = expected_makes(c2, r, location=True).iloc[0]
    assert abs(x2.x2pm - (0.5 * 1.3 + 0.5 * 1.0)) < 1e-12


def test_align_only_outside_the_band():
    rng = np.random.default_rng(0)
    season = np.repeat([2001, 2002], 500)
    poss = np.full(1000, 10.0)
    y_pts = rng.normal(110, 30, 1000)
    y = np.where(season == 2001, y_pts * 1.0, y_pts * 1.05)     # 2002 drifts 5% high
    ya, rep = align(y, y_pts, poss, season, poss)
    assert not rep[rep.season == 2001].applied.iloc[0]          # inside the band: recorded as such
    assert rep[rep.season == 2002].applied.iloc[0]
    for s in (2001, 2002):                                       # the level matches in every season
        m = season == s
        assert abs((ya[m] * poss[m]).sum() / (y_pts[m] * poss[m]).sum() - 1.0) < 1e-6
    m = season == 2002
    assert np.allclose(ya[m] / y[m], 1 / 1.05)                   # a scalar, not an affine: shape untouched
