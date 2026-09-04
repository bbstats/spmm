"""The expected-points closure (src/eracoef/xpts.py) against possessions simulated from its own model.

If the geometric closure is right, then on possessions generated with known make rates, rebound
rate and turnover leak, the expected points and expected attempts it produces from the FIRST
attempt alone must reproduce the realised totals in aggregate.  Anything else is a bookkeeping
error in the closure, not basketball.
"""
import numpy as np
import pandas as pd
import pytest

from eracoef.xpts import ATT1, BUCKETS, expected_points, league_constants

RNG = np.random.default_rng(7)
P_MAKE = {"rim": 0.62, "mid": 0.42, "thr": 0.36}
SHARE1 = {"rim": 0.32, "mid": 0.22, "thr": 0.36, "ft": 0.10}       # first-attempt mix
MIX2 = {"rim": 0.55, "mid": 0.20, "thr": 0.25}                      # continuation mix (putbacks)
Q, R, T2, P_TOV = 0.77, 0.30, 0.10, 0.13


def _simulate(n_rows=400, poss_per_row=12):
    cols = ["pts", "poss", "tov", "att", "fga", "fgm", "fg3m", "fta", "ftm", "xftm", "fta1",
            "fta_tech", "ftm_tech", "xftm_tech", "pts_tech", "reb_chance", "reb_cont", "cont_dead",
            "att_retained", "trip_shoot", "trip_ns"] + ATT1 + [f"{k}_{b}" for b in BUCKETS for k in ("fga", "fgm")]
    out = np.zeros((n_rows, len(cols)))
    ix = {c: i for i, c in enumerate(cols)}
    for i in range(n_rows):
        row = out[i]
        row[ix["poss"]] = poss_per_row
        for _ in range(poss_per_row):
            if RNG.random() < P_TOV:
                row[ix["tov"]] += 1
                continue
            b = RNG.choice(list(SHARE1), p=list(SHARE1.values()))
            row[ix[f"att1_{b}"]] += 1
            row[ix["att"]] += 1
            if b == "ft":
                row[ix["trip_shoot"]] += 1
                made = RNG.random(2) < Q
                row[ix["fta"]] += 2; row[ix["fta1"]] += 2
                row[ix["ftm"]] += made.sum(); row[ix["xftm"]] += 2 * Q; row[ix["pts"]] += made.sum()
                miss = not made[-1]
            else:
                row[ix["fga"]] += 1; row[ix[f"fga_{b}"]] += 1
                if b == "thr":
                    row[ix["fg3m"]] += 0
                made = RNG.random() < P_MAKE[b]
                if made:
                    row[ix["fgm"]] += 1; row[ix[f"fgm_{b}"]] += 1; row[ix["pts"]] += BUCKETS[b]
                    if b == "thr":
                        row[ix["fg3m"]] += 1
                miss = not made
            # the rebound chain
            while miss:
                row[ix["reb_chance"]] += 1
                if RNG.random() >= R:
                    break
                row[ix["reb_cont"]] += 1
                if RNG.random() < T2:
                    row[ix["cont_dead"]] += 1; row[ix["tov"]] += 1
                    break
                b2 = RNG.choice(list(MIX2), p=list(MIX2.values()))
                row[ix["att"]] += 1; row[ix["fga"]] += 1; row[ix[f"fga_{b2}"]] += 1
                made = RNG.random() < P_MAKE[b2]
                if made:
                    row[ix["fgm"]] += 1; row[ix[f"fgm_{b2}"]] += 1; row[ix["pts"]] += BUCKETS[b2]
                    if b2 == "thr":
                        row[ix["fg3m"]] += 1
                miss = not made
    return pd.DataFrame(out, columns=cols)


@pytest.fixture(scope="module")
def sim():
    cnt = _simulate()
    season = np.full(len(cnt), 2024)
    consts = league_constants(cnt, season)
    k = consts[2024]
    rates = pd.DataFrame({"efg": np.full(len(cnt), k.efg), "tov": np.full(len(cnt), k.tov),
                          "oreb": np.full(len(cnt), k.oreb), "ftr": np.full(len(cnt), k.ftr)})
    return cnt, season, consts, rates


def test_constants_recover_the_generator(sim):
    cnt, season, consts, _ = sim
    k = consts[2024]
    assert abs(k.oreb / 100 - R) < 0.03
    assert abs(k.q - Q) < 0.03
    for b, p in P_MAKE.items():
        assert abs(k.p_make[b] - p) < 0.04, b
    assert abs(k.t2 - T2) < 0.05


@pytest.mark.parametrize("variant", ["mult", "add"])
def test_closure_reproduces_points_and_attempts(sim, variant):
    """At league rates, expected points and expected attempts match the realised totals."""
    cnt, season, consts, rates = sim
    xp, diag, clips = expected_points(cnt, rates, season, variant=variant, consts=consts)
    assert clips["p_clip_rate"] == 0.0 and clips["mult_clip_rate"] == 0.0
    ratio = xp.sum() / cnt.pts.sum()
    assert abs(ratio - 1.0) < 0.02, ratio
    exp_mult = (diag.mult * diag.with_att).sum() / diag.with_att.sum()
    assert abs(exp_mult / consts[2024].mult - 1.0) < 0.02, (exp_mult, consts[2024].mult)


def test_no_attempt_means_only_technicals(sim):
    cnt, season, consts, rates = sim
    c = cnt.copy()
    c.loc[:, ATT1] = 0.0
    c["fta1"] = 0.0
    c["xftm_tech"] = 1.5
    xp, diag, _ = expected_points(c, rates, season, consts=consts)
    assert np.allclose(xp, 1.5)
    assert np.allclose(diag.mult, 1.0)


def test_variance_is_an_expectation(sim):
    """The target must vary LESS than the realised points across rows: luck has been removed."""
    cnt, season, consts, rates = sim
    xp, _, _ = expected_points(cnt, rates, season, consts=consts)
    assert xp.std() < 0.6 * cnt.pts.std()


def test_lineup_rates_move_the_target_the_right_way(sim):
    cnt, season, consts, rates = sim
    k = consts[2024]
    base, _, _ = expected_points(cnt, rates, season, consts=consts)
    hot = rates.assign(efg=k.efg + 5.0)
    more_boards = rates.assign(oreb=k.oreb + 10.0)
    careless = rates.assign(tov=k.tov * 2.0)
    assert expected_points(cnt, hot, season, consts=consts)[0].sum() > base.sum()
    assert expected_points(cnt, more_boards, season, consts=consts)[0].sum() > base.sum()
    assert expected_points(cnt, careless, season, consts=consts)[0].sum() < base.sum()
