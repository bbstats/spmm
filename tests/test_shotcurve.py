"""The per-season shot-distance curve (src/eracoef/shotcurve.py) on shots simulated from a known one."""
import numpy as np
import pandas as pd

from eracoef.shotcurve import MAX_DIST, UNLOCATED, ShotCurve, calibration, fit_curve, shot_bin, shot_rows


def _true_p(dist, three):
    if three:
        return 0.40 - 0.01 * (dist - 23)                       # 40% at the arc, drifting down
    return 0.66 * np.exp(-0.06 * dist) + 0.38 * (1 - np.exp(-0.06 * dist))   # rim 66% down to ~40%


def _simulate(n=60000, seed=5):
    rng = np.random.default_rng(seed)
    three = rng.random(n) < 0.35
    dist = np.where(three, rng.uniform(22, 30, n), rng.exponential(8, n).clip(0, 24)).round()
    p = np.array([_true_p(d, t) for d, t in zip(dist, three)])
    made = rng.random(n) < p
    # an unlocated share: 15% of threes at distance 0, and some twos at 0 with no rim word
    unloc = (three & (rng.random(n) < 0.15)) | (~three & (dist == 0) & (rng.random(n) < 0.3))
    desc = np.where(three, "X 25' 3PT Jump Shot", np.where(dist == 0, "X Layup", "X Jump Shot"))
    desc = np.where(unloc & ~three, "X Jump Shot", desc)
    dist = np.where(unloc, 0.0, dist)
    return pd.DataFrame({"actionType": np.where(made, "Made Shot", "Missed Shot"), "shotValue": np.where(three, 3, 2),
                         "shotDistance": dist, "description": desc}), p


def test_shot_bin_rules():
    assert shot_bin(-1.0, False) == UNLOCATED
    assert shot_bin(0.0, True) == UNLOCATED and shot_bin(15.0, True) == UNLOCATED
    assert shot_bin(0.0, False, "Hone Dunk (2 PTS)") == 0
    assert shot_bin(0.0, False, "Hone Jump Shot") == UNLOCATED
    assert shot_bin(25.4, True) == 25 and shot_bin(80.0, True) == MAX_DIST


def test_curve_recovers_the_truth():
    pbp, _ = _simulate()
    rows = shot_rows(pbp)
    assert set(rows.columns) == {"three", "made", "bin"}
    curve = fit_curve(rows, 2024)
    for d in (1, 5, 10, 15, 20):
        assert abs(curve.prob(d, False, "X Jump Shot" if d else "X Layup") - _true_p(d, False)) < 0.03, d
    for d in (23, 25, 28):
        assert abs(curve.prob(d, True) - _true_p(d, True)) < 0.03, d
    cal = calibration(curve, rows)
    assert cal.gap.abs().max() < 0.04
    # the unlocated cells carry their own rate, near the league rate of that shot value
    assert abs(curve.prob(0.0, True) - rows[rows.three].made.mean()) < 0.05
    assert curve.table.three.nunique() == 2 and (curve.table.bin == UNLOCATED).sum() == 2


def test_constant_curve_and_roundtrip():
    c = ShotCurve.constant(p2=0.5, p3=0.35)
    assert c.prob(7.0, False) == 0.5 and c.prob(24.0, True) == 0.35 and c.prob(-1.0, False) == 0.5
    again = ShotCurve(c.season, c.table.copy())
    assert again.prob(7.0, False) == 0.5
