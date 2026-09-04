"""Sanity tests for the shipped 2024-26 board against an external consensus of modern all-in-one
metrics (data/external/consensus.csv, refreshed by scripts/22_vs_consensus.py --refresh).

Why these exist.  The ratings were previously validated against our OWN next-window on-court RAPM.
That benchmark is built from the same margin data and shares the same blind spots, so it certified
a board that puts Robert Williams 25th and Jusuf Nurkic 21st while every modern metric combined has
them 134th and 166th.  An external benchmark is the only thing that catches a shared error, so it
belongs in the test suite rather than in a one-off script.

Two groups:
  * guards        properties that hold today; they fail if something regresses
  * targets       the defect itself, marked xfail; they flip to passing when it is fixed

Five of the original six targets flipped when the ratings moved to the hybrid prior and are guards
now.  One is left: pure on-court defensive RAPM still rates backup bigs above the consensus, which
is an attribution question the box prior was never going to answer.

Everything skips cleanly if the ratings or the consensus file are not built yet.
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RATINGS = ROOT / "outputs" / "player_ratings.parquet"
CONSENSUS = ROOT / "data" / "external" / "consensus.csv"
WINDOW = "2024-2026"
MIN_POSS = 1000
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("&", " ").replace("-", " ").replace("'", "").replace(".", " ")
    return re.sub(r"\s+", " ", SUFFIX.sub(" ", s)).strip()


def _bigness(player_ids):
    """Per-36 rebound-and-block minus playmaking index, the axis the defect runs along.

    Needs the box cache, so the tests that use it skip in a fresh clone.
    """
    from eracoef.boxtable import season_box
    from eracoef.config import load_config
    try:
        box = season_box([2024, 2025, 2026], ["RS"], load_config())
    except Exception:                                     # no box cache in a fresh clone
        pytest.skip("box scores not built; run scripts/01_ingest.py")
    g = box[box.phase == "RS"].groupby("player_id", as_index=False)[
        ["minutes", "orb", "drb", "blk", "ast", "fg3m"]].sum()
    for c in ("orb", "drb", "blk", "ast", "fg3m"):
        g[c] = g[c] / g.minutes.clip(lower=1) * 36
    g["bigness"] = g.orb + g.blk + 0.3 * g.drb - 0.5 * g.ast - 0.4 * g.fg3m
    return pd.Series(player_ids).map(g.set_index("player_id").bigness).to_numpy()


@pytest.fixture(scope="module")
def board():
    if not RATINGS.exists() or not CONSENSUS.exists():
        pytest.skip("run scripts/08_ratings.py and scripts/22_vs_consensus.py first")
    con = pd.read_csv(CONSENSUS)[["player_name", "team", "adj_offense", "adj_defense", "adj_overall"]]
    con = con.dropna(subset=["player_name", "adj_overall"])
    con["key"] = con.player_name.map(_norm)
    con = con.drop_duplicates("key")

    rat = pd.read_parquet(RATINGS)
    ours = rat[(rat.window == WINDOW) & rat.player_name.notna()].copy()
    ours["key"] = ours.player_name.map(_norm)
    ours = ours.sort_values("poss_off").drop_duplicates("key", keep="last")

    m = ours.merge(con, on="key", how="inner")
    m = m[m.poss_off >= MIN_POSS].copy()
    if len(m) < 300:
        pytest.skip(f"only {len(m)} players matched; the join or the window is wrong")
    m["bigness"] = _bigness(m.player_id)
    for side, con_col in (("total", "adj_overall"), ("off", "adj_offense"), ("def", "adj_defense")):
        m[f"rk_ours_{side}"] = m[f"rating_{side}"].rank(ascending=False)
        m[f"rk_con_{side}"] = m[con_col].rank(ascending=False)
        z_o = (m[f"rating_{side}"] - m[f"rating_{side}"].mean()) / m[f"rating_{side}"].std()
        z_c = (m[con_col] - m[con_col].mean()) / m[con_col].std()
        m[f"gap_{side}"] = z_o - z_c
    return m


def _rank(board, key, side="total"):
    g = board[board.key == key]
    assert len(g) == 1, f"{key!r} matched {len(g)} rows"
    return float(g.iloc[0][f"rk_ours_{side}"])


# --------------------------------------------------------------------------------- guards
def test_join_covers_the_consensus(board):
    """The name join has no ids on either side, so guard against it quietly rotting."""
    assert len(board) >= 400


def test_the_very_top_of_the_board_is_right(board):
    """Whatever else is wrong, the top five should be the consensus top five."""
    ours = set(board.nsmallest(5, "rk_ours_total").key)
    theirs = set(board.nsmallest(5, "rk_con_total").key)
    assert len(ours & theirs) >= 4, f"ours {sorted(ours)} vs theirs {sorted(theirs)}"


def test_offense_agrees_with_the_consensus(board):
    """Offense is the half that works: the box score genuinely measures it."""
    rho = spearmanr(board.rating_off, board.adj_offense).statistic
    assert rho >= 0.82, f"offensive rank agreement fell to {rho:.3f}"


def test_offensive_spread_is_calibrated(board):
    ratio = board.rating_off.std() / board.adj_offense.std()
    assert 0.8 <= ratio <= 1.3, f"offensive spread ratio {ratio:.2f} is off"


def test_offense_has_no_big_man_bias(board):
    """The offensive gap should not track archetype. It currently does not."""
    r = board.gap_off.corr(board.bigness) if "bigness" in board else None
    if r is None:
        pytest.skip("bigness not on the ratings table")
    assert abs(r) < 0.30, f"offensive gap correlates {r:+.3f} with bigness"


# ------------------------------------------------------------------ fixed by the hybrid prior
# These five were the defect.  They flipped when scripts/08_ratings.py moved to the hybrid prior --
# box score priced to predict a PLAYER on offense, no box prior at all on defense -- so they are
# guards now: defensive spread 2.13 -> 1.23, defensive agreement 0.755 -> 0.888, archetype bias
# +0.63 -> +0.20, overall agreement 0.784 -> 0.896, and the star guards came back up the board.
def test_defensive_spread_is_calibrated(board):
    ratio = board.rating_def.std() / board.adj_defense.std()
    assert ratio <= 1.4, f"defensive spread ratio is {ratio:.2f}"


def test_defense_agrees_with_the_consensus(board):
    rho = spearmanr(board.rating_def, board.adj_defense).statistic
    assert rho >= 0.82, f"defensive rank agreement is {rho:.3f}"


def test_no_archetype_bias_overall(board):
    r = board.gap_total.corr(board.bigness)
    assert abs(r) < 0.30, f"total gap correlates {r:+.3f} with bigness"


def test_star_guards_are_not_buried(board):
    """The consensus has all of these inside its top 80."""
    anchors = {"trae young": 200, "lamelo ball": 60, "devin booker": 55, "stephen curry": 40}
    bad = {k: _rank(board, k) for k, ceil in anchors.items()
           if k in set(board.key) and _rank(board, k) > ceil}
    assert not bad, f"ranked far too low: {bad}"


def test_overall_agrees_with_the_consensus(board):
    rho = spearmanr(board.rating_total, board.adj_overall).statistic
    assert rho >= 0.85, f"overall rank agreement is {rho:.3f}"


# --------------------------------------------------------------------------------- targets
@pytest.mark.xfail(reason="the last piece of the archetype tilt: with no defensive box prior the "
                          "bigs are no longer inflated by rebounds and blocks, but pure on-court "
                          "defensive RAPM still likes them more than the consensus does -- Robert "
                          "Williams 68th against 134th, Jonathan Isaac 55th against 156th",
                   strict=True)
def test_backup_bigs_are_not_in_the_top_twenty(board):
    """Named anchors the consensus puts nowhere near the top. Robert Williams is the one the
    project owner flagged: every modern metric combined has him about 134th."""
    anchors = {"robert williams": 100, "jusuf nurkic": 100, "jonathan isaac": 100,
               "luke kornet": 60, "dayron sharpe": 60, "moussa diabate": 45}
    bad = {k: _rank(board, k) for k, floor in anchors.items()
           if k in set(board.key) and _rank(board, k) < floor}
    assert not bad, f"ranked far too high: {bad}"
