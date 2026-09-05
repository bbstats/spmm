"""Role inputs (src/eracoef/roles.py) on synthetic V3 boxes and stints."""
import numpy as np
import pandas as pd

from eracoef.roles import INPUTS, design7, parse_minutes, player_season_inputs, roles_from_boxes, shares_from_stints

HOME, AWAY = 1610612700, 1610612800


def _box(team_players):
    rows = []
    for t, lst in team_players.items():
        for pid, mins in lst:
            rows.append(dict(teamId=t, personId=pid, minutes=mins))
    return pd.DataFrame(rows)


def _stint(game_id, h, a, poss_h, poss_a):
    d = {"game_id": game_id, "poss_h": poss_h, "poss_a": poss_a}
    d.update({f"h{i + 1}": p for i, p in enumerate(h)})
    d.update({f"a{i + 1}": p for i, p in enumerate(a)})
    return d


def test_parse_minutes():
    assert abs(parse_minutes("24:06") - 24.1) < 1e-9
    assert parse_minutes("") == 0.0 and parse_minutes(None) == 0.0 and parse_minutes(float("nan")) == 0.0
    assert parse_minutes("12") == 12.0


def _world():
    # game 1: HOME (1..7: 1-5 start, 6 plays 10 min, 7 does not play) vs AWAY (11..17 likewise)
    b1 = _box({HOME: [(1, "30:00"), (2, "30:00"), (3, "30:00"), (4, "30:00"), (5, "30:00"), (6, "10:00"), (7, "")],
               AWAY: [(11, "30:00"), (12, "30:00"), (13, "30:00"), (14, "30:00"), (15, "30:00"), (16, "10:00"), (17, "")]})
    # game 2: player 1 has been traded to AWAY and starts there; AWAY is now the home side
    b2 = _box({AWAY: [(11, "30:00"), (12, "30:00"), (13, "30:00"), (14, "30:00"), (1, "30:00"), (15, "10:00"), (16, "")],
               HOME: [(2, "30:00"), (3, "30:00"), (4, "30:00"), (5, "30:00"), (6, "30:00"), (7, "10:00")]})
    boxes = {"G1": b1, "G2": b2}
    games = pd.DataFrame({"game_id": ["G1", "G2"], "home_team_id": [HOME, AWAY], "away_team_id": [AWAY, HOME]})
    st = pd.DataFrame([
        _stint("G1", [1, 2, 3, 4, 5], [11, 12, 13, 14, 15], 10, 10),
        _stint("G1", [2, 3, 4, 5, 6], [11, 12, 13, 14, 16], 10, 10),
        _stint("G2", [11, 12, 13, 14, 1], [2, 3, 4, 5, 6], 10, 10),
        _stint("G2", [11, 12, 13, 14, 1], [3, 4, 5, 6, 7], 10, 10),
    ])
    return boxes, games, st


def test_roles_from_boxes_first_five_start():
    boxes, _, _ = _world()
    r = roles_from_boxes(boxes).set_index(["player_id", "team_id"])
    assert r.loc[(1, HOME), "games"] == 1 and r.loc[(1, HOME), "starts"] == 1
    assert r.loc[(1, AWAY), "games"] == 1 and r.loc[(1, AWAY), "starts"] == 1      # the traded player has two rows
    assert r.loc[(6, HOME), "games"] == 2 and r.loc[(6, HOME), "starts"] == 1      # bench in G1, sixth row = starter in G2? no: 6th row
    assert r.loc[(7, HOME), "games"] == 1 and r.loc[(7, HOME), "starts"] == 0      # DNP in G1, played in G2
    assert (r.starts <= r.games).all()


def test_shares_and_inputs():
    boxes, games, st = _world()
    s = shares_from_stints(st, games).set_index(["player_id", "team_id"])
    # every stint is 20 possessions both ends; each team has two games of two stints -> 80
    assert s.loc[(2, HOME), "team_poss"] == 80 and s.loc[(11, AWAY), "team_poss"] == 80
    assert s.loc[(1, HOME), "poss_on"] == 20 and s.loc[(1, AWAY), "poss_on"] == 40
    assert s.loc[(11, AWAY), "poss_on"] == 80                                     # on the floor for everything
    assert (s.poss_on <= s.team_poss).all()
    roles = roles_from_boxes(boxes).merge(s.reset_index(), on=["player_id", "team_id"], how="outer")
    roles[["games", "starts", "minutes", "poss_on"]] = roles[["games", "starts", "minutes", "poss_on"]].fillna(0.0)
    roles["season"] = 2001
    roles["age"] = np.where(roles.player_id == 11, 30.0, np.nan)                  # only one age known
    i = player_season_inputs(roles, cap=0.9).set_index("player_id")
    assert abs(i.loc[11, "share"] - 0.9) < 1e-12                                  # 80 / 80 capped
    assert abs(i.loc[1, "share"] - 60 / 80) < 1e-12                               # summed over both teams
    assert abs(i.loc[2, "share"] - 60 / 80) < 1e-12                               # on the floor for 3 of the 4 stints
    assert i.loc[6, "gs_pct"] == 0.5 and i.loc[7, "gs_pct"] == 0.0 and i.loc[1, "gs_pct"] == 1.0
    assert i.loc[11, "age_imputed"] == 0 and i.loc[1, "age_imputed"] == 1 and i.loc[1, "age"] == 30.0
    assert list(i.columns[-4:]) == ["share", "gs_pct", "age", "age_imputed"]


def test_design7():
    X = design7([0.5, 0.2], [1.0, 0.0], [25.0, 33.0])
    assert X.shape == (2, 7) and len(INPUTS) == 7
    assert X[0, 1] == 0.25 and X[1, 6] == 33.0 ** 3
