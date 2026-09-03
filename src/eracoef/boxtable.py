"""Per-game player box counts (from LeagueGameLog) in the 13-feature layout used by BoxExposure.

game_box columns: game_id, player_id, season, phase, fg3m, fg3_miss, fg2m, fg2_miss, ftm, ft_miss,
                  orb, drb, ast, tov, stl, blk, pf
Per-game player possessions come from the stints (design.build_game_poss), not from here.
"""
from __future__ import annotations

import pandas as pd

from .ingest import load_gamelog


def box_from_gamelog(gl: pd.DataFrame) -> pd.DataFrame:
    g = gl
    out = pd.DataFrame({
        "game_id": g["GAME_ID"].astype(str),
        "player_id": g["PLAYER_ID"].astype(int),
        "team_id": g["TEAM_ID"].astype(int),
        "player_name": g["PLAYER_NAME"].astype(str),
        "season": g["season"].astype(int),
        "phase": g["phase"].astype(str),
        "minutes": g["MIN"].fillna(0).astype(float),
        "fg3m": g["FG3M"].fillna(0).astype(float),
        "fg3_miss": (g["FG3A"].fillna(0) - g["FG3M"].fillna(0)).astype(float),
        "fg2m": (g["FGM"].fillna(0) - g["FG3M"].fillna(0)).astype(float),
        "fg2_miss": ((g["FGA"].fillna(0) - g["FG3A"].fillna(0)) - (g["FGM"].fillna(0) - g["FG3M"].fillna(0))).astype(float),
        "ftm": g["FTM"].fillna(0).astype(float),
        "ft_miss": (g["FTA"].fillna(0) - g["FTM"].fillna(0)).astype(float),
        "orb": g["OREB"].fillna(0).astype(float),
        "drb": g["DREB"].fillna(0).astype(float),
        "ast": g["AST"].fillna(0).astype(float),
        "tov": g["TOV"].fillna(0).astype(float),
        "stl": g["STL"].fillna(0).astype(float),
        "blk": g["BLK"].fillna(0).astype(float),
        "pf": g["PF"].fillna(0).astype(float),
        "pts": g["PTS"].fillna(0).astype(float),
    })
    return out


def season_box(seasons, phases, cfg) -> pd.DataFrame:
    parts = [box_from_gamelog(load_gamelog(s, p, cfg)) for s in seasons for p in phases]
    return pd.concat(parts, ignore_index=True)


def player_names(box: pd.DataFrame, pbp_names: pd.DataFrame | None = None) -> pd.DataFrame:
    """player_id, season -> display name as of that season.

    LeagueGameLog returns players' current names (Enes Freedom, Nene Hilario, Metta Sandiford-Artest).
    When the play-by-play of that season wrote a different family name, use first name + play-by-play name.
    """
    cur = (box.groupby(["player_id", "season"])["player_name"].agg(lambda s: s.value_counts().index[0]).reset_index())
    if pbp_names is None or len(pbp_names) == 0:
        return cur
    import unicodedata

    def norm(s):
        s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
        return "".join(ch for ch in s.lower() if ch.isalpha())

    m = cur.merge(pbp_names[["player_id", "season", "pbp_name"]], on=["player_id", "season"], how="left")
    out = []
    for name, pbp in zip(m["player_name"], m["pbp_name"]):
        if not isinstance(pbp, str) or not pbp.strip():
            out.append(name)
            continue
        toks = name.split()
        first = toks[0]
        fam = " ".join(toks[1:]) if len(toks) > 1 else name
        ptoks = pbp.split()
        # "Marc Morris" / "Ma. Morris": drop a leading token that abbreviates the first name
        if len(ptoks) > 1 and (norm(first).startswith(norm(ptoks[0])) or norm(ptoks[0]).startswith(norm(first))):
            ptoks = ptoks[1:]
        p = " ".join(ptoks)
        if norm(fam) == norm(p) or norm(fam).startswith(norm(p)) or norm(p).startswith(norm(fam)):
            out.append(name)                      # same family name (suffix / accent differences): keep the current name
        elif norm(p) == norm(first):
            out.append(p)                         # "Nene"
        else:
            out.append(f"{first} {p}")            # "Enes Kanter", "Metta World Peace"
    m["player_name"] = out
    return m[["player_id", "season", "player_name"]]
