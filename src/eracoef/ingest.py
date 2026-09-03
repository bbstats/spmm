"""Raw data pulls, cached per game / per season as parquet. Never re-download.

Sources (stats.nba.com via nba_api):
  - LeagueGameLog (player rows), one call per season and phase -> game list + per-game box counts
  - PlayByPlayV3 per game (V2 is dead: the API returns empty JSON)
  - BoxScoreTraditionalV3 per game (roster, starters, minutes) and, on demand, per period
    (range_type=2) to resolve period starters the play-by-play cannot identify
  - api.pbpstats.com get-games per season (possessions per team-game) as a cross-check, 2000-01+

Layout under cfg paths.raw:
  gamelog/{season}_{phase}.parquet
  pbp/{season}/{game_id}.parquet
  box/{season}/{game_id}.parquet
  periodbox/{season}/{game_id}_{period}.parquet
  pbpstats_games/{season}_{phase}.parquet
"""
from __future__ import annotations

import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import resolve

warnings.filterwarnings("ignore", category=DeprecationWarning)

SEASON_TYPES = {"RS": "Regular Season", "PO": "Playoffs"}
GAME_PREFIX = {"RS": "002", "PO": "004"}
TIMEOUT = 60
HEADERS_PBPSTATS = {"User-Agent": "Mozilla/5.0"}


def season_str(season: int) -> str:
    """2014 -> '2013-14'."""
    return f"{season - 1}-{str(season)[-2:]}"


def season_of_game(game_id: str) -> int:
    yy = int(game_id[3:5])
    return (1900 + yy + 1) if yy >= 90 else (2000 + yy + 1)


def raw_dir(cfg) -> Path:
    return resolve(cfg, "raw")


def _retry(fn, tries=4, base_sleep=5.0, what=""):
    last = None
    for t in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(base_sleep * (3 ** t))
    raise RuntimeError(f"failed after {tries} tries: {what}: {last!r}")


# ----------------------------------------------------------------------------- game logs
def load_gamelog(season: int, phase: str, cfg, force=False) -> pd.DataFrame:
    """Player-level game log for a season and phase (cached)."""
    path = raw_dir(cfg) / "gamelog" / f"{season}_{phase}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    from nba_api.stats.endpoints import leaguegamelog
    def pull():
        return leaguegamelog.LeagueGameLog(season=season_str(season), season_type_all_star=SEASON_TYPES[phase],
                                           player_or_team_abbreviation="P", timeout=TIMEOUT).get_data_frames()[0]
    gl = _retry(pull, what=f"gamelog {season} {phase}")
    gl = gl[gl["GAME_ID"].str.startswith(GAME_PREFIX[phase])].copy()   # drop play-in etc.
    gl["season"] = season
    gl["phase"] = phase
    path.parent.mkdir(parents=True, exist_ok=True)
    gl.to_parquet(path, index=False)
    return gl


def game_table(gl: pd.DataFrame) -> pd.DataFrame:
    """One row per game: game_id, game_date, home_team_id, away_team_id, season, phase."""
    g = gl[["GAME_ID", "GAME_DATE", "TEAM_ID", "MATCHUP", "season", "phase"]].drop_duplicates(["GAME_ID", "TEAM_ID"])
    g = g.assign(is_home=g["MATCHUP"].str.contains(" vs. "))
    # neutral-site games (NBA Cup knockout rounds, international games) list both teams as "@" (or both "vs."):
    # take the lower team id as the nominal home side and flag the game neutral (home effect = 0 in the design)
    n_home = g.groupby("GAME_ID")["is_home"].transform("sum")
    amb = n_home != 1
    g.loc[amb, "is_home"] = g.loc[amb, "TEAM_ID"] == g.loc[amb].groupby("GAME_ID")["TEAM_ID"].transform("min")
    home = g[g.is_home].set_index("GAME_ID")["TEAM_ID"]
    away = g[~g.is_home].set_index("GAME_ID")["TEAM_ID"]
    meta = g.drop_duplicates("GAME_ID").set_index("GAME_ID")[["GAME_DATE", "season", "phase"]]
    neutral = g.groupby("GAME_ID").apply(lambda d: bool((d["MATCHUP"].str.contains(" vs. ").sum()) != 1)).rename("neutral")
    out = meta.join(home.rename("home_team_id")).join(away.rename("away_team_id")).join(neutral).reset_index()
    out = out.rename(columns={"GAME_ID": "game_id", "GAME_DATE": "game_date"})
    out["game_date"] = pd.to_datetime(out["game_date"])
    return out.sort_values(["game_date", "game_id"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- per-game pulls
def _pbp_path(game_id, cfg):
    return raw_dir(cfg) / "pbp" / str(season_of_game(game_id)) / f"{game_id}.parquet"


def _box_path(game_id, cfg):
    return raw_dir(cfg) / "box" / str(season_of_game(game_id)) / f"{game_id}.parquet"


def _pbox_path(game_id, period, cfg):
    return raw_dir(cfg) / "periodbox" / str(season_of_game(game_id)) / f"{game_id}_{period}.parquet"


def fetch_pbp(game_id: str, cfg, force=False) -> pd.DataFrame:
    path = _pbp_path(game_id, cfg)
    if path.exists() and not force:
        return pd.read_parquet(path)
    from nba_api.stats.endpoints import playbyplayv3
    def pull():
        df = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=TIMEOUT).get_data_frames()[0]
        if len(df) == 0:
            raise ValueError("empty pbp")
        return df
    df = _retry(pull, what=f"pbp {game_id}")
    df = df.astype({c: str for c in df.columns if df[c].dtype == object})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def fetch_boxscore(game_id: str, cfg, force=False) -> pd.DataFrame:
    path = _box_path(game_id, cfg)
    if path.exists() and not force:
        return pd.read_parquet(path)
    from nba_api.stats.endpoints import boxscoretraditionalv3
    def pull():
        df = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, timeout=TIMEOUT).get_data_frames()[0]
        if len(df) == 0:
            raise ValueError("empty boxscore")
        return df
    df = _retry(pull, what=f"box {game_id}")
    df = df.astype({c: str for c in df.columns if df[c].dtype == object})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def period_range(period: int):
    if period <= 4:
        return (period - 1) * 7200, period * 7200
    start = 28800 + (period - 5) * 3000
    return start, start + 3000


def fetch_period_box(game_id: str, period: int, cfg, force=False) -> pd.DataFrame:
    """Players with minutes in one period (BoxScoreTraditionalV3 with range_type=2)."""
    path = _pbox_path(game_id, period, cfg)
    if path.exists() and not force:
        return pd.read_parquet(path)
    from nba_api.stats.endpoints import boxscoretraditionalv3
    s, e = period_range(period)
    def pull():
        return boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, start_period=period, end_period=period,
                                                           range_type=2, start_range=s, end_range=e,
                                                           timeout=TIMEOUT).get_data_frames()[0]
    df = _retry(pull, what=f"periodbox {game_id} p{period}")
    df = df.astype({c: str for c in df.columns if df[c].dtype == object})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def pbpstats_games(season: int, phase: str, cfg, force=False) -> pd.DataFrame:
    """Per-game possessions from api.pbpstats.com (cross-check only; empty before 2000-01)."""
    path = raw_dir(cfg) / "pbpstats_games" / f"{season}_{phase}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    def pull():
        r = requests.get("https://api.pbpstats.com/get-games/nba",
                         params={"Season": season_str(season), "SeasonType": SEASON_TYPES[phase]},
                         headers=HEADERS_PBPSTATS, timeout=TIMEOUT)
        r.raise_for_status()
        return pd.DataFrame(r.json().get("results", []))
    df = _retry(pull, what=f"pbpstats games {season} {phase}")
    if len(df) == 0:
        df = pd.DataFrame(columns=["GameId", "Date", "HomeTeamId", "AwayTeamId", "HomePoints", "AwayPoints",
                                   "HomePossessions", "AwayPossessions"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


# ----------------------------------------------------------------------------- season driver
def ingest_season(season: int, phases=("RS", "PO"), cfg=None, workers=2, delay=0.6, log_every=100, verbose=True) -> dict:
    """Pull game logs, then PBP + boxscore for every game not yet cached. Idempotent."""
    summary = {}
    for phase in phases:
        gl = load_gamelog(season, phase, cfg)
        games = game_table(gl)
        try:
            pbpstats_games(season, phase, cfg)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  pbpstats games {season} {phase}: {e!r}")
        todo = [g for g in games["game_id"] if not (_pbp_path(g, cfg).exists() and _box_path(g, cfg).exists())]
        if verbose:
            print(f"{season} {phase}: {len(games)} games, {len(todo)} to fetch")
        t0 = time.time()
        failed = []

        def work(gid):
            fetch_pbp(gid, cfg)
            time.sleep(delay)
            fetch_boxscore(gid, cfg)
            time.sleep(delay)
            return gid

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(work, g): g for g in todo}
            for f in as_completed(futs):
                done += 1
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001
                    failed.append((futs[f], repr(e)[:200]))
                if verbose and done % log_every == 0:
                    el = time.time() - t0
                    print(f"  {done}/{len(todo)}  {el:.0f}s  ({el / done:.2f}s/game)")
        summary[phase] = dict(n_games=len(games), fetched=len(todo), failed=failed, seconds=time.time() - t0)
        if verbose:
            print(f"  done {phase}: {len(todo)} fetched in {time.time() - t0:.0f}s, {len(failed)} failed")
    return summary
