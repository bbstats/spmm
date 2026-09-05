"""Role inputs for the Simple SPM prior: playing-time share, starts, and age, per player-season.

The role prior (spm.py) pulls every player toward "what a player with this role and age is worth"
before the box score says anything.  Three inputs, all from data already on disk or one request away:

  share    the player's on-floor possessions (both ends) summed over every team he played for,
           divided by ONE full team-season of possessions (the mean of his teams' season totals).
           The denominator counts the games he did not play, so injury lowers the share; a player
           traded mid-season keeps his full-season share because the numerator is summed, not
           averaged.  Capped at `share_cap` (0.9).
  gs_pct   games started / games played.  Starters are each team's first five rows in the V3 box
           file (the rule stints.GameParser already relies on); played = minutes > 0.
  age      from nba_api LeagueDashPlayerBioStats, one request per season, cached under
           data/raw/bio/.  A missing age gets the season median and `age_imputed` = 1.

`build_roles` writes data/cache/roles.parquet (one row per player-season-team); `window_inputs`
blends the per-season inputs down to the design's Z unit the way BoxExposure blends its padding
constants (possession-weighted over the player's seasons in the window).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve
from .design import AWAY_SLOTS, HOME_SLOTS
from .ingest import GAME_PREFIX, TIMEOUT, _retry, game_table, load_gamelog, raw_dir, season_str

INPUTS = ["share", "share2", "gs_pct", "gs_pct2", "age", "age2", "age3"]
RAW_INPUTS = ["share", "gs_pct", "age"]


def parse_minutes(s) -> float:
    """'24:06' -> 24.1; '' / None / NaN -> 0.0."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return 0.0
    s = str(s).strip()
    if not s:
        return 0.0
    if ":" in s:
        m, sec = s.split(":", 1)
        try:
            return float(m) + float(sec) / 60.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def roles_from_boxes(boxes: dict) -> pd.DataFrame:
    """Per (player_id, team_id): games played, games started, minutes, from V3 box frames keyed by game_id.

    Each team's first five rows in a V3 box are its starters (position is not a starter flag).
    """
    parts = []
    for gid, b in boxes.items():
        if len(b) == 0:
            continue
        b = b[["teamId", "personId", "minutes"]].copy()
        b["minutes"] = b["minutes"].map(parse_minutes)
        b["rank"] = b.groupby("teamId").cumcount()
        b["start"] = (b["rank"] < 5).astype(float)
        b["played"] = (b["minutes"] > 0).astype(float)
        parts.append(b[["teamId", "personId", "played", "start", "minutes"]])
    if not parts:
        return pd.DataFrame(columns=["player_id", "team_id", "games", "starts", "minutes"])
    d = pd.concat(parts, ignore_index=True)
    g = d.groupby(["personId", "teamId"], as_index=False).agg(games=("played", "sum"), starts=("start", "sum"),
                                                              minutes=("minutes", "sum"))
    return g.rename(columns={"personId": "player_id", "teamId": "team_id"}).astype(
        {"player_id": np.int64, "team_id": np.int64})


def shares_from_stints(stints: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Per (player_id, team_id): on-floor possessions (both ends), and the team's season total.

    `stints`: game_id, h1..h5, a1..a5, poss_h, poss_a.  `games`: game_id, home_team_id, away_team_id.
    """
    st = stints.merge(games[["game_id", "home_team_id", "away_team_id"]], on="game_id", how="inner")
    both = (st["poss_h"].to_numpy(dtype=float) + st["poss_a"].to_numpy(dtype=float))
    parts = []
    for slots, team_col in ((HOME_SLOTS, "home_team_id"), (AWAY_SLOTS, "away_team_id")):
        team = st[team_col].to_numpy()
        for c in slots:
            parts.append(pd.DataFrame({"player_id": st[c].to_numpy(), "team_id": team, "poss_on": both}))
    p = pd.concat(parts, ignore_index=True).groupby(["player_id", "team_id"], as_index=False)["poss_on"].sum()
    team_poss = pd.concat([pd.DataFrame({"team_id": st["home_team_id"].to_numpy(), "poss": both}),
                           pd.DataFrame({"team_id": st["away_team_id"].to_numpy(), "poss": both})],
                          ignore_index=True).groupby("team_id", as_index=False)["poss"].sum().rename(columns={"poss": "team_poss"})
    return p.merge(team_poss, on="team_id", how="left").astype({"player_id": np.int64, "team_id": np.int64})


def _box_dir(season: int, cfg) -> Path:
    return raw_dir(cfg) / "box" / str(season)


def season_roles(season: int, cfg) -> pd.DataFrame:
    """One season, regular season only: player_id, season, team_id, games, starts, minutes, poss_on, team_poss."""
    bdir = _box_dir(season, cfg)
    boxes = {}
    for f in sorted(bdir.glob(f"{GAME_PREFIX['RS']}*.parquet")):
        boxes[f.stem] = pd.read_parquet(f, columns=["teamId", "personId", "minutes"])
    roles = roles_from_boxes(boxes)
    stints = pd.read_parquet(Path(resolve(cfg, "stints")) / f"{season}_RS.parquet",
                             columns=["game_id", *HOME_SLOTS, *AWAY_SLOTS, "poss_h", "poss_a"])
    games = game_table(load_gamelog(season, "RS", cfg))
    games["game_id"] = games["game_id"].astype(str)
    stints["game_id"] = stints["game_id"].astype(str)
    shares = shares_from_stints(stints, games)
    out = roles.merge(shares, on=["player_id", "team_id"], how="outer")
    out[["games", "starts", "minutes", "poss_on"]] = out[["games", "starts", "minutes", "poss_on"]].fillna(0.0)
    team_poss = shares.drop_duplicates("team_id").set_index("team_id")["team_poss"]
    out["team_poss"] = out["team_poss"].fillna(out["team_id"].map(team_poss))
    out.insert(1, "season", int(season))
    return out


def season_ages(season: int, cfg, force: bool = False) -> pd.DataFrame:
    """player_id, season, age from LeagueDashPlayerBioStats (cached data/raw/bio/{season}_RS.parquet)."""
    path = raw_dir(cfg) / "bio" / f"{season}_RS.parquet"
    if path.exists() and not force:
        df = pd.read_parquet(path)
    else:
        from nba_api.stats.endpoints import leaguedashplayerbiostats

        def pull():
            d = leaguedashplayerbiostats.LeagueDashPlayerBioStats(
                season=season_str(season), season_type_all_star="Regular Season", timeout=TIMEOUT).get_data_frames()[0]
            if len(d) == 0:
                raise ValueError("empty bio stats")
            return d
        df = _retry(pull, what=f"bio {season}")
        df = df.astype({c: str for c in df.columns if df[c].dtype == object})
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    out = pd.DataFrame({"player_id": df["PLAYER_ID"].astype(np.int64), "season": int(season),
                        "age": pd.to_numeric(df["AGE"], errors="coerce")})
    return out.drop_duplicates("player_id")


def build_roles(cfg, seasons=None, force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Every season's roles + ages -> data/cache/roles.parquet (one row per player-season-team)."""
    path = roles_path(cfg)
    if path.exists() and not force:
        return pd.read_parquet(path)
    seasons = list(range(int(cfg["first_season"]), int(cfg["last_season"]) + 1)) if seasons is None else list(seasons)
    parts = []
    for s in seasons:
        r = season_roles(s, cfg)
        a = season_ages(s, cfg)
        r = r.merge(a, on=["player_id", "season"], how="left")
        miss = float(r.age.isna().mean()) if len(r) else 0.0
        parts.append(r)
        if verbose:
            print(f"  roles {s}: {r.player_id.nunique()} players, {len(r)} player-team rows, "
                  f"age missing {miss:.1%}", flush=True)
    out = pd.concat(parts, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def roles_path(cfg) -> Path:
    p = cfg.get("roles", {}).get("cache", "data/cache/roles.parquet")
    return Path(cfg["_root"]) / p if "_root" in cfg else Path(p)


def player_season_inputs(roles: pd.DataFrame, cap: float = 0.9) -> pd.DataFrame:
    """Per player-season: share (summed over teams / one team-season, capped), gs_pct, age (+ age_imputed)."""
    r = roles.copy()
    g = r.groupby(["player_id", "season"], as_index=False).agg(
        games=("games", "sum"), starts=("starts", "sum"), minutes=("minutes", "sum"), poss_on=("poss_on", "sum"),
        team_poss=("team_poss", "mean"), age=("age", "first"))
    g["share"] = np.where(g.team_poss > 0, g.poss_on / np.where(g.team_poss > 0, g.team_poss, 1.0), 0.0).clip(0.0, cap)
    g["gs_pct"] = np.where(g.games > 0, g.starts / np.where(g.games > 0, g.games, 1.0), 0.0).clip(0.0, 1.0)
    med = g.groupby("season")["age"].transform("median")
    g["age_imputed"] = g.age.isna().astype(int)
    g["age"] = g.age.fillna(med).fillna(float(g.age.median()) if g.age.notna().any() else 27.0)
    return g[["player_id", "season", "games", "starts", "minutes", "poss_on", "team_poss", "share", "gs_pct", "age", "age_imputed"]]


def design7(share, gs_pct, age) -> np.ndarray:
    """The Simple SPM design: share, share^2, gs_pct, gs_pct^2, age, age^2, age^3 (no intercept)."""
    s, g, a = (np.asarray(v, dtype=float) for v in (share, gs_pct, age))
    return np.column_stack([s, s ** 2, g, g ** 2, a, a ** 2, a ** 3])


def window_inputs(wd, inputs: pd.DataFrame, cap: float = 0.9) -> pd.DataFrame:
    """The per-season inputs blended to the design's Z unit, aligned to wd.spec.ps_table.

    Possession-weighted over the player's seasons in the window (RS offensive possessions from
    wd.game_poss, the weights the design itself uses); a player-season with no roles row takes the
    season's possession-weighted league mean and is counted in .attrs["n_missing"].
    """
    spec = wd.spec
    psx = spec.psx_table[["psx_idx", "player_id", "season", "ps_idx"]].copy()
    gp = wd.game_poss
    rs = wd.games.loc[wd.games["phase"] == "RS", "game_idx"].to_numpy()
    gp = gp[np.isin(gp["game_idx"], rs)]
    w_psx = np.zeros(len(psx))
    np.add.at(w_psx, gp["psx_idx"].to_numpy(), gp["poss_off"].to_numpy(dtype=float))
    psx["w"] = w_psx
    m = psx.merge(inputs[["player_id", "season", *RAW_INPUTS]], on=["player_id", "season"], how="left")
    missing = m["share"].isna()
    for c in RAW_INPUTS:
        lm = m.groupby("season").apply(
            lambda d: np.average(d[c].fillna(0.0), weights=np.where(d[c].notna(), d.w, 0.0))
            if (d[c].notna() & (d.w > 0)).any() else d[c].mean(), include_groups=False)
        m[c] = m[c].fillna(m["season"].map(lm)).fillna(float(m[c].mean()) if m[c].notna().any() else 0.0)
    m["share"] = m["share"].clip(0.0, cap)
    n_ps = spec.n_ps
    out = np.zeros((n_ps, len(RAW_INPUTS)))
    den = np.zeros(n_ps)
    cnt = np.zeros((n_ps, len(RAW_INPUTS)))
    n = np.zeros(n_ps)
    vals = m[RAW_INPUTS].to_numpy(dtype=float)
    idx = m["ps_idx"].to_numpy()
    ww = m["w"].to_numpy(dtype=float)
    np.add.at(out, idx, vals * ww[:, None])
    np.add.at(den, idx, ww)
    np.add.at(cnt, idx, vals)
    np.add.at(n, idx, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        blend = np.where(den[:, None] > 0, out / np.where(den > 0, den, 1.0)[:, None], cnt / np.maximum(n, 1.0)[:, None])
    res = pd.DataFrame(blend, columns=RAW_INPUTS)
    res.insert(0, "ps_idx", np.arange(n_ps))
    res.insert(1, "player_id", spec.ps_table["player_id"].to_numpy())
    res.attrs["n_missing"] = int(missing.sum())
    res.attrs["n_psx"] = int(len(m))
    return res
