"""Play-by-play (stats.nba.com V3) -> possessions with lineups -> stints.

Possession ends on: made FG (unless an and-1 free throw follows), defensive rebound,
turnover, made final free throw of a regular trip, end of period.  Offensive rebounds
continue the possession.  Technical, flagrant and clear-path free throws do not end a
possession.  A possession is attributed to the ten players on the floor when it ends.

Lineups: period starters are inferred from the play-by-play (players who act before
being substituted in); when fewer than five are found, the period box score
(BoxScoreTraditionalV3, range_type=2) supplies the players with minutes in the period.
Substitutions in V3 carry only the outgoing player's id, so the incoming player is
resolved by name against the team's box-score roster, with a look-ahead to the next
event involving a candidate when the name is ambiguous.

Stint = consecutive possessions in one game with the same ten players, split at every
substitution and at period ends.
"""
from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve
from .design import AWAY_SLOTS, HOME_SLOTS
from .ingest import fetch_boxscore, fetch_pbp, fetch_period_box, game_table, load_gamelog, pbpstats_games

CLOCK_RE = re.compile(r"PT(\d+)M(\d+(?:\.\d+)?)S")
SUB_RE = re.compile(r"SUB:\s*(.+?)\s+FOR\s+(.+)$")
TIP_RE = re.compile(r"Tip to\s+(.+)$")
FT_RE = re.compile(r"Free Throw(?: (\w[\w ]*?))? (\d) of (\d)")
OFFENSIVE_ACTIONS = {"Made Shot", "Missed Shot", "Turnover", "Heave"}
IGNORED = {"", "Timeout", "Instant Replay", "Violation", "Ejection"}


def parse_clock(s: str) -> float:
    m = CLOCK_RE.match(str(s))
    if not m:
        return np.nan
    return int(m.group(1)) * 60 + float(m.group(2))


def regulation_remaining(period: int, clock: float) -> float:
    return (4 - period) * 720.0 + clock if period <= 4 else 0.0


# A description token that ends the player's name: a number, a parenthesis, or one of these whole words.
STOP_TOKEN = re.compile(r"^(\d|\(|3PT|[A-Z](\.[A-Z])*\.FOUL$|T\.Foul$|OFF\.|FLAGRANT|"
                        r"(REBOUND|Free|Turnover|STEAL|BLOCK|Bad|Lost|Traveling|Offensive|Out|Step|Shot|Jump|Layup|Dunk|Hook|Tip|"
                        r"Running|Driving|Turnaround|Fadeaway|Floating|Pullup|Putback|Alley|Finger|Reverse|Cutting|Slam|Bank|"
                        r"Violation|Personal|Shooting|Loose|Away|Technical|Flagrant|No|Kicked|Def\.|Foul|Double|Inbound|Lane|"
                        r"Backcourt|Palming|Illegal|Discontinue|Punched|Poss|Hanging|Taunting|Excess|Delay|Jumper|Regular|"
                        r"Shooting|Clear|Transition|Elbow|Charge|Blocking|Defensive|Inbound)$)", re.IGNORECASE)


def _leading_name(descriptions) -> str:
    """The player's name as written in this game's play-by-play: the words before the action keyword."""
    from collections import Counter
    names = []
    for d in descriptions:
        out = []
        for tk in str(d).split():
            if STOP_TOKEN.match(tk):
                break
            out.append(tk)
        if out:
            names.append(" ".join(out))
    if not names:
        return ""
    return Counter(names).most_common(1)[0][0]


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


class GameParser:
    """Turn one game's V3 play-by-play into possessions with lineups."""

    def __init__(self, pbp: pd.DataFrame, box: pd.DataFrame, home_id: int, away_id: int, game_id: str,
                 period_box_fetcher=None, gt_rule=None):
        self.game_id = game_id
        self.home, self.away = int(home_id), int(away_id)
        self.teams = (self.home, self.away)
        self.fetch_period_box = period_box_fetcher
        self.gt_rule = gt_rule or {}
        self.diag = dict(unresolved_subs=0, sub_out_not_on_floor=0, forced_switches=0, jump_switches=0,
                         periods_from_box=0, periods_starters_failed=0, invalid_possessions=0, n_periods=0)
        # roster
        self.roster = {}                 # pid -> team
        self.names = {t: {} for t in self.teams}   # team -> norm family name -> [pid]
        self.name_i = {}                 # pid -> normalized "F. Family"
        self.box_minutes = {}
        for r in box.itertuples(index=False):
            pid, team = int(r.personId), int(r.teamId)
            self.roster[pid] = team
            fam = _norm(r.familyName)
            self.names.setdefault(team, {}).setdefault(fam, []).append(pid)
            self.name_i[pid] = _norm(r.nameI)
            self.box_minutes[pid] = str(r.minutes)
        # V3 lists each team's starters as its first five rows (the position field is not a starter flag)
        self.box_starters = {t: set() for t in self.teams}
        for t in self.teams:
            rows_t = box[box.teamId.astype(int) == t]
            self.box_starters[t] = set(rows_t["personId"].astype(int).head(5).tolist())
        self.box_pts = {}
        for t in self.teams:
            self.box_pts[t] = float(box.loc[box.teamId.astype(int) == t, "points"].astype(float).sum()) if "points" in box.columns else np.nan
        # events
        ev = pbp.copy()
        for c in ("actionType", "subType", "description", "clock", "playerName"):
            ev[c] = ev[c].astype(str).str.strip()      # some seasons pad these with trailing spaces
        ev["clock_s"] = ev["clock"].map(parse_clock)
        for c in ("teamId", "personId", "period", "shotValue"):
            ev[c] = pd.to_numeric(ev[c], errors="coerce").fillna(0).astype(int)
        for c in ("scoreHome", "scoreAway"):
            ev[c] = pd.to_numeric(ev[c], errors="coerce").fillna(0).astype(int)
        self.ev = ev.reset_index(drop=True)
        # names as written in this game's play-by-play (the box score carries players' current names,
        # the play-by-play the names of the day: Nene vs Hilario, Kanter vs Freedom, ...)
        self.pbp_names = {t: {} for t in self.teams}
        descs = {}
        for pid, at, desc in zip(self.ev["personId"], self.ev["actionType"], self.ev["description"]):
            pid = int(pid)
            if self.roster.get(pid) in self.teams and at in ("Made Shot", "Missed Shot", "Rebound", "Turnover", "Foul", "Free Throw", ""):
                d = str(desc)
                if d.startswith("MISS "):
                    d = d[5:]
                descs.setdefault(pid, []).append(d)
        self.day_names = {}      # pid -> name as written in this game's play-by-play
        for pid, ds in descs.items():
            t = self.roster[pid]
            name = _leading_name(ds)
            if name:
                self.day_names[pid] = name
                self.pbp_names[t].setdefault(_norm(name), set()).add(pid)
                last = _norm(name.split()[-1])
                if last != _norm(name):
                    self.pbp_names[t].setdefault(last, set()).add(pid)

    # ---------------------------------------------------------------- helpers
    def _row_team(self, r) -> int:
        if r.teamId in self.teams:
            return r.teamId
        if r.personId in self.teams:
            return r.personId
        return 0

    def _other(self, t):
        return self.away if t == self.home else self.home

    def _player_team(self, pid):
        return self.roster.get(pid, 0)

    def _resolve_in(self, team, in_name, on_floor, i, period_rows):
        """Incoming player id for 'SUB: <in_name> FOR <out>' on `team`."""
        key = _norm(in_name)
        cands = list(self.pbp_names.get(team, {}).get(key, []))
        if not cands:
            cands = list(self.names.get(team, {}).get(key, []))
        if not cands:
            # tolerate suffixes / partial matches
            for fam, pids in self.names.get(team, {}).items():
                if key.endswith(fam) or fam.endswith(key):
                    cands.extend(pids)
        cands = [c for c in cands if c not in on_floor]
        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1 and "." in in_name:
            ini = _norm(in_name)
            narrowed = [c for c in cands if self.name_i.get(c, "") == ini]
            if len(narrowed) == 1:
                return narrowed[0]
        if not cands:
            # no name match (spelling drift): closest roster name not on the floor, if it is close enough
            import difflib
            pool = {fam: pids for fam, pids in self.names.get(team, {}).items() if any(p not in on_floor for p in pids)}
            best = difflib.get_close_matches(key, list(pool), n=1, cutoff=0.75)
            if best:
                cands = [p for p in pool[best[0]] if p not in on_floor]
        if len(cands) > 1:
            # look ahead within the period for the first event involving a candidate
            for r in period_rows[i + 1:]:
                if r.personId in cands and r.actionType != "Substitution":
                    return r.personId
                if r.actionType == "Substitution" and r.personId in cands:
                    return r.personId
        if len(cands) == 1:
            return cands[0]
        return None

    def _starters_from_pbp(self, rows):
        starters = {t: [] for t in self.teams}
        entered = {t: set() for t in self.teams}
        for r in rows:
            at = r.actionType
            if at == "Substitution":
                t = self._player_team(r.personId)
                if t in self.teams and r.personId not in entered[t] and r.personId not in starters[t]:
                    starters[t].append(r.personId)
                m = SUB_RE.match(str(r.description))
                if m and t in self.teams:
                    pid_in = self._resolve_in(t, m.group(1), set(starters[t]) | entered[t], 0, [])
                    if pid_in is not None:
                        entered[t].add(pid_in)
                continue
            if at in IGNORED or at == "period":
                continue
            if at == "Foul" and "Technical" in str(r.subType):
                continue
            if at == "Free Throw" and "Technical" in str(r.subType):
                continue
            pid = r.personId
            t = self._player_team(pid)
            if t in self.teams and pid not in entered[t] and pid not in starters[t]:
                starters[t].append(pid)
        return starters, entered

    def _starters(self, period, rows):
        starters, entered = self._starters_from_pbp(rows)
        out = {}
        for t in self.teams:
            s = starters[t]
            if len(s) == 5:
                out[t] = set(s)
                continue
            if period == 1 and len(self.box_starters[t]) == 5 and set(s) <= self.box_starters[t]:
                out[t] = set(self.box_starters[t])
                continue
            if len(s) != 5 and self.fetch_period_box is not None:
                self.diag["periods_from_box"] += 1
                pb = self.fetch_period_box(self.game_id, period)
                played = set(int(p) for p, tm, mn in zip(pb.personId, pb.teamId, pb.minutes)
                             if int(tm) == t and str(mn) not in ("", "0:00", "None", "nan"))
                if len(s) > 5 and len(set(s) & played) == 5:
                    out[t] = set(s) & played
                    continue
                first_in = set()
                first_out = set()
                for r in rows:
                    if r.actionType != "Substitution":
                        continue
                    m = SUB_RE.match(str(r.description))
                    if self._player_team(r.personId) == t and r.personId not in first_in:
                        first_out.add(r.personId)
                    if m:
                        pid_in = self._resolve_in(t, m.group(1), set(), 0, []) if self._player_team(r.personId) == t else None
                        if pid_in is not None and pid_in not in first_out:
                            first_in.add(pid_in)
                cand = (played - first_in) | set(s)
                if len(cand) == 5:
                    out[t] = cand
                    continue
            self.diag["periods_starters_failed"] += 1
            out[t] = None
        return out

    # ---------------------------------------------------------------- main loop
    def possessions(self) -> pd.DataFrame:
        recs = []
        score = {self.home: 0, self.away: 0}
        pending = {self.home: 0, self.away: 0}
        for period, grp in self.ev.groupby("period", sort=True):
            self.diag["n_periods"] += 1
            rows = list(grp.itertuples(index=False))
            starters = self._starters(period, rows)
            on = {t: (set(starters[t]) if starters[t] is not None else set()) for t in self.teams}
            valid = {t: starters[t] is not None for t in self.teams}
            offense = None
            prev_offense = None
            active = False
            pts = 0
            and1 = False
            start_clock = 720.0 if period <= 4 else 300.0
            start_score = dict(score)

            def close(reason, end_clock, next_offense):
                nonlocal offense, active, pts, and1, start_clock, start_score
                if offense is not None and active:
                    p = pts + pending[offense]
                    pending[offense] = 0
                    ok = valid[self.home] and valid[self.away] and len(on[self.home]) == 5 and len(on[self.away]) == 5
                    if not ok:
                        self.diag["invalid_possessions"] += 1
                    recs.append(dict(period=period, offense=offense, points=p, start_clock=start_clock, end_clock=end_clock,
                                     score_home=start_score[self.home], score_away=start_score[self.away],
                                     home_lineup=tuple(sorted(on[self.home])), away_lineup=tuple(sorted(on[self.away])),
                                     valid=ok, reason=reason))
                offense = next_offense
                active = False
                pts = 0
                and1 = False
                start_clock = end_clock
                start_score = dict(score)

            def ensure_offense(team, i):
                """An offensive action by `team`: switch (closing the current possession) if needed."""
                nonlocal offense, active
                if offense is None:
                    offense = team
                    if prev_offense is not None and prev_offense != team:
                        self.diag["jump_switches"] += 1
                elif offense != team:
                    self.diag["forced_switches"] += 1
                    close("forced", rows[i].clock_s, team)
                active = True

            for i, r in enumerate(rows):
                at = r.actionType
                if at in IGNORED:
                    continue
                if at == "period":
                    if r.subType == "end":
                        close("period_end", 0.0, None)
                        # technical free throws made after a team's last possession: credit that team's last possession
                        for t in self.teams:
                            if pending[t] > 0:
                                for rec in reversed(recs):
                                    if rec["offense"] == t:
                                        rec["points"] += pending[t]
                                        pending[t] = 0
                                        break
                    continue
                if at == "Substitution":
                    t = self._player_team(r.personId)
                    if t not in self.teams:
                        continue
                    m = SUB_RE.match(str(r.description))
                    pid_in = self._resolve_in(t, m.group(1), on[t], i, rows) if m else None
                    if r.personId in on[t]:
                        on[t].discard(r.personId)
                    else:
                        self.diag["sub_out_not_on_floor"] += 1
                        valid[t] = False
                    if pid_in is None:
                        self.diag["unresolved_subs"] += 1
                        valid[t] = False
                    else:
                        on[t].add(pid_in)
                        if len(on[t]) != 5:
                            valid[t] = False
                        elif not valid[t] and starters[t] is not None:
                            valid[t] = True   # self-healed back to five known players
                    continue
                if at == "Jump Ball":
                    m = TIP_RE.search(str(r.description))
                    tip_team = None
                    if m:
                        key = _norm(m.group(1))
                        hits = [pid for t in self.teams for pid in self.names.get(t, {}).get(key, [])]
                        teams_hit = {self._player_team(p) for p in hits}
                        if len(teams_hit) == 1:
                            tip_team = teams_hit.pop()
                    if tip_team is not None and offense is not None and tip_team != offense and active:
                        self.diag["jump_switches"] += 1
                        close("jump_ball", r.clock_s, tip_team)
                    elif offense is None and tip_team is not None:
                        offense = tip_team
                    continue
                team = self._row_team(r)
                if at in ("Made Shot", "Missed Shot", "Heave"):
                    if team not in self.teams:
                        continue
                    ensure_offense(team, i)
                    if at == "Made Shot":
                        v = int(r.shotValue) if int(r.shotValue) in (2, 3) else 2
                        score[team] += v
                        pts += v
                        # and-1: a shooting foul on the defense at the same clock followed by a 1-of-1 free throw
                        deferred = False
                        for r2 in rows[i + 1:i + 12]:
                            if r2.clock_s != r.clock_s or r2.actionType in ("Made Shot", "Missed Shot", "Turnover", "period"):
                                break
                            if r2.actionType == "Rebound" and self._row_team(r2) == self._other(team):
                                break
                            if r2.actionType == "Free Throw" and self._row_team(r2) == team and "Technical" not in str(r2.subType):
                                deferred = True
                                break
                        if deferred:
                            and1 = True
                        else:
                            close("made_fg", r.clock_s, self._other(team))
                    continue
                if at == "Turnover":
                    if team not in self.teams:
                        continue
                    ensure_offense(team, i)
                    close("turnover", r.clock_s, self._other(team))
                    continue
                if at == "Rebound":
                    if team not in self.teams:
                        continue
                    if offense is None:
                        offense = team
                        active = True
                        continue
                    if team == offense:
                        continue      # offensive rebound: possession continues
                    close("dreb", r.clock_s, team)
                    active = True     # the rebounding team now has the ball
                    continue
                if at == "Free Throw":
                    if team not in self.teams:
                        continue
                    st = str(r.subType)
                    desc = str(r.description)
                    # made free throws carry "(N PTS)"; some misses have a blank description with no "MISS"
                    made = ("PTS" in desc) or (not desc.startswith("MISS") and (r.scoreHome + r.scoreAway) > 0)
                    technical = "Technical" in st
                    retain = ("Flagrant" in st) or ("Clear Path" in st)
                    if technical:
                        if made:
                            score[team] += 1
                            if offense == team and active:
                                pts += 1
                            else:
                                pending[team] += 1   # credited to the team's next (or, at period end, last) possession
                        continue
                    ensure_offense(team, i)
                    if made:
                        score[team] += 1
                        pts += 1
                    m = FT_RE.search(st)
                    if retain or not m:
                        continue
                    k, n = int(m.group(2)), int(m.group(3))
                    if k == n and made:
                        close("made_ft", r.clock_s, self._other(team))
                    continue
                # fouls and anything else: no possession effect
            prev_offense = None
        df = pd.DataFrame(recs)
        return df

    # ---------------------------------------------------------------- stints
    def stints(self, poss: pd.DataFrame | None = None) -> pd.DataFrame:
        if poss is None:
            poss = self.possessions()
        if len(poss) == 0:
            return pd.DataFrame()
        gt = self.gt_rule
        key = poss["period"].astype(str) + "|" + poss["home_lineup"].astype(str) + "|" + poss["away_lineup"].astype(str)
        new = (key != key.shift()) | (~poss["valid"]) | (~poss["valid"].shift(fill_value=True))
        sid = new.cumsum()
        rows = []
        for _, g in poss[poss["valid"]].groupby(sid[poss["valid"]]):
            first = g.iloc[0]
            margin_h = int(first["score_home"] - first["score_away"])
            clock = float(first["start_clock"])
            period = int(first["period"])
            frac_rem = regulation_remaining(period, clock) / 2880.0
            is_gt = False
            if period == 4:
                if abs(margin_h) >= gt.get("q4_margin_any", 20):
                    is_gt = True
                if abs(margin_h) >= gt.get("q4_margin_late", 15) and clock < 60 * gt.get("late_minutes", 6):
                    is_gt = True
            hl, al = first["home_lineup"], first["away_lineup"]
            rec = dict(period=period, poss_h=int((g["offense"] == self.home).sum()), poss_a=int((g["offense"] == self.away).sum()),
                       pts_h=float(g.loc[g["offense"] == self.home, "points"].sum()), pts_a=float(g.loc[g["offense"] == self.away, "points"].sum()),
                       margin_h=margin_h, frac_rem=frac_rem, is_gt=is_gt, start_clock=clock)
            rec.update({c: int(p) for c, p in zip(HOME_SLOTS, hl)})
            rec.update({c: int(p) for c, p in zip(AWAY_SLOTS, al)})
            rows.append(rec)
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- season driver
def build_game(game_id: str, home_id: int, away_id: int, cfg, gt_rule=None):
    pbp = fetch_pbp(game_id, cfg)
    box = fetch_boxscore(game_id, cfg)
    fetcher = lambda gid, p: fetch_period_box(gid, p, cfg)  # noqa: E731
    gp = GameParser(pbp, box, home_id, away_id, game_id, period_box_fetcher=fetcher, gt_rule=gt_rule)
    poss = gp.possessions()
    st = gp.stints(poss)
    d = dict(gp.diag)
    d.update(game_id=game_id, n_poss=len(poss), n_poss_home=int((poss["offense"] == gp.home).sum()) if len(poss) else 0,
             n_poss_away=int((poss["offense"] == gp.away).sum()) if len(poss) else 0,
             n_valid_poss=int(poss["valid"].sum()) if len(poss) else 0,
             pts_home=float(poss.loc[poss["offense"] == gp.home, "points"].sum()) if len(poss) else 0.0,
             pts_away=float(poss.loc[poss["offense"] == gp.away, "points"].sum()) if len(poss) else 0.0,
             box_pts_home=gp.box_pts.get(gp.home, np.nan), box_pts_away=gp.box_pts.get(gp.away, np.nan),
             n_stints=len(st))
    names = pd.DataFrame({"player_id": list(gp.day_names), "pbp_name": list(gp.day_names.values())})
    return st, d, names


def build_season(season: int, phase: str, cfg, force=False, verbose=True):
    """Stints for one season and phase -> data/stints/{season}_{phase}.parquet (+ _diag.parquet)."""
    out_dir = resolve(cfg, "stints")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{season}_{phase}.parquet"
    dpath = out_dir / f"{season}_{phase}_diag.parquet"
    if path.exists() and dpath.exists() and not force:
        return pd.read_parquet(path), pd.read_parquet(dpath)
    gl = load_gamelog(season, phase, cfg)
    games = game_table(gl)
    gt_rule = cfg.get("garbage_time", {})
    neutral_seasons = set(cfg.get("neutral_site_seasons_po", []))
    parts, diags, name_parts = [], [], []
    t0 = time.time()
    for k, g in enumerate(games.itertuples(index=False)):
        try:
            st, d, nm = build_game(g.game_id, g.home_team_id, g.away_team_id, cfg, gt_rule)
        except Exception as e:  # noqa: BLE001
            diags.append(dict(game_id=g.game_id, error=repr(e)[:300]))
            continue
        name_parts.append(nm)
        if len(st):
            st["game_id"] = g.game_id
            st["season"] = season
            st["phase"] = phase
            st["game_date"] = g.game_date
            st["series_id"] = f"{season}-{min(g.home_team_id, g.away_team_id)}-{max(g.home_team_id, g.away_team_id)}" if phase == "PO" else ""
            st["neutral"] = bool((phase == "PO" and season in neutral_seasons) or bool(getattr(g, "neutral", False)))
            parts.append(st)
        d["home_team_id"] = g.home_team_id
        d["away_team_id"] = g.away_team_id
        diags.append(d)
        if verbose and (k + 1) % 100 == 0:
            print(f"  {season} {phase}: {k + 1}/{len(games)} games  {time.time() - t0:.0f}s")
    stints = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    diag = pd.DataFrame(diags)
    diag["seconds_total"] = time.time() - t0
    stints.to_parquet(path, index=False)
    diag.to_parquet(dpath, index=False)
    if name_parts:
        nm = pd.concat(name_parts, ignore_index=True)
        nm = nm.groupby("player_id")["pbp_name"].agg(lambda s: s.value_counts().index[0]).reset_index()
        nm["season"] = season
        nm.to_parquet(out_dir / f"{season}_{phase}_names.parquet", index=False)
    return stints, diag


def season_names(season: int, phase: str, cfg) -> pd.DataFrame:
    """player_id, season, pbp_name: the player's name as written in that season's play-by-play."""
    p = resolve(cfg, "stints") / f"{season}_{phase}_names.parquet"
    if not p.exists():
        build_season(season, phase, cfg)
    return pd.read_parquet(p)


def season_diagnostics(season: int, phase: str, cfg) -> dict:
    """Per-season summary: lineup validity, possessions per game vs pbpstats, player-poss identity, points check."""
    st, diag = build_season(season, phase, cfg)
    out = dict(season=season, phase=phase, n_games=int(diag["game_id"].nunique()), n_stints=len(st))
    if "error" in diag.columns:
        out["games_failed"] = int(diag["error"].notna().sum())
    d = diag[diag.get("error", pd.Series(index=diag.index, dtype=object)).isna()] if "error" in diag.columns else diag
    out["stints_per_game"] = len(st) / max(out["n_games"], 1)
    out["poss_per_team_game"] = float((d["n_poss_home"] + d["n_poss_away"]).mean() / 2)
    out["valid_poss_frac"] = float(d["n_valid_poss"].sum() / d["n_poss"].sum())
    out["unresolved_subs_per_game"] = float(d["unresolved_subs"].mean())
    out["sub_out_not_on_floor_per_game"] = float(d["sub_out_not_on_floor"].mean())
    out["forced_switches_per_game"] = float(d["forced_switches"].mean())
    out["periods_from_box_frac"] = float(d["periods_from_box"].sum() / d["n_periods"].sum())
    out["periods_starters_failed"] = int(d["periods_starters_failed"].sum())
    out["points_match_frac"] = float(((d["pts_home"] == d["box_pts_home"]) & (d["pts_away"] == d["box_pts_away"])).mean())
    # possessions vs pbpstats
    try:
        pg = pbpstats_games(season, phase, cfg)
        if len(pg):
            m = d.merge(pg[["GameId", "HomePossessions", "AwayPossessions"]], left_on="game_id", right_on="GameId")
            diff = (m["n_poss_home"] - m["HomePossessions"]).abs() + (m["n_poss_away"] - m["AwayPossessions"]).abs()
            out["pbpstats_games_matched"] = len(m)
            out["poss_abs_diff_per_game_vs_pbpstats"] = float(diff.mean() / 2)
            out["poss_within_2_frac"] = float((diff <= 4).mean())
    except Exception as e:  # noqa: BLE001
        out["pbpstats_error"] = repr(e)[:100]
    # sum of player possessions = 5 x team possessions holds by construction; check five distinct players
    if len(st):
        hl = st[HOME_SLOTS].to_numpy(); al = st[AWAY_SLOTS].to_numpy()
        out["five_distinct_frac"] = float(np.mean([(len(set(h)) == 5) and (len(set(a)) == 5) for h, a in zip(hl, al)]))
    return out
