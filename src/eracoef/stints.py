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
SHOT_ACTIONS = ("Made Shot", "Missed Shot", "Heave")
RIM_FT = 3.0          # feet; at or inside this is the "rim" bucket

# Bumped whenever the stint schema changes.  It is written into the per-season diag frame and
# checked on load, because the cache is keyed on path existence alone: without this, adding a column
# leaves every stale file loading happily with the new columns silently absent.
#   1  the original schema: poss/pts/margin/lineups
#   2  per-possession counters (POSS_COUNTERS below), for the luck-adjusted target
#   3  xftm / xftm_tech: expected free-throw makes, the first luck-adjusted term
#   4  per-shooter attempt counters by lineup slot (SLOT_COUNTERS), the first attempt's realised
#      points and rebound chance (pts1, chance1), the cross-fitting half per game, and the per-game
#      shooter table -- the inputs of the shooter-level expected points (xshoot.py)
STINT_SCHEMA = 4


class StaleStintCache(RuntimeError):
    """A cached stint file predates the current schema and must be rebuilt."""

# Per-possession counters, carried alongside `points` and summed per side into the stints.
#
# `att` is an ATTEMPT EVENT, and it is the unit the rebound chain is built on: every field-goal
# attempt, plus every free-throw trip that is not an and-1.  A shot that draws a shooting foul
# records no FGA at all, so counting only FGA misses 9% of possessions -- see FINDINGS.md section 14,
# where this definition reconciles the possession count to 0.25% across 1998, 2005 and 2024 while
# the box-score convention (FGA only) is out by 9%.
#
# Two identities hold exactly, per possession, and are tested in tests/test_stints.py:
#     points - pts_tech == 2*(fgm - fg3m) + 3*fg3m + ftm
#     reb_cont - cont_dead + att_retained == att - 1     (whenever the possession reaches an attempt)
# There are exactly three ways a possession gets an attempt after its first, and the identity names
# all of them:
#   reb_cont      it rebounded its own miss (including `oreb_unattr`, where the feed recorded no
#                 rebound row -- a block or a tip -- but play plainly continued)
#   att_retained  a flagrant, clear-path or away-from-play foul handed the ball back
#   cont_dead     subtracts a continuation that led to NO further attempt: rebound your own miss and
#                 then turn it over, or have the period expire.  Only the last continuation can be
#                 dead, since between any two rebounds there is necessarily a shot.
POSS_COUNTERS = (
    "pts_tech",                                             # technical/deferred FT points
    "fga", "fgm", "fg3a", "fg3m", "fta", "ftm",             # box-score shooting
    "fta_tech", "ftm_tech", "fta1", "tov",
    "att", "and1", "trip_shoot", "trip_ns", "fga_fouled", "att_retained",
    "reb_chance", "reb_cont", "reb_drop", "cont_dead",      # the rebound chain
    "oreb_p", "oreb_t", "oreb_unattr",
    "fga_rim", "fgm_rim", "fga_mid", "fgm_mid", "fga_thr", "fgm_thr",
    "att1_rim", "att1_mid", "att1_thr", "att1_ft",          # the possession's FIRST attempt only
    "xftm", "xftm_tech",                                    # EXPECTED free-throw makes (see below)
    "pts1", "chance1",                                      # the first attempt's realised points (incl. its
                                                            # free throws) and whether it produced a rebound chance
)

# Per-SHOOTER counters, kept by lineup slot so the shooter's own padded rate can price them at
# window-build time (xshoot.py).  The rate depends on which seasons are pooled and on which half of
# them is the other half, so it cannot be baked into the stints; the counts can.
#   fg2a, fg3a, fta   his two-point, three-point and (non-technical) free-throw attempts
#   xl2, xl3          the sum over those attempts of the LEAGUE make probability at that distance this
#                     season (shotcurve.py); xv2, xv3 the sum of l(1-l), for the logit variant
#   *1                the same restricted to the possession's first attempt (fg2a1/xl2_1/fg3a1/xl3_1),
#                     the free throws that belong to it (fta1), and its trip's last free throw (ftlast1),
#                     the one whose miss creates a rebound chance
# Slots are positions in the stint's SORTED lineup (h1..h5 / a1..a5); `x` is a shooter who was no
# longer on the floor when the possession closed (subbed out after his free throws), priced at the
# league rate.
SHOOTER_COUNTERS = ("fg2a", "xl2", "xv2", "fg3a", "xl3", "xv3", "fta",
                    "fg2a1", "xl2_1", "fg3a1", "xl3_1", "fta1", "ftlast1")
SLOTS = ("1", "2", "3", "4", "5", "x")
SLOT_COUNTERS = tuple(f"{c}_s{s}" for c in SHOOTER_COUNTERS for s in SLOTS)
SHOT_TABLE = ("fg2a", "fg2m", "xl2", "fg3a", "fg3m", "xl3")     # the per-game shooter table

# Expected free throws replace the ones that actually went in, which is the first and cleanest piece
# of the luck adjustment: whether a free throw drops is close to pure noise, and it is a fifth of all
# scoring.  `xftm` is the sum over the possession's attempts of the SHOOTER's own leave-one-game-out
# padded percentage.
#
# The shooter's own rate, not the lineup's: free-throw percentage is a property of one identified
# player and is essentially unaffected by the defence, so a lineup-level RAPM would throw away the
# identification we already have for free.
FT_LEAGUE_DEFAULT = 0.75


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
                 period_box_fetcher=None, gt_rule=None, ft_rates=None, curve=None):
        self.game_id = game_id
        self.curve = curve               # shotcurve.ShotCurve for the season; None -> flat fallback rates
        self.shots = {}                  # pid -> per-game shooter totals (SHOT_TABLE), every attempt
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
        # shotDistance is 0 on free throws and on any season that does not carry it; the rim/mid
        # split falls back to the shotValue buckets when it is absent (see _shot_bucket)
        sd = ev["shotDistance"] if "shotDistance" in ev.columns else -1.0
        ev["shot_dist"] = pd.to_numeric(pd.Series(sd, index=ev.index), errors="coerce").fillna(-1.0).astype(float)
        for c in ("scoreHome", "scoreAway"):
            ev[c] = pd.to_numeric(ev[c], errors="coerce").fillna(0).astype(int)
        self.ev = ev.reset_index(drop=True)
        # leave-one-game-out free-throw percentage per shooter: season totals minus this game's, so a
        # player's own makes never price the possessions he is being scored on
        self._ft = None
        if ft_rates is not None:
            totals, league, k = ft_rates
            made_g, att_g = {}, {}
            for r in self.ev.itertuples(index=False):
                if r.actionType != "Free Throw":
                    continue
                pid = int(r.personId)
                d = str(r.description)
                made = ("PTS" in d) or (not d.startswith("MISS") and (r.scoreHome + r.scoreAway) > 0)
                att_g[pid] = att_g.get(pid, 0) + 1
                made_g[pid] = made_g.get(pid, 0) + int(made)
            self._ft = (totals, float(league), float(k), made_g, att_g)
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

    def shot_p(self, three: bool, dist: float, desc: str) -> float:
        """The LEAGUE make probability of this attempt at this distance this season (shotcurve)."""
        if self.curve is None:
            from .shotcurve import FALLBACK
            return FALLBACK[bool(three)]
        return self.curve.prob(dist, three, desc)

    def ft_prob(self, pid: int) -> float:
        """The shooter's padded FT%, with this game's own attempts removed."""
        if self._ft is None:
            return FT_LEAGUE_DEFAULT
        totals, league, k, made_g, att_g = self._ft
        m, a = totals.get(int(pid), (0.0, 0.0))
        m -= made_g.get(int(pid), 0)
        a -= att_g.get(int(pid), 0)
        return (max(m, 0.0) + k * league) / (max(a, 0.0) + k)

    # ---------------------------------------------------------------- main loop
    def possessions(self) -> pd.DataFrame:
        recs = []
        score = {self.home: 0, self.away: 0}
        pending = {self.home: 0, self.away: 0}
        # technical free throws taken with no possession in progress: their attempts and makes are
        # deferred exactly like their points, so the counters stay complete
        pend_fta = {self.home: 0, self.away: 0}
        pend_ftm = {self.home: 0, self.away: 0}
        pend_xftm = {self.home: 0.0, self.away: 0.0}
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
            # counters for the possession being built, and the small amount of state needed to
            # resolve them.  Both are mutated in place so `close` needs no extra nonlocals.
            ct = dict.fromkeys(POSS_COUNTERS, 0)
            sh: dict = {}                    # pid -> SHOOTER_COUNTERS for the possession being built
            sc = dict(pending_chance=0,      # a miss is waiting to be resolved into a rebound
                      and1_used=False,       # the deferred and-1 free throw has been seen
                      cont_team=False,       # the last continuation was a TEAM offensive rebound
                      att_since_cont=0,      # attempts since that continuation
                      retain_pending=0,      # a flagrant/clear-path/away-from-play gave the ball back
                      last_foul_shooting=False)

            def close(reason, end_clock, next_offense):
                nonlocal offense, active, pts, and1, start_clock, start_score
                # A team offensive rebound with the period expiring before any further attempt did
                # not continue anything, so it is neither a continuation nor a contested chance.
                # About 5% of all continuations; dropping it is what takes the possession
                # reconciliation from 0.7% to 0.004% on 2024 (FINDINGS.md section 14).
                if ct["reb_cont"] > 0 and sc["att_since_cont"] == 0:
                    if reason == "period_end" and sc["cont_team"]:
                        ct["reb_cont"] -= 1
                        ct["reb_chance"] -= 1
                        ct["oreb_t"] -= 1
                        ct["reb_drop"] += 1
                        if ct["att"] == 1:
                            ct["chance1"] -= 1          # it was the first attempt's chance
                    else:
                        ct["cont_dead"] += 1
                if offense is not None and active:
                    p = pts + pending[offense]
                    ct["pts_tech"] += pending[offense]
                    ct["fta_tech"] += pend_fta[offense]
                    ct["ftm_tech"] += pend_ftm[offense]
                    ct["xftm_tech"] += pend_xftm[offense]
                    pending[offense] = 0
                    pend_fta[offense] = 0
                    pend_ftm[offense] = 0
                    pend_xftm[offense] = 0.0
                    ok = valid[self.home] and valid[self.away] and len(on[self.home]) == 5 and len(on[self.away]) == 5
                    if not ok:
                        self.diag["invalid_possessions"] += 1
                    recs.append(dict(period=period, offense=offense, points=p, start_clock=start_clock, end_clock=end_clock,
                                     score_home=start_score[self.home], score_away=start_score[self.away],
                                     home_lineup=tuple(sorted(on[self.home])), away_lineup=tuple(sorted(on[self.away])),
                                     valid=ok, reason=reason, shooters=dict(sh), **ct))
                offense = next_offense
                active = False
                pts = 0
                and1 = False
                start_clock = end_clock
                start_score = dict(score)
                for k in ct:
                    ct[k] = 0
                sh.clear()
                sc.update(pending_chance=0, and1_used=False, cont_team=False, att_since_cont=0,
                          retain_pending=0)

            def note_attempt(bucket):
                """Register an attempt event and, if it is the possession's first, its bucket.

                Only the FIRST attempt is recorded by type.  Everything after it exists only because
                a rebound went the offense's way, and conditioning on that is exactly the luck the
                expected-points target is built to remove."""
                # A new attempt while a miss is still unresolved means the offense kept the ball
                # without the feed recording a rebound -- a block or a tip sequence.  About one
                # possession in 100,000, but it is a real continuation and leaving it out is what
                # breaks the identity below.
                if sc["retain_pending"] and ct["att"] > 0:
                    ct["att_retained"] += 1       # a retention foul, not a rebound, bought this one
                    sc["retain_pending"] = 0
                elif sc["pending_chance"]:
                    ct["reb_chance"] += 1
                    ct["reb_cont"] += 1
                    ct["oreb_unattr"] += 1
                    if ct["att"] == 1:
                        ct["chance1"] += 1
                    sc.update(pending_chance=0, cont_team=False, att_since_cont=0)
                if ct["att"] == 0:
                    ct[f"att1_{bucket}"] += 1
                ct["att"] += 1
                sc["att_since_cont"] += 1

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
                            if pending[t] > 0 or pend_fta[t] > 0:
                                for rec in reversed(recs):
                                    if rec["offense"] == t:
                                        rec["points"] += pending[t]
                                        rec["pts_tech"] += pending[t]   # keep the identity exact
                                        rec["fta_tech"] += pend_fta[t]
                                        rec["ftm_tech"] += pend_ftm[t]
                                        rec["xftm_tech"] += pend_xftm[t]
                                        pending[t] = pend_fta[t] = pend_ftm[t] = 0
                                        pend_xftm[t] = 0.0
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
                    ensure_offense(team, i)     # may close(); every counter below must follow it
                    made = at == "Made Shot"
                    three = int(r.shotValue) == 3
                    bucket = "thr" if three else ("rim" if 0 <= r.shot_dist <= RIM_FT else "mid")
                    ct["fga"] += 1
                    ct[f"fga_{bucket}"] += 1
                    ct["fg3a"] += three
                    if made:
                        ct["fgm"] += 1
                        ct[f"fgm_{bucket}"] += 1
                        ct["fg3m"] += three
                    note_attempt(bucket)          # resolves any earlier miss, before arming this one
                    # the shooter's own counters, and the league's view of the shot
                    pid = int(r.personId)
                    first = ct["att"] == 1
                    lp = self.shot_p(three, r.shot_dist, str(r.description))
                    s = sh.setdefault(pid, dict.fromkeys(SHOOTER_COUNTERS, 0.0))
                    kind = "3" if three else "2"
                    s[f"fg{kind}a"] += 1
                    s[f"xl{kind}"] += lp
                    s[f"xv{kind}"] += lp * (1.0 - lp)
                    if first:
                        s[f"fg{kind}a1"] += 1
                        s[f"xl{kind}_1"] += lp
                    g = self.shots.setdefault(pid, dict.fromkeys(SHOT_TABLE, 0.0))
                    g[f"fg{kind}a"] += 1
                    g[f"xl{kind}"] += lp
                    g[f"fg{kind}m"] += made
                    if not made:
                        sc["pending_chance"] = 1
                    if at == "Made Shot":
                        v = int(r.shotValue) if int(r.shotValue) in (2, 3) else 2
                        score[team] += v
                        pts += v
                        if first:
                            ct["pts1"] += v
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
                    ct["tov"] += 1
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
                        # offensive rebound: the possession continues, so this resolves the pending
                        # chance as a continuation.  Team rebounds (teamId 0) count here too; the
                        # period-end ones are taken back out in close().
                        if sc["pending_chance"]:
                            ct["reb_chance"] += 1
                            ct["reb_cont"] += 1
                            if ct["att"] == 1:
                                ct["chance1"] += 1
                            team_reb = int(r.teamId) == 0
                            ct["oreb_t" if team_reb else "oreb_p"] += 1
                            sc.update(pending_chance=0, cont_team=team_reb, att_since_cont=0)
                        continue
                    # defensive rebound: resolve the chance BEFORE close(), which flushes the record
                    if sc["pending_chance"]:
                        ct["reb_chance"] += 1
                        if ct["att"] == 1:
                            ct["chance1"] += 1
                        sc["pending_chance"] = 0
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
                        live = offense == team and active
                        if made:
                            score[team] += 1
                            if live:
                                pts += 1
                                ct["pts_tech"] += 1
                            else:
                                pending[team] += 1   # credited to the team's next (or, at period end, last) possession
                        q = self.ft_prob(r.personId)
                        if live:
                            ct["fta_tech"] += 1
                            ct["ftm_tech"] += made
                            ct["xftm_tech"] += q
                        else:
                            pend_fta[team] += 1
                            pend_ftm[team] += made
                            pend_xftm[team] += q
                        continue
                    ensure_offense(team, i)     # may close(); every counter below must follow it
                    m = FT_RE.search(st)
                    k, n = (int(m.group(2)), int(m.group(3))) if m else (1, 1)
                    if k == 1:
                        # The trip starts here.  Three kinds of trip are NOT new attempt events:
                        #   and-1          the field goal it follows already counted
                        #   fouled on a miss that the feed ALSO recorded as a missed FG.  Usually a
                        #     shooting foul on a miss records no FGA at all, which is why trips
                        #     count as attempts -- but when both appear they are one attempt, and
                        #     the whistle means that "miss" was never a live rebound either
                        #   retained      flagrant, clear path and away-from-play keep the ball, so
                        #     they do not terminate anything
                        retained = retain or "Away From Play" in st
                        double = sc["last_foul_shooting"] and sc["pending_chance"] == 1
                        if and1 and not sc["and1_used"]:
                            sc["and1_used"] = True
                            ct["and1"] += 1
                        elif double:
                            sc["pending_chance"] = 0     # the whistle killed the rebound chance
                            ct["fga_fouled"] += 1
                        elif not retained:
                            ct["trip_shoot" if sc["last_foul_shooting"] else "trip_ns"] += 1
                            note_attempt("ft")
                        if retained:
                            sc["retain_pending"] = 1
                    ct["fta"] += 1
                    ct["ftm"] += made
                    ct["xftm"] += self.ft_prob(r.personId)
                    s = sh.setdefault(int(r.personId), dict.fromkeys(SHOOTER_COUNTERS, 0.0))
                    s["fta"] += 1
                    if ct["att"] <= 1:
                        ct["fta1"] += 1     # free throws belonging to the possession's first attempt
                        s["fta1"] += 1
                        if made:
                            ct["pts1"] += 1
                        if k == n:
                            s["ftlast1"] += 1   # the trip's last one: its miss is the first attempt's rebound chance
                    if made:
                        score[team] += 1
                        pts += 1
                    if retain or not m:
                        continue
                    if k == n and not made:
                        sc["pending_chance"] = 1    # a missed last free throw is a live rebound
                    if k == n and made:
                        close("made_ft", r.clock_s, self._other(team))
                    continue
                if at == "Foul":
                    # observed but not acting: the trip that follows needs to know whether the foul
                    # was a shooting foul, because a shooting foul on a MISS records no FGA at all
                    sc["last_foul_shooting"] = "Shooting" in str(r.subType)
                    continue
                # anything else: no possession effect
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
        p = poss[poss["valid"]]
        if len(p) == 0:
            return pd.DataFrame()
        s = sid[poss["valid"]].to_numpy()

        # One groupby over a masked matrix, rather than a .loc[].sum() per counter per stint.  With
        # 35 counters on both sides that was ~2.4M pandas slice-sums a season and dominated the
        # whole parse; this is the same arithmetic in one pass.
        cols = ["points"] + list(POSS_COUNTERS)
        vals = p[cols].to_numpy(dtype=float)
        is_h = (p["offense"].to_numpy() == self.home)
        side = np.concatenate([vals * is_h[:, None], vals * (~is_h)[:, None]], axis=1)
        names = [f"{c}_h" for c in cols] + [f"{c}_a" for c in cols]
        agg = pd.DataFrame(side, columns=names).groupby(s, sort=True).sum()
        agg = agg.rename(columns={"points_h": "pts_h", "points_a": "pts_a"})
        cnt = pd.DataFrame({"poss_h": is_h.astype(int), "poss_a": (~is_h).astype(int)}
                           ).groupby(s, sort=True).sum()

        # per-stint context comes from the FIRST possession of the stint, as before
        first = p.groupby(s, sort=True).head(1)
        period = first["period"].to_numpy().astype(int)
        clock = first["start_clock"].to_numpy().astype(float)
        margin_h = (first["score_home"] - first["score_away"]).to_numpy().astype(int)
        frac_rem = np.array([regulation_remaining(int(q), float(c)) for q, c in zip(period, clock)]) / 2880.0
        is_gt = (period == 4) & (
            (np.abs(margin_h) >= gt.get("q4_margin_any", 20))
            | ((np.abs(margin_h) >= gt.get("q4_margin_late", 15)) & (clock < 60 * gt.get("late_minutes", 6))))

        out = pd.DataFrame({"period": period, "poss_h": cnt["poss_h"].to_numpy(),
                            "poss_a": cnt["poss_a"].to_numpy(), "pts_h": agg["pts_h"].to_numpy(),
                            "pts_a": agg["pts_a"].to_numpy(), "margin_h": margin_h,
                            "frac_rem": frac_rem, "is_gt": is_gt, "start_clock": clock})
        # every counter is defined for the team ON OFFENSE, so the opponent's defensive version of it
        # is just the other side's row -- build_design already emits both (design.py:245)
        for c in POSS_COUNTERS:
            out[f"{c}_h"] = agg[f"{c}_h"].to_numpy()
            out[f"{c}_a"] = agg[f"{c}_a"].to_numpy()
        for slots, col in ((HOME_SLOTS, "home_lineup"), (AWAY_SLOTS, "away_lineup")):
            L = np.array([list(t) for t in first[col]], dtype=np.int64)
            for i, name in enumerate(slots):
                out[name] = L[:, i]

        # the per-shooter counters, by the shooter's slot in the stint's sorted lineup.  Every
        # possession of a stint shares the lineup, so the slot of a player is fixed within it; a
        # shooter no longer on the floor when the possession closed goes to slot `x`.
        long = []
        stint_pos = {sid: i for i, sid in enumerate(agg.index)}
        for sid, off, hl, al, shooters in zip(s, p["offense"].to_numpy(), p["home_lineup"], p["away_lineup"],
                                              p["shooters"]):
            if not shooters:
                continue
            lineup = hl if off == self.home else al
            side = "h" if off == self.home else "a"
            pos = {pid: str(i + 1) for i, pid in enumerate(lineup)}
            for pid, cnt in shooters.items():
                slot = pos.get(pid, "x")
                for c, v in cnt.items():
                    if v:
                        long.append((stint_pos[sid], f"{c}_s{slot}_{side}", v))
        cols = [f"{c}_{side}" for c in SLOT_COUNTERS for side in ("h", "a")]
        block = np.zeros((len(agg), len(cols)))
        if long:
            ci = {c: j for j, c in enumerate(cols)}
            for i, c, v in long:
                block[i, ci[c]] += v
        return pd.concat([out, pd.DataFrame(block, columns=cols, index=out.index)], axis=1)


# ----------------------------------------------------------------------------- season driver
def build_game(game_id: str, home_id: int, away_id: int, cfg, gt_rule=None, ft_rates=None, curve=None):
    pbp = fetch_pbp(game_id, cfg)
    box = fetch_boxscore(game_id, cfg)
    fetcher = lambda gid, p: fetch_period_box(gid, p, cfg)  # noqa: E731
    gp = GameParser(pbp, box, home_id, away_id, game_id, period_box_fetcher=fetcher, gt_rule=gt_rule,
                    ft_rates=ft_rates, curve=curve)
    poss = gp.possessions()
    st = gp.stints(poss)
    shots = pd.DataFrame([dict(player_id=pid, **v) for pid, v in gp.shots.items()],
                         columns=["player_id", *SHOT_TABLE])
    shots["game_id"] = game_id
    d = dict(gp.diag)
    d.update(game_id=game_id, n_poss=len(poss), n_poss_home=int((poss["offense"] == gp.home).sum()) if len(poss) else 0,
             n_poss_away=int((poss["offense"] == gp.away).sum()) if len(poss) else 0,
             n_valid_poss=int(poss["valid"].sum()) if len(poss) else 0,
             pts_home=float(poss.loc[poss["offense"] == gp.home, "points"].sum()) if len(poss) else 0.0,
             pts_away=float(poss.loc[poss["offense"] == gp.away, "points"].sum()) if len(poss) else 0.0,
             box_pts_home=gp.box_pts.get(gp.home, np.nan), box_pts_away=gp.box_pts.get(gp.away, np.nan),
             n_stints=len(st))
    names = pd.DataFrame({"player_id": list(gp.day_names), "pbp_name": list(gp.day_names.values())})
    return st, d, names, shots


def assign_halves(games: pd.DataFrame) -> pd.Series:
    """The cross-fitting half of every game, by game_id: regular-season games alternate A/B in
    chronological order within a season, playoff games within a series.  The SAME rule as
    design._order_games, applied at stint-build time so the shooter tables and the design agree."""
    g = games.copy()
    g["_ph"] = (g["phase"] == "PO").astype(int)
    g = g.sort_values(["season", "_ph", "game_date", "game_id"]).reset_index(drop=True)
    rs = (g["phase"] == "RS").to_numpy()
    half = np.empty(len(g), dtype=object)
    if rs.any():
        half[rs] = np.where(g[rs].groupby("season").cumcount().to_numpy() % 2 == 0, "A", "B")
    if (~rs).any():
        half[~rs] = np.where(g[~rs].groupby(["season", "series_id"]).cumcount().to_numpy() % 2 == 0, "A", "B")
    return pd.Series(half, index=g["game_id"].to_numpy())


def build_season(season: int, phase: str, cfg, force=False, verbose=True):
    """Stints for one season and phase -> data/stints/{season}_{phase}.parquet (+ _diag.parquet)."""
    out_dir = resolve(cfg, "stints")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{season}_{phase}.parquet"
    dpath = out_dir / f"{season}_{phase}_diag.parquet"
    if path.exists() and dpath.exists() and not force:
        st, d = pd.read_parquet(path), pd.read_parquet(dpath)
        # A cache written before the per-possession counters existed still loads, and its missing
        # columns become NaN the moment it is concatenated with a fresh one -- which then flows
        # silently into y and w.  Refuse it instead.
        have = int(d["stint_schema"].iloc[0]) if "stint_schema" in d.columns and len(d) else 0
        if have >= STINT_SCHEMA:
            return st, d
        raise StaleStintCache(
            f"{path.name} was built at schema {have}, this code needs {STINT_SCHEMA}. "
            f"Rebuild with: python scripts/02_stints.py {season} {season} {phase} --force")
    gl = load_gamelog(season, phase, cfg)
    games = game_table(gl)
    gt_rule = cfg.get("garbage_time", {})
    # the shooters' season free-throw totals, for the leave-one-game-out expected-makes term
    from .boxtable import box_from_gamelog, ft_padding, ft_totals
    sb = box_from_gamelog(gl)
    ft_rates = (ft_totals(sb), *ft_padding(sb))
    # the league's make probability by distance this season (the regular season's curve, for the
    # playoffs too), so every attempt carries the league's view of it into the shooter counters
    from .shotcurve import curve_for
    curve = curve_for(season, cfg, verbose=verbose)
    neutral_seasons = set(cfg.get("neutral_site_seasons_po", []))
    parts, diags, name_parts, shot_parts = [], [], [], []
    t0 = time.time()
    for k, g in enumerate(games.itertuples(index=False)):
        try:
            st, d, nm, shots = build_game(g.game_id, g.home_team_id, g.away_team_id, cfg, gt_rule, ft_rates, curve)
        except Exception as e:  # noqa: BLE001
            diags.append(dict(game_id=g.game_id, error=repr(e)[:300]))
            continue
        name_parts.append(nm)
        shot_parts.append(shots)
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
    # the cross-fitting half of every game that produced stints, stored so the design and the
    # shooter tables use the same one
    if len(stints):
        halves = assign_halves(stints[["game_id", "season", "phase", "game_date", "series_id"]].drop_duplicates("game_id"))
        stints["half"] = stints["game_id"].map(halves)
    diag = pd.DataFrame(diags)
    diag["seconds_total"] = time.time() - t0
    diag["stint_schema"] = STINT_SCHEMA
    stints.to_parquet(path, index=False)
    diag.to_parquet(dpath, index=False)
    shots = pd.concat(shot_parts, ignore_index=True) if shot_parts else pd.DataFrame(columns=["player_id", *SHOT_TABLE, "game_id"])
    shots["season"] = season
    shots["phase"] = phase
    shots["half"] = shots["game_id"].map(halves) if len(stints) else None
    shots.to_parquet(out_dir / f"{season}_{phase}_shots.parquet", index=False)
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
