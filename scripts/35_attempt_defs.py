"""Stage 0 of the luck-adjusted target: pin down what an "attempt" is, before writing any counters.

HANDOFF.md quotes two reference numbers -- league OREB% 24.4% and a continuation multiplier of about
1.15 -- without saying which attempt definition they hold under.  They are not definition-free.  A
shot attempt that draws a shooting foul records no FGA, so it exists only as a free-throw trip; a
bonus free throw is not a shot attempt but its miss can still be rebounded; a team rebound is a
continuation for the offense but not a box-score offensive rebound; and a miss as the period expires
was never rebounded by anyone.  Each of those choices moves OREB% by one to three points.

Nothing here needs possession boundaries.  Writing `att` for attempt events and `cont` for offensive
rebounds that continue a possession, every possession that reaches an attempt has exactly one more
attempt than it has continuations, so

    possessions with an attempt = att - cont        and       multiplier = att / (att - cont)

both fall out of the event stream alone.  (The `1/(1-m*r)` form of the multiplier is the same
statement rearranged, so it agrees identically for every definition and is not a test of anything.)

The test that DOES discriminate is reconciliation against the possession count we already have from
the built stints.  Every possession ends in an attempt, a turnover, or nothing:

    (att - cont) + turnovers + leftover = possessions

so each definition predicts a `leftover` -- possessions it says reach neither an attempt nor a
turnover.  The right definition is the one whose leftover is small and explicable.  A definition
that undercounts attempts shows up immediately as a large positive leftover.

Also checks the claim in HANDOFF that only 89.4% of missed field goals are followed by a rebound.
That is what `shift(-1)` sees; a short forward scan should do much better, and if it does not, the
missing ones have to be accounted for or OREB% is biased.

Read-only: reads cached play-by-play and the built stints, writes one CSV.

usage: python scripts/35_attempt_defs.py [n_games=120] [seasons=2024,2005,1998]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config, resolve  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 4)
cfg = load_config()
ROOT = Path(cfg["_root"])
RAW = resolve(cfg, "raw") / "pbp"
STINTS = resolve(cfg, "stints")
OUT = ROOT / "outputs"
N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 120
SEASONS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [2024, 2005, 1998]

SCAN = 7          # how many events forward to look for the rebound that resolves a miss
FOUL_BACK = 6     # how many events back to look for the foul that caused a free-throw trip
SHOT = ("Made Shot", "Missed Shot", "Heave")


def load_game(path):
    ev = pd.read_parquet(path)
    for c in ("actionType", "subType", "description", "shotResult"):
        ev[c] = ev[c].astype(str).str.strip()
    for c in ("teamId", "personId", "period", "shotValue"):
        ev[c] = pd.to_numeric(ev[c], errors="coerce").fillna(0).astype(int)
    for c in ("scoreHome", "scoreAway"):
        ev[c] = pd.to_numeric(ev[c], errors="coerce").fillna(0).astype(int)
    return ev.reset_index(drop=True)


def row_team(r, teams):
    """Team rebounds carry teamId 0 and put the team id in personId (stints._row_team does this)."""
    if r.teamId in teams:
        return r.teamId
    if r.personId in teams:
        return r.personId
    return 0


def ft_made(r):
    """`shotResult` is blank on free-throw rows; made ones carry "(N PTS)" (stints.py:413)."""
    d = str(r.description)
    return ("PTS" in d) or (not d.startswith("MISS") and (r.scoreHome + r.scoreAway) > 0)


def scan_game(ev):
    """One pass over a game.  Returns per-game counts under every definition at once."""
    rows = list(ev.itertuples(index=False))
    n = len(rows)
    teams = set(x for x in ev.teamId.unique() if x > 0)
    c = dict.fromkeys(
        ("fga", "fg_miss", "fg_made", "trip_shoot", "trip_ns", "trip_and1", "trip_tech",
         "ft_final_miss_shoot", "ft_final_miss_ns",
         "chance_fg", "chance_ft_shoot", "chance_ft_ns",
         "oreb_p", "oreb_t", "dreb_p", "dreb_t", "terminal_fg", "terminal_ft",
         "oreb_t_period_end", "tov",
         "miss_reb_gap1", "miss_reb_found", "miss_reb_none"), 0)

    # the and-1 rule the parser already uses: a made shot followed within 12 events, at the same
    # clock, by a free throw from the same team (stints.py:376-383)
    def is_and1(i, team):
        r = rows[i]
        for j in range(i + 1, min(i + 13, n)):
            q = rows[j]
            if q.clock != r.clock:
                break
            if q.actionType == "Free Throw" and row_team(q, teams) == team and "Technical" not in str(q.subType):
                return True
        return False

    def resolve(i, team):
        """Find the rebound for the miss at row i.  Returns 'off', 'off_team', 'def', 'def_team' or
        None when the period ends first -- which is terminal, NOT a defensive rebound.  Counting
        those as defensive would bias OREB% down (HANDOFF:222-223)."""
        for j in range(i + 1, min(i + 1 + SCAN, n)):
            q = rows[j]
            at = str(q.actionType)
            if at == "period" or q.period != rows[i].period:
                return None
            if at == "Rebound":
                t = row_team(q, teams)
                team_reb = q.teamId == 0
                if j == i + 1:
                    c["miss_reb_gap1"] += 1
                if t == team:
                    return "off_team" if team_reb else "off"
                return "def_team" if team_reb else "def"
            if at in SHOT or at == "Turnover":
                return None      # a new attempt with no rebound row in between: unresolvable
        return None

    def period_ends_next(i):
        """An offensive TEAM rebound with the period expiring right after is not a continuation."""
        for j in range(i + 1, min(i + 4, n)):
            q = rows[j]
            if str(q.actionType) == "period" or q.period != rows[i].period:
                return True
            if str(q.actionType) in SHOT or str(q.actionType) in ("Turnover", "Free Throw"):
                return False
        return False

    for i, r in enumerate(rows):
        at = str(r.actionType)
        team = row_team(r, teams)
        if at == "Turnover":
            c["tov"] += 1
        elif at in SHOT:
            c["fga"] += 1
            missed = (at != "Made Shot")
            c["fg_miss" if missed else "fg_made"] += 1
            if missed:
                res = resolve(i, team)
                if res is None:
                    c["terminal_fg"] += 1
                    c["miss_reb_none"] += 1
                else:
                    c["miss_reb_found"] += 1
                    c["chance_fg"] += 1
                    c[{"off": "oreb_p", "off_team": "oreb_t",
                       "def": "dreb_p", "def_team": "dreb_t"}[res]] += 1
                    if res == "off_team" and period_ends_next(i):
                        c["oreb_t_period_end"] += 1
            elif is_and1(i, team):
                c["trip_and1"] += 1
        elif at == "Free Throw":
            st = str(r.subType)
            if "Technical" in st:
                if " 1 of " in st or st.endswith("Technical"):
                    c["trip_tech"] += 1
                continue
            first = " 1 of " in st
            k_of_n = st.rsplit(" ", 3)[-3:] if " of " in st else None
            final = bool(k_of_n) and k_of_n[0] == k_of_n[2]
            # classify the trip by the foul that caused it
            shooting = False
            for j in range(i - 1, max(i - 1 - FOUL_BACK, -1), -1):
                if str(rows[j].actionType) == "Foul":
                    shooting = "Shooting" in str(rows[j].subType)
                    break
            if first:
                c["trip_shoot" if shooting else "trip_ns"] += 1
            if final and not ft_made(r):
                c["ft_final_miss_shoot" if shooting else "ft_final_miss_ns"] += 1
                res = resolve(i, team)
                if res is None:
                    c["terminal_ft"] += 1
                else:
                    c["chance_ft_shoot" if shooting else "chance_ft_ns"] += 1
                    c[{"off": "oreb_p", "off_team": "oreb_t",
                       "def": "dreb_p", "def_team": "dreb_t"}[res]] += 1
    return c


# --------------------------------------------------------------------------- the definitions
# att   what counts as an attempt event (the thing the geometric series chains)
# chance/cont  which misses can be rebounded, and which rebounds continue the possession
DEFS = {
    "A  FGA only, player OREB": dict(
        att=lambda c: c["fga"],
        chance=lambda c: c["chance_fg"],
        cont=lambda c: c["oreb_p"],
        miss=lambda c: c["fg_miss"] - c["terminal_fg"]),
    "B  FGA only, incl team OREB": dict(
        att=lambda c: c["fga"],
        chance=lambda c: c["chance_fg"],
        cont=lambda c: c["oreb_p"] + c["oreb_t"],
        miss=lambda c: c["fg_miss"] - c["terminal_fg"]),
    "C  FGA + shooting trips (no and-1), incl team OREB": dict(
        att=lambda c: c["fga"] + c["trip_shoot"] - c["trip_and1"],
        chance=lambda c: c["chance_fg"] + c["chance_ft_shoot"],
        cont=lambda c: c["oreb_p"] + c["oreb_t"],
        miss=lambda c: (c["fg_miss"] - c["terminal_fg"]) + c["ft_final_miss_shoot"]),
    "D  C + non-shooting trips too": dict(
        att=lambda c: c["fga"] + c["trip_shoot"] + c["trip_ns"] - c["trip_and1"],
        chance=lambda c: c["chance_fg"] + c["chance_ft_shoot"] + c["chance_ft_ns"],
        cont=lambda c: c["oreb_p"] + c["oreb_t"],
        miss=lambda c: (c["fg_miss"] - c["terminal_fg"]) + c["ft_final_miss_shoot"] + c["ft_final_miss_ns"]),
    # E is D with the period-end team offensive rebounds taken out of BOTH sides: the period expired,
    # so the possession did not continue and the chance was never really contested.  About 5% of all
    # continuations, and it is the difference between a 0.7% and a 0.0% reconciliation residual.
    "E  D, period-end team OREB dropped": dict(
        att=lambda c: c["fga"] + c["trip_shoot"] + c["trip_ns"] - c["trip_and1"],
        chance=lambda c: (c["chance_fg"] + c["chance_ft_shoot"] + c["chance_ft_ns"]
                          - c["oreb_t_period_end"]),
        cont=lambda c: c["oreb_p"] + c["oreb_t"] - c["oreb_t_period_end"],
        miss=lambda c: (c["fg_miss"] - c["terminal_fg"]) + c["ft_final_miss_shoot"] + c["ft_final_miss_ns"]),
}

out, raws = [], []
for season in SEASONS:
    d = RAW / str(season)
    files = sorted(d.glob("*.parquet"))[:N_GAMES]
    if not files:
        print(f"season {season}: no cached play-by-play at {d}, skipping")
        continue
    tot = None
    for f in files:
        c = scan_game(load_game(f))
        tot = c if tot is None else {k: tot[k] + v for k, v in c.items()}
    tot["season"] = season
    tot["games"] = len(files)
    raws.append(tot)

    sp = STINTS / f"{season}_RS.parquet"
    poss = ppp = np.nan
    if sp.exists():
        st = pd.read_parquet(sp)
        poss = float((st.poss_h + st.poss_a).sum() / st.game_id.nunique())
        ppp = float((st.pts_h + st.pts_a).sum() / (st.poss_h + st.poss_a).sum())

    print(f"\n=== season {season}: {len(files)} games")
    print(f"  possessions/game {poss:.1f}   points/possession {ppp:.4f}   (all RS games, built stints)")
    found, none_ = tot["miss_reb_found"], tot["miss_reb_none"]
    print(f"  missed FG resolved to a rebound: {found}/{found + none_} = "
          f"{100 * found / max(found + none_, 1):.1f}%   "
          f"(of those, {100 * tot['miss_reb_gap1'] / max(found, 1):.1f}% at the very next event)")
    print(f"  free-throw trips: {tot['trip_shoot']} shooting, {tot['trip_ns']} non-shooting "
          f"({100 * tot['trip_ns'] / max(tot['trip_shoot'] + tot['trip_ns'], 1):.0f}% non-shooting), "
          f"{tot['trip_and1']} and-1, {tot['trip_tech']} technical")
    print(f"  team rebounds: {tot['oreb_t']} offensive, {tot['dreb_t']} defensive")
    print(f"  terminal misses dropped: {tot['terminal_fg']} FG, {tot['terminal_ft']} FT")

    g = len(files)
    for name, D in DEFS.items():
        att, ch, cont = D["att"](tot), D["chance"](tot), D["cont"](tot)
        with_att = (att - cont) / g                    # possessions that reach an attempt
        tov = tot["tov"] / g
        out.append(dict(season=season, definition=name, att_per_game=att / g,
                        oreb_pct=100 * cont / ch,
                        poss_with_att=with_att, tov=tov,
                        implied_poss=with_att + tov, actual_poss=poss,
                        leftover=poss - with_att - tov,
                        leftover_pct=100 * (poss - with_att - tov) / poss,
                        multiplier=att / (att - cont)))

R = pd.DataFrame(out)
print("\n\n=== reconciliation: (att - cont) + turnovers + leftover = possessions")
print("    `leftover` is what each definition cannot account for.  Big and positive means the")
print("    definition is missing attempts.")
for season in R.season.unique():
    print(f"\n-- season {season}")
    print(R[R.season == season].drop(columns="season").to_string(index=False))

print("\n\n=== averaged over seasons; HANDOFF quotes OREB% 24.4 and multiplier ~1.15")
piv = R.pivot_table(index="definition",
                    values=["att_per_game", "oreb_pct", "multiplier", "leftover", "leftover_pct"],
                    aggfunc="mean")
piv = piv[["att_per_game", "oreb_pct", "multiplier", "leftover", "leftover_pct"]]
piv = piv.reindex(piv.leftover.abs().sort_values().index)
print(piv.to_string())
best = piv.index[0]
print(f"\nsmallest unexplained residual: {best}")
print(f"  leftover {piv.loc[best, 'leftover']:+.2f} possessions/game "
      f"({piv.loc[best, 'leftover_pct']:+.2f}%), OREB% {piv.loc[best, 'oreb_pct']:.1f}, "
      f"multiplier {piv.loc[best, 'multiplier']:.3f}")

R.to_csv(OUT / "csv" / "attempt_definitions.csv", index=False)
pd.DataFrame(raws).to_csv(OUT / "csv" / "attempt_raw_counts.csv", index=False)
print("\nwrote outputs/csv/attempt_definitions.csv and outputs/csv/attempt_raw_counts.csv")
