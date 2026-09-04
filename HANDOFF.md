# Handoff: the luck-adjusted target is finished, and half of it did not pay

Written after the session that measured the untested `c_def` x xPTS combination, then built the
stage 4 shot term, gated it, and measured it out of season — where it lost. `FINDINGS.md` section
17 is the record; sections 13-16 are the previous session's. `HANDOFF_luck_archive.md` is the
previous version of this file, `HANDOFF_hybrid_archive.md` and `HANDOFF_diagnosis_archive.md` the
ones before.

**Branch:** everything is on `hybrid-and-xpts`, uncommitted at the time of writing (see the file
list in Part 3), and not merged to `main`. Pages serves `/docs` from `main`.

---

## Part 1: where things stand

### The chosen system is unchanged: hybrid + xPTS(ft), `c_def = 0`

Nothing in this session moved the shipped rating. Two things were measured that had not been.

**1. The combination (`scripts/42_defweight_xpts.py`).** The `c_def` sweep re-run on the
free-throw-adjusted target, both aggregations, both K, consensus beside each row. 28 held-out
seasons. Game level:

| target | c_def | K=2 | K=4 | consensus total | consensus defense | archetype bias |
|---|---|---|---|---|---|---|
| pts | 0 | 113.31 | 113.07 | **0.896** | **0.888** | **+0.195** |
| xpts_ft | 0 | 113.24 | 113.04 | 0.895 | 0.888 | +0.190 |
| pts | 0.5 | 112.89 | 112.76 | 0.854 | 0.839 | +0.498 |
| xpts_ft | **0.5** | **112.81** | **112.73** | 0.854 | 0.841 | +0.495 |
| xpts_ft | 0.75 | 112.85 | 112.74 | 0.824 | 0.804 | +0.572 |

- The two gains add. Argmin interior in every row (0.75 stint, 0.50 game, both targets, both K).
- **The knobs sit on different axes.** xPTS(ft) costs the consensus nothing; `c_def` costs exactly
  what it cost before, at every target. A frontier, not a winner. The project ships a rating, so
  `c_def` stays 0 and the luck adjustment was the direction to push.

**2. Stage 4, the shot term (`src/eracoef/xpts.py`, `scripts/43_xpts_gate.py`,
`scripts/44_yoy_shot.py`).** Built to the spec: condition on the possession's first attempt,
marginalise everything downstream with the lineup's four cross-fitted factor rates, geometric
closure with the turnover leak, per-era league constants from the counters, lineup-level multiplier
clip, factor lambdas fixed at the 2024-26 selection. It lives at window-build time
(`xpts.xpts_design`), with the rate fits cached per block in `data/xpts/` (443 MB, gitignored).

It **closes**: expected attempts within 0.4% of observed on every one of 112 training blocks,
points ratio 0.994-0.999, no clipping, `mult` and `add` eFG variants indistinguishable. Stint sd
63.6 → 27.8, team-game sd 12.9 → 9.2. `tests/test_xpts.py` checks the closure on simulated
possessions.

It **loses**, out of season, everywhere:

| target (c_def 0, K=2) | stint | game | paired vs pts, game | sd_def of ratings |
|---|---|---|---|---|
| pts | 3627.97 | 113.31 | — | 1.00 |
| xpts_ft | 3627.83 | 113.24 | −0.08 (z −2.2, 20/28) | 0.99 |
| xshot mult | 3628.95 | 113.61 | +0.32 (z +1.8, 9/28) | **0.72** |
| xshot, league make rates | 3631.90 | 115.23 | +1.96 (z +6.6, 2/28) | 0.61 |

The ordering `pts < lineup-scaled < league-rate` is the mechanism: shot-making is a large real
signal (about 15% of everything the ratings know), the lineup eFG fit recovers ~80% of it, and the
luck removed does not pay for the rest. It shows on defense — defensive spread down 28%, offensive
down 4% — because the eFG factor shrinks defense harder (`lam_ratio` 1.50), so shot suppression
reaches the target through a heavily shrunk rate while the points RAPM reads it off the makes.

**What generalises:** free-throw luck is separable because a free throw belongs to one identified
player and no defender. Field-goal luck is not separable from skill at lineup level with these
tools. The re-parameterisation trap the previous handoff named arrived in partial form: three
quarters of expected points is `att1 x league make rate x lineup eFG`, and the lineup eFG is a
shrunk estimate.

**Stage 5 is off.** It was conditional on stage 4 paying.

---

## Part 2: what to do next

There is no obvious next step on the target. If someone wants to keep pulling on it, in order:

1. **The rebound piece alone.** Keep the realised first-attempt outcome, marginalise only the
   continuation. Needs two counters the stints do not carry — points on the first attempt including
   its free throws (`pts1`) and whether it produced a rebound chance (`chance1`) — so
   `STINT_SCHEMA` 4 and the 45-minute rebuild. Then `xpts = pts1 + chance1 * r * V` with everything
   already in `xpts.py`. This says whether offensive-rebound luck is separable even though shot luck
   is not. It is the only untested piece with a mechanism argument for it (a rebound depends on ten
   players' positions, not one player's release; but the OREB% factor has the best split-half
   reliability of the four, 0.738, and it is the piece that is mostly about the lineup).
2. **A partial adjustment** `y = pts - a * (pts - xpts)`, `a` in [0, 1]. Legal to select on the
   criterion; report on seasons not used to select.
3. Nothing else from the original luck plan. And-one luck and opponent free-throw luck are inside
   the free-throw term's logic and would be marginal.

Otherwise the open item is the same as before this work started: the one remaining xfail in
`tests/test_vs_consensus.py` is an attribution defect on backup bigs, and no target change and no
prior touches it.

**Merging to `main`** is probably wanted; nothing in this session was merged or committed.

---

## Part 3: state of the machinery

### Built this session (all uncommitted on `hybrid-and-xpts`)

| file | what |
|---|---|
| `src/eracoef/xpts.py` | stage 4: `league_constants`, `expected_points` (variants `mult`, `add`, `league`), `gates`, `lineup_rates` (cached), `xpts_design` |
| `src/eracoef/design.py` | `WindowData.counters` (the offense's per-possession counters per row, plus `pts`, `poss`), `WindowData.with_target` |
| `scripts/42_defweight_xpts.py` | the `c_def` x target sweep with consensus columns; `outputs/defweight_xpts*.parquet` |
| `scripts/43_xpts_gate.py` | the closure gates on one window; `outputs/xpts_gate.parquet` |
| `scripts/44_yoy_shot.py` | the out-of-season test by target; `outputs/yoy_shot.parquet` (full run), `outputs/yoy_shot_league.parquet` (the diagnostic), `outputs/yoy_shot_gates.parquet` |
| `tests/test_xpts.py` | 6 tests; the suite is 42 passed, 1 xfailed |
| `HANDOFF_luck_archive.md` | the previous handoff |

### Settled, do not redo

Everything in the previous handoff's list still holds (attempt definition, the `shift(-1)`
artefact, four-factor reliability, the estimated O/D asymmetry, FT padding k, the booster). Added:

- **The shot term does not pay, and the reason is measured** (Part 1). Do not rebuild it with a
  different eFG scaling — `mult` and `add` are identical to four figures, neither clips.
- **The closure itself is right.** Attempt and points closures hold per season and out of sample
  with no calibration. If a future variant misses the band, the variant is wrong, not the anchors.
- **Factor lambdas are fixed** (`xpts.FIXED_LAMBDA`, from `scripts/37_factors.py`). Re-selecting per
  block would make the target a function of the block.
- **`c_def` and xPTS are orthogonal knobs**: one trades prediction against attribution, the other is
  free on attribution. Consensus columns at a given `c_def` are the same to three decimals
  whichever target sits under them.

### Traps specific to this codebase

All of the previous list still applies (grid-edge argmax, `STINT_SCHEMA`, `nohup` survivors,
`w` is not possessions, never write per-player factor effects). Added:

- **A bug in the closure can leave the attempt closure exact while the points closure is off.** The
  first version of `xpts.py` had the continuation count right (multiplier gate passed) and the
  continuation points 2x too large (points ratio 1.12). Gate both, separately, always.
- **A per-row multiplier clip on the realised bucket mix clips the wrong thing.** Short stints with
  one three-point attempt and a strong rebounding lineup hit the band at 0.6%; the clip belongs on
  the lineup's continuation factor, not the row.
- **`build_window` in-process caches stints; `xpts_design` caches rates on disk** by block label,
  with `+`-joined labels for a block with a hole (`1997+1999`). The cache checks row count and stint
  indices, so a stale file raises rather than being reused, but a change to the factor design or
  lambdas needs `data/xpts/` deleted by hand.
- **The two idle `python.exe` processes** from 9/3 23:25 (pids 43140/63880, ~2 s CPU total) are not
  a runaway job; they were left alone.

### Verification

```
.venv/Scripts/python -m pytest tests -q                       # 42 passed, 1 xfailed
.venv/Scripts/python scripts/43_xpts_gate.py 2024-2026        # ~1 s with the rate cache, ~1 min without
.venv/Scripts/python scripts/42_defweight_xpts.py 1998 2025 --k=2,4     # ~8 min
.venv/Scripts/python scripts/44_yoy_shot.py 1998 2025 --k=2,4 --c=0,0.5 # ~10 min with cache; overwrites outputs/yoy_shot.parquet
```

Never bare `python`. Run heavy numpy jobs one at a time. Write source files with the Write/Edit
tools; short `cat >> file << 'EOF'` appends did work this session.

### The standing warning, earned again

The previous handoff predicted stage 4 would be "the remaining and larger piece". It was larger,
and it went the other way. The closure was specified correctly, built correctly, gated correctly,
and lost — and the loss was only visible in the out-of-season test, never in any within-window
gate. The gates say the target is a good expectation; only prediction says whether an expectation
is what the ratings should be fit to. Measure first.
