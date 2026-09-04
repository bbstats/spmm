# Handoff: finish the luck-adjusted target

Written after the session that shipped the hybrid prior, built the per-possession counters and the
four-factor RAPMs, landed the free-throw luck adjustment, and — the important part — built an
internal criterion that can actually select things. `FINDINGS.md` sections 13-16 are the detailed
record. `HANDOFF_hybrid_archive.md` is the previous version of this file,
`HANDOFF_diagnosis_archive.md` the one before that.

**Branch:** all of this is on `hybrid-and-xpts`, not merged to `main`. Pages serves `/docs` from
`main`, so merging is probably wanted.

---

## Part 1: where things stand

### Shipped

The **hybrid prior**: box score priced to predict a PLAYER on offense, **no box prior at all on
defense**. `config.yaml` → `ratings_prior`; `windows.hybrid_beta`; `scripts/08_ratings.py`. It is
ONE fit — the box term is an offset `Xbox @ beta` with separate offensive and defensive columns, so
zeroing beta's defensive half IS "no defensive prior". No per-side lambda machinery.

Against the consensus (2024-26, 475 players, read once): total **0.896** vs 0.784 before, defense
0.888 vs 0.755, defensive spread 2.13 → 1.23, archetype bias +0.63 → +0.20. Five of the six strict
xfails flipped and are guards now; the suite is 36 passed, 1 xfailed.

### The criterion, which is the main asset from this session

`scripts/38_yoy.py`. Hold out season H, train on a **symmetric** neighbourhood excluding it (K=2 is
{H-1, H+1}; K=4 is {H-2..H+2} minus H), predict every stint of H from the ten players' ratings plus
an intercept and home term refit on H, score possession-weighted squared error in points per 100 —
at stint level and again after summing each team's points within a game.

**It is the first criterion here that is both external to the model and legal to select on.** The
next-window on-court benchmark shares the estimate's blind spot; the consensus can only be read,
never fit to. This one scores actual points in a season the model never saw, using no outside data.

Symmetric-around-H rather than train-on-a-block-predict-the-flanks, because only the symmetric form
can compare systems that pool different numbers of seasons: the held-out season stays identical
across every method and every K, so comparisons are exactly paired, and aging cancels.

### Results, 28 held-out seasons

Baseline with no player ratings (intercept + home only): stint **3653.18**, game **125.61**.

| model | K | stint | game | stint gain | game gain |
|---|---|---|---|---|---|
| RAPM, no prior | 2 | 3635.31 | 114.62 | 0.49% | 8.75% |
| PI-RAPM, team-priced | 2 | **3626.81** | 113.58 | 0.72% | 9.57% |
| hybrid (ships) | 2 | 3627.97 | 113.31 | 0.69% | 9.79% |
| hybrid + xPTS(ft) | 2 | 3627.83 | **113.24** | 0.69% | **9.85%** |
| RAPM, no prior | 4 | 3632.26 | 113.76 | 0.57% | 9.43% |
| PI-RAPM, team-priced | 4 | **3626.06** | 113.26 | 0.74% | 9.83% |
| hybrid (ships) | 4 | 3626.92 | 113.07 | 0.72% | 9.98% |
| hybrid + xPTS(ft) | 4 | 3626.82 | **113.04** | 0.72% | **10.01%** |

Three things to carry forward:

- **The ranking depends on the aggregation.** PI-RAPM wins every stint row; it is third of four at
  game level. Same predictions, same weights — only where errors cancel differs.
- **Game level is the meaningful magnitude.** Ratings remove ~10% of game error and only ~0.7% of
  stint error, because stint error is nearly all binomial noise. The project owner selected on game
  level; **hybrid + xPTS(ft) is the chosen system.**
- **K=4 beats K=2 for every method**, so pooling more seasons helps. That is the estimand-mismatch
  question from two handoffs ago, answered.

### The tension that is not resolved, and must not be quietly dropped

`scripts/41_defweight.py` sweeps one scalar `c_def` on the defensive half of the team-priced beta,
with the player-priced offense fixed. `c_def=0` is the shipped hybrid, `c_def=1` is PI-RAPM.

| c_def | stint | game | consensus total | consensus defense | archetype bias |
|---|---|---|---|---|---|
| 0.00 | 3627.97 | 113.31 | **0.896** | **0.888** | **+0.195** |
| 0.25 | 3627.05 | 113.01 | 0.878 | 0.872 | +0.376 |
| 0.50 | 3626.56 | **112.89** | 0.854 | 0.839 | +0.498 |
| 0.75 | **3626.49** | 112.93 | 0.824 | 0.802 | +0.575 |
| 1.00 | 3626.85 | 113.15 | 0.791 | 0.766 | +0.623 |

**Both aggregations have an INTERIOR optimum** (0.75 stint, 0.50 game). Section 13 recorded that no
criterion this project had could select the defensive shrinkage — within-season stint MSE prefers
the wrong end at 3.4 SE, the next-window benchmark is flat then declines. This one chooses. That is
the methodological result, independent of the value.

**And the consensus is monotone the other way**, read once after selection. Every step that improves
prediction costs rank agreement and adds archetype bias.

Both are real and they are not in conflict. The defensive box prior gets the lineup **sum** right
while getting the **split** wrong: a defensive rebound is available because someone else contested
the shot, so crediting the rebounder predicts his lineups well and describes him badly. Predicting
held-out points mostly rewards the sum. `scripts/40_movers.py` confirms it — the gap between hybrid
and PI-RAPM **halves as rosters turn over** (+1.94 with no movers on the floor, +0.91 with 3+), the
signature of credit that does not travel, but it does not reverse, so PI-RAPM's edge is not purely
artifact. Roster churn is ~50% a season, which is why the test has real attribution power and also
why it still is not a pure attribution test.

`scripts/39_why.py` ruled out the obvious alternative: the hybrid's deficit is **not** a coverage
problem. It is worst among well-established players (+1.63, z=+5.07) and smallest among players with
no training exposure at all.

---

## Part 2: what to do next

### 1. The untested combination — do this first, it is cheap

**The `c_def` sweep was run WITHOUT xPTS, and the four-system table was run at `c_def=0`.** So the
chosen system has never been measured at the `c_def` the criterion actually wants. `c_def=0.5`
scored 112.89 at game level against the hybrid's 113.31 — a bigger gain than xPTS delivered — so the
combination is very likely better than either alone.

Run the sweep with `target="xpts_ft"`, reporting game-level error and the consensus columns side by
side. If it lands where the pieces suggest, the honest framing is a **frontier**, not a winner:
`c_def` buys prediction and sells attribution, and xPTS buys a little of both.

### 2. Stage 4: the shot term and the geometric closure

Only free throws are luck-adjusted so far. The remaining and larger piece:

```
xpts(p) = x1(b) + fta1(p)*q + m1(b) * r * V2         (0 if the possession reaches no attempt)
V2      = (1 - t_sc) * x2 / (1 - m2 * r)
x2      = 2 * eFG_putback(lineup) + f*q
```

- `x1(b) = 2 * p_make(b)`, with `p_make(b) = p_make_league(b) * (eFG_lineup / eFG_league)` —
  **multiplicative on the make probability, not additive**, or a rim-heavy lineup's make probability
  exceeds 1 in the tail. One-line switch; test both.
- `r` is the OREB lineup rate, `f` the FT-rate lineup value, `t_sc` the TOV lineup value. All four
  factors are needed even though eFG and OREB do the work.
- **Condition on the possession's FIRST attempt; marginalise everything downstream of it.** This is
  the central design decision and the original spec left it out. Taken literally, "never use the
  number of attempts that happened" makes xPTS a deterministic function of the lineup rates, every
  possession in a stint identical, and the points RAPM a re-parameterisation of four fits whose
  individual splits we refuse to publish. The first attempt is not downstream of any rebound, so
  conditioning on it introduces none of the luck being removed. Use `att1_*`, `fta1`, `fta_tech`;
  do NOT use `fga`, `fta`, `reb_cont` (those are four-factor targets only).
- **Where it lives:** NOT in the stints. It depends on fitted lineup rates, which depend on the
  window, which is built from the stints — circular. Compute at window-build time, optionally cache
  at `data/xpts/{window}_{variant}.parquet`. (The FT term is different and *is* stored, as `xftm`,
  because it is a fixed linear combination of counters with no fitted quantity in it.)
- **Gates:** mean multiplier in [1.13, 1.20]; per-lineup multiplier clipped to [1.05, 1.40] with
  clip rate <0.5%; aggregate `sum(xpts)/sum(pts)` in [0.995, 1.005], with a single global affine
  calibration per season fitted on the training block only if it misses — never per lineup.
- **The trap that will fake a win:** the plain A/B cross-fit of the four factors assigns half-A rows
  the fit trained on half B. If half B is also the evaluation set, the evaluation half's outcomes
  leak into the training target. Not an issue for the out-of-season test (the held-out season is
  never in any training block), but it *is* an issue for any within-season split.

### 3. Then stage 5, only if 4 pays

And-one luck, opponent free-throw luck, the two-stage putback refinement. Diminishing returns.

---

## Part 3: state of the machinery

### Built this session

| file | what |
|---|---|
| `src/eracoef/stints.py` | 35 per-possession counters (`POSS_COUNTERS`), `xftm`, `STINT_SCHEMA=3` |
| `src/eracoef/design.py` | `TARGETS` — `target=` parameter on `build_design`; `xpts_ft` derived there |
| `src/eracoef/factors.py` | four-factor RAPM, lineup sums only, per-factor lambda selection |
| `src/eracoef/windows.py` | `player_priced_beta`, `hybrid_beta` |
| `src/eracoef/boxtable.py` | `ft_padding` (method of moments), `ft_totals` |
| `scripts/33-34` | the hybrid: penalty path under both criteria, single-fit verification |
| `scripts/35-37` | attempt definitions, counter gate, four-factor fits |
| `scripts/38-41` | the out-of-season criterion, the two diagnostics, the `c_def` sweep |

### Settled, do not redo

- **The old rebound anchors (OREB% 24.4, multiplier 1.15) are the box-score convention and wrong
  for this model.** Counting only FGA misses 9% of possessions, because a shot drawing a shooting
  foul records no FGA. An attempt = FGA + every non-and-1 free-throw trip. Era-dependent: 2024 is
  27.1% / 1.136, 1998 is 33.8% / 1.213, so gates must be per season.
- **The "89.4% of misses are followed by a rebound" figure is a `shift(-1)` artefact.** A ≤7-event
  forward scan finds ~100%.
- **Four-factor split-half reliability of the lineup sums** (the go/no-go for stage 4): eFG 0.666,
  TOV 0.754, OREB 0.738, FT rate 0.762, against a 0.30 threshold. Calibration within 0.55%.
- **The O/D asymmetry is estimated, not imposed**: `lam_ratio` (= λ_D/λ_O) is 1.50 for eFG and 3.00
  for OREB — little defensive skill — but **0.75 for turnovers**, so forcing them is a real
  defensive skill. Do not hard-code the asymmetry; the fit finds it.
- **FT padding k is 22-27 attempts**, measured by method of moments, not the 40 that was assumed.
- **The booster is out** (`ratings_prior.boost: false`): 0 on total, −0.020 on offense.

### Traps specific to this codebase

- **An argmax on a grid boundary has not chosen anything.** This recurred twice in one session —
  the ladder's defensive knobs, and OREB% selecting the top of `config.yaml`'s `lam_ratio_grid`.
  `factors.py` now uses a wider grid and flags edge selections. Check the argmax is interior.
- **`STINT_SCHEMA` is checked on load and raises `StaleStintCache`.** Bump it whenever the stint
  schema changes; a full rebuild of 60 season-phases is ~45 minutes at ~85s a season.
- **A `nohup`'d background job survives the tool wrapper.** A harness "task completed" notification
  is the wrapper exiting, not the job. An empty log means buffered, not dead. One such job ran on
  undetected for two hours overwriting freshly rebuilt files with stale-schema output, and the only
  symptom was each integrity check flagging one more file than the last. Check
  `Get-CimInstance Win32_Process -Filter "Name='python.exe'"` and read the CommandLine.
- **`w` is `poss * gt_weight`, not possessions.** Converting a rate back to points needs
  `wd.rows["poss"]`. Harmless at the shipped `gt_weight: 1.0`, wrong under the 0.5/0.0 robustness
  runs.
- **Never write the per-player four-factor effects anywhere.** Splitting a lineup's rebounding among
  five players is the badly-identified direction; only the lineup sums are trustworthy.

### Verification

```
.venv/Scripts/python -m pytest tests -q                      # 36 passed, 1 xfailed
.venv/Scripts/python scripts/36_counter_check.py 2024,2025,2026
.venv/Scripts/python scripts/37_factors.py 2024-2026         # the stage 2 gate
.venv/Scripts/python scripts/38_yoy.py 1998 2025 --k=2,4     # ~3 min
.venv/Scripts/python scripts/41_defweight.py 1998 2025       # ~70 s
.venv/Scripts/python scripts/02_stints.py 1997 2026 RS,PO --force   # ~45 min, only if the schema changes
```

Never bare `python` — the system interpreter is 3.10 without numpy. Run heavy numpy jobs one at a
time. Write source files with the Write/Edit tools, not shell heredocs.

### One standing warning, still earned

Across these sessions, every mechanism asserted before measurement was wrong at least once: the
archetype tilt, the booster's value, player history, the rebound-linkage rate, the coverage
explanation for the hybrid's deficit, and the assumed FT padding constant. Every one was caught by
measuring. The counters' continuation identity alone flushed out four cases the spec did not
anticipate. Measure first.
