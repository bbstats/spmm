# Handoff: one evaluation system, a rank map that ships, and a shooter target that does not

Written after the session that turned the out-of-season criterion into one module, built the
shooter-level expected-points target with a shot-location curve and rejected it, and found and
shipped the rank-calibration correction. `FINDINGS.md` section 18 is the record; sections 16-17
are the two sessions before. `HANDOFF_shot_archive.md` is the previous version of this file.

**Branch:** everything is committed on `hybrid-and-xpts`; nothing merged to `main`. Pages serves
`/docs` from `main`; `scripts/07_plots.py` has not been re-run on the new board.

---

## Part 1: where things stand

### The chosen system changed three times, each time by the criterion

**Board = offense from the free-throw target, defense from the opponent-3PM-replaced target
(`x3def`), rank map on top, `c_def = 0`.** `config.yaml` -> `ratings_prior`: `target: xpts_ft`,
`defense_target: x3def`, `rank_map` (table `outputs/holdout_def3_rank.parquet`, system `def3_p0`,
K=4). `08_ratings.py` runs two fits per window and takes every `*_def` column from the second.
Out of season it is 112.54 / 112.33 at team-game level (K=2 / K=4) against 112.83 / 112.68 for the
free-throw system with the map and 113.24 / 113.04 for the free-throw system alone.

`x3def`: every opponent three-point make becomes 3 x the shooter's 3P% from the OTHER half of the
block, padded toward the league with k = 450 attempts (the split-half reliability of high-volume
shooters; Blackport's 750 is the same order). Opponent 3P% needs 4,000-7,000 attempts to stabilise
(Nylon Calculus 2018), so what it removes is almost all noise. Seasons before the block add nothing
(p1, p2 within 0.001 of p0). Never leave-one-game-out: that estimate is negatively correlated
with the game's own outcome by 1/(N-1) (distributional bias, Science Advances 2025); the other
half of the block is not.

**Consensus, read once: total 0.882 / offense 0.879 / defense 0.814.** The defensive drop from
0.888 is the blend's composition, not a defect: the owner's defense blend is 50% xRAPM + 40% EPM
(raw points) + 10% LA-RAPM. Against each component separately the new defense agrees LESS with
every raw-points metric (td_drapm 0.933 -> 0.818, xDRAPM 0.907 -> 0.800, EPM 0.844 -> 0.794) and
MORE with every luck-adjusted one (td_ladrapm 0.720 -> 0.758, xDLEBRON 0.589 -> 0.636, DLEBRON
0.671 -> 0.704). The players who rise are drop-coverage centers (Sabonis 197 -> 56, Valanciunas
302 -> 80, Poeltl 94 -> 15); the fallers are perimeter role players. `22_vs_consensus.py` section 7
now prints the per-component agreement; `data/external/impact_metrics_2526.csv` is the source.
The owner's words: the blend is a sanity check; the game-level criterion is the test.
`tests/test_vs_consensus.py`: defensive guard lowered to 0.78 with the reason in its docstring;
10 passed, 1 xfailed.

### The evaluation system (`src/eracoef/holdout.py`, `scripts/45_holdout.py`)

Everything is one call now. `Holdout.from_config(cfg).run(systems, ctx)` scores any `System`
(`fit(train_seasons, ctx) -> Ratings`) on every held-out season at stint and team-game level,
with paired tests, movers/exposure splits, rank calibration, the consensus read and two
rating-semantics diagnostics. `45_holdout.py` is the single CLI (its docstring maps each deleted
script to an invocation) and reproduces the old `38_yoy.py` to 1e-12. Scripts 38-42 and 44 are
deleted. `scripts/48_ladder.py` applies the stop rules to any set of result files. Tests run
against `simulate(rho=0.85, turnover=0.3)`, which now has persistent talent and roster churn.

Two things learned building it, both in the code comments:
- **One slope per decile is not identified** with five players a side (headcounts sum to a
  constant, collinear with the intercept); the slope is a smooth curve in the rating.
- **The level refit matters on the simulator** (`holdout.level`: "home" or "full"); on real data
  the ratings were fit with the fixed block, so "home" stays the default.

### The rank map (ships)

Out of season, the worst offensive decile wants its ratings multiplied by 0.77 and the best by
1.01; defense the same in the good direction (0.78-0.84 for the worst, 1.05 for the best). Monotone
through every decile, both K, z to -10. `RankMappedSystem` fits a monotone per-side curve on the
pooled slopes leave-one-season-out and wins: team-game -0.41 per 100 (z -6.2, 26 of 28) at K=2,
-0.36 (z -5.7, 25 of 28) at K=4; stint -0.31 / -0.28. It preserves every within-side rank, so it
buys prediction (the same size as `c_def = 0.5` did) at no attribution cost. `08_ratings.py`
applies it from `outputs/holdout_final_rank.parquet`; raw columns kept as `rating_*_raw`.

### The shooter-level target (negative, measured, done)

Built to the owner's spec: the shooter's own padded 2P%/3P%/FT% from the other half of the block,
a per-season shot-distance curve (`shotcurve.py`), the counts by lineup slot in the stints (schema
4, `SLOT_COUNTERS`), priced at window-build time (`xshoot.py`). Every gate passed. Every
whole-target system loses (game level +1.5 to +2.4 per 100, z +5 to +8); splits with defense from
points are flat. The padding constants explain it: k = 175 attempts for 2P%, 226 for 3P%, 24 for
FT%. A half-season 3P% is trusted half-and-half with the league, so the expectation is mostly the
league rate, and the league-rate variant had already lost in section 17. The location curve is
worth 0.7 per 100 against the flat rate; pooling four seasons instead of two moves the gap from 1.71
to 1.52. Keep the machinery (the slot counters cost nothing and the curve is right); do not
rebuild the target with a different pooling.

---

## Part 2: what to do next

1. **Merge and republish.** `scripts/07_plots.py` on the new board, merge to `main`.
2. **DONE -- the rank map's mechanism** (FINDINGS 18, last subsection). Offense: the curve is the
   box prior's (no-prior ridge is calibrated at the bottom and 1.6x too narrow at the top; the
   player-priced prior fixes the top, overshoots the bottom). Defense: the ridge's (the hybrid has no
   defensive prior and its curve is the no-prior curve). Upstream fixes, if wanted: a prior that is
   nonlinear at the low end on offense; a heavier or rating-dependent defensive penalty. The map
   on the finished rating does both.
3. **DONE -- the map and `c_def` add**: −0.91 per 100 at game level together (z −10.9, 27/28).
   The forecasting product is `rankmap_hybrid_xft_c0.5` (game error 112.3, consensus 0.854); the
   rating product ships without `c_def` (112.8, consensus 0.894).
4. **Lineup-level calibration**: lineups predicted to be the worst score 0.7-0.8 of their deficit.
   The player-level map fixes part of that; `--rank` on the mapped system shows what is left.
5. **LATER (owner's request): the lineup three-point factor.** Replace each opponent three by
   3 x shooter's rate x the defensive lineup's fitted 3P% factor (a four-factor RAPM on opponent
   3P%, lineup sums only, cross-fitted, REML-shrunk), so the repeatable part of conceding open threes
   stays in the defensive target and only the residual noise leaves. `design.TARGETS` gets
   `{"fg3pct": num fg3m, den fg3a}`, `factors.py` handles the fit, `xshoot.def_three_design` gets a
   `lineup=True` switch; ~30 min to build, ~10 to run. Tells whether the drop-coverage centers'
   rise under `x3def` is luck removed or scheme forgiven.
6. **OWNER'S CALL (2026-09-05): the offensive prior is too strong at the top of the 1997-99
   board.** Stockton #1 (prior 4.62 + residual 0.56) over Jordan (4.31 + 0.96) is the prior, not the
   floor: the prior's spread is 1.76 per 100 against 0.36 for the on-court residual, correlation
   with the final offensive rating 0.98, and it is era-flat (a 1997 assist priced like a 2024 one;
   ast +0.32, stl +0.67, fg3m +1.37 per unit of the per-100 rate). The owner's suggestion: if there
   is a single prior, make it a GBDT on the box rates rather than a linear fit -- a gradient-boosted
   model of next-window on-court impact from the 13 rates (plus era/season as a feature so it is
   not era-flat), cross-fitted by window exactly as `player_priced_beta` is, then used as the
   offset in the same one-penalty fit. Not the old LRBoost (a correction on top of the linear
   prior, measured at 0): the prior itself. Test on the criterion against `def3_p0`, and read the
   rank curve -- a good nonlinear prior should flatten the offensive 0.80-1.01 curve by itself.
7. **A nonlinear offensive prior** (item 6 is one way) would be the principled replacement for the map on offense:
   fit `player_priced_beta` with a rank-dependent term, or apply the map to the prior before the
   ridge rather than after it, and test both against `rankmap_hybrid_xft` on the criterion.
6. The remaining xfail (backup bigs) is untouched by any of this.

---

## Part 3: state of the machinery

### Built this session (all committed)

| file | what |
|---|---|
| `src/eracoef/holdout.py` | the system: Context, PluginSystem / SplitSystem / MappedSystem / RankMappedSystem / TableSystem, predict_season, score, splits, Holdout.run, pooled / paired / report, rank_calibration / pooled_rank / fit_rank_map, vs_consensus, team_residual, replacement_quality |
| `src/eracoef/pad.py` | `shrink`, `mom_k`, `pad_rate`: the one padding helper; `ft_padding` and `BoxExposure` delegate |
| `src/eracoef/shotcurve.py` | per-season make probability by distance; `shot_bin` is the one location rule; cached in `data/shotcurve/` |
| `src/eracoef/xshoot.py` | shooter-level pricing of the slot counters; `TARGET_REGISTRY` |
| `src/eracoef/stints.py` | schema 4: `SLOT_COUNTERS`, `pts1`, `chance1`, stored `half`, per-game shots table (`{season}_{phase}_shots.parquet`) |
| `src/eracoef/design.py` | counters carry the slot columns, player ids by slot and the half; `_order_games` uses the stored half |
| `src/eracoef/simulate.py` | `rho`, `turnover`; `team` in the truth |
| `scripts/45_holdout.py` | the CLI; `--rankmap=<rank parquet>` adds `rankmap_<system>` |
| `scripts/46_xshoot_gate.py`, `47_shotcurve_gate.py`, `48_ladder.py` | the gates and the verdict table |
| `scripts/08_ratings.py` | `ratings_prior.target`, `ratings_prior.rank_map` |
| `tests/test_holdout.py`, `test_pad.py`, `test_shotcurve.py`, `test_xshoot.py`, `test_stints.py` | 60+ tests, 1 xfail |

### Settled, do not redo

- Everything in the previous two handoffs' lists still holds.
- **Shooter-level expectation loses at every pooling, with or without location.** Measured. The
  padding constants are the reason and they are properties of the sport.
- **An alignment is a scalar on the level, never an affine.** A row-level affine has slope ~0.6
  and is a partial shrink of the target in disguise, firing in some seasons and not others.
- **The stored half and the design's half are identical by construction** (`stints.assign_halves`
  = `design._order_games`); a game that produced no stints is skipped by both.
- **The per-bin curve gate is per season and per shot value**, unlocated cells included; the
  unlocated-twos cell is heaves (2% make rate) and is padded lightly for that reason.

### Traps specific to this codebase (added)

- **Background Bash has a 10-minute wrapper limit.** Anything longer must be `nohup ... &` from a
  subshell, watched with a `Monitor` that checks the pid via `ps -W` (not `tasklist /FI`, which
  Git Bash mangles into a path). The stint rebuild is ~45 min; the ladder ~6 min per K.
- **`ps -W | grep name` does not see arguments**; check by pid, or via PowerShell's
  `Get-CimInstance Win32_Process` with `$_.CommandLine`.
- **The stint frames are ~2x wider now** (156 slot columns). `windows._STINT_CACHE` holds every
  season loaded in-process; a 28-season run fits, but do not raise `Context.cache_size` casually.
- **`Ratings.aligned` merges on player_id**; with `player_unit: season` a block has one rating per
  player-season and `ratings_from_fit` pools them by possessions. Real data uses `window`.
- **The two idle `python.exe` from 9/3** are still there and still idle.

### Verification

```
.venv/Scripts/python -m pytest tests -q                                   # 60+ passed, 1 xfailed
.venv/Scripts/python scripts/47_shotcurve_gate.py                         # ~3 min; all 30 seasons within 2 points
.venv/Scripts/python scripts/46_xshoot_gate.py 2024-2026                  # ~5 s with the rebuilt stints
.venv/Scripts/python scripts/45_holdout.py --systems=hybrid,hybrid_xft --k=2,4 --rank --consensus --tag=x   # ~5 min
.venv/Scripts/python scripts/45_holdout.py --systems=hybrid_xft,rankmap_hybrid_xft --rankmap=outputs/holdout_final_rank.parquet --k=2,4 --tag=y
.venv/Scripts/python scripts/48_ladder.py outputs/holdout_ladder_k2.parquet outputs/holdout_ladder_k4.parquet
.venv/Scripts/python scripts/08_ratings.py && .venv/Scripts/python -m pytest tests/test_vs_consensus.py -q
```

Never bare `python`. Heavy numpy jobs one at a time; the runner is fast enough (~10 s per held-out
season for ten systems) that most comparisons fit in a few minutes.

### The standing warning, kept

Three things this session were asserted and then measured the other way: that pooling more
seasons would rescue the shooter rate (it moved the gap by 0.2 of 1.7), that one slope per decile
would measure rank calibration (collinear), and that the simulator's true ratings would score at
1.0 (its process was not stationary). Each was caught by a number. Measure first.
