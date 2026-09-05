# Handoff: one evaluation system, a rank map that ships, and a shooter target that does not

Written after the session that turned the out-of-season criterion into one module, built the
shooter-level expected-points target with a shot-location curve and rejected it, and found and
shipped the rank-calibration correction. `FINDINGS.md` section 18 is the record; sections 16-17
are the two sessions before. `HANDOFF_shot_archive.md` is the previous version of this file.

**Branch:** everything is committed on `hybrid-and-xpts`; nothing merged to `main`. Pages serves
`/docs` from `main`; `scripts/07_plots.py` has not been re-run on the new board.

---

## Part 1: where things stand

### The chosen system changed twice, both times by the criterion

**hybrid + xPTS(ft) + the rank map, `c_def = 0`.** `config.yaml` -> `ratings_prior` now carries
`target: xpts_ft` (the free-throw-adjusted target the previous session chose but never wired into
`08_ratings.py` -- the board had been on raw points) and `rank_map` (below). Consensus, read once:
total 0.894 / offense 0.879 / defense 0.888, defensive spread ratio 1.15 (was 1.23), archetype bias
0.18 (was 0.19). `tests/test_vs_consensus.py`: 10 passed, 1 xfailed, unchanged.

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
2. **The rank map's mechanism** is not separated: the bench box prior over-stating how bad a
   low-usage player is, or the ridge's shrinkage being too light at the low end. Test by running
   `--rank` on `rapm` (no prior) and `pi`: if the curve is flat without a prior, it is the prior.
   Either way the map is the right fix for the product; this is about understanding it.
3. **The `c_def` frontier** (section 17) is still available for a forecasting product; the rank
   map and `c_def` are likely additive but untested together (`hybrid_xft_c0.5` + `--rankmap`).
4. **Lineup-level calibration**: lineups predicted to be the worst score 0.7-0.8 of their deficit.
   The player-level map fixes part of that; `--rank` on the mapped system shows what is left.
5. The remaining xfail (backup bigs) is untouched by any of this.

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
