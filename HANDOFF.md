# Handoff: OpenRAPM ships the multi-stage prior-informed RAPM (mspi); one-page site

Written after the session that built the owner's prior chain (APM -> Simple SPM -> ridge -> GBDT),
ran it on the criterion, and found it beats the unmapped board at game level but not the board as it
ships. `FINDINGS.md` section 19 is the record; `HANDOFF_rankmap_archive.md` is the previous version
of this file (sections 16-18 of FINDINGS are its context).

**Shipped (owner's call, 2026-09-05 evening, game level is all that matters):** `config.yaml` ->
`ratings_prior: offense gbdt, defense gbdt, role_prior spm, rank_map null`, `gbdt.mode: full`;
`scripts/08_ratings.py` builds the board through `spm.chain_offset` (the board is `mspi`: 112.06 at
K=4 against 112.33 for the previous board, 20 of 28 seasons; consensus 0.772 / 0.778 / 0.785 against
0.883 / 0.879 / 0.814, read once). The project is now called **OpenRAPM**: `README.md` says what the
test is and how to contribute a system; `docs/index.html` is one page (block picker, a table of
offense / defense / total sortable by column, no chart) reading `docs/data/ratings.json` from `scripts/52_site.py`;
the old coefficient page is `docs/coefficients.html`. Merged to `main` and pushed at the end of the
session (see git log). `tests/test_vs_consensus.py` guards were lowered to the new board's values
with the reason in each docstring.

---

## Part 1: where things stand

### The previous board: offense from the free-throw target, defense from x3def, rank map on top

`rankmap_def3_p0`: 112.54 / 112.33 at team-game level (K=2 / K=4), consensus 0.883 / 0.879 / 0.814.
It is the reference the chain was measured against (and `rankmap_def3_p0_rep` the fair one). `def3_p0` reproduces `outputs/holdout_def3.parquet` to 1e-14 through
the new code paths.

### The chain, and what the criterion said (FINDINGS 19)

Per side and per window: **APM** (the plugin ridge with beta 0 at penalty 100), the **Simple SPM** (a
possession-weighted ridge of APM on share of team possessions, starts share, age and their powers,
leave-window-out), **RAPM_1** (the shipped ridge pulled toward the SPM), and a **chimeraboost GBDT**
fit to RAPM_1 (a player's value pooled over his OTHER windows, from his rates + season, and in the
"full" mode the role inputs too), leave-window-out, the excluded window kept out of every target.

| system | game K=2 / K=4 | vs def3_p0 (unmapped) | vs rankmap_def3_p0 (ships) | stint |
|---|---|---|---|---|
| spm (role prior alone) | 113.69 / 112.73 | flat (+0.85 / +0.17) | | loses |
| mspi_resid (SPM + GBDT on the residual) | 113.33 / 112.55 | flat | | loses |
| **mspi** (the GBDT alone, role inputs as features) | **112.32 / 112.06** | **WINS -0.55 / -0.52 (z -2.4 / -2.7)** | flat -0.23 / -0.29 (z -1.2 / -1.8) | loses +1.05 / +0.86 |
| rankmap_mspi | 112.64 / 112.40 | flat | flat | loses |

So: the GBDT prior takes the same 0.3-0.5 per 100 from the board that the rank map takes, and the two
do not stack; the map on top of the GBDT costs at game level. Against what ships the chain is flat at
game level and loses at stint level. **Not shipped**, by the stop rules.

### The second round (owner: game level only; rank work dropped; the chain is `mspi`)

| what was tried | result (game level, K=2 / K=4) |
|---|---|
| garbage-time rows split (`--splits=gt`) | the chain is BETTER on them (-1.78, z -2.1); the 7+-bench loss is competitive second-unit minutes |
| `gt_weight` 0.5 / 0 in the fit, board and chain | nothing (chain -0.51 vs -0.55 against the board; board +0.02 against itself) |
| calibration by role (`51_garbage.py`) | chain offense: starters 1.20, bench 1.05, deep bench 2.10; defense calibrated. Board: bench offense 0.85 |
| GBDT prior scaled 1.25 / 1.5 / 2.0 | worse the more it is scaled; amplitude is not the lever |
| trained on unshrunk APM (`mspi_apm`); mixed sides (`mspi_mix`) | scales fixed, error unchanged (112.14 / 112.15 at K=4 vs 112.06) |
| **replacement level for unseen players (`ReplacementSystem`)** | **board + map + rep 112.34 / 112.20: -0.20 / -0.14 vs what ships, z -4.5, WINS.** chain + rep 112.11 / 111.91: -0.24 / -0.30 vs board + rep, z -1.2 / -1.9, flat |

The replacement level (absent players score the block's mean rating of under-500-possession players,
per side, training data only) is a free, legal improvement to the criterion's treatment of the shipped
board, consensus untouched. With both systems given it, the chain's edge is 0.2-0.3 per 100 at game
level in 16-20 of 28 seasons and not significant at both K. The owner then ruled that game level is
all that matters and shipped `mspi` (see the top of this file).

What it did answer:
- **The Stockton problem is the linear prior's amplitude.** Under `mspi` on 1997-99 Jordan is
  first (prior 3.16 + residual 1.62) and Stockton seventh; the offensive prior sd is 1.15 against the
  linear prior's 1.69, the residual 0.67 against 0.40.
- **The era term is worth almost nothing**: an assist moves from 0.14 to 0.16 per 100 between 1998
  and 2025, threes 0.26 to 0.25 (`outputs/csv/gbdt_pdp.csv`); Boruta rejects `season` outright in the
  residual mode.
- **The rank curve flips**: the GBDT prior is calibrated at the bottom (1.02) and under-rates the top
  (1.24); the linear prior was 0.81 / 1.03. The role prior alone makes the bottom worse (0.76).
- **Where it loses**: floors with 7+ bench players (18% of possessions): +1.7 at game level, +3.3 at
  stint level (z +5). Where it wins: one or two movers on the floor (-0.67, z -2.5), all-established
  floors (-0.59), guard-heavy floors (-0.68).
- **Consensus**: `mspi` 0.772 / 0.778 / 0.785 against 0.883 / 0.879 / 0.814, the largest drop
  a criterion winner has carried; reported only, per the owner's ruling.
- **Drag** (RLOOCV): at most 0.05 per 100 per leave-window-out set; immaterial, measured.

### The evaluation system, extended

- `Holdout.run(held=)` and `holdout.run_parallel(ho, names, workers=4)`: spawned workers rebuild the
  systems by NAME from `systems.registry(cfg)` (the offset closures do not pickle) with BLAS / numba
  threads pinned to 12 // workers. The full ladder (6 systems, 2 K, 28 seasons, splits, rank) ran in
  322 s. Scripts that spawn need a `__main__` guard (45 has one now).
- New splits `bigs` (top-tercile big-man score on the floor) and `bench` (starts share under 0.5 on
  the floor); `--spread` (prior sd vs residual sd), `--top=1997-1999`, `--pdp`, `--norun`.
- `PluginSystem(offset=)` carries a per-player prior through `plugin_fit(prior_offset=)`;
  `Ratings` now has `prior_o` / `prior_d`; `SplitSystem` carries them.

---

## Part 2: what to do next

1. **Done: merged and republished** as OpenRAPM with the one-page site. `07_plots.py` / `site.py` are
   the old coefficient page's generator and now write `docs/coefficients.html` only if re-run by hand
   (they still write `docs/index.html`; do not run them without redirecting, or the new page is lost).
2. **Decide the replacement level.** It is an evaluation-time rule (what an unseen player scores when a
   held-out season is predicted), so it changes no rating on the board; it does change the criterion's
   number for every system, by -0.14 to -0.20 for the board and more for the chain. Make it the
   default in `Holdout` (one line: wrap every system in `ReplacementSystem`) or leave it a named
   variant; either way the reference for the next candidate is `rankmap_def3_p0_rep`.
3. **A prior with the chain's bench and the linear prior's starters.** The role calibration says the
   chain's offense is right on the bench (1.05 against the board's 0.85) and too timid for starters
   (1.20 against 1.01); the linear prior is the other way round. One system: `mspi`'s offset below a
   possession or role threshold, the linear player-priced prior above it, blended over a band; test
   against `rankmap_def3_p0_rep`. Scaling the whole prior is measured not to work; a role-dependent
   blend is the untested thing.
4. **The deep bench's spread** (2.10 on offense): the GBDT cannot learn it from RAPM_1 (all role level
   for them) and `mspi_apm` learned it at the cost of defensive width. A per-side choice of target
   (APM on offense, RAPM_1 on defense) is `mspi_mix`, which is calibrated on both sides and no better;
   so the deep bench's spread is real but small in possessions. Low priority.
5. **Merge Boruta's tentative features properly**: 25 trials left drb, fg3_miss, pf, tov tentative on
   offense; the config keeps them. A 50-trial run takes ~35 min at 12 threads.
6. LATER (owner's request, unchanged): the lineup three-point factor (HANDOFF_rankmap_archive item 5).
7. The remaining xfail (backup bigs) is untouched.

---

## Part 3: state of the machinery

### Built this session

| file | what |
|---|---|
| `src/eracoef/roles.py` | roles table (games, starts, minutes, on-floor and team possessions, age), `player_season_inputs`, `window_inputs`, `design7` |
| `src/eracoef/spm.py` | `apm_fit`, `fit_spm` / `SPMFit`, `spm_offset`, `apm_lambda_check`, `chain_offset(sides, mode)` |
| `src/eracoef/gbdt_prior.py` | `training_rows` (pooled other-window target), `drag` / `counterbalance`, `GBDTPrior` (modes residual / full, cached per exclusion set), `gbdt_offset`, `partial_dependence`, `make_boruta` / `run_boruta` |
| `src/eracoef/systems.py` | `registry(cfg, rankmap)`: every named system, incl. `spm`, `mspi_resid(_o)`, `mspi(_o)` |
| `src/eracoef/holdout.py` | `PluginSystem.offset`, `ratings_from_fit(offset=)`, `Context.roles / rpanel / gbdt / mspi / bigness / bench`, `by_bigs`, `by_bench`, `run(held=)`, `run_parallel` |
| `src/eracoef/windows.py` | `player_ratings_table(prior_parts=)` |
| `scripts/49_role_panel.py` | the panel: APM, SPM (leave-window-out), RAPM_1 -> `outputs/role_panel.parquet`; `--check` for the APM penalty |
| `scripts/50_boruta.py` | BorutaShap per side and mode -> `outputs/csv/boruta_*.csv`, YAML for `gbdt.features_*` |
| `scripts/51_garbage.py` | calibration slope per role group (deep bench / bench / starters) per side, all rows / garbage time / competitive -> `outputs/csv/garbage_slopes.csv` |
| `holdout.ReplacementSystem`, `Ratings.fill_o/fill_d`, `by_gt` split | the replacement level for unseen players; garbage-time row split |
| `systems.registry`: `mspi_s*`, `mspi_apm*`, `mspi_mix`, `*_rep`, `*_gt05`, `*_gt0`, `rankmap_def3_p0_rep` | the second-round variants (`chain_offset(scale=, target=)`) |
| `outputs/holdout_{gt,scale,mix,rep}.parquet`, `outputs/garbage_slopes.parquet` | the second-round results |
| `scripts/45_holdout.py` | registry from `systems.py`; `--workers`, `--spread`, `--top`, `--pdp`, `--norun`; `main()` guard |
| `scripts/08_ratings.py` | the shipping switch (`ratings_prior.role_prior: spm`, `offense` / `defense: gbdt`), inert |
| `tests/test_roles.py`, `test_spm.py`, `test_gbdt_prior.py`, `test_holdout.py` (+3) | 77 tests, 1 xfail |
| `data/cache/roles.parquet`, `data/raw/bio/`, `outputs/role_panel.parquet`, `outputs/holdout_chain*.parquet`, `outputs/csv/{spm_coefs, spm_lambda_check, role_panel_report, boruta_*, gbdt_pdp, holdout_chain_*}.csv` | data and results |

### Settled, do not redo

- Everything in the previous handoffs' lists still holds.
- **The chain's arithmetic**: `share` = summed on-floor possessions over one full team-season (the
  owner's definition: games not played count); the GBDT must NOT be trained on RAPM_1 and then added
  to the SPM (double-counts the role level: offensive scale 0.71). Either the residual `u` under the
  SPM, or RAPM_1 with the role inputs as features and no SPM.
- **The rank map does not stack with the GBDT prior** (measured leave-one-season-out).
- **APM at penalty 100 is fine**: the SPM coefficients agree at 30 / 100 / 300 except the half-point
  defensive starts-share term.
- **The era feature is inert**: Boruta rejects it on the residual, PD is flat.
- **BorutaShap needs shims** (`np.NaN`, `scipy.stats.binom_test`, `check_model`, `explain`) and 20+
  trials to decide anything under its Bonferroni test.

### Traps specific to this codebase (added)

- **`tests/test_core.py::test_calibration_slope_vs_lambda` is flaky and pre-existing**: passes alone,
  fails when the rest of `test_core.py` runs first (the crossfit u slope comes out 1.49 against a
  1.25 bound). Reproduced on the committed tree with this session's changes stashed, so it is
  order-dependent state in that file (an unseeded draw somewhere before it), not the new modules.
- **Windows spawn re-imports the script**: any script that calls `run_parallel` needs
  `if __name__ == "__main__":`, or the workers re-run its top level and the pool breaks.
- **`uv pip install` needs the .exe path** (`--python A:/code/spmm/.venv/Scripts/python.exe`).
- **Do not run Boruta (12 numba threads) beside the ladder**; ~27 s per trial when they share cores,
  10 s alone.

### Verification

```
.venv/Scripts/python -m pytest tests -q                                   # 77 passed, 1 xfailed (test_core flaky, see above)
.venv/Scripts/python scripts/49_role_panel.py --check                     # ~2 min with stints cached; asserts xrapm_panel unchanged
.venv/Scripts/python scripts/50_boruta.py --trials=25 --threads=12        # ~30 min
.venv/Scripts/python scripts/45_holdout.py --systems=def3_p0,mspi --k=2,4 --rank --consensus --splits=bigs,bench --spread --workers=4 --tag=x   # ~4 min
.venv/Scripts/python scripts/45_holdout.py --systems=def3_p0,rankmap_def3_p0,mspi,rankmap_mspi --rankmap=outputs/holdout_chain_rank.parquet --k=2,4 --workers=4 --tag=y
.venv/Scripts/python scripts/48_ladder.py outputs/holdout_y.parquet --ref=rankmap_def3_p0
.venv/Scripts/python scripts/45_holdout.py --systems=def3_p0,mspi --norun --top=1997-1999 --pdp
```

### The standing warning, kept

Three things this session were asserted and then measured the other way: that the GBDT could be
trained on RAPM_1 and added to the SPM (double count, caught by the smoke test's scale of 0.71); that
the rank map would fix the GBDT prior's narrow top (it costs 0.3 at game level); and that a one-season
smoke result (-1.6 on 2010) would hold (-0.5 over 28, and flat against the mapped board). Measure first.
