# Handoff: OpenRAPM ships the multi-stage prior-informed RAPM; the domain is half connected

Written 2026-09-05, late. The session built the owner's prior chain, measured it two rounds deep on
the out-of-season criterion, shipped it as the board, renamed the project OpenRAPM, replaced the
site with one table, and bought `openrapm.com`. `FINDINGS.md` section 19 is the record (both
rounds); `HANDOFF_rankmap_archive.md` is the previous handoff.

**Branch state:** `main` and `hybrid-and-xpts` are identical and pushed (`bbstats/openrapm`; the old
`spmm` URLs redirect). Live at <https://bbstats.github.io/openrapm/>.

---

## Part 0: unfinished right now

1. **DNS for openrapm.com.** The repo side is done (`docs/CNAME` = `openrapm.com`; the Pages custom
   domain is set, `https_enforced: false`). At the time of writing `nslookup openrapm.com` still
   returns Porkbun's parking addresses (207.207.210.x), so the records are not in yet. The owner adds
   them at Porkbun: delete the two parking records, add four `A` records for the bare host
   (185.199.108.153, .109.153, .110.153, .111.153) and a `CNAME` `www` -> `bbstats.github.io`.
   Then: `nslookup openrapm.com` should return the four GitHub addresses; wait for the certificate
   (`gh api repos/bbstats/openrapm/pages --jq .https_certificate`), then
   `gh api -X PUT repos/bbstats/openrapm/pages -F https_enforced=true`.
2. **The owner thinks something in the model needs fixing.** Part 1 lists what the measurements
   say is wrong with the shipped board, so whoever picks this up starts from the numbers.

---

## Part 1: what ships, and what is known to be wrong with it

### The board is `mspi` (multi-stage prior-informed RAPM)

Per side (offense on the free-throw-adjusted target, defense on the opponent-three-adjusted one),
per three-season block:

1. **APM**: the plugin ridge with no box prior at penalty 100 (under 1% of the shipped 18,352).
2. **Simple SPM**: a possession-weighted ridge of APM on share of team possessions, starts share,
   age and their powers, fit on the other blocks. `share` = on-floor possessions summed over teams,
   over ONE full team-season (games not played count), capped at 0.9.
3. **RAPM_1**: the shipped ridge (`lam_plugin` 18,352, ratio 0.287) pulled toward the Simple SPM.
4. **The GBDT prior**: chimeraboost fit to RAPM_1 pooled over the player's OTHER blocks, from his 13
   padded centred rates + season + share, gs_pct, age (Boruta survivors in `config.yaml -> gbdt`),
   leave-block-out, the excluded block kept out of every pooled target.
5. **The rating**: the same ridge pulled toward the GBDT prediction alone (no linear box prior, no
   Simple SPM in the final fit; the role information reaches it through the GBDT's features).

`config.yaml`: `ratings_prior: offense gbdt, defense gbdt, role_prior spm, rank_map null`,
`gbdt.mode: full`. `scripts/08_ratings.py` builds it through `spm.chain_offset(mode=cfg gbdt.mode)`
(a bug caught on the first build: the default mode is "residual", which is my variant, not the
owner's chain; the script now reads the mode from config).

Out of season (team-game level, points per 100, 28 held-out seasons): **112.32 / 112.06 at K=2 / K=4**
against 112.54 / 112.33 for the previous board with its rank map; 20 of 28 seasons at K=4, z -1.8.
Against the previous board without the map: -0.55 / -0.52, z -2.4 / -2.7. The owner shipped it on
the rule that game level is all that matters.

### What the measurements say is wrong with it (the candidates for "the thing to fix")

1. **The offensive prior is too timid for starters and has no spread among the deep bench.**
   `scripts/51_garbage.py` (calibration slope per role group, held-out seasons, K=4): the seasons
   want the chain's offensive ratings multiplied by **1.20 for starters** (z +11.6), 1.05 for bench
   players, **2.10 for deep-bench players** (under 10% of team possessions, z +11.4). Defense is
   calibrated in every group (1.01-1.05). The previous linear prior was the mirror image: 1.01 for
   starters and 0.85 for the bench. Cause: the GBDT is trained on RAPM_1, which is a shrunk number,
   and for a deep-bench player it is nearly all role level, so the model learns no spread there.
   Measured NOT to fix it: scaling the whole prior (1.25 / 1.5 / 2.0 all worse at game level), the
   rank map (costs 0.3), training on unshrunk APM (`mspi_apm`: offense scale right, defense goes
   wide, same error), APM offense + RAPM_1 defense (`mspi_mix`: both scales right, same error).
   **Not tried: a role-dependent blend** -- the chain's prior for bench players, the linear
   player-priced prior (or a scaled chain) for starters, blended over a band of share or starts.
2. **The consensus agreement fell from 0.883 / 0.879 / 0.814 to 0.772 / 0.778 / 0.785**, the largest
   drop any criterion winner has carried. The offensive spread is 0.62 of the consensus's. The top
   five on 2024-26 are Jokic, Wembanyama, Gilgeous-Alexander, Leonard, Holmgren where the consensus
   has Giannis and Doncic in place of the last two. Star guards sit lower (the timid top end above).
   Per the owner this is reported, not gated; `tests/test_vs_consensus.py` floors were lowered to
   0.75 / 0.76 / 0.75 with the reason in each docstring.
3. **The chain loses on floors with seven or more bench players** (18% of possessions): +1.7 per 100
   at game level against the previous board. It is NOT garbage time (the chain is better on
   garbage-time rows, -1.8; `gt_weight` 0.5 / 0 changes nothing). It is competitive second-unit
   minutes, and item 1 is the likely cause.
4. **Unseen players.** The criterion gives a player the training block never saw the average
   player's 0, which under this board makes an unseen rookie far better than the bench around him.
   `ReplacementSystem` (absent players score the block's mean rating of under-500-possession
   players) is worth -0.45 / -0.44 to the chain and -0.20 / -0.14 to the previous board (z -4.5),
   consensus untouched. It is an evaluation-time rule and changes no rating on the board. Not yet
   the default in `Holdout`; the fair reference for the next candidate is `rankmap_def3_p0_rep`.
5. **The era term is inert** (assist value 0.14 -> 0.16 per 100 across 28 seasons; Boruta rejects
   `season` on the residual). Not wrong, but the design's stated reason for the season feature did
   not materialise.
6. **The Simple SPM's defensive starts-share coefficient depends on the APM penalty** (-0.90 /
   -0.76 / -0.51 at 30 / 100 / 300; every other coefficient agrees within a few percent). Half a
   point; noted.

### What it got right (do not undo)

- The Stockton problem is gone: on 1997-99 Jordan is first on offense (prior 3.16 + residual 1.62),
  Stockton seventh; prior sd 1.15 against the linear prior's 1.69, residual 0.67 against 0.40.
- The bench end is calibrated for the first time (1.05 against 0.85).
- Defense is calibrated in every role group.
- `def3_p0` (the previous board, unmapped) reproduces `outputs/holdout_def3.parquet` to 1e-14
  through all the new code, and `outputs/xrapm_panel.parquet` is byte-identical.

---

## Part 2: what to do next

1. **Finish the domain** (Part 0).
2. **Make the replacement level the criterion's default** (wrap every system in `ReplacementSystem`
   inside `Holdout.run`, or in `systems.registry`), re-run the reference once, and record the new
   reference numbers (`rankmap_def3_p0_rep`: 112.34 / 112.20; `mspi_rep`: 112.11 / 111.91).
3. **The role-dependent blend** (Part 1, item 1). One system in `systems.py`: `chain_offset` for
   players below a share / starts threshold, `hybrid_beta`'s linear prior above it, a linear ramp
   between. Read `51_garbage.py` on it; ship only if it wins at game level at both K.
4. **If the owner's fix is elsewhere in the model**, the diagnostics that read it are all one
   command each: `45_holdout.py --systems=... --k=2,4 --workers=4 --rank --consensus --spread
   --splits=bench,exposure,gt --top=1997-1999 --pdp`, then `48_ladder.py`, then `51_garbage.py`.
5. Boruta's tentative features were kept (drb, fg3_miss, pf, tov on offense); a 50-trial run
   (~35 min at 12 threads) would settle them.
6. LATER (owner, unchanged): the lineup three-point factor (HANDOFF_rankmap_archive item 5).

---

## Part 3: state of the machinery

### Built this session

| file | what |
|---|---|
| `src/eracoef/roles.py` | roles table (games, starts, minutes, on-floor and team possessions, age via `leaguedashplayerbiostats`), `player_season_inputs`, `window_inputs`, `design7` |
| `src/eracoef/spm.py` | `apm_fit`, `fit_spm` / `SPMFit`, `spm_offset`, `apm_lambda_check`, `chain_offset(sides, mode, scale, target)` |
| `src/eracoef/gbdt_prior.py` | `training_rows` (pooled other-block target), `drag` / `counterbalance` (RLOOCV), `GBDTPrior` (modes residual / full, `target_col`, cached per exclusion set), `gbdt_offset`, `partial_dependence`, `make_boruta` / `run_boruta` |
| `src/eracoef/systems.py` | `registry(cfg, rankmap)`: every named system: `spm`, `mspi(_o)`, `mspi_resid(_o)`, `mspi_s*`, `mspi_apm*`, `mspi_mix`, `*_rep`, `*_gt05/_gt0`, `def3_*`, `rankmap_*` |
| `src/eracoef/holdout.py` | `PluginSystem.offset`, `ratings_from_fit(offset=)`, `Ratings.fill_o/fill_d`, `ReplacementSystem`, `Context.roles / rpanel / gbdt / mspi / mspi_apm / bigness / bench`, splits `bigs`, `bench`, `gt`, `run(held=)`, `run_parallel` |
| `src/eracoef/windows.py` | `player_ratings_table(prior_parts=)` |
| `scripts/49_role_panel.py` | APM, Simple SPM (leave-block-out), RAPM_1 -> `outputs/role_panel.parquet`; `--check` |
| `scripts/50_boruta.py` | BorutaShap per side and mode -> `outputs/csv/boruta_*.csv` |
| `scripts/51_garbage.py` | calibration slope per role group per side, all / garbage-time / competitive rows -> `outputs/csv/garbage_slopes.csv` |
| `scripts/52_site.py` | `docs/data/ratings.json` for the page |
| `scripts/45_holdout.py` | registry from `systems.py`; `--workers`, `--spread`, `--top`, `--pdp`, `--norun`; `main()` guard |
| `scripts/08_ratings.py` | the chain switch (reads `gbdt.mode`) |
| `docs/index.html`, `docs/CNAME`, `docs/coefficients.html` | the table page; the domain; the old coefficient page |
| `README.md` | OpenRAPM: the test, the chain, how to contribute |
| `tests/test_roles.py`, `test_spm.py`, `test_gbdt_prior.py`, `test_holdout.py` (+3), `test_vs_consensus.py` (floors) | 76 passed, 1 xfail, 1 flaky (below) |
| `outputs/holdout_{chain,chainmap,gt,scale,mix,rep}.parquet`, `outputs/garbage_slopes.parquet`, `outputs/csv/{spm_coefs,spm_lambda_check,role_panel_report,boruta_*,gbdt_pdp,holdout_*}.csv` | every result quoted above |

### Settled, do not redo

- Everything in the previous handoffs' lists.
- The GBDT must NOT be trained on RAPM_1 and then added on top of the Simple SPM (double count of
  the role level, offensive scale 0.71). Either the residual `u` under the SPM (`mspi_resid`, flat)
  or RAPM_1 with the role inputs as features and no SPM in the final fit (`mspi`, ships).
- The rank map does not stack with the GBDT prior. Scaling the prior does not help. Garbage-time
  weighting does not matter. The unshrunk-APM target does not move game-level error.
- `share` is summed over teams over one full team-season; averaging per-team shares halves a
  traded starter.
- APM at penalty 100 is fine (coefficients agree at 30 / 100 / 300 bar the one noted).
- BorutaShap needs shims (`np.NaN`, `scipy.stats.binom_test`, `check_model`, `explain`) and 20+
  trials to decide anything.

### Traps

- **`scripts/07_plots.py` overwrites `docs/index.html`** with the old coefficient page. Do not run it
  without redirecting `site.py`'s output, or the OpenRAPM page is lost (it is in git; `git checkout
  docs/index.html` restores it).
- **`tests/test_core.py::test_calibration_slope_vs_lambda` is flaky and pre-existing** (passes alone,
  fails after the rest of that file; reproduced on the committed tree before this session's changes).
- **Windows spawn re-imports the script**: anything that calls `run_parallel` needs a `__main__` guard.
- **`uv pip install` needs the .exe path**: `--python A:/code/spmm/.venv/Scripts/python.exe`.
- **Do not run Boruta (12 numba threads) beside the ladder**: 27 s per trial shared, 10 s alone.
- Names with non-ASCII characters (Jokic, Doncic) need `PYTHONIOENCODING=utf-8` when printing from
  the Windows shell.

### Verification

```
.venv/Scripts/python -m pytest tests -q                                   # 76 passed, 1 xfailed (+ the flaky core test)
.venv/Scripts/python scripts/49_role_panel.py --check                     # ~2 min with stints cached
.venv/Scripts/python scripts/08_ratings.py && .venv/Scripts/python scripts/52_site.py && .venv/Scripts/python scripts/22_vs_consensus.py
.venv/Scripts/python scripts/45_holdout.py --systems=rankmap_def3_p0,mspi --rankmap=outputs/holdout_chain_rank.parquet --k=2,4 --workers=4 --tag=x   # ~4 min
.venv/Scripts/python scripts/48_ladder.py outputs/holdout_x.parquet --ref=rankmap_def3_p0
.venv/Scripts/python scripts/51_garbage.py --systems=def3_p0,mspi          # ~4 min
```

### The standing warning, kept

Four things this session were asserted and then measured the other way: that the GBDT could be
trained on RAPM_1 and added to the SPM (double count); that the rank map would fix the GBDT prior's
narrow top (it costs 0.3); that the 7+-bench loss was garbage time (the chain is better there); and
that a one-season smoke result (-1.6) would hold (-0.5 over 28, -0.3 against the mapped board).
Measure first.
