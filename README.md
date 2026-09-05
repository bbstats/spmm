# OpenRAPM

Open player-impact ratings for the NBA, 1996-97 through 2025-26, in points per 100 possessions,
one rating per player per three-season block. Live at <https://bbstats.github.io/spmm/>.

The point of the project is the **test**, not any one model: every architecture change is judged
by whether it predicts the games of a season it never saw. Anyone is welcome to propose one; the
criterion decides.

## The test

Hold out one season. Fit the ratings on the seasons either side of it (two or four of them). Predict
every stint of the held-out season from the ten players on the floor, with only the level and home
court refit on that season. Score possession-weighted squared error in points per 100, summed to
team-games so possession noise cancels. Twenty-eight held-out seasons, every system paired by season.

    python scripts/45_holdout.py --systems=def3_p0,mspi --k=2,4 --workers=4 --tag=x
    python scripts/48_ladder.py outputs/holdout_x.parquet --ref=def3_p0

A system is anything with a name and `fit(train_seasons, ctx) -> Ratings` (`src/eracoef/holdout.py`);
`src/eracoef/systems.py` names the ones built so far. `FINDINGS.md` is the record of what the test
has decided, section by section; `HANDOFF.md` is where things stand.

## What ships

The multi-stage prior-informed RAPM (`mspi`, FINDINGS section 19):

1. **APM**: a ridge with an almost-zero penalty on the on-court result, offense on a free-throw-
   adjusted target and defense on an opponent-three-point-adjusted one.
2. **A role prior**: a ridge of APM on the player's share of team possessions, starts share and age,
   fit on the other blocks.
3. **A prior-informed RAPM**: the shipped ridge pulled toward the role prior.
4. **A boosted box-score prior**: a gradient-boosted model (chimeraboost) of that RAPM from the
   player's padded box rates, the season and the role inputs, cross-fitted by block.
5. **The rating**: the ridge pulled toward the boosted prior. Positive is good on both ends.

Out of season it predicts held-out games at 112.06 points per 100 (K=4) against 112.33 for the
previous board; the linear box prior it replaces over-rated its own top end (FINDINGS 18-19).

## Pipeline

    python scripts/01_ingest.py 1997 2026 RS,PO     # play-by-play, box scores
    python scripts/02_stints.py 1997 2026 RS,PO     # possessions with ten players on the floor
    python scripts/03_cv.py                         # the ridge penalties, once
    python scripts/04_fit_all.py                    # coefficient study (outputs/coefs.parquet)
    python scripts/27_xrapm_prior.py                # the player-level panel
    python scripts/49_role_panel.py --check         # APM, role prior, prior-informed RAPM per block
    python scripts/50_boruta.py                     # feature selection for the boosted prior
    python scripts/08_ratings.py                    # the board (outputs/player_ratings.parquet)
    python scripts/52_site.py                       # docs/data/ratings.json for the page

Raw and intermediate caches live under `data/` and are not in git; steps 1 and 2 regenerate them.
Use the project venv (`.venv/Scripts/python` on Windows); `pytest tests -q` runs the checks.

## Layout

    src/eracoef/
      stints.py      play-by-play to stints
      design.py      the sparse design: players, fixed effects, box exposures
      exposure.py    cross-fitted, empirical-Bayes padded per-100 rates
      estimator.py   the mixed model (Henderson), eigen lambda path
      cv.py          pipelines, cross-fitted beta, the plug-in ratings fit
      roles.py       playing-time share, starts, age per player-season
      spm.py         APM, the role prior, the chain's offset
      gbdt_prior.py  the boosted box prior, leave-block-out, Boruta on chimeraboost
      holdout.py     the out-of-season test: systems, runner, splits, diagnostics
      systems.py     every named system
      windows.py     the block loop and the ratings table
      xshoot.py      the luck-adjusted targets (free throws, opponent threes)

The earlier coefficient-drift page is kept at `docs/coefficients.html`.

## Contributing

Add a system to `src/eracoef/systems.py`, run the test against what ships, and open a pull request
with the paired result. Changes that win at game level at both K, in at least 60% of seasons, are the
ones that get merged; the consensus of public metrics is read once as a sanity check and never
selected on.
