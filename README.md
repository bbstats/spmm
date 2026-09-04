# spmm

NBA SPM based on RAPM mixed models for more stable coefficients, plus seasonal/playoff coefficient changes.

Thirty seasons of play-by-play (1996-97 through 2025-26) turned into stints, then one ridge
regression per three-season window. The box score enters as unpenalized fixed effects beside
ridge-penalized player offense and defense effects, solved jointly by Henderson's mixed model
equations, so the coefficients and the player ratings land on the same scale: points per 100
possessions. Live at <https://bbstats.github.io/spmm/>.

> **The ratings section below is under revision and parts of it are known to be wrong.**
> The gradient-boosted correction was measured at zero value against an external benchmark, and the
> defensive box-score prior is being removed. See `HANDOFF.md` for the current state and
> `FINDINGS.md` sections 7-12 for the evidence. The coefficient drift study is unaffected.

## What a rating is

    rating = box score prior + nonlinear correction + on-court residual

- **Box score prior** is the player's three-year padded per-100 rates times that window's
  coefficients, centred on the average player.
- **Nonlinear correction** is a gradient-boosted model (ChimeraBoost) on the ridge residual, pooled
  over all ten windows so it has enough players to fit, cross-fitted so nobody scores himself,
  shrunk by the slope of its own out-of-fold prediction, and stripped of any linear component so it
  can only supply curvature. Playing time is a feature during fitting and frozen at a starter
  reference when scoring, so the correction carries the shape of a box-score profile without the
  premium the residual pays for facing backups.
- **On-court residual** is what the margin says beyond both.

The random effect is one value per player per three-season window, matching the window the
coefficients are fit on and the window the playoff block uses.

## Pipeline

    python scripts/01_ingest.py 1997 2026 RS,PO     # play-by-play and box scores
    python scripts/02_stints.py 1997 2026 RS,PO     # possessions with ten players on the floor
    python scripts/03_cv.py                         # lambda and lambda_ratio, once, on two windows
    python scripts/04_validation.py                 # the regularization checkpoint
    python scripts/04_fit_all.py                    # coefs.parquet: base plus six robustness runs
    python scripts/05_playoffs.py                   # the playoff block
    python scripts/10_boost.py                      # the pooled LRBoost, ratings_boosted.parquet
    python scripts/11_gate.py                       # does the boosted prior calibrate better?
    python scripts/12_window_length.py              # three-season windows against five-season
    python scripts/06_export_csv.py                 # CSV mirrors of every table
    python scripts/07_plots.py                      # docs/index.html and docs/img/*.png

Raw and intermediate caches live under `data/` and are not in git; steps 1 and 2 regenerate them.

## Layout

    src/eracoef/
      stints.py     play-by-play to stints
      boxtable.py   per-game box scores, player names
      design.py     the sparse design: Z (players), F (fixed effects), box exposures
      exposure.py   cross-fitted, empirical-Bayes padded per-100 rates
      estimator.py  the mixed model: Henderson, Schur complement over u, eigen lambda path
      cv.py         pipelines, cross-fitted beta, the plug-in ratings fit
      boost.py      LRBoost: the player panel, the pooled booster, the prior offset
      checks.py     calibration, bootstrap, out-of-sample comparisons, variance components
      windows.py    the window loop and the ratings table
      plots.py      the coefficient figure and the static PNGs
      site.py       the single page

## Notes

- **Cross-fitted rates.** Each season's games are split in half; coefficients fit on one half use
  rates computed only from the other. Full-season rates inflate the three-point coefficient by more
  than seven standard errors.
- **One lambda for the whole era**, chosen once by cross-validation and REML on two windows, so
  shrinkage cannot manufacture drift.
- **Standard errors** are the model's, about 10% conservative against a game-level Bayesian
  bootstrap.
