"""Four-factor RAPM: lineup-specific expected rates for eFG%, TOV%, OREB% and FT rate.

These exist for one purpose -- to build expected points -- and they are consumed in exactly one
form: the LINEUP SUM `Z @ theta` evaluated on a row, never the individual player split.

That distinction is the whole reason this is safe.  Splitting a lineup's offensive rebounding among
its five players is the badly identified direction, and the per-player four-factor numbers would be
as shaky as the defensive box prior this project spent a session removing (FINDINGS.md sections
7-13).  The lineup sum is the part RAPM estimates *well*, and it is the only part xPTS needs.  So:

    **Never write the per-player four-factor effects to outputs/.  Never publish them.**

Two further choices, both deliberate:

* **No box prior** -- `features=[]`, pure zero-prior RAPM.  The 13 box features ARE the four factors
  at a different aggregation, so priming an OREB% fit with the player's own `orb` rate is close to
  regressing the target on itself; the prior would dominate and the fit would become a repackaged
  box score.  It also removes a whole leakage surface: with no features, BoxExposure never consults
  covariate games, so leakage is controlled by the row mask alone.
* **Lambda per factor** -- `lam_plugin` and `lam_beta` are sigma^2/tau^2 on the points-per-100 scale.
  eFG% (sigma ~ 50 per attempt) and OREB% (sigma ~ 43 per chance) have different sigma^2 AND
  different tau^2, so no transport formula exists.  Select them, do not carry them over.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cv import fit_pipeline, lambda_ratio_grid, make_pipeline

FACTORS = ("efg", "tov", "oreb", "ftr")

# Fitted lineup rates are clipped before they reach xPTS.  A ridge cannot guarantee a probability
# stays in range, and a lineup rate outside these bounds is a fit artefact, not basketball.  The
# clip RATE is what matters: above about 0.5% of rows, the shrinkage is too weak.
CLIP = {"efg": (15.0, 85.0), "tov": (2.0, 40.0), "oreb": (2.0, 60.0), "ftr": (0.0, 60.0)}

# config.yaml's lam_ratio_grid tops out at 2.0, which is right for the points target but TRUNCATES
# the factors: OREB% selects 2.0 there, i.e. the grid edge, and 3.0 once the grid is widened.  A
# criterion sitting on its boundary has not chosen anything (the same failure as the defensive
# shrinkage in FINDINGS.md section 13), so the factor grid is wide enough for every factor to land
# in the interior, and select_lambda reports when one does not.
FACTOR_RATIOS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0)


def select_lambda(wd_t, cfg, lams=None, ratios=None, cv=2, selection="reml_in_band"):
    """Choose (lam, lam_ratio) for one factor by REML-in-band over the joint grid.

    Same machinery and same selection rule as the points fit uses (`cv.lambda_ratio_grid`), applied
    to the factor's own design so the choice is made on the factor's own scale.
    """
    if lams is None:
        g = cfg["lam_grid"]
        lams = (np.geomspace(g["start"], g["stop"], g["num"]) if g.get("log", True)
                else np.linspace(g["start"], g["stop"], g["num"]))
    lams = [float(x) for x in lams]
    ratios = [float(r) for r in (ratios if ratios is not None else FACTOR_RATIOS)]

    def fit_fn(ratio):
        pipe = make_pipeline(wd_t, lams=lams, cv=cv, features=[], mode="full",
                             lam_ratio=ratio, selection=selection)
        return fit_pipeline(pipe, wd_t)["mm"]

    tab, lam, ratio, how = lambda_ratio_grid(fit_fn, ratios, lams, selection=selection)
    edge = ratio in (min(ratios), max(ratios)) or lam in (min(lams), max(lams))
    return lam, ratio, ("EDGE:" + how) if edge else how, tab


def fit_factor(wd_t, lam, ratio, mask=None):
    """One zero-prior RAPM on a factor's design.  `mask` restricts the FITTED rows (cross-fitting)."""
    pipe = make_pipeline(wd_t, lam=lam, features=[], mode="full", lam_ratio=ratio)
    return fit_pipeline(pipe, wd_t, mask)


def lineup_rate(pipe, wd_eval, factor=None, rows=None) -> np.ndarray:
    """The fitted rate for the ten players on the floor, per row of `wd_eval`.

    Evaluated on the POINTS design's rows rather than the factor's own, so every points row gets a
    rate -- including stints where that side took no field goals and so never appeared in the eFG%
    design at all.

    The whole fixed block is included, not just the season intercept: `home`, `is_po`, `is_gt`,
    `margin` and `margin_frem` are real conditioning (garbage time genuinely changes eFG%), and
    dropping them would break the calibration identity the aggregate check relies on.
    """
    X = wd_eval.X if rows is None else wd_eval.X[np.flatnonzero(rows)]
    exp = pipe["exposure"]
    c = pipe["mm"].components(exp.transform(X))
    r = c["fixed"] + c.get("u_O", 0.0) + c.get("u_D", 0.0)
    if factor is not None:
        lo, hi = CLIP[factor]
        r = np.clip(r, lo, hi)
    return r


def crossfit_rate(wd_eval, wd_t, factor, lam, ratio, halves=("A", "B"), include_po=True):
    """Per-row lineup rate on `wd_eval`, each row scored by the fit that did NOT see it.

    A player's own outcomes feed his factor effect, which prices the very possessions used to score
    him; the same-row channel is genuine leakage, so a row is always scored out of fold.  The A/B
    split is `design._order_games`, which already alternates games chronologically within a season,
    so it is interleaved by construction rather than chronological -- lineups, roster and context are
    held fixed and only the bounces differ.

    Returns (rate, coverage) where coverage is the fraction of rows that got a rate.

    CAUTION for the validation experiment: if half B is being held out to score a model, do NOT use
    this -- half-A rows would be scored by the fit trained on half B, leaking the evaluation half's
    outcomes into the training target.  Split A into A1/A2 and cross-fit within A instead.
    """
    n = wd_eval.X.shape[0]
    out = np.full(n, np.nan)
    for h in halves:
        other = "B" if h == "A" else "A"
        pipe = fit_factor(wd_t, lam, ratio, mask=wd_t.half_mask(h, include_po=include_po))
        score = wd_eval.half_mask(other, include_po=include_po)
        if score.any():
            out[score] = lineup_rate(pipe, wd_eval, factor=factor, rows=score)
    return out, float(np.isfinite(out).mean())


def fit_all(wd_eval, wd_by_factor, cfg, lams=None, ratios=None, crossfit=True, verbose=True):
    """Select lambda, fit and evaluate every factor.  Returns (rates, report).

    `rates` is a DataFrame with one column per factor, aligned to `wd_eval`'s rows.
    `report` carries the chosen knobs, the clip rate and the split-half reliability of the lineup
    sums -- which is the go/no-go for the whole approach, since a lineup rate that does not
    replicate across halves carries no signal to put into xPTS.
    """
    rates, rep = {}, []
    for f in FACTORS:
        wd_t = wd_by_factor[f]
        lam, ratio, how, _ = select_lambda(wd_t, cfg, lams=lams, ratios=ratios)
        full = fit_factor(wd_t, lam, ratio)
        r_full = lineup_rate(full, wd_eval, factor=f)
        raw = lineup_rate(full, wd_eval)
        lo, hi = CLIP[f]
        clip_rate = float(((raw < lo) | (raw > hi)).mean())

        # split-half reliability of the LINEUP SUM: fit on each half, correlate the two rates over
        # the same rows.  This is the number that says whether there is anything here at all.
        a = fit_factor(wd_t, lam, ratio, mask=wd_t.half_mask("A", include_po=True))
        b = fit_factor(wd_t, lam, ratio, mask=wd_t.half_mask("B", include_po=True))
        ra, rb = lineup_rate(a, wd_eval, factor=f), lineup_rate(b, wd_eval, factor=f)
        w = wd_eval.rows["poss"].to_numpy()
        rel = _wcorr(ra, rb, w)

        if crossfit:
            r, cov = crossfit_rate(wd_eval, wd_t, f, lam, ratio)
            r = np.where(np.isfinite(r), r, r_full)     # uncovered rows fall back to the full fit
        else:
            r, cov = r_full, 1.0
        rates[f] = r
        rep.append(dict(factor=f, lam=lam, lam_ratio=ratio, eff_def=lam * ratio, chosen_by=how,
                        at_grid_edge=how.startswith("EDGE"),
                        mean=float(np.average(r, weights=w)), sd=float(_wsd(r, w)),
                        sd_halffit=float(_wsd(ra, w)), sd_fullfit=float(_wsd(r_full, w)),
                        split_half_r=rel, clip_rate=clip_rate, coverage=cov))
        if verbose:
            print(f"  {f:5s} lam {lam:8.0f} ratio {ratio:5.2f} ({how})  mean {rep[-1]['mean']:6.2f}  "
                  f"sd {rep[-1]['sd']:5.2f}  split-half r {rel:.3f}  clip {100 * clip_rate:.2f}%", flush=True)
    return pd.DataFrame(rates), pd.DataFrame(rep)


def _wsd(x, w):
    mu = np.average(x, weights=w)
    return float(np.sqrt(np.average((x - mu) ** 2, weights=w)))


def _wcorr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return float(cov / (_wsd(x, w) * _wsd(y, w) + 1e-12))
