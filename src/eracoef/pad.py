"""The one place a box-score rate is padded.

Every rate this project derives from counts -- a shooter's free-throw percentage, his two- and
three-point percentages, the thirteen per-100 exposure rates -- is shrunk toward a target with the
same closed form, and the shrinkage constant comes from the same method of moments.  Padding by hand
anywhere else is a bug: it has produced numbers that disagreed with the shipped convention before
(FINDINGS.md section 8), and the owner's rule is that no box-score stat is used unpadded, ever.

    padded = (n * rate + k * target) / (n + k)

`k` is in the units of `n`.  For a made/attempt proportion that is ATTEMPTS, estimated by `mom_k`
below.  For the per-100-possession exposure rates it is possessions, estimated by
`exposure.split_half_k` from interleaved halves; that estimator stays where it is because it measures
a different quantity (split-half covariance of a rate, not the binomial variance of a proportion),
but the shrinkage it feeds into is this `shrink`.
"""
from __future__ import annotations

import numpy as np

K_FALLBACK = 40.0          # when there are too few players to estimate k; a mild, deliberately conservative pad


def shrink(rate, n, k, target):
    """(n * rate + k * target) / (n + k), elementwise with broadcasting; `target` where n + k == 0."""
    rate = np.asarray(rate, dtype=float)
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)
    target = np.asarray(target, dtype=float)
    denom = n + k
    safe = np.where(denom > 0, denom, 1.0)
    return np.where(denom > 0, (n * rate + k * target) / safe, target)


def mom_k(made, att, min_att: float = 20.0, fallback_rate: float | None = None) -> tuple[float, float]:
    """League rate and the empirical-Bayes padding constant k, in ATTEMPTS, for a made/attempt proportion.

    Method of moments on per-unit totals (a unit is a player-season, or a player-block).  A proportion
    has within-unit variance p(1-p)/n, so the between-unit variance is what remains after subtracting it:

        tau2 = Var_w(p_hat) - E_w[p_hat (1 - p_hat) / n]
        k    = E_w[p_hat (1 - p_hat)] / tau2

    with attempt weights.  Units below `min_att` attempts are dropped from the estimate (their p_hat is
    too coarse) but are of course still padded by the caller.  With fewer than 20 qualifying units the
    estimate is not trustworthy and (league, K_FALLBACK) is returned, the league rate from whatever
    totals exist or `fallback_rate` if there are none.
    """
    made = np.asarray(made, dtype=float)
    att = np.asarray(att, dtype=float)
    tot_m, tot_a = made.sum(), att.sum()
    league = float(tot_m / tot_a) if tot_a > 0 else (0.5 if fallback_rate is None else float(fallback_rate))
    keep = att >= min_att
    if keep.sum() < 20:
        return league, K_FALLBACK
    p, w = made[keep] / att[keep], att[keep]
    mu = np.average(p, weights=w)
    between = np.average((p - mu) ** 2, weights=w)
    within = np.average(p * (1.0 - p) / w, weights=w)
    tau2 = max(between - within, 1e-6)
    return league, float(np.average(p * (1.0 - p), weights=w) / tau2)


def pad_rate(made, att, k, target):
    """The padded proportion made/att; att may be zero (then the target)."""
    made = np.asarray(made, dtype=float)
    att = np.asarray(att, dtype=float)
    rate = np.where(att > 0, made / np.where(att > 0, att, 1.0), 0.0)
    return shrink(rate, att, k, target)
