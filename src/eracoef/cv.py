"""Pipelines, cross-fitting driver and cross-validation with metadata routing.

pipe = Pipeline([("exposure", BoxExposure(...)), ("mm", MixedModelRAPMCV(...))])
GridSearchCV over exposure__pad_scale / mm__lam_ratio / mm__lam_delta with GroupKFold,
sample_weight routed to the estimator (and the transformer, for centering) and to the
scorer; groups routed to the splitter and to MixedModelRAPMCV's inner lambda path.

crossfit_beta: two half-fits (rows of half A with rates from half B, and vice versa),
beta = (beta_A + beta_B)/2, Cov = (Cov_A + Cov_B)/4.
plugin_fit: beta plugged in, u fit on all rows with full-season padded rates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline

from .design import WindowData
from .estimator import MixedModelRAPM, MixedModelRAPMCV
from .exposure import BoxExposure

sklearn.set_config(enable_metadata_routing=True)


def weighted_mse_scorer():
    return make_scorer(mean_squared_error, greater_is_better=False).set_score_request(sample_weight=True)


def make_exposure(wd: WindowData, features=None, mode="crossfit", half="A", **kw) -> BoxExposure:
    feats = list(wd.spec.features) if features is None else list(features)
    exp = BoxExposure(game_box=wd.game_box, game_poss=wd.game_poss, features=feats, spec=wd.spec,
                      mode=mode, half=half, game_half=wd.game_half, **kw)
    exp.set_fit_request(sample_weight=True)
    return exp


def make_pipeline(wd: WindowData, lams=None, lam=None, cv=5, features=None, lam_ratio=1.0, lam_buckets=None,
                  lam_delta=None, beta_fixed=None, mode="crossfit", half="A", selection="reml_in_band",
                  prior_offset=None, free_prior_scale=False, **exposure_kw) -> Pipeline:
    """Pipeline of BoxExposure -> MixedModelRAPMCV (if `lams`) or MixedModelRAPM (fixed `lam`)."""
    exp = make_exposure(wd, features=features, mode=mode, half=half, **exposure_kw)
    if lams is not None:
        mm = MixedModelRAPMCV(lams=list(lams), cv=cv, selection=selection, lam_ratio=lam_ratio, lam_buckets=lam_buckets,
                              lam_delta=lam_delta, beta_fixed=beta_fixed, prior_offset=prior_offset, spec=wd.spec,
                              free_prior_scale=free_prior_scale)
        mm.set_fit_request(sample_weight=True, groups=True)
    else:
        mm = MixedModelRAPM(lam=lam, lam_ratio=lam_ratio, lam_buckets=lam_buckets, lam_delta=lam_delta,
                            beta_fixed=beta_fixed, prior_offset=prior_offset, spec=wd.spec,
                            free_prior_scale=free_prior_scale)
        mm.set_fit_request(sample_weight=True)
    mm.set_score_request(sample_weight=True)
    return Pipeline([("exposure", exp), ("mm", mm)])


def fit_pipeline(pipe: Pipeline, wd: WindowData, mask=None) -> Pipeline:
    sub = wd if mask is None else wd.subset(mask)
    if isinstance(pipe[-1], MixedModelRAPMCV):
        return pipe.fit(sub.X, sub.y, sample_weight=sub.w, groups=sub.groups)
    return pipe.fit(sub.X, sub.y, sample_weight=sub.w)


def grid_search(pipe: Pipeline, param_grid: dict, wd: WindowData, n_folds=5, n_jobs=-1, mask=None) -> GridSearchCV:
    gs = GridSearchCV(pipe, param_grid, cv=GroupKFold(n_folds), scoring=weighted_mse_scorer(), n_jobs=n_jobs,
                      refit=True, return_train_score=False)
    sub = wd if mask is None else wd.subset(mask)
    gs.fit(sub.X, sub.y, sample_weight=sub.w, groups=sub.groups)
    return gs


def lam_grid(cfg: dict) -> np.ndarray:
    g = cfg["lam_grid"]
    if g.get("log", True):
        return np.geomspace(g["start"], g["stop"], g["num"])
    return np.linspace(g["start"], g["stop"], g["num"])


def lambda_ratio_grid(fit_fn, ratios, lams, selection="reml_in_band"):
    """Joint (lam_ratio, lambda) selection with fold-paired SEs across the whole grid.

    fit_fn(ratio) returns one estimator or a list of them (e.g. the two half-fits); their
    per-fold OOS MSE and REML profiles are summed.  Returns (table, lam, ratio, chosen_by).
    """
    rows, fold_mse, reml = [], [], []
    for r in ratios:
        ests = fit_fn(r)
        ests = ests if isinstance(ests, (list, tuple)) else [ests]
        fm = sum(np.asarray(e.fold_mse_) for e in ests)
        rm = sum(e.cv_results_["reml_neg2ll"].to_numpy() for e in ests)
        fold_mse.append(fm)
        reml.append(rm)
        for j, lam in enumerate(lams):
            rows.append(dict(lam_ratio=r, lam=lam))
    F = np.concatenate(fold_mse, axis=1)          # n_folds x (n_ratio * n_lams)
    R = np.concatenate(reml)
    mean = F.mean(0)
    best = int(np.argmin(mean))
    diff = F - F[:, [best]]
    se = diff.std(0, ddof=1) / np.sqrt(F.shape[0])
    tab = pd.DataFrame(rows)
    tab["mean_mse"] = mean
    tab["diff_vs_best"] = diff.mean(0)
    tab["diff_se"] = se
    tab["in_1se_band"] = tab["diff_vs_best"] <= np.where(se > 0, se, 0)
    tab["reml_neg2ll"] = R - R.min()
    i_cv = best
    i_reml = int(np.argmin(R))
    if selection == "cv" or not bool(tab["in_1se_band"].iloc[i_reml]):
        pick, how = i_cv, "cv"
    else:
        pick, how = i_reml, "reml"
    return tab, float(tab["lam"].iloc[pick]), float(tab["lam_ratio"].iloc[pick]), how


def fold_indices(wd: WindowData, n_folds=5, mask=None):
    """GroupKFold splits (train_idx, test_idx) on the (optionally masked) window rows."""
    sub = wd if mask is None else wd.subset(mask)
    return list(GroupKFold(n_folds).split(sub.X, sub.y, sub.groups)), sub


# ----------------------------------------------------------------------------- cross-fitting
@dataclass
class CrossfitResult:
    """beta from two half-fits. Exposes beta_/beta_se_/cov_beta_/delta_/delta_se_/n_feat_/spec like an estimator."""
    beta_: np.ndarray
    beta_se_: np.ndarray
    cov_beta_: np.ndarray
    delta_: np.ndarray
    delta_se_: np.ndarray
    n_feat_: int
    spec: object
    fits: dict = field(default_factory=dict)      # "A" -> Pipeline, "B" -> Pipeline
    lam: dict = field(default_factory=dict)
    cov_delta_: np.ndarray = None
    cov_beta_delta_: np.ndarray = None

    def coef_table(self, flip_defense: bool = True) -> pd.DataFrame:
        nf = self.n_feat_
        feats = list(self.spec.features)
        rows = []
        for side, off in (("O", 0), ("D", nf)):
            sgn = -1.0 if (flip_defense and side == "D") else 1.0
            for j, f in enumerate(feats):
                r = dict(side=side, feature=f, beta=sgn * self.beta_[off + j], se=self.beta_se_[off + j])
                if len(self.delta_):
                    r.update(delta=sgn * self.delta_[off + j], delta_se=self.delta_se_[off + j])
                rows.append(r)
        return pd.DataFrame(rows)

    def half_table(self) -> pd.DataFrame:
        a = self.fits["A"]["mm"]; b = self.fits["B"]["mm"]
        nf = self.n_feat_
        feats = list(self.spec.features)
        return pd.DataFrame({"side": ["O"] * nf + ["D"] * nf, "feature": feats * 2,
                             "beta_A": a.beta_, "se_A": a.beta_se_, "beta_B": b.beta_, "se_B": b.beta_se_,
                             "beta": self.beta_, "se": self.beta_se_})

    def summary(self) -> str:
        lines = [f"crossfit beta: lam A={self.lam.get('A')}, B={self.lam.get('B')}"]
        for h in ("A", "B"):
            lines.append(f"  half {h}: " + self.fits[h]["mm"].summary().replace("\n", "\n    "))
        return "\n".join(lines)


def crossfit_beta(wd: WindowData, lam=None, lams=None, cv=5, lam_ratio=1.0, lam_buckets=None, lam_delta=None,
                  include_po=False, features=None, mask=None, selection="reml_in_band", w_mult=None,
                  exposure_kw_by_half=None, **exposure_kw) -> CrossfitResult:
    """Two half-fits -> beta = (beta_A + beta_B)/2, Cov = (Cov_A + Cov_B)/4 (same for delta).

    include_po: add each half's playoff rows to that half-fit (PO games alternate A/B within a series);
        they get the other half's RS rates, like every row, so delta_A and delta_B are independent.
    mask: optional row mask applied on top of the half split (e.g. RS rows only, or a CV training fold).
    w_mult: optional per-row weight multiplier (game-level bootstrap), used with exposure game_mult.
    """
    if w_mult is not None:
        wd = WindowData(wd.X, wd.y, wd.w * np.asarray(w_mult, dtype=float), wd.groups, wd.spec, wd.game_box,
                        wd.game_poss, wd.rows, wd.games)
    fits, lam_used = {}, {}
    for half in ("A", "B"):
        m = wd.half_mask(half, include_po=include_po)
        if mask is not None:
            m = m & mask
        kw = dict(exposure_kw)
        if exposure_kw_by_half:
            kw.update(exposure_kw_by_half.get(half, {}))
        pipe = make_pipeline(wd, lams=lams, lam=lam, cv=cv, features=features, lam_ratio=lam_ratio,
                             lam_buckets=lam_buckets, lam_delta=lam_delta, mode="crossfit", half=half,
                             selection=selection, **kw)
        fit_pipeline(pipe, wd, m)
        fits[half] = pipe
        lam_used[half] = float(pipe["mm"].lam_) if lams is not None else float(lam)
    a, b = fits["A"]["mm"], fits["B"]["mm"]
    beta = 0.5 * (a.beta_ + b.beta_)
    cov = 0.25 * (a.cov_beta_ + b.cov_beta_)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    if len(a.delta_):
        delta = 0.5 * (a.delta_ + b.delta_)
        cov_d = 0.25 * (a.cov_delta_ + b.cov_delta_)
        cov_bd = 0.25 * (a.cov_beta_delta_ + b.cov_beta_delta_)
        delta_se = np.sqrt(np.maximum(np.diag(cov_d), 0.0))
    else:
        delta, delta_se, cov_d, cov_bd = np.zeros(0), np.zeros(0), None, None
    return CrossfitResult(beta, se, cov, delta, delta_se, a.n_feat_, wd.spec, fits, lam_used,
                          cov_delta_=cov_d, cov_beta_delta_=cov_bd)


def plugin_fit(wd: WindowData, beta, lam=None, lams=None, cv=5, lam_ratio=1.0, lam_buckets=None, lam_delta=None,
               features=None, mask=None, selection="reml_in_band", prior_offset=None,
               free_prior_scale=False, **exposure_kw) -> Pipeline:
    """Ratings fit: beta plugged in, u (and F) fit on all rows with full-season padded rates.

    `prior_offset` is the per-Z-column boost from LRBoost, in raw sign, added to the linear box term.
    """
    pipe = make_pipeline(wd, lams=lams, lam=lam, cv=cv, features=features, lam_ratio=lam_ratio, lam_buckets=lam_buckets,
                         lam_delta=lam_delta, beta_fixed=np.asarray(beta, dtype=float), mode="full",
                         selection=selection, prior_offset=prior_offset,
                         free_prior_scale=free_prior_scale, **exposure_kw)
    return fit_pipeline(pipe, wd, mask)
