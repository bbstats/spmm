"""Mixed-model RAPM: box-score exposures as fixed effects beside ridged player-season effects.

    y = X_f theta_f + Z u + e,   u ~ N(0, sigma^2/lambda * D),  e ~ N(0, sigma^2 W^-1)

Henderson's mixed model equations are solved by a Schur complement over u.  Player-season
columns are disjoint across seasons, so G = Z'WZ is block-diagonal by season and every
solve is one ~1100x1100 factorization per season.  MixedModelRAPMCV eigendecomposes each
season block once so the whole lambda path (and the REML profile) costs microseconds per
lambda.

`beta_fixed`: plug a cross-fitted beta in and fit only F and u (ridge on y - X_box beta).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.linalg as sla
import scipy.sparse as sp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold, KFold
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, _check_sample_weight


# ----------------------------------------------------------------------------- layout
@dataclass
class _Layout:
    """Row-level pieces of one design: Z (sparse), X_f (dense), penalty on X_f columns."""
    Z: sp.csr_matrix | None
    Xf: np.ndarray
    fixed_names: list
    pen_diag: np.ndarray          # ridge on each X_f column (0 = unpenalized)
    n_feat: int
    n_delta: int
    box_slice: slice              # columns of X_f holding [Xbox_O | Xbox_D] (empty if beta_fixed)
    f_slice: slice
    delta_slice: slice
    offset: np.ndarray | None = None   # X_box beta_fixed (+ delta block) when beta is plugged in
    box: np.ndarray | None = None      # raw [Xbox_O | Xbox_D]
    is_po: np.ndarray | None = None

    def rows(self, idx):
        return _Layout(None if self.Z is None else self.Z[idx], self.Xf[idx], self.fixed_names, self.pen_diag,
                       self.n_feat, self.n_delta, self.box_slice, self.f_slice, self.delta_slice,
                       None if self.offset is None else self.offset[idx],
                       None if self.box is None else self.box[idx],
                       None if self.is_po is None else self.is_po[idx])


def _layout(X, spec, lam_delta, beta_fixed=None):
    if spec is None:
        Xf = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=float)
        p = Xf.shape[1]
        return _Layout(None, Xf, [f"x{i}" for i in range(p)], np.zeros(p), 0, 0,
                       slice(0, 0), slice(0, p), slice(p, p))
    X = sp.csr_matrix(X)
    n_base = spec.n_base
    n_box = X.shape[1] - n_base
    if n_box % 2:
        raise ValueError("exposure columns must come in O/D pairs")
    n_feat = n_box // 2
    feats = list(spec.features) if len(spec.features) == n_feat else [f"x{j}" for j in range(n_feat)]
    Z = X[:, spec.z]
    F = X[:, spec.f].toarray()
    box = X[:, n_base:].toarray() if n_feat else np.zeros((X.shape[0], 0))
    is_po = F[:, spec.f_names.index("is_po")]
    use_delta = lam_delta is not None and n_feat > 0
    offset = None
    parts, names, pen = [], [], []
    if beta_fixed is not None:
        bf = np.asarray(beta_fixed, dtype=float)
        if bf.shape[0] not in (2 * n_feat, 4 * n_feat):
            raise ValueError("beta_fixed must have 2*n_feat entries (beta) or 4*n_feat (beta then delta)")
        offset = box @ bf[:2 * n_feat]
        fix_delta = bf.shape[0] == 4 * n_feat
        if fix_delta:
            offset = offset + (box * is_po[:, None]) @ bf[2 * n_feat:]
        p0 = 0
    else:
        parts.append(box)
        names += [f"O:{f}" for f in feats] + [f"D:{f}" for f in feats]
        pen.append(np.zeros(2 * n_feat))
        fix_delta = False
        p0 = 2 * n_feat
    parts.append(F)
    names += list(spec.f_names)
    pen.append(np.zeros(len(spec.f_names)))
    p1 = p0 + len(spec.f_names)
    n_delta = 0
    if use_delta and not fix_delta:
        parts.append(box * is_po[:, None])
        names += [f"dO:{f}" for f in feats] + [f"dD:{f}" for f in feats]
        pen.append(np.full(2 * n_feat, float(lam_delta)))
        n_delta = 2 * n_feat
    Xf = np.hstack(parts)
    return _Layout(Z, Xf, names, np.concatenate(pen), n_feat, n_delta,
                   slice(0, p0), slice(p0, p1), slice(p1, p1 + n_delta), offset, box, is_po)


# ----------------------------------------------------------------------------- solution
@dataclass
class Solution:
    theta_f: np.ndarray
    u: np.ndarray            # unscaled, one per Z column
    S_inv: np.ndarray
    rss: float               # weighted residual sum of squares
    edf: float
    sigma2: float
    n: int
    p: int
    reml: float = np.nan     # -2 * restricted log-likelihood profile (up to a constant)


class Moments:
    """Weighted cross-products of one set of rows, per-season blocks, with column scaling.

    Scaled Z columns: Ztilde = Z diag(c), c_k = 1/sqrt(ratio_k), so a scalar ridge on
    Ztilde equals ratio-specific penalties on Z.
    """

    def __init__(self, layout: _Layout, y, w, season_cols, scale):
        Xf = layout.Xf
        Z = layout.Z
        y = np.asarray(y, dtype=float)
        if layout.offset is not None:
            y = y - layout.offset
        w = np.asarray(w, dtype=float)
        self.n = len(y)
        self.p = Xf.shape[1]
        self.pen_diag = layout.pen_diag
        self.scale = scale
        # accumulated in row chunks: a full w[:, None] * Xf copy is ~1 GB on a 30-season design
        self.A = np.zeros((self.p, self.p))
        self.b = np.zeros(self.p)
        self.yWy = 0.0
        step = max(1, int(2e7 // max(self.p, 1)))
        for i0 in range(0, self.n, step):
            sl = slice(i0, min(i0 + step, self.n))
            Xc, wc, yc = Xf[sl], w[sl], y[sl]
            self.A += Xc.T @ (wc[:, None] * Xc)
            self.b += Xc.T @ (wc * yc)
            self.yWy += float(yc @ (wc * yc))
        # fixed columns with no data (e.g. is_po in an RS-only window) are dropped from the solve
        self.active = np.diag(self.A) > 0
        self.p_act = int(self.active.sum())
        self.season_cols = season_cols
        self.n_z = 0 if Z is None else Z.shape[1]
        self.Gs, self.Bs, self.gs = [], [], []
        if Z is not None and Z.shape[1]:
            Zw = Z.T.multiply(w).tocsr()          # Z'W
            G = (Zw @ Z).tocsr()                  # Z'WZ  (block diagonal by season)
            B = (Zw @ Xf).T                       # X_f' W Z   (p x n_z)
            g = Zw @ y
            del Zw
            for cols in season_cols:
                c = scale[cols]
                Gs = G[cols][:, cols].toarray() * np.outer(c, c)
                self.Gs.append(Gs)
                self.Bs.append(B[:, cols] * c[None, :])
                self.gs.append(g[cols] * c)
        self._eig = None

    def _expand(self, theta_a, S_inv_a):
        theta = np.zeros(self.p)
        theta[self.active] = theta_a
        S_inv = np.zeros((self.p, self.p))
        S_inv[np.ix_(self.active, self.active)] = S_inv_a
        S_inv[~self.active, ~self.active] = np.nan
        return theta, S_inv

    # ---- dense Cholesky path (fixed lambda)
    def solve_chol(self, lam: float) -> Solution:
        act = self.active
        A, b = self.A[np.ix_(act, act)], self.b[act]
        S = A.copy() + np.diag(self.pen_diag[act])
        rhs = b.copy()
        facs = []
        for Gs, Bs, gs in zip(self.Gs, self.Bs, self.gs):
            Bs = Bs[act]
            C = Gs + lam * np.eye(len(Gs))
            cf = sla.cho_factor(C, lower=True, check_finite=False)
            CiB = sla.cho_solve(cf, Bs.T, check_finite=False)       # n_s x p
            S -= Bs @ CiB
            rhs -= Bs @ sla.cho_solve(cf, gs, check_finite=False)
            facs.append(cf)
        S_inv_a = np.linalg.inv(S) if self.p_act else np.zeros((0, 0))
        theta_a = S_inv_a @ rhs
        theta, S_inv = self._expand(theta_a, S_inv_a)
        u = np.zeros(self.n_z)
        edf = float(self.p_act)
        quad = 0.0
        for cols, Gs, Bs, gs, cf in zip(self.season_cols, self.Gs, self.Bs, self.gs, facs):
            us = sla.cho_solve(cf, gs - Bs.T @ theta, check_finite=False)
            u[cols] = us
            edf += float(np.trace(sla.cho_solve(cf, Gs, check_finite=False)))
            quad += us @ Gs @ us + 2.0 * theta @ Bs @ us - 2.0 * us @ gs
        rss = self.yWy - 2.0 * theta @ self.b + theta @ self.A @ theta + quad
        sigma2 = rss / max(self.n - edf, 1.0)
        u_unscaled = u * self.scale if self.n_z else u
        return Solution(theta, u_unscaled, S_inv, float(rss), edf, float(sigma2), self.n, self.p_act)

    # ---- eigen path (lambda grid)
    def eig(self):
        if self._eig is None:
            Vs, ds, Ps, qs = [], [], [], []
            for Gs, Bs, gs in zip(self.Gs, self.Bs, self.gs):
                d, V = sla.eigh(Gs, check_finite=False)
                d = np.maximum(d, 0.0)
                Vs.append(V); ds.append(d)
                Ps.append(Bs @ V)      # p x n_s
                qs.append(V.T @ gs)    # n_s
            self._eig = (Vs, ds, Ps, qs)
        return self._eig

    def solve_eig(self, lam: float, reml: bool = True, pen=None) -> Solution:
        Vs, ds, Ps, qs = self.eig()
        act = self.active
        pen_diag = self.pen_diag if pen is None else np.asarray(pen, dtype=float)
        A, b = self.A[np.ix_(act, act)], self.b[act]
        S = A.copy() + np.diag(pen_diag[act])
        rhs = b.copy()
        Ps = [P[act] for P in Ps]
        for d, P, q in zip(ds, Ps, qs):
            inv = 1.0 / (d + lam)
            S -= (P * inv[None, :]) @ P.T
            rhs -= P @ (q * inv)
        S_inv_a = np.linalg.inv(S) if self.p_act else np.zeros((0, 0))
        theta_a = S_inv_a @ rhs
        theta, S_inv = self._expand(theta_a, S_inv_a)
        u = np.zeros(self.n_z)
        edf = float(self.p_act)
        quad = 0.0
        unorm2 = 0.0
        logdet = 0.0
        for cols, V, d, P, q in zip(self.season_cols, Vs, ds, Ps, qs):
            inv = 1.0 / (d + lam)
            coef = (q - P.T @ theta_a) * inv
            u[cols] = V @ coef
            edf += float(np.sum(d * inv))
            quad += float(coef @ (d * coef)) + 2.0 * float(theta_a @ (P @ coef)) - 2.0 * float(coef @ q)
            unorm2 += float(coef @ coef)
            logdet += float(np.sum(np.log(d + lam) - np.log(lam)))
        rss = self.yWy - 2.0 * theta @ self.b + theta @ self.A @ theta + quad
        sigma2 = rss / max(self.n - edf, 1.0)
        sol = Solution(theta, u * self.scale if self.n_z else u, S_inv, float(rss), edf, float(sigma2), self.n, self.p_act)
        if reml:
            # -2 * REML profile log-likelihood (constants dropped; -sum(log w) added by the caller)
            Q = rss + lam * unorm2
            sign, ld_S = np.linalg.slogdet(S) if self.p_act else (1.0, 0.0)
            n_eff = max(self.n - self.p_act, 1)
            sol.reml = logdet + ld_S + n_eff * np.log(Q / n_eff)
        return sol


def _weighted_mse(y, yhat, w):
    w = np.asarray(w, dtype=float)
    return float(np.sum(w * (y - yhat) ** 2) / np.sum(w))


# ----------------------------------------------------------------------------- estimator
class MixedModelRAPM(RegressorMixin, BaseEstimator):
    """Mixed-model RAPM at a fixed ridge penalty.

    Parameters
    ----------
    lam : ridge penalty on the offensive player-season effects.
    lam_ratio : lambda_D / lambda_O.
    lam_buckets : dict name -> ratio for Z column groups in spec.col_groups (extra penalty multiplier).
    lam_delta : ridge on the playoff-only exposure block; None -> no delta block.
    beta_fixed : plug-in beta (2*n_feat, or 4*n_feat with delta); the box columns then enter as an offset.
    spec : DesignSpec.  None -> every column of X is an unpenalized fixed effect (plain WLS).
    """

    def __init__(self, lam=1000.0, lam_ratio=1.0, lam_buckets=None, lam_delta=None, beta_fixed=None, spec=None):
        self.lam = lam
        self.lam_ratio = lam_ratio
        self.lam_buckets = lam_buckets
        self.lam_delta = lam_delta
        self.beta_fixed = beta_fixed
        self.spec = spec

    # ---- helpers
    def _scale(self):
        spec = self.spec
        if spec is None:
            return np.zeros(0)
        c = np.ones(2 * spec.n_ps)
        c[spec.zd] /= np.sqrt(float(self.lam_ratio))
        for name, ratio in (self.lam_buckets or {}).items():
            c[spec.col_groups[name]] /= np.sqrt(float(ratio))
        return c

    def _season_cols(self):
        return [] if self.spec is None else self.spec.season_cols()

    def _validate(self, X, y, sample_weight):
        X, y = check_X_y(X, y, accept_sparse="csr", y_numeric=True, dtype=np.float64)
        w = _check_sample_weight(sample_weight, X, dtype=np.float64)
        return X, y, w

    def _layout(self, X):
        return _layout(X, self.spec, self.lam_delta, self.beta_fixed)

    def _moments(self, layout, y, w):
        return Moments(layout, y, w, self._season_cols(), self._scale())

    def _set_solution(self, sol: Solution, layout: _Layout, mom: Moments):
        self.solution_ = sol
        self.n_feat_ = layout.n_feat
        self.fixed_names_ = layout.fixed_names
        self.dropped_fixed_ = [n for n, a in zip(layout.fixed_names, mom.active) if not a]
        self.theta_f_ = sol.theta_f
        self.cov_theta_ = sol.sigma2 * sol.S_inv
        se = np.sqrt(np.maximum(np.diag(self.cov_theta_), 0.0))
        self.theta_se_ = se
        nf = layout.n_feat
        if self.beta_fixed is not None:
            bf = np.asarray(self.beta_fixed, dtype=float)
            self.beta_ = bf[:2 * nf]
            self.beta_se_ = np.full(2 * nf, np.nan)
            self.cov_beta_ = np.full((2 * nf, 2 * nf), np.nan)
        else:
            bs = layout.box_slice
            self.beta_ = sol.theta_f[bs]
            self.beta_se_ = se[bs]
            self.cov_beta_ = self.cov_theta_[bs, bs]
        self.gamma_ = sol.theta_f[layout.f_slice]
        self.gamma_se_ = se[layout.f_slice]
        ds = layout.delta_slice
        if self.beta_fixed is not None and len(np.asarray(self.beta_fixed)) == 4 * nf:
            self.delta_ = np.asarray(self.beta_fixed, dtype=float)[2 * nf:]
            self.delta_se_ = np.full(2 * nf, np.nan)
        else:
            self.delta_ = sol.theta_f[ds]
            self.delta_se_ = se[ds]
        self.u_ = sol.u
        self.sigma2_ = sol.sigma2
        self.edf_ = sol.edf
        self.rss_ = sol.rss
        self.n_rows_ = sol.n
        self.coef_ = sol.theta_f  # sklearn convention
        return self

    # ---- sklearn API
    def fit(self, X, y, sample_weight=None):
        X, y, w = self._validate(X, y, sample_weight)
        self.n_features_in_ = X.shape[1]
        layout = self._layout(X)
        mom = self._moments(layout, y, w)
        sol = mom.solve_chol(float(self.lam)) if mom.n_z else mom.solve_chol(0.0)
        return self._set_solution(sol, layout, mom)

    def _predict_layout(self, layout: _Layout, theta_f=None, u=None):
        theta_f = self.theta_f_ if theta_f is None else theta_f
        u = self.u_ if u is None else u
        yhat = layout.Xf @ theta_f
        if layout.offset is not None:
            yhat = yhat + layout.offset
        if layout.Z is not None and layout.Z.shape[1]:
            yhat = yhat + layout.Z @ u
        return yhat

    def predict(self, X):
        check_is_fitted(self, "theta_f_")
        X = check_array(X, accept_sparse="csr", dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}")
        return self._predict_layout(self._layout(X))

    def components(self, X):
        """Per-row pieces of the prediction: prior_O, prior_D, fixed (F gamma), delta, u_O, u_D."""
        check_is_fitted(self, "theta_f_")
        X = check_array(X, accept_sparse="csr", dtype=np.float64)
        lay = self._layout(X)
        nf = lay.n_feat
        out = {}
        if lay.box is not None:
            out["prior_O"] = lay.box[:, :nf] @ self.beta_[:nf]
            out["prior_D"] = lay.box[:, nf:] @ self.beta_[nf:]
            out["delta"] = (lay.box * lay.is_po[:, None]) @ self.delta_ if len(self.delta_) else np.zeros(lay.Xf.shape[0])
        out["fixed"] = lay.Xf[:, lay.f_slice] @ self.theta_f_[lay.f_slice]
        if lay.Z is not None:
            m = self.spec.n_ps
            out["u_O"] = lay.Z[:, :m] @ self.u_[:m]
            out["u_D"] = lay.Z[:, m:] @ self.u_[m:]
        return out

    def coef_table(self, flip_defense: bool = True) -> pd.DataFrame:
        """Box-score coefficients with SEs. Defense flipped so positive = good on both sides."""
        check_is_fitted(self, "theta_f_")
        nf = self.n_feat_
        feats = list(self.spec.features) if self.spec is not None and len(self.spec.features) == nf else [f"x{j}" for j in range(nf)]
        rows = []
        has_delta = len(self.delta_) == 2 * nf and nf > 0
        for side, off in (("O", 0), ("D", nf)):
            sgn = -1.0 if (flip_defense and side == "D") else 1.0
            for j, f in enumerate(feats):
                r = dict(side=side, feature=f, beta=sgn * self.beta_[off + j], se=self.beta_se_[off + j])
                if has_delta:
                    r.update(delta=sgn * self.delta_[off + j], delta_se=self.delta_se_[off + j])
                rows.append(r)
        return pd.DataFrame(rows)

    def summary(self) -> str:
        check_is_fitted(self, "theta_f_")
        lines = [f"MixedModelRAPM: n={self.n_rows_}  p_fixed={len(self.theta_f_) - len(self.dropped_fixed_)}  "
                 f"n_z={len(self.u_)}  sigma2={self.sigma2_:.1f}  edf={self.edf_:.1f}"]
        if self.dropped_fixed_:
            lines.append(f"  dropped all-zero fixed columns: {self.dropped_fixed_}")
        if self.beta_fixed is not None:
            lines.append("  beta plugged in (beta_fixed); box columns entered as an offset")
        return "\n".join(lines)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


# ----------------------------------------------------------------------------- lambda path
class MixedModelRAPMCV(MixedModelRAPM):
    """MixedModelRAPM with the ridge penalty chosen on a grid by grouped CV (eigen path).

    Parameters
    ----------
    lams : grid of lambda values.
    cv : int (GroupKFold if groups given else KFold) or a splitter.
    selection : "reml_in_band" (REML lambda if within the CV 1-SE band, else the CV argmin),
        "cv" (argmin of mean OOS MSE) or "reml".
    lam_ratio, lam_buckets, lam_delta, beta_fixed, spec : as MixedModelRAPM.
    """

    def __init__(self, lams=(100.0, 300.0, 1000.0, 3000.0, 10000.0), cv=5, selection="reml_in_band", lam_ratio=1.0,
                 lam_buckets=None, lam_delta=None, beta_fixed=None, spec=None):
        self.lams = lams
        self.cv = cv
        self.selection = selection
        self.lam_ratio = lam_ratio
        self.lam_buckets = lam_buckets
        self.lam_delta = lam_delta
        self.beta_fixed = beta_fixed
        self.spec = spec

    def _splitter(self, groups):
        if isinstance(self.cv, int):
            return GroupKFold(self.cv) if groups is not None else KFold(self.cv, shuffle=True, random_state=0)
        return self.cv

    def fit(self, X, y, sample_weight=None, groups=None):
        X, y, w = self._validate(X, y, sample_weight)
        self.n_features_in_ = X.shape[1]
        lams = np.asarray(list(self.lams), dtype=float)
        layout = self._layout(X)
        splitter = self._splitter(groups)
        fold_mse = []
        fold_w = []
        for tr, te in splitter.split(X, y, groups):
            mom = self._moments(layout.rows(tr), y[tr], w[tr])
            mom.eig()
            lay_te = layout.rows(te)
            mses = []
            for lam in lams:
                sol = mom.solve_eig(lam, reml=False)
                yhat = self._predict_layout(lay_te, sol.theta_f, sol.u)
                mses.append(_weighted_mse(y[te], yhat, w[te]))
            fold_mse.append(mses)
            fold_w.append(w[te].sum())
        fold_mse = np.asarray(fold_mse)              # n_folds x n_lams
        self.fold_mse_ = fold_mse
        self.fold_weight_ = np.asarray(fold_w)
        mean_mse = fold_mse.mean(0)
        best = int(np.argmin(mean_mse))
        nf = fold_mse.shape[0]
        diff = fold_mse - fold_mse[:, [best]]
        diff_se = diff.std(0, ddof=1) / np.sqrt(nf) if nf > 1 else np.full(len(lams), np.nan)

        # full-data eigen path (kept, so set_lam re-solves for free) + REML profile
        self.moments_ = self._moments(layout, y, w)
        self.moments_.eig()
        self.log_w_sum_ = float(np.sum(np.log(w)))
        reml = np.array([self.moments_.solve_eig(l, reml=True).reml for l in lams])
        in_band = diff.mean(0) <= np.where(np.isnan(diff_se), np.inf, diff_se)
        self.cv_results_ = pd.DataFrame({
            "lam": lams, "mean_mse": mean_mse, "diff_vs_best": diff.mean(0), "diff_se": diff_se,
            "in_1se_band": in_band, "reml_neg2ll": reml,
        })
        for k in range(nf):
            self.cv_results_[f"mse_fold{k}"] = fold_mse[k]
        self.lam_cv_ = float(lams[best])
        i_reml = int(np.argmin(reml))
        self.lam_reml_ = float(lams[i_reml])
        if self.selection == "cv":
            lam = self.lam_cv_
        elif self.selection == "reml":
            lam = self.lam_reml_
        elif self.selection == "reml_in_band":
            lam = self.lam_reml_ if in_band[i_reml] else self.lam_cv_
        else:
            raise ValueError(f"unknown selection {self.selection!r}")
        self.lam_selected_by_ = "reml" if lam == self.lam_reml_ and self.selection != "cv" else "cv"
        self._layout_ = layout
        return self.set_lam(lam)

    def set_lam(self, lam: float, lam_delta=None):
        """Re-solve at a new lambda (and optionally a new playoff-block penalty) from the stored
        eigendecomposition.  Only the diagonal of the Schur complement changes with lam_delta."""
        check_is_fitted(self, "moments_")
        self.lam_ = float(lam)
        pen = None
        if lam_delta is not None:
            lay = self._layout_
            if lay.n_delta == 0:
                raise ValueError("this fit has no delta block; pass lam_delta at construction")
            pen = self.moments_.pen_diag.copy()
            pen[lay.delta_slice] = float(lam_delta)
            self.lam_delta_ = float(lam_delta)
        sol = self.moments_.solve_eig(float(lam), reml=True, pen=pen)
        return self._set_solution(sol, self._layout_, self.moments_)

    def set_lam_delta(self, lam_delta: float):
        """Re-solve at a new playoff-block penalty, keeping lambda."""
        return self.set_lam(self.lam_, lam_delta=lam_delta)

    def reml_profile(self, lams):
        """-2 REML profile log-likelihood (with the -sum log w constant) on a grid."""
        check_is_fitted(self, "moments_")
        return np.array([self.moments_.solve_eig(float(l), reml=True).reml for l in lams]) - self.log_w_sum_

    def lambda_report(self) -> str:
        check_is_fitted(self, "moments_")
        band = self.cv_results_.loc[self.cv_results_["in_1se_band"], "lam"]
        return (f"lambda: CV argmin {self.lam_cv_:.0f}, REML {self.lam_reml_:.0f}, CV 1-SE band "
                f"[{band.min():.0f}, {band.max():.0f}] -> using {self.lam_:.0f} ({self.lam_selected_by_})")

    def theta_median_starter(self, starter_quantile=0.6):
        """theta = 1 - sqrt(lam/(lam+d)) at the median starter's possessions (d = diag of the O block of G)."""
        check_is_fitted(self, "moments_")
        d = np.concatenate([np.diag(G)[:len(G) // 2] for G in self.moments_.Gs])
        d = d[d > 0]
        starters = d[d >= np.quantile(d, starter_quantile)]
        d_med = float(np.median(starters))
        return 1.0 - np.sqrt(self.lam_ / (self.lam_ + d_med)), d_med

    def shrinkage_table(self, buckets=((0, 500), (500, 1500), (1500, np.inf))):
        """Per-player shrinkage diag(G (G+lam I)^-1) and the naive n/(n+lam), by season and possession bucket."""
        check_is_fitted(self, "moments_")
        Vs, ds, _, _ = self.moments_.eig()
        lam = self.lam_
        spec = self.spec
        c = self._scale()
        rows = []
        for s, (cols, V, d, Gs) in enumerate(zip(spec.season_cols(), Vs, ds, self.moments_.Gs)):
            a = np.einsum("ij,j,ij->i", V, d / (d + lam), V)   # diag of V diag(d/(d+lam)) V'
            n_i = np.diag(Gs) / c[cols] ** 2                    # possessions (unscaled diagonal)
            half = len(cols) // 2
            for side, sl in (("O", slice(0, half)), ("D", slice(half, None))):
                for lo, hi in buckets:
                    m = (n_i[sl] >= lo) & (n_i[sl] < hi)
                    if m.sum() == 0:
                        continue
                    lam_side = lam * (c[cols][sl][m] ** -2)
                    rows.append(dict(season=spec.seasons[s], side=side, bucket=f"[{lo},{hi})", n_players=int(m.sum()),
                                     shrink_mean=float(a[sl][m].mean()), shrink_p10=float(np.percentile(a[sl][m], 10)),
                                     shrink_p90=float(np.percentile(a[sl][m], 90)),
                                     naive_mean=float((n_i[sl][m] / (n_i[sl][m] + lam_side)).mean())))
        return pd.DataFrame(rows)
