"""The GBDT box prior: what a player's box line adds to his role level.

One chimeraboost model per side.  A training row is a player in one window W with W's 13 padded,
centred rates and the season he played most in W; its target is his RAPM_1 (the ridge pulled toward
the Simple SPM, spm.py) POOLED over his OTHER windows, possession-weighted, and its weight is that
pooled possession count.  So the model learns "a player whose box line looks like this in this era
is worth this much on court in the rest of his career", and a player seen in only one window is
scored but never trained on.

Cross-fitting is leave-window-out: the model used for a window (or for a holdout block) is fit
without every window in the exclusion set, and an excluded window's RAPM_1 never enters a pooled
target either.  The owner's point from the distributional-bias paper (RLOOCV): holding a block out
drags the training mean away from the full mean, in the direction opposite to the held-out block.
`drag` measures it per exclusion set and `counterbalance` rescales the weights of the training rows
on the drag's side so the weighted training mean equals the full-panel mean.  With the target centred
inside every window the drag is expected to be near zero; it is reported either way.

Feature selection: BorutaShap once per side on the pooled panel, with chimeraboost's own exact SHAP
values (BorutaShap's shap path calls shap.TreeExplainer, which does not know chimeraboost).
`ChimeraBorutaShap` overrides just that; the shadow-feature test and the binomial decision are the
library's.  The library also needs two compatibility shims (`np.NaN`, `scipy.stats.binom_test`).

Everything is in the model's raw sign; the offset is possession-centred per side before it is
handed to `plugin_fit(prior_offset=...)`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .design import FEATURES

SIDES = ("O", "D")
ROLE_INPUTS = ["share", "gs_pct", "age"]
DEFAULT_FEATURES = [*FEATURES, "season"]                 # mode "residual": what the box line adds to the role level
FULL_FEATURES = [*FEATURES, "season", *ROLE_INPUTS]       # mode "full": the GBDT is the whole prior


# ------------------------------------------------------------------------------------ training rows
def training_rows(panel: pd.DataFrame, side: str, exclude=(), features=None, target_col: str = "rapm1",
                  poss_col: str = "poss") -> pd.DataFrame:
    """Rows of one side with every window in `exclude` removed from BOTH the rows and the pooled targets.

    target = possession-weighted mean of `target_col` over the player's other (non-excluded) windows,
    weight = the possessions behind that mean.  Rows without another window are dropped.
    """
    feats = list(DEFAULT_FEATURES if features is None else features)
    ex = set(exclude)
    p = panel[(panel.side == side) & ~panel.window.isin(ex)].copy()
    w = p[poss_col].to_numpy(dtype=float)
    v = p[target_col].to_numpy(dtype=float)
    p["_wv"] = w * v
    p["_w"] = w
    g = p.groupby("player_id")
    tot_wv = g["_wv"].transform("sum").to_numpy()
    tot_w = g["_w"].transform("sum").to_numpy()
    other_w = tot_w - w
    other_wv = tot_wv - w * v
    keep = other_w > 0
    out = p.loc[keep, ["player_id", "window", *feats]].copy()
    out["target"] = other_wv[keep] / other_w[keep]
    out["weight"] = other_w[keep]
    return out.reset_index(drop=True)


def reference_mean(panel: pd.DataFrame, side: str, **kw) -> float:
    """The full-panel weighted target mean the leave-window-out training sets are compared against."""
    r = training_rows(panel, side, exclude=(), **kw)
    return float(np.average(r["target"], weights=r["weight"])) if len(r) else 0.0


def drag(rows: pd.DataFrame, full_mean: float) -> float:
    """Weighted training-target mean minus the full-panel mean (the distributional bias of this exclusion set)."""
    if len(rows) == 0:
        return 0.0
    return float(np.average(rows["target"], weights=rows["weight"]) - full_mean)


def counterbalance(rows: pd.DataFrame, full_mean: float, tol: float = 0.02) -> tuple[pd.DataFrame, dict]:
    """If the drag exceeds `tol` (points per 100), rescale the weights of the rows on the drag's side of the
    full mean by the one factor that brings the weighted mean back to `full_mean`.  Returns (rows, report)."""
    d = drag(rows, full_mean)
    rep = dict(drag_before=d, drag_after=d, factor=1.0, n_side=0, applied=False)
    if len(rows) == 0 or abs(d) <= tol:
        return rows, rep
    t = rows["target"].to_numpy(dtype=float)
    w = rows["weight"].to_numpy(dtype=float)
    side = t > full_mean if d > 0 else t < full_mean
    S_side, W_side = float((w[side] * t[side]).sum()), float(w[side].sum())
    S_other, W_other = float((w[~side] * t[~side]).sum()), float(w[~side].sum())
    denom = S_side - full_mean * W_side
    f = (full_mean * W_other - S_other) / denom if denom != 0 else 1.0
    f = float(np.clip(f, 0.0, 1.0))
    out = rows.copy()
    out.loc[side, "weight"] = w[side] * f
    rep.update(drag_after=drag(out, full_mean), factor=f, n_side=int(side.sum()), applied=True)
    return out, rep


# ------------------------------------------------------------------------------------ the model
def fit_gbdt(rows: pd.DataFrame, features, seed: int = 0, thread_count=None, **params):
    """chimeraboost at its defaults (early stopping on), weighted by `rows.weight`, with the player as the
    group so his rows never straddle the early-stopping split."""
    from chimeraboost import ChimeraBoostRegressor

    kw = dict(random_state=int(seed))
    if thread_count:
        kw["thread_count"] = int(thread_count)
    kw.update(params)
    m = ChimeraBoostRegressor(**kw)
    X = rows[list(features)].to_numpy(dtype=float)
    y = rows["target"].to_numpy(dtype=float)
    w = rows["weight"].to_numpy(dtype=float)
    m.fit(X, y, sample_weight=w, groups=rows["player_id"].to_numpy())
    return m


class GBDTPrior:
    """Leave-window-out chimeraboost priors over the role panel, one per side, cached per exclusion set.

    Two modes, both tested on the criterion (the first build counted the role level twice by training on
    RAPM_1 and then adding the GBDT on top of the Simple SPM):
      mode "residual"  target `u` (RAPM_1 minus the role prior, pooled over the player's other windows),
                       features = rates + season; the offset is Simple SPM + this.
      mode "full"      target `rapm1` itself, features = rates + season + the role inputs; the offset is
                       this alone, the GBDT being the whole prior.
    """

    def __init__(self, panel: pd.DataFrame, cfg: dict, seed: int | None = None, thread_count=None, features=None,
                 mode: str | None = None, target_col: str | None = None):
        g = cfg.get("gbdt", {})
        self.panel = panel
        self.cfg = cfg
        self.mode = str(g.get("mode", "residual") if mode is None else mode)
        if self.mode not in ("residual", "full"):
            raise ValueError(f"gbdt mode must be 'residual' or 'full', got {self.mode!r}")
        self.target_col = target_col or ("u" if self.mode == "residual" else "rapm1")   # "apm": the unshrunk target
        self.seed = int(g.get("seed", 0) if seed is None else seed)
        self.thread_count = g.get("thread_count") if thread_count is None else thread_count
        self.tol = float(g.get("drag_tol", 0.02))
        self.features = {}
        default = DEFAULT_FEATURES if self.mode == "residual" else FULL_FEATURES
        key = "features_{}" if self.mode == "residual" else "features_full_{}"
        for side in SIDES:
            f = (features or {}).get(side) if isinstance(features, dict) else features
            f = f or g.get(key.format(side)) or default
            self.features[side] = list(f)
        self.params = dict(g.get("params", {}) or {})
        self._models: dict = {}
        self._ref = {side: reference_mean(panel, side, target_col=self.target_col) for side in SIDES}
        self.reports: list = []

    def model(self, side: str, exclude=()):
        key = (side, frozenset(exclude))
        if key not in self._models:
            rows = training_rows(self.panel, side, exclude, self.features[side], target_col=self.target_col)
            rows, rep = counterbalance(rows, self._ref[side], self.tol)
            m = fit_gbdt(rows, self.features[side], seed=self.seed, thread_count=self.thread_count, **self.params)
            rep.update(mode=self.mode, side=side, exclude=",".join(sorted(exclude)), n_rows=int(len(rows)),
                       best_iteration=int(getattr(m, "best_iteration_", -1) or -1))
            self.reports.append(rep)
            self._models[key] = (m, rep)
        return self._models[key]

    def predict(self, side: str, X: pd.DataFrame, exclude=()) -> np.ndarray:
        m, _ = self.model(side, exclude)
        return np.asarray(m.predict(X[self.features[side]].to_numpy(dtype=float)), dtype=float)


def gbdt_offset(prior: GBDTPrior, ro, rd, season, poss_o, poss_d, exclude=(), sides=SIDES, features=None,
                extra: pd.DataFrame | None = None) -> np.ndarray:
    """The (2m,) raw-sign GBDT offset from a window's centred rates and seasons (and, for mode "full", the role
    inputs in `extra`, aligned to ps_idx), possession-centred per side; zeros on a side not in `sides`."""
    feats = list(FEATURES if features is None else features)
    m = len(season)
    out = np.zeros(2 * m)
    for side, R, poss, off in (("O", ro, poss_o, 0), ("D", rd, poss_d, m)):
        if side not in sides:
            continue
        X = pd.DataFrame(np.asarray(R, dtype=float), columns=feats)
        X["season"] = np.asarray(season, dtype=float)
        if extra is not None:
            for c in extra.columns:
                if c in prior.features[side] and c not in X.columns:
                    X[c] = np.asarray(extra[c], dtype=float)
        g = prior.predict(side, X, exclude)
        w = np.maximum(np.asarray(poss, dtype=float), 0.0)
        if w.sum() > 0:
            g = g - np.average(g, weights=w)
        out[off:off + m] = g
    return out


def partial_dependence(model, X: pd.DataFrame, features, feature: str, grid, seasons=None) -> pd.DataFrame:
    """Mean prediction with `feature` set to each grid value (and `season` to each of `seasons`), the rest of
    the rows as they are.  chimeraboost has no PD helper; this is the plain definition."""
    feats = list(features)
    rows = []
    base = X[feats].to_numpy(dtype=float)
    j = feats.index(feature)
    js = feats.index("season") if "season" in feats else None
    for s in ([None] if (seasons is None or js is None) else seasons):
        Xs = base.copy()
        if s is not None:
            Xs[:, js] = float(s)
        for v in grid:
            Xv = Xs.copy()
            Xv[:, j] = float(v)
            rows.append(dict(feature=feature, season=s, value=float(v), pd=float(np.mean(model.predict(Xv)))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------ Boruta on chimeraboost
def _import_borutashap():
    """BorutaShap 1.0.17 against numpy 2 / scipy 1.16: two names it imports were removed."""
    import scipy.stats as st
    if not hasattr(np, "NaN"):
        np.NaN = np.nan
    if not hasattr(st, "binom_test"):
        def binom_test(x, n=None, p=0.5, alternative="two-sided"):
            return st.binomtest(int(x), int(n), p, alternative=alternative).pvalue
        st.binom_test = binom_test
    import BorutaShap
    return BorutaShap


def make_boruta(model, importance_measure: str = "shap"):
    """A BorutaShap whose SHAP importances come from chimeraboost's exact `shap_values`."""
    BS = _import_borutashap()

    class ChimeraBorutaShap(BS.BorutaShap):
        def check_model(self):          # chimeraboost exposes feature_importances_ only once fitted
            pass

        def explain(self):
            X = self.X_boruta
            sv = np.asarray(self.model.shap_values(X.to_numpy(dtype=float) if hasattr(X, "to_numpy") else X))
            if sv.ndim == 3:
                sv = np.abs(sv).sum(axis=0)
            self.shap_values = np.abs(sv).mean(0)

    return ChimeraBorutaShap(model=model, importance_measure=importance_measure, classification=False)


def run_boruta(rows: pd.DataFrame, features, n_trials: int = 50, seed: int = 0, thread_count=None,
               verbose: bool = False, **params) -> dict:
    """BorutaShap on the pooled training rows of one side; returns accepted / tentative / rejected and the history."""
    from chimeraboost import ChimeraBoostRegressor

    kw = dict(random_state=int(seed))
    if thread_count:
        kw["thread_count"] = int(thread_count)
    kw.update(params)
    fs = make_boruta(ChimeraBoostRegressor(**kw))
    X = rows[list(features)].reset_index(drop=True)
    y = pd.Series(rows["target"].to_numpy(dtype=float))
    w = pd.Series(rows["weight"].to_numpy(dtype=float))
    fs.fit(X=X, y=y, sample_weight=w, n_trials=int(n_trials), random_state=int(seed), sample=False,
           train_or_test="test", normalize=True, verbose=verbose)
    hist = getattr(fs, "history_x", None)
    return dict(accepted=sorted(fs.accepted), tentative=sorted(fs.tentative), rejected=sorted(fs.rejected),
                history=hist.copy() if hist is not None else None)
