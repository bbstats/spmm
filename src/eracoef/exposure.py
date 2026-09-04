"""BoxExposure: padded per-100 box-score exposure columns, cross-fitted by default.

mode="crossfit" (the estimator of record): the rows passed to fit/transform are one
half of the season's games (`half` = "A" or "B"); every player-season's covariate is
his padded rate from the OTHER half's games only -- one number per player-season,
independent of the fitting half's outcomes.
mode="full": rates from all games among the fitted rows (diagnostic; same-game leakage).
mode="loo": totals minus the row's own game (diagnostic; within-player jackknife artifact).

Padding: (n_eff * rate + k * target) / (n_eff + k) with a per-feature, per-season
k = sigma2_within / tau2_between (possession units) from odd/even-game split halves of the
covariate games.  `pad_target` is either the season league mean ("league") or the mean of
players in the same possession bin ("poss_conditional"), estimated on the covariate games.
n_eff is the possession count, or (sum w n)^2 / sum w^2 n when games carry bootstrap weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

K_MAX = 1e5
MODES = ("crossfit", "full", "loo")
TARGETS = ("league", "poss_conditional")
N_PAD_BINS = 6


def _lineup(X: sp.csr_matrix, cols: slice) -> np.ndarray:
    """(n, 5) player-season indices of the +1 entries in the Z block `cols`."""
    Z = X[:, cols].tocsr()
    Z.sort_indices()
    counts = np.diff(Z.indptr)
    if not np.all(counts == 5):
        raise ValueError("every row must have exactly five players in each Z block")
    return Z.indices.reshape(-1, 5)


def split_half_k(ps_idx: np.ndarray, game_idx: np.ndarray, counts: np.ndarray, poss: np.ndarray,
                 season_of_ps: np.ndarray, n_seasons: int, min_half_poss: float = 20.0):
    """Per-season, per-feature k = sigma2_within / tau2_between (possession units).

    Odd/even split by game order within each player-season.  sigma2_within is the
    per-possession noise variance of a per-100 rate (equal-weight mean over players
    of (rA-rB)^2 / (1/PA + 1/PB)); tau2 is the possession-weighted split-half
    covariance of the two half rates, which is the between-player variance of the
    true rates.  Returns (k, sigma2_w, tau2, n_players) each (n_seasons, n_feat).
    """
    n_ps = len(season_of_ps)
    n_feat = counts.shape[1]
    order = np.lexsort((game_idx, ps_idx))
    p_s, c_s, P_s = ps_idx[order], counts[order], poss[order]
    start = np.r_[True, p_s[1:] != p_s[:-1]]
    grp_start = np.maximum.accumulate(np.where(start, np.arange(len(p_s)), 0))
    rank = np.arange(len(p_s)) - grp_start
    parity = rank % 2
    CA = np.zeros((n_ps, n_feat)); CB = np.zeros((n_ps, n_feat))
    PA = np.zeros(n_ps); PB = np.zeros(n_ps)
    a = parity == 0
    np.add.at(CA, p_s[a], c_s[a]); np.add.at(PA, p_s[a], P_s[a])
    np.add.at(CB, p_s[~a], c_s[~a]); np.add.at(PB, p_s[~a], P_s[~a])
    k = np.full((n_seasons, n_feat), np.nan)
    s2 = np.full((n_seasons, n_feat), np.nan)
    t2 = np.full((n_seasons, n_feat), np.nan)
    npl = np.zeros((n_seasons, n_feat), dtype=int)
    for s in range(n_seasons):
        m = (season_of_ps == s) & (PA >= min_half_poss) & (PB >= min_half_poss)
        if m.sum() < 3:
            continue
        rA = 100.0 * CA[m] / PA[m, None]
        rB = 100.0 * CB[m] / PB[m, None]
        inv = (1.0 / PA[m] + 1.0 / PB[m])[:, None]
        sigma2_w = np.mean((rA - rB) ** 2 / inv, axis=0)
        wgt = (PA[m] + PB[m])[:, None]
        mA = (wgt * rA).sum(0) / wgt.sum()
        mB = (wgt * rB).sum(0) / wgt.sum()
        tau2 = ((wgt * (rA - mA) * (rB - mB)).sum(0) / wgt.sum())
        s2[s] = sigma2_w
        t2[s] = tau2
        npl[s] = m.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            kk = np.where(tau2 > 0, sigma2_w / np.maximum(tau2, 1e-12), K_MAX)
        k[s] = np.minimum(kk, K_MAX)
    return k, s2, t2, npl


class BoxExposure(BaseEstimator, TransformerMixin):
    """Append [Xbox_O | Xbox_D] (sum of the on-court players' padded per-100 rates) to X.

    Parameters
    ----------
    game_box : DataFrame with game_idx, ps_idx, phase and one count column per feature.
    game_poss : DataFrame with game_idx, ps_idx, poss_off, poss_def (from the stints).
    features : list of feature names (columns of game_box). None -> spec.features.
    pad_k : "auto" (per-feature, per-season k from split halves, estimated in fit), a dict
        feature -> k (same k for every season), or an (n_seasons, n_feat) array.
    pad_scale : multiplier on the k's (sanity check only).
    pad_target : "league" (season league mean) or "poss_conditional" (mean of the same
        possession bin), estimated on the covariate games.
    mode : "crossfit" | "full" | "loo".
    half : "A" | "B"; in crossfit mode the rows are this half, covariates come from the other.
    game_half : array indexed by game_idx with values "A" / "B" (RS games) or "PO".
    center : subtract the possession-weighted training mean of each exposure column.
    spec : DesignSpec describing the column layout of X.
    min_half_poss : minimum possessions in each split half for the k estimate.
    game_mult : per-game weight (bootstrap); counts and possessions are weighted by it and
        the padding uses n_eff = (sum w n)^2 / sum w^2 n.
    fixed_padding : dict with keys pad_k / league_rates / bins / target to freeze the padding
        constants at full-sample values (used inside the bootstrap).
    """

    def __init__(self, game_box=None, game_poss=None, features=None, pad_k="auto", pad_scale=1.0,
                 pad_target="league", mode="crossfit", half="A", game_half=None, center=True, spec=None,
                 min_half_poss=20.0, game_mult=None, fixed_padding=None):
        self.game_box = game_box
        self.game_poss = game_poss
        self.features = features
        self.pad_k = pad_k
        self.pad_scale = pad_scale
        self.pad_target = pad_target
        self.mode = mode
        self.half = half
        self.game_half = game_half
        self.center = center
        self.spec = spec
        self.min_half_poss = min_half_poss
        self.game_mult = game_mult
        self.fixed_padding = fixed_padding

    # ------------------------------------------------------------------ helpers
    def _feature_list(self):
        if self.features is not None:
            return list(self.features)
        return list(self.spec.features)

    def _table(self, feats):
        """Merged per-(game, player-season) counts and possessions for RS games."""
        gp = self.game_poss[["game_idx", "psx_idx", "poss_off", "poss_def"]]
        gb = self.game_box[["game_idx", "psx_idx", "phase"] + feats]
        rs_games = np.unique(gb.loc[gb["phase"] == "RS", "game_idx"].to_numpy())
        gb = gb[gb["phase"] == "RS"].drop(columns="phase")
        tab = gp.merge(gb, on=["game_idx", "psx_idx"], how="outer")
        tab[feats] = tab[feats].fillna(0.0)
        tab[["poss_off", "poss_def"]] = tab[["poss_off", "poss_def"]].fillna(0.0)
        return tab, rs_games

    def _game_idx(self, X):
        return np.asarray(X[:, self.spec.game_idx_col].toarray()).ravel().astype(np.int64)

    def _covariate_games(self, train_games, rs_games):
        """Games whose box counts form the covariates."""
        if self.mode == "crossfit":
            if self.game_half is None:
                raise ValueError("crossfit mode needs game_half")
            if self.half not in ("A", "B"):
                raise ValueError("half must be 'A' or 'B'")
            other = "B" if self.half == "A" else "A"
            gh = np.asarray(self.game_half)
            cov = np.flatnonzero(gh == other)
            own = np.flatnonzero(gh == self.half)
            bad = np.setdiff1d(train_games, own)
            if bad.size:
                raise ValueError(f"crossfit half {self.half}: {bad.size} fitted games belong to the other half")
            return np.intersect1d(cov, rs_games)
        if self.mode in ("full", "loo"):
            return np.intersect1d(train_games, rs_games)
        raise ValueError(f"unknown mode {self.mode!r}; use one of {MODES}")

    @staticmethod
    def _accumulate(n_ps, n_feat, p_idx, counts, poss, mult):
        """Weighted totals and the effective (independent-game) count for padding."""
        C = np.zeros((n_ps, n_feat)); np.add.at(C, p_idx, counts * mult[:, None])
        P = np.zeros(n_ps); np.add.at(P, p_idx, poss * mult)
        P2 = np.zeros(n_ps); np.add.at(P2, p_idx, poss * mult ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            n_eff = np.where(P2 > 0, P ** 2 / np.where(P2 > 0, P2, 1.0), 0.0)
        return C, P, n_eff

    # ------------------------------------------------------------------ sklearn API
    def fit(self, X, y=None, sample_weight=None):
        X = check_array(X, accept_sparse="csr")
        spec = self.spec
        feats = self._feature_list()
        n_feat = len(feats)
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = feats
        n_ps = spec.n_ps
        n_seasons = spec.n_seasons
        if self.pad_target not in TARGETS:
            raise ValueError(f"unknown pad_target {self.pad_target!r}; use one of {TARGETS}")

        game_idx = self._game_idx(X)
        train_games = np.unique(game_idx)
        self.train_games_ = train_games

        tab, rs_games = self._table(feats)
        cov_games = self._covariate_games(train_games, rs_games)
        self.covariate_games_ = cov_games
        n_games = int(max(tab["game_idx"].max(), game_idx.max())) + 1
        is_cov = np.zeros(n_games, dtype=bool)
        is_cov[cov_games] = True
        tab_g = tab["game_idx"].to_numpy().astype(np.int64)
        tab_cov = tab[is_cov[tab_g]]

        g_tr = tab_cov["game_idx"].to_numpy().astype(np.int64)
        x_tr = tab_cov["psx_idx"].to_numpy().astype(np.int64)
        c_tr = tab_cov[feats].to_numpy(dtype=float) if n_feat else np.zeros((len(tab_cov), 0))
        po_tr = tab_cov["poss_off"].to_numpy(dtype=float)
        pd_tr = tab_cov["poss_def"].to_numpy(dtype=float)
        mult = np.ones(len(g_tr)) if self.game_mult is None else np.asarray(self.game_mult, dtype=float)[g_tr]

        # per player-season, which is the unit the padding constants are estimated on
        n_psx = spec.n_psx
        ps_of_psx = spec.ps_of_psx if spec.ps_of_psx is not None else np.arange(n_psx)
        season_of_psx = spec.season_of_psx if spec.season_of_psx is not None else spec.season_of_ps
        Cx, Pox, Pox_eff = self._accumulate(n_psx, n_feat, x_tr, c_tr, po_tr, mult)
        _, Pdx, Pdx_eff = self._accumulate(n_psx, n_feat, x_tr, c_tr, pd_tr, mult)
        self.psx_totals_, self.psx_poss_off_, self.psx_poss_def_ = Cx, Pox, Pdx
        self.ps_of_psx_, self.season_of_psx_ = ps_of_psx, season_of_psx

        # per Z unit, summing the player-seasons that belong to it
        def to_ps(v):
            out = np.zeros((n_ps,) + v.shape[1:])
            np.add.at(out, ps_of_psx, v)
            return out

        C, Po, Pd = to_ps(Cx), to_ps(Pox), to_ps(Pdx)
        Po_eff, Pd_eff = to_ps(Pox_eff), to_ps(Pdx_eff)
        self.totals_ = C
        self.poss_off_ = Po
        self.poss_def_ = Pd
        self.poss_off_eff_ = Po_eff
        self.poss_def_eff_ = Pd_eff

        fixed = self.fixed_padding or {}
        # league rates per season (total counts / total possessions)
        if "league_rates" in fixed:
            L = np.asarray(fixed["league_rates"], dtype=float)
        else:
            L = np.zeros((n_seasons, n_feat))
            for s in range(n_seasons):
                m = season_of_psx == s
                denom = Pox[m].sum()
                L[s] = 100.0 * Cx[m].sum(0) / denom if denom > 0 else 0.0
        self.league_rates_ = L

        # padding constants from the covariate games
        if "pad_k" in fixed:
            k = np.asarray(fixed["pad_k"], dtype=float)
        elif isinstance(self.pad_k, str) and self.pad_k == "auto":
            k, s2, t2, npl = split_half_k(x_tr, g_tr, c_tr, po_tr, season_of_psx, n_seasons, self.min_half_poss)
            self.k_sigma2_within_ = s2
            self.k_tau2_between_ = t2
            self.k_n_players_ = npl
            if n_feat and np.isfinite(k).any():
                k = np.where(np.isnan(k), np.nanmean(k, axis=0, keepdims=True), k)
            else:
                k = np.zeros((n_seasons, n_feat))
        elif isinstance(self.pad_k, dict):
            k = np.tile(np.asarray([float(self.pad_k[f]) for f in feats])[None, :], (n_seasons, 1))
        else:
            k = np.asarray(self.pad_k, dtype=float)
        self.pad_k_ = k * float(self.pad_scale)

        # padding target: league mean, or the mean of the same possession bin
        if "bins" in fixed:
            self.pad_bins_ = fixed["bins"]
            self.pad_target_ = fixed.get("target")
        elif self.pad_target == "poss_conditional" and n_feat:
            self.pad_bins_, self.pad_target_ = self._fit_target_bins(Cx, Pox, season_of_psx, n_seasons, n_feat, L,
                                                                     P_bins=Po[ps_of_psx])
        else:
            self.pad_bins_, self.pad_target_ = None, None

        # LOO lookup (only used in loo mode): sorted composite key game*n_ps + ps
        key = g_tr * n_ps + ps_of_psx[x_tr]
        order = np.argsort(key)
        self.loo_keys_ = key[order]
        self.loo_counts_ = c_tr[order]
        self.loo_poss_off_ = po_tr[order]
        self.loo_poss_def_ = pd_tr[order]

        # One k and one target per Z unit: the player's own possession-weighted blend of the
        # per-season constants, so k stays a per-season measurement property and the target keeps
        # tracking league drift inside the window.
        self.pad_k_ps_ = self._blend(self.pad_k_[season_of_psx], Pox, ps_of_psx, n_ps)
        self.pad_k_ps_d_ = self._blend(self.pad_k_[season_of_psx], Pdx, ps_of_psx, n_ps)
        Tx = self._target_psx(Po, season_of_psx, ps_of_psx)
        Txd = self._target_psx(Pd, season_of_psx, ps_of_psx)
        self.target_ps_ = self._blend(Tx, Pox, ps_of_psx, n_ps)
        self.target_ps_d_ = self._blend(Txd, Pdx, ps_of_psx, n_ps)

        # padded covariate rates, one number per Z unit, plus full-window rates for reporting
        self.rates_ = self._pad_ps(C, Po, Po_eff, self.pad_k_ps_, self.target_ps_)
        self.rates_d_ = self._pad_ps(C, Pd, Pd_eff, self.pad_k_ps_d_, self.target_ps_d_)
        if self.mode == "crossfit":
            all_rs = tab[np.isin(tab_g, rs_games)]
            p_all = all_rs["psx_idx"].to_numpy().astype(np.int64)
            c_all = all_rs[feats].to_numpy(dtype=float) if n_feat else np.zeros((len(all_rs), 0))
            m_all = np.ones(len(p_all)) if self.game_mult is None else np.asarray(self.game_mult, dtype=float)[all_rs["game_idx"].to_numpy().astype(np.int64)]
            Cxs, Poxs, _ = self._accumulate(n_psx, n_feat, p_all, c_all, all_rs["poss_off"].to_numpy(dtype=float), m_all)
            _, Pdxs, _ = self._accumulate(n_psx, n_feat, p_all, c_all, all_rs["poss_def"].to_numpy(dtype=float), m_all)
            Cs, Pos, Pds = to_ps(Cxs), to_ps(Poxs), to_ps(Pdxs)
            self.season_totals_, self.season_poss_off_, self.season_poss_def_ = Cs, Pos, Pds
            ko = self._blend(self.pad_k_[season_of_psx], Poxs, ps_of_psx, n_ps)
            kd = self._blend(self.pad_k_[season_of_psx], Pdxs, ps_of_psx, n_ps)
            self.season_rates_ = self._pad_ps(
                Cs, Pos, Pos, ko, self._blend(self._target_psx(Pos, season_of_psx, ps_of_psx), Poxs, ps_of_psx, n_ps))
            self.season_rates_d_ = self._pad_ps(
                Cs, Pds, Pds, kd, self._blend(self._target_psx(Pds, season_of_psx, ps_of_psx), Pdxs, ps_of_psx, n_ps))
        else:
            self.season_totals_, self.season_poss_off_, self.season_poss_def_ = C, Po, Pd
            self.season_rates_ = self.rates_
            self.season_rates_d_ = self.rates_d_

        # centering means (possession-weighted over the training rows)
        self.means_o_ = np.zeros(n_feat)
        self.means_d_ = np.zeros(n_feat)
        if self.center and n_feat:
            Xo, Xd = self._exposures(X, game_idx)
            wgt = np.ones(X.shape[0]) if sample_weight is None else np.asarray(sample_weight, dtype=float)
            self.means_o_ = (wgt[:, None] * Xo).sum(0) / wgt.sum()
            self.means_d_ = (wgt[:, None] * Xd).sum(0) / wgt.sum()
        return self

    @staticmethod
    def _blend(v_psx, w_psx, ps_of_psx, n_ps):
        """Possession-weighted average of a per-player-season quantity, down to the Z unit."""
        num = np.zeros((n_ps, v_psx.shape[1]))
        den = np.zeros(n_ps)
        np.add.at(num, ps_of_psx, v_psx * w_psx[:, None])
        np.add.at(den, ps_of_psx, w_psx)
        cnt = np.zeros((n_ps, v_psx.shape[1]))
        np.add.at(cnt, ps_of_psx, v_psx)
        n = np.zeros(n_ps)
        np.add.at(n, ps_of_psx, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(den[:, None] > 0, num / np.where(den > 0, den, 1.0)[:, None],
                           cnt / np.maximum(n, 1.0)[:, None])
        return out

    def _target_psx(self, P_ps, season_of_psx, ps_of_psx):
        """(n_psx, n_feat) padding target: the season's level, in the bin set by the Z unit's size."""
        if self.pad_bins_ is None or self.pad_target_ is None:
            return self.league_rates_[season_of_psx]
        b = np.clip(np.searchsorted(self.pad_bins_, P_ps[ps_of_psx], side="right") - 1,
                    0, self.pad_target_.shape[1] - 1)
        return self.pad_target_[season_of_psx, b]

    @staticmethod
    def _pad_ps(C, P, n_eff, k, target):
        """(n_eff * rate + k * target) / (n_eff + k), with one k and one target per Z unit (pad.shrink)."""
        from .pad import shrink
        pos = P > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(pos[:, None], 100.0 * C / np.where(pos, P, 1.0)[:, None], 0.0)
        return shrink(rate, n_eff[:, None], k, target)

    def _fit_target_bins(self, C, P, season_of, n_seasons, n_feat, league, P_bins=None):
        """Possession-conditional padding target: league mean within log-spaced possession bins."""
        ok = P > 0
        if ok.sum() < 5 * N_PAD_BINS:
            return None, None
        Pb = P if P_bins is None else P_bins
        lo, hi = np.log(np.maximum(np.percentile(Pb[ok], 1), 1.0)), np.log(Pb[ok].max())
        edges = np.r_[0.0, np.exp(np.linspace(lo, hi, N_PAD_BINS + 1))[1:-1], np.inf]
        b_of = np.clip(np.searchsorted(edges, Pb, side="right") - 1, 0, N_PAD_BINS - 1)
        pooled = np.zeros((N_PAD_BINS, n_feat))
        for b in range(N_PAD_BINS):
            m = ok & (b_of == b)
            pooled[b] = 100.0 * C[m].sum(0) / P[m].sum() if m.sum() and P[m].sum() > 0 else np.nan
        T = np.zeros((n_seasons, N_PAD_BINS, n_feat))
        for s in range(n_seasons):
            for b in range(N_PAD_BINS):
                m = ok & (b_of == b) & (season_of == s)
                if m.sum() >= 5 and P[m].sum() > 0:
                    T[s, b] = 100.0 * C[m].sum(0) / P[m].sum()
                elif np.isfinite(pooled[b]).all():
                    T[s, b] = pooled[b]
                else:
                    T[s, b] = league[s]
        return edges, T

    def _target(self, P, season_of):
        if self.pad_bins_ is None or self.pad_target_ is None:
            return self.league_rates_[season_of]
        b = np.clip(np.searchsorted(self.pad_bins_, P, side="right") - 1, 0, self.pad_target_.shape[1] - 1)
        return self.pad_target_[season_of, b]

    def _pad(self, C, P, season_of, n_eff=None):
        """(n_eff * rate + k * target) / (n_eff + k); rate = 100 C / P."""
        k = self.pad_k_[season_of]
        T = self._target(P, season_of)
        n = P if n_eff is None else n_eff
        pos = P > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(pos[:, None], 100.0 * C / np.where(pos, P, 1.0)[:, None], 0.0)
        denom = n[:, None] + k
        return np.where(denom > 0, (n[:, None] * rate + k * T) / np.where(denom > 0, denom, 1.0), T)

    def _rates_for(self, ps, game_idx, side):
        """Padded rates (n, n_feat) for player-season `ps` on the row's game."""
        if self.mode != "loo":
            return (self.rates_ if side == "O" else self.rates_d_)[ps]
        C = self.totals_[ps].copy()
        P = (self.poss_off_ if side == "O" else self.poss_def_)[ps].copy()
        key = game_idx * self.spec.n_ps + ps
        pos = np.searchsorted(self.loo_keys_, key)
        pos_c = np.minimum(pos, len(self.loo_keys_) - 1)
        found = (pos < len(self.loo_keys_)) & (self.loo_keys_[pos_c] == key)
        if found.any():
            idx = pos_c[found]
            C[found] -= self.loo_counts_[idx]
            P[found] -= (self.loo_poss_off_ if side == "O" else self.loo_poss_def_)[idx]
        k = (self.pad_k_ps_ if side == "O" else self.pad_k_ps_d_)[ps]
        t = (self.target_ps_ if side == "O" else self.target_ps_d_)[ps]
        return self._pad_ps(C, P, P, k, t)

    def _exposures(self, X, game_idx):
        n_feat = len(self.feature_names_)
        n = X.shape[0]
        Xo = np.zeros((n, n_feat)); Xd = np.zeros((n, n_feat))
        if n_feat == 0:
            return Xo, Xd
        lo = _lineup(X, self.spec.zo)
        ld = _lineup(X, self.spec.zd)
        for j in range(5):
            Xo += self._rates_for(lo[:, j], game_idx, "O")
            Xd += self._rates_for(ld[:, j], game_idx, "D")
        return Xo, Xd

    def transform(self, X):
        check_is_fitted(self, "totals_")
        X = check_array(X, accept_sparse="csr")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of columns than at fit time")
        Xo, Xd = self._exposures(X, self._game_idx(X))
        if self.center:
            Xo = Xo - self.means_o_
            Xd = Xd - self.means_d_
        if Xo.shape[1] == 0:
            return X
        return sp.hstack([X, sp.csr_matrix(Xo), sp.csr_matrix(Xd)], format="csr")

    def get_feature_names_out(self, input_features=None):
        base = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(base + [f"O:{f}" for f in self.feature_names_] + [f"D:{f}" for f in self.feature_names_], dtype=object)

    def pad_k_table(self) -> pd.DataFrame:
        check_is_fitted(self, "pad_k_")
        return pd.DataFrame(self.pad_k_, index=self.spec.seasons, columns=self.feature_names_)

    def padding_state(self) -> dict:
        """Frozen padding constants, to reuse inside a bootstrap."""
        check_is_fitted(self, "pad_k_")
        return dict(pad_k=self.pad_k_ / float(self.pad_scale), league_rates=self.league_rates_,
                    bins=self.pad_bins_, target=self.pad_target_)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.requires_fit = True
        return tags
