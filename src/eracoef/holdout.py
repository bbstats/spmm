"""The out-of-season criterion, once.

Hold out season H.  Train on the K seasons nearest to H, balanced either side of it and never
including it.  Predict every stint of H from the ten players on the floor, with only an intercept
and a home term refit on H.  Score possession-weighted squared error in points per 100 -- at stint
level, and again after summing each team's points within a game so that noise inside a team's ~100
possessions cancels before it is squared.

This is the one test in the project that is both EXTERNAL to the model (it scores actual points in a
season the ratings never saw) and LEGAL to select on (it uses no outside data, unlike the consensus,
which is validation only).  Symmetric-around-H rather than train-on-a-block-predict-the-flanks
because the held-out season is then identical across every method and every K, so every comparison
is exactly paired and aging cancels.  FINDINGS.md sections 16-17 are the record of what it has
decided so far.

How to use it:

    ctx = Context.load(cfg)
    ho = Holdout.from_config(cfg)
    res = ho.run([PluginSystem("hybrid"), PluginSystem("hybrid_xft", target="xpts_ft")], ctx)
    print(report(res, ref="hybrid"))

A System is anything with a name and `fit(train_seasons, ctx) -> Ratings`.  `PluginSystem` is the
fit every previous script used (the box prior as an offset, one penalty); `SplitSystem` takes offense
from one system and defense from another; `TableSystem` scores a table of ratings you already have,
which is how the tests check the runner against a known truth.

Conventions, applied identically to every system so none is favoured:
  * Ratings are in the model's RAW sign: `o` adds points scored, `d` adds points ALLOWED.
  * Players absent from the training block score 0 (the average player).
  * The level (intercept, home) is refit on the held-out season.  Scoring drifts across eras and
    that is not what is under test.
  * Team-game aggregation converts rates back to points with `rows["poss"]`, never with `w`, because
    `w` is `poss * gt_weight` and the project runs gt_weight at 0.5 and 0 for robustness.
  * Lambda is FIXED across systems.  Tuning it on this criterion and then reporting this criterion
    would be circular.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from .cv import plugin_fit
from .design import WindowData
from .windows import build_window, hybrid_beta, window_label, window_seasons

RESULT_COLUMNS = ["held_out", "k", "train", "system", "lam", "split", "group", "n", "mse", "base", "calib",
                  "calib_side", "scale", "scale_off", "scale_def", "covered", "tg", "tg_base", "tg_n", "seconds"]


# ---------------------------------------------------------------------------------------- ratings
@dataclass(frozen=True)
class Ratings:
    """One row per player: `o`, `d` in raw sign, `poss` = training regular-season possessions.
    `fill_o` / `fill_d` are what a player ABSENT from the training block scores: 0 (the average player,
    the criterion's default) unless a system sets a replacement level (ReplacementSystem)."""
    df: pd.DataFrame
    fill_o: float = 0.0
    fill_d: float = 0.0

    def aligned(self, player_ids) -> pd.DataFrame:
        """The ratings in the order of `player_ids`; absent players score the fill (0 = the average player)."""
        r = pd.DataFrame({"player_id": np.asarray(player_ids)}).merge(self.df, on="player_id", how="left")
        return r.fillna({"o": float(self.fill_o), "d": float(self.fill_d), "poss": 0.0})


class System(Protocol):
    name: str

    def fit(self, train: list[int], ctx: "Context") -> Ratings: ...


# ---------------------------------------------------------------------------------------- context
def default_loader(seasons, cfg, target):
    return build_window(list(seasons), cfg, target=target)


@dataclass
class Context:
    """Everything a System needs that is not the training seasons.  Built once per run.

    `loader(seasons, cfg, target)` builds a WindowData; the default reads the real stints, the tests
    inject one that slices simulated frames.  `panel` and `base` are the inputs of the two box priors
    (outputs/xrapm_panel.parquet, outputs/coefs.parquet); None if not built, and then only systems
    that do not need them can run.  `current_h` is the one piece of state: the runner sets it before
    each fit so a prior can be kept off every window the held-out season touches.
    """
    cfg: dict
    panel: pd.DataFrame | None = None
    base: pd.DataFrame | None = None
    loader: Callable = default_loader
    current_h: int | None = None
    current_k: int | None = None
    cache_size: int = 4
    roles: pd.DataFrame | None = None          # data/cache/roles.parquet (roles.build_roles)
    role_inputs: pd.DataFrame | None = None    # roles.player_season_inputs(roles)
    rpanel: pd.DataFrame | None = None         # outputs/role_panel.parquet (scripts/49_role_panel.py)
    gbdt: object | None = None                 # gbdt_prior.GBDTPrior, mode "residual", models cached per exclusion set
    mspi: object | None = None            # the same in mode "full" (the GBDT as the whole prior)
    mspi_apm: object | None = None        # mode "full" trained on APM (unshrunk) instead of RAPM_1
    _cache: dict = field(default_factory=dict, repr=False)
    _teams: dict = field(default_factory=dict, repr=False)
    _bigness: dict = field(default_factory=dict, repr=False)
    _bench: dict = field(default_factory=dict, repr=False)
    reports: list = field(default_factory=list, repr=False)

    @classmethod
    def load(cls, cfg, loader=None) -> "Context":
        out = Path(cfg["_root"]) / "outputs"
        panel = pd.read_parquet(out / "xrapm_panel.parquet") if (out / "xrapm_panel.parquet").exists() else None
        base = None
        if (out / "coefs.parquet").exists():
            coefs = pd.read_parquet(out / "coefs.parquet")
            base = coefs[coefs.run == "base"]
        ctx = cls(cfg=cfg, panel=panel, base=base, loader=loader or default_loader)
        from .roles import player_season_inputs, roles_path
        rp = roles_path(cfg)
        if rp.exists():
            ctx.roles = pd.read_parquet(rp)
            ctx.role_inputs = player_season_inputs(ctx.roles, cap=float(cfg.get("roles", {}).get("share_cap", 0.9)))
        pp = Path(cfg["_root"]) / cfg.get("paths", {}).get("role_panel", "outputs/role_panel.parquet")
        if pp.exists():
            ctx.rpanel = pd.read_parquet(pp)
            from .gbdt_prior import GBDTPrior
            ctx.gbdt = GBDTPrior(ctx.rpanel, cfg, mode="residual")
            ctx.mspi = GBDTPrior(ctx.rpanel, cfg, mode="full")
            ctx.mspi_apm = GBDTPrior(ctx.rpanel, cfg, mode="full", target_col="apm")   # trained on unshrunk APM
        return ctx

    def bigness(self, season: int, min_minutes: float = 500.0) -> dict:
        """player_id -> True if in the top tercile of the big-man score (orb + blk + 0.3 drb - 0.5 ast - 0.4 fg3m
        per 36, the definition of scripts/22_vs_consensus.py) among players with `min_minutes` that season."""
        if season not in self._bigness:
            from .boxtable import season_box
            b = season_box([season], ["RS"], self.cfg)
            g = b.groupby("player_id")[["orb", "blk", "drb", "ast", "fg3m", "minutes"]].sum()
            g = g[g.minutes >= min_minutes]
            per36 = g[["orb", "blk", "drb", "ast", "fg3m"]].div(g.minutes.clip(lower=1), axis=0) * 36
            score = per36.orb + per36.blk + 0.3 * per36.drb - 0.5 * per36.ast - 0.4 * per36.fg3m
            cut = float(score.quantile(2 / 3)) if len(score) else np.inf
            self._bigness[season] = {int(k): bool(v >= cut) for k, v in score.items()}
        return self._bigness[season]

    def bench(self, season: int) -> dict:
        """player_id -> True if his games-started share that season is below cfg holdout.bench_gs_pct."""
        if season not in self._bench:
            thr = float(self.cfg.get("holdout", {}).get("bench_gs_pct", 0.5))
            if self.role_inputs is None:
                self._bench[season] = {}
            else:
                r = self.role_inputs[(self.role_inputs.season == season) & (self.role_inputs.games > 0)]
                self._bench[season] = {int(k): bool(v < thr) for k, v in zip(r.player_id, r.gs_pct)}
        return self._bench[season]

    @property
    def features(self) -> list:
        return list(self.cfg["features"])

    @property
    def win_of(self) -> dict:
        return {s: window_label(list(range(w[0], w[1] + 1)))
                for w in window_seasons(self.cfg) for s in range(w[0], w[1] + 1)}

    def neighbourhood(self, h: int, k: int) -> list[int]:
        """The K seasons nearest H, excluding H, balanced either side where the era allows."""
        s0, s1 = int(self.cfg["first_season"]), int(self.cfg["last_season"])
        out, d = [], 1
        while len(out) < k and d <= (s1 - s0):
            for s in (h - d, h + d):
                if s0 <= s <= s1 and len(out) < k:
                    out.append(s)
            d += 1
        return sorted(out)

    def labels(self, train) -> set:
        """Window labels a prior must stay off: the training block's and the held-out season's."""
        seasons = list(train) + ([self.current_h] if self.current_h is not None else [])
        return {self.win_of[s] for s in seasons}

    def design(self, seasons, target="pts") -> WindowData:
        """A cached design.  `target` is a design.TARGETS key, or a callable
        (seasons, cfg, wd_pts) -> WindowData | (WindowData, report) for a derived target."""
        key = (tuple(int(s) for s in seasons), target if isinstance(target, str) else getattr(target, "__name__", repr(target)))
        if key not in self._cache:
            if len(self._cache) >= self.cache_size:
                self._cache.clear()
            if isinstance(target, str):
                wd = self.loader(list(seasons), self.cfg, target)
            else:
                wd = target(list(seasons), self.cfg, self.design(seasons, "pts"))
                if isinstance(wd, tuple):
                    wd, rep = wd
                    self.reports.append(dict(seasons=key[0], target=key[1], report=rep))
            self._cache[key] = wd
        return self._cache[key]

    def main_team(self, season: int) -> dict:
        """player_id -> the team he played the most minutes for that season (box scores)."""
        if season not in self._teams:
            from .boxtable import season_box
            b = season_box([season], ["RS"], self.cfg).groupby(["player_id", "team_id"], as_index=False)["minutes"].sum()
            b = b.sort_values("minutes").drop_duplicates("player_id", keep="last")
            self._teams[season] = dict(zip(b.player_id.astype(int), b.team_id.astype(int)))
        return self._teams[season]


# ---------------------------------------------------------------------------------------- box priors
def beta_none(labs, ctx: Context) -> np.ndarray:
    """No box prior: pure RAPM."""
    return np.zeros(2 * len(ctx.features))


def beta_team(labs, ctx: Context) -> np.ndarray:
    """The team-priced prior (stint margin on LINEUP SUMS), averaged over the windows not in `labs`.

    The coefficients are near-flat across eras, so averaging the remaining windows costs almost
    nothing, and using a window's own beta where it overlaps the block or the held-out season would
    let the prior see the answer.
    """
    if ctx.base is None:
        raise RuntimeError("outputs/coefs.parquet is not built; run scripts/04_fit_all.py")
    fe = ctx.features
    d = ctx.base[~ctx.base.window.isin(labs)]
    o = d[d.side == "O"].groupby("feature")["beta"].mean().reindex(fe).to_numpy()
    dd = d[d.side == "D"].groupby("feature")["beta"].mean().reindex(fe).to_numpy()
    return np.concatenate([o, -dd])           # defense is stored flipped; the model wants it raw


def beta_hybrid(labs, ctx: Context) -> np.ndarray:
    """The shipped prior: player-priced offense, no defensive prior at all."""
    if ctx.panel is None:
        raise RuntimeError("outputs/xrapm_panel.parquet is not built; run scripts/27_xrapm_prior.py")
    return hybrid_beta(ctx.panel, ctx.features, labs)


def beta_mixed(c_def: float) -> Callable:
    """Player-priced offense with the team-priced defensive prior scaled by `c_def` (0 = hybrid, 1 = PI-RAPM)."""
    def build(labs, ctx: Context) -> np.ndarray:
        nf = len(ctx.features)
        return np.concatenate([beta_hybrid(labs, ctx)[:nf], c_def * beta_team(labs, ctx)[nf:]])
    build.__name__ = f"beta_mixed_{c_def}"
    return build


# ---------------------------------------------------------------------------------------- systems
def ratings_from_fit(wd: WindowData, pipe, beta, offset=None) -> Ratings:
    """Per-player offense and defense from a plug-in fit: the padded rates times the prior, plus the per-player
    offset when one was carried into the fit, plus the residual.  `prior_o` / `prior_d` keep the prior part."""
    exp, mm = pipe["exposure"], pipe["mm"]
    nf = len(beta) // 2
    m = wd.spec.n_ps
    ro = exp.season_rates_ - exp.means_o_ / 5.0
    rd = exp.season_rates_d_ - exp.means_d_ / 5.0
    po, pdd = ro @ beta[:nf], rd @ beta[nf:]
    if offset is not None:
        off = np.asarray(offset, dtype=float)
        po, pdd = po + off[:m], pdd + off[m:]
    df = pd.DataFrame({"player_id": wd.spec.ps_table["player_id"].to_numpy(),
                       "o": po + mm.u_[:m], "d": pdd + mm.u_[m:],
                       "poss": np.asarray(exp.season_poss_off_, dtype=float), "prior_o": po, "prior_d": pdd})
    if df.player_id.duplicated().any():
        # player-SEASON units (player_unit: season): one rating per player, pooled by possessions
        w = df.poss.clip(lower=1.0)
        df = (df.assign(o=df.o * w, d=df.d * w, prior_o=df.prior_o * w, prior_d=df.prior_d * w, w=w)
              .groupby("player_id", as_index=False)
              .agg(o=("o", "sum"), d=("d", "sum"), poss=("poss", "sum"), prior_o=("prior_o", "sum"),
                   prior_d=("prior_d", "sum"), w=("w", "sum")))
        for c in ("o", "d", "prior_o", "prior_d"):
            df[c] = df[c] / df.w
        df = df.drop(columns="w")
    return Ratings(df)


@dataclass
class PluginSystem:
    """The ratings fit every script in this project has used: a box prior as an offset, one penalty.

    `offset(train, ctx, wd) -> (2 * n_ps,)` adds a per-player prior in raw sign on top of the linear box term
    (`plugin_fit(prior_offset=...)`): the role prior and the boosted box prior of spm.py / gbdt_prior.py."""
    name: str
    target: str | Callable = "pts"
    beta: Callable = beta_hybrid
    lam: float | None = None
    lam_ratio: float | None = None
    offset: Callable | None = None

    def fit(self, train, ctx: Context) -> Ratings:
        cfg = ctx.cfg
        wd = ctx.design(train, self.target)
        b = self.beta(ctx.labels(train), ctx)
        off = None if self.offset is None else np.asarray(self.offset(train, ctx, wd), dtype=float)
        pipe = plugin_fit(wd, b, lam=float(cfg["lam_plugin"] if self.lam is None else self.lam),
                          lam_ratio=float(cfg["lam_ratio_plugin"] if self.lam_ratio is None else self.lam_ratio),
                          pad_target=cfg["pad_target"], prior_offset=off)
        return ratings_from_fit(wd, pipe, b, offset=off)


@dataclass
class SplitSystem:
    """Offense from one system, defense from another -- e.g. offense on a luck-adjusted target, defense on points."""
    name: str
    offense: System
    defense: System

    def fit(self, train, ctx: Context) -> Ratings:
        fo = self.offense.fit(train, ctx).df
        fd = self.defense.fit(train, ctx).df
        o = fo[["player_id", "o", "poss", *([c for c in ("prior_o",) if c in fo.columns])]]
        d = fd[["player_id", "d", "poss", *([c for c in ("prior_d",) if c in fd.columns])]].rename(columns={"poss": "poss_d"})
        r = o.merge(d, on="player_id", how="outer")
        r["poss"] = r.poss.fillna(r.poss_d)
        cols = ["player_id", "o", "d", "poss", *([c for c in ("prior_o", "prior_d") if c in r.columns])]
        return Ratings(r.fillna({c: 0.0 for c in cols if c != "player_id"})[cols])


@dataclass
class MappedSystem:
    """A system with a per-side monotone map applied to its ratings (the rank-calibration correction)."""
    name: str
    inner: System
    map_o: Callable = lambda x: x
    map_d: Callable = lambda x: x

    def fit(self, train, ctx: Context) -> Ratings:
        r = self.inner.fit(train, ctx).df.copy()
        r["o"] = self.map_o(r.o.to_numpy())
        r["d"] = self.map_d(r.d.to_numpy())
        return Ratings(r)


@dataclass
class RankMappedSystem:
    """The rank-calibration correction, scored honestly: the map for held-out season H is fitted on
    the pooled decile slopes of every OTHER held-out season (`fit_rank_map`, exclude_h = H), at the
    same K, so H's own outcomes never choose H's correction.  `rank` is Holdout.rank_ from an earlier
    run of `inner` with rank=True."""
    name: str
    inner: System
    rank: pd.DataFrame
    sides: tuple = ("o", "d")

    def fit(self, train, ctx: Context) -> Ratings:
        r = self.inner.fit(train, ctx).df.copy()
        k = ctx.current_k if ctx.current_k is not None else int(self.rank.k.iloc[0])
        for side in self.sides:
            f = fit_rank_map(self.rank, self.inner.name, side, k, exclude_h=ctx.current_h)
            r[side] = f(r[side].to_numpy())
        return Ratings(r)


@dataclass
class ReplacementSystem:
    """`inner` with a replacement level for players the training block never saw: the possession-weighted
    mean rating of the block's low-exposure players (under `max_poss` training possessions), per side.
    Uses no held-out data; the criterion's default gives an unseen rookie the average player's 0."""
    name: str
    inner: System
    max_poss: float = 500.0

    def fit(self, train, ctx: Context) -> Ratings:
        r = self.inner.fit(train, ctx)
        low = r.df[(r.df.poss > 0) & (r.df.poss < self.max_poss)]
        if len(low) == 0:
            return r
        w = low.poss.to_numpy(dtype=float)
        return Ratings(r.df, fill_o=float(np.average(low.o, weights=w)), fill_d=float(np.average(low.d, weights=w)))


@dataclass
class TableSystem:
    """Ratings from a table you already have (player_id, season, o, d[, poss]): the mean over the training seasons."""
    name: str
    table: pd.DataFrame

    def fit(self, train, ctx: Context) -> Ratings:
        t = self.table[self.table.season.isin(list(train))]
        if "poss" not in t.columns:
            t = t.assign(poss=1.0)
        g = t.groupby("player_id", as_index=False).agg(o=("o", "mean"), d=("d", "mean"), poss=("poss", "sum"))
        return Ratings(g)


# ---------------------------------------------------------------------------------------- scoring
@dataclass
class Prediction:
    """Everything about one system's prediction of one held-out season that the scores and splits need."""
    y: np.ndarray
    w: np.ndarray
    poss: np.ndarray
    home: np.ndarray
    c_off: np.ndarray          # the offensive five's ratings summed, per row
    c_def: np.ndarray
    pred: np.ndarray           # intercept + home refit on the season, plus c_off + c_def
    base_pred: np.ndarray      # intercept + home only
    game_idx: np.ndarray
    is_home_off: np.ndarray
    rat: pd.DataFrame          # ratings aligned to the season's Z columns
    A: np.ndarray              # the level columns refit on the season (level_columns)


def _wls(A, y, w):
    """Weighted least squares; a zero column (a system with no ratings) gets coefficient 0, not an error."""
    AtW = (A * w[:, None]).T
    return np.linalg.lstsq(AtW @ A, AtW @ y, rcond=None)[0]


def level_columns(wd_h: WindowData, level: str = "home") -> np.ndarray:
    """The nuisance columns refit on the held-out season.

    "home"  an intercept and the home term -- what every earlier script used
    "full"  the design's whole fixed block: intercept, home, playoff, garbage time, margin and the
            margin-by-time term.  Those are context, not attribution, and leaving them out attenuates
            the ratings wherever a strong lineup builds a lead and then scores less on it.
    """
    m = wd_h.spec.n_ps
    if level == "home":
        home = np.asarray(wd_h.X[:, wd_h.spec.f_col("home")].todense()).ravel()
        return np.column_stack([np.ones(wd_h.X.shape[0]), home])
    if level == "full":
        nfix = len(wd_h.spec.f_names)
        F = np.asarray(wd_h.X[:, 2 * m:2 * m + nfix].todense())
        return np.column_stack([np.ones(wd_h.X.shape[0]), F])
    raise ValueError(f"level must be 'home' or 'full', got {level!r}")


def predict_season(rat: Ratings, wd_h: WindowData, level: str = "home") -> Prediction:
    """Predict every row of the held-out season from the ratings; only the level (`level_columns`) is refit."""
    m = wd_h.spec.n_ps
    r = rat.aligned(wd_h.spec.ps_table["player_id"].to_numpy())
    c_off = wd_h.X[:, :m] @ r["o"].to_numpy()
    c_def = wd_h.X[:, m:2 * m] @ r["d"].to_numpy()
    y, w = wd_h.y, wd_h.w
    home = np.asarray(wd_h.X[:, wd_h.spec.f_col("home")].todense()).ravel()
    A = level_columns(wd_h, level)
    contrib = c_off + c_def
    pred = A @ _wls(A, y - contrib, w) + contrib
    base_pred = A @ _wls(A, y, w)
    return Prediction(y=y, w=w, poss=wd_h.rows["poss"].to_numpy(dtype=float), home=home, c_off=c_off, c_def=c_def,
                      pred=pred, base_pred=base_pred, game_idx=wd_h.rows["game_idx"].to_numpy(),
                      is_home_off=wd_h.rows["is_home_off"].to_numpy(), rat=r, A=A)


def team_game_mse(y, pred, poss, game_idx, is_home_off) -> tuple[float, float]:
    """Sum actual and predicted POINTS over each team's rows in a game, then score; returns (mse, possessions)."""
    g = pd.DataFrame({"g": game_idx, "h": is_home_off, "poss": poss,
                      "act": y * poss / 100.0, "prd": pred * poss / 100.0}).groupby(["g", "h"]).sum()
    return float(np.average(((g.act - g.prd) / g.poss * 100.0) ** 2, weights=g.poss)), float(g.poss.sum())


def score(p: Prediction, mask=None) -> dict:
    """The scores of a prediction, on all rows or on a subset (the level stays the full-season refit).

      mse, base        weighted stint MSE with the ratings and with the level only
      calib, calib_side  MSE after a scalar (per side: two scalars) fitted ON the held-out season --
                       an amplitude diagnostic and an upper bound, never a score
      scale_*          those scalars; 1.0 = calibrated, below 1 = the ratings are too wide
      covered          share of the season's players the training block had a rating for
      tg, tg_base      the same two errors after team-game aggregation
    """
    idx = np.arange(len(p.y)) if mask is None else np.flatnonzero(mask)
    y, w = p.y[idx], p.w[idx]

    def wmse(pred):
        return float(np.average((y - pred) ** 2, weights=w))

    Ac = np.column_stack([p.A[idx], p.c_off[idx] + p.c_def[idx]])
    cc = _wls(Ac, y, w)
    A2 = np.column_stack([p.A[idx], p.c_off[idx], p.c_def[idx]])
    c2 = _wls(A2, y, w)
    tg, tg_n = team_game_mse(y, p.pred[idx], p.poss[idx], p.game_idx[idx], p.is_home_off[idx])
    tg_base, _ = team_game_mse(y, p.base_pred[idx], p.poss[idx], p.game_idx[idx], p.is_home_off[idx])
    return dict(n=float(w.sum()), mse=wmse(p.pred[idx]), base=wmse(p.base_pred[idx]), calib=wmse(Ac @ cc),
                calib_side=wmse(A2 @ c2), scale=float(cc[-1]), scale_off=float(c2[-2]), scale_def=float(c2[-1]),
                covered=float((p.rat.poss.to_numpy() > 0.0).mean()), tg=tg, tg_base=tg_base, tg_n=tg_n)


# ---------------------------------------------------------------------------------------- splits
SplitFn = Callable[[Prediction, WindowData, Context, int, list], np.ndarray]   # -> group label per row


def by_movers(p: Prediction, wd_h: WindowData, ctx: Context, h: int, train: list) -> np.ndarray:
    """How many of the ten on the floor changed team between the training block and the held-out season.

    A player MOVED if his main team in H differs from his main team in the nearest training season he
    appears in.  Players absent from either are not movers, they are unknown.  Mid-season trades count
    only if the bulk of his minutes shifted.  This is the traded-to-an-average-team test: a rating
    that is really his old teammates' credit stops predicting once he plays with new ones.
    """
    m = wd_h.spec.n_ps
    ids = wd_h.spec.ps_table["player_id"].to_numpy()
    now = ctx.main_team(h)
    before: dict = {}
    for s in sorted(train, key=lambda s: abs(s - h)):
        for pid, t in ctx.main_team(s).items():
            before.setdefault(pid, t)
    moved = np.array([1.0 if (q in now and q in before and now[q] != before[q]) else 0.0 for q in ids])
    n_movers = wd_h.X[:, :2 * m] @ np.tile(moved, 2)
    return np.where(n_movers == 0, "no movers", np.where(n_movers <= 2, "1-2 movers", "3+ movers"))


def by_exposure(p: Prediction, wd_h: WindowData, ctx: Context, h: int, train: list) -> np.ndarray:
    """The SMALLEST training exposure among the ten players on the row, binned (cfg holdout.exposure_edges)."""
    edges = [float(e) for e in ctx.cfg.get("holdout", {}).get("exposure_edges", [0, 1, 500, 1500, 4000, 1e9])]
    labels = [f"{int(a)}-{int(b) - 1}" if b < 1e8 else f"{int(a)}+" for a, b in zip(edges[:-1], edges[1:])]
    labels[0] = "none (0)" if edges[1] == 1 else labels[0]
    m = wd_h.spec.n_ps
    big = 1e9
    poss = p.rat.poss.to_numpy(dtype=float)
    # Z has exactly five ones per block, so a min over the on-court players is a max over (big - poss)
    minposs = big - (wd_h.X[:, :2 * m].multiply(np.tile(big - poss, 2))).max(axis=1).toarray().ravel()
    b = np.clip(np.digitize(minposs, edges) - 1, 0, len(labels) - 1)
    return np.asarray(labels, dtype=object)[b]


def _count_split(flag: np.ndarray, wd_h: WindowData, edges, what: str) -> np.ndarray:
    """Rows binned by how many of the ten on the floor carry `flag`, with edges like [0, 3, 4, 5, 11]."""
    m = wd_h.spec.n_ps
    edges = [float(e) for e in edges]
    labels = []
    for a, b in zip(edges[:-1], edges[1:]):
        a, b = int(a), int(b)
        labels.append(f"{a} {what}" if b == a + 1 else (f"{a}+ {what}" if b > 10 else f"{a}-{b - 1} {what}"))
    n = wd_h.X[:, :2 * m] @ np.tile(np.asarray(flag, dtype=float), 2)
    b = np.clip(np.digitize(n, edges) - 1, 0, len(labels) - 1)
    return np.asarray(labels, dtype=object)[b]


def by_bigs(p: Prediction, wd_h: WindowData, ctx: Context, h: int, train: list) -> np.ndarray:
    """How many of the ten on the floor are big men (top tercile of the big-man score in season H).
    The loss-wise archetype check: is a system worse where the bigs are, not just tilted toward them."""
    big = ctx.bigness(h)
    ids = wd_h.spec.ps_table["player_id"].to_numpy()
    flag = np.array([1.0 if big.get(int(q), False) else 0.0 for q in ids])
    return _count_split(flag, wd_h, ctx.cfg.get("holdout", {}).get("bigs_edges", [0, 3, 4, 5, 11]), "bigs")


def by_bench(p: Prediction, wd_h: WindowData, ctx: Context, h: int, train: list) -> np.ndarray:
    """How many of the ten on the floor are bench players in season H (games-started share below
    cfg holdout.bench_gs_pct): where the role prior is meant to help, and where garbage time lives."""
    bench = ctx.bench(h)
    ids = wd_h.spec.ps_table["player_id"].to_numpy()
    flag = np.array([1.0 if bench.get(int(q), False) else 0.0 for q in ids])
    return _count_split(flag, wd_h, ctx.cfg.get("holdout", {}).get("bench_edges", [0, 3, 5, 7, 11]), "bench")


def by_gt(p: Prediction, wd_h: WindowData, ctx: Context, h: int, train: list) -> np.ndarray:
    """Garbage-time rows (the stint parser's rule: a 4th-quarter margin of 15 late or 20 at any point) against the rest."""
    gt = wd_h.rows["is_gt"].to_numpy().astype(bool)
    return np.where(gt, "garbage time", "competitive")


SPLITS = {"movers": by_movers, "exposure": by_exposure, "bigs": by_bigs, "bench": by_bench, "gt": by_gt}


# ---------------------------------------------------------------------------------------- the runner
@dataclass
class Holdout:
    cfg: dict
    first: int
    last: int
    ks: list
    lams: list
    rank_bins: int = 10
    rank_min_poss: float = 1000.0
    level: str = "home"
    rank_: pd.DataFrame | None = field(default=None, repr=False)

    @classmethod
    def from_config(cls, cfg, **override) -> "Holdout":
        h = dict(cfg.get("holdout", {}))
        h.update({k: v for k, v in override.items() if v is not None})
        lam = h.get("lam")
        lams = [float(cfg["lam_plugin"])] if lam is None else [float(x) for x in (lam if isinstance(lam, list) else [lam])]
        return cls(cfg=cfg, first=int(h.get("first", cfg["first_season"] + 1)), last=int(h.get("last", cfg["last_season"] - 1)),
                   ks=[int(k) for k in h.get("ks", [2, 4])], lams=lams, rank_bins=int(h.get("rank_bins", 10)),
                   rank_min_poss=float(h.get("rank_min_poss", 1000)), level=str(h.get("level", "home")))

    def seasons(self) -> list[int]:
        s0, s1 = int(self.cfg["first_season"]), int(self.cfg["last_season"])
        return [h for h in range(self.first, self.last + 1) if s0 <= h <= s1]

    def run(self, systems: list, ctx: Context, splits: dict | None = None, rank: bool = False,
            out: Path | None = None, verbose: bool = True, held: list | None = None) -> pd.DataFrame:
        """One row per held-out season x K x lambda x system x split x group, columns RESULT_COLUMNS.
        `held` restricts the held-out seasons (a worker's share of them); default = all of `seasons()`."""
        splits = splits or {}
        rows, rank_rows = [], []
        t0 = time.time()
        held = self.seasons() if held is None else [int(h) for h in held]
        if verbose:
            print(f"held-out {held[0]}-{held[-1]} ({len(held)}), K {self.ks}, lambdas {self.lams}, "
                  f"systems {[s.name for s in systems]}", flush=True)
        for h in held:
            ctx.current_h = h
            wd_h = ctx.design([h], "pts")                       # ALWAYS scored against actual points
            for k in self.ks:
                ctx.current_k = k
                train = ctx.neighbourhood(h, k)
                for system in systems:
                    for lam in self.lams:
                        t1 = time.time()
                        rat = _fit(system, train, ctx, lam)
                        p = predict_season(rat, wd_h, level=self.level)
                        base = dict(held_out=h, k=k, train=",".join(map(str, train)), system=system.name, lam=lam)
                        rows.append(dict(**base, split="all", group="all", **score(p), seconds=time.time() - t1))
                        for sname, fn in splits.items():
                            grp = fn(p, wd_h, ctx, h, train)
                            for g in pd.unique(grp):
                                rows.append(dict(**base, split=sname, group=str(g), **score(p, grp == g), seconds=0.0))
                        if rank:
                            rc = rank_calibration(p, wd_h, n_bins=self.rank_bins, min_poss=self.rank_min_poss)
                            rank_rows.append(rc.assign(**base))
            if verbose:
                print(f"  {h} done ({time.time() - t0:.0f}s)", flush=True)
            if out is not None:
                pd.DataFrame(rows)[RESULT_COLUMNS].to_parquet(out, index=False)       # checkpoint
        res = pd.DataFrame(rows)[RESULT_COLUMNS]
        if rank_rows:
            self.rank_ = pd.concat(rank_rows, ignore_index=True)
        if out is not None:
            res.to_parquet(out, index=False)
        ctx.current_h = None
        return res


def _fit(system, train, ctx, lam):
    """Fit with the run's lambda where the system leaves it open."""
    if isinstance(system, PluginSystem) and system.lam is None:
        return replace(system, lam=lam).fit(train, ctx)
    return system.fit(train, ctx)


# ---------------------------------------------------------------------------------------- parallel runner
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")


def _worker(job: dict):
    """One process's share of the held-out seasons.  Systems are rebuilt by name from `systems.registry`
    inside the child (closures do not pickle), on a fresh Context with its own stint cache."""
    from .systems import registry
    cfg = job["cfg"]
    ctx = Context.load(cfg)
    reg = registry(cfg, rankmap=job.get("rankmap"))
    systems = [reg[n] for n in job["names"]]
    ho: Holdout = job["holdout"]
    res = ho.run(systems, ctx, splits={s: SPLITS[s] for s in job.get("splits", [])}, rank=job.get("rank", False),
                 held=job["held"], verbose=job.get("verbose", True))
    gb = [r for prior in (ctx.gbdt, ctx.mspi, ctx.mspi_apm) if prior is not None for r in getattr(prior, "reports", [])]
    return res, ho.rank_, gb, list(ctx.reports)


def run_parallel(ho: "Holdout", names: list, splits=(), rank: bool = False, out: Path | None = None,
                 verbose: bool = True, workers: int = 4, rankmap=None) -> tuple[pd.DataFrame, pd.DataFrame | None, list]:
    """`Holdout.run` over `workers` spawned processes, contiguous blocks of held-out seasons each, with the
    BLAS / numba thread count pinned to cpu_count // workers so the processes do not oversubscribe the
    machine (the handoff's 20x trap).  Returns (results, rank rows or None, the GBDT drag reports)."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    held = ho.seasons()
    workers = max(1, min(int(workers), len(held)))
    threads = max(1, (os.cpu_count() or workers) // workers)
    chunks = [c.tolist() for c in np.array_split(np.asarray(held), workers)]
    old = {k: os.environ.get(k) for k in THREAD_VARS}
    os.environ.update({k: str(threads) for k in THREAD_VARS})
    t0 = time.time()
    try:
        if verbose:
            print(f"parallel: {workers} workers x {threads} threads, seasons {[ (c[0], c[-1]) for c in chunks ]}", flush=True)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            jobs = [dict(cfg=ho.cfg, holdout=ho, names=list(names), held=c, splits=list(splits), rank=rank,
                         rankmap=rankmap, verbose=verbose) for c in chunks if c]
            parts = [f.result() for f in [ex.submit(_worker, j) for j in jobs]]
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    res = pd.concat([p[0] for p in parts], ignore_index=True).sort_values(["held_out", "k", "lam", "system"]).reset_index(drop=True)
    ranks = [p[1] for p in parts if p[1] is not None]
    rank_ = pd.concat(ranks, ignore_index=True) if ranks else None
    ho.rank_ = rank_
    reports = [r for p in parts for r in p[2]]
    if out is not None:
        res[RESULT_COLUMNS].to_parquet(out, index=False)
    if verbose:
        print(f"parallel run done ({time.time() - t0:.0f}s)", flush=True)
    return res[RESULT_COLUMNS], rank_, reports


# ---------------------------------------------------------------------------------------- summaries
POOL_BY = ("k", "lam", "system", "split", "group")


def pooled(res: pd.DataFrame, by=POOL_BY) -> pd.DataFrame:
    """Possession-weighted means over the held-out seasons.  `vs_no_ratings` = share of error the ratings remove."""
    def agg(d):
        n, tn = d.n.sum(), d.tg_n.sum()
        return pd.Series({
            "seasons": int(d.held_out.nunique()),
            "mse": float((d.mse * d.n).sum() / n),
            "base": float((d.base * d.n).sum() / n),
            "vs_no_ratings": 1.0 - float((d.mse * d.n).sum() / (d.base * d.n).sum()),
            "calib_side": float((d.calib_side * d.n).sum() / n),
            "scale_off": float((d.scale_off * d.n).sum() / n),
            "scale_def": float((d.scale_def * d.n).sum() / n),
            "covered": float(d.covered.mean()),
            "game": float((d.tg * d.tg_n).sum() / tn),
            "game_base": float((d.tg_base * d.tg_n).sum() / tn),
            "game_vs_no_ratings": 1.0 - float((d.tg * d.tg_n).sum() / (d.tg_base * d.tg_n).sum()),
            "share": float(n),
        })
    out = res.groupby(list(by), sort=True).apply(agg, include_groups=False).reset_index()
    tot = out.groupby([b for b in by if b != "group"] if "group" in by else list(by))["share"].transform("sum")
    out["share"] = out["share"] / tot
    return out


def paired(res: pd.DataFrame, ref: str, value: str = "mse", by=("k", "lam", "split", "group")) -> pd.DataFrame:
    """Every system against `ref`, paired by held-out season: mean difference, its SE, z, seasons won.

    `value` is "mse" (stint) or "tg" (team-game).  Negative = better than the reference.
    """
    rows = []
    for key, d in res.groupby(list(by), sort=True):
        p = d.pivot_table(index="held_out", columns="system", values=value)
        if ref not in p.columns:
            continue
        for s in p.columns:
            if s == ref:
                continue
            diff = (p[s] - p[ref]).dropna()
            n = len(diff)
            sd = diff.std(ddof=1) if n > 1 else np.nan
            rows.append(dict(**dict(zip(by, key)), system=s, ref=ref, value=value, mean_diff=float(diff.mean()),
                             se=float(sd / np.sqrt(n)) if n > 1 else np.nan,
                             z=float(diff.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else np.nan,
                             wins=int((diff < 0).sum()), n_seasons=n))
    return pd.DataFrame(rows)


def report(res: pd.DataFrame, ref: str | None = None, digits: int = 4) -> str:
    """The printed summary: pooled scores per K, then the paired tests against `ref` at stint and game level."""
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", digits)
    lines = ["=== out-of-season reliability, pooled over held-out seasons (LOWER mse/game IS BETTER)",
             "    vs_no_ratings: share of held-out error the ratings remove.  scale_*: what the held-out season",
             "    wants the ratings multiplied by (1 = calibrated; below 1 = too wide) -- a diagnostic, not a score."]
    P = pooled(res)
    cols = ["system", "seasons", "mse", "vs_no_ratings", "scale_off", "scale_def", "covered", "game", "game_vs_no_ratings"]
    for (k, lam, split, group), d in P.groupby(["k", "lam", "split", "group"], sort=True):
        head = f"\n-- K = {k}, lambda {lam:.0f}" + ("" if split == "all" else f", {split} = {group} (share {d.share.iloc[0]:.2f})")
        lines.append(head)
        lines.append(d[cols].to_string(index=False))
    if ref is not None and ref in set(res.system):
        lines.append(f"\n=== paired by held-out season against {ref}; negative = better")
        for value, label in (("mse", "stint"), ("tg", "team-game")):
            t = paired(res, ref, value)
            if len(t):
                lines.append(f"\n-- {label} level")
                lines.append(t.drop(columns=["ref", "value"]).to_string(index=False))
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------- rank calibration
def rank_calibration(p: Prediction, wd_h: WindowData, n_bins: int = 10, min_poss: float = 1000.0) -> pd.DataFrame:
    """What the held-out season wants each decile of the ratings multiplied by, out of sample.

    Player rank (side "o" / "d"): players with `min_poss` training possessions are ranked into deciles
    by their rating; everyone else is the "unranked" group.  Regress the held-out stints on the ten
    per-decile lineup contributions (plus the other side's whole contribution, the intercept and home):
    slope_j is the multiplier the season wants on decile j, 1.0 = calibrated.  A slope that rises with
    rank says the top is under-rated; one that falls says the top is over-rated.

    Lineup rank (side "lineup"): rows binned by their predicted margin; within each, the WLS slope of
    actual on predicted and the mean residual -- the calibration of "players with these coefficients,
    when they get together".
    """
    m = wd_h.spec.n_ps
    y, w = p.y, p.w
    one = np.ones(len(y))
    rows = []
    for side, col, Z, other in (("o", "o", wd_h.X[:, :m], p.c_def), ("d", "d", wd_h.X[:, m:2 * m], p.c_off)):
        v = p.rat[col].to_numpy(dtype=float)
        poss = p.rat.poss.to_numpy(dtype=float)
        ranked = (poss >= min_poss) & (v != 0.0)
        dec = np.full(len(v), -1)
        if ranked.sum() >= n_bins * 2:
            dec[ranked] = pd.qcut(v[ranked], n_bins, labels=False, duplicates="drop")
        groups = sorted(set(dec.tolist()))
        # The slope as a smooth function of the rating, s(v) = c1 + c2 u + c3 u^2 with u the
        # standardised rating, reported at each decile's mean.  NOT one slope per decile: every row
        # has exactly five players a side, so the decile headcounts sum to a constant and a per-decile
        # fit is collinear with the intercept (it came out 0.3 to 0.9 on ratings that were exactly
        # right).  The curve is identified, and it answers the question -- does the top of the board
        # want a bigger or a smaller multiplier than the bottom -- directly.
        u = np.zeros(len(v))
        if ranked.sum() >= 3:
            u[ranked] = (v[ranked] - v[ranked].mean()) / max(v[ranked].std(), 1e-9)
        A = np.column_stack([p.A, other, Z @ v, Z @ (v * u), Z @ (v * u ** 2)])
        AtW = (A * w[:, None]).T
        G = AtW @ A
        coef = np.linalg.lstsq(G, AtW @ y, rcond=None)[0]
        e = y - A @ coef
        sigma2 = float(np.average(e ** 2, weights=w))
        cov = sigma2 * np.linalg.pinv(G) * (w.sum() / len(w))    # weights are possessions: rescale to a per-row variance
        c, V = coef[-3:], cov[-3:, -3:]
        for g in groups:
            sel = dec == g
            ug = float(u[sel].mean()) if sel.any() else 0.0
            b = np.array([1.0, ug, ug ** 2])
            rows.append(dict(side=side, decile=int(g), n_players=int(sel.sum()),
                             mean_rating=float(v[sel].mean()) if sel.any() else np.nan,
                             slope=float(b @ c), se=float(np.sqrt(max(b @ V @ b, 0.0)))))
    # lineup rank
    margin = p.pred - p.base_pred
    q = (np.asarray(pd.qcut(margin, n_bins, labels=False, duplicates="drop"), dtype=float)
         if np.ptp(margin) > 0 else np.full(len(margin), np.nan))      # a system with no ratings has no lineup rank
    for g in sorted(pd.unique(q[np.isfinite(q)])):
        sel = q == g
        yy, ww, pp = y[sel], w[sel], p.pred[sel]
        A = np.column_stack([np.ones(sel.sum()), pp])
        c = _wls(A, yy, ww)
        rows.append(dict(side="lineup", decile=int(g), n_players=int(sel.sum()),
                         mean_rating=float(np.average(pp - p.base_pred[sel], weights=ww)),
                         slope=float(c[1]), se=np.nan,
                         mean_resid=float(np.average(yy - pp, weights=ww))))
    return pd.DataFrame(rows)


def pooled_rank(rank: pd.DataFrame, exclude_h=None) -> pd.DataFrame:
    """Inverse-variance pooled decile slopes over held-out seasons; z = (slope - 1) / se."""
    r = rank if exclude_h is None else rank[rank.held_out != exclude_h]
    out = []
    for (k, lam, system, side, dec), d in r.groupby(["k", "lam", "system", "side", "decile"]):
        if side == "lineup" or d.se.isna().all():
            out.append(dict(k=k, lam=lam, system=system, side=side, decile=dec, slope=float(d.slope.mean()),
                            se=float(d.slope.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan,
                            mean_rating=float(d.mean_rating.mean()), seasons=len(d)))
        else:
            iv = 1.0 / np.maximum(d.se.to_numpy() ** 2, 1e-12)
            slope = float((d.slope.to_numpy() * iv).sum() / iv.sum())
            out.append(dict(k=k, lam=lam, system=system, side=side, decile=dec, slope=slope,
                            se=float(np.sqrt(1.0 / iv.sum())), mean_rating=float(d.mean_rating.mean()), seasons=len(d)))
    o = pd.DataFrame(out)
    o["z"] = (o.slope - 1.0) / o.se
    return o


def fit_rank_map(rank: pd.DataFrame, system: str, side: str, k: int, exclude_h=None) -> Callable:
    """A monotone piecewise-linear map rating -> slope(rating) * rating from the pooled decile slopes,
    leaving `exclude_h` out so the correction is scored on a season that did not choose it."""
    pr = pooled_rank(rank, exclude_h)
    pr = pr[(pr.system == system) & (pr.side == side) & (pr.k == k) & (pr.decile >= 0)].sort_values("mean_rating")
    if len(pr) < 2:
        return lambda x: x
    xs = pr.mean_rating.to_numpy()
    ys = np.maximum.accumulate(xs * pr.slope.to_numpy())          # enforce monotone

    def f(x):
        x = np.asarray(x, dtype=float)
        lo, hi = pr.slope.iloc[0], pr.slope.iloc[-1]
        out = np.interp(x, xs, ys)
        out = np.where(x < xs[0], ys[0] + lo * (x - xs[0]), out)
        return np.where(x > xs[-1], ys[-1] + hi * (x - xs[-1]), out)
    return f


# ---------------------------------------------------------------------------------------- validation and diagnostics
def vs_consensus(rat: Ratings, cfg, min_poss: float | None = None) -> dict:
    """The external check, read once: rank agreement with the consensus of modern metrics on 2024-26.

    Nothing is ever selected on this.  Ratings arrive in raw sign; defense is flipped here so that
    positive = good, as on the board.
    """
    out = Path(cfg["_root"]) / "outputs" / "vs_consensus.parquet"
    if not out.exists():
        raise RuntimeError("outputs/vs_consensus.parquet is not built; run scripts/22_vs_consensus.py")
    from scipy.stats import spearmanr
    min_poss = float(cfg.get("holdout", {}).get("consensus", {}).get("min_poss", 1000) if min_poss is None else min_poss)
    con = pd.read_parquet(out)[["player_id", "adj_offense", "adj_defense", "adj_overall", "bigness"]]
    g = rat.df.merge(con, on="player_id")
    g = g[g.poss >= min_poss].copy()
    g["dd"] = -g.d
    g["t"] = g.o + g.dd

    def z(s):
        return (s - s.mean()) / s.std()

    return dict(n=int(len(g)), con_total=float(spearmanr(g.t, g.adj_overall).statistic),
                con_off=float(spearmanr(g.o, g.adj_offense).statistic),
                con_def=float(spearmanr(g.dd, g.adj_defense).statistic),
                spread_off=float(g.o.std() / g.adj_offense.std()), spread_def=float(g.dd.std() / g.adj_defense.std()),
                bias_total=float((z(g.t) - z(g.adj_overall)).corr(g.bigness)))


def _team_of_rows(p: Prediction, wd_h: WindowData, main_team: dict) -> tuple[np.ndarray, np.ndarray]:
    """The offensive and defensive team of every row: the modal main team of the five on each side."""
    m = wd_h.spec.n_ps
    ids = wd_h.spec.ps_table["player_id"].to_numpy()
    team = np.array([main_team.get(int(q), -1) for q in ids])
    Zo, Zd = wd_h.X[:, :m].tocsr(), wd_h.X[:, m:2 * m].tocsr()

    def modal(Z):
        t = team[Z.indices].reshape(Z.shape[0], 5)
        out = np.empty(Z.shape[0], dtype=int)
        for i in range(Z.shape[0]):
            vals, cnt = np.unique(t[i], return_counts=True)
            out[i] = vals[np.argmax(cnt)]
        return out
    return modal(Zo), modal(Zd)


def team_residual(p: Prediction, wd_h: WindowData, main_team: dict) -> pd.DataFrame:
    """How much each team out- or under-performed its players' ratings, on offense and on defense.

    The residual mean per team with its SE (delta method on the possession weights).  A rating that
    captures "how much he helps his current team" leaves nothing here; a systematic residual is the
    part of team quality the additive model does not carry -- fit, scheme, or chemistry.
    """
    to, td = _team_of_rows(p, wd_h, main_team)
    e = p.y - p.pred
    rows = []
    for t in sorted(set(to.tolist()) - {-1}):
        for side, sel in (("offense", to == t), ("defense", td == t)):
            w = p.w[sel]
            mean = float(np.average(e[sel], weights=w))
            se = float(np.sqrt((w ** 2 * (e[sel] - mean) ** 2).sum()) / w.sum())
            rows.append(dict(team_id=int(t), side=side, poss=float(p.poss[sel].sum()), resid=mean, se=se,
                             z=mean / se if se > 0 else np.nan))
    return pd.DataFrame(rows)


def replacement_quality(p: Prediction, wd_h: WindowData, main_team: dict, min_poss: float = 1000.0) -> pd.DataFrame:
    """Is a player's rating tied to how bad the players who replace him are?

    For each player on a team: the mean rating-sum of his team's OFFENSIVE lineups when he sits, minus the
    mean rating-sum of the OTHER FOUR when he plays (same for defense).  Positive = good replacements.
    A rating that says "he is good because his replacement is bad" correlates negatively with this;
    the additive model's coefficient should not, and `corr` at the bottom of the frame is the test.
    """
    m = wd_h.spec.n_ps
    ids = wd_h.spec.ps_table["player_id"].to_numpy()
    to, td = _team_of_rows(p, wd_h, main_team)
    o, d, poss = p.rat.o.to_numpy(), p.rat.d.to_numpy(), p.rat.poss.to_numpy()
    Zo, Zd = wd_h.X[:, :m].tocsr(), wd_h.X[:, m:2 * m].tocsr()
    rows = []
    for j in np.flatnonzero(poss >= min_poss):
        t = main_team.get(int(ids[j]), -1)
        if t < 0:
            continue
        for side, Z, team_rows, v in (("o", Zo, to == t, o), ("d", Zd, td == t, d)):
            on = np.asarray(Z[:, j].todense()).ravel() > 0
            sel_on, sel_off = team_rows & on, team_rows & ~on
            if sel_on.sum() < 20 or sel_off.sum() < 20:
                continue
            lineup = Z @ v
            with_him = np.average(lineup[sel_on] - v[j], weights=p.w[sel_on])
            without = np.average(lineup[sel_off], weights=p.w[sel_off])
            rows.append(dict(player_id=int(ids[j]), team_id=int(t), side=side, rating=float(v[j]),
                             replacement=float(without - with_him), poss_on=float(p.poss[sel_on].sum())))
    out = pd.DataFrame(rows)
    if len(out):
        out.attrs["corr"] = {s: float(g.rating.corr(g.replacement)) for s, g in out.groupby("side")}
    return out
