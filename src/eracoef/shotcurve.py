"""The league's make probability by shot distance, one curve per season.

Shot validity by distance is not a constant: the mid-range two of 1998 and of 2024 are different
shots, and the corner three moved the three-point curve.  So the expected value of an attempt is
read off a per-season curve, fitted on every field-goal attempt of that regular season -- a
logistic regression on a natural cubic spline of distance, separately for twos and threes -- and
cached as a one-foot lookup table in data/shotcurve/.  A single shot's own contribution to a curve
fitted on ~200,000 is 1e-5 and is not leakage.

What the feed does and does not carry.  Coordinates are present for 70-76% of attempts in 1998-2008
and ~100% from 2013 on, but about 15% of THREES have distance 0 in every era including 2024, and a
TWO at distance 0 is either at the rim or missing.  So every curve carries an explicit UNLOCATED cell
per shot value with its own make rate; a three with distance below 20 feet is unlocated; a two at
distance 0 is at the rim only if the description names an at-rim shot (dunk, layup, tip, putback,
alley-oop, finger roll), else unlocated.  `shot_bin` is the ONE place that rule lives, used by the
fit and by the stint parser alike, so the two can never disagree.

The shooter's own skill enters elsewhere (xshoot.py): this module is the league's view of the shot.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

UNLOCATED = -1
MAX_DIST = 40                                       # feet; longer shots share the last bin
RIM_WORDS = ("Dunk", "Layup", "Tip", "Putback", "Alley", "Finger Roll", "Slam")
KNOTS = {False: (1.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0, 24.0),     # twos
         True: (24.0, 25.0, 26.0, 27.0, 28.0, 30.0, 34.0)}        # threes: the arc is 23.75 ft; below it is empty
RANGE = {False: (0.0, 30.0), True: (24.0, float(MAX_DIST))}      # the spline is fit inside, flat outside
FALLBACK = {False: 0.50, True: 0.36}                            # no curve at all (synthetic games in tests)
RIDGE = 1e-3
# A one-foot bin with many shots keeps its own observed rate, padded toward the spline: the rim
# structure (dunks at 0 ft, layups at 1, contested shots at 2-3) is sharper than any smooth curve
# can follow, and the smooth curve is only needed where the bins are sparse.
K_BIN = 150.0


def shot_bin(dist: float, three: bool, desc: str = "") -> int:
    """The curve's cell for one attempt: an integer foot, or UNLOCATED."""
    if dist is None or not np.isfinite(dist) or dist < 0:
        return UNLOCATED
    if three:
        return UNLOCATED if dist < 20 else int(min(round(dist), MAX_DIST))
    if dist == 0 and not any(w in desc for w in RIM_WORDS):
        return UNLOCATED
    return int(min(round(dist), MAX_DIST))


def shot_rows(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per field-goal attempt of a game: three, made, bin."""
    at = pbp["actionType"].astype(str).str.strip()
    fg = pbp[at.isin(["Made Shot", "Missed Shot", "Heave"])]
    if not len(fg):
        return pd.DataFrame(columns=["three", "made", "bin"])
    three = pd.to_numeric(fg["shotValue"], errors="coerce").fillna(0).astype(int).eq(3).to_numpy()
    dist = (pd.to_numeric(fg["shotDistance"], errors="coerce").fillna(-1.0).to_numpy(dtype=float)
            if "shotDistance" in fg.columns else np.full(len(fg), -1.0))
    desc = fg["description"].astype(str).to_numpy()
    made = (at[fg.index] == "Made Shot").to_numpy()
    b = np.array([shot_bin(d, t, s) for d, t, s in zip(dist, three, desc)], dtype=int)
    return pd.DataFrame({"three": three, "made": made, "bin": b})


def _natural_spline(x: np.ndarray, knots) -> np.ndarray:
    """Natural cubic spline basis (ESL 5.4): [1, x, N_3 .. N_K]."""
    k = np.asarray(knots, dtype=float)
    K = len(k)

    def d(j):
        return (np.clip(x - k[j], 0, None) ** 3 - np.clip(x - k[-1], 0, None) ** 3) / (k[-1] - k[j])

    cols = [np.ones_like(x), x] + [d(j) - d(K - 2) for j in range(K - 2)]
    return np.column_stack(cols)


def _logistic(X, y, w=None, ridge=RIDGE, iters=25):
    """IRLS for a logistic regression with a small ridge on everything but the intercept."""
    n, p = X.shape
    w = np.ones(n) if w is None else w
    beta = np.zeros(p)
    P = ridge * np.eye(p)
    P[0, 0] = 0.0
    for _ in range(iters):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        s = np.maximum(mu * (1 - mu), 1e-6) * w
        z = eta + (y - mu) / np.maximum(mu * (1 - mu), 1e-6)
        new = np.linalg.solve((X * s[:, None]).T @ X + P, (X * s[:, None]).T @ z)
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new
    return beta


@dataclass
class ShotCurve:
    season: int
    table: pd.DataFrame          # three, bin, prob, n, made  (bin UNLOCATED = the unlocated cell)

    def __post_init__(self):
        self._p = {(bool(t), int(b)): float(p) for t, b, p in zip(self.table.three, self.table.bin, self.table.prob)}

    def prob(self, dist: float, three: bool, desc: str = "") -> float:
        """League make probability for one attempt this season."""
        return self._p.get((bool(three), shot_bin(dist, three, desc)), FALLBACK[bool(three)])

    def prob_bin(self, b: int, three: bool) -> float:
        return self._p.get((bool(three), int(b)), FALLBACK[bool(three)])

    @classmethod
    def constant(cls, season: int = 0, p2: float = FALLBACK[False], p3: float = FALLBACK[True]) -> "ShotCurve":
        """A flat curve (tests, and the no-location fallback)."""
        rows = [dict(three=t, bin=b, prob=(p3 if t else p2), n=0, made=0)
                for t in (False, True) for b in range(UNLOCATED, MAX_DIST + 1)]
        return cls(season, pd.DataFrame(rows))


def fit_curve(rows: pd.DataFrame, season: int) -> ShotCurve:
    """Fit the two curves and the two unlocated cells from a season's shot rows."""
    from .pad import shrink
    out = []
    for three in (False, True):
        r = rows[rows.three == three]
        located = r[r.bin != UNLOCATED]
        league = float(r.made.mean()) if len(r) else FALLBACK[three]
        lo, hi = RANGE[three]
        if len(located) >= 500:
            x = np.clip(located.bin.to_numpy(dtype=float), lo, hi)
            X = _natural_spline(x, KNOTS[three])
            beta = _logistic(X, located.made.to_numpy(dtype=float))
            grid = np.clip(np.arange(0, MAX_DIST + 1, dtype=float), lo, hi)
            fitted = 1.0 / (1.0 + np.exp(-(_natural_spline(grid, KNOTS[three]) @ beta)))
        else:
            fitted = np.full(MAX_DIST + 1, league)
        cnt = located.groupby("bin").made.agg(["size", "sum"]).reindex(range(MAX_DIST + 1)).fillna(0)
        n_b, m_b = cnt["size"].to_numpy(dtype=float), cnt["sum"].to_numpy(dtype=float)
        obs = np.where(n_b > 0, m_b / np.where(n_b > 0, n_b, 1.0), 0.0)
        prob = shrink(obs, n_b, K_BIN, fitted)             # dense bins: their own rate; sparse: the spline
        for b in range(MAX_DIST + 1):
            out.append(dict(three=three, bin=b, prob=float(prob[b]), n=int(n_b[b]), made=int(m_b[b])))
        u = r[r.bin == UNLOCATED]
        # the unlocated cell: its own rate, lightly padded toward the shot value's league rate.  Unlocated
        # twos are mostly heaves (2026: 1135 of them made at 2%), so the pad must not drag them upward.
        p_u = float(shrink(u.made.mean() if len(u) else league, len(u), 10.0, league))
        out.append(dict(three=three, bin=UNLOCATED, prob=p_u, n=int(len(u)), made=int(u.made.sum()) if len(u) else 0))
    return ShotCurve(season, pd.DataFrame(out))


def calibration(curve: ShotCurve, rows: pd.DataFrame, min_n: int = 200) -> pd.DataFrame:
    """Observed against fitted make rate per bin (the gate: within 2% where there are shots)."""
    g = rows.groupby(["three", "bin"]).made.agg(["size", "mean"]).reset_index()
    g["fitted"] = [curve.prob_bin(b, t) for t, b in zip(g.three, g.bin)]
    g["gap"] = g["mean"] - g["fitted"]
    return g[g["size"] >= min_n].rename(columns={"size": "n", "mean": "observed"})


def curve_path(season: int, cfg) -> Path:
    return Path(cfg["_root"]) / "data" / "shotcurve" / f"{season}_RS.parquet"


def season_shot_rows(season: int, cfg, phase: str = "RS") -> pd.DataFrame:
    """Every field-goal attempt of a season, from the cached play-by-play."""
    from .ingest import fetch_pbp, game_table, load_gamelog
    games = game_table(load_gamelog(season, phase, cfg))
    parts = []
    for g in games.itertuples(index=False):
        try:
            parts.append(shot_rows(fetch_pbp(g.game_id, cfg)))
        except Exception:  # noqa: BLE001  a game that cannot be read is not a curve's problem
            continue
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["three", "made", "bin"])


def curve_for(season: int, cfg, force: bool = False, verbose: bool = False) -> ShotCurve:
    """The season's curve, fitted on its regular season and cached.  Playoff games use it too."""
    path = curve_path(season, cfg)
    if path.exists() and not force:
        return ShotCurve(season, pd.read_parquet(path))
    rows = season_shot_rows(season, cfg, "RS")
    curve = fit_curve(rows, season)
    path.parent.mkdir(parents=True, exist_ok=True)
    curve.table.to_parquet(path, index=False)
    if verbose:
        share = rows.groupby("three").bin.apply(lambda b: float((b == UNLOCATED).mean()))
        print(f"  shot curve {season}: {len(rows)} attempts, unlocated twos {100 * share.get(False, 0):.1f}% "
              f"threes {100 * share.get(True, 0):.1f}%", flush=True)
    return curve
