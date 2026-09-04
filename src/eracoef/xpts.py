"""Stage 4 of the luck adjustment: expected points from the possession's FIRST attempt.

The target the ratings are fit to is actual points, and actual points are a lot of luck: whether a
contested jumper drops, whether the long rebound bounces to the offense, whether the free throw goes
in.  Stage 3 (`xftm` in the stints, `xpts_ft` in design.TARGETS) replaced the last of these.  This
module replaces the other two.

For every possession the stints record what the offense did FIRST -- the bucket of its first attempt
(rim / mid / three / a free-throw trip) and the free throws that belong to it -- and nothing about
what happened after, because everything after exists only because a rebound went the offense's way.
Conditioning on the first attempt introduces none of the luck being removed: it is upstream of
every rebound.  Everything downstream is marginalised with the LINEUP's fitted four-factor rates:

    xpts(p) = x1(b) + fta1(p) * q + m1(b) * r * V           (0 if the possession reaches no attempt)
    V       = (1 - t2) * x_att / (1 - (1 - t2) * m_att * r)   the value of a continuation

    x1(b)   expected points of a first attempt in bucket b, the league make rate for that bucket
            scaled by the lineup's eFG% (multiplicatively on the make probability, or additively --
            `variant`)
    m1(b)   the probability that first attempt produces a rebound chance
    r       the lineup's offensive-rebound rate (the OREB% RAPM, per chance)
    t2      the chance a continuation dies without another attempt, scaled by the lineup's TOV%
    x_att   expected points per attempt on the continuation: field-goal part scaled by the lineup's
            eFG%, free-throw part by its FT rate
    m_att   the probability an attempt on the continuation produces another rebound chance
    q       free-throw percentage: the SHOOTER's own padded rate for the first attempt's free throws
            (that is what `xftm` carries), the league's for the marginalised ones

Every league constant comes from the stint counters of the same seasons, so the closure is anchored
to the era it is applied to (offensive rebounding fell six points between 1998 and 2024, FINDINGS.md
section 14).  The geometric form is `1 / (1 - m * r)` with the turnover leak in the same place the
identity `reb_cont - cont_dead + att_retained == att - 1` puts it.

Where it lives.  NOT in the stints: it depends on fitted lineup rates, which depend on the window,
which is built from the stints.  It is computed at window-build time by `xpts_design`, with the
expensive part -- the four cross-fitted factor fits -- cached per window in data/xpts/.

What it must never use: `fga`, `fta`, `reb_cont`, `fgm` of the possession itself.  Those are the
realised downstream events, i.e. the luck.  They enter only as league totals for the constants and as
the four-factor targets.  The one exception is `xftm / fta` as the shooter-specific free-throw rate
for the first attempt's trip, which is a rate, not a count.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .factors import CLIP, FACTORS, crossfit_rate, fit_factor, lineup_rate

BUCKETS = {"rim": 2.0, "mid": 2.0, "thr": 3.0}          # first-attempt bucket -> points if made
ATT1 = ["att1_rim", "att1_mid", "att1_thr", "att1_ft"]

# Selected once by scripts/37_factors.py on 2024-26 (REML in band, interior of a widened grid) and
# held fixed everywhere, exactly as lam_plugin is: re-selecting per training block would cost an
# hour a block and would make the target itself a function of the block.
FIXED_LAMBDA = {"efg": (3494.76, 1.50), "tov": (2175.85, 0.75), "oreb": (414.35, 3.00), "ftr": (1354.69, 1.00)}

# Per-lineup expected-attempt multiplier, 1 + (first-attempt miss rate) * (continuation value).  A
# lineup outside this range is a fit artefact, not basketball; the CLIP RATE is the gate (< 0.5%).
MULT_CLIP = (1.05, 1.40)
P_CLIP = (0.02, 0.98)                                   # a make probability, after lineup scaling
CALIB_BAND = (0.995, 1.005)                             # aggregate sum(xpts) / sum(pts), per season


@dataclass
class LeagueConstants:
    season: int
    p_make: dict            # bucket -> league make probability (all attempts in the bucket)
    q: float                # league FT%
    efg: float              # per 100 attempts, the units of the eFG% RAPM
    tov: float              # per 100 possessions
    oreb: float             # per 100 chances
    ftr: float              # FTA per 100 attempts
    x_fg: float             # field-goal points per attempt (league, all attempts)
    x_ft: float             # non-technical free-throw points per attempt
    m_att: float            # rebound chances per attempt
    fg_miss: float          # 1 - FG% over all field-goal attempts
    t2: float               # continuations that die without another attempt
    mult: float             # observed attempts per possession reaching one
    avg_pts_b: float        # sum_b share_b * pts_b over first attempts, for the additive variant


def league_constants(cnt: pd.DataFrame, season_of_row: np.ndarray) -> dict[int, LeagueConstants]:
    """The closure's anchors, per season, from the counters of every row in that season."""
    out = {}
    for s in np.unique(season_of_row):
        c = cnt[season_of_row == s].sum()
        with_att = sum(c[a] for a in ATT1)
        share = {b: c[f"att1_{b}"] / with_att for b in BUCKETS}
        out[int(s)] = LeagueConstants(
            season=int(s),
            p_make={b: c[f"fgm_{b}"] / max(c[f"fga_{b}"], 1.0) for b in BUCKETS},
            q=(c.ftm + c.ftm_tech) / max(c.fta + c.fta_tech, 1.0),
            efg=100.0 * (c.fgm + 0.5 * c.fg3m) / c.fga,
            tov=100.0 * c.tov / c.poss,
            oreb=100.0 * c.reb_cont / c.reb_chance,
            ftr=100.0 * c.fta / c.att,
            x_fg=(2.0 * c.fgm + c.fg3m) / c.att,
            x_ft=c.ftm / c.att,
            m_att=c.reb_chance / c.att,
            fg_miss=1.0 - c.fgm / c.fga,
            t2=c.cont_dead / max(c.reb_cont + c.att_retained, 1.0),
            mult=c.att / with_att,
            avg_pts_b=sum(share[b] * BUCKETS[b] for b in BUCKETS),
        )
    return out


def expected_points(cnt: pd.DataFrame, rates: pd.DataFrame, season_of_row: np.ndarray,
                    variant: str = "mult", consts: dict | None = None):
    """Expected points per row (POINTS, not per 100) and the per-row diagnostics.

    `cnt` is WindowData.counters, `rates` has the four lineup rates per row in the units of
    design.TARGETS (efg per 100 attempts, tov per 100 possessions, oreb per 100 chances, ftr per 100
    attempts), already clipped to factors.CLIP.
    """
    if variant not in ("mult", "add", "league"):
        raise ValueError(f"variant must be 'mult', 'add' or 'league', got {variant!r}")
    consts = consts or league_constants(cnt, season_of_row)
    n = len(cnt)
    # broadcast the season constants to rows
    def per_row(attr):
        m = {s: getattr(k, attr) for s, k in consts.items()}
        return pd.Series(season_of_row).map(m).to_numpy(dtype=float)

    efg_L, tov_L, oreb_L, ftr_L = per_row("efg"), per_row("tov"), per_row("oreb"), per_row("ftr")
    q_L, x_fg, x_ft, m_att_L, fg_miss_L, t2_L, avg_pts_b = (
        per_row("q"), per_row("x_fg"), per_row("x_ft"), per_row("m_att"), per_row("fg_miss"),
        per_row("t2"), per_row("avg_pts_b"))
    efg, tov, oreb, ftr = (rates[f].to_numpy(dtype=float) for f in FACTORS)

    # the lineup's shot-making, applied to a league make probability
    if variant == "league":                                         # diagnostic: NO lineup shot-making
        efg = efg_L
    s_efg = efg / efg_L                                             # multiplicative
    d_efg = (efg - efg_L) / 100.0 / (avg_pts_b / 2.0)               # additive, same eFG shift
    p_clips = 0

    def scale_p(p):
        nonlocal p_clips
        raw = p * s_efg if variant == "mult" else p + d_efg
        p_clips += int(((raw < P_CLIP[0]) | (raw > P_CLIP[1])).sum())
        return np.clip(raw, *P_CLIP)

    # the continuation: one attempt's worth, then the geometric closure
    r = oreb / 100.0
    t2 = np.clip(t2_L * tov / tov_L, 0.0, 0.95)
    x_att = x_fg * s_efg + x_ft * (ftr / ftr_L)
    fg_pct_row = scale_p(1.0 - fg_miss_L)
    m_att = m_att_L * (1.0 - fg_pct_row) / fg_miss_L
    G = (1.0 - t2) / (1.0 - (1.0 - t2) * m_att * r)                 # attempts per continuation
    V = x_att * G                                                   # points per continuation

    # the first attempt, by bucket; the free-throw bucket's points ride on fta1 * q below
    with_att = cnt[ATT1].sum(axis=1).to_numpy(dtype=float)
    x_first = np.zeros(n)
    miss_first = np.zeros(n)                                        # rebound chances from first attempts
    for b, pts in BUCKETS.items():
        p_L = pd.Series(season_of_row).map({s: k.p_make[b] for s, k in consts.items()}).to_numpy(dtype=float)
        p = scale_p(p_L)
        a = cnt[f"att1_{b}"].to_numpy(dtype=float)
        x_first += a * pts * p
        miss_first += a * (1.0 - p)
    miss_first += cnt["att1_ft"].to_numpy(dtype=float) * (1.0 - q_L)

    # the shooter's own free-throw rate for the first attempt's trip; league where he took none
    fta = cnt["fta"].to_numpy(dtype=float)
    q_row = np.where(fta > 0, cnt["xftm"].to_numpy(dtype=float) / np.maximum(fta, 1.0), q_L)
    x_ft1 = cnt["fta1"].to_numpy(dtype=float) * q_row

    # the per-lineup multiplier, clipped by scaling the continuation
    # The per-LINEUP multiplier: the continuation factor r * G applied to the league's first-attempt
    # miss rate, so the clip is a property of the ten players and not of a short stint's realised
    # bucket mix.  Clipping is done on r * G and the row's own mix is then applied to the clipped value.
    has = with_att > 0
    m1_L = pd.Series(season_of_row).map({s: float(miss_first[season_of_row == s].sum()
                                                  / max(with_att[season_of_row == s].sum(), 1.0))
                                         for s in consts}).to_numpy(dtype=float)
    m1_L = np.maximum(m1_L, 1e-6)                       # no first attempts anywhere: nothing to clip
    rG = r * G
    lo, hi = (MULT_CLIP[0] - 1.0) / m1_L, (MULT_CLIP[1] - 1.0) / m1_L
    rG_c = np.clip(rG, lo, hi)
    mult_clips = int(((rG != rG_c) & has).sum())
    mult_c = np.where(has, 1.0 + (miss_first / np.maximum(with_att, 1.0)) * rG_c, 1.0)
    cont = miss_first * rG_c                            # expected continuation attempts
    xp = x_first + x_ft1 + cont * x_att + cnt["xftm_tech"].to_numpy(dtype=float)

    diag = pd.DataFrame({"xpts": xp, "mult": mult_c, "with_att": with_att,
                         "x_first": x_first, "x_ft1": x_ft1, "x_cont": cont * x_att})
    return xp, diag, dict(p_clips=p_clips, p_clip_rate=p_clips / (4.0 * n),
                          mult_clip_rate=mult_clips / max(int(has.sum()), 1))


def gates(cnt: pd.DataFrame, diag: pd.DataFrame, season_of_row: np.ndarray, consts: dict) -> pd.DataFrame:
    """Per-season calibration of the closure: expected against observed attempts and points."""
    rows = []
    for s, k in consts.items():
        m = season_of_row == s
        c, d = cnt[m], diag[m]
        wa = d.with_att.sum()
        rows.append(dict(season=s, obs_mult=k.mult, exp_mult=float((d.mult * d.with_att).sum() / wa),
                         oreb=k.oreb, t2=k.t2,
                         pts=float(c.pts.sum()), xpts=float(d.xpts.sum()),
                         ratio=float(d.xpts.sum() / c.pts.sum()),
                         share_first=float(d.x_first.sum() / d.xpts.sum()),
                         share_ft1=float(d.x_ft1.sum() / d.xpts.sum()),
                         share_cont=float(d.x_cont.sum() / d.xpts.sum())))
    g = pd.DataFrame(rows)
    g["mult_ok"] = (g.exp_mult / g.obs_mult).between(0.98, 1.02)
    g["calib_ok"] = g.ratio.between(*CALIB_BAND)
    return g


# ------------------------------------------------------------------------------ the lineup rates
def lineup_rates(wd_pts, seasons, cfg, cache_dir: Path | None = None, label: str | None = None,
                 lambdas: dict | None = None, verbose: bool = False) -> pd.DataFrame:
    """The four cross-fitted lineup rates per row of the points design, cached per window.

    Each row's rate comes from the half-fit that did NOT see it (factors.crossfit_rate), with the
    full fit filling any row neither half covers.  The cache is keyed on the window label and checked
    against the row count and the row's stint index, so a stale file cannot be silently reused.
    """
    from .windows import build_window
    lambdas = lambdas or FIXED_LAMBDA
    path = None
    if cache_dir is not None and label is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"rates_{label}.parquet"
        if path.exists():
            r = pd.read_parquet(path)
            same = (len(r) == wd_pts.X.shape[0]
                    and np.array_equal(r["stint"].to_numpy(), wd_pts.rows["stint"].to_numpy())
                    and np.array_equal(r["is_home_off"].to_numpy(), wd_pts.rows["is_home_off"].to_numpy()))
            if same:
                return r[list(FACTORS)]
    rates = {}
    for f in FACTORS:
        wd_f = build_window(list(seasons), cfg, target=f)
        lam, ratio = lambdas[f]
        r, cov = crossfit_rate(wd_pts, wd_f, f, lam, ratio)
        if not np.isfinite(r).all():
            full = fit_factor(wd_f, lam, ratio)
            r = np.where(np.isfinite(r), r, lineup_rate(full, wd_pts, factor=f))
        rates[f] = r
        if verbose:
            w = wd_pts.rows["poss"].to_numpy()
            print(f"    {f:5s} mean {np.average(r, weights=w):6.2f}  coverage {cov:.3f}", flush=True)
    R = pd.DataFrame(rates)
    if path is not None:
        R.assign(stint=wd_pts.rows["stint"].to_numpy(),
                 is_home_off=wd_pts.rows["is_home_off"].to_numpy()).to_parquet(path, index=False)
    return R


def xpts_design(seasons, cfg, variant: str = "mult", calibrate: bool = True, cache: bool = True,
                verbose: bool = False, wd_pts=None):
    """The points design with y replaced by expected points per 100.  Returns (WindowData, report).

    `calibrate` applies one affine map per season -- fitted on these rows, so on a training block it
    is legal -- and only where the aggregate misses CALIB_BAND.  Never per lineup.
    """
    from .windows import build_window, window_label
    if wd_pts is None:
        wd_pts = build_window(list(seasons), cfg)
    if wd_pts.counters is None:
        raise RuntimeError("the stints carry no per-possession counters; rebuild them (STINT_SCHEMA)")
    seasons = sorted(int(s) for s in seasons)
    contiguous = seasons == list(range(seasons[0], seasons[-1] + 1))
    lab = window_label(seasons) if contiguous else "+".join(map(str, seasons))   # a block with a hole
    root = Path(cfg["_root"]) / "data" / "xpts" if cache else None
    rates = lineup_rates(wd_pts, seasons, cfg, cache_dir=root, label=lab, verbose=verbose)
    cnt, season = wd_pts.counters, wd_pts.rows["season"].to_numpy()
    consts = league_constants(cnt, season)
    xp, diag, clips = expected_points(cnt, rates, season, variant=variant, consts=consts)
    g = gates(cnt, diag, season, consts)
    poss = wd_pts.rows["poss"].to_numpy(dtype=float)
    y = 100.0 * xp / poss
    y_pts = wd_pts.y
    w = wd_pts.w
    calib = []
    if calibrate:
        for s in consts:
            m = season == s
            row = g[g.season == s].iloc[0]
            if row.calib_ok:
                calib.append(dict(season=s, applied=False, a=0.0, b=1.0))
                continue
            A = np.column_stack([np.ones(m.sum()), y[m]])
            ab = np.linalg.solve((A * w[m, None]).T @ A, (A * w[m, None]).T @ y_pts[m])
            y[m] = A @ ab
            calib.append(dict(season=s, applied=True, a=float(ab[0]), b=float(ab[1])))
    report = dict(window=lab, variant=variant, gates=g, clips=clips, calibration=pd.DataFrame(calib),
                  consts=consts)
    return wd_pts.with_target(y), report
