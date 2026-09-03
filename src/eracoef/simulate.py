"""Simulated seasons with known beta, u, per-game box counts and stints.

The outcome model mirrors the leakage story: made shots enter a stint's points
mechanically (3 / 2 / 1 per make, realized counts), while every other feature and
the "net" value of a make enter through the player's true rate.  So the true
coefficient on 3PM is beta_3 (< 3) but the realized 3PM count of the game is
correlated with the stint residual with slope 3 -- exactly what full-season rates
leak and leave-one-game-out rates do not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .design import FEATURES, HOME_SLOTS, AWAY_SLOTS

LEAGUE_MEAN = np.array([2.4, 4.2, 5.6, 5.2, 3.6, 1.0, 2.0, 6.6, 4.8, 2.8, 1.6, 1.0, 4.0])
BETA_O = np.array([1.6, -0.7, 0.9, -0.6, 0.5, -0.4, 0.5, 0.1, 0.4, -1.0, 0.1, 0.0, -0.1])
BETA_D = np.array([-0.1, 0.1, -0.1, 0.2, 0.0, 0.1, 0.0, -0.3, -0.1, 0.1, -0.8, -0.6, 0.3])
# Same-game mechanical slopes: realized counts enter the stint's points with these values
# (a make scores; a miss forfeits the possession's ~1 point unless an ORB follows; an ORB
# restores it; a turnover forfeits it).  The player's TRUE rate enters with beta, so the
# true coefficient is beta while the game's box count leaks with slope m.
MECH = {"fg3m": 3.0, "fg2m": 2.0, "ftm": 1.0, "fg3_miss": -0.75, "fg2_miss": -0.75, "orb": 0.75, "tov": -1.0}
GAMMA = dict(home=1.5, intercept=60.0, is_gt=-2.0, margin=-0.15, margin_frem=-0.10, po_int=-1.0, po_home=0.5)


def _sample_lineup(rng, weights):
    """Five distinct players, probability ~ weights (Gumbel top-k)."""
    g = np.log(weights) + rng.gumbel(size=len(weights))
    return np.sort(np.argpartition(-g, 5)[:5])


def simulate(n_seasons=2, n_teams=10, players_per_team=12, games_per_season=150, stints_per_game=(30, 50),
             beta_O=BETA_O, beta_D=BETA_D, tau_u=(1.5, 1.5), eps_var=0.5, rate_cv=0.5, seed=0, leak=True,
             po_series_per_season=0, delta_O=None, delta_D=None, first_season=2001):
    """Return dict(stints=DataFrame, box=DataFrame, truth=dict)."""
    rng = np.random.default_rng(seed)
    feats = FEATURES
    nf = len(feats)
    beta_O = np.asarray(beta_O, dtype=float)
    beta_D = np.asarray(beta_D, dtype=float)
    delta_O = np.zeros(nf) if delta_O is None else np.asarray(delta_O, dtype=float)
    delta_D = np.zeros(nf) if delta_D is None else np.asarray(delta_D, dtype=float)
    mech = np.array([MECH.get(f, 0.0) for f in feats])
    is_mech = mech != 0

    stint_rows, box_rows, truth_rows = [], [], []
    game_counter = 0
    for s in range(n_seasons):
        season = first_season + s
        n_pl = n_teams * players_per_team
        team_of = np.repeat(np.arange(n_teams), players_per_team)
        pid = 1000 * (team_of + 1) + np.tile(np.arange(players_per_team), n_teams) + 1
        # true rates: gamma with mean LEAGUE_MEAN and cv rate_cv
        shape = 1.0 / rate_cv ** 2
        mu = rng.gamma(shape, LEAGUE_MEAN / shape, size=(n_pl, nf))
        eta_O = rng.normal(0, tau_u[0], n_pl)
        eta_D = rng.normal(0, tau_u[1], n_pl)
        imp_O = mu @ beta_O + eta_O
        imp_D = mu @ beta_D + eta_D
        prop = rng.gamma(2.0, 1.0, n_pl) * np.where(np.tile(np.arange(players_per_team), n_teams) < 5, 3.0, 1.0)
        truth_rows.append(pd.DataFrame({"player_id": pid, "season": season, "u_O": eta_O, "u_D": eta_D,
                                        "impact_O": imp_O, "impact_D": imp_D, **{f"rate_{f}": mu[:, j] for j, f in enumerate(feats)}}))
        po_imp_O = imp_O + mu @ delta_O
        po_imp_D = imp_D + mu @ delta_D

        # schedule
        games = []
        for g in range(games_per_season):
            h, a = rng.choice(n_teams, 2, replace=False)
            games.append((h, a, "RS", ""))
        for k in range(po_series_per_season):
            h, a = rng.choice(n_teams, 2, replace=False)
            n_g = int(rng.integers(4, 8))
            for gi in range(n_g):
                hh, aa = (h, a) if gi in (0, 1, 4, 6) else (a, h)
                games.append((hh, aa, "PO", f"{season}-{min(h, a)}-{max(h, a)}"))

        for (h, a, phase, series) in games:
            game_counter += 1
            gid = f"{season}{game_counter:05d}"
            n_st = int(rng.integers(stints_per_game[0], stints_per_game[1] + 1))
            score_h = score_a = 0.0
            gbox = np.zeros((n_pl, nf))
            gposs = np.zeros(n_pl)
            hp = np.flatnonzero(team_of == h); ap = np.flatnonzero(team_of == a)
            for t in range(n_st):
                frac_rem = 1.0 - t / n_st
                margin_h = score_h - score_a
                is_gt = (frac_rem < 0.125 and abs(margin_h) >= 15) or (frac_rem < 0.25 and abs(margin_h) >= 20)
                lh = hp[_sample_lineup(rng, prop[hp])]
                la = ap[_sample_lineup(rng, prop[ap])]
                poss_h = 1 + rng.poisson(6)
                poss_a = max(1, poss_h + int(rng.integers(-1, 2)))
                pts = {}
                for off, de, poss, home, margin in ((lh, la, poss_h, 1.0, margin_h), (la, lh, poss_a, -1.0, -margin_h)):
                    iO, iD = (po_imp_O, po_imp_D) if phase == "PO" else (imp_O, imp_D)
                    e100 = iO[off].sum() + iD[de].sum() + GAMMA["intercept"] + GAMMA["home"] * home \
                        + GAMMA["is_gt"] * is_gt + GAMMA["margin"] * np.clip(margin, -25, 25) \
                        + GAMMA["margin_frem"] * np.clip(margin, -25, 25) * frac_rem
                    if phase == "PO":
                        e100 += GAMMA["po_int"] + GAMMA["po_home"] * home
                    e_pts = e100 * poss / 100.0
                    p = e_pts + rng.normal(0, np.sqrt(eps_var * poss))
                    if leak:
                        cnt = rng.poisson(mu[off][:, is_mech] * poss / 100.0)          # 5 x n_mech
                        p += (cnt * mech[is_mech]).sum() - (mu[off][:, is_mech] * mech[is_mech]).sum() * poss / 100.0
                        gbox[off[:, None], np.flatnonzero(is_mech)[None, :]] += cnt
                    gposs[off] += poss
                    pts[home] = p
                stint_rows.append(dict(game_id=gid, season=season, phase=phase, series_id=series, game_date=game_counter,
                                       period=1 + min(int(4 * (1 - frac_rem)), 3),
                                       **{c: int(pid[i]) for c, i in zip(HOME_SLOTS, lh)},
                                       **{c: int(pid[i]) for c, i in zip(AWAY_SLOTS, la)},
                                       poss_h=poss_h, poss_a=poss_a, pts_h=pts[1.0], pts_a=pts[-1.0],
                                       margin_h=margin_h, frac_rem=frac_rem, is_gt=bool(is_gt), neutral=False))
                score_h += pts[1.0]; score_a += pts[-1.0]
            # non-mechanical (or all, if no leak) per-game counts
            played = np.flatnonzero(gposs > 0)
            other = ~is_mech if leak else np.ones(nf, bool)
            gbox[played[:, None], np.flatnonzero(other)[None, :]] = rng.poisson(mu[played][:, other] * gposs[played, None] / 100.0)
            for i in played:
                box_rows.append(dict(game_id=gid, player_id=int(pid[i]), season=season, phase=phase,
                                     **{f: float(gbox[i, j]) for j, f in enumerate(feats)}))
    stints = pd.DataFrame(stint_rows)
    box = pd.DataFrame(box_rows)
    truth = dict(ps=pd.concat(truth_rows, ignore_index=True), beta_O=beta_O, beta_D=beta_D,
                 delta_O=delta_O, delta_D=delta_D, gamma=GAMMA, features=feats)
    return dict(stints=stints, box=box, truth=truth)


def truth_aligned(truth: dict, spec) -> pd.DataFrame:
    """Truth table aligned to spec.ps_table (one row per Z column of the O block)."""
    t = spec.ps_table.merge(truth["ps"], on=["player_id", "season"], how="left")
    return t
