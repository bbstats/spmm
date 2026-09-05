"""The out-of-season runner (src/eracoef/holdout.py) against simulated seasons with a known truth.

The simulator is run with persistent talent (rho) and roster turnover, so a rating fit on the
seasons around H genuinely carries information about H and players appear with new teammates --
the two things the criterion is built to see.
"""
import numpy as np
import pandas as pd
import pytest

from eracoef.design import FEATURES, build_design
from eracoef.holdout import (RESULT_COLUMNS, SPLITS, Context, Holdout, PluginSystem, Ratings, SplitSystem,
                             TableSystem, beta_none, fit_rank_map, paired, pooled, pooled_rank, predict_season,
                             rank_calibration, ratings_from_fit, replacement_quality, report, score, team_residual)
from eracoef.simulate import simulate

CFG = {"gt_weight": 1.0, "margin_clip": 25, "low_poss_threshold": 500, "features": FEATURES,
       "first_season": 2001, "last_season": 2005, "windows": [[2001, 2003], [2004, 2006]],
       "lam_plugin": 4000.0, "lam_ratio_plugin": 1.0, "pad_target": "league",
       "holdout": {"first": 2002, "last": 2004, "ks": [2], "rank_bins": 5, "rank_min_poss": 1,   # a TableSystem's poss is seasons
                   "level": "full",           # the simulator has a rubber-band term; the full fixed block absorbs it
                   "exposure_edges": [0, 1, 300, 1000, 1e9]}}


@pytest.fixture(scope="module")
def world():
    # quieter than the default simulator (eps_var 0.5, the mechanical leak on) so the truth-recovery
    # assertions below have standard errors of a few hundredths rather than a tenth
    sim = simulate(n_seasons=5, n_teams=8, players_per_team=10, games_per_season=60, stints_per_game=(20, 30),
                   rho=0.85, turnover=0.3, seed=11, eps_var=0.2, leak=False)
    st, box, truth = sim["stints"], sim["box"], sim["truth"]["ps"]

    def loader(seasons, cfg, target):
        assert target == "pts"
        return build_design(st[st.season.isin(seasons)], box[box.season.isin(seasons)], FEATURES, cfg)

    ctx = Context(cfg=CFG, loader=loader)
    for s, g in truth.groupby("season"):                       # main teams from the truth, not box scores
        ctx._teams[int(s)] = dict(zip(g.player_id.astype(int), g.team.astype(int)))
        ids = g.player_id.astype(int).to_numpy()
        ctx._bigness[int(s)] = {int(q): bool(q % 3 == 0) for q in ids}   # archetype flags: the split only needs a partition
        ctx._bench[int(s)] = {int(q): bool(q % 2 == 0) for q in ids}
    table = truth.rename(columns={"impact_O": "o", "impact_D": "d"})[["player_id", "season", "o", "d"]]
    return sim, ctx, table


@pytest.fixture(scope="module")
def results(world):
    sim, ctx, table = world
    rng = np.random.default_rng(3)
    noisy = table.assign(o=table.o + rng.normal(0, 3.0, len(table)), d=table.d + rng.normal(0, 3.0, len(table)))
    systems = [TableSystem("true", table), TableSystem("noisy", noisy), TableSystem("zero", table.assign(o=0.0, d=0.0)),
               TableSystem("double", table.assign(o=2 * table.o, d=2 * table.d)),
               PluginSystem("rapm", beta=beta_none)]
    ho = Holdout.from_config(CFG)
    res = ho.run(systems, ctx, splits=SPLITS, rank=True, verbose=False)
    return ho, res


def test_neighbourhood_and_labels(world):
    _, ctx, _ = world
    assert ctx.neighbourhood(2003, 2) == [2002, 2004]
    assert ctx.neighbourhood(2001, 2) == [2002, 2003]             # edge: both neighbours on one side
    assert ctx.neighbourhood(2003, 4) == [2001, 2002, 2004, 2005]
    ctx.current_h = 2004
    assert ctx.labels([2003, 2005]) == {"2001-2003", "2004-2006"}
    ctx.current_h = None


def test_results_schema(results):
    _, res = results
    assert list(res.columns) == RESULT_COLUMNS
    allrows = res[res.split == "all"]
    assert set(allrows.held_out) == {2002, 2003, 2004}
    assert allrows.groupby(["held_out", "k", "system", "lam"]).size().max() == 1


def test_true_beats_noisy_beats_zero(results):
    _, res = results
    P = pooled(res[res.split == "all"]).set_index("system")
    assert P.loc["true", "mse"] < P.loc["noisy", "mse"] < P.loc["zero", "mse"]
    assert P.loc["true", "game"] < P.loc["zero", "game"]
    assert P.loc["rapm", "mse"] < P.loc["zero", "mse"]            # the real fit carries information across seasons
    z = paired(res[res.split == "all"], ref="zero").set_index("system")
    assert z.loc["true", "z"] < -2 and z.loc["true", "wins"] == 3
    assert "true" in report(res, ref="zero")


def test_scale_diagnoses_amplitude(results):
    _, res = results
    P = pooled(res[res.split == "all"]).set_index("system")
    assert abs(P.loc["true", "scale_off"] - 1.0) < 0.15
    assert abs(P.loc["double", "scale_off"] - 0.5) < 0.1          # twice-too-wide ratings want half


def test_splits_partition_rows(results):
    _, res = results
    for name in SPLITS:
        g = res[(res.split == name) & (res.system == "true")].groupby("held_out").n.sum()
        a = res[(res.split == "all") & (res.system == "true")].set_index("held_out").n
        assert np.allclose(g.reindex(a.index), a)
    movers = res[(res.split == "movers") & (res.system == "true")]
    assert {"no movers", "1-2 movers", "3+ movers"} & set(movers.group)
    assert res[(res.split == "bigs") & (res.system == "true")].group.nunique() >= 2
    assert res[(res.split == "bench") & (res.system == "true")].group.nunique() >= 2


def test_rank_calibration_sees_amplitude(results):
    ho, _ = results
    pr = pooled_rank(ho.rank_)
    t = pr[(pr.system == "true") & (pr.side == "o") & (pr.decile >= 0)]
    d = pr[(pr.system == "double") & (pr.side == "o") & (pr.decile >= 0)]
    assert abs(t.slope.mean() - 1.0) < 0.2
    assert abs(d.slope.mean() - 0.5) < 0.12
    assert (pr[pr.side == "lineup"].slope > 0).all()
    f = fit_rank_map(ho.rank_, "double", "o", 2, exclude_h=2003)
    x = np.array([-4.0, 0.0, 4.0])
    assert np.all(np.diff(f(x)) > 0)                              # monotone
    assert abs(f(4.0) - 2.0) < 1.0                                # roughly halves


def test_rank_map_corrects_a_doubled_system(results, world):
    """RankMappedSystem with the doubled ratings' own rank table brings them back to the truth's scale,
    with the map for each held-out season fitted on the other seasons."""
    from eracoef.holdout import RankMappedSystem
    ho, res = results
    _, ctx, table = world
    inner = TableSystem("double", table.assign(o=2 * table.o, d=2 * table.d))
    mapped = RankMappedSystem("mapped", inner, ho.rank_)
    res2 = Holdout.from_config(CFG).run([inner, mapped], ctx, verbose=False)
    P = pooled(res2).set_index("system")
    assert P.loc["mapped", "mse"] < P.loc["double", "mse"]
    assert abs(P.loc["mapped", "scale_off"] - 1.0) < abs(P.loc["double", "scale_off"] - 1.0)
    z = paired(res2, ref="double").set_index("system")
    assert z.loc["mapped", "wins"] == 3


def test_diagnostics_run(world):
    sim, ctx, table = world
    ctx.current_h = 2003
    wd_h = ctx.design([2003], "pts")
    rat = TableSystem("true", table).fit([2002, 2004], ctx)
    p = predict_season(rat, wd_h, level="full")
    s = score(p)
    assert s["n"] > 0 and s["mse"] < s["base"]
    tr = team_residual(p, wd_h, ctx.main_team(2003))
    assert set(tr.side) == {"offense", "defense"} and len(tr) == 16
    assert tr.z.abs().mean() < 3.0                                # the true ratings leave no team effect
    rq = replacement_quality(p, wd_h, ctx.main_team(2003), min_poss=1)   # a TableSystem's poss is seasons, not possessions
    assert len(rq) > 0 and "corr" in rq.attrs
    ctx.current_h = None


def test_offset_system_equals_plugin_fit(world):
    """A PluginSystem with a per-player offset is the plug-in fit with that prior_offset, and its rating splits
    into the prior part and the ridge residual exactly; a SplitSystem carries both priors through."""
    from eracoef.cv import plugin_fit
    sim, ctx, table = world
    train = [2002, 2004]
    wd = ctx.design(train, "pts")
    m = wd.spec.n_ps
    rng = np.random.default_rng(5)
    v = rng.normal(0, 1.0, 2 * m)
    sys_off = PluginSystem("off", beta=beta_none, offset=lambda t, c, w: v)
    r = sys_off.fit(train, ctx).df
    pipe = plugin_fit(wd, np.zeros(2 * len(FEATURES)), lam=CFG["lam_plugin"], lam_ratio=CFG["lam_ratio_plugin"],
                      pad_target=CFG["pad_target"], prior_offset=v)
    ref = ratings_from_fit(wd, pipe, np.zeros(2 * len(FEATURES)), offset=v).df
    assert np.allclose(r.o, ref.o, atol=1e-12) and np.allclose(r.d, ref.d, atol=1e-12)
    assert np.allclose(r.prior_o, ref.prior_o) and np.allclose(r.prior_d, ref.prior_d)
    # the simulator's units are player-seasons, pooled to one row per player; before pooling the split is exact
    raw = pd.DataFrame({"o": v[:m] + pipe["mm"].u_[:m], "prior_o": v[:m], "d": v[m:] + pipe["mm"].u_[m:], "prior_d": v[m:]})
    assert np.allclose(raw.o - raw.prior_o, pipe["mm"].u_[:m]) and np.allclose(raw.d - raw.prior_d, pipe["mm"].u_[m:])
    assert abs(r.prior_o.std()) > 0 and abs(r.prior_d.std()) > 0
    plain = PluginSystem("plain", beta=beta_none).fit(train, ctx).df
    assert not np.allclose(plain.o, r.o)                                   # the offset changed the fit
    assert np.allclose(plain.prior_o, 0.0)
    split = SplitSystem("split", offense=sys_off, defense=PluginSystem("plain", beta=beta_none)).fit(train, ctx).df
    assert np.allclose(split.prior_o, r.prior_o) and np.allclose(split.prior_d, 0.0)
    assert np.allclose(split.o, r.o) and np.allclose(split.d, plain.d)


def test_ratings_table_prior_parts(world):
    """windows.player_ratings_table with the chain offset: prior_* carry the offset (defense flipped), and
    rating = prior + u holds with no linear box term."""
    from eracoef.windows import player_ratings_table
    sim, ctx, table = world
    seasons = [2002, 2004]
    wd = ctx.design(seasons, "pts")
    m = wd.spec.n_ps
    v = np.random.default_rng(7).normal(0, 1.0, 2 * m)
    zeros = np.zeros(2 * len(FEATURES))
    t = player_ratings_table(wd, zeros, CFG, seasons, prior_offset=v, prior_parts=(v[:m], v[m:]))
    assert np.allclose(t.prior_off, v[:m]) and np.allclose(t.prior_def, -v[m:])
    plain = player_ratings_table(wd, zeros, CFG, seasons)
    assert np.allclose(plain.prior_off, 0.0) and not np.allclose(plain.u_off, t.u_off)
    assert np.allclose(t.prior_total, t.prior_off + t.prior_def)
