"""Core tests: BoxExposure (crossfit / full / loo), MixedModelRAPM(CV), routing, REML, calibration, leakage."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold

from eracoef.checks import beta_table, bootstrap_beta, calibration_slopes, u_recovery
from eracoef.cv import crossfit_beta, make_pipeline, plugin_fit, weighted_mse_scorer
from eracoef.design import FEATURES, build_design
from eracoef.estimator import MixedModelRAPM, MixedModelRAPMCV
from eracoef.exposure import BoxExposure, split_half_k
from eracoef.simulate import MECH, simulate

CFG = {"gt_weight": 1.0, "margin_clip": 25, "low_poss_threshold": 500}


@pytest.fixture(scope="module")
def small():
    sim = simulate(n_seasons=2, n_teams=6, players_per_team=10, games_per_season=40, stints_per_game=(20, 30), seed=1)
    wd = build_design(sim["stints"], sim["box"], FEATURES, CFG)
    return sim, wd


@pytest.fixture(scope="module")
def medium():
    sim = simulate(n_seasons=2, n_teams=10, players_per_team=12, games_per_season=150, stints_per_game=(30, 50), seed=2)
    wd = build_design(sim["stints"], sim["box"], FEATURES, CFG)
    return sim, wd


@pytest.fixture(scope="module")
def tiny():
    sim = simulate(n_seasons=1, n_teams=4, players_per_team=8, games_per_season=8, stints_per_game=(4, 8), seed=3)
    wd = build_design(sim["stints"], sim["box"], FEATURES, CFG)
    return sim, wd


def _exp(wd, **kw):
    kw.setdefault("pad_k", {f: 50.0 for f in FEATURES})
    kw.setdefault("center", False)
    return BoxExposure(wd.game_box, wd.game_poss, FEATURES, spec=wd.spec, game_half=wd.game_half, **kw)


def _pad(exp, C, P, s):
    return (100 * C + exp.pad_k_[s] * exp.league_rates_[s]) / (P + exp.pad_k_[s])


# ----------------------------------------------------------------------------- exposure
def test_crossfit_covariates_from_other_half_only(small):
    _, wd = small
    gh = wd.game_half
    for half, other in (("A", "B"), ("B", "A")):
        m = wd.half_mask(half)
        exp = _exp(wd, mode="crossfit", half=half).fit(wd.X[m])
        # covariate games are exactly the other half's RS games
        assert set(exp.covariate_games_) == set(np.flatnonzero(gh == other))
        gb = wd.game_box[(wd.game_box.phase == "RS") & (gh[wd.game_box.game_idx.to_numpy()] == other)]
        tot = np.zeros_like(exp.totals_)
        np.add.at(tot, gb.ps_idx.to_numpy(), gb[FEATURES].to_numpy())
        np.testing.assert_allclose(exp.totals_, tot)
        # every row of a player-season gets the same number: Xbox_O == sum of rates_[ps]
        Xt = exp.transform(wd.X[m])
        nf = len(FEATURES)
        box_o = Xt[:, wd.spec.box_o(nf)].toarray()
        Zo = wd.X[m][:, wd.spec.zo].tocsr()
        for r in (0, 5, 50):
            ps = Zo.indices[Zo.indptr[r]:Zo.indptr[r + 1]]
            np.testing.assert_allclose(box_o[r], exp.rates_[ps].sum(0), rtol=1e-12)
    # mixing halves in one fit is refused
    with pytest.raises(ValueError):
        _exp(wd, mode="crossfit", half="A").fit(wd.X)


def test_loo_rate_excludes_own_game(small):
    _, wd = small
    exp = _exp(wd, mode="loo")
    Xt = exp.fit_transform(wd.X)
    nf = len(FEATURES)
    box_o = Xt[:, wd.spec.box_o(nf)].toarray()
    Zo = wd.X[:, wd.spec.zo].tocsr()
    game = wd.rows["game_idx"].to_numpy()
    for r in [0, 7, 123, len(wd.y) - 1]:
        po = Zo.indices[Zo.indptr[r]:Zo.indptr[r + 1]]
        manual = 0
        for p in po:
            C = exp.totals_[p].copy(); P = exp.poss_off_[p]
            gb = wd.game_box[(wd.game_box.game_idx == game[r]) & (wd.game_box.ps_idx == p) & (wd.game_box.phase == "RS")]
            gp = wd.game_poss[(wd.game_poss.game_idx == game[r]) & (wd.game_poss.ps_idx == p)]
            if len(gb):
                C -= gb[FEATURES].to_numpy()[0]
            if len(gp):
                P -= gp["poss_off"].to_numpy()[0]
            manual = manual + _pad(exp, C, P, wd.spec.season_of_ps[p])
        np.testing.assert_allclose(box_o[r], manual, rtol=1e-10)


def test_full_mode_test_rows_use_training_totals_only(small):
    _, wd = small
    game = wd.rows["game_idx"].to_numpy()
    held = np.unique(game)[:5]
    tr = ~np.isin(game, held); te = ~tr
    exp = _exp(wd, mode="full").fit(wd.X[tr])
    gb = wd.game_box[(wd.game_box.phase == "RS") & ~wd.game_box.game_idx.isin(held)]
    tot = np.zeros_like(exp.totals_)
    np.add.at(tot, gb.ps_idx.to_numpy(), gb[FEATURES].to_numpy())
    np.testing.assert_allclose(exp.totals_, tot)
    assert not np.isin(held, exp.covariate_games_).any()
    Xt = exp.transform(wd.X[te])
    nf = len(FEATURES)
    box_o = Xt[:, wd.spec.box_o(nf)].toarray()
    Zo = wd.X[te][:, wd.spec.zo].tocsr()
    r = 3
    po = Zo.indices[Zo.indptr[r]:Zo.indptr[r + 1]]
    np.testing.assert_allclose(box_o[r], exp.rates_[po].sum(0), rtol=1e-10)


def test_padding_and_centering_formulas(small):
    _, wd = small
    k = {f: 10.0 * (j + 1) for j, f in enumerate(FEATURES)}
    exp = _exp(wd, mode="full", pad_k=k, center=True)
    exp.fit(wd.X, sample_weight=wd.w)
    ps = 5
    s = wd.spec.season_of_ps[ps]
    kk = np.array([k[f] for f in FEATURES])
    expect = (100 * exp.totals_[ps] + kk * exp.league_rates_[s]) / (exp.poss_off_[ps] + kk)
    np.testing.assert_allclose(exp.rates_[ps], expect)
    np.testing.assert_allclose(exp.season_rates_[ps], expect)   # full mode: same thing
    Xt = exp.transform(wd.X)
    B = Xt[:, wd.spec.n_base:].toarray()
    np.testing.assert_allclose((wd.w[:, None] * B).sum(0) / wd.w.sum(), 0.0, atol=1e-9)
    exp2 = _exp(wd, mode="full", pad_k=k, pad_scale=2.0).fit(wd.X)
    np.testing.assert_allclose(exp2.pad_k_, 2 * exp.pad_k_)
    # crossfit: season_rates_ (reporting) use both halves, rates_ (covariates) only the other half
    m = wd.half_mask("A")
    exp3 = _exp(wd, mode="crossfit", half="A", pad_k=k).fit(wd.X[m])
    assert exp3.season_poss_off_.sum() > exp3.poss_off_.sum()


def test_split_half_k_recovers_truth():
    rng = np.random.default_rng(0)
    n_pl, n_games = 400, 60
    mu = rng.gamma(4.0, 2.0 / 4.0, size=(n_pl, 2))
    mu[:, 1] = rng.gamma(4.0, 6.0 / 4.0, size=n_pl)
    poss = rng.integers(20, 60, size=(n_pl, n_games)).astype(float)
    counts = rng.poisson(mu[:, None, :] * poss[:, :, None] / 100.0).astype(float)
    ps = np.repeat(np.arange(n_pl), n_games)
    g = np.tile(np.arange(n_games), n_pl)
    k, s2, t2, _ = split_half_k(ps, g, counts.reshape(-1, 2), poss.ravel(), np.zeros(n_pl, int), 1)
    k_true = 100 * mu.mean(0) / mu.var(0)
    np.testing.assert_allclose(k[0], k_true, rtol=0.2)


# ----------------------------------------------------------------------------- estimator
def test_estimator_recovers_beta_and_u(medium):
    sim, wd = medium
    cf = crossfit_beta(wd, lam=8000.0, lam_ratio=2.0)
    tb = beta_table(cf, sim["truth"])
    assert np.abs(tb["z"]).max() < 4.0
    assert np.abs(tb["z"]).mean() < 1.5
    pipe = plugin_fit(wd, cf.beta_, lam=8000.0, lam_ratio=2.0)
    mm = pipe["mm"]
    np.testing.assert_allclose(mm.beta_, cf.beta_)
    assert np.isnan(mm.beta_se_).all()
    ur = u_recovery(mm, wd.spec, sim["truth"])
    assert (ur["corr_u_eta"] > 0.3).all()
    # joint (full mode) fit also recovers
    jm = make_pipeline(wd, lam=8000.0, lam_ratio=2.0, mode="full").fit(wd.X, wd.y, sample_weight=wd.w)["mm"]
    assert np.abs(beta_table(jm, sim["truth"])["z"]).max() < 4.0
    assert "is_po" in jm.dropped_fixed_ and "po_home" in jm.dropped_fixed_


def test_leakage_bias_ordering():
    """At a CV-like lambda: full above truth on makes, loo below, crossfit unbiased (12 reps, small sim)."""
    lam = 8000.0
    feats = [FEATURES.index(f) for f in ("fg3m", "fg2m", "fg2_miss")]
    est = {m: [] for m in ("full", "loo", "crossfit")}
    z_cf = []
    for r in range(12):
        sim = simulate(seed=500 + r)
        wd = build_design(sim["stints"], sim["box"], FEATURES, CFG)
        for mode in est:
            if mode == "crossfit":
                e = crossfit_beta(wd, lam=lam)
                z_cf.append(((e.beta_ - np.r_[sim["truth"]["beta_O"], sim["truth"]["beta_D"]]) / e.beta_se_)[feats])
            else:
                e = make_pipeline(wd, lam=lam, mode=mode).fit(wd.X, wd.y, sample_weight=wd.w)["mm"]
            est[mode].append(e.beta_[feats])
    mean = {m: np.mean(v, 0) for m, v in est.items()}
    true = sim["truth"]["beta_O"][feats]
    # makes: full > crossfit > loo; miss (m < 0): the signs flip
    for i, f in enumerate(("fg3m", "fg2m", "fg2_miss")):
        sgn = np.sign(MECH[f])
        assert sgn * mean["full"][i] > sgn * mean["loo"][i], (f, mean)
    assert np.abs(np.mean(z_cf, 0)).max() < 0.7, np.mean(z_cf, 0)


def test_predict_matches_manual(medium):
    _, wd = medium
    tr, te = next(GroupKFold(4).split(wd.X, wd.y, wd.groups))
    pipe = make_pipeline(wd, lam=5000.0, lam_ratio=1.5, mode="full")
    pipe.fit(wd.X[tr], wd.y[tr], sample_weight=wd.w[tr])
    Xt = pipe["exposure"].transform(wd.X[te])
    mm = pipe["mm"]
    spec = wd.spec
    Xf = sp.hstack([Xt[:, spec.n_base:], Xt[:, spec.f]]).toarray()
    manual = Xf @ mm.theta_f_ + Xt[:, spec.z] @ mm.u_
    np.testing.assert_allclose(pipe.predict(wd.X[te]), manual, rtol=1e-12)
    seen = np.flatnonzero(np.asarray(abs(wd.X[tr][:, spec.z]).sum(0)).ravel() > 0)
    unseen = np.setdiff1d(np.arange(2 * spec.n_ps), seen)
    assert np.all(mm.u_[unseen] == 0)
    # plug-in: prediction = X_box beta_fixed + F gamma + Z u
    pp = plugin_fit(wd, mm.beta_, lam=5000.0, lam_ratio=1.5, mask=np.isin(np.arange(len(wd.y)), tr))
    Xt2 = pp["exposure"].transform(wd.X[te])
    nf = mm.n_feat_
    man2 = Xt2[:, spec.n_base:] @ mm.beta_ + Xt2[:, spec.f] @ pp["mm"].gamma_ + Xt2[:, spec.z] @ pp["mm"].u_
    np.testing.assert_allclose(pp.predict(wd.X[te]), man2, rtol=1e-12)


def test_lam_delta_inf_equals_no_delta():
    sim = simulate(n_seasons=1, n_teams=6, players_per_team=9, games_per_season=30, stints_per_game=(10, 20),
                   po_series_per_season=2, seed=4)
    wd = build_design(sim["stints"], sim["box"], FEATURES, CFG)
    assert (~wd.rs_mask()).sum() > 0
    a = make_pipeline(wd, lam=3000.0, lam_delta=None, mode="full").fit(wd.X, wd.y, sample_weight=wd.w)["mm"]
    b = make_pipeline(wd, lam=3000.0, lam_delta=1e12, mode="full").fit(wd.X, wd.y, sample_weight=wd.w)["mm"]
    assert len(b.delta_) == 2 * len(FEATURES)
    np.testing.assert_allclose(b.beta_, a.beta_, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(b.u_, a.u_, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(b.delta_, 0.0, atol=1e-6)
    # crossfit with playoff rows included in both halves
    cf = crossfit_beta(wd, lam=3000.0, lam_delta=1000.0, include_po=True)
    assert len(cf.delta_) == 2 * len(FEATURES) and np.isfinite(cf.delta_se_).all()


def test_clone_and_params_roundtrip(small):
    _, wd = small
    pipe = make_pipeline(wd, lams=[100.0, 1000.0], lam_ratio=1.3, lam_buckets={"low_poss": 2.0}, pad_scale=1.5)
    c = clone(pipe)
    assert c.get_params()["mm__lam_ratio"] == 1.3
    assert c.get_params()["exposure__pad_scale"] == 1.5
    assert c.get_params()["exposure__mode"] == "crossfit"
    c.set_params(mm__lam_ratio=0.7, exposure__pad_scale=0.5, exposure__half="B")
    assert c["mm"].lam_ratio == 0.7 and c["exposure"].pad_scale == 0.5 and c["exposure"].half == "B"
    est = MixedModelRAPM(lam=3.0, lam_ratio=2.0, lam_buckets={"a": 1.0}, lam_delta=4.0, beta_fixed=None, spec=None)
    assert clone(est).get_params() == est.get_params()


def test_gridsearch_pipeline_end_to_end(small):
    _, wd = small
    m = wd.half_mask("A")
    sub = wd.subset(m)
    pipe = make_pipeline(wd, lams=[300.0, 3000.0], cv=3, mode="crossfit", half="A")
    gs = GridSearchCV(pipe, {"exposure__pad_scale": [0.5, 2.0], "mm__lam_ratio": [0.5, 2.0]}, cv=GroupKFold(3),
                      scoring=weighted_mse_scorer(), n_jobs=1)
    gs.fit(sub.X, sub.y, sample_weight=sub.w, groups=sub.groups)
    assert len(gs.cv_results_["mean_test_score"]) == 4
    assert np.isfinite(gs.cv_results_["mean_test_score"]).all()
    assert hasattr(gs.best_estimator_["mm"], "lam_")
    for tr, te in GroupKFold(3).split(sub.X, sub.y, sub.groups):
        assert not np.intersect1d(sub.groups[tr], sub.groups[te]).size


def test_cv_matches_fixed_lambda(small):
    _, wd = small
    lam = 2000.0
    kw = dict(lam_ratio=1.7, lam_buckets={"low_poss": 1.5}, mode="full")
    fixed = make_pipeline(wd, lam=lam, **kw).fit(wd.X, wd.y, sample_weight=wd.w)["mm"]
    cvm = make_pipeline(wd, lams=[lam], cv=3, **kw).fit(wd.X, wd.y, sample_weight=wd.w, groups=wd.groups)["mm"]
    assert cvm.lam_ == lam
    np.testing.assert_allclose(cvm.beta_, fixed.beta_, rtol=1e-9)
    np.testing.assert_allclose(cvm.u_, fixed.u_, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(cvm.cov_beta_, fixed.cov_beta_, rtol=1e-8)
    np.testing.assert_allclose(cvm.sigma2_, fixed.sigma2_, rtol=1e-10)
    np.testing.assert_allclose(cvm.edf_, fixed.edf_, rtol=1e-9)
    cvm.set_lam(3 * lam)
    fixed3 = make_pipeline(wd, lam=3 * lam, **kw).fit(wd.X, wd.y, sample_weight=wd.w)["mm"]
    np.testing.assert_allclose(cvm.beta_, fixed3.beta_, rtol=1e-9)


def test_reml_profile_matches_dense(tiny):
    _, wd = tiny
    spec = wd.spec
    ratio = 1.6
    pipe = make_pipeline(wd, lams=[500.0, 2000.0], cv=2, lam_ratio=ratio, mode="full").fit(wd.X, wd.y, sample_weight=wd.w, groups=wd.groups)
    mm = pipe["mm"]
    Xt = pipe["exposure"].transform(wd.X)
    Xf = sp.hstack([Xt[:, spec.n_base:], Xt[:, spec.f]]).toarray()[:, mm.moments_.active]
    Z = Xt[:, spec.z].toarray() * mm._scale()[None, :]
    w = wd.w; y = wd.y
    n, p = Xf.shape
    lams = np.array([300.0, 1000.0, 5000.0])
    direct = []
    for lam in lams:
        V = np.diag(1 / w) + Z @ Z.T / lam
        Vi = np.linalg.inv(V)
        S = Xf.T @ Vi @ Xf
        th = np.linalg.solve(S, Xf.T @ Vi @ y)
        r = y - Xf @ th
        Q = r @ Vi @ r
        direct.append(np.linalg.slogdet(V)[1] + np.linalg.slogdet(S)[1] + (n - p) * np.log(Q / (n - p)))
    np.testing.assert_allclose(mm.reml_profile(lams), direct, rtol=1e-8)


def test_lambda_selection_rule(small):
    _, wd = small
    lams = np.geomspace(300, 30000, 6)
    mm = make_pipeline(wd, lams=lams, cv=3, mode="full").fit(wd.X, wd.y, sample_weight=wd.w, groups=wd.groups)["mm"]
    res = mm.cv_results_
    assert res.loc[res.lam == mm.lam_cv_, "in_1se_band"].all()
    if res.loc[res.lam == mm.lam_reml_, "in_1se_band"].iloc[0]:
        assert mm.lam_ == mm.lam_reml_ and mm.lam_selected_by_ == "reml"
    else:
        assert mm.lam_ == mm.lam_cv_ and mm.lam_selected_by_ == "cv"
    assert "lambda" in mm.lambda_report()


def test_check_estimator_subset():
    from sklearn.utils.estimator_checks import check_estimator
    res = check_estimator(MixedModelRAPM(lam=1.0, spec=None), on_fail=None)
    failed = [r["check_name"] for r in res if r["status"] == "failed"]
    core = {"check_estimators_dtypes", "check_fit_score_takes_y", "check_estimators_fit_returns_self",
            "check_estimators_pickle", "check_regressors_train", "check_regressor_data_not_an_array",
            "check_estimators_nan_inf", "check_fit2d_predict1d", "check_regressors_int"}
    assert not (core & set(failed)), failed


def test_calibration_slope_vs_lambda():
    tau = 1.5
    eps_var = 1.0
    sim = simulate(n_seasons=2, n_teams=16, players_per_team=12, games_per_season=320, stints_per_game=(30, 50),
                   tau_u=(tau, tau), eps_var=eps_var, leak=False, seed=5)
    wd = build_design(sim["stints"], sim["box"], FEATURES, CFG)
    lam_true = 1e4 * eps_var / tau ** 2
    u_slopes, eblup_slopes, prior_slopes = {}, {}, {}
    for f in (0.25, 1.0, 4.0):
        cs = calibration_slopes(wd, lam=f * lam_true, n_folds=4, mode="full")
        u_slopes[f] = cs[(cs.kind == "joint_u") & (cs.side == "O+D")]["slope"].iloc[0]
        eblup_slopes[f] = cs[(cs.kind == "eblup") & (cs.side == "O+D")]["slope"].iloc[0]
        prior_slopes[f] = cs[(cs.kind == "prior") & (cs.side == "O+D")]["slope"].iloc[0]
    assert 0.8 < u_slopes[1.0] < 1.2, u_slopes
    assert u_slopes[4.0] > u_slopes[1.0] > u_slopes[0.25], u_slopes
    assert eblup_slopes[4.0] > eblup_slopes[1.0] > eblup_slopes[0.25], eblup_slopes
    row = cs[(cs.kind == "prior") & (cs.side == "O+D")].iloc[0]
    assert abs(prior_slopes[1.0] - row["expected_slope"]) < 0.15, (prior_slopes, row["expected_slope"])
    # crossfit + plug-in path runs and gives a sane u slope at the true lambda
    cs2 = calibration_slopes(wd, lam=lam_true, n_folds=3, mode="crossfit")
    assert 0.75 < cs2[(cs2.kind == "joint_u") & (cs2.side == "O+D")]["slope"].iloc[0] < 1.25
    # per side and per bucket the prior is calibrated too (fold fixed effects: each fold has its own beta/gamma)
    pr = cs2[cs2.kind == "prior"]
    assert 0.8 < pr[(pr.side == "D") & (pr.bucket == "all")]["slope"].iloc[0] < 1.2, pr
    assert 0.8 < pr[(pr.side == "O") & (pr.bucket == "all")]["slope"].iloc[0] < 1.2, pr
    big = pr[pr.bucket == "[1500,inf)"]
    assert (big["slope"].between(0.8, 1.2)).all(), big


def test_pad_target_and_fixed_padding(small):
    _, wd = small
    a = crossfit_beta(wd, lam=8000.0)
    exp = a.fits["A"]["exposure"]
    # unit game weights: n_eff equals the possession count, so nothing changes
    np.testing.assert_allclose(exp.poss_off_eff_, exp.poss_off_, rtol=1e-10)
    # frozen padding constants reproduce the fit exactly
    frozen = {h: {"fixed_padding": a.fits[h]["exposure"].padding_state()} for h in ("A", "B")}
    c = crossfit_beta(wd, lam=8000.0, exposure_kw_by_half=frozen)
    np.testing.assert_allclose(a.beta_, c.beta_, rtol=1e-12)
    # possession-conditional target: one target per possession bin, different from the league mean
    b = crossfit_beta(wd, lam=8000.0, pad_target="poss_conditional")
    eb = b.fits["A"]["exposure"]
    assert eb.pad_bins_ is not None and eb.pad_target_.shape[1] >= 2
    assert not np.allclose(eb.pad_target_[0, 0], eb.pad_target_[0, -1])
    # a player with no covariate possessions gets the target, not a rate
    p = np.flatnonzero(eb.poss_off_ == 0)
    if len(p):
        np.testing.assert_allclose(eb.rates_[p[0]], eb._target(eb.poss_off_[p[:1]], wd.spec.season_of_ps[p[:1]])[0])


def test_bootstrap_schemes_and_neff(small):
    _, wd = small
    for scheme in ("multinomial", "bayesian"):
        bt = bootstrap_beta(wd, lam=8000.0, n_boot=8, scheme=scheme, verbose=False)
        assert bt["beta"].shape == (8, 26)
        assert np.isfinite(bt["se_boot"]).all()
    # n_eff halves under Exp(1)-style weights with the same totals
    exp = crossfit_beta(wd, lam=8000.0).fits["A"]["exposure"]
    gm = np.full(int(wd.games.game_idx.max()) + 1, 2.0)
    exp2 = crossfit_beta(wd, lam=8000.0, game_mult=gm).fits["A"]["exposure"]
    np.testing.assert_allclose(exp2.poss_off_, 2 * exp.poss_off_, rtol=1e-10)
    np.testing.assert_allclose(exp2.poss_off_eff_, exp.poss_off_, rtol=1e-10)
