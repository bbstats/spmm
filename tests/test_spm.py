"""The Simple SPM (src/eracoef/spm.py): recovers a known role-and-age function from noisy APM, leaves the
excluded window out, centres its offset, and reads the same at different APM penalties."""
import numpy as np
import pandas as pd

from eracoef.spm import apm_lambda_check, fit_spm, spm_offset, spm_predict


def _truth(share, gs, age):
    return 6.0 * share - 4.0 * share ** 2 + 2.0 * gs - 0.02 * (age - 27.0) ** 2


def _panel(seed=0, n=3000, noise=3.0):
    rng = np.random.default_rng(seed)
    parts = []
    for side in ("O", "D"):
        share = rng.uniform(0.02, 0.9, n)
        gs = rng.uniform(0, 1, n)
        age = rng.uniform(19, 38, n)
        poss = share * 15000
        f = _truth(share, gs, age) * (1.0 if side == "O" else -1.0)
        apm = f + rng.normal(0, noise, n) * np.sqrt(3000.0 / poss)
        parts.append(pd.DataFrame({"window": [f"W{i % 10}" for i in range(n)], "side": side, "player_id": np.arange(n),
                                   "poss": poss, "share": share, "gs_pct": gs, "age": age, "apm": apm, "truth": f}))
    return pd.concat(parts, ignore_index=True)


def test_spm_recovers_the_function_leaving_the_window_out():
    p = _panel()
    fit = fit_spm(p, "O", exclude={"W0"}, pen=1.0, min_poss=500)
    assert fit.n == int(((p.side == "O") & (p.window != "W0") & (p.poss >= 500)).sum())
    held = p[(p.side == "O") & (p.window == "W0")]
    pred = spm_predict(fit, held.share, held.gs_pct, held.age)
    rmse = float(np.sqrt(np.average((pred - held.truth) ** 2, weights=held.poss)))
    assert rmse < 0.6, rmse
    d = fit_spm(p, "D", exclude={"W0"})
    assert d.coef[0] < 0 < fit.coef[0]                     # raw sign on defense: more share = more points allowed? No: better players allow fewer
    t = fit.table()
    assert list(t.columns) == ["side", "input", "coef_std", "coef_raw"] and len(t) == 7


def test_offset_is_centred_and_ordered():
    p = _panel()
    fo, fd = fit_spm(p, "O"), fit_spm(p, "D")
    inputs = pd.DataFrame({"share": [0.8, 0.3, 0.05], "gs_pct": [1.0, 0.2, 0.0], "age": [27.0, 27.0, 27.0]})
    poss = np.array([12000.0, 4500.0, 700.0])
    off = spm_offset(fo, fd, inputs, poss, poss)
    assert off.shape == (6,)
    assert abs(np.average(off[:3], weights=poss)) < 1e-9 and abs(np.average(off[3:], weights=poss)) < 1e-9
    assert off[0] > off[1] > off[2]                        # offense: the starter is best
    assert off[3] < off[4] < off[5]                        # defense raw sign: the starter allows fewest


def test_lambda_check_reads_zero_on_identical_panels():
    p = _panel()
    t = apm_lambda_check({30: p, 100: p, 300: p}, "O")
    assert list(t.columns) == ["lam_30", "lam_100", "lam_300", "max_rel_diff"]
    assert float(t.max_rel_diff.max()) < 1e-9
