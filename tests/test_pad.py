"""The one padding helper (src/eracoef/pad.py)."""
import numpy as np

from eracoef.pad import K_FALLBACK, mom_k, pad_rate, shrink


def test_shrink_limits():
    assert shrink(0.9, 100.0, 0.0, 0.5) == 0.9                     # no padding: the raw rate
    assert shrink(0.9, 0.0, 30.0, 0.5) == 0.5                      # no data: the target
    assert shrink(0.9, 0.0, 0.0, 0.5) == 0.5                       # nothing at all: still the target
    x = shrink(np.array([0.9, 0.3]), np.array([10.0, 1000.0]), 30.0, 0.5)
    assert abs(x[0] - (10 * 0.9 + 30 * 0.5) / 40) < 1e-12
    assert abs(x[1] - 0.3) < 0.01                                  # plenty of data: barely moved


def test_mom_k_recovers_a_beta_binomial():
    """Rates drawn Beta(a, b) have between-unit variance mu(1-mu)/(a+b+1), so the padding constant is ~a+b."""
    rng = np.random.default_rng(0)
    a, b = 30.0, 10.0                                              # mean 0.75, k_true = 40
    p = rng.beta(a, b, 2000)
    att = rng.integers(100, 400, 2000).astype(float)
    made = rng.binomial(att.astype(int), p).astype(float)
    league, k = mom_k(made, att)
    assert abs(league - 0.75) < 0.01
    assert 0.75 * (a + b) < k < 1.25 * (a + b)


def test_mom_k_falls_back_with_too_few_units():
    league, k = mom_k(np.array([7.0, 8.0]), np.array([10.0, 10.0]))
    assert league == 0.75 and k == K_FALLBACK
    league, k = mom_k(np.array([]), np.array([]), fallback_rate=0.6)
    assert league == 0.6 and k == K_FALLBACK


def test_pad_rate_handles_zero_attempts():
    x = pad_rate(np.array([0.0, 50.0]), np.array([0.0, 100.0]), 20.0, 0.4)
    assert x[0] == 0.4
    assert abs(x[1] - (50 + 20 * 0.4) / 120) < 1e-12
