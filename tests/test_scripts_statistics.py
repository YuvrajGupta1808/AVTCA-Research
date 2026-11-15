"""Paired tests and the participant bootstrap of the paper scripts (scripts/paper/)."""
import importlib.util
import os

import numpy as np
import pytest
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


st = _load('significance_tests_under_test', 'scripts/paper/significance_tests.py')


def _pair(n_a, n_b, n_both=10, n_neither=3):
    """Predictions where only a is right on n_a clips and only b on n_b clips."""
    y = np.zeros(n_a + n_b + n_both + n_neither, int)
    a = np.ones_like(y); b = np.ones_like(y)
    a[:n_a] = 0                       # only a right
    b[n_a:n_a + n_b] = 0              # only b right
    a[n_a + n_b:n_a + n_b + n_both] = 0
    b[n_a + n_b:n_a + n_b + n_both] = 0
    return a, b, y


def test_exact_mcnemar_matches_the_binomial_test():
    a, b, y = _pair(7, 2)
    r = st.mcnemar(a, b, y)
    assert (r['only_a'], r['only_b']) == (7, 2)
    assert r['p'] == pytest.approx(binomtest(7, 9, 0.5).pvalue)
    assert r['delta'] == pytest.approx(100 * 5 / len(y))


def test_exact_mcnemar_hand_values():
    a, b, y = _pair(5, 0)
    assert st.mcnemar(a, b, y)['p'] == pytest.approx(2 * 0.5 ** 5)  # two-sided: 0.0625
    a, b, y = _pair(0, 0)
    assert st.mcnemar(a, b, y)['p'] == 1.0


def test_holm_adjustment_is_monotone_and_capped():
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> raised to 0.06 (monotone)
    assert st.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert st.holm([0.6, 0.9]) == pytest.approx([1.0, 1.0])
    assert st.holm([0.2]) == pytest.approx([0.2])


def _bootstrap_module():
    return _load('participant_bootstrap_under_test', 'scripts/paper/participant_bootstrap.py')


def test_participant_of_reads_the_number():
    pbm = _bootstrap_module()
    assert pbm.participant_of('subject_105_jwdq6sdejx_vid_0_1') == '105'


def test_bootstrap_is_deterministic_and_resamples_whole_participants():
    pbm = _bootstrap_module()
    rng = np.random.default_rng(1)
    part = np.repeat(['3', '11', '7', '20'], [5, 2, 9, 4])
    a = rng.random(part.size) < 0.7
    b = rng.random(part.size) < 0.5
    units, idx, d1 = pbm.bootstrap([(a, b)], part, draws=200, seed=0)
    _, _, d2 = pbm.bootstrap([(a, b)], part, draws=200, seed=0)
    _, _, d3 = pbm.bootstrap([(a, b)], part, draws=200, seed=1)
    assert list(units) == ['11', '20', '3', '7']  # sorted as strings
    assert np.array_equal(d1, d2) and not np.array_equal(d1, d3)
    # a participant who is always right under a and b contributes no difference
    same = np.ones(part.size, bool)
    _, _, d0 = pbm.bootstrap([(same, same)], part, draws=50, seed=0)
    assert not d0.any()
    # every draw is a mean over whole participants: the set of possible values is finite
    per = {u: 100 * (a[idx[u]].sum() - b[idx[u]].sum()) for u in units}
    draws = pbm.bootstrap([(a, b)], part, draws=1, seed=5)[2][0, 0]
    rng5 = np.random.default_rng(5)
    pick = rng5.choice(units, len(units))
    expected = sum(per[u] for u in pick) / sum(len(idx[u]) for u in pick)
    assert draws == pytest.approx(expected)
