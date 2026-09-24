"""Unit tests for the parts of the pipeline that are easy to get quietly wrong.

These run on synthetic data and never touch the network, so they are fast and
deterministic. What they cover is the machinery whose failures would not show
up as a crash: a Qini coefficient that silently rewards the wrong ranking, a
downloaded file that is not the published experiment, learner selection that
returns something the rest of the code cannot use.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hillstrom_ab_analysis as hb  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic fixtures                                                          #
# --------------------------------------------------------------------------- #
def _two_arm_frame(n=4000, seed=0):
    """A two-arm trial where half the customers have uplift and half have none.

    Uplift is 30pp for a random half and 0 for the rest, so the true ranking is
    known and anything claiming to measure ranking quality has a right answer to
    check against. The responsive half is scattered rather than placed first on
    purpose: qini_curve sorts with a stable argsort, so tied scores are ranked
    in row order, and a fixture whose row order carries the signal would let a
    constant score look like a perfect ranker.
    """
    rng = np.random.default_rng(seed)
    true_uplift = np.zeros(n)
    true_uplift[rng.permutation(n)[: n // 2]] = 0.30
    t = np.zeros(n, int)
    t[rng.permutation(n)[: n // 2]] = 1
    y = (rng.random(n) < 0.10 + t * true_uplift).astype(int)
    return y, t, true_uplift


def _hillstrom_shaped_frame():
    """A frame with the published columns, row count and arm sizes."""
    segments = (
        [hb.MENS] * hb.EXPECTED_ARMS[hb.MENS]
        + [hb.WOMENS] * hb.EXPECTED_ARMS[hb.WOMENS]
        + [hb.CONTROL] * hb.EXPECTED_ARMS[hb.CONTROL]
    )
    n = len(segments)
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "recency": rng.integers(1, 13, n),
        "history_segment": "1) $0 - $100",
        "history": rng.random(n) * 100,
        "mens": rng.integers(0, 2, n),
        "womens": rng.integers(0, 2, n),
        "zip_code": "Rural",
        "newbie": rng.integers(0, 2, n),
        "channel": "Web",
        "segment": segments,
        "visit": rng.integers(0, 2, n),
        "conversion": 0,
        "spend": 0.0,
    })


# --------------------------------------------------------------------------- #
# qini_curve                                                                  #
# --------------------------------------------------------------------------- #
def test_qini_rewards_the_true_ranking_over_random_ones():
    """Against a null of random rankings, not against one lucky draw.

    A single random score can land a Qini of 15 on 4,000 rows by chance, so
    comparing to one draw makes a flaky test. The claim worth asserting is the
    one randomization inference makes in Layer 8: the true ranking beats the
    whole null distribution, and that distribution is centred on zero.
    """
    y, t, true_uplift = _two_arm_frame()
    good = hb.qini_curve(y, t, true_uplift)[3]
    null = np.array([
        hb.qini_curve(y, t, np.random.default_rng(1000 + i).random(len(y)))[3]
        for i in range(30)
    ])
    assert good > null.max()
    assert abs(null.mean()) < 0.1 * good


def test_qini_of_a_constant_score_is_about_zero():
    """No ranking means no lift over the random baseline, by construction."""
    y, t, _ = _two_arm_frame()
    coef = hb.qini_curve(y, t, np.zeros(len(y)))[3]
    total = y[t == 1].sum() - y[t == 0].sum() * (t.sum() / (1 - t).sum())
    assert abs(coef) < 0.02 * abs(total)


def test_qini_is_invariant_to_monotone_rescaling():
    """The property that makes the Layer 6b calibration check necessary.

    Qini sees only the order of the scores, so a model that ranks correctly and
    predicts magnitudes that are wildly off earns the same coefficient as one
    that predicts the truth. If this test ever fails, the calibration layer's
    justification has changed and so should its prose.
    """
    y, t, true_uplift = _two_arm_frame()
    base = hb.qini_curve(y, t, true_uplift)[3]
    for transform in (lambda s: s * 7.0 + 3.0, np.exp, lambda s: s ** 3):
        assert hb.qini_curve(y, t, transform(true_uplift))[3] == pytest.approx(base)


def test_qini_breaks_ties_in_row_order():
    """Documented, not accidental: tied scores are ranked by position.

    qini_curve sorts with a stable mergesort, so a score that cannot separate
    customers hands the ranking to whatever order the rows arrive in. That is
    harmless in the pipeline, where the reporting split comes out of a shuffled
    train_test_split, and it is worth knowing before anyone feeds qini_curve a
    frame sorted by outcome.
    """
    y, t, true_uplift = _two_arm_frame()
    flat = np.zeros(len(y))
    order = np.argsort(-true_uplift, kind="mergesort")   # responsive rows first
    sorted_coef = hb.qini_curve(y[order], t[order], flat)[3]
    assert sorted_coef > 10 * abs(hb.qini_curve(y, t, flat)[3])


def test_uplift_at_k_finds_the_responsive_half():
    y, t, true_uplift = _two_arm_frame()
    top = hb.uplift_at_k(y, t, true_uplift, k=0.30)
    overall = y[t == 1].mean() - y[t == 0].mean()
    assert top > overall


# --------------------------------------------------------------------------- #
# _validate_dataset                                                           #
# --------------------------------------------------------------------------- #
def test_validate_accepts_a_correctly_shaped_file(tmp_path):
    path = tmp_path / "ok.csv"
    _hillstrom_shaped_frame().to_csv(path, index=False)
    hb._validate_dataset(path)          # must not raise


@pytest.mark.parametrize("corrupt, expected", [
    (lambda df: df.drop(columns=["visit"]), "missing columns"),
    (lambda df: df.iloc[:-1], "rows"),
    (lambda df: df.assign(segment=df["segment"].str.replace(
        hb.CONTROL, hb.MENS, regex=False)), "arm sizes"),
])
def test_validate_rejects_a_file_that_is_not_the_experiment(tmp_path, corrupt, expected):
    path = tmp_path / "bad.csv"
    corrupt(_hillstrom_shaped_frame()).to_csv(path, index=False)
    with pytest.raises(ValueError, match=expected):
        hb._validate_dataset(path)


def test_validate_rejects_an_html_error_page(tmp_path):
    """The failure the cache is most likely to see: a 200 that isn't the CSV."""
    path = tmp_path / "error.html"
    path.write_text("<html><body>404 Not Found</body></html>", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        hb._validate_dataset(path)


# --------------------------------------------------------------------------- #
# Learner fitting and selection                                               #
# --------------------------------------------------------------------------- #
def _learner_frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    df = _hillstrom_shaped_frame().sample(n, random_state=seed).reset_index(drop=True)
    df["T"] = rng.integers(0, 2, n)
    # Give the models something real to find, so selection is not a coin flip.
    df["visit"] = (rng.random(n) < 0.10 + df["T"] * df["womens"] * 0.30).astype(int)
    return df


def test_fit_learners_scores_every_row_of_the_scoring_frame():
    df = _learner_frame()
    train, score = df.iloc[:2000], df.iloc[2000:]
    score_t, score_s = hb._fit_learners(train, score)
    assert score_t.shape == score_s.shape == (len(score),)
    assert np.isfinite(score_t).all() and np.isfinite(score_s).all()


def test_select_learner_returns_a_usable_choice():
    sel = hb._select_learner(_learner_frame())
    assert sel["selected"] in {"T-learner", "S-learner"}
    assert sel["n"] == 3000
    # The winner is whichever cross-fitted coefficient is larger, ties to T.
    assert (sel["selected"] == "T-learner") == (sel["qini_t"] >= sel["qini_s"])


# --------------------------------------------------------------------------- #
# Policy value and its bootstrap                                              #
# --------------------------------------------------------------------------- #
def _policy_frame(n=6000, seed=0):
    """A split where true uplift tracks the score, so targeting should pay."""
    rng = np.random.default_rng(seed)
    score = rng.normal(0.03, 0.05, n)          # straddles the 3pp break-even
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.10 + t * np.clip(score, 0, None)).astype(int)
    return y, t, score


def _breakeven_policy(score):
    return (score > hb.COST_PER_EMAIL / hb.VALUE_PER_VISIT).astype(int)


def test_contacting_everyone_reproduces_the_blanket_figure():
    """The targeted arm of the estimator must reduce to blanket at policy = 1."""
    y, t, score = _policy_frame()
    qini, nv_blanket, nv_targeted, frac = hb._reporting_stats(
        y, t, score, np.ones(len(y), int))
    assert frac == 1.0
    assert nv_targeted == pytest.approx(nv_blanket)


def test_contacting_no_one_is_worth_nothing():
    """Net value is measured against contacting no one, so that policy is 0."""
    y, t, score = _policy_frame()
    _, _, nv_targeted, frac = hb._reporting_stats(
        y, t, score, np.zeros(len(y), int))
    assert frac == 0.0
    assert nv_targeted == pytest.approx(0.0)


def test_reporting_stats_qini_agrees_with_qini_curve():
    y, t, score = _policy_frame()
    stats_qini = hb._reporting_stats(y, t, score, _breakeven_policy(score))[0]
    assert stats_qini == pytest.approx(hb.qini_curve(y, t, score)[3])


def test_targeting_pays_when_uplift_tracks_the_score():
    y, t, score = _policy_frame()
    _, nv_blanket, nv_targeted, frac = hb._reporting_stats(
        y, t, score, _breakeven_policy(score))
    assert 0.0 < frac < 1.0
    assert nv_targeted > nv_blanket


def test_bootstrap_centres_on_the_observed_gain():
    """Why _reporting_stats re-reads the propensity instead of fixing it.

    Resamples land with a slightly different treated share. Weighting them by
    the design constant rather than by their own share biases each resample,
    and the bootstrap drifts off the point estimate it is meant to describe.
    Re-reading it keeps the distribution centred, which is what makes the
    percentile interval mean anything.
    """
    y, t, score = _policy_frame()
    pol = _breakeven_policy(score)
    obs = hb._reporting_stats(y, t, score, pol)
    gain_obs = obs[2] - obs[1]

    rng = np.random.default_rng(3)
    n = len(y)
    gains = np.array([
        (lambda s: s[2] - s[1])(
            hb._reporting_stats(*(a[i] for a in (y, t, score, pol))))
        for i in (rng.integers(0, n, n) for _ in range(300))
    ])
    assert abs(gains.mean() - gain_obs) < 0.25 * gains.std()
    assert gains.std() > 0
