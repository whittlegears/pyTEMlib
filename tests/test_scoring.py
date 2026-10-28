"""Scorer interface and the baseline scorer."""

import numpy as np
from scipy import ndimage

from pyTEMlib.scoring import BaselineScorer, Scorer, available_scorers, score_windows


def particles(n=20, seed=0, size=128):
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size))
    image[rng.integers(0, size, n), rng.integers(0, size, n)] = 1.0
    return ndimage.gaussian_filter(image, 2) * 2000 + 50 + rng.normal(0, 2, (size, size))


def test_baseline_is_a_scorer_and_listed():
    assert isinstance(BaselineScorer(), Scorer)
    assert "baseline" in available_scorers()


def test_particles_score_higher_than_blank_support():
    blank = 50 + np.random.default_rng(1).normal(0, 2, (128, 128))
    scorer = BaselineScorer()
    high, reason = scorer.score(particles(), {})
    low, _ = scorer.score(blank, {})
    assert 0.0 <= low < 0.2 < 0.5 < high <= 1.0
    assert "contrast" in reason


def test_score_windows_finds_the_busy_corner():
    image = 50 + np.random.default_rng(2).normal(0, 2, (256, 256))
    image[:128, 128:] = particles(seed=3)
    scores = sorted(score_windows(image, BaselineScorer(), 128), key=lambda s: s["value"], reverse=True)
    assert (scores[0]["row"], scores[0]["col"]) == (0, 128)
    assert all(0.0 <= s["value"] <= 1.0 for s in scores)


def test_deterministic():
    image = particles(seed=4)
    assert BaselineScorer().score(image, {}) == BaselineScorer().score(image, {})
