"""image_scoring: the Scorer interface, the baseline scorer, scoring the windows of an image and choosing regions."""

import numpy as np
import pytest

from pyTEMlib.image_tools import BaselineScorer, Scorer, score_windows, select_regions

from .conftest import image_dataset, particles


def test_baseline_is_a_scorer():
    scorer = BaselineScorer()
    assert isinstance(scorer, Scorer) and scorer.name == "baseline" and scorer.version


def test_particles_score_higher_than_blank_support():
    blank = 50 + np.random.default_rng(1).normal(0, 2, (128, 128))
    scorer = BaselineScorer()
    high, reason = scorer.score(particles(), {})
    low, _ = scorer.score(blank, {})
    assert 0.0 <= low < 0.2 < 0.5 < high <= 1.0
    assert "edge contrast" in reason


def test_blurred_particles_score_lower():
    scorer = BaselineScorer()
    sharp, _ = scorer.score(particles(seed=5), {})
    blurred, reason = scorer.score(particles(seed=5, blur=3.0), {})
    assert blurred < 0.2 < 0.5 < sharp
    assert "edge width" in reason


def test_score_windows_finds_the_busy_corner_and_tells_the_scorer_the_pixel_size():
    image = 50 + np.random.default_rng(2).normal(0, 2, (256, 256))
    image[:128, 128:] = particles(seed=3)
    seen = []

    class Recording(BaselineScorer):
        def score(self, region, metadata):
            seen.append(metadata)
            return super().score(region, metadata)

    scores = sorted(score_windows(image_dataset(image, 0.25e-9), Recording(), 128), key=lambda s: s["value"], reverse=True)
    assert (scores[0]["row"], scores[0]["col"]) == (0, 128)
    assert all(0.0 <= s["value"] <= 1.0 for s in scores) and len(scores) == 4
    assert seen[0]["pixel_size_m"] == pytest.approx(0.25e-9) and seen[0]["window"] == 128


def test_score_windows_needs_an_image_dataset():
    with pytest.raises(TypeError):
        score_windows(np.zeros((64, 64)))


def test_select_regions_keeps_the_best_windows_that_do_not_overlap():
    scores = [
        {"row": 0, "col": 0, "window": 64, "value": 0.9},
        {"row": 32, "col": 32, "window": 64, "value": 0.8},  # overlaps the best: skipped
        {"row": 0, "col": 64, "window": 64, "value": 0.7},
        {"row": 128, "col": 0, "window": 64, "value": 0.6},
    ]
    assert [(s["row"], s["col"]) for s in select_regions(scores, top_k=2)] == [(0, 0), (0, 64)]
    assert [(s["row"], s["col"]) for s in select_regions(scores, top_k=5, min_separation=16)] == [(0, 0), (32, 32), (0, 64), (128, 0)]
    assert select_regions(scores, top_k=0) == []


def test_deterministic():
    image = particles(seed=4)
    assert BaselineScorer().score(image, {}) == BaselineScorer().score(image, {})
