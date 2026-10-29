"""Evaluated field-of-view reward: precision@5 against known particles of several kinds, a random reference, and sensitivity to image faults.

The reward is a detection proxy (it finds sharp, clean particles); none of this
checks it against what a microscopist would choose. Seeds 0-2 were used to tune
the baseline; the tests here use held-out seeds.
"""

import json
from math import comb

import numpy as np
import pytest

from pyTEMlib.scoring import BaselineScorer, BaselineScorerV2
from pyTEMlib.scoring.bench import main as bench
from pyTEMlib.scoring.evaluate import (
    DEFAULT_THRESHOLD, FEATURE_KINDS, HELDOUT_SEEDS, TUNING_SEEDS, LabelledMap, _render_particles, _signal_per_nm,
    defocus, evaluate, make_labelled_map, render_cell,
)

SEEDS = HELDOUT_SEEDS[:4]  # the bench reports all of 3-29; a few keep the tests quick
SUPPORT = 3.0 * _signal_per_nm("C")


class EdgeCountScorer:
    """baseline 1.0.0's rule (share of the region covered by edges): rewards blur, so it must fail."""

    name = "edge_count"
    version = "test"
    description = "fraction of pixels on an edge"

    def score(self, region, metadata):
        from scipy import ndimage

        smooth = ndimage.gaussian_filter(np.asarray(region, float), 1.0)
        noise = np.std(region - smooth) + 1e-9
        return float((np.hypot(*np.gradient(smooth)) > 1.5 * noise).mean()), "edges"


@pytest.fixture(scope="module")
def baseline():
    return evaluate("baseline", seeds=SEEDS)


@pytest.fixture(scope="module")
def baseline_v2():
    return evaluate("baseline_v2", seeds=SEEDS)


@pytest.fixture(scope="module")
def random_result():
    return evaluate("random", seeds=SEEDS)


def test_threshold_is_far_above_chance():
    # 30 of 100 regions are clean features; a random ranking puts >= 4 of them in the top 5 with p ~ 2.7 %,
    # and >= 4 of one kind's 5 regions among its 75 candidates with p ~ 2e-5.
    overall = sum(comb(30, h) * comb(70, 5 - h) for h in (4, 5)) / comb(100, 5)
    per_kind = sum(comb(5, h) * comb(70, 5 - h) for h in (4, 5)) / comb(75, 5)
    assert DEFAULT_THRESHOLD == 0.8 and overall < 0.03 and per_kind < 1e-4


def test_tuning_and_held_out_seeds_are_disjoint(baseline):
    assert set(TUNING_SEEDS) == {0, 1, 2} and set(HELDOUT_SEEDS) == set(range(3, 30))
    assert baseline["held_out"] and not set(baseline["seeds"]) & set(TUNING_SEEDS)


def test_baseline_meets_threshold_on_held_out_seeds(baseline):
    assert [r["seed"] for r in baseline["per_seed"]] == list(SEEDS)
    assert all(r["precision_at_k"] >= DEFAULT_THRESHOLD for r in baseline["per_seed"])
    assert baseline["passed"]


def test_every_feature_kind_ranks_above_the_negatives(baseline):
    assert set(baseline["per_category_mean_precision"]) == set(FEATURE_KINDS)
    for kind, precision in baseline["per_category_mean_precision"].items():
        assert precision >= DEFAULT_THRESHOLD, kind
    for kind in ("separated", "crowded", "small", "large", "sampling"):
        assert baseline["per_category_min_precision"][kind] >= DEFAULT_THRESHOLD, kind


def test_sharp_crowded_and_small_regions_outrank_defocused_and_contaminated():
    scorer = BaselineScorer()
    for seed in SEEDS[:2]:
        m = make_labelled_map(seed)
        w = m.window
        values = {}
        for r in m.regions():
            values.setdefault(r["category"], []).append(scorer.score(m.image[r["row"] : r["row"] + w, r["col"] : r["col"] + w], {})[0])
        spoiled = np.array(values["defocused"] + values["contaminated"])
        for kind in ("crowded", "small"):
            good = np.array(values[kind])
            assert good.min() > np.percentile(spoiled, 95), (seed, kind)
            assert (good[:, None] > spoiled[None, :]).mean() >= 0.98, (seed, kind)


def test_overall_precision_alone_would_have_missed_the_2_0_0_bias(baseline_v2):
    # 2.0.0 fills its top 5 with well-separated particles, so overall precision looks fine, but it ranks
    # crowded, small and large particles below defocused ones. The per-kind numbers catch that.
    assert baseline_v2["scorer_version"] == "2.0.0"
    assert baseline_v2["mean_precision"] >= DEFAULT_THRESHOLD
    assert not baseline_v2["passed"]
    for kind in ("crowded", "small", "large"):
        assert baseline_v2["per_category_mean_precision"][kind] < 0.5, kind


def test_random_scorer_falls_below(baseline, random_result):
    assert not random_result["passed"]
    assert random_result["mean_precision"] < 0.5 < baseline["mean_precision"]
    assert all(v < 0.3 for v in random_result["per_category_mean_precision"].values())


def test_every_negative_lowers_the_score(baseline):
    for r in baseline["per_seed"]:
        for fault in ("defocused", "contaminated", "empty"):
            sens = r["sensitivity"][fault]
            assert sens["fraction_lowered"] >= 0.9, (r["seed"], fault)
            assert sens["mean_change"] < -0.3, (r["seed"], fault)
            assert set(sens["fraction_lowered_by_kind"]) == set(FEATURE_KINDS)


def test_a_scorer_that_rewards_blur_fails():
    result = evaluate(EdgeCountScorer(), seeds=SEEDS[:1])
    assert not result["passed"]
    assert result["per_seed"][0]["sensitivity"]["defocused"]["mean_change"] > 0


def test_result_records_version_seeds_and_claim(baseline):
    assert baseline["scorer"] == "baseline" and baseline["scorer_version"] == BaselineScorer.version == "3.0.0"
    assert baseline["seeds"] == list(SEEDS) and baseline["dataset"] == "synthetic-fov-2"
    assert "detection proxy" in baseline["claim"] and "not validated" in baseline["claim"]
    json.dumps(baseline)  # storable as one JSONL line


def test_dataset_is_seeded_and_balanced():
    a, b = make_labelled_map(1), make_labelled_map(1)
    assert np.array_equal(a.image, b.image)
    regions = a.regions()
    labels = [r["label"] for r in regions]
    assert {lab: labels.count(lab) for lab in set(labels)} == {"feature": 30, "empty": 24, "defocused": 23, "contaminated": 23}
    kinds = [r["category"] for r in regions if r["label"] == "feature"]
    assert {k: kinds.count(k) for k in set(kinds)} == {k: 5 for k in FEATURE_KINDS}
    assert not np.array_equal(a.image, make_labelled_map(2).image)


def test_feature_kinds_cover_what_they_claim():
    rng = np.random.default_rng(0)
    window = 96
    small = [render_cell(rng, "small", window) for _ in range(20)]
    assert all(len(p) == 1 and (s > 0).mean() < 0.01 for s, p in small)
    radii = [p[0]["semi_axes_px"][0] for _ in range(60) for _, p in [render_cell(rng, "sampling", window)]]
    assert min(radii) < 3 and max(radii) > 30 and 1.8 <= min(radii) and max(radii) <= 40
    crowded = [render_cell(rng, "crowded", window)[1] for _ in range(10)]
    overlapping = sum(
        any(np.hypot(*np.subtract(p["center_px"], q["center_px"])) < p["semi_axes_px"][0] + q["semi_axes_px"][0] for q in ps[:i])
        for ps in crowded for i, p in enumerate(ps)
    )
    assert all(len(ps) >= 8 for ps in crowded) and overlapping > 10
    large = [render_cell(rng, "large", window)[1] for _ in range(10)]
    assert all(p["semi_axes_px"][0] >= 14 for ps in large for p in ps)
    low, _ = render_cell(rng, "low_contrast", window)
    assert 4 * np.sqrt(SUPPORT) * 0.99 <= low.max() <= 8 * np.sqrt(SUPPORT) * 1.01


# ---------------------------------------------------------------- the review's three probes (regression)


def _noisy(signal, seed):
    return np.random.default_rng(seed).poisson(SUPPORT + signal).astype(np.float64)


def _ellipsoid(size, centre, a, element):
    return _render_particles((size, size), [{"center_px": centre, "semi_axes_px": (a, a, 0.4 * a), "theta_rad": 0.0, "element": element}], 0.25)


def _overlapping_disks(size=128, r=25, height=150.0):
    rows, cols = np.mgrid[:size, :size]
    signal = np.zeros((size, size))
    for cr, cc in [(40, 40), (40, 75), (64, 58), (88, 40), (88, 80)]:
        signal += height * ((rows - cr) ** 2 + (cols - cc) ** 2 <= r * r)
    return signal


def test_probe_one_sharp_particle_of_radius_10():
    scorer = BaselineScorer()
    for seed in range(4):
        signal = _ellipsoid(64, (32 + seed, 30 - seed), 10.0, "Au")
        sharp, _ = scorer.score(_noisy(signal, seed), {})
        blurred, _ = scorer.score(_noisy(defocus(signal, 4.0), seed), {})
        assert sharp > 0.7 and blurred < 0.1, seed


def test_probe_five_overlapping_sharp_disks_of_radius_25():
    signal = _overlapping_disks()
    for seed in range(3):
        image = _noisy(signal, seed)
        assert BaselineScorerV2().score(image, {})[0] < 0.05  # 2.0.0: overlap doubles the height, read as blur
        value, reason = BaselineScorer().score(image, {})
        assert value > 0.7, reason
        assert BaselineScorer().score(_noisy(defocus(signal, 4.0), seed), {})[0] < 0.1


def test_probe_single_small_particle_under_one_percent_of_the_region():
    signal = _ellipsoid(64, (20, 41), 3.0, "Fe")
    assert (signal > 0).mean() < 0.01
    for seed in range(4):
        sharp_v2 = BaselineScorerV2().score(_noisy(signal, seed), {})[0]
        blurred_v2 = BaselineScorerV2().score(_noisy(defocus(signal, 4.0), seed), {})[0]
        assert sharp_v2 < 0.3 and abs(sharp_v2 - blurred_v2) < 0.1  # 2.0.0: low, whatever the focus
        sharp = BaselineScorer().score(_noisy(signal, seed), {})[0]
        blurred = BaselineScorer().score(_noisy(defocus(signal, 4.0), seed), {})[0]
        assert sharp > 0.5 and blurred < 0.1, seed


def test_from_twin_ground_truth_records():
    # export_ground_truth gives stage centres in metres; rows follow y, columns follow x.
    gt = [{"id": 0, "center_m": [10e-9, 2e-9], "semi_axes_m": [1e-9, 1e-9, 0.4e-9], "theta_rad": 0.0, "element": "Au"}]
    m = LabelledMap.from_twin(np.zeros((64, 128)), gt, origin_m=(0.0, 0.0), pixel_size_m=0.25e-9, window=64,
                              bad_regions=[{"kind": "contaminated", "box_px": (0, 64, 64, 128)}])
    assert m.particles[0]["center_px"] == pytest.approx((8.0, 40.0))
    assert [r["label"] for r in m.regions()] == ["feature", "contaminated"]


def test_bench_command_line(tmp_path, capsys):
    out = tmp_path / "results.jsonl"
    assert bench(["--scorer", "random", "--seeds", "3", "--brief", "--out", str(out)]) == 1
    record = json.loads(out.read_text().splitlines()[0])
    assert record["scorer"] == "random" and record["seeds"] == [3]
    assert bench(["--scorer", "tests.test_reward_eval:EdgeCountScorer", "--seeds", "3", "--brief"]) == 1
    assert '"scorer": "edge_count"' in capsys.readouterr().out
