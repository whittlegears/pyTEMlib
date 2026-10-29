"""Evaluated field-of-view reward: precision@5 against known particles, a random reference, and sensitivity to image faults.

The reward is a detection proxy (it finds sharp, clean particles); none of this
checks it against what a microscopist would choose.
"""

import json

import numpy as np
import pytest

from pyTEMlib.scoring import BaselineScorer
from pyTEMlib.scoring.bench import main as bench
from pyTEMlib.scoring.evaluate import DEFAULT_THRESHOLD, LabelledMap, evaluate, make_labelled_map

SEEDS = (0, 1, 2)


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
def random_result():
    return evaluate("random", seeds=SEEDS)


def test_threshold_is_far_above_chance():
    # 16 of 64 regions are clean features; a random ranking puts >= 4 of them in the top 5 with p ~ 1.6 %.
    from math import comb

    p = sum(comb(16, h) * comb(48, 5 - h) for h in (4, 5)) / comb(64, 5)
    assert DEFAULT_THRESHOLD == 0.8 and p < 0.02


def test_baseline_meets_threshold_on_three_seeds(baseline):
    assert [r["seed"] for r in baseline["per_seed"]] == list(SEEDS)
    assert all(r["precision_at_k"] >= DEFAULT_THRESHOLD for r in baseline["per_seed"])
    assert baseline["passed"]


def test_random_scorer_falls_below(baseline, random_result):
    assert not random_result["passed"]
    assert all(r["precision_at_k"] < DEFAULT_THRESHOLD for r in random_result["per_seed"])
    assert random_result["mean_precision"] < baseline["mean_precision"]


def test_blur_and_contamination_lower_the_score(baseline):
    for r in baseline["per_seed"]:
        for fault in ("defocused", "contaminated"):
            assert r["sensitivity"][fault]["fraction_lowered"] >= 0.9, (r["seed"], fault)
            assert r["sensitivity"][fault]["mean_change"] < -0.3, (r["seed"], fault)


def test_a_scorer_that_rewards_blur_fails():
    result = evaluate(EdgeCountScorer(), seeds=SEEDS[:1])
    assert not result["passed"]
    assert result["per_seed"][0]["sensitivity"]["defocused"]["mean_change"] > 0


def test_result_records_version_seeds_and_claim(baseline):
    assert baseline["scorer"] == "baseline" and baseline["scorer_version"] == BaselineScorer.version
    assert baseline["seeds"] == list(SEEDS) and baseline["dataset"]
    assert "detection proxy" in baseline["claim"] and "not validated" in baseline["claim"]
    json.dumps(baseline)  # storable as one JSONL line


def test_dataset_is_seeded_and_balanced():
    a, b = make_labelled_map(1), make_labelled_map(1)
    assert np.array_equal(a.image, b.image)
    labels = [r["label"] for r in a.regions()]
    assert {lab: labels.count(lab) for lab in set(labels)} == {"feature": 16, "empty": 16, "defocused": 16, "contaminated": 16}
    assert not np.array_equal(a.image, make_labelled_map(2).image)


def test_from_twin_ground_truth_records():
    # export_ground_truth gives stage centres in metres; rows follow y, columns follow x.
    gt = [{"id": 0, "center_m": [10e-9, 2e-9], "semi_axes_m": [1e-9, 1e-9, 0.4e-9], "theta_rad": 0.0, "element": "Au"}]
    m = LabelledMap.from_twin(np.zeros((64, 128)), gt, origin_m=(0.0, 0.0), pixel_size_m=0.25e-9, window=64,
                              bad_regions=[{"kind": "contaminated", "box_px": (0, 64, 64, 128)}])
    assert m.particles[0]["center_px"] == pytest.approx((8.0, 40.0))
    assert [r["label"] for r in m.regions()] == ["feature", "contaminated"]


def test_bench_command_line(tmp_path, capsys):
    out = tmp_path / "results.jsonl"
    assert bench(["--scorer", "random", "--seeds", "0", "--brief", "--out", str(out)]) == 1
    record = json.loads(out.read_text().splitlines()[0])
    assert record["scorer"] == "random" and record["seeds"] == [0]
    assert bench(["--scorer", "tests.test_reward_eval:EdgeCountScorer", "--seeds", "0", "--brief"]) == 1
    assert '"scorer": "edge_count"' in capsys.readouterr().out
