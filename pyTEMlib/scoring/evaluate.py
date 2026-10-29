"""
Evaluate a field-of-view scorer against labelled maps.

What this measures, honestly: the labelled maps know where the particles are
and which regions were deliberately spoiled (defocused, contaminated) or left
empty. A scorer passes if its top regions are the ones that hold sharp, clean
particles. That is a **detection proxy** for "a good field of view": it checks
that the scorer finds visible features of several kinds and is pushed down by
the two image faults we can simulate. It has not been checked against what a
microscopist would pick; there are no human-preference labels yet (operator
picks at first light are planned as the first ones).

Clean feature regions come in six kinds, five of each per map, so a scorer
tuned to one particle look cannot pass:

* ``separated``: 2-5 well-separated particles of ~6 px radius (the twin's sample);
* ``crowded``: 8-14 particles of 4-14 px that may overlap (summed thickness);
* ``small``: one particle of 2-4.5 px radius, under 1 % of the region;
* ``large``: one or two particles of 16-32 px radius;
* ``sampling``: the same 1.6 nm particles at another pixel size (radius 2-40 px);
* ``low_contrast``: sharp particles whose peak is only 4-8 x the background noise.

Negatives: 24 empty regions, 23 defocused and 23 contaminated ones per map;
each spoiled region starts from the layout of a randomly chosen clean kind.

Numbers per scorer:

* ``precision@k`` (k = 5): of the k regions the scorer ranks highest on a map,
  the fraction that are clean feature regions (30 of 100 per map). Per
  category: the same over that kind's 5 regions plus the 70 negatives. Regions
  are the cells of a grid of ``window`` pixels, so every ranked region has one
  label; ``score_map`` ranks half-overlapping windows instead.
* sensitivity: clean regions of every kind are re-rendered with the same
  particles and noise draw, once defocused, once contaminated and once with
  the particles removed (empty); the check reports how often, and by how
  much, each negative lowers the score.

Seeds 0-2 are for tuning (``TUNING_SEEDS``); seeds 3-29 are held out
(``HELDOUT_SEEDS``, the default) and are the numbers to report.

The dataset is synthetic and generated in-process from a seed, so the
evaluation runs without a microscope, a twin or a server. Its image model
follows the asyncroscopy digital twin's sample (ellipsoidal Au/Pt/Fe particles
on a carbon support, HAADF intensity ~ Z^1.7 x thickness, Poisson counts) but
does not import it: pyTEMlib must not depend on asyncroscopy. Twin images can
be evaluated too: build a ``LabelledMap`` from the image and the twin's
``export_ground_truth`` records with ``LabelledMap.from_twin``.
"""

from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from scipy import ndimage

from . import Scorer, available_scorers

DATASET_VERSION = "synthetic-fov-2"
TUNING_SEEDS = (0, 1, 2)
HELDOUT_SEEDS = tuple(range(3, 30))
DEFAULT_SEEDS = HELDOUT_SEEDS
DEFAULT_K = 5
# Proposed D6 threshold (to be agreed with WG2): at least 4 of the top 5 regions clean features, on
# every held-out seed, and on average for each feature kind. With 30 of 100 regions clean features, a
# scorer that ranks at random reaches 4 of 5 with probability 2.7 % per map (hypergeometric); per kind
# (5 of 75) with probability 2e-5.
DEFAULT_THRESHOLD = 0.8

# HAADF signal: counts per (Z^1.7 x atoms per cubic angstrom x nm): 3 nm of carbon support gives
# ~16 counts per pixel, the centre of a typical Au particle (~1.3 nm thick) ~330 and of Fe ~70.
_ELEMENTS = {"C": (6, 0.100), "Fe": (26, 0.085), "Pt": (78, 0.066), "Au": (79, 0.059)}
_COUNTS = 2.6
_PIXEL_NM = 0.25
_SUPPORT_NM = 3.0


def _signal_per_nm(element: str) -> float:
    z, density = _ELEMENTS[element]
    return _COUNTS * z**1.7 * density


@dataclass
class LabelledMap:
    """A map image with its particles and known bad regions, both in pixels of ``image``.

    particles: dicts with ``center_px`` (row, col); other keys are carried along.
    bad_regions: dicts with ``kind`` ("defocused", "contaminated", ...) and ``box_px``
    (row0, col0, row1, col1), end-exclusive.
    good_regions: optional dicts with ``kind`` (a feature category such as "crowded") and
    ``box_px``; they only name the category of a clean feature region.
    """

    image: np.ndarray
    window: int
    particles: list[dict]
    bad_regions: list[dict] = field(default_factory=list)
    pixel_size_m: float | None = None
    seed: int | None = None
    good_regions: list[dict] = field(default_factory=list)

    @classmethod
    def from_twin(cls, image, ground_truth: list[dict], origin_m: tuple[float, float], pixel_size_m: float, window: int, bad_regions=None, seed=None):
        """From a twin image and its ``export_ground_truth`` particle records (``center_m`` = stage x, y).

        origin_m: stage (x, y) of pixel (0, 0); rows follow y and columns follow x.
        """
        particles = []
        for p in ground_truth:
            x, y = p["center_m"][:2]
            particles.append({**p, "center_px": ((y - origin_m[1]) / pixel_size_m, (x - origin_m[0]) / pixel_size_m)})
        return cls(np.asarray(image, dtype=np.float64), window, particles, list(bad_regions or []), pixel_size_m, seed)

    def regions(self) -> list[dict]:
        """Grid regions (row, col top-left) with their label ("feature", "empty" or the bad kind) and category."""
        h, w = self.image.shape[:2]
        out = []
        for r in range(0, h - self.window + 1, self.window):
            for c in range(0, w - self.window + 1, self.window):
                box = (r, c, r + self.window, c + self.window)
                bad = next((b["kind"] for b in self.bad_regions if _overlap(box, b["box_px"])), None)
                has_feature = any(r <= p["center_px"][0] < box[2] and c <= p["center_px"][1] < box[3] for p in self.particles)
                label = bad or ("feature" if has_feature else "empty")
                category = next((g["kind"] for g in self.good_regions if _overlap(box, g["box_px"])), None) if label == "feature" else None
                out.append({"row": r, "col": c, "label": label, "category": category or label})
        return out


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# ---------------------------------------------------------------- synthetic labelled maps

FEATURE_KINDS = ("separated", "crowded", "small", "large", "sampling", "low_contrast")
NEGATIVE_KINDS = ("empty", "defocused", "contaminated")
CLASSES = ("feature",) + NEGATIVE_KINDS
PER_FEATURE_KIND = 5
NEGATIVE_COUNTS = {"empty": 24, "defocused": 23, "contaminated": 23}


def _place_particles(rng, window, count, radius, overlap=False, elements=("Au", "Pt", "Fe"), p=(0.4, 0.4, 0.2)) -> list[dict]:
    """Up to ``count`` ellipsoids fully inside a window x window cell (local pixels); ``radius(rng)`` draws a semi-axis."""
    out = []
    for _ in range(50 * count):
        if len(out) >= count:
            break
        a = float(radius(rng))
        b = a * float(rng.uniform(0.8, 1.2))
        margin = max(a, b) + 2
        if window - 2 * margin <= 0:
            continue
        centre = rng.uniform(margin, window - margin, 2)
        if overlap or all(np.hypot(*(centre - np.array(q["center_px"]))) > margin + max(q["semi_axes_px"][:2]) for q in out):
            out.append({
                "center_px": (float(centre[0]), float(centre[1])),
                "semi_axes_px": (a, b, 0.4 * a),
                "theta_rad": float(rng.uniform(0, np.pi)),
                "element": str(rng.choice(list(elements), p=list(p))),
            })
    return out


def _render_particles(shape, particles, pixel_nm) -> np.ndarray:
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float64)
    signal = np.zeros(shape)
    for p in particles:
        a, b, c = p["semi_axes_px"]
        dr, dc = rows - p["center_px"][0], cols - p["center_px"][1]
        cos_t, sin_t = np.cos(p["theta_rad"]), np.sin(p["theta_rad"])
        u, v = (dc * cos_t + dr * sin_t) / a, (-dc * sin_t + dr * cos_t) / b
        thickness_nm = 2 * c * pixel_nm * np.sqrt(np.clip(1 - u * u - v * v, 0, None))
        signal += thickness_nm * _signal_per_nm(p["element"])
    return signal


def render_cell(rng: np.random.Generator, kind: str, window: int) -> tuple[np.ndarray, list[dict]]:
    """Noiseless signal (counts above the support) and particles (local pixels) of one clean feature region."""
    pixel_nm = _PIXEL_NM
    scale = None
    if kind == "separated":  # the twin's sample: 1.6 +- 0.2 nm semi-axis at 0.25 nm per pixel (~6.4 px)
        particles = _place_particles(rng, window, int(rng.integers(2, 6)), lambda g: np.clip(g.normal(6.4, 0.8), 4.0, 9.0))
    elif kind == "crowded":
        particles = _place_particles(rng, window, int(rng.integers(8, 15)), lambda g: np.clip(g.normal(8.0, 2.0), 4.0, 14.0), overlap=True)
    elif kind == "small":
        particles = _place_particles(rng, window, 1, lambda g: g.uniform(2.0, 4.5))
    elif kind == "large":
        a = float(rng.uniform(16.0, 32.0))
        particles = _place_particles(rng, window, 1 if a > 22 else 2, lambda g: a * g.uniform(0.9, 1.0))
    elif kind == "sampling":
        # The twin's 1.6 nm particles at a pixel size that makes them 2-40 px in radius; counts per pixel
        # as at 0.25 nm (the dwell time is assumed to follow the pixel size).
        a = float(np.exp(rng.uniform(np.log(2.0), np.log(40.0))))
        pixel_nm = 1.6 / a
        particles = _place_particles(rng, window, 1 if a > 14 else int(rng.integers(1, 4)), lambda g: a * g.uniform(0.9, 1.0))
    elif kind == "low_contrast":
        particles = _place_particles(rng, window, int(rng.integers(2, 6)), lambda g: np.clip(g.normal(6.4, 0.8), 4.0, 9.0))
        scale = float(rng.uniform(4.0, 8.0)) * np.sqrt(_SUPPORT_NM * _signal_per_nm("C"))  # peak = 4-8 x background noise
    else:
        raise ValueError(f"unknown feature kind {kind!r}")
    signal = _render_particles((window, window), particles, pixel_nm)
    if scale is not None and signal.max() > 0:
        signal *= scale / signal.max()
    return signal, particles


def defocus(signal: np.ndarray, sigma_px: float) -> np.ndarray:
    """Probe defocus as a Gaussian blur of the noiseless signal (a disc would be closer; the trend is the same)."""
    return ndimage.gaussian_filter(signal, sigma_px, mode="nearest")


def contaminate(signal: np.ndarray, rng: np.random.Generator, thickness_nm: float = 12.0) -> np.ndarray:
    """Cover a region with a lumpy hydrocarbon (carbon-like) layer of about ``thickness_nm``.

    The layer adds its own HAADF signal and texture and spreads the beam, which blurs what lies under it.
    """
    lumps = ndimage.gaussian_filter(rng.normal(size=signal.shape), 6.0, mode="wrap")
    lumps = (lumps - lumps.min()) / (np.ptp(lumps) + 1e-12)
    layer_nm = thickness_nm * (0.5 + lumps)
    spread = ndimage.gaussian_filter(signal, 1.5 + thickness_nm / 10.0, mode="nearest")
    return spread + layer_nm * _signal_per_nm("C")


def make_labelled_map(seed: int, grid: int = 10, window: int = 96, pixel_size_m: float = 0.25e-9, support_nm: float = _SUPPORT_NM) -> LabelledMap:
    """A grid x grid map: 5 clean regions of each feature kind, the rest empty, defocused or contaminated.

    Needs grid * grid == 100 (the category counts are fixed); defocused and contaminated regions start
    from the layout of a randomly chosen feature kind.
    """
    if grid * grid != len(FEATURE_KINDS) * PER_FEATURE_KIND + sum(NEGATIVE_COUNTS.values()):
        raise ValueError("make_labelled_map needs a 10 x 10 grid")
    rng = np.random.default_rng(seed)
    kinds = [k for k in FEATURE_KINDS for _ in range(PER_FEATURE_KIND)] + [k for k, n in NEGATIVE_COUNTS.items() for _ in range(n)]
    kinds = [kinds[i] for i in rng.permutation(len(kinds))]
    support = support_nm * _signal_per_nm("C") * (1 + 0.05 * ndimage.gaussian_filter(rng.normal(size=(grid * window,) * 2), 20))
    image = np.array(support)
    particles, bad, good = [], [], []
    for i, kind in enumerate(kinds):
        r0, c0 = (i // grid) * window, (i % grid) * window
        box = (r0, c0, r0 + window, c0 + window)
        if kind == "empty":
            continue
        layout = kind if kind in FEATURE_KINDS else str(rng.choice(FEATURE_KINDS))
        signal, local = render_cell(rng, layout, window)
        if kind == "defocused":
            signal = defocus(signal, float(rng.uniform(3.0, 5.0)))
        elif kind == "contaminated":
            signal = contaminate(signal, rng, float(rng.uniform(8.0, 16.0)))
        image[r0 : r0 + window, c0 : c0 + window] += signal
        particles += [{**p, "center_px": (p["center_px"][0] + r0, p["center_px"][1] + c0), "layout": layout} for p in local]
        (good if kind in FEATURE_KINDS else bad).append({"kind": kind, "box_px": box})
    image = rng.poisson(np.clip(image, 0, None)).astype(np.float64)
    return LabelledMap(image, window, particles, bad, pixel_size_m, seed, good)


# ---------------------------------------------------------------- metrics


class RandomScorer:
    """Reference scorer: a seeded random number per region, ignoring the image."""

    name = "random"
    version = "1.0.0"
    description = "Uniform random value; the chance level any real scorer must beat."

    def __init__(self, seed: int = 0):
        self.seed = seed

    def score(self, region: np.ndarray, metadata: dict) -> tuple[float, str]:
        key = f"{self.seed}:{metadata.get('row')}:{metadata.get('col')}:{metadata.get('map_seed')}:{metadata.get('variant', '')}".encode()
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") / 2**64
        return value, "random"


def resolve_scorer(scorer: str | Scorer) -> Scorer:
    if not isinstance(scorer, str):
        return scorer
    if scorer == RandomScorer.name:
        return RandomScorer()
    scorers = available_scorers()
    if scorer not in scorers:
        raise ValueError(f"unknown scorer {scorer!r}; available: {sorted(scorers) + [RandomScorer.name]}")
    return scorers[scorer]


def _score_region(scorer: Scorer, region: np.ndarray, metadata: dict) -> float:
    value, _ = scorer.score(region, metadata)
    return float(np.clip(value, 0.0, 1.0))


def _precision(ranked: list[dict], k: int) -> float:
    return sum(s["label"] == "feature" for s in ranked[:k]) / k


def precision_at_k(scorer: Scorer, labelled: LabelledMap, k: int = DEFAULT_K) -> dict:
    """Fraction of the scorer's k highest-ranked grid regions that are clean feature regions, overall and per category.

    Per category: the ranking restricted to that category's clean regions plus every negative region.
    Ties are broken by position (row, then column), never in the scorer's favour.
    """
    w = labelled.window
    ranked = []
    for region in labelled.regions():
        meta = {"row": region["row"], "col": region["col"], "window": w, "pixel_size_m": labelled.pixel_size_m, "map_seed": labelled.seed}
        value = _score_region(scorer, labelled.image[region["row"] : region["row"] + w, region["col"] : region["col"] + w], meta)
        ranked.append({**region, "value": value})
    ranked.sort(key=lambda s: (-s["value"], s["row"], s["col"]))
    categories = sorted({s["category"] for s in ranked if s["label"] == "feature"}, key=lambda c: (FEATURE_KINDS + (c,)).index(c))
    per_category = {c: _precision([s for s in ranked if s["label"] != "feature" or s["category"] == c], k) for c in categories}
    mean_value = {lab: float(np.mean([s["value"] for s in ranked if s["category"] == lab])) for lab in {s["category"] for s in ranked}}
    return {
        "precision_at_k": _precision(ranked, k),
        "k": k,
        "top_labels": [s["category"] for s in ranked[:k]],
        "chance": sum(s["label"] == "feature" for s in ranked) / len(ranked),
        "per_category": per_category,
        "mean_score": mean_value,
    }


def sensitivity(scorer: Scorer, seed: int, per_kind: int = 4, window: int = 96, pixel_size_m: float = 0.25e-9) -> dict:
    """Score clean regions of every feature kind and the same regions spoiled (same particles, same noise draw).

    The spoiled versions are the negative categories: defocused, contaminated, and empty (particles removed).
    """
    rng = np.random.default_rng([seed, 7])
    support = _SUPPORT_NM * _signal_per_nm("C")
    faults = ("defocused", "contaminated", "empty")
    deltas = {f: [] for f in faults}
    by_kind = {f: {} for f in faults}
    clean_scores = []
    for kind in FEATURE_KINDS:
        for i in range(per_kind):
            signal, _ = render_cell(rng, kind, window)
            variants = {
                "clean": signal,
                "defocused": defocus(signal, float(rng.uniform(3.0, 5.0))),
                "contaminated": contaminate(signal, rng, float(rng.uniform(8.0, 16.0))),
                "empty": np.zeros_like(signal),
            }
            noise_seed = int(rng.integers(2**31))
            scores = {}
            for name, s in variants.items():
                image = np.random.default_rng(noise_seed).poisson(support + s).astype(np.float64)
                meta = {"row": i, "col": FEATURE_KINDS.index(kind), "window": window, "pixel_size_m": pixel_size_m, "map_seed": seed, "variant": name}
                scores[name] = _score_region(scorer, image, meta)
            clean_scores.append(scores["clean"])
            for f in faults:
                deltas[f].append(scores[f] - scores["clean"])
                by_kind[f].setdefault(kind, []).append(scores[f] - scores["clean"])
    result = {"mean_clean": float(np.mean(clean_scores)), "regions": len(clean_scores)}
    for f in faults:
        d = np.array(deltas[f])
        result[f] = {
            "mean_change": float(d.mean()),
            "fraction_lowered": float((d < 0).mean()),
            "fraction_lowered_by_kind": {k: float((np.array(v) < 0).mean()) for k, v in by_kind[f].items()},
        }
    return result


def evaluate(scorer: str | Scorer, seeds=DEFAULT_SEEDS, k: int = DEFAULT_K, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Evaluate a scorer by name (or instance) on the seeded synthetic dataset; a JSON-ready record.

    ``passed``: precision@k >= threshold on every seed, and the mean per-category precision@k >= threshold for
    every feature kind (overall precision alone can be met by one easy kind: baseline 2.0.0 reaches 0.99 overall
    while ranking crowded, small and large particles below defocused ones). Report held-out seeds (the default);
    seeds 0-2 were used for tuning.
    """
    s = resolve_scorer(scorer)
    seeds = [int(x) for x in seeds]
    per_seed = []
    for seed in seeds:
        p = precision_at_k(s, make_labelled_map(seed), k)
        per_seed.append({"seed": seed, **p, "sensitivity": sensitivity(s, seed)})
    precisions = [r["precision_at_k"] for r in per_seed]
    categories = list(per_seed[0]["per_category"])
    per_category_mean = {c: float(np.mean([r["per_category"][c] for r in per_seed])) for c in categories}
    faults = ("defocused", "contaminated", "empty")
    return {
        "scorer": s.name,
        "scorer_version": s.version,
        "dataset": DATASET_VERSION,
        "seeds": seeds,
        "held_out": all(x not in TUNING_SEEDS for x in seeds),
        "metric": f"precision@{k}",
        "threshold": threshold,
        "min_precision": min(precisions),
        "mean_precision": float(np.mean(precisions)),
        "share_of_seeds_passing": float(np.mean([x >= threshold for x in precisions])),
        "passed": min(precisions) >= threshold and all(v >= threshold for v in per_category_mean.values()),
        "per_category_mean_precision": per_category_mean,
        "per_category_min_precision": {c: float(min(r["per_category"][c] for r in per_seed)) for c in categories},
        "fraction_lowered": {f: float(np.mean([r["sensitivity"][f]["fraction_lowered"] for r in per_seed])) for f in faults},
        "mean_change": {f: float(np.mean([r["sensitivity"][f]["mean_change"] for r in per_seed])) for f in faults},
        "per_seed": per_seed,
        "claim": "detection proxy: finds sharp, clean particles of several kinds; not validated against human preference",
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
    }


__all__ = [
    "DEFAULT_THRESHOLD", "FEATURE_KINDS", "HELDOUT_SEEDS", "LabelledMap", "NEGATIVE_KINDS", "RandomScorer", "TUNING_SEEDS",
    "contaminate", "defocus", "evaluate", "make_labelled_map", "precision_at_k", "render_cell", "resolve_scorer", "sensitivity",
]
