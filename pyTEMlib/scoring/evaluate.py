"""
Evaluate a field-of-view scorer against labelled maps.

What this measures, honestly: the labelled maps know where the particles are
and which regions were deliberately spoiled (defocused, contaminated). A
scorer passes if its top regions are the ones that hold sharp, clean
particles. That is a **detection proxy** for "a good field of view": it checks
that the scorer finds visible features and is pushed down by the two image
faults we can simulate. It has not been checked against what a microscopist
would pick; there are no human-preference labels yet (operator picks at first
light are planned as the first ones).

Two numbers per scorer:

* ``precision@k`` (k = 5): of the k regions the scorer ranks highest on a map,
  the fraction that are clean feature regions. Regions are the cells of a
  grid of ``window`` pixels, so every ranked region has exactly one label.
  ``score_map`` ranks half-overlapping windows instead; the grid keeps the
  labels unambiguous.
* sensitivity: every clean feature region is re-rendered with the same
  particles and the same noise draw, once defocused and once contaminated;
  the check reports how often, and by how much, each fault lowers the score.

The default dataset is synthetic and generated in-process from a seed, so the
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

DATASET_VERSION = "synthetic-fov-1"
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_K = 5
# Proposed D6 threshold (to be agreed with WG2): at least 4 of the top 5 regions clean features,
# on every seed. With 16 of 64 regions clean features, a scorer that ranks at random reaches
# 4 of 5 with probability 1.6 % per map (hypergeometric), so ~4e-6 on all three seeds.
DEFAULT_THRESHOLD = 0.8

# HAADF signal: counts per (Z^1.7 x atoms per cubic angstrom x nm): 3 nm of carbon support gives
# ~16 counts per pixel, the centre of a typical Au particle (~1.3 nm thick) ~330 and of Fe ~70.
_ELEMENTS = {"C": (6, 0.100), "Fe": (26, 0.085), "Pt": (78, 0.066), "Au": (79, 0.059)}
_COUNTS = 2.6


def _signal_per_nm(element: str) -> float:
    z, density = _ELEMENTS[element]
    return _COUNTS * z**1.7 * density


@dataclass
class LabelledMap:
    """A map image with its particles and known bad regions, both in pixels of ``image``.

    particles: dicts with ``center_px`` (row, col); other keys are carried along.
    bad_regions: dicts with ``kind`` ("defocused", "contaminated", ...) and ``box_px``
    (row0, col0, row1, col1), end-exclusive.
    """

    image: np.ndarray
    window: int
    particles: list[dict]
    bad_regions: list[dict] = field(default_factory=list)
    pixel_size_m: float | None = None
    seed: int | None = None

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
        """Grid regions (row, col top-left) with their label: "feature", "empty" or the bad kind."""
        h, w = self.image.shape[:2]
        out = []
        for r in range(0, h - self.window + 1, self.window):
            for c in range(0, w - self.window + 1, self.window):
                box = (r, c, r + self.window, c + self.window)
                bad = next((b["kind"] for b in self.bad_regions if _overlap(box, b["box_px"])), None)
                has_feature = any(r <= p["center_px"][0] < box[2] and c <= p["center_px"][1] < box[3] for p in self.particles)
                out.append({"row": r, "col": c, "label": bad or ("feature" if has_feature else "empty")})
        return out


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# ---------------------------------------------------------------- synthetic labelled maps

CLASSES = ("feature", "empty", "defocused", "contaminated")


def _particles_in_cell(rng, origin, window, count) -> list[dict]:
    out = []
    for _ in range(50 * count):
        if len(out) >= count:
            break
        a = float(np.clip(rng.normal(6.4, 0.8), 4.0, 9.0))  # semi-axis in pixels: twin's 1.6 +- 0.2 nm at 0.25 nm per pixel
        margin = a + 3
        centre = origin + rng.uniform(margin, window - margin, 2)
        if all(np.hypot(*(centre - np.array(p["center_px"]))) > a + p["semi_axes_px"][0] + 2 for p in out):
            out.append({
                "center_px": (float(centre[0]), float(centre[1])),
                "semi_axes_px": (a, a * float(rng.uniform(0.8, 1.2)), 0.4 * a),
                "theta_rad": float(rng.uniform(0, np.pi)),
                "element": str(rng.choice(["Au", "Pt", "Fe"], p=[0.4, 0.4, 0.2])),
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


def make_labelled_map(seed: int, grid: int = 8, window: int = 64, pixel_size_m: float = 0.25e-9, support_nm: float = 3.0) -> LabelledMap:
    """A grid x grid map with equal numbers of clean-feature, empty, defocused and contaminated regions."""
    rng = np.random.default_rng(seed)
    pixel_nm = pixel_size_m * 1e9
    n = grid * grid
    labels = [CLASSES[i] for i in rng.permutation(np.arange(n) % len(CLASSES))]
    support = support_nm * _signal_per_nm("C") * (1 + 0.05 * ndimage.gaussian_filter(rng.normal(size=(grid * window,) * 2), 20))
    image = np.array(support)
    particles, bad = [], []
    for i, label in enumerate(labels):
        r0, c0 = (i // grid) * window, (i % grid) * window
        cell = np.s_[r0 : r0 + window, c0 : c0 + window]
        if label == "empty":
            continue
        found = _particles_in_cell(rng, np.array([r0, c0], float), window, int(rng.integers(2, 6)))
        local = [{**p, "center_px": (p["center_px"][0] - r0, p["center_px"][1] - c0)} for p in found]
        signal = _render_particles((window, window), local, pixel_nm)
        if label == "defocused":
            signal = defocus(signal, float(rng.uniform(3.0, 5.0)))
        elif label == "contaminated":
            signal = contaminate(signal, rng, float(rng.uniform(8.0, 16.0)))
        image[cell] += signal
        particles += found
        if label != "feature":
            bad.append({"kind": label, "box_px": (r0, c0, r0 + window, c0 + window)})
    image = rng.poisson(np.clip(image, 0, None)).astype(np.float64)
    return LabelledMap(image, window, particles, bad, pixel_size_m, seed)


# ---------------------------------------------------------------- metrics


class RandomScorer:
    """Reference scorer: a seeded random number per region, ignoring the image."""

    name = "random"
    version = "1.0.0"
    description = "Uniform random value; the chance level any real scorer must beat."

    def __init__(self, seed: int = 0):
        self.seed = seed

    def score(self, region: np.ndarray, metadata: dict) -> tuple[float, str]:
        key = f"{self.seed}:{metadata.get('row')}:{metadata.get('col')}:{metadata.get('map_seed')}".encode()
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


def precision_at_k(scorer: Scorer, labelled: LabelledMap, k: int = DEFAULT_K) -> dict:
    """Fraction of the scorer's k highest-ranked grid regions that are clean feature regions.

    Ties are broken by position (row, then column), never in the scorer's favour.
    """
    w = labelled.window
    ranked = []
    for region in labelled.regions():
        meta = {"row": region["row"], "col": region["col"], "window": w, "pixel_size_m": labelled.pixel_size_m, "map_seed": labelled.seed}
        value = _score_region(scorer, labelled.image[region["row"] : region["row"] + w, region["col"] : region["col"] + w], meta)
        ranked.append({**region, "value": value})
    ranked.sort(key=lambda s: (-s["value"], s["row"], s["col"]))
    top = ranked[:k]
    hits = sum(s["label"] == "feature" for s in top)
    return {
        "precision_at_k": hits / k,
        "k": k,
        "top_labels": [s["label"] for s in top],
        "chance": sum(s["label"] == "feature" for s in ranked) / len(ranked),
    }


def sensitivity(scorer: Scorer, seed: int, n_regions: int = 16, window: int = 64, pixel_size_m: float = 0.25e-9) -> dict:
    """Score clean feature regions and the same regions defocused or contaminated (same particles, same noise draw)."""
    rng = np.random.default_rng([seed, 7])
    pixel_nm = pixel_size_m * 1e9
    support = 3.0 * _signal_per_nm("C")
    out = {"defocused": [], "contaminated": []}
    clean_scores = []
    for i in range(n_regions):
        particles = _particles_in_cell(rng, np.zeros(2), window, int(rng.integers(2, 6)))
        signal = _render_particles((window, window), particles, pixel_nm)
        variants = {
            "clean": signal,
            "defocused": defocus(signal, float(rng.uniform(3.0, 5.0))),
            "contaminated": contaminate(signal, rng, float(rng.uniform(8.0, 16.0))),
        }
        noise_seed = int(rng.integers(2**31))
        scores = {}
        for name, s in variants.items():
            image = np.random.default_rng(noise_seed).poisson(support + s).astype(np.float64)
            scores[name] = _score_region(scorer, image, {"row": i, "col": 0, "window": window, "pixel_size_m": pixel_size_m, "map_seed": seed})
        clean_scores.append(scores["clean"])
        for name in out:
            out[name].append(scores[name] - scores["clean"])
    result = {"mean_clean": float(np.mean(clean_scores))}
    for name, deltas in out.items():
        deltas = np.array(deltas)
        result[name] = {"mean_change": float(deltas.mean()), "fraction_lowered": float((deltas < 0).mean())}
    return result


def evaluate(scorer: str | Scorer, seeds=DEFAULT_SEEDS, k: int = DEFAULT_K, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Evaluate a scorer by name (or instance) on the seeded synthetic dataset; a JSON-ready record."""
    s = resolve_scorer(scorer)
    per_seed = []
    for seed in seeds:
        p = precision_at_k(s, make_labelled_map(seed), k)
        per_seed.append({"seed": int(seed), **p, "sensitivity": sensitivity(s, seed)})
    precisions = [r["precision_at_k"] for r in per_seed]
    return {
        "scorer": s.name,
        "scorer_version": s.version,
        "dataset": DATASET_VERSION,
        "seeds": [int(x) for x in seeds],
        "metric": f"precision@{k}",
        "threshold": threshold,
        "min_precision": min(precisions),
        "mean_precision": float(np.mean(precisions)),
        "passed": min(precisions) >= threshold,
        "per_seed": per_seed,
        "claim": "detection proxy: finds sharp, clean particles; not validated against human preference",
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
    }


__all__ = ["DEFAULT_THRESHOLD", "LabelledMap", "RandomScorer", "contaminate", "defocus", "evaluate", "make_labelled_map", "precision_at_k", "resolve_scorer", "sensitivity"]
