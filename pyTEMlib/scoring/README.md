# Field-of-view scorers and their evaluation

A scorer answers "how good is this region of a map as a field of view?" with a
value in [0, 1], a short reason and its own version. `score_map` (the MCP
analysis server) ranks map windows with any registered scorer and stores the
best as points of interest, each with its `scorer_version`.

## What the evaluation measures (and what it does not)

`pyTEMlib.scoring.evaluate` scores a scorer against labelled maps: maps where
the particle positions are known and some regions were deliberately spoiled.

* **precision@5**: of the 5 grid regions a scorer ranks highest, the fraction
  that hold clean, in-focus particles. Each synthetic map is an 8 x 8 grid of
  64-pixel regions, 16 of each kind: clean features, empty support,
  defocused particles, contaminated particles.
* **sensitivity**: the same particles under the same noise draw are scored
  clean, defocused and contaminated. A good scorer drops on both faults.

Pass mark (proposed for D6, to be agreed with WG2): precision@5 >= 0.8 on each
of seeds 0, 1 and 2. A random ranking reaches 0.8 on one map with probability
about 1.6 % (hypergeometric, 16 of 64), so about 4 in a million on all three.

**This is a detection proxy.** It shows that a scorer finds sharp, clean
particles and is pushed down by defocus and contamination. It is not
validated against human preference: there are no microscopist labels yet.
Operator picks at first light are meant to become the first ones.

The maps are generated in-process from a seed (`make_labelled_map`), with
an image model that follows the asyncroscopy digital twin (ellipsoidal
Au/Pt/Fe particles on carbon, HAADF ~ Z^1.7 x thickness, Poisson counts). The
twin itself is not imported. To evaluate on twin images, build a
`LabelledMap.from_twin(image, twin.export_ground_truth(...), origin_m, pixel_size_m, window, bad_regions)`.

Limits: defocus is modelled as a Gaussian blur and contamination as a lumpy
carbon layer that also spreads the beam. Regions are a fixed grid, while
`score_map` ranks half-overlapping windows. The synthetic set is easy: a
scorer that passes here can still fail on real data.

## Results so far

| scorer | version | precision@5, seeds 0 / 1 / 2 | defocus lowers score | contamination lowers score |
|---|---|---|---|---|
| baseline | 1.0.0 (edge coverage) | 0.0 / 0.0 / 0.0 | 0-6 % of regions (score rose) | 6-25 % of regions |
| baseline | 2.0.0 (edge width) | 1.0 / 1.0 / 1.0 | 100 %, mean -0.93 | 100 %, mean -0.85 |
| random | reference | 0.2 / 0.0 / 0.2 | - | - |

Records with timestamps are in `benchmarks/results/reward_eval.jsonl`. The
edge-width midpoint of baseline 2.0.0 was set while looking at seeds 0-2; seeds
3-29 of the same generator also give 1.0, which shows only that it generalises
within this synthetic set.

## Benchmark your own scorer

```
python -m pyTEMlib.scoring.bench --scorer baseline
python -m pyTEMlib.scoring.bench --scorer random
python -m pyTEMlib.scoring.bench --scorer my_scorers:BlobScorer --out benchmarks/results/reward_eval.jsonl
```

`--scorer` takes a registered name or `module:attribute`. The bench prints
JSON with `scorer_version`, the seeds and precision@5, and exits 1 below the
threshold. An example scorer, `my_scorers.py`:

```python
"""Count compact bright blobs that are in focus."""
import numpy as np
from scipy import ndimage


class BlobScorer:
    name = "blobs"
    version = "0.1.0"
    description = "Number of sharp bright blobs, saturating at 3."

    def score(self, region, metadata):
        data = np.asarray(region, dtype=float)
        smooth = ndimage.gaussian_filter(data, 1.0)
        background = np.median(smooth)
        noise = np.std(data - smooth) + 1e-9
        mask = smooth > background + 8 * noise
        labels, n = ndimage.label(mask)
        sharp = 0
        for index in range(1, n + 1):
            blob = labels == index
            if blob.sum() < 9:
                continue  # a few hot pixels, not a particle
            height = smooth[blob].max() - background
            ring = ndimage.binary_dilation(blob, iterations=2) & ~blob
            steepest = np.hypot(*np.gradient(smooth))[ring | blob].max()
            if height / steepest < 4.8:  # edge width in pixels
                sharp += 1
        return min(sharp / 3, 1.0), f"{sharp} sharp blobs of {n}"
```

To make it available to `score_map` and `list_scorers`, register it from
your package:

```toml
[project.entry-points."pytemlib.scorers"]
blobs = "my_scorers:BlobScorer"
```
