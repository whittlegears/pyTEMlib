# Field-of-view scorers and their evaluation

A scorer answers "how good is this region of a map as a field of view?" with a
value in [0, 1], a short reason and its own version. `score_map` (the MCP
analysis server) ranks map windows with any registered scorer and stores the
best as points of interest, each with its `scorer_version`.

## The baseline scorer (3.0.0)

`baseline` scores focus per feature, not over the whole region. It finds edge
points (ridges of the smoothed gradient at least 5 x the local noise), groups
them into features (connected edge contours), and measures each feature's edge
width from how fast the gradient falls off on the darker side of its edge.
That ratio does not depend on the step height, the feature size or how much
of the region features cover. Faint, short edges are judged by the blurred end
of their uncertainty, because the noise draws that get detected look sharp.
The region's edge width is the median over features, each weighted by its
edge points (capped) and reliability; score = sharpness x edge contrast, with
a reason such as "4 features, 4 with sharp edges; median edge width 1.35 px".

2.0.0 (one global edge width from the 99th-percentile height over the steepest
gradient) is still selectable as `baseline_v2`, for comparison only.

## What the evaluation measures (and what it does not)

`pyTEMlib.scoring.evaluate` scores a scorer against labelled maps: maps where
the particle positions are known and some regions were deliberately spoiled.
Each synthetic map (`synthetic-fov-2`) is a 10 x 10 grid of 96-pixel regions:

* 30 clean feature regions, 5 of each kind: `separated` (the twin's 2-5
  particles of ~6 px radius), `crowded` (8-14 particles of 4-14 px that may
  overlap), `small` (one particle of 2-4.5 px, under 1 % of the region),
  `large` (one or two of 16-32 px), `sampling` (the same 1.6 nm particles at a
  pixel size that makes them 2-40 px in radius) and `low_contrast` (sharp
  particles whose peak is 4-8 x the background noise);
* 70 negatives: 24 empty, 23 defocused (Gaussian blur of 3-5 px) and 23
  contaminated (lumpy carbon layer that also spreads the beam). Each spoiled
  region starts from the layout of a randomly chosen clean kind.

Numbers:

* **precision@5**: of the 5 regions a scorer ranks highest, the fraction that
  are clean features; and **per kind**, the same over that kind's 5 regions
  plus the 70 negatives.
* **sensitivity**: clean regions of every kind are re-rendered with the same
  particles and noise draw, defocused, contaminated and emptied; the share of
  each that lowers the score, and by how much.

Seeds 0-2 are for tuning; seeds 3-29 are held out and are what is reported
(the bench's default). Pass mark (proposed for D6, to be agreed with WG2):
precision@5 >= 0.8 on every held-out seed **and** mean per-kind precision@5
>= 0.8 for every kind. A random ranking reaches 0.8 on one map with
probability 2.7 % (30 of 100), and for one kind with probability 2e-5.

**This is a detection proxy.** It shows that a scorer finds sharp, clean
particles of several sizes, densities and contrasts and is pushed down by
defocus, contamination and empty support. It is not validated against human
preference: there are no microscopist labels yet. Operator picks at first
light are meant to become the first ones.

The maps are generated in-process from a seed (`make_labelled_map`), with
an image model that follows the asyncroscopy digital twin (ellipsoidal
Au/Pt/Fe particles on carbon, HAADF ~ Z^1.7 x thickness, Poisson counts). The
twin itself is not imported. To evaluate on twin images, build a
`LabelledMap.from_twin(image, twin.export_ground_truth(...), origin_m, pixel_size_m, window, bad_regions)`.

## Results so far

Held-out seeds 3-29 of `synthetic-fov-2` (27 maps):

| scorer | precision@5 mean (min) | separated | crowded | small | large | sampling | low contrast | defocus / contamination / empty lowers score |
|---|---|---|---|---|---|---|---|---|
| baseline 3.0.0 (per-feature edge width) | 1.00 (1.0) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 (min 0.8) | 100 / 100 / 100 %, mean -0.78 / -0.78 / -0.79 |
| baseline_v2 2.0.0 (global edge width) | 0.99 (0.8) | 0.99 | 0.16 | 0.02 | 0.00 | 0.39 | 0.34 | 82 / 80 / 64 %, mean -0.23 / -0.17 / -0.24 |
| random | 0.27 (0.0) | 0.04 | 0.07 | 0.04 | 0.05 | 0.06 | 0.05 | 53 / 48 / 52 % |

Note the 2.0.0 row: overall precision@5 alone looks fine because its top 5 are
always well-separated particles; only the per-kind numbers show it ranking
crowded, small and large particles below defocused and contaminated regions.
On the earlier, single-kind set (`synthetic-fov-1`, seeds 0-2) 1.0.0 scored
0.0 and 2.0.0 1.0; those records are kept in `benchmarks/results/reward_eval.jsonl`
with the new ones.

Probes from an independent review (`tests/test_reward_eval.py`), score of the
sharp version / the same defocused by 4 px:

| probe | 2.0.0 | 3.0.0 |
|---|---|---|
| one sharp particle, radius 10 px | 0.59-0.66 / 0.00 | 0.88-0.92 / 0.00 |
| five overlapping sharp disks, radius 25 px | 0.00 / 0.00 | 0.76-0.86 / 0.00 |
| one small particle, 0.7 % of the region | 0.16-0.18 / 0.16-0.17 | 0.56-0.68 / 0.00 |

**Honesty note.** 3.0.0's constants (edge midpoint 1.8 px, uncertainty
margin 1.5 standard errors) were chosen on seeds 0-2 only, where many settings
tie at 1.0; the held-out numbers above come from the same generator, so they
show that the scorer generalises within this synthetic family, not to real
data. The family was widened after a review found that 2.0.0, which had also
scored 1.0 here on one particle look, was biased against crowded and small
particles; other biases the generator cannot produce (real defocus with
Fresnel fringes, astigmatism, drift, scan noise, hot pixels, supports with
lattice fringes) are untested. Low-contrast regions are the weakest kind: on
4 of 27 held-out maps one spoiled region outranks one of them, each time a
defocused or contaminated low-contrast layout with a single short edge that
noise makes look sharp. Edge widths are in pixels: a finite probe sampled much
finer than its size would read as blurred, so features must be scored at a
level where the probe is about a pixel (`score_map` chooses the level).
Defocus is modelled as a Gaussian blur and contamination as a lumpy carbon
layer; regions are a fixed grid, while `score_map` ranks half-overlapping
windows.

## Benchmark your own scorer

```
python -m pyTEMlib.scoring.bench --scorer baseline
python -m pyTEMlib.scoring.bench --scorer random
python -m pyTEMlib.scoring.bench --scorer baseline_v2
python -m pyTEMlib.scoring.bench --scorer my_scorers:BlobScorer --out benchmarks/results/reward_eval.jsonl
```

`--scorer` takes a registered name or `module:attribute`. The bench prints
JSON with `scorer_version`, the seeds, precision@5 overall and per kind, and
the sensitivity, and exits 1 below the pass mark. An example scorer, `my_scorers.py`:

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
