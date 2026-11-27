"""Synthetic images shared by the image tests: scenes, tiles cut from them, and calibrated datasets.

Everything here is generated from a seed, so the tests need no data files and no microscope.
"""

from functools import lru_cache

import numpy as np
from scipy import ndimage

import sidpy

PIXEL = 1e-9  # metres per pixel of the synthetic scenes
TILE = 256  # side of a tile cut by make_tiles


def image_dataset(array, pixel_size_m=PIXEL, stage_xy_m=None, title='image', acquired_s=None):
    """A sidpy IMAGE dataset with spatial dimensions y (rows) and x (columns) in nm.

    stage_xy_m: the stage (x, y) of the image centre in metres, put in metadata['experiment']['stage_position'].
    """
    array = np.asarray(array)
    dataset = sidpy.Dataset.from_array(array, title=title)
    dataset.data_type = 'IMAGE'
    dataset.quantity = 'intensity'
    dataset.units = 'counts'
    pixel_nm = pixel_size_m * 1e9
    dataset.set_dimension(0, sidpy.Dimension(np.arange(array.shape[0]) * pixel_nm, name='y', units='nm',
                                             quantity='Length', dimension_type='spatial'))
    dataset.set_dimension(1, sidpy.Dimension(np.arange(array.shape[1]) * pixel_nm, name='x', units='nm',
                                             quantity='Length', dimension_type='spatial'))
    experiment = {}
    if stage_xy_m is not None:
        experiment['stage_position'] = [float(stage_xy_m[0]), float(stage_xy_m[1])]
    if acquired_s is not None:
        experiment['acquired_s'] = float(acquired_s)
    dataset.metadata['experiment'] = experiment
    return dataset


# ---------------------------------------------------------------- particles on a support

def particles(n=20, seed=0, size=128, blur=0.5):
    """Sharp discs of radius 3 px (barely blurred) on a noisy support; a 2-px blur is half way to defocus."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[:size, :size]
    image = np.zeros((size, size))
    for r, c in zip(rng.integers(0, size, n), rng.integers(0, size, n)):
        image[(rows - r) ** 2 + (cols - c) ** 2 <= 9] = 1.0
    return ndimage.gaussian_filter(image, blur) * 100 + 50 + rng.normal(0, 2, (size, size))


def gaussian_spots(shape=(256, 256), spots=((60, 80, 6.0, 40.0), (180, 150, 4.0, 25.0), (128, 230, 5.0, 2.0)), seed=1):
    """Gaussian particles (row, col, sigma px, height) on a background of 10 with noise of standard deviation 1."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    image = 10.0 + rng.normal(0.0, 1.0, shape)
    for row, col, sigma, height in spots:
        image += height * np.exp(-((yy - row) ** 2 + (xx - col) ** 2) / (2 * sigma**2))
    return image


# ---------------------------------------------------------------- a fine image and its coarse parent

STEP = 8  # parent pixels are STEP child pixels wide


def fine_scene(size=2400, seed=0):
    """Smooth texture with brighter blobs, seen at the child's (fine) pixel size."""
    rng = np.random.default_rng(seed)
    image = ndimage.gaussian_filter(rng.normal(0, 1, (size, size)), 6)
    blobs = np.zeros((size, size))
    blobs[rng.integers(0, size, 300), rng.integers(0, size, 300)] = 1.0
    return image / image.std() + 8 * ndimage.gaussian_filter(blobs, 10) / ndimage.gaussian_filter(blobs, 10).max()


def parent_of(scene, step=STEP):
    """What a coarse scan sees: the scene smoothed, then sampled at every step-th pixel centre."""
    smooth = ndimage.gaussian_filter(scene, 0.4 * step)
    centres = np.arange(step // 2, scene.shape[0], step) - 0.5
    rr, cc = np.meshgrid(centres, centres, indexing="ij")
    return ndimage.map_coordinates(smooth, [rr, cc], order=1)


# ---------------------------------------------------------------- tiles of a map

def scene(size=1300, blobs=500, seed=0):
    """Blurred blobs of random brightness on a noisy background."""
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size))
    rows, cols = rng.integers(0, size, blobs), rng.integers(0, size, blobs)
    image[rows, cols] = rng.uniform(50, 200, blobs)
    image = ndimage.gaussian_filter(image, 4) * 30 + 20
    return image + rng.normal(0, 1.0, image.shape)


def make_tiles(stage_error_px, overlap=0.2, grid=4, seed=1, reported_noise_px=0.0, image=None):
    """Tiles cut where the stage really was; positions reported where it was told to go.

    Returns the tiles and their true top-left (row, col) in the scene.
    """
    from pyTEMlib.image_tools.image_mosaic import Tile

    rng = np.random.default_rng(seed)
    image = scene() if image is None else image
    step = int(TILE * (1 - overlap))
    tiles, true_top_left = [], []
    for r in range(grid):
        for c in range(grid):
            commanded = np.array([40 + r * step, 40 + c * step], dtype=float)
            true = commanded + rng.normal(0, stage_error_px, 2)
            ti = np.clip(np.round(true).astype(int), 0, image.shape[0] - TILE)
            tiles.append(image[ti[0] : ti[0] + TILE, ti[1] : ti[1] + TILE].copy())
            true_top_left.append(ti.astype(float))
            reported = commanded + rng.normal(0, reported_noise_px, 2) if reported_noise_px else commanded
            centre_xy = ((reported[1] + TILE / 2) * PIXEL, (reported[0] + TILE / 2) * PIXEL)
            tiles[-1] = Tile(tiles[-1], centre_xy, PIXEL, index=len(tiles) - 1)
    return tiles, np.array(true_top_left)


def placement_error(positions, truth):
    """Per-tile error after removing the unobservable global offset."""
    diff = positions - truth
    diff -= np.median(diff, axis=0)
    return np.hypot(diff[:, 0], diff[:, 1])


def errors(placed, truth, subset=None) -> dict:
    """Median, 95th percentile and largest placement error in pixels, after removing the global offset."""
    e = placement_error(placed, truth)
    if subset is not None:
        e = e[subset]
    return {"median_px": round(float(np.median(e)), 3), "p95_px": round(float(np.percentile(e, 95)), 3), "max_px": round(float(e.max()), 3)}


class Scene:
    """Gaussian particles spread over a large map, rendered per tile through a grid (O(particles in the tile))."""

    def __init__(self, extent_px: float, density: float, seed: int):
        rng = np.random.default_rng(seed)
        count = int(density * extent_px * extent_px)
        self.xy = rng.uniform(-50, extent_px + 50, size=(count, 2))
        self.sigma = rng.uniform(1.5, 4.0, size=count)
        self.amp = rng.uniform(200, 900, size=count)
        self.cell = 64.0
        keys = np.floor(self.xy / self.cell).astype(np.int64)
        self.buckets: dict[tuple[int, int], np.ndarray] = {}
        order = np.lexsort((keys[:, 1], keys[:, 0]))
        keys = keys[order]
        starts = np.flatnonzero(np.r_[True, np.any(np.diff(keys, axis=0) != 0, axis=1)])
        ends = np.r_[starts[1:], len(keys)]
        for s, e in zip(starts, ends):
            self.buckets[(int(keys[s, 0]), int(keys[s, 1]))] = order[s:e]

    def render(self, top: float, left: float, size: int, rng: np.random.Generator) -> np.ndarray:
        rows = top + np.arange(size)
        cols = left + np.arange(size)
        image = np.full((size, size), 50.0)
        r0, r1 = int(np.floor((top - 12) / self.cell)), int(np.floor((top + size + 12) / self.cell))
        c0, c1 = int(np.floor((left - 12) / self.cell)), int(np.floor((left + size + 12) / self.cell))
        for a in range(r0, r1 + 1):
            for b in range(c0, c1 + 1):
                for k in self.buckets.get((a, b), ()):
                    y, x = self.xy[k]
                    gy = np.exp(-0.5 * ((rows - y) / self.sigma[k]) ** 2)
                    gx = np.exp(-0.5 * ((cols - x) / self.sigma[k]) ** 2)
                    image += self.amp[k] * np.outer(gy, gx)
        return rng.poisson(image).astype(np.uint16)


class SyntheticRun:
    """A serpentine run of ``side`` x ``side`` tiles with stage error and sample drift, rendered on demand.

    ``truth`` is each tile's true top-left (row, col), ``reported`` where the stage said it was (the truth
    off by the readout error and the drift since the start), ``drift`` the sample drift and ``times``
    the acquisition times (one tile per second).
    """

    def __init__(self, side: int, tile: int, overlap: float, stage_noise: float, drift_px_per_tile: float, seed: int, density: float = 8e-4):
        step = tile * (1 - overlap)
        self.tile = tile
        rng = np.random.default_rng(seed + 1)
        order = []
        for r in range(side):
            cols = range(side) if r % 2 == 0 else reversed(range(side))
            order += [(r, c) for c in cols]
        n = len(order)
        self.commanded = np.array([[r * step, c * step] for r, c in order], dtype=np.float64)
        self.times = np.arange(n, dtype=np.float64)
        walk = np.cumsum(rng.normal(0, 0.15 * drift_px_per_tile, size=(n, 2)), axis=0)
        self.drift = np.outer(self.times, [drift_px_per_tile, 0.6 * drift_px_per_tile]) + walk
        self.truth = self.commanded + rng.normal(0, 0.05 * tile, size=(n, 2))
        self.reported = self.truth + self.drift + rng.normal(0, stage_noise * tile, size=(n, 2))
        extent = side * step + tile + abs(self.drift).max() + 100
        self.scene = Scene(extent, density=density, seed=seed)
        self.seed = seed

    def __len__(self):
        return len(self.truth)

    @lru_cache(maxsize=4096)
    def image(self, k: int) -> np.ndarray:
        top, left = self.truth[k]
        return self.scene.render(top, left, self.tile, np.random.default_rng(self.seed * 1_000_003 + k))

    def __getitem__(self, k):
        return SyntheticTile(self, k)


class SyntheticTile:
    """One tile of a SyntheticRun, shaped like image_mosaic.Tile; its image is rendered when read."""

    def __init__(self, run: SyntheticRun, k: int):
        self.run, self.index = run, k
        top, left = run.reported[k]
        half = run.tile / 2
        self.stage_xy_m = ((left + half) * PIXEL, (top + half) * PIXEL)
        self.pixel_size_m = PIXEL
        self.attrs = {}
        self.acquired_s = float(run.times[k])
        self.shape = (run.tile, run.tile)

    @property
    def image(self):
        return self.run.image(self.index)
