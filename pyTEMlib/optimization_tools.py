"""
Measurement-driven optimisation and trend statistics

Instrument-independent pieces of the autofocus / stage-z / aberration
hill-climbing notebooks of asyncroscopy, generalised to any number of
parameters: a ridge-regularised quadratic response surface, a trust-region
proposer that never repeats a measured point, and an ask / tell optimiser.
The caller owns the instrument: it sets the parameters, measures a score
(e.g. pyTEMlib.image_tools.image_metrics) and reports it back.

Also contains the parabola-vertex helper of the defocus-series notebook and
the linear-trend-with-confidence-interval helper of the lattice-profile
notebook.

part of pyTEMlib
"""
import itertools

import numpy as np
from scipy import stats


def _quadratic_design(normalized):
    count, dims = normalized.shape
    columns = [np.ones(count)]
    columns += [normalized[:, i] for i in range(dims)]
    columns += [normalized[:, i] * normalized[:, j]
                for i, j in itertools.combinations_with_replacement(range(dims), 2)]
    return np.column_stack(columns)


def fit_quadratic_response(points, scores, ridge=1e-6):
    """Fit a full quadratic surface score(points) in any number of dimensions.

    Parameters
    ----------
    points: array_like (n, d) or (n,)
        measured parameter settings
    scores: array_like (n,)
    ridge: float
        Tikhonov regularisation; keeps under-determined fits stable

    Returns
    -------
    callable mapping candidate points (m, d) or (m,) to predicted scores (m,)
    """
    points = np.asarray(points, dtype=np.float64)
    points = points[:, None] if points.ndim == 1 else points
    scores = np.asarray(scores, dtype=np.float64)
    center = points.mean(axis=0)
    scale = np.maximum(points.std(axis=0), 1e-12 * np.maximum(np.abs(center), 1.0))
    design = _quadratic_design((points - center) / scale)
    coefficients = np.linalg.solve(design.T @ design + ridge * np.eye(design.shape[1]), design.T @ scores)

    def predict(candidates):
        candidates = np.asarray(candidates, dtype=np.float64)
        candidates = candidates.reshape(-1, points.shape[1])
        return _quadratic_design((candidates - center) / scale) @ coefficients

    return predict


def _candidates_in_ball(center, radius, dims, rng, count):
    if dims == 1:
        return np.linspace(center[0] - radius[0], center[0] + radius[0], count)[:, None]
    if dims == 2:
        side = int(np.sqrt(count * 4 / np.pi)) + 1
        axis = np.linspace(-1, 1, side)
        grid = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
        unit = grid[np.linalg.norm(grid, axis=1) <= 1]
    else:
        direction = rng.normal(size=(count, dims))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        unit = direction * rng.random((count, 1)) ** (1 / dims)
        unit = np.vstack([unit, np.eye(dims), -np.eye(dims)])
    return center + unit * radius


def propose_next_point(points, scores, trust_radius, bounds=None, maximize=True,
                       candidate_count=401, min_separation_fraction=1 / 40, seed=0):
    """Next setting to measure: the best predicted point within the trust region.

    The trust region is a ball (scaled per dimension by ``trust_radius``)
    around the best measured point. Candidates closer than
    ``min_separation_fraction * trust_radius`` to a measured point are skipped.

    Parameters
    ----------
    points: array_like (n, d) or (n,)
    scores: array_like (n,)
    trust_radius: float or array_like (d,)
    bounds: array_like (d, 2) or (2,) or None
        hard lower / upper limits of every parameter
    maximize: bool
    candidate_count: int
    min_separation_fraction: float
    seed: int
        random seed for the candidate cloud in more than two dimensions

    Returns
    -------
    numpy.ndarray (d,) or None when every candidate was already measured
    """
    points = np.asarray(points, dtype=np.float64)
    one_dimensional = points.ndim == 1
    points = points[:, None] if one_dimensional else points
    scores = np.asarray(scores, dtype=np.float64)
    sign = 1.0 if maximize else -1.0
    dims = points.shape[1]
    radius = np.broadcast_to(np.asarray(trust_radius, dtype=np.float64), (dims,))
    best = points[int(np.argmax(sign * scores))]
    candidates = _candidates_in_ball(best, radius, dims, np.random.default_rng(seed), candidate_count)
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=np.float64).reshape(dims, 2)
        candidates = candidates[np.all((candidates >= bounds[:, 0]) & (candidates <= bounds[:, 1]), axis=1)]
    scaled_distance = np.linalg.norm((candidates[:, None, :] - points[None, :, :]) / radius, axis=2)
    candidates = candidates[scaled_distance.min(axis=1) > min_separation_fraction]
    if not len(candidates):
        return None
    predict = fit_quadratic_response(points, scores)
    choice = candidates[int(np.argmax(sign * predict(candidates)))]
    return choice


class TrustRegionOptimizer:
    """Ask / tell optimiser of a noisy, expensive score (focus, stigmation, stage z, ...).

    Usage::

        optimizer = TrustRegionOptimizer(start=[0.0], step=[5e-9], bounds=[[-1e-6, 1e-6]])
        while (setting := optimizer.ask()) is not None:
            instrument.set(setting)
            optimizer.tell(setting, metric(instrument.acquire()))
        best_setting, best_score = optimizer.best()

    The first 1 + 2d asks are the start point and +/- one step along each
    axis; then quadratic-model proposals follow. A proposal that improves the
    best score by more than ``improvement_tolerance`` (relative) grows the
    trust radius by ``grow``, otherwise it shrinks by ``shrink``. The search
    stops after ``max_iterations`` model proposals or when the radius falls
    below ``min_step``.
    """

    def __init__(self, start, step, bounds=None, maximize=True, max_iterations=8, min_step=None,
                 max_step=None, grow=1.2, shrink=0.65, improvement_tolerance=1e-3):
        self.start = np.atleast_1d(np.asarray(start, dtype=np.float64))
        dims = len(self.start)
        self.step = np.broadcast_to(np.asarray(step, dtype=np.float64), (dims,)).copy()
        if np.any(self.step <= 0):
            raise ValueError('step must be positive')
        self.bounds = None if bounds is None else np.asarray(bounds, dtype=np.float64).reshape(dims, 2)
        self.maximize = bool(maximize)
        self.max_iterations = int(max_iterations)
        self.min_step = self.step / 16 if min_step is None else np.broadcast_to(np.asarray(min_step, float), (dims,))
        self.max_step = self.step * 2 if max_step is None else np.broadcast_to(np.asarray(max_step, float), (dims,))
        self.grow, self.shrink = float(grow), float(shrink)
        self.improvement_tolerance = float(improvement_tolerance)
        self.trust_radius = self.step.copy()
        self.points, self.scores = [], []
        self.iterations = 0
        self._initial = [self.start] + [self.start + sign * self.step * np.eye(dims)[i]
                                        for i in range(dims) for sign in (-1, 1)]
        if self.bounds is not None:
            self._initial = [point for point in self._initial
                             if np.all((point >= self.bounds[:, 0]) & (point <= self.bounds[:, 1]))]
        self._pending_model_step = False
        self.finished = False

    def _signed(self, score):
        return score if self.maximize else -score

    def ask(self):
        """Next parameter setting to measure, or None when the search is finished."""
        if self.finished:
            return None
        if len(self.points) < len(self._initial):
            return self._initial[len(self.points)].copy()
        if self.iterations >= self.max_iterations or np.all(self.trust_radius < self.min_step):
            self.finished = True
            return None
        proposal = propose_next_point(np.array(self.points), np.array(self.scores), self.trust_radius,
                                      self.bounds, self.maximize)
        if proposal is None:
            self.finished = True
            return None
        self._pending_model_step = True
        return proposal

    def tell(self, point, score):
        """Report the measured score of a setting returned by :meth:`ask`."""
        score = float(score)
        if self._pending_model_step:
            previous = max(self._signed(value) for value in self.scores)
            improved = self._signed(score) > previous + self.improvement_tolerance * abs(previous)
            if improved:
                self.trust_radius = np.minimum(self.trust_radius * self.grow, self.max_step)
            else:
                self.trust_radius = self.trust_radius * self.shrink
            self.iterations += 1
            self._pending_model_step = False
        self.points.append(np.atleast_1d(np.asarray(point, dtype=np.float64)))
        self.scores.append(score)

    def best(self):
        """(best setting, best score) measured so far."""
        if not self.scores:
            raise ValueError('nothing measured yet')
        index = int(np.argmax([self._signed(value) for value in self.scores]))
        return self.points[index].copy(), self.scores[index]

    def history(self):
        """List of dicts with point, score and whether it is the current best."""
        best_index = int(np.argmax([self._signed(value) for value in self.scores])) if self.scores else -1
        return [{'point': point.tolist(), 'score': score, 'best': index == best_index}
                for index, (point, score) in enumerate(zip(self.points, self.scores))]


def parabola_vertex(x, y):
    """Fit y = a x^2 + b x + c and return the vertex.

    Returns
    -------
    dict with vertex_x, vertex_y, curvature (a) and coefficients (a, b, c);
    curvature < 0 means the vertex is a maximum
    """
    a, b, c = np.polyfit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), 2)
    if a == 0:
        raise ValueError('data are linear, no vertex')
    vertex_x = -b / (2 * a)
    return {'vertex_x': float(vertex_x), 'vertex_y': float(np.polyval([a, b, c], vertex_x)),
            'curvature': float(a), 'coefficients': [float(a), float(b), float(c)]}


def smoothstep(x):
    """Cubic smoothstep 3x^2 - 2x^3 of x clipped to [0, 1]."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0, 1)
    return x * x * (3 - 2 * x)


def linear_trend(x, y, confidence=0.95, transform=None):
    """Least-squares slope of y versus x with a Student-t confidence interval.

    Parameters
    ----------
    x, y: array_like
    confidence: float
    transform: callable or None
        applied to x before fitting (e.g. :func:`smoothstep` for a centre-to-edge profile)

    Returns
    -------
    dict with slope, intercept, slope_ci_low, slope_ci_high (NaN with fewer than 3 points)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if transform is not None:
        x = transform(x)
    design = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    dof = len(y) - 2
    if dof <= 0:
        return {'slope': float(beta[1]), 'intercept': float(beta[0]),
                'slope_ci_low': float('nan'), 'slope_ci_high': float('nan')}
    variance = np.sum((y - design @ beta) ** 2) / dof
    standard_error = np.sqrt(variance * np.linalg.inv(design.T @ design)[1, 1])
    critical = stats.t.ppf(0.5 + confidence / 2, dof)
    return {'slope': float(beta[1]), 'intercept': float(beta[0]),
            'slope_ci_low': float(beta[1] - critical * standard_error),
            'slope_ci_high': float(beta[1] + critical * standard_error)}
