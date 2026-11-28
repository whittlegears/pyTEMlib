"""
Bayesian optimisation of measured scores (optional: needs pyTEMlib[ml])

Gaussian-process models of settings -> scores, and the next setting to try:
one score with log expected improvement or an upper confidence bound, or several
scores at once with expected hypervolume improvement. Moved from the asyncroscopy
notebook AberrationsBO.ipynb, where it tunes C1 and A1 on image contrast and an
FFT score. Nothing here touches an instrument; the caller measures.

part of pyTEMlib
"""
import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.multi_objective import qLogExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from gpytorch.mlls import ExactMarginalLogLikelihood

dtype = torch.float64


class SuggestionError(ValueError):
    pass


def _bounds(bounds, dimensions: int) -> torch.Tensor:
    array = np.asarray(bounds, dtype=float).reshape(dimensions, 2)
    if np.any(array[:, 0] >= array[:, 1]):
        raise SuggestionError("every bound must be [low, high] with low < high")
    return torch.tensor(array.T, dtype=dtype)


def _points(points, minimum: int) -> torch.Tensor:
    x = torch.tensor(np.asarray(points, dtype=float), dtype=dtype)
    x = x[:, None] if x.ndim == 1 else x
    if x.shape[0] < minimum:
        raise SuggestionError(f"at least {minimum} measured points are needed, got {x.shape[0]}")
    return x


def _fit(x: torch.Tensor, y: torch.Tensor, bounds: torch.Tensor) -> SingleTaskGP:
    model = SingleTaskGP(x, y, input_transform=Normalize(d=x.shape[-1], bounds=bounds),
                         outcome_transform=Standardize(m=y.shape[-1]))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def suggest_setting(points, scores, bounds, maximize=True, acquisition="log_ei", beta=2.0, seed=0,
                    num_restarts=10, raw_samples=256) -> dict:
    """Next setting for one score from a Gaussian-process model of all measurements."""
    x = _points(points, 2)
    sign = 1.0 if maximize else -1.0
    y = sign * torch.tensor(np.asarray(scores, dtype=float), dtype=dtype).reshape(-1, 1)
    if y.shape[0] != x.shape[0]:
        raise SuggestionError("points and scores differ in length")
    box = _bounds(bounds, x.shape[1])
    torch.manual_seed(seed)
    model = _fit(x, y, box)
    if acquisition == "log_ei":
        acquisition_function = LogExpectedImprovement(model, best_f=y.max())
    elif acquisition == "ucb":
        acquisition_function = UpperConfidenceBound(model, beta=beta)
    else:
        raise SuggestionError("acquisition must be 'log_ei' or 'ucb'")
    candidate, _ = optimize_acqf(acquisition_function, bounds=box, q=1, num_restarts=num_restarts,
                                 raw_samples=raw_samples)
    with torch.no_grad():
        posterior = model.posterior(candidate)
    best = int(torch.argmax(y))
    return {"next_point": candidate[0].tolist(),
            "predicted_score": float(sign * posterior.mean.item()),
            "predicted_std": float(posterior.variance.sqrt().item()),
            "best_measured_point": x[best].tolist(),
            "best_measured_score": float(sign * y[best].item()),
            "measurements": int(x.shape[0])}


def suggest_pareto_setting(points, objectives, bounds, maximize=None, ref_point=None, seed=0,
                           num_restarts=10, raw_samples=256) -> dict:
    """Next setting that most increases the Pareto hypervolume of several scores."""
    x = _points(points, 2)
    y = torch.tensor(np.asarray(objectives, dtype=float), dtype=dtype)
    if y.ndim != 2 or y.shape[1] < 2 or y.shape[0] != x.shape[0]:
        raise SuggestionError("objectives must be one row per point with at least two scores")
    signs = torch.tensor([1.0 if flag else -1.0 for flag in (maximize or [True] * y.shape[1])], dtype=dtype)
    y = y * signs
    box = _bounds(bounds, x.shape[1])
    reference = (y.min(dim=0).values - 0.1 * y.std(dim=0)) if ref_point is None \
        else torch.tensor(ref_point, dtype=dtype) * signs
    torch.manual_seed(seed)
    model = _fit(x, y, box)
    pareto = is_non_dominated(y)
    acquisition_function = qLogExpectedHypervolumeImprovement(
        model=model, ref_point=reference.tolist(),
        partitioning=NondominatedPartitioning(ref_point=reference, Y=y[pareto]))
    candidate, _ = optimize_acqf(acquisition_function, bounds=box, q=1, num_restarts=num_restarts,
                                 raw_samples=raw_samples)
    with torch.no_grad():
        predicted = model.posterior(candidate).mean[0] * signs
    return {"next_point": candidate[0].tolist(), "predicted_objectives": predicted.tolist(),
            "pareto_indices": torch.where(pareto)[0].tolist(), "measurements": int(x.shape[0])}
