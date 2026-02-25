from __future__ import annotations

from collections.abc import Callable

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def compute_pareto_front(points: np.ndarray) -> np.ndarray:
    """Compute the Pareto front from a set of points (maximization convention).

    Args:
        points: shape (N, d) — N solution vectors with d objectives.

    Returns:
        shape (K, d) — the K non-dominated points.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= 1:
        return points.copy()

    # pymoo uses minimization — negate for maximization
    nds = NonDominatedSorting()
    front_idx = nds.do(-points, only_non_dominated_front=True)

    # Deduplicate
    front = points[front_idx]
    unique = np.unique(front, axis=0)
    return unique


def hypervolume(front: np.ndarray, ref_point: np.ndarray) -> float:
    """Compute the hypervolume indicator (maximization convention).

    Args:
        front: shape (K, d) — Pareto front points.
        ref_point: shape (d,) — reference point (must be dominated by all front points).

    Returns:
        Hypervolume scalar value.
    """
    front = np.asarray(front, dtype=np.float64)
    ref = np.asarray(ref_point, dtype=np.float64)

    # pymoo uses minimization — negate both
    indicator = HV(ref_point=-ref)
    return float(indicator(-front))


def evaluate_ser(
    episode_returns: np.ndarray,
    utility_fn: Callable[[np.ndarray], float],
) -> float:
    """Scalarised Expected Returns: apply utility to expected return vector.
    V_SER = u(E[returns])"""
    expected = np.mean(episode_returns, axis=0)
    return utility_fn(expected)


def evaluate_esr(
    episode_returns: np.ndarray,
    utility_fn: Callable[[np.ndarray], float],
) -> float:
    """Expected Scalarised Returns: average utility over individual episodes.
    V_ESR = E[u(returns)]"""
    utilities = [utility_fn(r) for r in episode_returns]
    return float(np.mean(utilities))
