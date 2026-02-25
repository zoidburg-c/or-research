from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


def linear_utility(weights: np.ndarray) -> Callable[[np.ndarray], float]:
    """Weighted sum: u(r) = w . r"""
    w = np.asarray(weights, dtype=np.float64)

    def _u(rewards: np.ndarray) -> float:
        return float(np.dot(w, rewards))

    return _u


def threshold_utility(
    thresholds: np.ndarray,
    weights: np.ndarray,
) -> Callable[[np.ndarray], float]:
    """Returns -inf if any objective is below its threshold,
    otherwise returns the weighted sum of the surplus above thresholds."""
    t = np.asarray(thresholds, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    def _u(rewards: np.ndarray) -> float:
        r = np.asarray(rewards, dtype=np.float64)
        if np.any(r < t):
            return float("-inf")
        return float(np.dot(w, r - t))

    return _u


def chebyshev_utility(
    weights: np.ndarray,
    ideal: np.ndarray,
) -> Callable[[np.ndarray], float]:
    """Negated weighted Chebyshev distance from ideal point.
    u(r) = -max_d( w_d * |ideal_d - r_d| )
    Higher (less negative) is better. 0 at ideal point."""
    w = np.asarray(weights, dtype=np.float64)
    z = np.asarray(ideal, dtype=np.float64)

    def _u(rewards: np.ndarray) -> float:
        r = np.asarray(rewards, dtype=np.float64)
        return float(-np.max(w * np.abs(z - r)))

    return _u


def utilitarian_welfare(utilities: list[float]) -> float:
    """Sum of individual utilities."""
    return sum(utilities)


def nash_welfare(utilities: list[float]) -> float:
    """Product of individual utilities."""
    return float(math.prod(utilities))


def lorenz_welfare(utilities: list[float]) -> float:
    """Sum of cumulative sorted utilities (Lorenz-based).
    Prefers more equal distributions: for same total,
    equal distributions produce higher Lorenz welfare."""
    sorted_u = sorted(utilities)
    cumulative = 0.0
    total = 0.0
    for u in sorted_u:
        cumulative += u
        total += cumulative
    return total
