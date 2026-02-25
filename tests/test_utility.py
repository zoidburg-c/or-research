import numpy as np
import pytest
from momas.utility import (
    linear_utility,
    threshold_utility,
    chebyshev_utility,
    utilitarian_welfare,
    nash_welfare,
    lorenz_welfare,
)


class TestLinearUtility:
    def test_weighted_sum(self):
        u = linear_utility(np.array([0.5, 0.3, 0.2]))
        result = u(np.array([10.0, 20.0, 30.0]))
        assert result == pytest.approx(17.0)

    def test_uniform_weights(self):
        u = linear_utility(np.array([1.0, 1.0]))
        assert u(np.array([3.0, 4.0])) == pytest.approx(7.0)

    def test_zero_reward(self):
        u = linear_utility(np.array([0.5, 0.5]))
        assert u(np.array([0.0, 0.0])) == pytest.approx(0.0)


class TestThresholdUtility:
    def test_above_thresholds(self):
        u = threshold_utility(
            thresholds=np.array([5.0, 3.0]),
            weights=np.array([1.0, 1.0]),
        )
        result = u(np.array([10.0, 10.0]))
        assert result == pytest.approx(12.0)

    def test_below_threshold_returns_negative_infinity(self):
        u = threshold_utility(
            thresholds=np.array([5.0, 3.0]),
            weights=np.array([1.0, 1.0]),
        )
        result = u(np.array([4.0, 10.0]))
        assert result == float("-inf")

    def test_exactly_at_threshold(self):
        u = threshold_utility(
            thresholds=np.array([5.0, 3.0]),
            weights=np.array([1.0, 1.0]),
        )
        result = u(np.array([5.0, 3.0]))
        assert result == pytest.approx(0.0)


class TestChebyshevUtility:
    def test_balanced_solution(self):
        u = chebyshev_utility(
            weights=np.array([1.0, 1.0]),
            ideal=np.array([10.0, 10.0]),
        )
        result = u(np.array([5.0, 5.0]))
        assert result == pytest.approx(-5.0)

    def test_unbalanced_penalized(self):
        u = chebyshev_utility(
            weights=np.array([1.0, 1.0]),
            ideal=np.array([10.0, 10.0]),
        )
        result = u(np.array([9.0, 1.0]))
        assert result == pytest.approx(-9.0)

    def test_at_ideal(self):
        u = chebyshev_utility(
            weights=np.array([1.0, 1.0]),
            ideal=np.array([10.0, 10.0]),
        )
        assert u(np.array([10.0, 10.0])) == pytest.approx(0.0)


class TestWelfareFunctions:
    def test_utilitarian(self):
        assert utilitarian_welfare([3.0, 4.0, 5.0]) == pytest.approx(12.0)

    def test_nash(self):
        assert nash_welfare([2.0, 3.0, 4.0]) == pytest.approx(24.0)

    def test_nash_with_zero(self):
        assert nash_welfare([0.0, 3.0]) == pytest.approx(0.0)

    def test_lorenz_prefers_equality(self):
        equal = lorenz_welfare([5.0, 5.0, 5.0])
        unequal = lorenz_welfare([1.0, 5.0, 9.0])
        assert equal > unequal

    def test_lorenz_same_sum(self):
        equal = lorenz_welfare([5.0, 5.0])
        unequal = lorenz_welfare([2.0, 8.0])
        assert equal > unequal
