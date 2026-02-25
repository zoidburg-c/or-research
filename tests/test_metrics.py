import numpy as np
import pytest
from momas.metrics import compute_pareto_front, hypervolume, evaluate_ser, evaluate_esr
from momas.utility import linear_utility


class TestParetoFront:
    def test_simple_2d(self):
        points = np.array([[1, 5], [2, 3], [3, 2], [4, 1], [3, 4]])
        front = compute_pareto_front(points)
        assert len(front) == 3
        expected = {(1, 5), (3, 4), (4, 1)}
        actual = {tuple(p) for p in front}
        assert actual == expected

    def test_single_point(self):
        points = np.array([[1.0, 2.0]])
        front = compute_pareto_front(points)
        assert len(front) == 1

    def test_all_same(self):
        points = np.array([[1.0, 1.0], [1.0, 1.0]])
        front = compute_pareto_front(points)
        assert len(front) == 1


class TestHypervolume:
    def test_simple_2d(self):
        front = np.array([[3.0, 1.0], [1.0, 3.0]])
        ref = np.array([0.0, 0.0])
        hv = hypervolume(front, ref)
        assert hv == pytest.approx(5.0)

    def test_single_point(self):
        front = np.array([[2.0, 3.0]])
        ref = np.array([0.0, 0.0])
        hv = hypervolume(front, ref)
        assert hv == pytest.approx(6.0)


class TestSERESR:
    def test_ser_linear(self):
        episode_returns = np.array([[10.0, 20.0], [30.0, 40.0]])
        u = linear_utility(np.array([0.5, 0.5]))
        ser = evaluate_ser(episode_returns, u)
        assert ser == pytest.approx(25.0)

    def test_esr_linear(self):
        episode_returns = np.array([[10.0, 20.0], [30.0, 40.0]])
        u = linear_utility(np.array([0.5, 0.5]))
        esr = evaluate_esr(episode_returns, u)
        assert esr == pytest.approx(25.0)

    def test_ser_esr_differ_nonlinear(self):
        episode_returns = np.array([[1.0, 9.0], [9.0, 1.0]])
        def product_u(r):
            return float(r[0] * r[1])
        ser = evaluate_ser(episode_returns, product_u)
        esr = evaluate_esr(episode_returns, product_u)
        assert ser == pytest.approx(25.0)
        assert esr == pytest.approx(9.0)
        assert ser != pytest.approx(esr)
