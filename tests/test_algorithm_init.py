import numpy as np
import pytest

from autolyap.algorithms.algorithm import Algorithm


# Basic initialization and validation tests for Algorithm.
class DummyAlgorithm(Algorithm):
    def get_ABCD(self, k: int):
        A = np.eye(self.n)
        B = np.zeros((self.n, self.m_bar))
        C = np.zeros((self.m_bar, self.n))
        D = np.zeros((self.m_bar, self.m_bar))
        return A, B, C, D


def test_algorithm_init_sets_dimensions():
    algo = DummyAlgorithm(n=2, m=2, m_bar_is=[1, 2], I_func=[1], I_op=[2])
    assert algo.m_bar == 3
    assert algo.m_bar_func == 1
    assert algo.m_bar_op == 2
    assert algo.kappa == {1: 1}


def test_algorithm_init_rejects_invalid_sizes():
    with pytest.raises(ValueError):
        DummyAlgorithm(n=0, m=1, m_bar_is=[1], I_func=[1], I_op=[])
    with pytest.raises(ValueError):
        DummyAlgorithm(n=1, m=0, m_bar_is=[1], I_func=[1], I_op=[])


def test_algorithm_init_rejects_m_bar_length_mismatch():
    with pytest.raises(ValueError):
        DummyAlgorithm(n=1, m=2, m_bar_is=[1], I_func=[1], I_op=[2])


def test_algorithm_init_requires_index_cover():
    with pytest.raises(ValueError):
        DummyAlgorithm(n=1, m=2, m_bar_is=[1, 1], I_func=[1], I_op=[])


def test_algorithm_init_rejects_nonpositive_m_bar():
    with pytest.raises(ValueError):
        DummyAlgorithm(n=1, m=1, m_bar_is=[0], I_func=[1], I_op=[])
    with pytest.raises(ValueError):
        DummyAlgorithm(n=1, m=1, m_bar_is=[-1], I_func=[1], I_op=[])


def test_algorithm_init_rejects_overlapping_indices():
    with pytest.raises(ValueError):
        DummyAlgorithm(n=1, m=1, m_bar_is=[1], I_func=[1], I_op=[1])
