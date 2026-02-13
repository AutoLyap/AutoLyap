import numpy as np
import pytest

from autolyap.algorithms.algorithm import Algorithm


# Validation tests for Algorithm._get_AsBsCsDs and get_ABCD outputs.
class BadTupleAlgorithm(Algorithm):
    def __init__(self):
        super().__init__(n=1, m=1, m_bar_is=[1], I_func=[1], I_op=[])

    def get_ABCD(self, k: int):
        return [np.eye(1), np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))]


class BadShapeAlgorithm(Algorithm):
    def __init__(self):
        super().__init__(n=1, m=1, m_bar_is=[1], I_func=[1], I_op=[])

    def get_ABCD(self, k: int):
        A = np.eye(1)
        B = np.zeros((2, 1))
        C = np.zeros((1, 1))
        D = np.zeros((1, 1))
        return A, B, C, D


class BadTypeAlgorithm(Algorithm):
    def __init__(self):
        super().__init__(n=1, m=1, m_bar_is=[1], I_func=[1], I_op=[])

    def get_ABCD(self, k: int):
        A = [[1.0]]
        B = np.zeros((1, 1))
        C = np.zeros((1, 1))
        D = np.zeros((1, 1))
        return A, B, C, D


def test_get_asbscsds_requires_tuple():
    algo = BadTupleAlgorithm()
    with pytest.raises(ValueError):
        algo._get_AsBsCsDs(0, 0)


def test_get_asbscsds_rejects_bad_shapes():
    algo = BadShapeAlgorithm()
    with pytest.raises(ValueError):
        algo._get_AsBsCsDs(0, 0)


def test_get_asbscsds_rejects_non_arrays():
    algo = BadTypeAlgorithm()
    with pytest.raises(ValueError):
        algo._get_AsBsCsDs(0, 0)
