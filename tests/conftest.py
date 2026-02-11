import numpy as np
import pytest

from autolyap.algorithms.algorithm import Algorithm

pytest_plugins = ("tests.shared.cvxpy_fixtures",)


# Shared algorithm fixtures for matrix and parameter tests.
class _ConstantAlgorithm(Algorithm):
    def __init__(self):
        super().__init__(n=1, m=2, m_bar_is=[2, 1], I_func=[1], I_op=[2])

    def get_ABCD(self, k: int):
        A = np.array([[2.0]])
        B = np.array([[10.0, 20.0, 30.0]])
        C = np.array([[1.0], [2.0], [3.0]])
        D = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return A, B, C, D


class _MultiFuncAlgorithm(Algorithm):
    def __init__(self):
        super().__init__(n=1, m=2, m_bar_is=[1, 2], I_func=[1, 2], I_op=[])

    def get_ABCD(self, k: int):
        A = np.array([[1.0]])
        B = np.array([[1.0, 2.0, 3.0]])
        C = np.array([[1.0], [1.0], [1.0]])
        D = np.eye(3)
        return A, B, C, D


class _TinyFunctionalAlgorithm(Algorithm):
    def __init__(self):
        super().__init__(n=1, m=1, m_bar_is=[1], I_func=[1], I_op=[])

    def get_ABCD(self, k: int):
        A = np.array([[1.0]])
        B = np.array([[0.5]])
        C = np.array([[1.0]])
        D = np.array([[0.0]])
        return A, B, C, D


class _TinyOperatorAlgorithm(Algorithm):
    def __init__(self):
        super().__init__(n=1, m=1, m_bar_is=[1], I_func=[], I_op=[1])

    def get_ABCD(self, k: int):
        A = np.array([[1.0]])
        B = np.array([[0.5]])
        C = np.array([[1.0]])
        D = np.array([[0.0]])
        return A, B, C, D


@pytest.fixture
def constant_algorithm() -> Algorithm:
    return _ConstantAlgorithm()


@pytest.fixture
def multi_func_algorithm() -> Algorithm:
    return _MultiFuncAlgorithm()


@pytest.fixture
def tiny_functional_algorithm() -> Algorithm:
    return _TinyFunctionalAlgorithm()


@pytest.fixture
def tiny_operator_algorithm() -> Algorithm:
    return _TinyOperatorAlgorithm()
