import numpy as np
import pytest

from autolyap.problemclass import (
    MaximallyMonotone,
    StronglyMonotone,
    LipschitzOperator,
    Cocoercive,
    WeakMintyVariationalInequality,
)
from autolyap.problemclass.problemclass import OperatorInterpolationCondition


# Tests for operator interpolation condition data and parameter validation.
def _assert_operator_data(cond: OperatorInterpolationCondition, expected_index: str) -> None:
    data = cond.get_data()
    assert len(data) == 1
    matrix, interp_idx = data[0]
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix, matrix.T)
    assert str(interp_idx) == expected_index


def test_operator_conditions_shapes_and_indices():
    _assert_operator_data(MaximallyMonotone(), "r1<r2")
    _assert_operator_data(StronglyMonotone(0.5), "r1<r2")
    _assert_operator_data(LipschitzOperator(2.0), "r1<r2")
    _assert_operator_data(Cocoercive(1.0), "r1<r2")
    _assert_operator_data(WeakMintyVariationalInequality(0.0), "r1!=star")


@pytest.mark.parametrize("bad_mu", [0, -1, float("inf"), "x"])
def test_strongly_monotone_rejects_invalid_mu(bad_mu):
    with pytest.raises(ValueError):
        StronglyMonotone(bad_mu)


@pytest.mark.parametrize("bad_L", [0, -1, float("inf"), "x"])
def test_lipschitz_operator_rejects_invalid_L(bad_L):
    with pytest.raises(ValueError):
        LipschitzOperator(bad_L)


@pytest.mark.parametrize("bad_beta", [0, -1, float("inf"), "x"])
def test_cocoercive_rejects_invalid_beta(bad_beta):
    with pytest.raises(ValueError):
        Cocoercive(bad_beta)


@pytest.mark.parametrize("bad_rho", [float("inf"), "x"])
def test_weak_minty_rejects_invalid_rho(bad_rho):
    with pytest.raises(ValueError):
        WeakMintyVariationalInequality(bad_rho)
