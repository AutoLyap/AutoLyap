import numpy as np
import pytest

from autolyap.problemclass import (
    Convex,
    StronglyConvex,
    WeaklyConvex,
    Smooth,
    SmoothConvex,
    SmoothStronglyConvex,
    SmoothWeaklyConvex,
    IndicatorFunctionOfClosedConvexSet,
    SupportFunctionOfClosedConvexSet,
    GradientDominated,
)
from autolyap.problemclass.problemclass import (
    ParametrizedFunctionInterpolationCondition,
    FunctionInterpolationCondition,
)


# Tests for function interpolation condition data and parameter validation.
def _assert_function_data_item(
    item, expected_index: str, eq_flag: bool, matrix_shape: tuple, vector_len: int
) -> None:
    matrix, vector, eq, interp_idx = item
    assert matrix.shape == matrix_shape
    assert np.allclose(matrix, matrix.T)
    assert vector.shape == (vector_len,)
    assert eq is eq_flag
    assert str(interp_idx) == expected_index


def test_parametrized_function_condition_requires_mu_lt_L():
    with pytest.raises(ValueError):
        ParametrizedFunctionInterpolationCondition(mu=1.0, L=1.0)


def test_parametrized_function_condition_rejects_negative_infinity_mu():
    with pytest.raises(ValueError):
        ParametrizedFunctionInterpolationCondition(mu=float("-inf"), L=1.0)


def test_parametrized_function_condition_basic_shape():
    cond = ParametrizedFunctionInterpolationCondition(mu=0.0, L=float("inf"))
    data = cond.get_data()
    assert len(data) == 1
    _assert_function_data_item(data[0], "j1!=j2", False, (4, 4), 2)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Convex(),
        lambda: StronglyConvex(0.1),
        lambda: WeaklyConvex(0.2),
        lambda: Smooth(1.0),
        lambda: SmoothConvex(1.0),
        lambda: SmoothStronglyConvex(0.1, 1.0),
        lambda: SmoothWeaklyConvex(0.2, 1.0),
    ],
    ids=[
        "convex",
        "strongly_convex",
        "weakly_convex",
        "smooth",
        "smooth_convex",
        "smooth_strongly_convex",
        "smooth_weakly_convex",
    ],
)
def test_function_conditions_shapes_and_indices(factory):
    data = factory().get_data()
    assert len(data) == 1
    _assert_function_data_item(data[0], "j1!=j2", False, (4, 4), 2)


def test_smooth_strongly_convex_rejects_mu_ge_L():
    with pytest.raises(ValueError):
        SmoothStronglyConvex(1.0, 1.0)


@pytest.mark.parametrize("bad_mu_tilde", [0.0, -1.0, float("inf"), "x"])
def test_weakly_convex_rejects_invalid_mu_tilde(bad_mu_tilde):
    with pytest.raises(ValueError):
        WeaklyConvex(bad_mu_tilde)


@pytest.mark.parametrize("bad_mu_tilde", [0.0, -1.0, float("inf"), "x"])
def test_smooth_weakly_convex_rejects_invalid_mu_tilde(bad_mu_tilde):
    with pytest.raises(ValueError):
        SmoothWeaklyConvex(bad_mu_tilde, 1.0)


@pytest.mark.parametrize(
    "factory, expected",
    [
        (
            lambda: IndicatorFunctionOfClosedConvexSet(),
            [
                ("j1!=j2", False, (4, 4), 2),
                ("j1", True, (2, 2), 1),
            ],
        ),
        (
            lambda: SupportFunctionOfClosedConvexSet(),
            [
                ("j1!=j2", False, (4, 4), 2),
                ("j1", True, (2, 2), 1),
            ],
        ),
        (
            lambda: GradientDominated(1.0),
            [
                ("j1!=star", False, (4, 4), 2),
                ("j1!=star", False, (4, 4), 2),
            ],
        ),
    ],
    ids=["indicator", "support", "gradient_dominated"],
)
def test_function_condition_multi_item_shapes(factory, expected):
    data = factory().get_data()
    assert len(data) == len(expected)
    for item, spec in zip(data, expected):
        _assert_function_data_item(item, *spec)
