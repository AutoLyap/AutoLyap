import numpy as np
import pytest


# Tests for X matrix construction and immutability.
def test_get_xs_basic_k0(constant_algorithm):
    Xs = constant_algorithm._get_Xs(0, 0)
    assert set(Xs.keys()) == {0, 1}

    X0 = Xs[0]
    X1 = Xs[1]
    assert X0.shape == (1, 6)
    assert X1.shape == (1, 6)

    expected_X0 = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    expected_X1 = np.array([[2.0, 10.0, 20.0, 30.0, 0.0, 0.0]])
    assert np.allclose(X0, expected_X0)
    assert np.allclose(X1, expected_X1)


def test_get_xs_multi_k_k2_values(constant_algorithm):
    Xs = constant_algorithm._get_Xs(0, 1)
    X2 = Xs[2]

    expected_X2 = np.array([[4.0, 20.0, 40.0, 60.0, 10.0, 20.0, 30.0, 0.0, 0.0]])
    assert X2.shape == (1, 9)
    assert np.allclose(X2, expected_X2)


def test_get_xs_returns_readonly_views(constant_algorithm):
    Xs = constant_algorithm._get_Xs(0, 0)
    with pytest.raises(ValueError):
        Xs[0][0, 0] = 2.0


def test_get_xs_invalid_range_raises(constant_algorithm):
    with pytest.raises(ValueError):
        constant_algorithm._get_Xs(1, 0)


def test_get_xs_large_horizon_shapes(constant_algorithm):
    k_min = 0
    k_max = 25
    Xs = constant_algorithm._get_Xs(k_min, k_max)
    assert set(Xs.keys()) == set(range(k_min, k_max + 2))
    total_cols = 1 + (k_max - k_min + 1) * 3 + 2
    assert Xs[0].shape == (1, total_cols)
    assert Xs[k_max + 1].shape == (1, total_cols)
