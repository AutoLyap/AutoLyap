import numpy as np
import pytest


# Tests for U and Y matrix construction and immutability.
def test_get_us_and_star_shapes_and_values(constant_algorithm):
    Us = constant_algorithm._get_Us(0, 0)
    assert set(Us.keys()) == {0, "star"}

    U0 = Us[0]
    U_star = Us["star"]
    assert U0.shape == (3, 6)
    assert U_star.shape == (2, 6)

    expected_U0 = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    expected_U_star = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        ]
    )
    assert np.allclose(U0, expected_U0)
    assert np.allclose(U_star, expected_U_star)


def test_get_us_returns_readonly_views(constant_algorithm):
    Us = constant_algorithm._get_Us(0, 0)
    with pytest.raises(ValueError):
        Us[0][0, 0] = 1.0
    with pytest.raises(ValueError):
        Us["star"][0, 0] = 1.0


def test_get_ys_basic_k0(constant_algorithm):
    Ys = constant_algorithm._get_Ys(0, 0)
    assert set(Ys.keys()) == {0, "star"}

    Y0 = Ys[0]
    Y_star = Ys["star"]
    assert Y0.shape == (3, 6)
    assert Y_star.shape == (2, 6)

    C = np.array([[1.0], [2.0], [3.0]])
    D = np.eye(3)
    assert np.allclose(Y0[:, 0:1], C)
    assert np.allclose(Y0[:, 1:4], D)
    assert np.allclose(Y0[:, 4:6], np.zeros((3, 2)))

    expected_star = np.zeros((2, 6))
    expected_star[:, -1] = 1.0
    assert np.allclose(Y_star, expected_star)


def test_get_ys_returns_readonly_views(constant_algorithm):
    Ys = constant_algorithm._get_Ys(0, 0)
    with pytest.raises(ValueError):
        Ys[0][0, 0] = 1.0
    with pytest.raises(ValueError):
        Ys["star"][0, 0] = 1.0


def test_get_ys_multi_k_k1_values(constant_algorithm):
    Ys = constant_algorithm._get_Ys(0, 1)
    Y1 = Ys[1]
    Y_star = Ys["star"]

    assert Y1.shape == (3, 9)
    assert Y_star.shape == (2, 9)

    C = np.array([[1.0], [2.0], [3.0]])
    B = np.array([[10.0, 20.0, 30.0]])
    D = np.eye(3)

    assert np.allclose(Y1[:, 0:1], 2.0 * C)
    assert np.allclose(Y1[:, 1:4], C @ B)
    assert np.allclose(Y1[:, 4:7], D)
    assert np.allclose(Y1[:, 7:9], np.zeros((3, 2)))

    expected_star = np.zeros((2, 9))
    expected_star[:, -1] = 1.0
    assert np.allclose(Y_star, expected_star)


def test_get_ys_invalid_range_raises(constant_algorithm):
    with pytest.raises(ValueError):
        constant_algorithm._get_Ys(1, 0)
