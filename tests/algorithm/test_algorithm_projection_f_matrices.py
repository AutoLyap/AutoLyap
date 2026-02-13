import numpy as np
import pytest


# Tests for projection (P) and function (F) matrices.
def test_get_ps_values(constant_algorithm):
    Ps = constant_algorithm._get_Ps()
    expected_keys = {(1, 1), (1, 2), (2, 1), (1, "star"), (2, "star")}
    assert set(Ps.keys()) == expected_keys

    assert np.allclose(Ps[(1, 1)], np.array([[1.0, 0.0, 0.0]]))
    assert np.allclose(Ps[(1, 2)], np.array([[0.0, 1.0, 0.0]]))
    assert np.allclose(Ps[(2, 1)], np.array([[0.0, 0.0, 1.0]]))
    assert np.allclose(Ps[(1, "star")], np.array([[1.0, 0.0]]))
    assert np.allclose(Ps[(2, "star")], np.array([[0.0, 1.0]]))


def test_get_fs_values(constant_algorithm):
    Fs = constant_algorithm._get_Fs(0, 1)
    expected_keys = {
        (1, 1, 0),
        (1, 2, 0),
        (1, 1, 1),
        (1, 2, 1),
        (1, "star", "star"),
    }
    assert set(Fs.keys()) == expected_keys

    assert np.allclose(Fs[(1, 1, 0)], np.array([[1.0, 0.0, 0.0, 0.0, 0.0]]))
    assert np.allclose(Fs[(1, 2, 0)], np.array([[0.0, 1.0, 0.0, 0.0, 0.0]]))
    assert np.allclose(Fs[(1, 1, 1)], np.array([[0.0, 0.0, 1.0, 0.0, 0.0]]))
    assert np.allclose(Fs[(1, 2, 1)], np.array([[0.0, 0.0, 0.0, 1.0, 0.0]]))
    assert np.allclose(Fs[(1, "star", "star")], np.array([[0.0, 0.0, 0.0, 0.0, 1.0]]))


def test_get_fs_multiple_function_components_offsets(multi_func_algorithm):
    Fs = multi_func_algorithm._get_Fs(0, 0)
    expected_keys = {
        (1, 1, 0),
        (2, 1, 0),
        (2, 2, 0),
        (1, "star", "star"),
        (2, "star", "star"),
    }
    assert set(Fs.keys()) == expected_keys

    assert np.allclose(Fs[(1, 1, 0)], np.array([[1.0, 0.0, 0.0, 0.0, 0.0]]))
    assert np.allclose(Fs[(2, 1, 0)], np.array([[0.0, 1.0, 0.0, 0.0, 0.0]]))
    assert np.allclose(Fs[(2, 2, 0)], np.array([[0.0, 0.0, 1.0, 0.0, 0.0]]))
    assert np.allclose(Fs[(1, "star", "star")], np.array([[0.0, 0.0, 0.0, 1.0, 0.0]]))
    assert np.allclose(Fs[(2, "star", "star")], np.array([[0.0, 0.0, 0.0, 0.0, 1.0]]))


def test_get_fs_returns_readonly_views(constant_algorithm):
    Fs = constant_algorithm._get_Fs(0, 0)
    with pytest.raises(ValueError):
        Fs[(1, 1, 0)][0, 0] = 1.0
    with pytest.raises(ValueError):
        Fs[(1, "star", "star")][0, 0] = 1.0


def test_get_fs_invalid_range_raises(constant_algorithm):
    with pytest.raises(ValueError):
        constant_algorithm._get_Fs(1, 0)
