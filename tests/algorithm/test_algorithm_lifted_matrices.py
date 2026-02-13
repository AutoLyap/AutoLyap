import numpy as np
import pytest


# Tests for lifted constraint matrices (E/W/F aggregation).
def test_compute_e_star_and_nonstar_shape(constant_algorithm):
    E = constant_algorithm._compute_E(1, [(1, 0), ("star", "star")], 0, 0)
    assert E.shape == (4, 6)

    Ys = constant_algorithm._get_Ys(0, 0)
    Us = constant_algorithm._get_Us(0, 0)
    assert np.allclose(E[0], Ys[0][0])
    assert np.allclose(E[1], Ys["star"][0])
    assert np.allclose(E[2], Us[0][0])
    assert np.allclose(E[3], Us["star"][0])


def test_compute_e_rejects_empty_pairs(constant_algorithm):
    with pytest.raises(ValueError):
        constant_algorithm._compute_E(1, [], 0, 0)


def test_compute_e_rejects_out_of_range_component(constant_algorithm):
    with pytest.raises(ValueError):
        constant_algorithm._compute_E(3, [(1, 0)], 0, 0)


def test_compute_e_rejects_out_of_range_j_or_k(constant_algorithm):
    with pytest.raises(ValueError):
        constant_algorithm._compute_E(1, [(3, 0)], 0, 0)
    with pytest.raises(ValueError):
        constant_algorithm._compute_E(1, [(1, 1)], 0, 0)


def test_compute_w_validates_matrix_shape(constant_algorithm):
    with pytest.raises(ValueError):
        constant_algorithm._compute_W(1, [(1, 0)], 0, 0, np.eye(3))


def test_compute_w_rejects_nonsymmetric_matrix(constant_algorithm):
    M = np.array([[1.0, 2.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        constant_algorithm._compute_W(1, [(1, 0)], 0, 0, M)


def test_compute_f_aggregated_rejects_partial_star_pair(constant_algorithm):
    with pytest.raises(ValueError):
        constant_algorithm._compute_F_aggregated(1, [("star", 0)], 0, 0, np.array([1.0]))


def test_compute_f_aggregated_rejects_bad_weight_length(constant_algorithm):
    with pytest.raises(ValueError):
        constant_algorithm._compute_F_aggregated(1, [(1, 0), (1, 0)], 0, 0, np.array([1.0]))
