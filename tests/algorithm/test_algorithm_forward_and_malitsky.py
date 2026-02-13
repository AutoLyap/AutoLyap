import numpy as np

from autolyap.algorithms import ForwardMethod, MalitskyTamFRB


# Focused regression coverage for restored ForwardMethod and MalitskyTamFRB classes.
def test_forward_method_structure_and_abcd_are_k_invariant():
    algo = ForwardMethod(gamma=0.2)

    assert algo.n == 1
    assert algo.m == 1
    assert algo.m_bar_is == [1]
    assert algo.m_bar == 1
    assert algo.I_func == []
    assert algo.I_op == [1]

    expected_A = np.array([[1.0]])
    expected_B = np.array([[-0.2]])
    expected_C = np.array([[1.0]])
    expected_D = np.array([[0.0]])

    for k in [0, 5]:
        A, B, C, D = algo.get_ABCD(k)
        assert np.allclose(A, expected_A)
        assert np.allclose(B, expected_B)
        assert np.allclose(C, expected_C)
        assert np.allclose(D, expected_D)


def test_malitsky_tam_frb_structure_and_abcd_are_k_invariant():
    algo = MalitskyTamFRB(gamma=0.2)

    assert algo.n == 2
    assert algo.m == 2
    assert algo.m_bar_is == [2, 1]
    assert algo.m_bar == 3
    assert algo.I_func == []
    assert algo.I_op == [1, 2]

    expected_A = np.array([[1.0, 0.0], [1.0, 0.0]])
    expected_B = np.array([[-0.4, 0.2, -0.2], [0.0, 0.0, 0.0]])
    expected_C = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    expected_D = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.4, 0.2, -0.2]]
    )

    for k in [0, 5]:
        A, B, C, D = algo.get_ABCD(k)
        assert np.allclose(A, expected_A)
        assert np.allclose(B, expected_B)
        assert np.allclose(C, expected_C)
        assert np.allclose(D, expected_D)


def test_forward_method_set_gamma_updates_b_matrix():
    algo = ForwardMethod(gamma=0.2)
    _, B_before, _, _ = algo.get_ABCD(0)
    algo.set_gamma(0.5)
    _, B_after, _, _ = algo.get_ABCD(0)

    assert np.allclose(B_before, np.array([[-0.2]]))
    assert np.allclose(B_after, np.array([[-0.5]]))


def test_malitsky_tam_frb_set_gamma_updates_only_scaled_blocks():
    algo = MalitskyTamFRB(gamma=0.2)
    A_before, B_before, C_before, D_before = algo.get_ABCD(0)
    algo.set_gamma(0.5)
    A_after, B_after, C_after, D_after = algo.get_ABCD(0)

    assert np.allclose(A_before, A_after)
    assert np.allclose(C_before, C_after)
    assert np.allclose(B_before, np.array([[-0.4, 0.2, -0.2], [0.0, 0.0, 0.0]]))
    assert np.allclose(B_after, np.array([[-1.0, 0.5, -0.5], [0.0, 0.0, 0.0]]))
    assert np.allclose(
        D_before,
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.4, 0.2, -0.2]]),
    )
    assert np.allclose(
        D_after,
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.5, -0.5]]),
    )
