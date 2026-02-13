import numpy as np
import pytest

from autolyap.algorithms import (
    ForwardMethod,
    GradientMethod,
    MalitskyTamFRB,
    DouglasRachford,
    ProximalPoint,
    TripleMomentum,
    DavisYin,
    HeavyBallMethod,
    ChambollePock,
)


# Sanity checks for concrete algorithm ABCD implementations.
@pytest.mark.parametrize(
    "factory, shapes, expected, expected_indices",
    [
        (
            lambda: ForwardMethod(gamma=0.3),
            ((1, 1), (1, 1), (1, 1), (1, 1)),
            (
                np.array([[1.0]]),
                np.array([[-0.3]]),
                np.array([[1.0]]),
                np.array([[0.0]]),
            ),
            ({"I_op": [1], "I_func": []}),
        ),
        (
            lambda: GradientMethod(gamma=0.3),
            ((1, 1), (1, 1), (1, 1), (1, 1)),
            (
                np.array([[1.0]]),
                np.array([[-0.3]]),
                np.array([[1.0]]),
                np.array([[0.0]]),
            ),
            None,
        ),
        (
            lambda: ProximalPoint(gamma=0.4),
            ((1, 1), (1, 1), (1, 1), (1, 1)),
            (
                np.array([[1.0]]),
                np.array([[-0.4]]),
                np.array([[1.0]]),
                np.array([[-0.4]]),
            ),
            None,
        ),
        (
            lambda: DouglasRachford(gamma=0.2, lambda_value=0.5, type="operator"),
            ((1, 1), (1, 2), (2, 1), (2, 2)),
            (
                np.array([[1.0]]),
                np.array([[-0.1, -0.1]]),
                np.array([[1.0], [1.0]]),
                np.array([[-0.2, 0.0], [-0.4, -0.2]]),
            ),
            ({"I_op": [1, 2], "I_func": []}),
        ),
        (
            lambda: DavisYin(gamma=0.2, lambda_value=0.5),
            ((1, 1), (1, 3), (3, 1), (3, 3)),
            (
                np.array([[1.0]]),
                np.array([[-0.1, -0.1, -0.1]]),
                np.array([[1.0], [1.0], [1.0]]),
                np.array([[-0.2, 0.0, 0.0], [-0.2, 0.0, 0.0], [-0.4, -0.2, -0.2]]),
            ),
            None,
        ),
        (
            lambda: HeavyBallMethod(gamma=0.3, delta=0.1),
            ((2, 2), (2, 1), (1, 2), (1, 1)),
            (
                np.array([[1.1, -0.1], [1.0, 0.0]]),
                np.array([[-0.3], [0.0]]),
                np.array([[1.0, 0.0]]),
                np.array([[0.0]]),
            ),
            None,
        ),
        (
            lambda: ChambollePock(tau=0.4, sigma=2.0, theta=0.5),
            ((2, 2), (2, 2), (2, 2), (2, 2)),
            (
                np.array([[1.0, -0.4], [0.0, 0.0]]),
                np.array([[-0.4, 0.0], [0.0, 1.0]]),
                np.array([[1.0, -0.4], [1.0, -0.1]]),
                np.array([[-0.4, 0.0], [-0.6, -0.5]]),
            ),
            None,
        ),
        (
            lambda: MalitskyTamFRB(gamma=0.2),
            ((2, 2), (2, 3), (3, 2), (3, 3)),
            (
                np.array([[1.0, 0.0], [1.0, 0.0]]),
                np.array([[-0.4, 0.2, -0.2], [0.0, 0.0, 0.0]]),
                np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.4, 0.2, -0.2]]),
            ),
            ({"I_op": [1, 2], "I_func": []}),
        ),
    ],
    ids=[
        "forward_method",
        "gradient_method",
        "proximal_point",
        "douglas_rachford_op",
        "davis_yin",
        "heavy_ball",
        "chambolle_pock",
        "malitsky_tam_frb",
    ],
)
def test_abcd_values(factory, shapes, expected, expected_indices):
    algo = factory()
    A, B, C, D = algo.get_ABCD(0)
    assert A.shape == shapes[0]
    assert B.shape == shapes[1]
    assert C.shape == shapes[2]
    assert D.shape == shapes[3]
    assert np.allclose(A, expected[0])
    assert np.allclose(B, expected[1])
    assert np.allclose(C, expected[2])
    assert np.allclose(D, expected[3])
    if expected_indices is not None:
        for name, value in expected_indices.items():
            assert getattr(algo, name) == value


def test_douglas_rachford_function_type_sets_indices():
    algo = DouglasRachford(gamma=0.2, lambda_value=0.5, type="function")
    assert algo.I_func == [1, 2]
    assert algo.I_op == []


def test_triple_momentum_abcd_values():
    mu = 1.0
    L = 4.0
    algo = TripleMomentum(mu=mu, L=L)
    A, B, C, D = algo.get_ABCD(0)

    q = mu / L
    sqrt_q = np.sqrt(q)
    alpha = (2 - sqrt_q) / L
    _beta = (1 - sqrt_q) ** 2 / (1 + sqrt_q)
    gamma = (1 - sqrt_q) ** 2 / ((2 - sqrt_q) * (1 + sqrt_q))

    assert A.shape == (2, 2)
    assert B.shape == (2, 1)
    assert C.shape == (1, 2)
    assert D.shape == (1, 1)
    assert np.allclose(A, np.array([[1 + _beta, -_beta], [1.0, 0.0]]))
    assert np.allclose(B, np.array([[-alpha], [0.0]]))
    assert np.allclose(C, np.array([[1 + gamma, -gamma]]))
    assert np.allclose(D, np.array([[0.0]]))
