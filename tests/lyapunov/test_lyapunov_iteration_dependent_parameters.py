import numpy as np
import pytest

from autolyap.problemclass import InclusionProblem, Convex
from autolyap.iteration_dependent import IterationDependent


# Tests for iteration-dependent parameter construction and validation.
def test_validate_iteration_dependent_inputs_accepts_valid(tiny_functional_algorithm):
    prob = InclusionProblem([Convex()])
    Q0 = np.eye(3)
    QK = np.eye(3)
    q0 = np.zeros(2)
    qK = np.zeros(2)
    K, *_ = IterationDependent._validate_iteration_dependent_inputs(
        prob, tiny_functional_algorithm, 1, Q0, QK, q0, qK
    )
    assert K == 1


def test_validate_iteration_dependent_inputs_rejects_missing_q(tiny_functional_algorithm):
    prob = InclusionProblem([Convex()])
    Q0 = np.eye(3)
    QK = np.eye(3)
    with pytest.raises(ValueError):
        IterationDependent._validate_iteration_dependent_inputs(
            prob, tiny_functional_algorithm, 1, Q0, QK, None, None
        )


def test_iteration_dependent_distance_to_solution_shapes(tiny_functional_algorithm):
    Q_k, q_k = IterationDependent.get_parameters_distance_to_solution(
        tiny_functional_algorithm, k=0, i=1, j=1
    )
    assert Q_k.shape == (3, 3)
    assert q_k.shape == (2,)


@pytest.mark.parametrize(
    "i,j",
    [
        (0, 1),
        (2, 1),
        (1, 2),
    ],
    ids=["i_zero", "i_too_large", "j_too_large"],
)
def test_iteration_dependent_distance_to_solution_rejects_bad_indices(
    tiny_functional_algorithm, i, j
):
    with pytest.raises(ValueError):
        IterationDependent.get_parameters_distance_to_solution(
            tiny_functional_algorithm, k=0, i=i, j=j
        )


def test_iteration_dependent_distance_to_solution_rejects_negative_k(tiny_functional_algorithm):
    with pytest.raises(ValueError):
        IterationDependent.get_parameters_distance_to_solution(
            tiny_functional_algorithm, k=-1, i=1, j=1
        )


def test_iteration_dependent_state_component_distance_to_solution_values(tiny_functional_algorithm):
    Q_k, q_k = IterationDependent.get_parameters_state_component_distance_to_solution(
        tiny_functional_algorithm, k=0, ell=1
    )
    expected_Q_k = np.array(
        [
            [1.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(Q_k, expected_Q_k)
    assert q_k.shape == (2,)
    assert np.allclose(q_k, 0.0)


@pytest.mark.parametrize("ell", [0, 2], ids=["ell_zero", "ell_too_large"])
def test_iteration_dependent_state_component_distance_to_solution_rejects_bad_ell(
    tiny_functional_algorithm, ell
):
    with pytest.raises(ValueError):
        IterationDependent.get_parameters_state_component_distance_to_solution(
            tiny_functional_algorithm, k=0, ell=ell
        )


def test_iteration_dependent_state_component_distance_to_solution_rejects_negative_k(
    tiny_functional_algorithm,
):
    with pytest.raises(ValueError):
        IterationDependent.get_parameters_state_component_distance_to_solution(
            tiny_functional_algorithm, k=-1, ell=1
        )


def test_iteration_dependent_state_component_cross_iteration_difference_values(tiny_functional_algorithm):
    Q_k, q_k = IterationDependent.get_parameters_state_component_cross_iteration_difference(
        tiny_functional_algorithm, k=0, ell=1, ell_prime=1
    )
    expected_Q_k = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.25, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    assert np.allclose(Q_k, expected_Q_k)
    assert q_k.shape == (2,)
    assert np.allclose(q_k, 0.0)


@pytest.mark.parametrize("ell", [0, 2], ids=["ell_zero", "ell_too_large"])
def test_iteration_dependent_state_component_cross_iteration_difference_rejects_bad_ell(
    tiny_functional_algorithm, ell
):
    with pytest.raises(ValueError):
        IterationDependent.get_parameters_state_component_cross_iteration_difference(
            tiny_functional_algorithm, k=0, ell=ell, ell_prime=1
        )


@pytest.mark.parametrize("ell_prime", [0, 2], ids=["ell_prime_zero", "ell_prime_too_large"])
def test_iteration_dependent_state_component_cross_iteration_difference_rejects_bad_ell_prime(
    tiny_functional_algorithm, ell_prime
):
    with pytest.raises(ValueError):
        IterationDependent.get_parameters_state_component_cross_iteration_difference(
            tiny_functional_algorithm, k=0, ell=1, ell_prime=ell_prime
        )


def test_iteration_dependent_state_component_cross_iteration_difference_rejects_negative_k(
    tiny_functional_algorithm,
):
    with pytest.raises(ValueError):
        IterationDependent.get_parameters_state_component_cross_iteration_difference(
            tiny_functional_algorithm, k=-1, ell=1, ell_prime=1
        )


def test_iteration_dependent_state_component_difference_values(tiny_functional_algorithm):
    Q_k, q_k = IterationDependent.get_parameters_state_component_difference(
        tiny_functional_algorithm, k=0, ell=1, ell_prime=1
    )
    assert np.allclose(Q_k, 0.0)
    assert q_k.shape == (2,)
    assert np.allclose(q_k, 0.0)


@pytest.mark.parametrize("ell", [0, 2], ids=["ell_zero", "ell_too_large"])
def test_iteration_dependent_state_component_difference_rejects_bad_ell(
    tiny_functional_algorithm, ell
):
    with pytest.raises(ValueError):
        IterationDependent.get_parameters_state_component_difference(
            tiny_functional_algorithm, k=0, ell=ell, ell_prime=1
        )


@pytest.mark.parametrize("ell_prime", [0, 2], ids=["ell_prime_zero", "ell_prime_too_large"])
def test_iteration_dependent_state_component_difference_rejects_bad_ell_prime(
    tiny_functional_algorithm, ell_prime
):
    with pytest.raises(ValueError):
        IterationDependent.get_parameters_state_component_difference(
            tiny_functional_algorithm, k=0, ell=1, ell_prime=ell_prime
        )


def test_iteration_dependent_state_component_difference_rejects_negative_k(
    tiny_functional_algorithm,
):
    with pytest.raises(ValueError):
        IterationDependent.get_parameters_state_component_difference(
            tiny_functional_algorithm, k=-1, ell=1, ell_prime=1
        )


def test_iteration_dependent_function_value_suboptimality_values(tiny_functional_algorithm):
    Q_k, q_k = IterationDependent.get_parameters_function_value_suboptimality(
        tiny_functional_algorithm, k=0, j=1
    )
    assert np.allclose(Q_k, 0.0)
    assert q_k.shape == (2,)
    assert np.allclose(q_k, np.array([1.0, -1.0]))


def test_iteration_dependent_function_value_suboptimality_rejects_bad_j(tiny_functional_algorithm):
    with pytest.raises(ValueError):
        IterationDependent.get_parameters_function_value_suboptimality(
            tiny_functional_algorithm, k=0, j=2
        )


def test_iteration_dependent_function_value_suboptimality_rejects_negative_k(tiny_functional_algorithm):
    with pytest.raises(ValueError):
        IterationDependent.get_parameters_function_value_suboptimality(
            tiny_functional_algorithm, k=-1, j=1
        )


def test_iteration_dependent_compute_thetas_requires_functional_components(tiny_operator_algorithm):
    with pytest.raises(ValueError):
        IterationDependent._compute_thetas(tiny_operator_algorithm)
