import numpy as np
import pytest

from autolyap.problemclass import InclusionProblem, Convex, MaximallyMonotone
from autolyap.iteration_independent import IterationIndependent


# Tests for iteration-independent parameter construction and validation.
def test_validate_iteration_independent_inputs_accepts_valid(tiny_functional_algorithm):
    prob = InclusionProblem([Convex()])
    P = np.eye(3)
    T = np.eye(4)
    p = np.zeros(2)
    t = np.zeros(3)
    h, alpha, *_ = IterationIndependent._validate_iteration_independent_inputs(
        prob, tiny_functional_algorithm, P, T, p, t, 0, 0
    )
    assert h == 0
    assert alpha == 0


def test_validate_iteration_independent_inputs_rejects_mismatch(tiny_functional_algorithm):
    prob = InclusionProblem([MaximallyMonotone()])
    P = np.eye(3)
    T = np.eye(4)
    with pytest.raises(ValueError):
        IterationIndependent._validate_iteration_independent_inputs(
            prob, tiny_functional_algorithm, P, T, np.zeros(2), np.zeros(3), 0, 0
        )


def test_validate_iteration_independent_inputs_requires_p_t_for_functional(tiny_functional_algorithm):
    prob = InclusionProblem([Convex()])
    P = np.eye(3)
    T = np.eye(4)
    with pytest.raises(ValueError):
        IterationIndependent._validate_iteration_independent_inputs(
            prob, tiny_functional_algorithm, P, T, None, None, 0, 0
        )


def test_validate_iteration_independent_inputs_rejects_p_t_for_operator_only(tiny_operator_algorithm):
    prob = InclusionProblem([MaximallyMonotone()])
    P = np.eye(3)
    T = np.eye(4)
    with pytest.raises(ValueError):
        IterationIndependent._validate_iteration_independent_inputs(
            prob, tiny_operator_algorithm, P, T, np.zeros(1), np.zeros(1), 0, 0
        )


def test_linear_convergence_distance_to_solution_shapes(tiny_functional_algorithm):
    P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
        tiny_functional_algorithm, h=0, alpha=0, i=1, j=1, tau=0
    )
    assert P.shape == (3, 3)
    assert T.shape == (4, 4)
    assert p.shape == (2,)
    assert t.shape == (3,)
    assert np.allclose(T, 0.0)


def test_linear_convergence_distance_to_solution_rejects_tau(tiny_functional_algorithm):
    with pytest.raises(ValueError):
        IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
            tiny_functional_algorithm, h=0, alpha=0, i=1, j=1, tau=1
        )


@pytest.mark.parametrize(
    "i,j",
    [
        (0, 1),
        (2, 1),
        (1, 2),
    ],
    ids=["i_zero", "i_too_large", "j_too_large"],
)
def test_linear_convergence_distance_to_solution_rejects_bad_indices(tiny_functional_algorithm, i, j):
    with pytest.raises(ValueError):
        IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
            tiny_functional_algorithm, h=0, alpha=0, i=i, j=j, tau=0
        )


def test_linear_convergence_function_value_suboptimality_t_values(tiny_functional_algorithm):
    _P, _p, _T, t = IterationIndependent.LinearConvergence.get_parameters_function_value_suboptimality(
        tiny_functional_algorithm, h=0, alpha=0, j=1, tau=0
    )
    assert np.allclose(t, 0.0)


def test_linear_convergence_function_value_suboptimality_rejects_bad_j(tiny_functional_algorithm):
    with pytest.raises(ValueError):
        IterationIndependent.LinearConvergence.get_parameters_function_value_suboptimality(
            tiny_functional_algorithm, h=0, alpha=0, j=2, tau=0
        )


def test_sublinear_fixed_point_residual_matches_diff(tiny_functional_algorithm):
    P, p, T, t = IterationIndependent.SublinearConvergence.get_parameters_fixed_point_residual(
        tiny_functional_algorithm, h=0, alpha=0, tau=0
    )
    assert P.shape == (3, 3)
    assert T.shape == (4, 4)
    assert p.shape == (2,)
    assert t.shape == (3,)
    Xs = tiny_functional_algorithm._get_Xs(0, 1)
    expected_T = (Xs[1] - Xs[0]).T @ (Xs[1] - Xs[0])
    assert np.allclose(T, expected_T)


def test_sublinear_fixed_point_residual_rejects_bad_tau(tiny_functional_algorithm):
    with pytest.raises(ValueError):
        IterationIndependent.SublinearConvergence.get_parameters_fixed_point_residual(
            tiny_functional_algorithm, h=0, alpha=0, tau=2
        )


def test_sublinear_duality_gap_rejects_operator_only(tiny_operator_algorithm):
    with pytest.raises(ValueError):
        IterationIndependent.SublinearConvergence.get_parameters_duality_gap(
            tiny_operator_algorithm, h=0, alpha=0, tau=0
        )


def test_sublinear_duality_gap_rejects_bad_tau(tiny_functional_algorithm):
    with pytest.raises(ValueError):
        IterationIndependent.SublinearConvergence.get_parameters_duality_gap(
            tiny_functional_algorithm, h=0, alpha=0, tau=2
        )


def test_sublinear_duality_gap_t_values(tiny_functional_algorithm):
    _P, _p, _T, t = IterationIndependent.SublinearConvergence.get_parameters_duality_gap(
        tiny_functional_algorithm, h=0, alpha=0, tau=0
    )
    assert np.allclose(t, np.array([1.0, 0.0, -1.0]))


def test_iteration_independent_compute_thetas_invalid_condition(tiny_functional_algorithm):
    with pytest.raises(ValueError):
        IterationIndependent._compute_Thetas(
            tiny_functional_algorithm, h=0, alpha=0, condition="C2"
        )


def test_iteration_independent_compute_thetas_shapes(tiny_functional_algorithm):
    Theta0, Theta1 = IterationIndependent._compute_Thetas(
        tiny_functional_algorithm, h=0, alpha=0, condition="C1"
    )
    assert Theta0.shape == (3, 4)
    assert Theta1.shape == (3, 4)
