import pytest

from autolyap import IterationDependent
from autolyap.algorithms import OptimizedGradientMethod
from autolyap.problemclass import InclusionProblem, SmoothConvex


pytestmark = [
    pytest.mark.public_api,
    pytest.mark.filterwarnings("ignore:Solution may be inaccurate.*:UserWarning"),
]


def test_convergence_optimized_gradient_method_c_matches_theory_cvxpy_clarabel(
    cvxpy_clarabel_solver_options,
):
    L = 1.0
    problem = InclusionProblem([SmoothConvex(L)])
    algorithm = OptimizedGradientMethod(L=L, K=1)
    rel_tol, abs_tol = 0.15, 5e-5

    for k in range(1, 7):
        algorithm.set_K(k)
        Q_0, q_0 = IterationDependent.get_parameters_distance_to_solution(
            algorithm, 0, i=1, j=1
        )
        Q_k, q_k = IterationDependent.get_parameters_function_value_suboptimality(
            algorithm, k
        )
        result = IterationDependent.search_lyapunov(
            problem,
            algorithm,
            k,
            Q_0,
            Q_k,
            q_0=q_0,
            q_K=q_k,
            solver_options=cvxpy_clarabel_solver_options,
        )
        assert result["success"] is True
        assert result["c_K"] is not None

        theta_k = algorithm.compute_theta(k, k)
        c_theory = L / (2.0 * theta_k ** 2)
        assert result["c_K"] == pytest.approx(c_theory, rel=rel_tol, abs=abs_tol)
