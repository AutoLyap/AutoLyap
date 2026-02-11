import pytest

from autolyap import IterationDependent
from autolyap.algorithms import OptimizedGradientMethod
from autolyap.problemclass import InclusionProblem, SmoothConvex
from tests.shared.cvxpy_fixtures import cvxpy_open_source_solver_name, cvxpy_open_source_solver_options


pytestmark = pytest.mark.filterwarnings("ignore:Solution may be inaccurate.*:UserWarning")


def _ogm_tolerance(solver_name: str) -> tuple[float, float]:
    if solver_name == "CLARABEL":
        return 0.15, 5e-5
    return 0.25, 2e-3


def test_convergence_optimized_gradient_method_c_matches_theory_cvxpy(
    cvxpy_open_source_solver_name, cvxpy_open_source_solver_options
):
    L = 1.0
    problem = InclusionProblem([SmoothConvex(L)])
    algorithm = OptimizedGradientMethod(L=L, K=1)
    rel_tol, abs_tol = _ogm_tolerance(cvxpy_open_source_solver_name)

    for k in range(1, 7):
        algorithm.set_K(k)
        Q_0, q_0 = IterationDependent.get_parameters_distance_to_solution(
            algorithm, 0, i=1, j=1
        )
        Q_k, q_k = IterationDependent.get_parameters_function_value_suboptimality(
            algorithm, k
        )
        result = IterationDependent.verify_iteration_dependent_Lyapunov(
            problem,
            algorithm,
            k,
            Q_0,
            Q_k,
            q_0=q_0,
            q_K=q_k,
            solver_options=cvxpy_open_source_solver_options,
        )
        assert result["success"] is True
        assert result["c_K"] is not None

        theta_k = algorithm._compute_theta(k, k)
        c_theory = L / (2.0 * theta_k ** 2)
        assert result["c_K"] == pytest.approx(c_theory, rel=rel_tol, abs=abs_tol)
