import numpy as np
import pytest

from autolyap import IterationDependent, IterationIndependent
from autolyap.algorithms import GradientMethod, OptimizedGradientMethod
from autolyap.problemclass import InclusionProblem, SmoothConvex, SmoothStronglyConvex
from tests.shared.cvxpy_fixtures import cvxpy_open_source_solver_name, cvxpy_open_source_solver_options
from tests.shared.cvxpy_test_utils import make_cvxpy_solver_options


pytestmark = pytest.mark.filterwarnings("ignore:Solution may be inaccurate.*:UserWarning")


def _rho_equivalence_tolerance(solver_name: str) -> float:
    return 1e-4 if solver_name == "CLARABEL" else 5e-3


def _c_equivalence_tolerance(solver_name: str) -> float:
    return 1e-6 if solver_name == "CLARABEL" else 1e-2


def test_iteration_independent_bisection_cvxpy_warm_start_equivalence_open_source(
    cvxpy_open_source_solver_name, cvxpy_open_source_solver_options
):
    mu = 1.0
    L = 4.0
    gamma = 0.2
    problem = InclusionProblem([SmoothStronglyConvex(mu, L)])
    algorithm = GradientMethod(gamma=gamma)
    P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
        algorithm
    )

    warm_options = cvxpy_open_source_solver_options
    cold_options = make_cvxpy_solver_options(
        cvxpy_open_source_solver_name,
        extra_params={"warm_start": False},
    )

    kwargs = dict(
        prob=problem,
        algo=algorithm,
        P=P,
        T=T,
        p=p,
        t=t,
        S_equals_T=True,
        s_equals_t=True,
        remove_C3=True,
        lower_bound=0.0,
        upper_bound=1.0,
        tol=1e-6,
    )
    result_warm = IterationIndependent.LinearConvergence.bisection_search_rho(
        **kwargs,
        solver_options=warm_options,
    )
    result_cold = IterationIndependent.LinearConvergence.bisection_search_rho(
        **kwargs,
        solver_options=cold_options,
    )

    assert result_warm["success"] and result_cold["success"]
    assert result_warm["rho"] is not None and result_cold["rho"] is not None
    assert np.isclose(
        result_warm["rho"],
        result_cold["rho"],
        atol=_rho_equivalence_tolerance(cvxpy_open_source_solver_name),
    )


def test_iteration_dependent_verify_cvxpy_warm_start_equivalence_open_source(
    cvxpy_open_source_solver_name, cvxpy_open_source_solver_options
):
    L = 1.0
    K = 1
    problem = InclusionProblem([SmoothConvex(L)])
    algorithm = OptimizedGradientMethod(L=L, K=K)
    Q_0, q_0 = IterationDependent.get_parameters_distance_to_solution(
        algorithm, 0, i=1, j=1
    )
    Q_K, q_K = IterationDependent.get_parameters_function_value_suboptimality(
        algorithm, K
    )

    warm_options = cvxpy_open_source_solver_options
    cold_options = make_cvxpy_solver_options(
        cvxpy_open_source_solver_name,
        extra_params={"warm_start": False},
    )

    kwargs = dict(
        prob=problem,
        algo=algorithm,
        K=K,
        Q_0=Q_0,
        Q_K=Q_K,
        q_0=q_0,
        q_K=q_K,
    )
    result_warm = IterationDependent.verify_iteration_dependent_Lyapunov(
        **kwargs,
        solver_options=warm_options,
    )
    result_cold = IterationDependent.verify_iteration_dependent_Lyapunov(
        **kwargs,
        solver_options=cold_options,
    )

    assert result_warm["success"] and result_cold["success"]
    assert result_warm["c"] is not None and result_cold["c"] is not None
    assert np.isclose(
        result_warm["c"],
        result_cold["c"],
        atol=_c_equivalence_tolerance(cvxpy_open_source_solver_name),
    )
