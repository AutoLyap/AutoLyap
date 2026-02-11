import numpy as np
import pytest

from autolyap import IterationDependent, IterationIndependent
from autolyap.algorithms import GradientMethod, OptimizedGradientMethod
from autolyap.problemclass import InclusionProblem, SmoothConvex, SmoothStronglyConvex
from tests.shared.cvxpy_fixtures import cvxpy_mosek_solver_options, mosek_fusion_solver_options


pytestmark = pytest.mark.mosek


def test_iteration_independent_verify_cross_backend_equivalence(
    mosek_fusion_solver_options, cvxpy_mosek_solver_options
):
    mu = 1.0
    L = 4.0
    gamma = 0.2
    problem = InclusionProblem([SmoothStronglyConvex(mu, L)])
    algorithm = GradientMethod(gamma=gamma)
    P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
        algorithm
    )
    rho_theoretical = max(gamma * L - 1.0, 1.0 - gamma * mu) ** 2

    kwargs = dict(
        prob=problem,
        algo=algorithm,
        P=P,
        T=T,
        p=p,
        t=t,
        rho=rho_theoretical,
        S_equals_T=True,
        s_equals_t=True,
        remove_C3=True,
    )
    result_mosek = IterationIndependent.verify_iteration_independent_Lyapunov(
        **kwargs,
        solver_options=mosek_fusion_solver_options,
    )
    result_cvxpy = IterationIndependent.verify_iteration_independent_Lyapunov(
        **kwargs,
        solver_options=cvxpy_mosek_solver_options,
    )

    assert result_mosek["success"] and result_cvxpy["success"]
    assert np.isclose(result_mosek["rho"], result_cvxpy["rho"], atol=1e-12)

    cert_mosek = result_mosek["certificate"]
    cert_cvxpy = result_cvxpy["certificate"]
    assert cert_mosek is not None and cert_cvxpy is not None
    assert np.allclose(cert_mosek["S"], cert_cvxpy["S"], atol=1e-9)
    assert np.allclose(cert_mosek["s"], cert_cvxpy["s"], atol=1e-9)


def test_iteration_independent_bisection_cross_backend_equivalence(
    mosek_fusion_solver_options, cvxpy_mosek_solver_options
):
    mu = 1.0
    L = 4.0
    gamma = 0.2
    problem = InclusionProblem([SmoothStronglyConvex(mu, L)])
    algorithm = GradientMethod(gamma=gamma)
    P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
        algorithm
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
    result_mosek = IterationIndependent.LinearConvergence.bisection_search_rho(
        **kwargs,
        solver_options=mosek_fusion_solver_options,
    )
    result_cvxpy = IterationIndependent.LinearConvergence.bisection_search_rho(
        **kwargs,
        solver_options=cvxpy_mosek_solver_options,
    )

    assert result_mosek["success"] and result_cvxpy["success"]
    assert result_mosek["rho"] is not None and result_cvxpy["rho"] is not None
    assert np.isclose(result_mosek["rho"], result_cvxpy["rho"], atol=1e-7)

    cert_mosek = result_mosek["certificate"]
    cert_cvxpy = result_cvxpy["certificate"]
    assert cert_mosek is not None and cert_cvxpy is not None
    assert np.allclose(cert_mosek["S"], cert_cvxpy["S"], atol=1e-9)
    assert np.allclose(cert_mosek["s"], cert_cvxpy["s"], atol=1e-9)


def test_iteration_dependent_verify_cross_backend_equivalence(
    mosek_fusion_solver_options, cvxpy_mosek_solver_options
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

    kwargs = dict(
        prob=problem,
        algo=algorithm,
        K=K,
        Q_0=Q_0,
        Q_K=Q_K,
        q_0=q_0,
        q_K=q_K,
    )
    result_mosek = IterationDependent.verify_iteration_dependent_Lyapunov(
        **kwargs,
        solver_options=mosek_fusion_solver_options,
    )
    result_cvxpy = IterationDependent.verify_iteration_dependent_Lyapunov(
        **kwargs,
        solver_options=cvxpy_mosek_solver_options,
    )

    assert result_mosek["success"] and result_cvxpy["success"]
    assert result_mosek["c_K"] is not None
    assert result_cvxpy["c_K"] is not None
    assert np.isclose(result_mosek["c_K"], result_cvxpy["c_K"], atol=1e-7)

    cert_mosek = result_mosek["certificate"]
    cert_cvxpy = result_cvxpy["certificate"]
    assert cert_mosek is not None and cert_cvxpy is not None
    assert len(cert_mosek["Q_sequence"]) == len(cert_cvxpy["Q_sequence"]) == K + 1
    assert len(cert_mosek["q_sequence"]) == len(cert_cvxpy["q_sequence"]) == K + 1
    assert np.allclose(cert_mosek["Q_sequence"][0], cert_cvxpy["Q_sequence"][0], atol=1e-9)
    assert np.allclose(cert_mosek["Q_sequence"][-1], cert_cvxpy["Q_sequence"][-1], atol=1e-9)
    assert np.allclose(cert_mosek["q_sequence"][0], cert_cvxpy["q_sequence"][0], atol=1e-9)
    assert np.allclose(cert_mosek["q_sequence"][-1], cert_cvxpy["q_sequence"][-1], atol=1e-9)
