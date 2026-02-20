# Equivalence checks between CVXPY+SDPA regular and multiprecision profiles.

import numpy as np
import pytest

from autolyap import IterationDependent, IterationIndependent
from autolyap.algorithms import GradientMethod, OptimizedGradientMethod
from autolyap.problemclass import InclusionProblem, SmoothConvex, SmoothStronglyConvex


pytestmark = pytest.mark.filterwarnings("ignore:Solution may be inaccurate.*:UserWarning")


def test_iteration_independent_verify_cvxpy_sdpa_regular_vs_multiprecision(
    cvxpy_sdpa_solver_options, cvxpy_sdpa_multiprecision_solver_options
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
    result_regular = IterationIndependent.search_lyapunov(
        **kwargs,
        solver_options=cvxpy_sdpa_solver_options,
    )
    result_multiprecision = IterationIndependent.search_lyapunov(
        **kwargs,
        solver_options=cvxpy_sdpa_multiprecision_solver_options,
    )

    assert result_regular["status"] == "feasible"
    assert result_multiprecision["status"] == "feasible"
    assert np.isclose(result_regular["rho"], result_multiprecision["rho"], atol=1e-8)

    cert_regular = result_regular["certificate"]
    cert_multiprecision = result_multiprecision["certificate"]
    assert cert_regular is not None and cert_multiprecision is not None
    assert np.allclose(cert_regular["S"], cert_multiprecision["S"], atol=1e-6)
    assert np.allclose(cert_regular["s"], cert_multiprecision["s"], atol=1e-6)


def test_iteration_dependent_verify_cvxpy_sdpa_regular_vs_multiprecision(
    cvxpy_sdpa_solver_options, cvxpy_sdpa_multiprecision_solver_options
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
    result_regular = IterationDependent.search_lyapunov(
        **kwargs,
        solver_options=cvxpy_sdpa_solver_options,
    )
    result_multiprecision = IterationDependent.search_lyapunov(
        **kwargs,
        solver_options=cvxpy_sdpa_multiprecision_solver_options,
    )

    assert result_regular["status"] == "feasible"
    assert result_multiprecision["status"] == "feasible"
    assert result_regular["c_K"] is not None
    assert result_multiprecision["c_K"] is not None
    assert np.isclose(result_regular["c_K"], result_multiprecision["c_K"], atol=1e-6)
