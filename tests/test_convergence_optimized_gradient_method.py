import numpy as np
import pytest
from autolyap.algorithms import OptimizedGradientMethod
from autolyap.problemclass import InclusionProblem, SmoothConvex
from autolyap.iteration_dependent import IterationDependent
from tests.mosek_utils import require_mosek_license


# Convergence-rate test for the optimized gradient method using MOSEK.
def test_convergence_optimized_gradient_method_c_matches_theory_first_10_ks():
    require_mosek_license()
    L = 1.0
    problem = InclusionProblem([SmoothConvex(L)])
    algorithm = OptimizedGradientMethod(L=L, K=1)

    for k in range(1, 11):
        algorithm.set_K(k)
        Q_0, q_0 = IterationDependent.get_parameters_distance_to_solution(algorithm, 0, i=1, j=1)
        theta_K = algorithm._compute_theta(k, k)
        bound_theoretical = L / (2 * theta_K ** 2)

        Q_k, q_k = IterationDependent.get_parameters_function_value_suboptimality(algorithm, k)
        ok, c = IterationDependent.verify_iteration_dependent_Lyapunov(
            problem,
            algorithm,
            k,
            Q_0,
            Q_k,
            q_0=q_0,
            q_K=q_k,
        )
        assert ok is True
        assert c is not None
        assert c == pytest.approx(bound_theoretical, rel=0.1, abs=1e-5)
