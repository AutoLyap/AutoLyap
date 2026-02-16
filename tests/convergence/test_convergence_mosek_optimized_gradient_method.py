import numpy as np
import pytest
from autolyap.algorithms import OptimizedGradientMethod
from autolyap.problemclass import InclusionProblem, SmoothConvex
from autolyap.iteration_dependent import IterationDependent
from tests.shared.mosek_utils import require_mosek_license

pytestmark = [pytest.mark.mosek, pytest.mark.public_api]


# Convergence-rate test for the optimized gradient method using MOSEK.
def test_convergence_optimized_gradient_method_c_matches_theory_first_10_ks():
    require_mosek_license()
    L = 1.0
    problem = InclusionProblem([SmoothConvex(L)])
    algorithm = OptimizedGradientMethod(L=L, K=1)

    for k in range(1, 11):
        algorithm.set_K(k)
        Q_0, q_0 = IterationDependent.get_parameters_distance_to_solution(algorithm, 0, i=1, j=1)
        theta_K = algorithm.compute_theta(k, k)
        bound_theoretical = L / (2 * theta_K ** 2)

        Q_k, q_k = IterationDependent.get_parameters_function_value_suboptimality(algorithm, k)
        result = IterationDependent.search_lyapunov(
            problem,
            algorithm,
            k,
            Q_0,
            Q_k,
            q_0=q_0,
            q_K=q_k,
        )
        assert result["status"] == "feasible"
        assert result["c_K"] is not None
        assert result["certificate"] is not None
        certificate = result["certificate"]
        assert len(certificate["Q_sequence"]) == k + 1
        assert np.allclose(certificate["Q_sequence"][0], Q_0)
        assert np.allclose(certificate["Q_sequence"][-1], Q_k)
        assert certificate["q_sequence"] is not None
        assert len(certificate["q_sequence"]) == k + 1
        assert np.allclose(certificate["q_sequence"][0], q_0)
        assert np.allclose(certificate["q_sequence"][-1], q_k)
        assert result["c_K"] == pytest.approx(bound_theoretical, rel=0.1, abs=1e-5)
