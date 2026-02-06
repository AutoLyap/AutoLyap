import numpy as np
from autolyap.algorithms import ProximalPoint
from autolyap.problemclass import InclusionProblem, StronglyConvex
from autolyap.iteration_independent import IterationIndependent
from tests.mosek_utils import require_mosek_license


# Convergence-rate test for the proximal point method using MOSEK.
def test_convergence_proximal_point_rho_matches_theory():
    require_mosek_license()
    mu = 1.0
    problem = InclusionProblem([StronglyConvex(mu)])
    algorithm = ProximalPoint(gamma=1.0)

    gammas = np.linspace(0.05, 0.95, 10)
    for gamma in gammas:
        algorithm.set_gamma(gamma)
        P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(algorithm)
        result = IterationIndependent.LinearConvergence.bisection_search_rho(
            problem,
            algorithm,
            P,
            T,
            p=p,
            t=t,
            S_equals_T=True,
            s_equals_t=True,
            remove_C3=True,
        )
        assert result["success"]
        assert result["certificate"] is not None
        assert "Q" in result["certificate"]
        assert "S" in result["certificate"]
        assert "multipliers" in result["certificate"]
        rho_al = result["rho"]
        assert rho_al is not None
        rho_theoretical = (1 / (1 + gamma * mu)) ** 2
        assert abs(rho_al - rho_theoretical) < 1e-5
