import numpy as np
from autolyap.algorithms import GradientMethod
from autolyap.problemclass import InclusionProblem, SmoothStronglyConvex
from autolyap.iteration_independent import IterationIndependent
from tests.mosek_utils import require_mosek_license


# Convergence-rate test for the gradient method using MOSEK.
def test_convergence_gradient_method_rho_matches_theory():
    require_mosek_license()
    mu = 1.0
    L = 4.0
    problem = InclusionProblem([SmoothStronglyConvex(mu, L)])
    algorithm = GradientMethod(gamma=1.0)

    gammas = np.linspace(0.05, 0.45, 10)
    for gamma in gammas:
        algorithm.set_gamma(gamma)
        P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(algorithm)
        rho_al = IterationIndependent.LinearConvergence.bisection_search_rho(
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
        assert rho_al is not None
        rho_theoretical = max(gamma * L - 1, 1 - gamma * mu) ** 2
        assert abs(rho_al - rho_theoretical) < 1e-5
