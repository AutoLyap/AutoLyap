import numpy as np
import pytest

from autolyap.algorithms import DouglasRachford
from autolyap.problemclass import (
    InclusionProblem,
    MaximallyMonotone,
    StronglyMonotone,
    LipschitzOperator,
    Cocoercive,
    SmoothStronglyConvex,
    Convex,
)
from autolyap.iteration_independent import IterationIndependent
from tests.mosek_utils import require_mosek_license


def _bisection_rho(problem, algorithm, h: int = 0, alpha: int = 0):
    if algorithm.m_func > 0:
        P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
            algorithm, h, alpha
        )
        return IterationIndependent.LinearConvergence.bisection_search_rho(
            problem,
            algorithm,
            P,
            T,
            p=p,
            t=t,
            h=h,
            alpha=alpha,
            S_equals_T=True,
            s_equals_t=True,
            remove_C3=True,
        )

    P, T = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
        algorithm, h, alpha
    )
    return IterationIndependent.LinearConvergence.bisection_search_rho(
        problem,
        algorithm,
        P,
        T,
        h=h,
        alpha=alpha,
        S_equals_T=True,
        s_equals_t=True,
        remove_C3=True,
    )


def _giselsson_thm65_delta(mu: float, L: float, gamma: float) -> float:
    return np.sqrt(1 - (4 * gamma * mu) / (1 + 2 * gamma * mu + (gamma * L) ** 2))


def _giselsson_thm65_rate_sq(lambda_value: float, mu: float, L: float, gamma: float) -> float:
    alpha = lambda_value / 2
    delta = _giselsson_thm65_delta(mu, L, gamma)
    return (abs(1 - alpha) + alpha * delta) ** 2


def _giselsson_thm74_delta(mu: float, L: float, gamma: float) -> float:
    return np.sqrt(1 - (4 * gamma * mu) / (1 + 2 * gamma * mu + (gamma**2) * mu * L))


def _giselsson_thm74_rate_sq(lambda_value: float, mu: float, L: float, gamma: float) -> float:
    alpha = lambda_value / 2
    delta = _giselsson_thm74_delta(mu, L, gamma)
    return (abs(1 - alpha) + alpha * delta) ** 2


def _ryu_thm41_rate(mu: float, beta: float, lambda_value: float, gamma: float) -> float:
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    scaled_mu = gamma * mu
    scaled_beta = beta / gamma
    if not (scaled_mu > 0 and scaled_beta > 0 and 0 < lambda_value < 2):
        raise ValueError("invalid scaled parameters")

    if (
        (scaled_mu * scaled_beta - scaled_mu + scaled_beta < 0)
        and (
            lambda_value
            <= 2
            * ((scaled_beta + 1) * (scaled_mu - scaled_beta - scaled_mu * scaled_beta))
            / (
                scaled_mu
                + scaled_mu * scaled_beta
                - scaled_beta
                - scaled_beta**2
                - 2 * scaled_mu * scaled_beta**2
            )
        )
    ):
        return abs(1 - lambda_value * (scaled_beta / (scaled_beta + 1)))

    if (
        (scaled_mu * scaled_beta - scaled_mu - scaled_beta > 0)
        and (
            lambda_value
            <= 2
            * (
                scaled_mu**2
                + scaled_beta**2
                + scaled_mu * scaled_beta
                + scaled_mu
                + scaled_beta
                - scaled_mu**2 * scaled_beta**2
            )
            / (
                scaled_mu**2
                + scaled_beta**2
                + scaled_mu**2 * scaled_beta
                + scaled_mu * scaled_beta**2
                + scaled_mu
                + scaled_beta
                - 2 * scaled_mu**2 * scaled_beta**2
            )
        )
    ):
        return abs(
            1
            - lambda_value
            * ((1 + scaled_mu * scaled_beta) / ((scaled_mu + 1) * (scaled_beta + 1)))
        )

    if lambda_value >= 2 * (scaled_mu * scaled_beta + scaled_mu + scaled_beta) / (
        2 * scaled_mu * scaled_beta + scaled_mu + scaled_beta
    ):
        return abs(1 - lambda_value)

    if (
        (scaled_mu * scaled_beta + scaled_mu - scaled_beta < 0)
        and (
            lambda_value
            <= 2
            * ((scaled_mu + 1) * (scaled_beta - scaled_mu - scaled_mu * scaled_beta))
            / (
                scaled_beta
                + scaled_mu * scaled_beta
                - scaled_mu
                - scaled_mu**2
                - 2 * scaled_mu**2 * scaled_beta
            )
        )
    ):
        return abs(1 - lambda_value * (scaled_mu / (scaled_mu + 1)))

    numerator = (
        ((2 - lambda_value) * scaled_mu * (scaled_beta + 1) + lambda_value * scaled_beta * (1 - scaled_mu))
        * ((2 - lambda_value) * scaled_beta * (scaled_mu + 1) + lambda_value * scaled_mu * (1 - scaled_beta))
    )
    denominator = scaled_mu * scaled_beta * (
        2 * scaled_mu * scaled_beta * (1 - lambda_value) + (2 - lambda_value) * (scaled_mu + scaled_beta + 1)
    )
    return (np.sqrt(2 - lambda_value) / 2) * np.sqrt(numerator / denominator)


def _ryu_thm43_rate(mu: float, lambda_value: float, L: float, gamma: float) -> float:
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    scaled_mu = gamma * mu
    scaled_L = gamma * L
    if not (scaled_mu > 0 and scaled_L > 0 and 0 < lambda_value < 2):
        raise ValueError("invalid scaled parameters")

    term1 = 2 * (lambda_value - 1) * scaled_mu + lambda_value - 2
    term2 = scaled_L**2 * (lambda_value - 2 * (scaled_mu + 1))
    denominator = np.sqrt(term1**2 + scaled_L**2 * (lambda_value - 2 * (scaled_mu + 1)) ** 2)
    condition_a_lhs = scaled_mu * (-term1 + term2) / denominator

    if condition_a_lhs <= np.sqrt(scaled_L**2 + 1):
        numerator_a = lambda_value + np.sqrt(
            (term1**2 + scaled_L**2 * (lambda_value - 2 * (scaled_mu + 1)) ** 2) / (scaled_L**2 + 1)
        )
        return numerator_a / (2 * (scaled_mu + 1))

    condition_b_threshold = (
        2
        * (scaled_mu + 1)
        * (scaled_L + 1)
        * (scaled_mu + scaled_mu * scaled_L**2 - scaled_L**2 - 2 * scaled_mu * scaled_L - 1)
    ) / (
        2 * scaled_mu**2
        - scaled_mu
        + scaled_mu * scaled_L**3
        - scaled_L**3
        - 3 * scaled_mu * scaled_L**2
        - scaled_L**2
        - 2 * scaled_mu**2 * scaled_L
        - scaled_mu * scaled_L
        - scaled_L
        - 1
    )
    if (
        scaled_L < 1
        and scaled_mu > (scaled_L**2 + 1) / ((scaled_L - 1) ** 2)
        and lambda_value <= condition_b_threshold
    ):
        return abs(1 - lambda_value * (scaled_L + scaled_mu) / ((scaled_mu + 1) * (scaled_L + 1)))

    term3 = lambda_value * (scaled_L**2 + 1) - 2 * scaled_mu * (lambda_value + scaled_L**2 - 1)
    term4 = lambda_value * (1 + 2 * scaled_mu + scaled_L**2) - 2 * (scaled_mu + 1) * (scaled_L**2 + 1)
    numerator_c = (2 - lambda_value) * term3 * term4 / (4 * scaled_mu * (scaled_L**2 + 1))
    denominator_c = 2 * scaled_mu * (lambda_value + scaled_L**2 - 1) - (2 - lambda_value) * (1 - scaled_L**2)
    return np.sqrt(numerator_c / denominator_c)


def _giselsson_boyd_delta(L: float, mu: float, gamma: float) -> float:
    return max((gamma * L - 1) / (gamma * L + 1), (1 - gamma * mu) / (1 + gamma * mu))


def _giselsson_boyd_rate_sq(lambda_value: float, mu: float, L: float, gamma: float) -> float:
    delta = _giselsson_boyd_delta(L, mu, gamma)
    return (abs(1 - lambda_value / 2) + (lambda_value / 2) * delta) ** 2


def test_convergence_douglas_rachford_experiment1_operator_mm_strong_lipschitz():
    require_mosek_license()

    mu = 1.0
    L = 2.0
    lambda_value = 2.0
    problem = InclusionProblem(
        [MaximallyMonotone(), [StronglyMonotone(mu=mu), LipschitzOperator(L=L)]]
    )
    algorithm = DouglasRachford(gamma=1.0, lambda_value=lambda_value, type="operator")

    for gamma in np.linspace(0.01, 5.0, 5):
        delta = _giselsson_thm65_delta(mu, L, gamma)
        if (lambda_value / 2) >= (2 / (1 + delta)):
            continue
        algorithm.set_gamma(float(gamma))
        result = _bisection_rho(problem, algorithm)
        assert result["success"]
        assert result["certificate"] is not None
        rho_al = result["rho"]
        assert rho_al is not None
        rho_theoretical = _giselsson_thm65_rate_sq(lambda_value, mu, L, gamma)
        assert rho_al == pytest.approx(rho_theoretical, abs=5e-5)


def test_convergence_douglas_rachford_experiment2_operator_mm_strong_cocoercive():
    require_mosek_license()

    mu = 1.0
    beta = 0.5
    L = 1.0 / beta
    lambda_value = 2.0
    problem = InclusionProblem(
        [MaximallyMonotone(), [StronglyMonotone(mu=mu), Cocoercive(beta=beta)]]
    )
    algorithm = DouglasRachford(gamma=1.0, lambda_value=lambda_value, type="operator")

    for gamma in np.linspace(0.01, 5.0, 5):
        delta = _giselsson_thm74_delta(mu, L, gamma)
        if (lambda_value / 2) >= (2 / (1 + delta)):
            continue
        algorithm.set_gamma(float(gamma))
        result = _bisection_rho(problem, algorithm)
        assert result["success"]
        assert result["certificate"] is not None
        rho_al = result["rho"]
        assert rho_al is not None
        rho_theoretical = _giselsson_thm74_rate_sq(lambda_value, mu, L, gamma)
        assert rho_al == pytest.approx(rho_theoretical, abs=5e-6)


def test_convergence_douglas_rachford_experiment3_operator_cocoercive_strong_lambda_sweep():
    require_mosek_license()

    mu = 1.0
    beta = 1.0
    gamma = 1.0
    problem = InclusionProblem([Cocoercive(beta=beta), StronglyMonotone(mu=mu)])
    algorithm = DouglasRachford(gamma=gamma, lambda_value=2.0, type="operator")

    for lambda_value in np.linspace(0.01, 1.99, 5):
        algorithm.set_lambda(float(lambda_value))
        result = _bisection_rho(problem, algorithm)
        assert result["success"]
        assert result["certificate"] is not None
        rho_al = result["rho"]
        assert rho_al is not None
        rho_theoretical = _ryu_thm41_rate(mu, beta, lambda_value, gamma) ** 2
        assert rho_al == pytest.approx(rho_theoretical, abs=5e-6)


def test_convergence_douglas_rachford_experiment4_operator_mm_lipschitz_strong_lambda_sweep():
    require_mosek_license()

    mu = 1.0
    L = 1.0
    gamma = 1.0
    problem = InclusionProblem(
        [[MaximallyMonotone(), LipschitzOperator(L=L)], StronglyMonotone(mu=mu)]
    )
    algorithm = DouglasRachford(gamma=gamma, lambda_value=2.0, type="operator")

    for lambda_value in np.linspace(0.01, 1.99, 5):
        algorithm.set_lambda(float(lambda_value))
        result = _bisection_rho(problem, algorithm)
        assert result["success"]
        assert result["certificate"] is not None
        rho_al = result["rho"]
        assert rho_al is not None
        rho_theoretical = _ryu_thm43_rate(mu, lambda_value, L, gamma) ** 2
        assert rho_al == pytest.approx(rho_theoretical, abs=5e-6)


def test_convergence_douglas_rachford_experiment5_function_smooth_strong_plus_convex():
    require_mosek_license()

    mu = 1.0
    L = 2.0
    lambda_value = 1.0
    problem = InclusionProblem([SmoothStronglyConvex(mu=mu, L=L), Convex()])
    algorithm = DouglasRachford(gamma=1.0, lambda_value=lambda_value, type="function")

    for gamma in np.linspace(0.01, 5.0, 5):
        delta = _giselsson_boyd_delta(L, mu, gamma)
        if (lambda_value / 2) >= (2 / (1 + delta)):
            continue
        algorithm.set_gamma(float(gamma))
        result = _bisection_rho(problem, algorithm)
        assert result["success"]
        assert result["certificate"] is not None
        rho_al = result["rho"]
        assert rho_al is not None
        rho_theoretical = _giselsson_boyd_rate_sq(lambda_value, mu, L, gamma)
        assert rho_al == pytest.approx(rho_theoretical, abs=5e-6)
