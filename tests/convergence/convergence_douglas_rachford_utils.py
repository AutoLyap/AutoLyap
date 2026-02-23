"""Shared helper formulas and solve wrappers for Douglas-Rachford convergence tests."""

import numpy as np

from autolyap.iteration_independent import IterationIndependent


def bisection_rho(problem, algorithm, h: int = 0, alpha: int = 0, solver_options=None):
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
            solver_options=solver_options,
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
        solver_options=solver_options,
    )


def dr_maximally_monotone_plus_strongly_monotone_lipschitz_delta(mu: float, L: float, gamma: float) -> float:
    return np.sqrt(1 - (4 * gamma * mu) / (1 + 2 * gamma * mu + (gamma * L) ** 2))


def dr_maximally_monotone_plus_strongly_monotone_lipschitz_rate_sq(lambda_value: float, mu: float, L: float, gamma: float) -> float:
    alpha = lambda_value / 2
    delta = dr_maximally_monotone_plus_strongly_monotone_lipschitz_delta(mu, L, gamma)
    return (abs(1 - alpha) + alpha * delta) ** 2


def dr_maximally_monotone_plus_strongly_monotone_cocoercive_delta(mu: float, L: float, gamma: float) -> float:
    return np.sqrt(1 - (4 * gamma * mu) / (1 + 2 * gamma * mu + (gamma**2) * mu * L))


def dr_maximally_monotone_plus_strongly_monotone_cocoercive_rate_sq(lambda_value: float, mu: float, L: float, gamma: float) -> float:
    alpha = lambda_value / 2
    delta = dr_maximally_monotone_plus_strongly_monotone_cocoercive_delta(mu, L, gamma)
    return (abs(1 - alpha) + alpha * delta) ** 2


def dr_cocoercive_plus_strongly_monotone_rate(mu: float, beta: float, lambda_value: float, gamma: float) -> float:
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


def dr_maximally_monotone_lipschitz_plus_strongly_monotone_rate(mu: float, lambda_value: float, L: float, gamma: float) -> float:
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


def dr_smooth_strongly_convex_plus_convex_delta(L: float, mu: float, gamma: float) -> float:
    return max((gamma * L - 1) / (gamma * L + 1), (1 - gamma * mu) / (1 + gamma * mu))


def dr_smooth_strongly_convex_plus_convex_rate_sq(lambda_value: float, mu: float, L: float, gamma: float) -> float:
    delta = dr_smooth_strongly_convex_plus_convex_delta(L, mu, gamma)
    return (abs(1 - lambda_value / 2) + (lambda_value / 2) * delta) ** 2
