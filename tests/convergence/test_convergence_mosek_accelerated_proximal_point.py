import numpy as np
import pytest

from autolyap.algorithms import AcceleratedProximalPoint
from autolyap.problemclass import InclusionProblem, MaximallyMonotone, Convex
from autolyap.iteration_dependent import IterationDependent
from tests.shared.mosek_utils import require_mosek_license

pytestmark = pytest.mark.mosek


# Iteration-dependent bound test for accelerated proximal point (operator mode).
# The test mirrors the old experiment setup:
#   V(0) = ||y^0 - y^*||^2,  V(k) = ||x^{k+1} - y^k||^2,
# and compares c_k with Kim's 1/(k+1)^2 bound.
def test_convergence_accelerated_proximal_point_operator_mode_c_bounded_by_kim_rate():
    require_mosek_license()

    problem_operator = InclusionProblem([MaximallyMonotone()])
    algorithm_operator = AcceleratedProximalPoint(gamma=1.0, type="operator")

    # V(0) = ||y^0 - y^*||^2 (same construction as in the original experiment code).
    Ys0 = algorithm_operator._get_Ys(0, 0)
    Xs0 = algorithm_operator._get_Xs(0, 0)
    diff0 = Xs0[0][1, :] - Ys0["star"][0, :]
    Q_0_operator = np.outer(diff0, diff0)

    for k in range(1, 11):
        # V(k) = ||x^{k+1} - y^k||^2
        Xs = algorithm_operator._get_Xs(k, k)
        diff = Xs[k + 1][0, :] - Xs[k][1, :]
        Q_k_operator = np.outer(diff, diff)

        result = IterationDependent.search_lyapunov(
            problem_operator,
            algorithm_operator,
            k,
            Q_0_operator,
            Q_k_operator,
        )

        assert result["status"] == "feasible"
        assert result["c_K"] is not None
        assert result["certificate"] is not None
        certificate = result["certificate"]
        assert len(certificate["Q_sequence"]) == k + 1
        assert np.allclose(certificate["Q_sequence"][0], Q_0_operator)
        assert np.allclose(certificate["Q_sequence"][-1], Q_k_operator)
        assert certificate["q_sequence"] is None

        kim_bound = 1.0 / (k + 1) ** 2
        # The SDP certificate should match Kim's bound up to solver tolerances.
        assert result["c_K"] == pytest.approx(kim_bound, abs=5e-6)


def test_convergence_accelerated_proximal_point_function_mode_c_bounded_by_kim_rate():
    require_mosek_license()

    problem_function = InclusionProblem([Convex()])
    algorithm_function = AcceleratedProximalPoint(gamma=1.0, type="function")

    # V(0) = ||y^0 - y^*||^2 (same construction as in the original experiment code).
    Ys0 = algorithm_function._get_Ys(0, 0)
    Xs0 = algorithm_function._get_Xs(0, 0)
    diff0 = Xs0[0][1, :] - Ys0["star"][0, :]
    Q_0_function = np.outer(diff0, diff0)
    q_0_function = np.zeros(algorithm_function.m_bar_func + algorithm_function.m_func)

    for k in range(1, 11):
        # V(k) = ||x^{k+1} - y^k||^2
        Xs = algorithm_function._get_Xs(k, k)
        diff = Xs[k + 1][0, :] - Xs[k][1, :]
        Q_k_function = np.outer(diff, diff)
        q_k_function = np.zeros(algorithm_function.m_bar_func + algorithm_function.m_func)

        result = IterationDependent.search_lyapunov(
            problem_function,
            algorithm_function,
            k,
            Q_0_function,
            Q_k_function,
            q_0=q_0_function,
            q_K=q_k_function,
        )

        assert result["status"] == "feasible"
        assert result["c_K"] is not None
        assert result["certificate"] is not None
        certificate = result["certificate"]
        assert len(certificate["Q_sequence"]) == k + 1
        assert np.allclose(certificate["Q_sequence"][0], Q_0_function)
        assert np.allclose(certificate["Q_sequence"][-1], Q_k_function)
        assert certificate["q_sequence"] is not None
        assert len(certificate["q_sequence"]) == k + 1
        assert np.allclose(certificate["q_sequence"][0], q_0_function)
        assert np.allclose(certificate["q_sequence"][-1], q_k_function)

        kim_bound = 1.0 / (k + 1) ** 2
        # The SDP certificate should match Kim's bound up to solver tolerances.
        assert result["c_K"] == pytest.approx(kim_bound, abs=5e-6)
