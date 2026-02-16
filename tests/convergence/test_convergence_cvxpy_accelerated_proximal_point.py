import numpy as np
import pytest

from autolyap.algorithms import AcceleratedProximalPoint
from autolyap.iteration_dependent import IterationDependent
from autolyap.problemclass import Convex, InclusionProblem, MaximallyMonotone


pytestmark = pytest.mark.filterwarnings("ignore:Solution may be inaccurate.*:UserWarning")


def test_convergence_accelerated_proximal_point_operator_mode_c_bounded_by_kim_rate_cvxpy_clarabel(
    cvxpy_clarabel_solver_options,
):
    problem_operator = InclusionProblem([MaximallyMonotone()])
    algorithm_operator = AcceleratedProximalPoint(gamma=1.0, type="operator")

    Ys0 = algorithm_operator._get_Ys(0, 0)
    Xs0 = algorithm_operator._get_Xs(0, 0)
    diff0 = Xs0[0][1, :] - Ys0["star"][0, :]
    Q_0_operator = np.outer(diff0, diff0)

    for k in range(1, 8):
        Xs = algorithm_operator._get_Xs(k, k)
        diff = Xs[k + 1][0, :] - Xs[k][1, :]
        Q_k_operator = np.outer(diff, diff)

        result = IterationDependent.search_lyapunov(
            problem_operator,
            algorithm_operator,
            k,
            Q_0_operator,
            Q_k_operator,
            solver_options=cvxpy_clarabel_solver_options,
        )

        assert result["status"] == "feasible"
        assert result["c_K"] is not None
        kim_bound = 1.0 / (k + 1) ** 2
        assert result["c_K"] == pytest.approx(kim_bound, abs=2e-4)


def test_convergence_accelerated_proximal_point_function_mode_c_bounded_by_kim_rate_cvxpy_clarabel(
    cvxpy_clarabel_solver_options,
):
    problem_function = InclusionProblem([Convex()])
    algorithm_function = AcceleratedProximalPoint(gamma=1.0, type="function")

    Ys0 = algorithm_function._get_Ys(0, 0)
    Xs0 = algorithm_function._get_Xs(0, 0)
    diff0 = Xs0[0][1, :] - Ys0["star"][0, :]
    Q_0_function = np.outer(diff0, diff0)
    q_0_function = np.zeros(algorithm_function.m_bar_func + algorithm_function.m_func)

    for k in range(1, 8):
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
            solver_options=cvxpy_clarabel_solver_options,
        )

        assert result["status"] == "feasible"
        assert result["c_K"] is not None
        kim_bound = 1.0 / (k + 1) ** 2
        assert result["c_K"] == pytest.approx(kim_bound, abs=2e-4)
