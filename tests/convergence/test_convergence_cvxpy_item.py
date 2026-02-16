import numpy as np
import pytest

from autolyap.algorithms import ITEM
from autolyap.iteration_dependent import IterationDependent
from autolyap.problemclass import InclusionProblem, SmoothStronglyConvex


pytestmark = pytest.mark.filterwarnings("ignore:Solution may be inaccurate.*:UserWarning")


def test_convergence_item_c_matches_theoretical_bound_cvxpy_clarabel(
    cvxpy_clarabel_solver_options,
):
    mu = 1.0
    L = 200.0
    q = mu / L

    problem = InclusionProblem([SmoothStronglyConvex(mu=mu, L=L)])
    algorithm = ITEM(L=L, mu=mu)

    Ys0 = algorithm._get_Ys(0, 0)
    Xs0 = algorithm._get_Xs(0, 0)
    Q_0 = (Xs0[0][1, :] - Ys0["star"]).T @ (Xs0[0][1, :] - Ys0["star"])
    q_0 = np.zeros(algorithm.m_bar_func + algorithm.m_func)

    for k in range(1, 11):
        A_k = algorithm.get_A(k)
        bound_theoretical = 1.0 / (1.0 + q * A_k)

        Ysk = algorithm._get_Ys(k, k)
        Xsk = algorithm._get_Xs(k, k)
        Q_k = (Xsk[k][1, :] - Ysk["star"]).T @ (Xsk[k][1, :] - Ysk["star"])
        q_k = np.zeros(algorithm.m_bar_func + algorithm.m_func)

        result = IterationDependent.search_lyapunov(
            problem,
            algorithm,
            k,
            Q_0,
            Q_k,
            q_0=q_0,
            q_K=q_k,
            solver_options=cvxpy_clarabel_solver_options,
        )

        assert result["status"] == "feasible"
        assert result["c_K"] is not None
        assert result["c_K"] == pytest.approx(bound_theoretical, abs=1e-3)
