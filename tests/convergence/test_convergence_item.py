import numpy as np
import pytest

from autolyap.algorithms import ITEM
from autolyap.problemclass import InclusionProblem, SmoothStronglyConvex
from autolyap.iteration_dependent import IterationDependent
from tests.shared.mosek_utils import require_mosek_license


# Iteration-dependent convergence test for ITEM.
# Mirrors the old experiment:
#   V(0) = ||z^0 - z^*||^2, V(k) = ||z^k - z^*||^2,
# with theoretical bound c_k = 1 / (1 + q A_k), q = mu / L.
def test_convergence_item_c_matches_theoretical_bound():
    require_mosek_license()

    mu = 1.0
    L = 200.0
    q = mu / L

    problem = InclusionProblem([SmoothStronglyConvex(mu=mu, L=L)])
    algorithm = ITEM(L=L, mu=mu)

    Ys0 = algorithm.get_Ys(0, 0)
    Xs0 = algorithm.get_Xs(0, 0)
    Q_0 = (Xs0[0][1, :] - Ys0["star"]).T @ (Xs0[0][1, :] - Ys0["star"])
    q_0 = np.zeros(algorithm.m_bar_func + algorithm.m_func)

    for k in range(1, 21):
        A_k = algorithm.get_A(k)
        bound_theoretical = 1.0 / (1.0 + q * A_k)

        Ysk = algorithm.get_Ys(k, k)
        Xsk = algorithm.get_Xs(k, k)
        Q_k = (Xsk[k][1, :] - Ysk["star"]).T @ (Xsk[k][1, :] - Ysk["star"])
        q_k = np.zeros(algorithm.m_bar_func + algorithm.m_func)

        result = IterationDependent.verify_iteration_dependent_Lyapunov(
            problem,
            algorithm,
            k,
            Q_0,
            Q_k,
            q_0=q_0,
            q_K=q_k,
        )

        assert result["success"] is True
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
        assert result["c_K"] == pytest.approx(bound_theoretical, abs=1e-6)
