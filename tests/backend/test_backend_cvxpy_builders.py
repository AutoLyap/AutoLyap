from autolyap import IterationDependent, IterationIndependent
from autolyap.problemclass import Convex, InclusionProblem, MaximallyMonotone
from tests.shared.cvxpy_fixtures import cvxpy_module


def test_iteration_independent_cvxpy_builder_respects_fixed_flags(
    tiny_functional_algorithm, cvxpy_module
):
    problem = InclusionProblem([Convex()])
    P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
        tiny_functional_algorithm
    )
    h, alpha, *_ = IterationIndependent._validate_iteration_independent_inputs(
        problem, tiny_functional_algorithm, P, T, p, t, 0, 0
    )
    _, handles = IterationIndependent._build_iteration_independent_problem_cvxpy(
        problem,
        tiny_functional_algorithm,
        P,
        T,
        p,
        t,
        h,
        alpha,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        rho_term=1.0,
        cp=cvxpy_module,
    )
    assert handles["Q_var"] is None
    assert handles["S_var"] is None
    assert handles["q_var"] is None
    assert handles["s_var"] is None


def test_iteration_independent_cvxpy_builder_creates_variables_when_unfixed(
    tiny_functional_algorithm, cvxpy_module
):
    problem = InclusionProblem([Convex()])
    P, p, T, t = IterationIndependent.LinearConvergence.get_parameters_distance_to_solution(
        tiny_functional_algorithm
    )
    h, alpha, *_ = IterationIndependent._validate_iteration_independent_inputs(
        problem, tiny_functional_algorithm, P, T, p, t, 0, 0
    )
    _, handles = IterationIndependent._build_iteration_independent_problem_cvxpy(
        problem,
        tiny_functional_algorithm,
        P,
        T,
        p,
        t,
        h,
        alpha,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        rho_term=1.0,
        cp=cvxpy_module,
    )
    assert handles["Q_var"] is not None
    assert handles["S_var"] is not None
    assert handles["q_var"] is not None
    assert handles["s_var"] is not None


def test_iteration_dependent_cvxpy_builder_creates_internal_sequences(
    tiny_functional_algorithm, cvxpy_module
):
    problem = InclusionProblem([Convex()])
    Q_0, q_0 = IterationDependent.get_parameters_distance_to_solution(
        tiny_functional_algorithm, k=0, i=1, j=1
    )
    Q_K, q_K = IterationDependent.get_parameters_function_value_suboptimality(
        tiny_functional_algorithm, k=2, j=1
    )
    K, _, _, _, _, m_func, m_op, _, dim_Q, dim_q = IterationDependent._validate_iteration_dependent_inputs(
        problem, tiny_functional_algorithm, 2, Q_0, Q_K, q_0, q_K
    )
    _, handles = IterationDependent._build_iteration_dependent_problem_cvxpy(
        problem,
        tiny_functional_algorithm,
        K,
        Q_0,
        Q_K,
        q_0,
        q_K,
        dim_Q,
        dim_q,
        m_func,
        m_op,
        cp=cvxpy_module,
    )
    assert len(handles["Q_vars"]) == 1
    assert len(handles["q_vars"]) == 1


def test_iteration_dependent_cvxpy_builder_operator_only_has_no_q_vars(
    tiny_operator_algorithm, cvxpy_module
):
    problem = InclusionProblem([MaximallyMonotone()])
    Q_0 = IterationDependent.get_parameters_distance_to_solution(
        tiny_operator_algorithm, k=0, i=1, j=1
    )
    Q_K = IterationDependent.get_parameters_distance_to_solution(
        tiny_operator_algorithm, k=2, i=1, j=1
    )
    K, _, _, _, _, m_func, m_op, _, dim_Q, dim_q = IterationDependent._validate_iteration_dependent_inputs(
        problem, tiny_operator_algorithm, 2, Q_0, Q_K, None, None
    )
    _, handles = IterationDependent._build_iteration_dependent_problem_cvxpy(
        problem,
        tiny_operator_algorithm,
        K,
        Q_0,
        Q_K,
        None,
        None,
        dim_Q,
        dim_q,
        m_func,
        m_op,
        cp=cvxpy_module,
    )
    assert handles["q_vars"] == {}
