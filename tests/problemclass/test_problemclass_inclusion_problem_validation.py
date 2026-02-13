import numpy as np
import pytest

from autolyap.problemclass import (
    Convex,
    MaximallyMonotone,
    GradientDominated,
    InclusionProblem,
)
from autolyap.problemclass.indices import _InterpolationIndices
from autolyap.problemclass.base import (
    _OperatorInterpolationCondition,
    _FunctionInterpolationCondition,
)


# Tests for InclusionProblem validation and component typing rules.
class _BadOperatorCondition(_OperatorInterpolationCondition):
    def get_data(self):
        matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        return [(matrix, _InterpolationIndices("r1<r2"))]


class _BadOperatorNonsymmetric(_OperatorInterpolationCondition):
    def get_data(self):
        matrix = np.array([[1.0, 2.0], [0.0, 1.0]])
        return [(matrix, _InterpolationIndices("r1<r2"))]


class _BadFunctionCondition(_FunctionInterpolationCondition):
    def get_data(self):
        matrix = np.eye(4)
        vector = np.array([1.0, 2.0, 3.0])
        return [(matrix, vector, False, _InterpolationIndices("r1!=r2"))]


def test_inclusionproblem_rejects_empty_components():
    with pytest.raises(ValueError):
        InclusionProblem([])


def test_inclusionproblem_rejects_empty_component_entry():
    with pytest.raises(ValueError):
        InclusionProblem([[]])


def test_inclusionproblem_rejects_invalid_component_type():
    with pytest.raises(ValueError):
        InclusionProblem([object()])


def test_inclusionproblem_rejects_bad_operator_matrix_shape():
    with pytest.raises(ValueError):
        InclusionProblem([_BadOperatorCondition()])


def test_inclusionproblem_rejects_nonsymmetric_operator_matrix():
    with pytest.raises(ValueError):
        InclusionProblem([_BadOperatorNonsymmetric()])


def test_inclusionproblem_rejects_function_vector_length_mismatch():
    with pytest.raises(ValueError):
        InclusionProblem([_BadFunctionCondition()])


def test_inclusionproblem_operator_component_sets_indices():
    prob = InclusionProblem([MaximallyMonotone()])
    assert prob.m == 1
    assert prob.I_op == [1]
    assert prob.I_func == []


def test_inclusionproblem_function_component_sets_indices():
    prob = InclusionProblem([Convex()])
    assert prob.m == 1
    assert prob.I_func == [1]
    assert prob.I_op == []


def test_inclusionproblem_rejects_mixed_component_types():
    with pytest.raises(ValueError):
        InclusionProblem([[MaximallyMonotone(), Convex()]])


def test_gradient_dominated_requires_single_component():
    with pytest.raises(ValueError):
        InclusionProblem([GradientDominated(1.0), Convex()])
