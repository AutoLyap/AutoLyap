import pytest

from autolyap.problemclass import (
    Convex,
    MaximallyMonotone,
    InclusionProblem,
)


# Tests for cache invalidation and update behavior in InclusionProblem.
def test_component_data_is_readonly():
    prob = InclusionProblem([Convex()])
    data = prob._get_component_data(1)
    matrix, vector, _eq, _idx = data[0]
    with pytest.raises(ValueError):
        matrix[0, 0] = 1.0
    with pytest.raises(ValueError):
        vector[0] = 1.0


def test_update_component_instances_refreshes_indices_and_cache():
    prob = InclusionProblem([Convex()])
    data_before = prob._get_component_data(1)
    assert len(data_before) == 1
    assert len(data_before[0]) == 4

    prob._update_component_instances(1, MaximallyMonotone())
    assert prob.I_op == [1]
    assert prob.I_func == []

    data_after = prob._get_component_data(1)
    assert len(data_after) == 1
    assert len(data_after[0]) == 2


def test_update_component_instances_rejects_mixed_types():
    prob = InclusionProblem([Convex()])
    with pytest.raises(ValueError):
        prob._update_component_instances(1, [Convex(), MaximallyMonotone()])
