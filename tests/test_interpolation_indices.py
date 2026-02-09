import pytest

from autolyap.problemclass.problemclass import InterpolationIndices


# Tests for interpolation index parsing and equality semantics.
def test_interpolation_indices_rejects_invalid_value():
    with pytest.raises(ValueError):
        InterpolationIndices("not-a-valid-index")


def test_interpolation_indices_equality_and_str():
    idx = InterpolationIndices("r1<r2")
    assert str(idx) == "r1<r2"
    assert idx == "r1<r2"
    assert idx == InterpolationIndices("r1<r2")
