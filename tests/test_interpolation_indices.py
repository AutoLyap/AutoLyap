import pytest

from autolyap.problemclass.problemclass import InterpolationIndices


# Tests for interpolation index parsing and equality semantics.
def test_interpolation_indices_rejects_invalid_value():
    with pytest.raises(ValueError):
        InterpolationIndices("not-a-valid-index")


def test_interpolation_indices_equality_and_str():
    idx = InterpolationIndices("p1<p2")
    assert str(idx) == "p1<p2"
    assert idx == "p1<p2"
    assert idx == InterpolationIndices("p1<p2")
