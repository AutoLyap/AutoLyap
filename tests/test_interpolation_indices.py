import pytest

from autolyap.problemclass.problemclass import InterpolationIndices


# Tests for interpolation index parsing and equality semantics.
def test_interpolation_indices_rejects_invalid_value():
    with pytest.raises(ValueError):
        InterpolationIndices("not-a-valid-index")


def test_interpolation_indices_equality_and_str():
    idx = InterpolationIndices("j1<j2")
    assert str(idx) == "j1<j2"
    assert idx == "j1<j2"
    assert idx == InterpolationIndices("j1<j2")
