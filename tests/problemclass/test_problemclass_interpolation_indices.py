import pytest

from autolyap.problemclass.indices import _InterpolationIndices


# Tests for interpolation index parsing and equality semantics.
def test_interpolation_indices_rejects_invalid_value():
    with pytest.raises(ValueError):
        _InterpolationIndices("not-a-valid-index")


def test_interpolation_indices_equality_and_str():
    idx = _InterpolationIndices("r1<r2")
    assert str(idx) == "r1<r2"
    assert idx == "r1<r2"
    assert idx == _InterpolationIndices("r1<r2")
