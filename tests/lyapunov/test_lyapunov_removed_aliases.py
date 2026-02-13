import pytest

from autolyap import IterationDependent, IterationIndependent


def test_iteration_independent_removed_alias_has_migration_message():
    with pytest.raises(AttributeError) as exc_info:
        getattr(IterationIndependent, "verify_iteration_independent_Lyapunov")

    message = str(exc_info.value)
    assert "removed in v0.2.0" in message
    assert "IterationIndependent.search_lyapunov" in message
    assert "release_notes/v0_2_0.html" in message
    assert "quick_start.html" in message


def test_iteration_dependent_removed_alias_has_migration_message():
    with pytest.raises(AttributeError) as exc_info:
        getattr(IterationDependent, "verify_iteration_dependent_Lyapunov")

    message = str(exc_info.value)
    assert "removed in v0.2.0" in message
    assert "IterationDependent.search_lyapunov" in message
    assert "release_notes/v0_2_0.html" in message
    assert "quick_start.html" in message
