import pytest


# Tests for cache behavior and read-only protections.
def test_cached_matrices_are_readonly(constant_algorithm):
    sys_mats = constant_algorithm._get_AsBsCsDs(0, 0)
    A = sys_mats[0][0]
    with pytest.raises(ValueError):
        A[0, 0] = 2.0

    Ps = constant_algorithm._get_Ps()
    with pytest.raises(ValueError):
        Ps[(1, 1)][0, 0] = 1.0


def test_cache_invalidation_non_structural_keeps_static(constant_algorithm):
    constant_algorithm._get_Us(0, 0)
    constant_algorithm._get_AsBsCsDs(0, 0)
    assert len(constant_algorithm._cache_Us) == 1
    assert len(constant_algorithm._cache_AsBsCsDs) == 1

    constant_algorithm.step_size = 0.5
    assert len(constant_algorithm._cache_AsBsCsDs) == 0
    assert len(constant_algorithm._cache_Us) == 1


def test_cache_invalidation_structural_clears_all(constant_algorithm):
    constant_algorithm._get_Us(0, 0)
    constant_algorithm._get_AsBsCsDs(0, 0)
    constant_algorithm._get_Fs(0, 0)
    assert len(constant_algorithm._cache_Us) == 1
    assert len(constant_algorithm._cache_AsBsCsDs) == 1
    assert len(constant_algorithm._cache_Fs) == 1

    constant_algorithm.n = constant_algorithm.n
    assert len(constant_algorithm._cache_Us) == 0
    assert len(constant_algorithm._cache_AsBsCsDs) == 0
    assert len(constant_algorithm._cache_Fs) == 0


def test_cache_eviction_respects_lru_order(constant_algorithm):
    for k in range(9):
        constant_algorithm._get_AsBsCsDs(k, k)
    cache = constant_algorithm._cache_AsBsCsDs
    assert len(cache) == constant_algorithm._cache_maxsize
    assert (0, 0) not in cache

    constant_algorithm._get_AsBsCsDs(1, 1)
    constant_algorithm._get_AsBsCsDs(9, 9)
    cache = constant_algorithm._cache_AsBsCsDs
    assert len(cache) == constant_algorithm._cache_maxsize
    assert (2, 2) not in cache
