"""Validation helpers shared across algorithms and problem classes."""

from numbers import Integral, Real
from typing import Any, Iterable, List, Optional, Sequence, cast

import numpy as np

_INDEX_LIST_TYPES = (list, tuple, np.ndarray)


def _ensure_index_container(values: Any, name: str, error_message: str) -> Sequence[Any]:
    r"""
    Ensure index-like input is a concrete finite-length sequence container.

    Accepted container types are ``list``, ``tuple``, and ``numpy.ndarray``.

    **Parameters**

    - `values`: Candidate container to validate.
    - `name`: Logical input name (reserved for caller-side error context).
    - `error_message`: Error text raised when validation fails.

    **Returns**

    - A sequence view of `values` suitable for indexed iteration.
    """
    if values is None or not isinstance(values, _INDEX_LIST_TYPES):
        raise ValueError(error_message)
    return cast(Sequence[Any], values)


def ensure_integral(value: Any, name: str, minimum: Optional[int] = None,
                    maximum: Optional[int] = None) -> int:
    r"""
    Validate and normalize an integer-valued scalar.

    Rejects booleans explicitly and optionally enforces lower/upper bounds.
    """
    # Reject bools explicitly; they are subclasses of int in Python.
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    value = int(value)
    # Bounds checks are optional; use when caller cares about valid ranges.
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}.")
    return value


def ensure_real_number(value: Any, name: str, finite: bool = False,
                       minimum: Optional[float] = None,
                       maximum: Optional[float] = None) -> float:
    r"""
    Validate and normalize a real-valued scalar.

    Rejects booleans and NaNs explicitly. Optional flags enforce finiteness
    and lower/upper bounds.
    """
    # Reject bools explicitly; they are subclasses of int/float in Python.
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number.")
    value = float(value)
    if np.isnan(value):
        raise ValueError(f"{name} must be a real number.")
    if finite and not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}.")
    return value


def ensure_finite_array(array: np.ndarray, name: str) -> None:
    r"""Raise if `array` contains non-finite entries (NaN or inf)."""
    # Centralized finite check used across matrix/vector inputs.
    try:
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite entries.")
    except TypeError as exc:
        raise ValueError(f"{name} must contain only finite entries.") from exc


def ensure_index_list(values: Iterable[Any], name: str, m: int) -> List[int]:
    r"""
    Validate a sorted, duplicate-free component-index list.

    The returned list is guaranteed to contain integers in ``[1, m]`` in
    strictly increasing order.
    """
    values = _ensure_index_container(values, name, f"{name} must be a list of integers.")
    items: List[int] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, Integral):
            raise ValueError(f"{name} must contain only integers.")
        v_int = int(v)
        if v_int < 1 or v_int > m:
            raise ValueError(f"{name} entries must be in [1, {m}].")
        items.append(v_int)
    if len(set(items)) != len(items):
        raise ValueError(f"{name} must not contain duplicates.")
    if items != sorted(items):
        raise ValueError(f"{name} must be in increasing order.")
    return items


def ensure_m_bar_list(values: Iterable[Any], m: int) -> List[int]:
    r"""
    Validate per-component evaluation counts.

    Ensures the sequence length equals ``m`` and every entry is an integer
    strictly greater than zero.
    """
    values = _ensure_index_container(values, "m_bar_is", "m_bar_is must be a list of positive integers.")
    if len(values) != m:
        raise ValueError("m must equal the length of m_bar_is")
    out: List[int] = []
    for idx, v in enumerate(values, start=1):
        if isinstance(v, bool) or not isinstance(v, Integral):
            raise ValueError("m_bar_is must contain only integers.")
        v_int = int(v)
        if v_int <= 0:
            raise ValueError(f"m_bar_is entries must be > 0. Got {v_int} at index {idx}.")
        out.append(v_int)
    return out
