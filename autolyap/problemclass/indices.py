"""Interpolation index helpers for problem classes."""

class InterpolationIndices:
    r"""
    Wrapper for interpolation indices that enforces allowed values.

    Class-level reference
    =====================

    This class-level docstring centralizes the allowed symbolic labels used to
    index interpolation pairs in problem classes.

    **Allowed values**

    - ``p1<p2``: unordered pairs of distinct interpolation points :math:`p_1` and :math:`p_2`.
    - ``p1!=p2``: ordered pairs of distinct interpolation points :math:`p_1` and :math:`p_2`.
    - ``p1``: single-point interpolation constraints (index :math:`p_1`).
    - ``p1!=star``: pairs between any interpolation point :math:`p_1` and the distinguished star point.

    **Parameters**

    - `value` (:class:`str`): The interpolation index string.

    **Raises**

    - `ValueError`: If `value` is not in the allowed set.
    """
    # Frozen set prevents accidental runtime mutation of allowed values.
    ALLOWED_VALUES = frozenset({"p1<p2", "p1!=p2", "p1", "p1!=star"})
    __slots__ = ("_value",)

    def __init__(self, value: str):
        self.value = value

    @property
    def value(self) -> str:
        return self._value

    @value.setter
    def value(self, value: str) -> None:
        if not isinstance(value, str):
            raise ValueError("Interpolation index must be a string.")
        if value not in self.ALLOWED_VALUES:
            raise ValueError(f"Invalid interpolation index: {value}. Allowed values: {self.ALLOWED_VALUES}")
        self._value = value

    def __str__(self) -> str:
        return self._value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, InterpolationIndices):
            return self._value == other._value
        if isinstance(other, str):
            return self._value == other
        return False

    def __repr__(self) -> str:
        return f"InterpolationIndices({self._value})"
