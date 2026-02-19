from typing import Any

import numpy as np
from autolyap.utils.backend_types import (
    MosekExprProtocol,
    MosekUpperTriangleVectorProtocol,
    UpperTriangleValuesProtocol,
)


def _import_mosek_expr() -> MosekExprProtocol:
    r"""Import MOSEK Fusion `Expr` lazily for MOSEK-backed model builders."""
    try:
        import mosek.fusion as mf
        import mosek.fusion.pythonic  # noqa: F401  # enables Fusion operator overloads
    except ImportError as exc:
        raise ImportError(
            "MOSEK Fusion backend requested, but `mosek` is not installed. "
            "Install it with `pip install autolyap[mosek]`."
        ) from exc
    return mf.Expr


def _upper_triangle_size(n: int) -> int:
    r"""Return the number of entries in the upper triangle of an `n x n` matrix."""
    return n * (n + 1) // 2


def create_symmetric_matrix_expression(
    Xij: MosekUpperTriangleVectorProtocol,
    n: int,
) -> Any:
    r"""
    Convert a list of upper triangle variables to a symmetric matrix expression.

    **Parameters**

    - `Xij`: MOSEK variable containing the upper triangle and diagonal values.
    - `n`: Size of the symmetric matrix.

    **Returns**

    - Symmetric matrix expression of size :math:`n \times n`.
    """
    Expr = _import_mosek_expr()
    # Keep upper-triangle ordering consistent with create_symmetric_matrix.
    X_expr = [[None] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i, n):
            X_expr[i][j] = Xij.index(idx)
            if i != j:
                # Mirror the upper-triangle entry to enforce symmetry.
                X_expr[j][i] = Xij.index(idx)
            idx += 1
    X_rows = []
    for i in range(n):
        X_rows.append(Expr.hstack(X_expr[i]))
    X = Expr.vstack(X_rows)
    return X


def create_symmetric_matrix(upper_triangle_values: UpperTriangleValuesProtocol, n: int) -> np.ndarray:
    r"""
    Convert a list of upper triangle values to a symmetric matrix.

    **Parameters**

    - `upper_triangle_values`: List of length :math:`n(n+1)/2` containing the upper triangle
      and diagonal values.
    - `n`: Size of the symmetric matrix.

    **Returns**

    - Symmetric matrix of size :math:`n \times n`.

    **Raises**

    - `ValueError`: If the length of `upper_triangle_values` is not :math:`n(n+1)/2`.
    """
    # Guard against mismatched upper-triangle length to avoid silent shape errors.
    if len(upper_triangle_values) != _upper_triangle_size(n):
        raise ValueError("The length of upper_triangle_values must be n(n+1)/2")

    symmetric_matrix = np.zeros((n, n))

    idx = 0
    for i in range(n):
        for j in range(i, n):
            symmetric_matrix[i, j] = upper_triangle_values[idx]
            if i != j:
                # Mirror the upper-triangle entry to enforce symmetry.
                symmetric_matrix[j, i] = upper_triangle_values[idx]
            idx += 1

    return symmetric_matrix
