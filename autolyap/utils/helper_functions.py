import numpy as np
from mosek.fusion import Expr

def create_symmetric_matrix_expression(Xij, n):
    r"""
    Convert a list of upper triangle variables to a symmetric matrix expression.

    **Parameters**

    - `Xij`: MOSEK variable containing the upper triangle and diagonal values.
    - `n`: Size of the symmetric matrix.

    **Returns**

    - Symmetric matrix expression of size :math:`n \times n`.
    """
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

def create_symmetric_matrix(upper_triangle_values, n):
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
    if len(upper_triangle_values) != n * (n + 1) // 2:
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
