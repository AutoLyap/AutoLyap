import numpy as np
from mosek.fusion import Expr

def create_symmetric_matrix_expression(Xij, n):
    r"""
    Convert a list of upper triangle variables to a symmetric matrix expression.

    :param Xij: MOSEK variable containing the upper triangle and diagonal values.
    :type Xij: Variable
    :param n: Size of the symmetric matrix.
    :type n: int
    :return: Symmetric matrix expression of size n x n.
    :rtype: Expr
    """
    X_expr = [[None] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i, n):
            X_expr[i][j] = Xij.index(idx)
            if i != j:
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

    :param upper_triangle_values: List of length n(n+1)/2 containing the upper triangle and diagonal values.
    :type upper_triangle_values: list
    :param n: Size of the symmetric matrix.
    :type n: int
    :return: Symmetric matrix of size n x n.
    :rtype: np.ndarray
    :raises ValueError: If the length of upper_triangle_values is not n(n+1)/2.
    """
    if len(upper_triangle_values) != n * (n + 1) // 2:
        raise ValueError("The length of upper_triangle_values must be n(n+1)/2")

    symmetric_matrix = np.zeros((n, n))

    idx = 0
    for i in range(n):
        for j in range(i, n):
            symmetric_matrix[i, j] = upper_triangle_values[idx]
            if i != j:
                symmetric_matrix[j, i] = upper_triangle_values[idx]
            idx += 1

    return symmetric_matrix
