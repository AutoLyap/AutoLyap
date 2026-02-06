import numpy as np
from typing import Tuple
from .algorithm import Algorithm
from autolyap.utils.validation import ensure_real_number

class DouglasRachford(Algorithm):
    r"""
    Douglas--Rachford splitting.

    Class-level reference
    =====================

    Mathematical notation and shared definitions used by methods are defined in this class docstring.

    Notation-driven assumptions are declared by the user via
    :class:`~autolyap.problemclass.InclusionProblem`: when present, terms written with
    :math:`\nabla` use differentiable functions, terms written with
    :math:`\prox_{\gamma f}` use proper, lower semicontinuous, convex functions,
    and terms written with :math:`J_{\gamma G}` use maximally monotone operators.

    Standard form
    -------------


    For an initial point :math:`x^0 \in \calH`, step size :math:`\gamma \in \reals_{++}`,
    and relaxation parameter :math:`\lambda \in \reals`,

    .. math::
        (\forall k \in \naturals)\quad
        \left[
        \begin{aligned}
            v^k &=
            \begin{cases}
                J_{\gamma G_1}(x^k), & \text{if type = operator}, \\
                \prox_{\gamma f_1}(x^k), & \text{if type = function},
            \end{cases} \\
            w^k &=
            \begin{cases}
                J_{\gamma G_2}(2v^k - x^k), & \text{if type = operator}, \\
                \prox_{\gamma f_2}(2v^k - x^k), & \text{if type = function},
            \end{cases} \\
            x^{k+1} &= x^k + \lambda (w^k - v^k).
        \end{aligned}
        \right.

    State-space representation
    --------------------------

    The update can be written in the algorithm representation with
    :math:`\bx^k = x^k`,
    :math:`\bu^k = \left((x^k-v^k)/\gamma,\; (2v^k-x^k-w^k)/\gamma\right)`, and
    :math:`\by^k = \left(x^k,\; 2v^k-x^k\right)`.

    In :meth:`get_ABCD`, the matrices are

    .. math::
        \begin{aligned}
            A_k &= \begin{bmatrix} 1 \end{bmatrix}, &
            B_k &= \begin{bmatrix} -\gamma\lambda & -\gamma\lambda \end{bmatrix}, \\
            C_k &=
            \begin{bmatrix}
            1 \\
            1
            \end{bmatrix}, &
            D_k &=
            \begin{bmatrix}
            -\gamma & 0 \\
            -2\gamma & -\gamma
            \end{bmatrix}.
        \end{aligned}
    """
    def __init__(self, gamma, lambda_value, type: str = "operator"):
        r"""
        Initialize the Douglas--Rachford method.
        """
        if type == "operator":
            super().__init__(1, 2, [1, 1], [], [1, 2])
        elif type == "function":
            super().__init__(1, 2, [1, 1], [1, 2], [])
        else:
            raise ValueError("type must be either 'operator' or 'function'")
        self.gamma = gamma
        self.lambda_value = lambda_value
    
    def set_gamma(self, gamma: float) -> None:
        r"""
        Set the step-size parameter :math:`\gamma`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.DouglasRachford`.

        **Parameters**

        - `gamma` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`\gamma`.

        **Raises**

        - `ValueError`: If `gamma` is not a finite real number or if :math:`\gamma \le 0`.
        """
        gamma = ensure_real_number(gamma, "gamma", finite=True)
        if gamma <= 0:
            raise ValueError("gamma must be > 0.")
        self.gamma = gamma
    
    def set_lambda(self, lambda_value: float) -> None:
        r"""
        Set the relaxation parameter :math:`\lambda`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.DouglasRachford`.

        **Parameters**

        - `lambda_value` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`\lambda`.

        **Raises**

        - `ValueError`: If `lambda_value` is not a finite real number.
        """
        lambda_value = ensure_real_number(lambda_value, "lambda_value", finite=True)
        self.lambda_value = lambda_value
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A = np.array([[1]])
        B = np.array([[-self.gamma*self.lambda_value, -self.gamma*self.lambda_value]])
        C = np.array([[1], 
                      [1]])
        D = np.array([[-self.gamma, 0], 
                      [-2*self.gamma, -self.gamma]])
        return (A, B, C, D)
