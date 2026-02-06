import numpy as np
from typing import Tuple
from .algorithm import Algorithm
from autolyap.utils.validation import ensure_real_number

class AcceleratedProximalPoint(Algorithm):
    r"""
    Accelerated proximal point method.

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


    Let :math:`\lambda_k = k/(k+2)`. For initial points
    :math:`x^0,y^0,y^{-1} \in \calH` and :math:`\gamma \in \reals_{++}`,

    .. math::
        (\forall k \in \naturals)\quad
        \left[
        \begin{aligned}
            x^{k+1} &=
            \begin{cases}
                J_{\gamma G}(y^k), & \text{if type = operator}, \\
                \prox_{\gamma f}(y^k), & \text{if type = function},
            \end{cases} \\
            y^{k+1} &= x^{k+1} + \lambda_k(x^{k+1} - x^k) - \lambda_k(x^k - y^{k-1}).
        \end{aligned}
        \right.

    State-space representation
    --------------------------

    The update can be written in the algorithm representation with
    :math:`\bx^k = (x^k, y^k, y^{k-1})`, :math:`\by^k = y^k`, and
    :math:`\bu^k = (y^k - x^{k+1})/\gamma`.

    With :math:`\lambda_k = k/(k+2)`, :meth:`get_ABCD` returns

    .. math::
        \begin{aligned}
            A_k &=
            \begin{bmatrix}
            0 & 1 & 0 \\
            -2\lambda_k & 1+\lambda_k & \lambda_k \\
            0 & 1 & 0
            \end{bmatrix}, &
            B_k &=
            \begin{bmatrix}
            -\gamma \\
            -\gamma(1+\lambda_k) \\
            0
            \end{bmatrix}, \\
            C_k &= \begin{bmatrix} 0 & 1 & 0 \end{bmatrix}, &
            D_k &= \begin{bmatrix} -\gamma \end{bmatrix}.
        \end{aligned}
    """
    def __init__(self, gamma, type: str = "operator"):
        r"""
        Initialize the accelerated proximal point method.
        """
        if type == "operator":
            super().__init__(3, 1, [1], [], [1])
        elif type == "function":
            super().__init__(3, 1, [1], [1], [])
        else:
            raise ValueError("type must be either 'operator' or 'function'")
        self.gamma = gamma
    
    def set_gamma(self, gamma: float) -> None:
        r"""
        Set the step-size parameter :math:`\gamma`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.AcceleratedProximalPoint`.

        **Parameters**

        - `gamma` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`\gamma`.

        **Raises**

        - `ValueError`: If `gamma` is not a finite real number or if :math:`\gamma \le 0`.
        """
        gamma = ensure_real_number(gamma, "gamma", finite=True)
        if gamma <= 0:
            raise ValueError("gamma must be > 0.")
        self.gamma = gamma
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lambda_var = k / (k + 2)
        A = np.array([[0, 1 , 0], 
                      [-2*lambda_var, 1+lambda_var, lambda_var],
                      [0, 1, 0]])
        B = np.array([[-self.gamma],
                      [-self.gamma*(1+lambda_var)],
                      [0]])
        C = np.array([[0, 1, 0]])
        D = np.array([[-self.gamma]])
        return (A, B, C, D)
