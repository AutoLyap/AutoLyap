import numpy as np
from typing import Tuple
from .algorithm import Algorithm
from autolyap.utils.validation import ensure_real_number

class Extragradient(Algorithm):
    r"""
    Extragradient method.

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


    For an initial point :math:`x^0 \in \calH`, step sizes
    :math:`\gamma,\delta \in \reals_{++}`, and :math:`k \in \naturals`,

    .. math::
        \left[
        \begin{aligned}
            &\bar{x}^k = x^k - \gamma G_1(x^k), \\
            &x^{k+1} = x^k - \delta G_1(\bar{x}^k),
        \end{aligned}
        \right.
        \qquad \text{if type = unconstrained},

    and

    .. math::
        \left[
        \begin{aligned}
            &\bar{x}^k = \prox_{\gamma f}(x^k - \gamma G_1(x^k)), \\
            &x^{k+1} = \prox_{\delta f}(x^k - \delta G_1(\bar{x}^k)),
        \end{aligned}
        \right.
        \qquad \text{if type = constrained}.

    State-space representation
    --------------------------

    The update can be written in the algorithm representation with
    :math:`\bx^k = x^k`.

    If `type = unconstrained`, use

    .. math::
        \bu^k = (G_1(x^k), G_1(\bar{x}^k)), \qquad
        \by^k = (x^k, \bar{x}^k),

    with

    .. math::
        \begin{aligned}
            A_k &= \begin{bmatrix} 1 \end{bmatrix}, &
            B_k &= \begin{bmatrix} 0 & -\delta \end{bmatrix}, \\
            C_k &=
            \begin{bmatrix}
            1 \\
            1
            \end{bmatrix}, &
            D_k &=
            \begin{bmatrix}
            0 & 0 \\
            -\gamma & 0
            \end{bmatrix}.
        \end{aligned}

    If `type = constrained`, use

    .. math::
        \bu^k = \left(
        G_1(x^k),\;
        G_1(\bar{x}^k),\;
        \frac{x^k - \gamma G_1(x^k) - \bar{x}^k}{\gamma},\;
        \frac{x^k - \delta G_1(\bar{x}^k) - x^{k+1}}{\delta}
        \right),

    .. math::
        \by^k = (x^k, \bar{x}^k, \bar{x}^k, x^{k+1}),

    with

    .. math::
        \begin{aligned}
            A_k &= \begin{bmatrix} 1 \end{bmatrix}, &
            B_k &= \begin{bmatrix} 0 & -\delta & 0 & -\delta \end{bmatrix}, \\
            C_k &=
            \begin{bmatrix}
            1 \\
            1 \\
            1 \\
            1
            \end{bmatrix}, &
            D_k &=
            \begin{bmatrix}
            0 & 0 & 0 & 0 \\
            -\gamma & 0 & -\gamma & 0 \\
            -\gamma & 0 & -\gamma & 0 \\
            0 & -\delta & 0 & -\delta
            \end{bmatrix}.
        \end{aligned}
    """
    def __init__(self, gamma: float, delta: float, type: str = "unconstrained"):
        r"""
        Initialize the extragradient method.
        """
        if type == "unconstrained":
            super().__init__(1, 1, [2], [], [1]) 
        elif type == "constrained":
            super().__init__(1, 2, [2, 2], [2], [1])
        else:
            raise ValueError("Not valid type/implemented yet")
        self.type = type
        self.gamma = gamma
        self.delta = delta
    
    def set_gamma(self, gamma: float) -> None:
        r"""
        Set the first step-size parameter :math:`\gamma`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.Extragradient`.

        **Parameters**

        - `gamma` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`\gamma`.

        **Raises**

        - `ValueError`: If `gamma` is not a finite real number or if :math:`\gamma \le 0`.
        """
        gamma = ensure_real_number(gamma, "gamma", finite=True)
        if gamma <= 0:
            raise ValueError("gamma must be > 0.")
        self.gamma = gamma
    
    def set_delta(self, delta: float) -> None:
        r"""
        Set the second step-size parameter :math:`\delta`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.Extragradient`.

        **Parameters**

        - `delta` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`\delta`.

        **Raises**

        - `ValueError`: If `delta` is not a finite real number or if :math:`\delta \le 0`.
        """
        delta = ensure_real_number(delta, "delta", finite=True)
        if delta <= 0:
            raise ValueError("delta must be > 0.")
        self.delta = delta
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.type == "unconstrained":
            A = np.array([[1]])
            B = np.array([[0, -self.delta]])
            C = np.array([[1], [1]])
            D = np.array([[0, 0], 
                          [-self.gamma, 0]])
        elif self.type == "constrained":
            A = np.array([[1]])
            B = np.array([[0, -self.delta, 0 , -self.delta]])
            C = np.array([[1],
                          [1],
                          [1],
                          [1]])
            D = np.array([[0, 0, 0, 0], 
                          [-self.gamma, 0, -self.gamma, 0],
                          [-self.gamma, 0, -self.gamma, 0],
                          [0, -self.delta, 0, -self.delta]])
        else:
            raise ValueError("Not valid type/implemented yet")
        return (A, B, C, D)
