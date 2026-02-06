import numpy as np
from typing import Tuple
from .algorithm import Algorithm
from autolyap.utils.validation import ensure_real_number

class HeavyBallMethod(Algorithm):
    r"""
    Heavy-ball method.

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


    For initial points :math:`x^{-1},x^0 \in \calH`, step size :math:`\gamma \in \reals_{++}`,
    and momentum parameter :math:`\delta \in \reals`,

    .. math::
        (\forall k \in \naturals)\quad
        x^{k+1} = x^k - \gamma \nabla f(x^k) + \delta (x^k - x^{k-1}).

    State-space representation
    --------------------------

    The update can be written in the algorithm representation with
    :math:`\bx^k = (x^k, x^{k-1})`, :math:`\bu^k = \nabla f(x^k)`, and
    :math:`\by^k = x^k`.

    In :meth:`get_ABCD`, the matrices are

    .. math::
        \begin{aligned}
            A_k &=
            \begin{bmatrix}
            1+\delta & -\delta \\
            1 & 0
            \end{bmatrix}, &
            B_k &=
            \begin{bmatrix}
            -\gamma \\
            0
            \end{bmatrix}, \\
            C_k &=
            \begin{bmatrix}
            1 & 0
            \end{bmatrix}, &
            D_k &= \begin{bmatrix} 0 \end{bmatrix}.
        \end{aligned}
    """
    def __init__(self, gamma, delta):
        r"""
        Initialize the heavy-ball method.
        """
        super().__init__(2, 1, [1], [1], [])
        self.gamma = gamma
        self.delta = delta
    
    def set_gamma(self, gamma: float) -> None:
        r"""
        Set the step-size parameter :math:`\gamma`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.HeavyBallMethod`.

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
        Set the momentum parameter :math:`\delta`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.HeavyBallMethod`.

        **Parameters**

        - `delta` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`\delta`.

        **Raises**

        - `ValueError`: If `delta` is not a finite real number.
        """
        delta = ensure_real_number(delta, "delta", finite=True)
        self.delta = delta
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A = np.array([[1+self.delta, -self.delta], [1, 0]]) 
        B = np.array([[-self.gamma],[0]])
        C = np.array([[1, 0]])
        D = np.array([[0]])
        return (A, B, C, D)
