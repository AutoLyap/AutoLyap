import numpy as np
from typing import Tuple
from .algorithm import Algorithm

class ProximalPoint(Algorithm):
    r"""
    Proximal point method.

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


    For an initial point :math:`x^0 \in \calH` and step size
    :math:`\gamma \in \reals_{++}`,

    .. math::
        (\forall k \in \naturals)\quad
        x^{k+1} = \prox_{\gamma f}(x^k).

    Equivalently,

    .. math::
        (\forall k \in \naturals)\quad
        x^{k+1} = x^k - \gamma u^k, \qquad u^k \in \partial f(x^{k+1}).

    State-space representation
    --------------------------

    The update can be written in the algorithm representation with
    :math:`\bx^k = x^k`, :math:`\bu^k = u^k`, and :math:`\by^k = x^{k+1}`
    (using :math:`u^k \in \partial f(x^{k+1})` from the equivalent standard form).

    In :meth:`get_ABCD`, the matrices are

    .. math::
        \begin{aligned}
            A_k &= \begin{bmatrix} 1 \end{bmatrix}, & B_k &= \begin{bmatrix} -\gamma \end{bmatrix}, \\
            C_k &= \begin{bmatrix} 1 \end{bmatrix}, & D_k &= \begin{bmatrix} -\gamma \end{bmatrix}.
        \end{aligned}
    """
    def __init__(self, gamma):
        r"""
        Initialize the proximal point method.
        """
        super().__init__(1, 1, [1], [1], [])
        self.gamma = gamma
    
    def set_gamma(self, gamma: float) -> None:
        r"""
        Set the step-size parameter :math:`\gamma`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.ProximalPoint`.

        **Parameters**

        - `gamma` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`\gamma`.

        **Raises**

        - `ValueError`: If `gamma` is not a finite real number or if :math:`\gamma \le 0`.
        """
        gamma = self._validate_positive_finite_real(gamma, "gamma")
        self._set_dynamic_parameter("gamma", gamma)
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A = np.array([[1]])
        B = np.array([[-self.gamma]])
        C = np.array([[1]])
        D = np.array([[-self.gamma]])
        return (A, B, C, D)
