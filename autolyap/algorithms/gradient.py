import numpy as np
from typing import Tuple
from .algorithm import Algorithm

class GradientMethod(Algorithm):
    r"""
    Gradient method.

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


    For an initial point :math:`x^0 \in \calH` and step size :math:`\gamma \in \reals_{++}`,

    .. math::
        (\forall k \in \naturals)\quad x^{k+1} = x^k - \gamma \nabla f(x^k).

    State-space representation
    --------------------------

    The update can be written in the algorithm representation with
    :math:`\bx^k = x^k`, :math:`\bu^k = \nabla f(x^k)`, and :math:`\by^k = x^k`.

    In :meth:`get_ABCD`, the matrices are

    .. math::
        \begin{aligned}
            A_k &= \begin{bmatrix} 1 \end{bmatrix}, & B_k &= \begin{bmatrix} -\gamma \end{bmatrix}, \\
            C_k &= \begin{bmatrix} 1 \end{bmatrix}, & D_k &= \begin{bmatrix} 0 \end{bmatrix}.
        \end{aligned}
    """
    def __init__(self, gamma):
        r"""
        Initialize the gradient method.
        """
        super().__init__(n=1, m=1, m_bar_is=[1], I_func=[1], I_op=[])
        self.gamma = gamma
    
    def set_gamma(self, gamma: float) -> None:
        r"""
        Set the step-size parameter :math:`\gamma`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.GradientMethod`.

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
        D = np.array([[0]])
        return (A, B, C, D)
