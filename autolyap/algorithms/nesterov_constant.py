import numpy as np
from typing import Tuple
from .algorithm import Algorithm

class NesterovConstant(Algorithm):
    r"""
    Nesterov's constant-step scheme.

    See :doc:`3. Algorithm representation </theory/algorithm_representation>`
    for mathematical notation and definitions.

    Notation-driven assumptions are declared by the user via
    :class:`~autolyap.problemclass.InclusionProblem`: when present, terms written with
    :math:`\nabla` use differentiable functions, terms written with
    :math:`\prox_{\gamma f}` use proper, lower semicontinuous, convex functions,
    and terms written with :math:`J_{\gamma G}` use maximally monotone operators.

    Standard form
    -------------


    Let :math:`q = \mu/L` and
    :math:`\eta = (1-\sqrt{q})/(1+\sqrt{q})`. For initial points
    :math:`x^{-1},x^0 \in \calH`,

    .. math::
        (\forall k \in \naturals)\quad
        \left[
        \begin{aligned}
            y^k &= x^k + \eta (x^k - x^{k-1}), \\
            x^{k+1} &= y^k - \frac{1}{L}\nabla f(y^k).
        \end{aligned}
        \right.

    State-space representation
    --------------------------

    The update can be written in the algorithm representation with

    .. math::
        \bx^k = (x^k, x^{k-1}), \qquad
        \bu^k = \nabla f(y^k), \qquad
        \by^k = y^k.

    Let :math:`q = \mu/L`. With this representation, the system matrices are

    .. math::
        \begin{aligned}
            A_k &=
            \begin{bmatrix}
            \frac{2}{1+\sqrt{q}} & -\frac{1-\sqrt{q}}{1+\sqrt{q}} \\
            1 & 0
            \end{bmatrix}, &
            B_k &=
            \begin{bmatrix}
            -\frac{1}{L} \\
            0
            \end{bmatrix}, \\
            C_k &=
            \begin{bmatrix}
            \frac{2}{1+\sqrt{q}} & -\frac{1-\sqrt{q}}{1+\sqrt{q}}
            \end{bmatrix}, &
            D_k &= \begin{bmatrix} 0 \end{bmatrix}.
        \end{aligned}

    These are the system matrices returned by :meth:`~autolyap.algorithms.Algorithm.get_ABCD`.

    Structural parameters
    ---------------------

    .. math::
        n = 2,\quad m = 1,\quad (\bar{m}_i)_{i=1}^{m} = (1),\quad \bar{m} = 1.

    .. math::
        I_{\text{func}} = \{1\},\quad I_{\text{op}} = \varnothing.
    """
    def __init__(self, mu: float, L: float) -> None:
        r"""
        Initialize the constant-step scheme.

        Structural inputs passed to :class:`~autolyap.algorithms.Algorithm` are

        .. math::
            n = 2,\quad m = 1,\quad (\bar m_i)_{i=1}^{m} = (1),\quad \bar m = 1,\quad
            I_{\mathrm{func}} = \{1\},\quad I_{\mathrm{op}} = \varnothing.
        """
        super().__init__(2, 1, [1], [1], [])
        self.set_L(L)
        self.set_mu(mu)

    def set_mu(self, mu: float) -> None:
        r"""
        Set the strong-convexity parameter :math:`\mu`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.NesterovConstant`.

        **Parameters**

        - `mu` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`\mu`.

        **Raises**

        - `ValueError`: If `mu` is not a finite real number or if :math:`\mu \le 0`.
        """
        mu = self._validate_positive_finite_real(mu, "mu")
        self._set_dynamic_parameter("mu", mu)

    def set_L(self, L: float) -> None:
        r"""
        Set the smoothness parameter :math:`L`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.NesterovConstant`.

        **Parameters**

        - `L` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`L`.

        **Raises**

        - `ValueError`: If `L` is not a finite real number or if :math:`L \le 0`.
        """
        L = self._validate_positive_finite_real(L, "L")
        self._set_dynamic_parameter("L", L)
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = self.mu / self.L
        
        A = np.array([[2/(1+np.sqrt(q)),-(1-np.sqrt(q))/(1+np.sqrt(q))],[1,0]])
        B = np.array([[-1 / self.L], [0]])
        C = np.array([[2/(1+np.sqrt(q)),-(1-np.sqrt(q))/(1+np.sqrt(q))]])
        D = np.array([[0]])
        return (A, B, C, D)
