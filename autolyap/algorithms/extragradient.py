import numpy as np
from typing import Tuple
from .algorithm import Algorithm

class Extragradient(Algorithm):
    r"""
    Extragradient method.

    See :doc:`3. Algorithm representation </theory/algorithm_representation>`
    for mathematical notation and definitions.

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

    .. math::
        \bx^k = x^k.

    If `type = unconstrained`, use

    .. math::
        \bu^k = (G_1(x^k), G_1(\bar{x}^k)), \qquad
        \by^k = (x^k, \bar{x}^k),

    with the system matrices

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

    These are the system matrices returned by :meth:`~autolyap.algorithms.Algorithm.get_ABCD` when
    `type = unconstrained`.

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

    with the system matrices

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

    These are the system matrices returned by :meth:`~autolyap.algorithms.Algorithm.get_ABCD` when
    `type = constrained`.

    Structural parameters
    ---------------------

    .. math::
        \text{type}=\text{"unconstrained"}:\quad
        n = 1,\quad m = 1,\quad (\bar{m}_i)_{i=1}^{m} = (2),\quad \bar{m} = 2,\quad
        I_{\text{func}} = \varnothing,\quad I_{\text{op}} = \{1\}.

    .. math::
        \text{type}=\text{"constrained"}:\quad
        n = 1,\quad m = 2,\quad (\bar{m}_i)_{i=1}^{m} = (2,2),\quad \bar{m} = 4,\quad
        I_{\text{func}} = \{2\},\quad I_{\text{op}} = \{1\}.
    """
    def __init__(self, gamma: float, delta: float, type: str = "unconstrained") -> None:
        r"""
        Initialize the extragradient method.

        Structural inputs passed to :class:`~autolyap.algorithms.Algorithm` are
        case-dependent:

        - If `type = unconstrained`:

          .. math::
              n = 1,\quad m = 1,\quad (\bar m_i)_{i=1}^{m} = (2),\quad \bar m = 2,\quad
              I_{\mathrm{func}} = \varnothing,\quad I_{\mathrm{op}} = \{1\}.

        - If `type = constrained`:

          .. math::
              n = 1,\quad m = 2,\quad (\bar m_i)_{i=1}^{m} = (2,2),\quad \bar m = 4,\quad
              I_{\mathrm{func}} = \{2\},\quad I_{\mathrm{op}} = \{1\}.
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
        gamma = self._validate_positive_finite_real(gamma, "gamma")
        self._set_dynamic_parameter("gamma", gamma)
    
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
        delta = self._validate_positive_finite_real(delta, "delta")
        self._set_dynamic_parameter("delta", delta)
    
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
