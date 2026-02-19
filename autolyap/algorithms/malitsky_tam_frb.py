import numpy as np
from typing import Tuple
from .algorithm import Algorithm


class MalitskyTamFRB(Algorithm):
    r"""
    Malitsky--Tam forward-reflected-backward method.

    See :doc:`3. Algorithm representation </theory/algorithm_representation>`
    for mathematical notation and definitions.

    Notation-driven assumptions are declared by the user via
    :class:`~autolyap.problemclass.InclusionProblem`: when present, terms written with
    :math:`\nabla` use differentiable functions, terms written with
    :math:`\prox_{\gamma f}` use proper, lower semicontinuous, convex functions,
    and terms written with :math:`J_{\gamma G}` use maximally monotone operators.

    Standard form
    -------------

    For an initial pair :math:`(x^{-1}, x^0) \in \calH^2` and step size
    :math:`\gamma \in \reals_{++}`,

    .. math::
        (\forall k \in \naturals)\quad
        x^{k+1} = J_{\gamma G_2}\left(x^k - 2\gamma G_1(x^k) + \gamma G_1(x^{k-1})\right).

    Equivalently,

    .. math::
        (\forall k \in \naturals)\quad
        x^{k+1} = x^k - 2\gamma G_1(x^k) + \gamma G_1(x^{k-1}) - \gamma G_2(x^{k+1}).

    State-space representation
    --------------------------

    The update can be written in the algorithm representation with

    .. math::
        \bx^k = (x^k,\; x^{k-1}), \qquad
        \bu^k = \left(G_1(x^k),\; G_1(x^{k-1}),\; G_2(x^{k+1})\right), \qquad
        \by^k = \left(x^k,\; x^{k-1},\; x^{k+1}\right).

    With this representation, the system matrices are

    .. math::
        \begin{aligned}
            A_k &=
            \begin{bmatrix}
            1 & 0 \\
            1 & 0
            \end{bmatrix}, &
            B_k &=
            \begin{bmatrix}
            -2\gamma & \gamma & -\gamma \\
            0 & 0 & 0
            \end{bmatrix}, \\
            C_k &=
            \begin{bmatrix}
            1 & 0 \\
            0 & 1 \\
            1 & 0
            \end{bmatrix}, &
            D_k &=
            \begin{bmatrix}
            0 & 0 & 0 \\
            0 & 0 & 0 \\
            -2\gamma & \gamma & -\gamma
            \end{bmatrix}.
        \end{aligned}

    These are the system matrices returned by :meth:`~autolyap.algorithms.Algorithm.get_ABCD`.

    Structural parameters
    ---------------------

    .. math::
        n = 2,\quad m = 2,\quad (\bar{m}_i)_{i=1}^{m} = (2,1),\quad \bar{m} = 3.

    .. math::
        I_{\text{func}} = \varnothing,\quad I_{\text{op}} = \{1,2\}.
    """
    def __init__(self, gamma: float) -> None:
        r"""
        Initialize the Malitsky--Tam forward-reflected-backward method.

        Structural inputs passed to :class:`~autolyap.algorithms.Algorithm` are

        .. math::
            n = 2,\quad m = 2,\quad (\bar m_i)_{i=1}^{m} = (2,1),\quad \bar m = 3,\quad
            I_{\mathrm{func}} = \varnothing,\quad I_{\mathrm{op}} = \{1,2\}.
        """
        super().__init__(2, 2, [2, 1], [], [1, 2])
        self.gamma = gamma

    def set_gamma(self, gamma: float) -> None:
        r"""
        Set the step-size parameter :math:`\gamma`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.MalitskyTamFRB`.

        **Parameters**

        - `gamma` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`\gamma`.

        **Raises**

        - `ValueError`: If `gamma` is not a finite real number or if :math:`\gamma \le 0`.
        """
        gamma = self._validate_positive_finite_real(gamma, "gamma")
        self._set_dynamic_parameter("gamma", gamma)

    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A = np.array([[1, 0],
                      [1, 0]])
        B = np.array([[-2 * self.gamma, self.gamma, -self.gamma],
                      [0, 0, 0]])
        C = np.array([[1, 0],
                      [0, 1],
                      [1, 0]])
        D = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [-2 * self.gamma, self.gamma, -self.gamma]])
        return (A, B, C, D)
