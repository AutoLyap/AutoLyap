import numpy as np
from typing import Tuple
from .algorithm import Algorithm

class TsengFBF(Algorithm):
    r"""
    Tseng's forward-backward-forward method.

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


    For an initial point :math:`x^0 \in \calH`, step size
    :math:`\gamma \in \reals_{++}`, and relaxation parameter
    :math:`\theta \in \reals`,

    .. math::
        (\forall k \in \naturals)\quad
        \left[
        \begin{aligned}
            \bar{x}^k &= J_{\gamma G_2}(x^k - \gamma G_1(x^k)), \\
            x^{k+1} &= x^k + \theta\Big(\bar{x}^k - \gamma G_1(\bar{x}^k) - (x^k - \gamma G_1(x^k))\Big).
        \end{aligned}
        \right.

    State-space representation
    --------------------------

    The update can be written in the algorithm representation with

    .. math::
        \begin{aligned}
            \bx^k &= x^k, \\
            \bu^k &= \left(
            G_1(x^k),\;
            G_1(\bar{x}^k),\;
            \frac{x^k-\gamma G_1(x^k)-\bar{x}^k}{\gamma}
            \right), \\
            \by^k &= (x^k, \bar{x}^k, \bar{x}^k).
        \end{aligned}

    With this representation, the system matrices are

    .. math::
        \begin{aligned}
            A_k &= \begin{bmatrix} 1 \end{bmatrix}, &
            B_k &= \begin{bmatrix} 0 & -\gamma\theta & -\gamma\theta \end{bmatrix}, \\
            C_k &=
            \begin{bmatrix}
            1 \\
            1 \\
            1
            \end{bmatrix}, &
            D_k &=
            \begin{bmatrix}
            0 & 0 & 0 \\
            -\gamma & 0 & -\gamma \\
            -\gamma & 0 & -\gamma
            \end{bmatrix}.
        \end{aligned}

    These are the system matrices returned by :meth:`~autolyap.algorithms.Algorithm.get_ABCD`.

    Structural parameters
    ---------------------

    .. math::
        n = 1,\quad m = 2,\quad (\bar{m}_i)_{i=1}^{m} = (2,1),\quad \bar{m} = 3.

    .. math::
        I_{\text{func}} = \varnothing,\quad I_{\text{op}} = \{1,2\}.
    """
    def __init__(self, gamma, theta):
        r"""
        Initialize the Tseng FBF method.

        Structural inputs passed to :class:`~autolyap.algorithms.Algorithm` are

        .. math::
            n = 1,\quad m = 2,\quad (\bar m_i)_{i=1}^{m} = (2,1),\quad \bar m = 3,\quad
            I_{\mathrm{func}} = \varnothing,\quad I_{\mathrm{op}} = \{1,2\}.
        """
        super().__init__(1, 2, [2, 1], [], [1, 2]) 
        self.gamma = gamma
        self.theta = theta
    
    def set_gamma(self, gamma: float) -> None:
        r"""
        Set the step-size parameter :math:`\gamma`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.TsengFBF`.

        **Parameters**

        - `gamma` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`\gamma`.

        **Raises**

        - `ValueError`: If `gamma` is not a finite real number or if :math:`\gamma \le 0`.
        """
        gamma = self._validate_positive_finite_real(gamma, "gamma")
        self._set_dynamic_parameter("gamma", gamma)
    
    def set_theta(self, theta: float) -> None:
        r"""
        Set the relaxation parameter :math:`\theta`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.TsengFBF`.

        **Parameters**

        - `theta` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`\theta`.

        **Raises**

        - `ValueError`: If `theta` is not a finite real number.
        """
        theta = self._validate_finite_real(theta, "theta")
        self._set_dynamic_parameter("theta", theta)
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A = np.array([[1]])
        B = np.array([[0, -self.gamma*self.theta, -self.gamma*self.theta]])
        C = np.array([[1],
                      [1],
                      [1]])
        D = np.array([[0, 0, 0], 
                      [-self.gamma, 0, -self.gamma],
                      [-self.gamma, 0, -self.gamma]])
        return (A, B, C, D)
