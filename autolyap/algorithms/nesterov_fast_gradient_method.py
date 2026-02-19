import numpy as np
from typing import Tuple
from .algorithm import Algorithm

class NesterovFastGradientMethod(Algorithm):
    r"""
    Nesterov's fast gradient method.

    See :doc:`3. Algorithm representation </theory/algorithm_representation>`
    for mathematical notation and definitions.

    Notation-driven assumptions are declared by the user via
    :class:`~autolyap.problemclass.InclusionProblem`: when present, terms written with
    :math:`\nabla` use differentiable functions, terms written with
    :math:`\prox_{\gamma f}` use proper, lower semicontinuous, convex functions,
    and terms written with :math:`J_{\gamma G}` use maximally monotone operators.

    Standard form
    -------------


    For initial points :math:`x^{-1},x^0 \in \calH`, step size :math:`\gamma \in \reals_{++}`,
    and :math:`\lambda_0 = 1`,

    .. math::
        (\forall k \in \naturals)\quad
        \left[
        \begin{aligned}
            y^k &= x^k + \delta_k (x^k - x^{k-1}), \\
            x^{k+1} &= y^k - \gamma \nabla f(y^k), \\
            \lambda_{k+1} &= \frac{1 + \sqrt{1 + 4\lambda_k^2}}{2}, \\
            \delta_k &= \frac{\lambda_k - 1}{\lambda_{k+1}}.
        \end{aligned}
        \right.

    This implementation also keeps an additional evaluation of :math:`\nabla f(x^k)`
    for analysis templates that require it.

    State-space representation
    --------------------------

    The update can be written in the algorithm representation with

    .. math::
        \bx^k = (x^k, x^{k-1}), \qquad
        \bu^k = (\nabla f(y^k), \nabla f(x^k)), \qquad
        \by^k = (y^k, x^k).

    With :math:`\lambda_0 = 1`, :math:`\lambda_{k+1} = \frac{1+\sqrt{1+4\lambda_k^2}}{2}`,
    and :math:`\alpha_k = \frac{\lambda_k - 1}{\lambda_{k+1}}`, the system
    matrices are

    .. math::
        \begin{aligned}
            A_k &=
            \begin{bmatrix}
            1+\alpha_k & -\alpha_k \\
            1 & 0
            \end{bmatrix}, &
            B_k &=
            \begin{bmatrix}
            -\gamma & 0 \\
            0 & 0
            \end{bmatrix}, \\
            C_k &=
            \begin{bmatrix}
            1+\alpha_k & -\alpha_k \\
            1 & 0
            \end{bmatrix}, &
            D_k &=
            \begin{bmatrix}
            0 & 0 \\
            0 & 0
            \end{bmatrix}.
        \end{aligned}

    These are the system matrices returned by :meth:`~autolyap.algorithms.Algorithm.get_ABCD`.

    Structural parameters
    ---------------------

    .. math::
        n = 2,\quad m = 1,\quad (\bar{m}_i)_{i=1}^{m} = (2),\quad \bar{m} = 2.

    .. math::
        I_{\text{func}} = \{1\},\quad I_{\text{op}} = \varnothing.
    """
    def __init__(self, gamma: float) -> None:
        r"""
        Initialize the fast gradient method.

        Structural inputs passed to :class:`~autolyap.algorithms.Algorithm` are

        .. math::
            n = 2,\quad m = 1,\quad (\bar m_i)_{i=1}^{m} = (2),\quad \bar m = 2,\quad
            I_{\mathrm{func}} = \{1\},\quad I_{\mathrm{op}} = \varnothing.
        """
        super().__init__(2, 1, [2], [1], [])
        self.gamma = gamma
    
    def set_gamma(self, gamma: float) -> None:
        r"""
        Set the step-size parameter :math:`\gamma`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.NesterovFastGradientMethod`.

        **Parameters**

        - `gamma` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`\gamma`.

        **Raises**

        - `ValueError`: If `gamma` is not a finite real number or if :math:`\gamma \le 0`.
        """
        gamma = self._validate_positive_finite_real(gamma, "gamma")
        self._set_dynamic_parameter("gamma", gamma)
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lambda_var = 1
        for _ in range(0, k + 1):
            lambda_var_prev = lambda_var
            lambda_var = (1 + np.sqrt(1 + 4 * lambda_var ** 2)) / 2

        alpha = (lambda_var_prev - 1) / lambda_var

        A = np.array([[1+alpha, -alpha],
                      [1, 0]])
        
        B = np.array([[-self.gamma, 0],
                      [0, 0]])
        
        C = np.array([[1+alpha, -alpha],
                      [1, 0]])
        
        D = np.array([[0, 0],
                      [0, 0]])
        
        return (A, B, C, D)
