import numpy as np
from typing import Tuple
from .algorithm import Algorithm

class DavisYin(Algorithm):
    r"""
    Davis--Yin three-operator splitting.

    See :doc:`3. Algorithm representation </theory/algorithm_representation>`
    for mathematical notation and definitions.

    Notation-driven assumptions are declared by the user via
    :class:`~autolyap.problemclass.InclusionProblem`: when present, terms written with
    :math:`\nabla` use differentiable functions, terms written with
    :math:`\prox_{\gamma f}` use proper, lower semicontinuous, convex functions,
    and terms written with :math:`J_{\gamma G}` use maximally monotone operators.

    Standard form
    -------------


    For an initial point :math:`x^0 \in \calH`, step size
    :math:`\gamma \in \reals_{++}`, and relaxation parameter
    :math:`\lambda \in \reals`, define,
    for all :math:`k \in \naturals`,

    .. math::
        \left[
        \begin{aligned}
            v^k &= \prox_{\gamma f_1}(x^k), \\
            w^k &= \prox_{\gamma f_3}\!\left(2v^k - x^k - \gamma \nabla f_2(v^k)\right), \\
            x^{k+1} &= x^k + \lambda (w^k - v^k).
        \end{aligned}
        \right.

    State-space representation
    --------------------------

    The update can be written in the algorithm representation with

    .. math::
        \begin{aligned}
            \bx^k &= x^k, \\
            \bu^k &= \left(
            \frac{x^k-v^k}{\gamma},\;
            \nabla f_2(v^k),\;
            \frac{2v^k-x^k-\gamma\nabla f_2(v^k)-w^k}{\gamma}
            \right), \\
            \by^k &= (v^k, v^k, w^k).
        \end{aligned}

    With this representation, the system matrices are

    .. math::
        \begin{aligned}
            A_k &= \begin{bmatrix} 1 \end{bmatrix}, &
            B_k &= \begin{bmatrix} -\gamma\lambda & -\gamma\lambda & -\gamma\lambda \end{bmatrix}, \\
            C_k &=
            \begin{bmatrix}
            1 \\
            1 \\
            1
            \end{bmatrix}, &
            D_k &=
            \begin{bmatrix}
            -\gamma & 0 & 0 \\
            -\gamma & 0 & 0 \\
            -2\gamma & -\gamma & -\gamma
            \end{bmatrix}.
        \end{aligned}

    These are the system matrices returned by :meth:`~autolyap.algorithms.Algorithm.get_ABCD`.

    Structural parameters
    ---------------------

    .. math::
        n = 1,\quad m = 3,\quad (\bar{m}_i)_{i=1}^{m} = (1,1,1),\quad \bar{m} = 3.

    .. math::
        I_{\text{func}} = \{1,2,3\},\quad I_{\text{op}} = \varnothing.
    """
    def __init__(self, gamma: float, lambda_value: float) -> None:
        r"""
        Initialize the Davis--Yin method.

        Structural inputs passed to :class:`~autolyap.algorithms.Algorithm` are

        .. math::
            n = 1,\quad m = 3,\quad (\bar m_i)_{i=1}^{m} = (1,1,1),\quad \bar m = 3,\quad
            I_{\mathrm{func}} = \{1,2,3\},\quad I_{\mathrm{op}} = \varnothing.
        """
        super().__init__(1, 3, [1, 1, 1], [1, 2, 3], [])
        self.gamma = gamma
        self.lambda_value = lambda_value
    
    def set_gamma(self, gamma: float) -> None:
        r"""
        Set the step-size parameter :math:`\gamma`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.DavisYin`.

        **Parameters**

        - `gamma` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`\gamma`.

        **Raises**

        - `ValueError`: If `gamma` is not a finite real number or if :math:`\gamma \le 0`.
        """
        gamma = self._validate_positive_finite_real(gamma, "gamma")
        self._set_dynamic_parameter("gamma", gamma)

    def set_lambda(self, lambda_value: float) -> None:
        r"""
        Set the relaxation parameter :math:`\lambda`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.DavisYin`.

        **Parameters**

        - `lambda_value` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`\lambda`.

        **Raises**

        - `ValueError`: If `lambda_value` is not a finite real number.
        """
        lambda_value = self._validate_finite_real(lambda_value, "lambda_value")
        self._set_dynamic_parameter("lambda_value", lambda_value)
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A = np.array([[1]])
        B = -self.gamma*self.lambda_value*np.array([[1,1,1]])
        C = np.array([[1],[1],[1]])
        D = -self.gamma*np.array([[1,0,0],[1,0,0],[2,1,1]])
        return (A, B, C, D)
