import numpy as np
from typing import Tuple
from .algorithm import Algorithm

class DouglasRachford(Algorithm):
    r"""
    Douglas--Rachford splitting :cite:`douglas1956numericalsolutionheat`, :cite:`eckstein1992douglasrachfordsplittingmethod`, :cite:`lions1979splittingalgorithmssum`.

    See :doc:`3. Algorithm representation </theory/algorithm_representation>`
    for mathematical notation and definitions.

    Notation-driven assumptions are declared by the user via
    :class:`~autolyap.problemclass.InclusionProblem`: when present, terms written with
    :math:`\nabla` use differentiable functions, terms written with
    :math:`\prox_{\gamma f}` use proper, lower semicontinuous, convex functions,
    and terms written with :math:`J_{\gamma G}` use maximally monotone operators.

    Standard form
    -------------


    For an initial point :math:`x^0 \in \calH`, step size :math:`\gamma \in \reals_{++}`,
    and relaxation parameter :math:`\lambda \in \reals`,

    .. math::
        (\forall k \in \naturals)\quad
        \left[
        \begin{aligned}
            v^k &=
            \begin{cases}
                J_{\gamma G_1}(x^k), & \text{if type = operator}, \\
                \prox_{\gamma f_1}(x^k), & \text{if type = function},
            \end{cases} \\
            w^k &=
            \begin{cases}
                J_{\gamma G_2}(2v^k - x^k), & \text{if type = operator}, \\
                \prox_{\gamma f_2}(2v^k - x^k), & \text{if type = function},
            \end{cases} \\
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
            \frac{2v^k-x^k-w^k}{\gamma}
            \right), \\
            \by^k &= \left(x^k,\; 2v^k-x^k\right).
        \end{aligned}

    With this representation, the system matrices are

    .. math::
        \begin{aligned}
            A_k &= \begin{bmatrix} 1 \end{bmatrix}, &
            B_k &= \begin{bmatrix} -\gamma\lambda & -\gamma\lambda \end{bmatrix}, \\
            C_k &=
            \begin{bmatrix}
            1 \\
            1
            \end{bmatrix}, &
            D_k &=
            \begin{bmatrix}
            -\gamma & 0 \\
            -2\gamma & -\gamma
            \end{bmatrix}.
        \end{aligned}

    These are the system matrices returned by :meth:`~autolyap.algorithms.Algorithm.get_ABCD`.

    Structural parameters
    ---------------------

    .. math::
        \text{type}=\text{"operator"}:\quad
        n = 1,\quad m = 2,\quad (\bar{m}_i)_{i=1}^{m} = (1,1),\quad \bar{m} = 2,\quad
        I_{\text{func}} = \varnothing,\quad I_{\text{op}} = \{1,2\}.

    .. math::
        \text{type}=\text{"function"}:\quad
        n = 1,\quad m = 2,\quad (\bar{m}_i)_{i=1}^{m} = (1,1),\quad \bar{m} = 2,\quad
        I_{\text{func}} = \{1,2\},\quad I_{\text{op}} = \varnothing.

    """
    def __init__(self, gamma: float, lambda_value: float, type: str = "operator") -> None:
        r"""
        Initialize the Douglas--Rachford method.

        Structural inputs passed to :class:`~autolyap.algorithms.Algorithm` are
        case-dependent:

        - If `type = operator`:

          .. math::
              n = 1,\quad m = 2,\quad (\bar m_i)_{i=1}^{m} = (1,1),\quad \bar m = 2,\quad
              I_{\mathrm{func}} = \varnothing,\quad I_{\mathrm{op}} = \{1,2\}.

        - If `type = function`:

          .. math::
              n = 1,\quad m = 2,\quad (\bar m_i)_{i=1}^{m} = (1,1),\quad \bar m = 2,\quad
              I_{\mathrm{func}} = \{1,2\},\quad I_{\mathrm{op}} = \varnothing.
        """
        if type == "operator":
            super().__init__(1, 2, [1, 1], [], [1, 2])
        elif type == "function":
            super().__init__(1, 2, [1, 1], [1, 2], [])
        else:
            raise ValueError("type must be either 'operator' or 'function'")
        self.gamma = gamma
        self.lambda_value = lambda_value
    
    def set_gamma(self, gamma: float) -> None:
        r"""
        Set the step-size parameter :math:`\gamma`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.DouglasRachford`.

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
        :class:`~autolyap.algorithms.DouglasRachford`.

        **Parameters**

        - `lambda_value` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`\lambda`.

        **Raises**

        - `ValueError`: If `lambda_value` is not a finite real number.
        """
        lambda_value = self._validate_finite_real(lambda_value, "lambda_value")
        self._set_dynamic_parameter("lambda_value", lambda_value)
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A = np.array([[1]])
        B = np.array([[-self.gamma*self.lambda_value, -self.gamma*self.lambda_value]])
        C = np.array([[1], 
                      [1]])
        D = np.array([[-self.gamma, 0], 
                      [-2*self.gamma, -self.gamma]])
        return (A, B, C, D)
