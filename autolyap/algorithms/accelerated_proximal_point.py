import numpy as np
from typing import Tuple
from .algorithm import Algorithm

class AcceleratedProximalPoint(Algorithm):
    r"""
    Accelerated proximal point method.

    See :doc:`3. Algorithm representation </theory/algorithm_representation>`
    for mathematical notation and definitions.

    Notation-driven assumptions are declared by the user via
    :class:`~autolyap.problemclass.InclusionProblem`: when present, terms written with
    :math:`\nabla` use differentiable functions, terms written with
    :math:`\prox_{\gamma f}` use proper, lower semicontinuous, convex functions,
    and terms written with :math:`J_{\gamma G}` use maximally monotone operators.

    Standard form
    -------------


    Let :math:`\lambda_k = k/(k+2)`. For initial points
    :math:`x^0,y^0,y^{-1} \in \calH` and :math:`\gamma \in \reals_{++}`,

    .. math::
        (\forall k \in \naturals)\quad
        \left[
        \begin{aligned}
            x^{k+1} &=
            \begin{cases}
                J_{\gamma G}(y^k), & \text{if type = operator}, \\
                \prox_{\gamma f}(y^k), & \text{if type = function},
            \end{cases} \\
            y^{k+1} &= x^{k+1} + \lambda_k(x^{k+1} - x^k) - \lambda_k(x^k - y^{k-1}).
        \end{aligned}
        \right.

    State-space representation
    --------------------------

    The update can be written in the algorithm representation with

    .. math::
        \bx^k = (x^k, y^k, y^{k-1}), \qquad
        \bu^k = \frac{y^k - x^{k+1}}{\gamma}, \qquad
        \by^k = y^k.

    With :math:`\lambda_k = k/(k+2)`, the system matrices are

    .. math::
        \begin{aligned}
            A_k &=
            \begin{bmatrix}
            0 & 1 & 0 \\
            -2\lambda_k & 1+\lambda_k & \lambda_k \\
            0 & 1 & 0
            \end{bmatrix}, &
            B_k &=
            \begin{bmatrix}
            -\gamma \\
            -\gamma(1+\lambda_k) \\
            0
            \end{bmatrix}, \\
            C_k &= \begin{bmatrix} 0 & 1 & 0 \end{bmatrix}, &
            D_k &= \begin{bmatrix} -\gamma \end{bmatrix}.
        \end{aligned}

    These are the system matrices returned by :meth:`~autolyap.algorithms.Algorithm.get_ABCD`.

    Structural parameters
    ---------------------

    .. math::
        \text{type}=\text{"operator"}:\quad
        n = 3,\quad m = 1,\quad (\bar{m}_i)_{i=1}^{m} = (1),\quad \bar{m} = 1,\quad
        I_{\text{func}} = \varnothing,\quad I_{\text{op}} = \{1\}.

    .. math::
        \text{type}=\text{"function"}:\quad
        n = 3,\quad m = 1,\quad (\bar{m}_i)_{i=1}^{m} = (1),\quad \bar{m} = 1,\quad
        I_{\text{func}} = \{1\},\quad I_{\text{op}} = \varnothing.
    """
    def __init__(self, gamma: float, type: str = "operator") -> None:
        r"""
        Initialize the accelerated proximal point method.

        Structural inputs passed to :class:`~autolyap.algorithms.Algorithm` are
        case-dependent:

        - If `type = operator`:

          .. math::
              n = 3,\quad m = 1,\quad (\bar m_i)_{i=1}^{m} = (1),\quad \bar m = 1,\quad
              I_{\mathrm{func}} = \varnothing,\quad I_{\mathrm{op}} = \{1\}.

        - If `type = function`:

          .. math::
              n = 3,\quad m = 1,\quad (\bar m_i)_{i=1}^{m} = (1),\quad \bar m = 1,\quad
              I_{\mathrm{func}} = \{1\},\quad I_{\mathrm{op}} = \varnothing.
        """
        if type == "operator":
            super().__init__(3, 1, [1], [], [1])
        elif type == "function":
            super().__init__(3, 1, [1], [1], [])
        else:
            raise ValueError("type must be either 'operator' or 'function'")
        self.gamma = gamma
    
    def set_gamma(self, gamma: float) -> None:
        r"""
        Set the step-size parameter :math:`\gamma`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.AcceleratedProximalPoint`.

        **Parameters**

        - `gamma` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`\gamma`.

        **Raises**

        - `ValueError`: If `gamma` is not a finite real number or if :math:`\gamma \le 0`.
        """
        gamma = self._validate_positive_finite_real(gamma, "gamma")
        self._set_dynamic_parameter("gamma", gamma)
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lambda_var = k / (k + 2)
        A = np.array([[0, 1 , 0], 
                      [-2*lambda_var, 1+lambda_var, lambda_var],
                      [0, 1, 0]])
        B = np.array([[-self.gamma],
                      [-self.gamma*(1+lambda_var)],
                      [0]])
        C = np.array([[0, 1, 0]])
        D = np.array([[-self.gamma]])
        return (A, B, C, D)
