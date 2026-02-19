import numpy as np
import math
from typing import Tuple
from .algorithm import Algorithm

class OptimizedGradientMethod(Algorithm):
    r"""
    Optimized gradient method :cite:`kim2015optimizedfirstorder`.

    See :doc:`3. Algorithm representation </theory/algorithm_representation>`
    for mathematical notation and definitions.

    Notation-driven assumptions are declared by the user via
    :class:`~autolyap.problemclass.InclusionProblem`: when present, terms written with
    :math:`\nabla` use differentiable functions, terms written with
    :math:`\prox_{\gamma f}` use proper, lower semicontinuous, convex functions,
    and terms written with :math:`J_{\gamma G}` use maximally monotone operators.

    Standard form
    -------------


    For initial points :math:`x^0,y^0 \in \calH`, smoothness constant
    :math:`L \in \reals_{++}`, and iteration budget :math:`K \in \naturals`,

    .. math::
        (\forall k \in \llbracket 0, K-1 \rrbracket)\quad
        \left[
        \begin{aligned}
            y^{k+1} &= x^k - \frac{1}{L}\nabla f(x^k), \\
            x^{k+1} &= y^{k+1} + \frac{\theta_k - 1}{\theta_{k+1}}(y^{k+1} - y^k) + \frac{\theta_k}{\theta_{k+1}}(y^{k+1} - x^k).
        \end{aligned}
        \right.

    .. math::
        \theta_k =
        \begin{cases}
            1, & \text{if } k = 0, \\
            \dfrac{1 + \sqrt{1 + 4\theta_{k-1}^2}}{2},
            & \text{if } k \in \llbracket 1, K-1 \rrbracket, \\
            \dfrac{1 + \sqrt{1 + 8\theta_{k-1}^2}}{2},
            & \text{if } k = K.
        \end{cases}

    State-space representation
    --------------------------

    The update can be written in the algorithm representation with

    .. math::
        \bx^k = (x^k, y^k), \qquad
        \bu^k = \nabla f(x^k), \qquad
        \by^k = x^k.

    For :math:`k \in \llbracket 0, K-1 \rrbracket`, with :math:`\theta_k` as in
    the standard form, the system matrices are

    .. math::
        \begin{aligned}
            A_k &=
            \begin{bmatrix}
            1+\frac{\theta_k-1}{\theta_{k+1}} & \frac{1-\theta_k}{\theta_{k+1}} \\
            1 & 0
            \end{bmatrix}, &
            B_k &=
            \begin{bmatrix}
            -\frac{1+(2\theta_k-1)/\theta_{k+1}}{L} \\
            -\frac{1}{L}
            \end{bmatrix}, \\
            C_k &= \begin{bmatrix} 1 & 0 \end{bmatrix}, &
            D_k &= \begin{bmatrix} 0 \end{bmatrix}.
        \end{aligned}

    These are the system matrices returned by :meth:`~autolyap.algorithms.Algorithm.get_ABCD` for
    :math:`k \in \llbracket 0, K-1 \rrbracket`.

    For :math:`k = K`, the system matrices are chosen for convenience as

    .. math::
        \begin{aligned}
            A_K &=
            \begin{bmatrix}
            0 & 0 \\
            0 & 0
            \end{bmatrix}, &
            B_K &=
            \begin{bmatrix}
            0 \\
            0
            \end{bmatrix}, \\
            C_K &= \begin{bmatrix} 1 & 0 \end{bmatrix}, &
            D_K &= \begin{bmatrix} 0 \end{bmatrix}.
        \end{aligned}

    These are the terminal system matrices returned by :meth:`~autolyap.algorithms.Algorithm.get_ABCD` for
    :math:`k = K`.

    Structural parameters
    ---------------------

    .. math::
        n = 2,\quad m = 1,\quad (\bar{m}_i)_{i=1}^{m} = (1),\quad \bar{m} = 1.

    .. math::
        I_{\text{func}} = \{1\},\quad I_{\text{op}} = \varnothing.
    """
    def __init__(self, L: float, K: int) -> None:
        r"""
        Initialize the optimized gradient method.

        Structural inputs passed to :class:`~autolyap.algorithms.Algorithm` are

        .. math::
            n = 2,\quad m = 1,\quad (\bar m_i)_{i=1}^{m} = (1),\quad \bar m = 1,\quad
            I_{\mathrm{func}} = \{1\},\quad I_{\mathrm{op}} = \varnothing.
        """
        super().__init__(2, 1, [1], [1], [])
        self.set_L(L)
        self.set_K(K)
    
    def set_L(self, L: float) -> None:
        r"""
        Set the smoothness parameter :math:`L`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.OptimizedGradientMethod`.

        **Parameters**

        - `L` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`L`.

        **Raises**

        - `ValueError`: If `L` is not a finite real number or if :math:`L \le 0`.
        """
        L = self._validate_positive_finite_real(L, "L")
        self._set_dynamic_parameter("L", L)
    
    def set_K(self, K: int) -> None:
        r"""
        Set the iteration budget :math:`K`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.OptimizedGradientMethod`.

        **Parameters**

        - `K` (:class:`int`): The value corresponding to :math:`K`.

        **Raises**

        - `ValueError`: If `K` is not an integer or if :math:`K < 0`.
        """
        K = self._validate_nonnegative_integral(K, "K")
        self._set_dynamic_parameter("K", K)
    
    def compute_theta(self, k: int, K: int) -> float:
        r"""
        Compute :math:`\theta_k` for horizon :math:`K`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.OptimizedGradientMethod`.

        **Parameters**

        - `k` (:class:`int`): Target index :math:`k`, with :math:`0 \le k \le K`.
        - `K` (:class:`int`): Horizon index :math:`K`.

        **Returns**

        - (:class:`float`): The scalar :math:`\theta_k` defined by the OGM recurrence.

        **Raises**

        - `ValueError`: If `k` or `K` are invalid, or if `k > K`.
        """
        k = self._validate_nonnegative_integral(k, "k")
        K = self._validate_nonnegative_integral(K, "K")
        if k > K:
            raise ValueError("k must be less than or equal to K.")

        theta = 1.0
        for i in range(1, k + 1):
            if i == K:
                theta = (1 + math.sqrt(1 + 8 * theta ** 2)) / 2
            else:
                theta = (1 + math.sqrt(1 + 4 * theta ** 2)) / 2
        return theta
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        k = self._validate_nonnegative_integral(k, "k")

        if k < self.K:
            theta_k = self.compute_theta(k, self.K)
            theta_kp1 = self.compute_theta(k + 1, self.K)
            
            A = np.array([[1+(theta_k-1)/theta_kp1, (1-theta_k)/theta_kp1],
                          [1, 0]])
            
            B = np.array([[-(1+(2*theta_k-1)/theta_kp1)/self.L],
                          [-1/self.L]])
            
            C = np.array([[1, 0]])
            
            D = np.array([[0]])
        
        elif k == self.K:
            A = np.array([[0, 0],
                          [0, 0]])
            
            B = np.array([[0],
                          [0]])
            
            C = np.array([[1, 0]])
            
            D = np.array([[0]])
        
        else:
            raise ValueError("k must be less than or equal to K.")
        
        return (A, B, C, D)
