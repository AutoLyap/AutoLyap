import numpy as np
import math
from typing import Tuple
from .algorithm import Algorithm
from autolyap.utils.validation import ensure_integral, ensure_real_number

class OptimizedGradientMethod(Algorithm):
    r"""
    Optimized gradient method.

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
    :math:`\bx^k = (x^k, y^k)`, :math:`\bu^k = \nabla f(x^k)`, and
    :math:`\by^k = x^k`.

    For :math:`k \in \llbracket 0, K-1 \rrbracket`, with :math:`\theta_k` as in
    the standard form, :meth:`get_ABCD` returns

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

    For :math:`k = K`, :meth:`get_ABCD` returns

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
    """
    def __init__(self, L, K):
        r"""
        Initialize the optimized gradient method.
        """
        super().__init__(2, 1, [1], [1], [])
        self.L = L
        self.K = K
    
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
        L = ensure_real_number(L, "L", finite=True)
        if L <= 0:
            raise ValueError("L must be > 0.")
        self.L = L
    
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
        K = ensure_integral(K, "K", minimum=0)
        self.K = K
    
    def _compute_theta(self, k: int, K: int) -> float:
        if k < 0 or k > K:
            raise ValueError("k must be a non-negative integer and less than or equal to K.")
        
        theta = 1.0 
        for i in range(1, k + 1):
            if i == K:
                theta = (1 + math.sqrt(1 + 8 * theta ** 2)) / 2
            else:
                theta = (1 + math.sqrt(1 + 4 * theta ** 2)) / 2
        return theta
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if k < self.K:
            theta_k = self._compute_theta(k, self.K)
            theta_kp1 = self._compute_theta(k + 1, self.K)
            
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
