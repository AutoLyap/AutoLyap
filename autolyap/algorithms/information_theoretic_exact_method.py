import numpy as np
import math
from typing import Tuple
from .algorithm import Algorithm
from autolyap.utils.validation import ensure_real_number

class ITEM(Algorithm):
    r"""
    Information-theoretic exact method (ITEM).

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


    Let :math:`q = \mu/L`, :math:`A_0 = 0`, and for all :math:`k \in \naturals`,

    .. math::
        A_{k+1}
        = \frac{(1+q)A_k + 2\left(1 + \sqrt{(1+A_k)(1+qA_k)}\right)}{(1-q)^2}.

    .. math::
        \beta_k = \frac{A_k}{(1-q)A_{k+1}}.

    .. math::
        \delta_k = \frac{(1-q)^2A_{k+1} - (1+q)A_k}{2(1+q+qA_k)}.

    For initial points :math:`x^0,z^0 \in \calH`,

    .. math::
        (\forall k \in \naturals)\quad
        \left[
        \begin{aligned}
            y^k &= (1-\beta_k)z^k + \beta_k x^k, \\
            x^{k+1} &= y^k - \frac{1}{L}\nabla f(y^k), \\
            z^{k+1} &= (1-q\delta_k)z^k + q\delta_k y^k - \frac{\delta_k}{L}\nabla f(y^k).
        \end{aligned}
        \right.

    State-space representation
    --------------------------

    The update can be written in the algorithm representation with
    :math:`\bx^k = (x^k, z^k)`, :math:`\bu^k = \nabla f(y^k)`, and
    :math:`\by^k = y^k`.

    Let :math:`q = \mu/L`, with :math:`\beta_k` and :math:`\delta_k` as in the
    standard form above. In :meth:`get_ABCD`, the matrices are

    .. math::
        \begin{aligned}
            A_k &=
            \begin{bmatrix}
            \beta_k & 1-\beta_k \\
            q\beta_k\delta_k & 1-q\beta_k\delta_k
            \end{bmatrix}, &
            B_k &=
            \begin{bmatrix}
            -\frac{1}{L} \\
            -\frac{\delta_k}{L}
            \end{bmatrix}, \\
            C_k &= \begin{bmatrix} \beta_k & 1-\beta_k \end{bmatrix}, &
            D_k &= \begin{bmatrix} 0 \end{bmatrix}.
        \end{aligned}
    """
    def __init__(self, mu, L):
        r"""
        Initialize ITEM.
        """
        super().__init__(2, 1, [1], [1], [])
        self.mu = mu
        self.L = L
    
    def set_L(self, L: float) -> None:
        r"""
        Set the smoothness parameter :math:`L`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.ITEM`.

        **Parameters**

        - `L` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`L`.

        **Raises**

        - `ValueError`: If `L` is not a finite real number or if :math:`L \le 0`.
        """
        L = ensure_real_number(L, "L", finite=True)
        if L <= 0:
            raise ValueError("L must be > 0.")
        self.L = L

    def set_mu(self, mu: float) -> None:
        r"""
        Set the strong-convexity parameter :math:`\mu`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.ITEM`.

        **Parameters**

        - `mu` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`\mu`.

        **Raises**

        - `ValueError`: If `mu` is not a finite real number or if :math:`\mu \le 0`.
        """
        mu = ensure_real_number(mu, "mu", finite=True)
        if mu <= 0:
            raise ValueError("mu must be > 0.")
        self.mu = mu
    
    def get_A(self, k: int) -> float:
        q = self.mu / self.L
        A = 0.0 
        for _ in range(k):
            A = ((1 + q) * A + 2 * (1 + math.sqrt((1 + A) * (1 + q * A)))) / ((1 - q) ** 2)
        return A

    def compute_beta(self, k: int) -> float:
        q = self.mu / self.L
        A_k = self.get_A(k)
        A_k1 = self.get_A(k + 1)
        return A_k / ((1 - q) * A_k1)

    def compute_delta(self, k: int) -> float:
        q = self.mu / self.L
        A_k = self.get_A(k)
        A_k1 = self.get_A(k + 1)
        numerator = ((1 - q) ** 2) * A_k1 - (1 + q) * A_k
        denominator = 2 * (1 + q + q * A_k)
        return numerator / denominator

    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = self.mu / self.L
        A_k = self.get_A(k)
        A_k1 = self.get_A(k + 1)
        beta = self.compute_beta(k)
        delta = self.compute_delta(k)

        A = np.array([[beta, 1-beta],
                      [q*beta*delta, 1-q*beta*delta]])
        
        B = np.array([[-1/self.L],
                      [-delta/self.L]])
        
        C = np.array([[beta, 1-beta]])
        
        D = np.array([[0]])
        
        return (A, B, C, D)
