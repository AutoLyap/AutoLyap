import numpy as np
import math
from typing import Tuple
from .algorithm import Algorithm

class ITEM(Algorithm):
    r"""
    Information-theoretic exact method (ITEM).

    See :doc:`3. Algorithm representation </theory/algorithm_representation>`
    for mathematical notation and definitions.

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

    .. math::
        \bx^k = (x^k, z^k), \qquad
        \bu^k = \nabla f(y^k), \qquad
        \by^k = y^k.

    Let :math:`q = \mu/L`, with :math:`\beta_k` and :math:`\delta_k` as in the
    standard form above. The system matrices are

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

    These are the system matrices returned by :meth:`~autolyap.algorithms.Algorithm.get_ABCD`.

    Structural parameters
    ---------------------

    .. math::
        n = 2,\quad m = 1,\quad (\bar{m}_i)_{i=1}^{m} = (1),\quad \bar{m} = 1.

    .. math::
        I_{\text{func}} = \{1\},\quad I_{\text{op}} = \varnothing.
    """
    def __init__(self, mu: float, L: float) -> None:
        r"""
        Initialize ITEM.

        Structural inputs passed to :class:`~autolyap.algorithms.Algorithm` are

        .. math::
            n = 2,\quad m = 1,\quad (\bar m_i)_{i=1}^{m} = (1),\quad \bar m = 1,\quad
            I_{\mathrm{func}} = \{1\},\quad I_{\mathrm{op}} = \varnothing.
        """
        super().__init__(2, 1, [1], [1], [])
        self.set_L(L)
        self.set_mu(mu)

    @staticmethod
    def _validate_mu_lt_L(mu: float, L: float) -> None:
        if mu >= L:
            raise ValueError(f"Require mu < L. Got mu={mu}, L={L}.")
    
    def set_L(self, L: float) -> None:
        r"""
        Set the smoothness parameter :math:`L`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.ITEM`.

        **Parameters**

        - `L` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`L`.

        **Raises**

        - `ValueError`: If `L` is not a finite real number, if :math:`L \le 0`,
          or if :math:`\mu \ge L`.
        """
        L = self._validate_positive_finite_real(L, "L")
        if hasattr(self, "mu"):
            self._validate_mu_lt_L(self.mu, L)
        self._set_dynamic_parameter("L", L)

    def set_mu(self, mu: float) -> None:
        r"""
        Set the strong-convexity parameter :math:`\mu`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.ITEM`.

        **Parameters**

        - `mu` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`\mu`.

        **Raises**

        - `ValueError`: If `mu` is not a finite real number, if :math:`\mu \le 0`,
          or if :math:`\mu \ge L`.
        """
        mu = self._validate_positive_finite_real(mu, "mu")
        if hasattr(self, "L"):
            self._validate_mu_lt_L(mu, self.L)
        self._set_dynamic_parameter("mu", mu)
    
    def get_A(self, k: int) -> float:
        k = self._validate_nonnegative_integral(k, "k")
        q = self.mu / self.L
        A = 0.0 
        for _ in range(k):
            A = ((1 + q) * A + 2 * (1 + math.sqrt((1 + A) * (1 + q * A)))) / ((1 - q) ** 2)
        return A

    def compute_beta(self, k: int) -> float:
        k = self._validate_nonnegative_integral(k, "k")
        q = self.mu / self.L
        A_k = self.get_A(k)
        A_k1 = self.get_A(k + 1)
        return A_k / ((1 - q) * A_k1)

    def compute_delta(self, k: int) -> float:
        k = self._validate_nonnegative_integral(k, "k")
        q = self.mu / self.L
        A_k = self.get_A(k)
        A_k1 = self.get_A(k + 1)
        numerator = ((1 - q) ** 2) * A_k1 - (1 + q) * A_k
        denominator = 2 * (1 + q + q * A_k)
        return numerator / denominator

    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        k = self._validate_nonnegative_integral(k, "k")
        q = self.mu / self.L
        beta = self.compute_beta(k)
        delta = self.compute_delta(k)

        A = np.array([[beta, 1-beta],
                      [q*beta*delta, 1-q*beta*delta]])
        
        B = np.array([[-1/self.L],
                      [-delta/self.L]])
        
        C = np.array([[beta, 1-beta]])
        
        D = np.array([[0]])
        
        return (A, B, C, D)
