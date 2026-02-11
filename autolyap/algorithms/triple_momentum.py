import numpy as np
from typing import Tuple
from .algorithm import Algorithm

class TripleMomentum(Algorithm):
    r"""
    Triple-momentum method.

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


    Let :math:`q = \mu/L`,

    .. math::
        \alpha = \frac{2-\sqrt{q}}{L}, \qquad
        \beta_{\mathrm{tm}} = \frac{(1-\sqrt{q})^2}{1+\sqrt{q}}, \qquad
        \gamma_{\mathrm{tm}} = \frac{(1-\sqrt{q})^2}{(2-\sqrt{q})(1+\sqrt{q})}.

    For initial points :math:`x^{-1},x^0 \in \calH`,

    .. math::
        (\forall k \in \naturals)\quad
        \left[
        \begin{aligned}
            y^k &= x^k + \gamma_{\mathrm{tm}}(x^k - x^{k-1}), \\
            x^{k+1} &= x^k + \beta_{\mathrm{tm}}(x^k - x^{k-1}) - \alpha \nabla f(y^k).
        \end{aligned}
        \right.

    State-space representation
    --------------------------

    The update can be written in the algorithm representation with
    :math:`\bx^k = (x^k, x^{k-1})`, :math:`\bu^k = \nabla f(y^k)`, and
    :math:`\by^k = y^k`.

    Let :math:`q = \mu/L`, and define

    .. math::
        \alpha = \frac{2-\sqrt{q}}{L}, \qquad
        \beta_{\mathrm{tm}} = \frac{(1-\sqrt{q})^2}{1+\sqrt{q}}, \qquad
        \gamma_{\mathrm{tm}} = \frac{(1-\sqrt{q})^2}{(2-\sqrt{q})(1+\sqrt{q})}.

    In :meth:`get_ABCD`, the matrices are

    .. math::
        \begin{aligned}
            A_k &=
            \begin{bmatrix}
            1+\beta_{\mathrm{tm}} & -\beta_{\mathrm{tm}} \\
            1 & 0
            \end{bmatrix}, &
            B_k &=
            \begin{bmatrix}
            -\alpha \\
            0
            \end{bmatrix}, \\
            C_k &=
            \begin{bmatrix}
            1+\gamma_{\mathrm{tm}} & -\gamma_{\mathrm{tm}}
            \end{bmatrix}, &
            D_k &= \begin{bmatrix} 0 \end{bmatrix}.
        \end{aligned}
    """
    def __init__(self, mu, L):
        r"""
        Initialize the triple-momentum method.
        """
        super().__init__(2, 1, [1], [1], [])
        self.set_L(L)
        self.set_mu(mu)

    def set_mu(self, mu: float) -> None:
        r"""
        Set the strong-convexity parameter :math:`\mu`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.TripleMomentum`.

        **Parameters**

        - `mu` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`\mu`.

        **Raises**

        - `ValueError`: If `mu` is not a finite real number or if :math:`\mu \le 0`.
        """
        mu = self._validate_positive_finite_real(mu, "mu")
        self._set_dynamic_parameter("mu", mu)

    def set_L(self, L: float) -> None:
        r"""
        Set the smoothness parameter :math:`L`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.TripleMomentum`.

        **Parameters**

        - `L` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`L`.

        **Raises**

        - `ValueError`: If `L` is not a finite real number or if :math:`L \le 0`.
        """
        L = self._validate_positive_finite_real(L, "L")
        self._set_dynamic_parameter("L", L)

    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = self.mu / self.L
        
        alpha = (2-np.sqrt(q))/self.L
        _beta = (1-np.sqrt(q))**2/(1+np.sqrt(q))
        gamma = (1-np.sqrt(q))**2/((2-np.sqrt(q))*(1+np.sqrt(q)))    
        
        A = np.array([[1+_beta,-_beta],[1,0]])
        B = np.array([[-alpha],[0]])
        C = np.array([[1+gamma,-gamma]])
        D = np.array([[0]])
        
        return (A, B, C, D)
