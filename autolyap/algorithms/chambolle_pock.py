import numpy as np
from typing import Tuple
from .algorithm import Algorithm
from autolyap.utils.validation import ensure_real_number

class ChambollePock(Algorithm):
    r"""
    Chambolle--Pock primal-dual method.

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


    In the identity-operator case, for initial points :math:`x^0,y^0 \in \calH`,
    primal and dual step sizes :math:`\tau,\sigma \in \reals_{++}`, and
    relaxation parameter :math:`\theta \in \reals`,

    .. math::
        (\forall k \in \naturals)\quad
        \left[
        \begin{aligned}
            x^{k+1} &= \prox_{\tau f_1}(x^k - \tau y^k), \\
            y^{k+1} &= \prox_{\sigma f_2^*}\!\left(y^k + \sigma \left(x^{k+1} + \theta (x^{k+1} - x^k)\right)\right).
        \end{aligned}
        \right.

    State-space representation
    --------------------------

    Using only variables and parameters from the standard form, set

    .. math::
        \bx^k = (x^k, y^k), \qquad
        \bu^k = \left(\frac{x^k-\tau y^k-x^{k+1}}{\tau},\; y^{k+1}\right),

    .. math::
        \by^k = \left(
        x^{k+1},\;
        \frac{y^k + \sigma\left(x^{k+1} + \theta(x^{k+1}-x^k)\right)-y^{k+1}}{\sigma}
        \right).

    In :meth:`get_ABCD`, the matrices are

    .. math::
        \begin{aligned}
            A_k &=
            \begin{bmatrix}
            1 & -\tau \\
            0 & 0
            \end{bmatrix}, &
            B_k &=
            \begin{bmatrix}
            -\tau & 0 \\
            0 & 1
            \end{bmatrix}, \\
            C_k &=
            \begin{bmatrix}
            1 & -\tau \\
            1 & \frac{1}{\sigma} - \tau(1+\theta)
            \end{bmatrix}, &
            D_k &=
            \begin{bmatrix}
            -\tau & 0 \\
            -\tau(1+\theta) & -\frac{1}{\sigma}
            \end{bmatrix}.
        \end{aligned}
    """
    def __init__(self, tau, sigma, theta):
        r"""
        Initialize the Chambolle--Pock method.
        """
        super().__init__(2, 2, [1, 1], [1, 2], [])
        self.tau = tau
        self.sigma = sigma
        self.theta = theta
    
    def set_tau(self, tau: float) -> None:
        r"""
        Set the primal step size :math:`\tau`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.ChambollePock`.

        **Parameters**

        - `tau` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`\tau`.

        **Raises**

        - `ValueError`: If `tau` is not a finite real number or if :math:`\tau \le 0`.
        """
        tau = ensure_real_number(tau, "tau", finite=True)
        if tau <= 0:
            raise ValueError("tau must be > 0.")
        self.tau = tau
    
    def set_sigma(self, sigma: float) -> None:
        r"""
        Set the dual step size :math:`\sigma`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.ChambollePock`.

        **Parameters**

        - `sigma` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`\sigma`.

        **Raises**

        - `ValueError`: If `sigma` is not a finite real number or if :math:`\sigma \le 0`.
        """
        sigma = ensure_real_number(sigma, "sigma", finite=True)
        if sigma <= 0:
            raise ValueError("sigma must be > 0.")
        self.sigma = sigma
    
    def set_theta(self, theta: float) -> None:
        r"""
        Set the relaxation parameter :math:`\theta`.

        Shared notation follows the class-level reference in
        :class:`~autolyap.algorithms.ChambollePock`.

        **Parameters**

        - `theta` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The value corresponding to :math:`\theta`.

        **Raises**

        - `ValueError`: If `theta` is not a finite real number.
        """
        theta = ensure_real_number(theta, "theta", finite=True)
        self.theta = theta
    
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A = np.array([[1, -self.tau],
                      [0, 0]])
        B = np.array([[-self.tau, 0],
                      [0, 1]])
        C = np.array([[1, -self.tau], 
                      [1, 1/self.sigma - self.tau*(1+self.theta)]])
        D = np.array([[-self.tau, 0], 
                      [-self.tau*(1+self.theta), -1/self.sigma]])
        return (A, B, C, D)
