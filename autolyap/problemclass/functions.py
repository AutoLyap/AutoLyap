"""Concrete function interpolation-condition classes."""

import numpy as np
from typing import List, Union, Tuple

from autolyap.problemclass.base import FunctionInterpolationCondition
from autolyap.problemclass.indices import InterpolationIndices
from autolyap.utils.validation import ensure_real_number

INF = float("inf")


def _ensure_positive_finite(value: Union[int, float], parameter_name: str, error_message: str) -> float:
    numeric_value = ensure_real_number(value, parameter_name)
    if not np.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(error_message)
    return numeric_value


def _ensure_positive_mu_tilde(mu_tilde: Union[int, float], context_name: str) -> Tuple[float, float]:
    validated = _ensure_positive_finite(
        mu_tilde,
        "Parameter mu_tilde",
        f"For {context_name}, mu_tilde must be > 0 and finite.",
    )
    return validated, -validated


class ParametrizedFunctionInterpolationCondition(FunctionInterpolationCondition):
    r"""
    Base class for function interpolation conditions parameterized by :math:`\mu` and :math:`L`.

    Class-level reference
    =====================

    This class-level docstring centralizes notation for the shared
    :math:`(\mu, L)` interpolation family used by its concrete subclasses.

    Provides a helper to compute interpolation data from :math:`\mu` and :math:`L`.
    This base class supports both smooth and nonsmooth conditions by setting :math:`L` appropriately.

    Let :math:`(\calH,\langle\cdot,\cdot\rangle)` be a real Hilbert space and
    :math:`f: \calH \to \mathbb{R} \cup \{\pm\infty\}`.
    The parameters satisfy :math:`-\infty < \mu < L \le +\infty` with :math:`L > 0`
    (nonsmooth cases are encoded by :math:`L = +\infty`).

    - Interpolation inequality:

      For any :math:`x_{j_1}, x_{j_2} \in \calH` with
      :math:`g_{j_1} \in \partial f(x_{j_1})`, :math:`g_{j_2} \in \partial f(x_{j_2})`,
      and :math:`F_{j_1} = f(x_{j_1})`, :math:`F_{j_2} = f(x_{j_2})`, the condition enforced is

      .. math::
          F_{j_1} \ge F_{j_2} + \langle g_{j_2}, x_{j_1} - x_{j_2} \rangle + \frac{\mu}{2}\|x_{j_1} - x_{j_2}\|^2
          + \frac{1}{2(L-\mu)}\|g_{j_1} - g_{j_2} - \mu(x_{j_1} - x_{j_2})\|^2,

      where :math:`\frac{1}{2(L-\mu)}` is interpreted as :math:`0` when :math:`L = +\infty`.

    - Interpolation data:

      The returned data uses interpolation indices ``j1!=j2`` and an inequality constraint with
      :math:`a = (-1,1)` applied to the function values :math:`(F_{j_1},F_{j_2})`. The quadratic term uses
      :math:`z = (x_{j_1},x_{j_2},g_{j_1},g_{j_2})` and a matrix :math:`M` given by

      .. math::
          M =
          \begin{cases}
            \frac{1}{2(L-\mu)}
            \begin{bmatrix}
              L\mu & -L\mu & -\mu & L \\
              -L\mu & L\mu & \mu & -L \\
              -\mu & \mu & 1 & -1 \\
              L & -L & -1 & 1
            \end{bmatrix}, & \text{if } L < +\infty, \\
            \frac{1}{2}
            \begin{bmatrix}
              \mu & -\mu & 0 & 1 \\
              -\mu & \mu & 0 & -1 \\
              0 & 0 & 0 & 0 \\
              1 & -1 & 0 & 0
            \end{bmatrix}, & \text{if } L = +\infty.
          \end{cases}

      The `eq` flag returned by :meth:`get_data` is ``False`` for this class because the
      interpolation constraint is an inequality.

    Note: Many classes below are specializations obtained by choosing :math:`\mu` and :math:`L`
    (e.g., weakly convex uses :math:`\mu = -\tilde{\mu}` with :math:`L=+\infty`; smooth uses
    :math:`\mu = -L`).

    **Parameters**

    - `mu` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The convexity parameter corresponding to :math:`\mu`.
      For strongly convex functions, :math:`\mu > 0`; for convex functions, :math:`\mu = 0`; for weakly convex
      functions, :math:`\mu < 0`.
    - `L` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The smoothness parameter corresponding to :math:`L`.
      For nonsmooth functions, :math:`L` is set to infinity.

    **Raises**

    - `ValueError`: If `mu` is not a number, if :math:`L \le 0` with :math:`L < +\infty`,
      or if :math:`\mu \geq L`.
    """
    def __init__(self, mu: Union[int, float], L: Union[int, float]):
        mu, L = self._validate_mu_L(mu, L)
        self.mu = mu
        self.L = L
        self._interpolation_data = self._compute_interpolation_data(mu, L)

    @staticmethod
    def _validate_mu_L(mu: Union[int, float], L: Union[int, float]) -> Tuple[float, float]:
        mu = ensure_real_number(mu, "Parameter mu")
        L = ensure_real_number(L, "Parameter L")
        if L != INF and L <= 0:
            raise ValueError("Parameter L must be positive or +inf.")
        if mu == -INF:
            raise ValueError("ParametrizedFunctionInterpolationCondition: mu cannot be -inf.")
        if not (mu < L):
            raise ValueError("ParametrizedFunctionInterpolationCondition requires that -inf < mu < L <= +inf.")
        return mu, L

    @staticmethod
    def _compute_interpolation_data(mu: float, L: float) -> Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]:
        r"""
        Compute the interpolation data based on mu and L.

        The tuple format and notation follow the class-level reference in
        :class:`~autolyap.problemclass.ParametrizedFunctionInterpolationCondition`.

        When L is infinite, the condition is nonsmooth and a simpler interpolation matrix is used.
        When L is finite, the interpolation data follows the formula for smooth functions.

        **Parameters**

        - `mu` (:class:`float`): The convexity parameter corresponding to :math:`\mu`.
        - `L` (:class:`float`): The smoothness parameter corresponding to :math:`L`.

        **Returns**

        - (:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`bool`, :class:`~autolyap.problemclass.problemclass.InterpolationIndices`\]): A tuple containing the
          matrix, vector, eq flag, and interpolation indices.

        """
        if L == INF:
            matrix = 0.5 * np.array([
                [mu, -mu, 0, 1],
                [-mu, mu, 0, -1],
                [0, 0, 0, 0],
                [1, -1, 0, 0]
            ])
        else:
            matrix = (1 / (2 * (L - mu))) * np.array([
                [L * mu, -L * mu, -mu, L],
                [-L * mu, L * mu, mu, -L],
                [-mu, mu, 1, -1],
                [L, -L, -1, 1]
            ])
        vector = np.array([-1, 1])
        eq = False
        interp_idx = InterpolationIndices("j1!=j2")
        return matrix, vector, eq, interp_idx

    def get_data(self) -> List[Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]]:
        r"""
        Return interpolation data for the function condition.

        The tuple format and notation follow the class-level reference in
        :class:`~autolyap.problemclass.ParametrizedFunctionInterpolationCondition`.

        **Returns**

        - (:class:`~typing.List`\[:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`bool`, :class:`~autolyap.problemclass.problemclass.InterpolationIndices`\]\]): A list containing one
          tuple with the matrix, vector, eq flag, and interpolation indices.

        """
        return [self._interpolation_data]

class Convex(ParametrizedFunctionInterpolationCondition):
    r"""
    Function interpolation condition for proper, lower semicontinuous, and convex functions.

    Let :math:`(\calH,\langle\cdot,\cdot\rangle)` be a real Hilbert space and
    :math:`f: \calH \to \mathbb{R} \cup \{\pm\infty\}`.

    - Effective domain:

      :math:`\operatorname{dom} f = \{x \in \calH \mid f(x) < +\infty\}`.

    - Proper:

      :math:`-\infty \notin f(\calH)` and :math:`\operatorname{dom} f \neq \emptyset`.

    - Lower semicontinuous:

      :math:`\liminf_{y \to x} f(y) \geq f(x)` for each :math:`x \in \calH`.

    - Convex:

    .. math::
        f((1-\lambda)x + \lambda y) \leq (1-\lambda) f(x) + \lambda f(y)
        \quad \text{for each } x,y \in \calH,\; \lambda \in [0,1].

    - Interpolation inequality:

      For any :math:`x_{j_1}, x_{j_2} \in \calH` with
      :math:`g_{j_1} \in \partial f(x_{j_1})`, :math:`g_{j_2} \in \partial f(x_{j_2})`,
      and :math:`F_{j_1} = f(x_{j_1})`, :math:`F_{j_2} = f(x_{j_2})`,

      .. math::
          F_{j_1} \ge F_{j_2} + \langle g_{j_2}, x_{j_1} - x_{j_2} \rangle.

    """
    def __init__(self):
        super().__init__(mu=0.0, L=INF)

class StronglyConvex(ParametrizedFunctionInterpolationCondition):
    r"""
    Function interpolation condition for proper, lower semicontinuous, and strongly convex functions.

    Let :math:`(\calH,\langle\cdot,\cdot\rangle)` be a real Hilbert space and
    :math:`f: \calH \to \mathbb{R} \cup \{\pm\infty\}`.

    - Effective domain:

      :math:`\operatorname{dom} f = \{x \in \calH \mid f(x) < +\infty\}`.

    - Proper:

      :math:`-\infty \notin f(\calH)` and :math:`\operatorname{dom} f \neq \emptyset`.

    - Lower semicontinuous:

      :math:`\liminf_{y \to x} f(y) \geq f(x)` for each :math:`x \in \calH`.

    - :math:`\mu`-strongly convex with :math:`\mu \in \mathbb{R}_{++}`:

      .. math::
          f - \frac{\mu}{2}\|\cdot\|^2 \quad \text{is convex}.

    - Interpolation inequality:

      For any :math:`x_{j_1}, x_{j_2} \in \calH` with
      :math:`g_{j_1} \in \partial f(x_{j_1})`, :math:`g_{j_2} \in \partial f(x_{j_2})`,
      and :math:`F_{j_1} = f(x_{j_1})`, :math:`F_{j_2} = f(x_{j_2})`,

      .. math::
          F_{j_1} \ge F_{j_2} + \langle g_{j_2}, x_{j_1} - x_{j_2} \rangle + \frac{\mu}{2}\|x_{j_1} - x_{j_2}\|^2.

    **Parameters**

    - `mu` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): Strong convexity parameter corresponding to
      :math:`\mu` (must be :math:`> 0` and finite).

    **Raises**

    - `ValueError`: If `mu` is not valid.
    """
    def __init__(self, mu: Union[int, float]):
        mu = _ensure_positive_finite(mu, "Parameter mu", "For StronglyConvex, mu must be > 0 and finite.")
        super().__init__(mu=mu, L=INF)

class WeaklyConvex(ParametrizedFunctionInterpolationCondition):
    r"""
    Function interpolation condition for proper, lower semicontinuous, and weakly convex functions.

    Let :math:`(\calH,\langle\cdot,\cdot\rangle)` be a real Hilbert space and
    :math:`f: \calH \to \mathbb{R} \cup \{\pm\infty\}`.

    - Effective domain:

      :math:`\operatorname{dom} f = \{x \in \calH \mid f(x) < +\infty\}`.

    - Proper:

      :math:`-\infty \notin f(\calH)` and :math:`\operatorname{dom} f \neq \emptyset`.

    - Lower semicontinuous:

      :math:`\liminf_{y \to x} f(y) \geq f(x)` for each :math:`x \in \calH`.

    - :math:`\tilde{\mu}`-weakly convex with :math:`\tilde{\mu} \in \mathbb{R}_{++}`:

      .. math::
          f + \frac{\tilde{\mu}}{2}\|\cdot\|^2 \quad \text{is convex}.

    - Interpolation inequality:

      For any :math:`x_{j_1}, x_{j_2} \in \calH` with
      :math:`g_{j_1} \in \partial f(x_{j_1})`, :math:`g_{j_2} \in \partial f(x_{j_2})`,
      and :math:`F_{j_1} = f(x_{j_1})`, :math:`F_{j_2} = f(x_{j_2})`,

      .. math::
          F_{j_1} \ge F_{j_2} + \langle g_{j_2}, x_{j_1} - x_{j_2} \rangle - \frac{\tilde{\mu}}{2}\|x_{j_1} - x_{j_2}\|^2.

    **Parameters**

    - `mu_tilde` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): Weak convexity parameter corresponding to
      :math:`\tilde{\mu}` (must be :math:`> 0` and finite).

    **Raises**

    - `ValueError`: If `mu_tilde` is not valid.
    """
    def __init__(self, mu_tilde: Union[int, float]):
        mu_tilde, mu = _ensure_positive_mu_tilde(mu_tilde, "WeaklyConvex")
        super().__init__(mu=mu, L=INF)
        self.mu_tilde = mu_tilde

class Smooth(ParametrizedFunctionInterpolationCondition):
    r"""
    Function interpolation condition for smooth functions.

    Let :math:`(\calH,\langle\cdot,\cdot\rangle)` be a real Hilbert space and
    :math:`f: \calH \to \mathbb{R} \cup \{\pm\infty\}`.

    - :math:`L`-smooth with :math:`L \in \mathbb{R}_{++}`:
      The function :math:`f` is Fréchet differentiable and its gradient is
      :math:`L`-Lipschitz continuous, i.e.,

      .. math::
          \|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|
          \quad \text{for each } x,y \in \calH.

    - Interpolation inequality:

      For any :math:`x_{j_1}, x_{j_2} \in \calH` with
      :math:`g_{j_1} = \nabla f(x_{j_1})`, :math:`g_{j_2} = \nabla f(x_{j_2})`,
      and :math:`F_{j_1} = f(x_{j_1})`, :math:`F_{j_2} = f(x_{j_2})`,

      .. math::
          F_{j_1} \ge F_{j_2} + \langle g_{j_2}, x_{j_1} - x_{j_2} \rangle
          + \frac{1}{4L}\|g_{j_1} - g_{j_2} + L(x_{j_1} - x_{j_2})\|^2
          - \frac{L}{2}\|x_{j_1} - x_{j_2}\|^2.

    **Parameters**

    - `L` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): Smoothness parameter corresponding to
      :math:`L` (must be :math:`> 0` and finite).

    **Raises**

    - `ValueError`: If `L` is not valid.
    """
    def __init__(self, L: Union[int, float]):
        L = _ensure_positive_finite(L, "Parameter L", "For Smooth, L must be > 0 and finite.")
        super().__init__(mu=-L, L=L)

class SmoothConvex(ParametrizedFunctionInterpolationCondition):
    r"""
    Function interpolation condition for smooth and convex functions.

    Let :math:`(\calH,\langle\cdot,\cdot\rangle)` be a real Hilbert space and
    :math:`f: \calH \to \mathbb{R} \cup \{\pm\infty\}`.

    - :math:`L`-smooth with :math:`L \in \mathbb{R}_{++}`:
      The function :math:`f` is Fréchet differentiable and its gradient is
      :math:`L`-Lipschitz continuous, i.e.,

      .. math::
          \|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|
          \quad \text{for each } x,y \in \calH.

    - Convex:

      .. math::
          f((1-\lambda)x + \lambda y) \leq (1-\lambda) f(x) + \lambda f(y)
          \quad \text{for each } x,y \in \calH,\; \lambda \in [0,1].

    - Interpolation inequality:

      For any :math:`x_{j_1}, x_{j_2} \in \calH` with
      :math:`g_{j_1} = \nabla f(x_{j_1})`, :math:`g_{j_2} = \nabla f(x_{j_2})`,
      and :math:`F_{j_1} = f(x_{j_1})`, :math:`F_{j_2} = f(x_{j_2})`,

      .. math::
          F_{j_1} \ge F_{j_2} + \langle g_{j_2}, x_{j_1} - x_{j_2} \rangle
          + \frac{1}{2L}\|g_{j_1} - g_{j_2}\|^2.

    **Parameters**

    - `L` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): Smoothness parameter corresponding to
      :math:`L` (must be :math:`> 0` and finite).

    **Raises**

    - `ValueError`: If `L` is not valid.
    """
    def __init__(self, L: Union[int, float]):
        L = _ensure_positive_finite(L, "Parameter L", "For SmoothConvex, L must be > 0 and finite.")
        super().__init__(mu=0.0, L=L)

class SmoothStronglyConvex(ParametrizedFunctionInterpolationCondition):
    r"""
    Function interpolation condition for smooth and strongly convex functions.

    Let :math:`(\calH,\langle\cdot,\cdot\rangle)` be a real Hilbert space and
    :math:`f: \calH \to \mathbb{R} \cup \{\pm\infty\}`.

    - :math:`L`-smooth with :math:`L \in \mathbb{R}_{++}`:
      The function :math:`f` is Fréchet differentiable and its gradient is
      :math:`L`-Lipschitz continuous, i.e.,

      .. math::
          \|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|
          \quad \text{for each } x,y \in \calH.

    - :math:`\mu`-strongly convex with :math:`\mu \in \mathbb{R}_{++}`
      and :math:`\mu < L`:

      .. math::
          f - \frac{\mu}{2}\|\cdot\|^2 \quad \text{is convex}.

    - Interpolation inequality:

      For any :math:`x_{j_1}, x_{j_2} \in \calH` with
      :math:`g_{j_1} = \nabla f(x_{j_1})`, :math:`g_{j_2} = \nabla f(x_{j_2})`,
      and :math:`F_{j_1} = f(x_{j_1})`, :math:`F_{j_2} = f(x_{j_2})`,

      .. math::
          F_{j_1} \ge F_{j_2} + \langle g_{j_2}, x_{j_1} - x_{j_2} \rangle + \frac{\mu}{2}\|x_{j_1} - x_{j_2}\|^2
          + \frac{1}{2(L-\mu)}\|g_{j_1} - g_{j_2} - \mu(x_{j_1} - x_{j_2})\|^2.

    **Parameters**

    - `mu` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): Strong convexity parameter corresponding to
      :math:`\mu` (must be :math:`> 0` and finite).
    - `L` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): Smoothness parameter corresponding to :math:`L`
      (must be :math:`> 0` and finite) with :math:`\mu < L`.

    **Raises**

    - `ValueError`: If parameters are not valid.
    """
    def __init__(self, mu: Union[int, float], L: Union[int, float]):
        mu = _ensure_positive_finite(
            mu,
            "Parameter mu",
            "For SmoothStronglyConvex, mu must be > 0 and finite.",
        )
        L = _ensure_positive_finite(
            L,
            "Parameter L",
            "For SmoothStronglyConvex, L must be > 0 and finite.",
        )
        if mu >= L:
            raise ValueError("For SmoothStronglyConvex, mu must be less than L.")
        super().__init__(mu=mu, L=L)

class SmoothWeaklyConvex(ParametrizedFunctionInterpolationCondition):
    r"""
    Function interpolation condition for smooth and weakly convex functions.

    Let :math:`(\calH,\langle\cdot,\cdot\rangle)` be a real Hilbert space and
    :math:`f: \calH \to \mathbb{R} \cup \{\pm\infty\}`.

    - :math:`L`-smooth with :math:`L \in \mathbb{R}_{++}`:
      The function :math:`f` is Fréchet differentiable and its gradient is
      :math:`L`-Lipschitz continuous, i.e.,

      .. math::
          \|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|
          \quad \text{for each } x,y \in \calH.

    - :math:`\tilde{\mu}`-weakly convex with :math:`\tilde{\mu} \in \mathbb{R}_{++}`:

      .. math::
          f + \frac{\tilde{\mu}}{2}\|\cdot\|^2 \quad \text{is convex}.

    - Interpolation inequality:

      For any :math:`x_{j_1}, x_{j_2} \in \calH` with
      :math:`g_{j_1} = \nabla f(x_{j_1})`, :math:`g_{j_2} = \nabla f(x_{j_2})`,
      and :math:`F_{j_1} = f(x_{j_1})`, :math:`F_{j_2} = f(x_{j_2})`,

      .. math::
          F_{j_1} \ge F_{j_2} + \langle g_{j_2}, x_{j_1} - x_{j_2} \rangle - \frac{\tilde{\mu}}{2}\|x_{j_1} - x_{j_2}\|^2
          + \frac{1}{2(L+\tilde{\mu})}\|g_{j_1} - g_{j_2} + \tilde{\mu}(x_{j_1} - x_{j_2})\|^2.

    **Parameters**

    - `mu_tilde` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): Weak convexity parameter corresponding to
      :math:`\tilde{\mu}` (must be :math:`> 0` and finite).
    - `L` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): Smoothness parameter corresponding to
      :math:`L` (must be :math:`> 0` and finite).

    **Raises**

    - `ValueError`: If parameters are not valid.
    """
    def __init__(self, mu_tilde: Union[int, float], L: Union[int, float]):
        mu_tilde, mu = _ensure_positive_mu_tilde(mu_tilde, "SmoothWeaklyConvex")
        L = _ensure_positive_finite(
            L,
            "Parameter L",
            "For SmoothWeaklyConvex, L must be > 0 and finite.",
        )
        super().__init__(mu=mu, L=L)
        self.mu_tilde = mu_tilde

class IndicatorFunctionOfClosedConvexSet(FunctionInterpolationCondition):
    r"""
    Function interpolation condition for indicator functions of nonempty, closed, and convex sets.

    Let :math:`(\calH,\langle\cdot,\cdot\rangle)` be a real Hilbert space and
    :math:`C \subseteq \calH` be nonempty, closed, and convex.

    - Indicator function:

      Functions :math:`\delta_C: \calH \to \mathbb{R} \cup \{\pm\infty\}` that can be written as

    .. math::
        \delta_C(x) =
        \begin{cases}
            0, & x \in C, \\
            +\infty, & x \notin C.
        \end{cases}

    - Normal cone:

      .. math::
          N_C(x) = \{g \in \calH \mid \langle g, y - x \rangle \le 0
          \ \text{for all } y \in C\}.

      For :math:`x \in C`, the subdifferential of the indicator function satisfies

      .. math::
          \partial \delta_C(x) = N_C(x).

    - Interpolation inequalities used:

      For any :math:`x_{j_1}, x_{j_2} \in C` with
      :math:`g_{j_1} \in N_C(x_{j_1})`, :math:`g_{j_2} \in N_C(x_{j_2})`,
      and :math:`F_{j_1} = \delta_C(x_{j_1})`, :math:`F_{j_2} = \delta_C(x_{j_2})`,

      .. math::
          \langle g_{j_2}, x_{j_1} - x_{j_2} \rangle \le 0 \quad \text{for } j_1 \ne j_2,

    and

      .. math::
          F_{j_1} = 0 \quad \text{for each interpolation point } x_{j_1} \in C.

    This condition has no parameters.

    """
    def get_data(self) -> List[Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]]:
        r"""
        Return interpolation data for the indicator function.

        The tuple format and notation follow the class-level reference in
        :class:`~autolyap.problemclass.IndicatorFunctionOfClosedConvexSet`.

        **Returns**

        - (:class:`~typing.List`\[:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`bool`, :class:`~autolyap.problemclass.problemclass.InterpolationIndices`\]\]): A list containing two
          tuples with the interpolation data.

        """
        interp_idx_ineq = InterpolationIndices("j1!=j2")
        matrix_ineq = 0.5 * np.array([
            [0, 0, 0, 1],
            [0, 0, 0, -1],
            [0, 0, 0, 0],
            [1, -1, 0, 0]
        ])
        vector_ineq = np.array([0, 0])
        interp_idx_eq = InterpolationIndices("j1")
        matrix_eq = np.array([[0, 0], [0, 0]])
        vector_eq = np.array([1])
        return [
            (matrix_ineq, vector_ineq, False, interp_idx_ineq),
            (matrix_eq, vector_eq, True, interp_idx_eq)
        ]

class SupportFunctionOfClosedConvexSet(FunctionInterpolationCondition):
    r"""
    Function interpolation condition for support functions of nonempty, closed, and convex sets.

    Let :math:`(\calH,\langle\cdot,\cdot\rangle)` be a real Hilbert space and
    :math:`C \subseteq \calH` be nonempty, closed, and convex.

    - Support function:

      Functions :math:`\sigma_C: \calH \to \mathbb{R} \cup \{\pm\infty\}` that can be written as

    .. math::
        \sigma_C(x) = \sup_{y \in C} \langle x, y \rangle.

    - Interpolation inequalities used:

      For any :math:`x_{j_1}, x_{j_2} \in \calH` with
      :math:`g_{j_1} \in \partial \sigma_C(x_{j_1})`, :math:`g_{j_2} \in \partial \sigma_C(x_{j_2})`,
      and :math:`F_{j_1} = \sigma_C(x_{j_1})`, :math:`F_{j_2} = \sigma_C(x_{j_2})`,

      .. math::
          F_{j_1} = \langle x_{j_1}, g_{j_1} \rangle,

    and

      .. math::
          \langle x_{j_2}, g_{j_1} - g_{j_2} \rangle \le 0 \quad \text{for } j_1 \ne j_2

    This condition has no parameters.

    """
    def get_data(self) -> List[Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]]:
        r"""
        Return interpolation data for the support function.

        The tuple format and notation follow the class-level reference in
        :class:`~autolyap.problemclass.SupportFunctionOfClosedConvexSet`.

        **Returns**

        - (:class:`~typing.List`\[:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`bool`, :class:`~autolyap.problemclass.problemclass.InterpolationIndices`\]\]): A list containing two
          tuples with the interpolation data.

        """
        interp_idx_ineq = InterpolationIndices("j1!=j2")
        matrix_ineq = 0.5 * np.array([
            [0, 0, 0, 0],
            [0, 0, 1, -1],
            [0, 1, 0, 0],
            [0, -1, 0, 0]
        ])
        vector_ineq = np.array([0, 0])
        interp_idx_eq = InterpolationIndices("j1")
        matrix_eq = 0.5 * np.array([[0, 1], [1, 0]])
        vector_eq = np.array([-1])
        return [
            (matrix_ineq, vector_ineq, False, interp_idx_ineq),
            (matrix_eq, vector_eq, True, interp_idx_eq)
        ]

class GradientDominated(FunctionInterpolationCondition):
    r"""
    Function interpolation condition for gradient-dominated functions.

    Let :math:`(\calH,\langle\cdot,\cdot\rangle)` be a real Hilbert space and
    :math:`f: \calH \to \mathbb{R} \cup \{\pm\infty\}`.

    - :math:`\mu_{\textup{gd}}`-gradient dominated with :math:`\mu_{\textup{gd}} \in \mathbb{R}_{++}`, i.e.,
      the function :math:`f` is Fréchet differentiable and

      .. math::
          f(x) - \inf_{y \in \calH} f(y) \leq \frac{1}{2\mu_{\textup{gd}}}\|\nabla f(x)\|^2
          \quad \text{for each } x \in \calH.

    - Interpolation inequalities used:

      Let :math:`x_\star` be a minimizer with :math:`F_\star = f(x_\star)`.
      For any :math:`x_{j_1} \in \calH` with :math:`g_{j_1} = \nabla f(x_{j_1})` and
      :math:`F_{j_1} = f(x_{j_1})`,

      .. math::
          F_{j_1} - F_\star \le \frac{1}{2\mu_{\textup{gd}}}\|g_{j_1}\|^2
          \quad \text{and} \quad
          F_{j_1} \ge F_\star.

    Note: This gradient-dominated condition is only sufficient for the analysis in AutoLyap,
    and there is no guarantee of tightness of the resulting analysis.

    Note: When used inside :class:`~autolyap.problemclass.InclusionProblem`, AutoLyap enforces that the total number
    of components is exactly one (i.e., :math:`m = 1`) if any component uses this condition.

    **Parameters**

    - `mu_gd` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The gradient-dominated parameter corresponding
      to :math:`\mu_{\textup{gd}}` (must be :math:`> 0` and finite).

    **Raises**

    - `ValueError`: If `mu_gd` is not a number, :math:`\le 0`, or infinite.
    """
    def __init__(self, mu_gd: Union[int, float]):
        mu_gd = _ensure_positive_finite(
            mu_gd,
            "Gradient-dominated parameter",
            "Gradient-dominated parameter (mu_gd) must be greater than 0 and finite.",
        )
        self.mu_gd = mu_gd

    def get_data(self) -> List[Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]]:
        r"""
        Return interpolation data for gradient-dominated functions.

        The tuple format and notation follow the class-level reference in
        :class:`~autolyap.problemclass.GradientDominated`.

        **Returns**

        - (:class:`~typing.List`\[:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`bool`, :class:`~autolyap.problemclass.problemclass.InterpolationIndices`\]\]): A list containing two
          tuples with the interpolation data.

        """
        a1 = np.array([-1, 1])
        M1 = np.zeros((4, 4))
        
        a2 = np.array([1, -1])
        M2 = np.zeros((4, 4))
        M2[2, 2] = -1 / (2 * self.mu_gd)
        
        interp_idx = InterpolationIndices("j1!=star")
        eq_flag = False
        return [
            (M1, a1, eq_flag, interp_idx),
            (M2, a2, eq_flag, interp_idx)
        ]
