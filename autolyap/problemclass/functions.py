"""Concrete function interpolation-condition classes."""

import numpy as np
from typing import List, Union, Tuple

from autolyap.problemclass.base import _FunctionInterpolationCondition
from autolyap.problemclass.indices import _InterpolationIndices
from autolyap.utils.validation import ensure_real_number

INF = float("inf")


def _ensure_positive_finite(value: Union[int, float], parameter_name: str, error_message: str) -> float:
    r"""Return `value` as float after enforcing strict positivity and finiteness."""
    numeric_value = ensure_real_number(value, parameter_name)
    if not np.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(error_message)
    return numeric_value


def _ensure_positive_mu_tilde(mu_tilde: Union[int, float], context_name: str) -> Tuple[float, float]:
    r"""
    Validate :math:`\tilde{\mu} > 0` and return both :math:`(\tilde{\mu}, -\tilde{\mu})`.

    Many weakly-convex templates are written with :math:`\mu=-\tilde{\mu}`.
    This helper keeps that conversion explicit and centralized.
    """
    validated = _ensure_positive_finite(
        mu_tilde,
        "Parameter mu_tilde",
        f"For {context_name}, mu_tilde must be > 0 and finite.",
    )
    return validated, -validated


class _ParametrizedFunctionInterpolationCondition(_FunctionInterpolationCondition):
    r"""
    Base class for function interpolation conditions parameterized by :math:`\mu` and :math:`L`.

    Class-level reference
    =====================

    This class-level docstring centralizes notation for the shared
    :math:`(\mu, L)` interpolation family used by its concrete subclasses.

    Provides a helper to compute interpolation data from :math:`\mu` and :math:`L`.
    This base class supports both smooth and nonsmooth conditions by setting :math:`L` appropriately.

    This template applies to :math:`f: \calH \to \mathbb{R} \cup \{\pm\infty\}` with
    parameters satisfying :math:`-\infty < \mu < L \le +\infty` and :math:`L > 0`
    (nonsmooth cases are encoded by :math:`L = +\infty`).

    - Interpolation inequality:

      For any :math:`x_{r_1}, x_{r_2} \in \calH` with
      :math:`g_{r_1} \in \partial f(x_{r_1})`, :math:`g_{r_2} \in \partial f(x_{r_2})`,
      and :math:`F_{r_1} = f(x_{r_1})`, :math:`F_{r_2} = f(x_{r_2})`, the condition enforced is

      .. math::
          F_{r_1} \ge F_{r_2} + \langle g_{r_2}, x_{r_1} - x_{r_2} \rangle + \frac{\mu}{2}\|x_{r_1} - x_{r_2}\|^2
          + \frac{1}{2(L-\mu)}\|g_{r_1} - g_{r_2} - \mu(x_{r_1} - x_{r_2})\|^2,

      where :math:`\frac{1}{2(L-\mu)}` is interpreted as :math:`0` when :math:`L = +\infty`.

    - Matrix/vector form used in :doc:`Interpolation conditions </theory/interpolation_conditions>`:

      With :math:`F = (F_{r_1}, F_{r_2})` and
      :math:`z = (x_{r_1}, x_{r_2}, g_{r_1}, g_{r_2})`, the interpolation inequality is encoded as

      .. math::
          a^\top F + \mathcal{Q}\p{M, z} \le 0, \qquad a = (-1, 1),

      with

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

      The returned interpolation indices are ``r1!=r2``.
      The `eq` flag returned by :meth:`get_data` is ``False``.

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
    def __init__(self, mu: Union[int, float], L: Union[int, float]) -> None:
        mu, L = self._validate_mu_L(mu, L)
        self.mu = mu
        self.L = L
        self._interpolation_data = self._compute_interpolation_data(mu, L)

    @staticmethod
    def _validate_mu_L(mu: Union[int, float], L: Union[int, float]) -> Tuple[float, float]:
        r"""
        Validate interpolation parameters and return normalized ``(mu, L)``.

        Enforces the admissible region ``-inf < mu < L <= +inf`` with the
        additional requirement that finite ``L`` values are strictly positive.
        """
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
    def _compute_interpolation_data(mu: float, L: float) -> Tuple[np.ndarray, np.ndarray, bool, _InterpolationIndices]:
        r"""
        Compute the interpolation data based on mu and L.

        The tuple format and notation follow the class-level reference in
        :class:`~autolyap.problemclass.functions._ParametrizedFunctionInterpolationCondition`.

        When L is infinite, the condition is nonsmooth and a simpler interpolation matrix is used.
        When L is finite, the interpolation data follows the formula for smooth functions.

        **Parameters**

        - `mu` (:class:`float`): The convexity parameter corresponding to :math:`\mu`.
        - `L` (:class:`float`): The smoothness parameter corresponding to :math:`L`.

        **Returns**

        - (:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`bool`, :class:`~autolyap.problemclass.indices._InterpolationIndices`\]): A tuple containing the
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
        interp_idx = _InterpolationIndices("r1!=r2")
        return matrix, vector, eq, interp_idx

    def get_data(self) -> List[Tuple[np.ndarray, np.ndarray, bool, _InterpolationIndices]]:
        r"""
        Return interpolation data for the function condition.

        The tuple format and notation follow the class-level reference in
        :class:`~autolyap.problemclass.functions._ParametrizedFunctionInterpolationCondition`.

        **Returns**

        - (:class:`~typing.List`\[:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`bool`, :class:`~autolyap.problemclass.indices._InterpolationIndices`\]\]): A list containing one
          tuple with the matrix, vector, eq flag, and interpolation indices.

        """
        return [self._interpolation_data]

class Convex(_ParametrizedFunctionInterpolationCondition):
    r"""
    Function interpolation condition for proper, lower semicontinuous, and convex functions.

    Let :math:`f: \calH \to \mathbb{R} \cup \{\pm\infty\}` be proper, lower semicontinuous, and convex.

    - Interpolation inequality:

      For any :math:`x_{r_1}, x_{r_2} \in \calH` with
      :math:`g_{r_1} \in \partial f(x_{r_1})`, :math:`g_{r_2} \in \partial f(x_{r_2})`,
      and :math:`F_{r_1} = f(x_{r_1})`, :math:`F_{r_2} = f(x_{r_2})`,

      .. math::
          F_{r_1} \ge F_{r_2} + \langle g_{r_2}, x_{r_1} - x_{r_2} \rangle.

    - Matrix/vector form used in :doc:`Interpolation conditions </theory/interpolation_conditions>`:

      With :math:`z = (x_{r_1}, x_{r_2}, g_{r_1}, g_{r_2})` and
      :math:`F = (F_{r_1}, F_{r_2})`, the same inequality is encoded as

      .. math::
          a^\top F + \mathcal{Q}\p{M, z} \le 0,

      with

      .. math::
          a = (-1, 1), \qquad
          M = \frac{1}{2}
          \begin{bmatrix}
              0 & 0 & 0 & 1 \\
              0 & 0 & 0 & -1 \\
              0 & 0 & 0 & 0 \\
              1 & -1 & 0 & 0
          \end{bmatrix}.

    """
    def __init__(self) -> None:
        super().__init__(mu=0.0, L=INF)

class StronglyConvex(_ParametrizedFunctionInterpolationCondition):
    r"""
    Function interpolation condition for proper, lower semicontinuous, and strongly convex functions.

    Let :math:`\mu \in \mathbb{R}_{++}` and
    :math:`f: \calH \to \mathbb{R} \cup \{\pm\infty\}` be proper, lower semicontinuous,
    and :math:`\mu`-strongly convex.

    - Interpolation inequality:

      For any :math:`x_{r_1}, x_{r_2} \in \calH` with
      :math:`g_{r_1} \in \partial f(x_{r_1})`, :math:`g_{r_2} \in \partial f(x_{r_2})`,
      and :math:`F_{r_1} = f(x_{r_1})`, :math:`F_{r_2} = f(x_{r_2})`,

      .. math::
          F_{r_1} \ge F_{r_2} + \langle g_{r_2}, x_{r_1} - x_{r_2} \rangle + \frac{\mu}{2}\|x_{r_1} - x_{r_2}\|^2.

    - Matrix/vector form used in :doc:`Interpolation conditions </theory/interpolation_conditions>`:

      With :math:`z = (x_{r_1}, x_{r_2}, g_{r_1}, g_{r_2})` and
      :math:`F = (F_{r_1}, F_{r_2})`, the same inequality is encoded as

      .. math::
          a^\top F + \mathcal{Q}\p{M, z} \le 0,

      with

      .. math::
          a = (-1, 1), \qquad
          M = \frac{1}{2}
          \begin{bmatrix}
              \mu & -\mu & 0 & 1 \\
              -\mu & \mu & 0 & -1 \\
              0 & 0 & 0 & 0 \\
              1 & -1 & 0 & 0
          \end{bmatrix}.

    **Parameters**

    - `mu` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): Strong convexity parameter corresponding to
      :math:`\mu` (must be :math:`> 0` and finite).

    **Raises**

    - `ValueError`: If `mu` is not valid.
    """
    def __init__(self, mu: Union[int, float]) -> None:
        mu = _ensure_positive_finite(mu, "Parameter mu", "For StronglyConvex, mu must be > 0 and finite.")
        super().__init__(mu=mu, L=INF)

class WeaklyConvex(_ParametrizedFunctionInterpolationCondition):
    r"""
    Function interpolation condition for proper, lower semicontinuous, and weakly convex functions.

    Let :math:`\tilde{\mu} \in \mathbb{R}_{++}` and
    :math:`f: \calH \to \mathbb{R} \cup \{\pm\infty\}` be proper, lower semicontinuous,
    and :math:`\tilde{\mu}`-weakly convex.

    - Interpolation inequality:

      For any :math:`x_{r_1}, x_{r_2} \in \calH` with
      :math:`g_{r_1} \in \partial f(x_{r_1})`, :math:`g_{r_2} \in \partial f(x_{r_2})`,
      and :math:`F_{r_1} = f(x_{r_1})`, :math:`F_{r_2} = f(x_{r_2})`,

      .. math::
          F_{r_1} \ge F_{r_2} + \langle g_{r_2}, x_{r_1} - x_{r_2} \rangle - \frac{\tilde{\mu}}{2}\|x_{r_1} - x_{r_2}\|^2.

    - Matrix/vector form used in :doc:`Interpolation conditions </theory/interpolation_conditions>`:

      With :math:`z = (x_{r_1}, x_{r_2}, g_{r_1}, g_{r_2})` and
      :math:`F = (F_{r_1}, F_{r_2})`, the same inequality is encoded as

      .. math::
          a^\top F + \mathcal{Q}\p{M, z} \le 0,

      with

      .. math::
          a = (-1, 1), \qquad
          M = \frac{1}{2}
          \begin{bmatrix}
              -\tilde{\mu} & \tilde{\mu} & 0 & 1 \\
              \tilde{\mu} & -\tilde{\mu} & 0 & -1 \\
              0 & 0 & 0 & 0 \\
              1 & -1 & 0 & 0
          \end{bmatrix}.

    **Parameters**

    - `mu_tilde` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): Weak convexity parameter corresponding to
      :math:`\tilde{\mu}` (must be :math:`> 0` and finite).

    **Raises**

    - `ValueError`: If `mu_tilde` is not valid.
    """
    def __init__(self, mu_tilde: Union[int, float]) -> None:
        mu_tilde, mu = _ensure_positive_mu_tilde(mu_tilde, "WeaklyConvex")
        super().__init__(mu=mu, L=INF)
        self.mu_tilde = mu_tilde

class Smooth(_ParametrizedFunctionInterpolationCondition):
    r"""
    Function interpolation condition for smooth functions.

    Let :math:`L \in \mathbb{R}_{++}` and
    :math:`f: \calH \to \mathbb{R} \cup \{\pm\infty\}` be Fréchet differentiable
    with :math:`L`-Lipschitz gradient.

    - Interpolation inequality:

      For any :math:`x_{r_1}, x_{r_2} \in \calH` with
      :math:`g_{r_1} = \nabla f(x_{r_1})`, :math:`g_{r_2} = \nabla f(x_{r_2})`,
      and :math:`F_{r_1} = f(x_{r_1})`, :math:`F_{r_2} = f(x_{r_2})`,

      .. math::
          F_{r_1} \ge F_{r_2} + \langle g_{r_2}, x_{r_1} - x_{r_2} \rangle
          + \frac{1}{4L}\|g_{r_1} - g_{r_2} + L(x_{r_1} - x_{r_2})\|^2
          - \frac{L}{2}\|x_{r_1} - x_{r_2}\|^2.

    - Matrix/vector form used in :doc:`Interpolation conditions </theory/interpolation_conditions>`:

      With :math:`z = (x_{r_1}, x_{r_2}, g_{r_1}, g_{r_2})` and
      :math:`F = (F_{r_1}, F_{r_2})`, the same inequality is encoded as

      .. math::
          a^\top F + \mathcal{Q}\p{M, z} \le 0,

      with

      .. math::
          a = (-1, 1), \qquad
          M = \frac{1}{4L}
          \begin{bmatrix}
              -L^2 & L^2 & L & L \\
              L^2 & -L^2 & -L & -L \\
              L & -L & 1 & -1 \\
              L & -L & -1 & 1
          \end{bmatrix}.

    **Parameters**

    - `L` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): Smoothness parameter corresponding to
      :math:`L` (must be :math:`> 0` and finite).

    **Raises**

    - `ValueError`: If `L` is not valid.
    """
    def __init__(self, L: Union[int, float]) -> None:
        L = _ensure_positive_finite(L, "Parameter L", "For Smooth, L must be > 0 and finite.")
        super().__init__(mu=-L, L=L)

class SmoothConvex(_ParametrizedFunctionInterpolationCondition):
    r"""
    Function interpolation condition for smooth and convex functions.

    Let :math:`L \in \mathbb{R}_{++}` and
    :math:`f: \calH \to \mathbb{R} \cup \{\pm\infty\}` be convex, Fréchet differentiable,
    with :math:`L`-Lipschitz gradient.

    - Interpolation inequality:

      For any :math:`x_{r_1}, x_{r_2} \in \calH` with
      :math:`g_{r_1} = \nabla f(x_{r_1})`, :math:`g_{r_2} = \nabla f(x_{r_2})`,
      and :math:`F_{r_1} = f(x_{r_1})`, :math:`F_{r_2} = f(x_{r_2})`,

      .. math::
          F_{r_1} \ge F_{r_2} + \langle g_{r_2}, x_{r_1} - x_{r_2} \rangle
          + \frac{1}{2L}\|g_{r_1} - g_{r_2}\|^2.

    - Matrix/vector form used in :doc:`Interpolation conditions </theory/interpolation_conditions>`:

      With :math:`z = (x_{r_1}, x_{r_2}, g_{r_1}, g_{r_2})` and
      :math:`F = (F_{r_1}, F_{r_2})`, the same inequality is encoded as

      .. math::
          a^\top F + \mathcal{Q}\p{M, z} \le 0,

      with

      .. math::
          a = (-1, 1), \qquad
          M = \frac{1}{2L}
          \begin{bmatrix}
              0 & 0 & 0 & L \\
              0 & 0 & 0 & -L \\
              0 & 0 & 1 & -1 \\
              L & -L & -1 & 1
          \end{bmatrix}.

    **Parameters**

    - `L` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): Smoothness parameter corresponding to
      :math:`L` (must be :math:`> 0` and finite).

    **Raises**

    - `ValueError`: If `L` is not valid.
    """
    def __init__(self, L: Union[int, float]) -> None:
        L = _ensure_positive_finite(L, "Parameter L", "For SmoothConvex, L must be > 0 and finite.")
        super().__init__(mu=0.0, L=L)

class SmoothStronglyConvex(_ParametrizedFunctionInterpolationCondition):
    r"""
    Function interpolation condition for smooth and strongly convex functions.

    Let :math:`0 < \mu < L` and
    :math:`f: \calH \to \mathbb{R} \cup \{\pm\infty\}` be :math:`\mu`-strongly convex,
    Fréchet differentiable, with :math:`L`-Lipschitz gradient.

    - Interpolation inequality:

      For any :math:`x_{r_1}, x_{r_2} \in \calH` with
      :math:`g_{r_1} = \nabla f(x_{r_1})`, :math:`g_{r_2} = \nabla f(x_{r_2})`,
      and :math:`F_{r_1} = f(x_{r_1})`, :math:`F_{r_2} = f(x_{r_2})`,

      .. math::
          F_{r_1} \ge F_{r_2} + \langle g_{r_2}, x_{r_1} - x_{r_2} \rangle + \frac{\mu}{2}\|x_{r_1} - x_{r_2}\|^2
          + \frac{1}{2(L-\mu)}\|g_{r_1} - g_{r_2} - \mu(x_{r_1} - x_{r_2})\|^2.

    - Matrix/vector form used in :doc:`Interpolation conditions </theory/interpolation_conditions>`:

      With :math:`z = (x_{r_1}, x_{r_2}, g_{r_1}, g_{r_2})` and
      :math:`F = (F_{r_1}, F_{r_2})`, the same inequality is encoded as

      .. math::
          a^\top F + \mathcal{Q}\p{M, z} \le 0,

      with

      .. math::
          a = (-1, 1), \qquad
          M = \frac{1}{2(L-\mu)}
          \begin{bmatrix}
              L\mu & -L\mu & -\mu & L \\
              -L\mu & L\mu & \mu & -L \\
              -\mu & \mu & 1 & -1 \\
              L & -L & -1 & 1
          \end{bmatrix}.

    **Parameters**

    - `mu` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): Strong convexity parameter corresponding to
      :math:`\mu` (must be :math:`> 0` and finite).
    - `L` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): Smoothness parameter corresponding to :math:`L`
      (must be :math:`> 0` and finite) with :math:`\mu < L`.

    **Raises**

    - `ValueError`: If parameters are not valid.
    """
    def __init__(self, mu: Union[int, float], L: Union[int, float]) -> None:
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

class SmoothWeaklyConvex(_ParametrizedFunctionInterpolationCondition):
    r"""
    Function interpolation condition for smooth and weakly convex functions.

    Let :math:`\tilde{\mu} \in \mathbb{R}_{++}` and :math:`L \in \mathbb{R}_{++}`, with
    :math:`f: \calH \to \mathbb{R} \cup \{\pm\infty\}` be :math:`\tilde{\mu}`-weakly convex,
    Fréchet differentiable, with :math:`L`-Lipschitz gradient.

    - Interpolation inequality:

      For any :math:`x_{r_1}, x_{r_2} \in \calH` with
      :math:`g_{r_1} = \nabla f(x_{r_1})`, :math:`g_{r_2} = \nabla f(x_{r_2})`,
      and :math:`F_{r_1} = f(x_{r_1})`, :math:`F_{r_2} = f(x_{r_2})`,

      .. math::
          F_{r_1} \ge F_{r_2} + \langle g_{r_2}, x_{r_1} - x_{r_2} \rangle - \frac{\tilde{\mu}}{2}\|x_{r_1} - x_{r_2}\|^2
          + \frac{1}{2(L+\tilde{\mu})}\|g_{r_1} - g_{r_2} + \tilde{\mu}(x_{r_1} - x_{r_2})\|^2.

    - Matrix/vector form used in :doc:`Interpolation conditions </theory/interpolation_conditions>`:

      With :math:`z = (x_{r_1}, x_{r_2}, g_{r_1}, g_{r_2})` and
      :math:`F = (F_{r_1}, F_{r_2})`, the same inequality is encoded as

      .. math::
          a^\top F + \mathcal{Q}\p{M, z} \le 0,

      with

      .. math::
          a = (-1, 1), \qquad
          M = \frac{1}{2(L+\tilde{\mu})}
          \begin{bmatrix}
              -L\tilde{\mu} & L\tilde{\mu} & \tilde{\mu} & L \\
              L\tilde{\mu} & -L\tilde{\mu} & -\tilde{\mu} & -L \\
              \tilde{\mu} & -\tilde{\mu} & 1 & -1 \\
              L & -L & -1 & 1
          \end{bmatrix}.

    **Parameters**

    - `mu_tilde` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): Weak convexity parameter corresponding to
      :math:`\tilde{\mu}` (must be :math:`> 0` and finite).
    - `L` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): Smoothness parameter corresponding to
      :math:`L` (must be :math:`> 0` and finite).

    **Raises**

    - `ValueError`: If parameters are not valid.
    """
    def __init__(self, mu_tilde: Union[int, float], L: Union[int, float]) -> None:
        mu_tilde, mu = _ensure_positive_mu_tilde(mu_tilde, "SmoothWeaklyConvex")
        L = _ensure_positive_finite(
            L,
            "Parameter L",
            "For SmoothWeaklyConvex, L must be > 0 and finite.",
        )
        super().__init__(mu=mu, L=L)
        self.mu_tilde = mu_tilde

class IndicatorFunctionOfClosedConvexSet(_FunctionInterpolationCondition):
    r"""
    Function interpolation condition for indicator functions of nonempty, closed, and convex sets.

    Let :math:`C \subseteq \calH` be nonempty, closed, and convex, and define
    its indicator function :math:`\delta_C : \calH \to \reals \cup \{\pm\infty\}`
    by

    .. math::
        \delta_C(x) =
        \begin{cases}
            0, & \text{if } x \in C, \\
            +\infty, & \text{if } x \notin C.
        \end{cases}

    Let :math:`N_C:\calH\rightrightarrows\calH` denote the normal cone of :math:`C`, defined by

    .. math::
        N_C(x) =
        \begin{cases}
            \{g \in \calH : \langle g, z - x \rangle \le 0,\ \forall z \in C\},
            & \text{if } x \in C, \\
            \emptyset, & \text{if } x \notin C,
        \end{cases}

    which coincides with the subdifferential of the indicator:

    .. math::
        \partial \delta_C(x) = N_C(x), \qquad \forall x \in \calH.

    - Interpolation inequalities used:

      For any :math:`x_{r_1}, x_{r_2} \in C` with
      :math:`g_{r_1} \in N_C(x_{r_1})`, :math:`g_{r_2} \in N_C(x_{r_2})`,
      and :math:`F_{r_1} = \delta_C(x_{r_1})`, :math:`F_{r_2} = \delta_C(x_{r_2})`,

      .. math::
          \langle g_{r_2}, x_{r_1} - x_{r_2} \rangle \le 0 \quad \text{for } r_1 \ne r_2,

      and

      .. math::
          F_{r_1} = 0.

    - Matrix/vector form used in :doc:`Interpolation conditions </theory/interpolation_conditions>`:

      With :math:`F = (F_{r_1}, F_{r_2})` and
      :math:`z = (x_{r_1}, x_{r_2}, g_{r_1}, g_{r_2})`, the inequality
      constraint has

      .. math::
          a_1 = (0, 0), \qquad
          M_1 = \frac{1}{2}
          \begin{bmatrix}
              0 & 0 & 0 & 1 \\
              0 & 0 & 0 & -1 \\
              0 & 0 & 0 & 0 \\
              1 & -1 & 0 & 0
          \end{bmatrix}.

      For the equality :math:`F_{r_1}=0`, with
      :math:`\hat{z}=(x_{r_1}, g_{r_1})`, the coefficients are

      .. math::
          a_2 = (1), \qquad
          M_2 =
          \begin{bmatrix}
              0 & 0 \\
              0 & 0
          \end{bmatrix}.

    This condition has no parameters.

    """
    def get_data(self) -> List[Tuple[np.ndarray, np.ndarray, bool, _InterpolationIndices]]:
        r"""
        Return interpolation data for the indicator function.

        The tuple format and notation follow the class-level reference in
        :class:`~autolyap.problemclass.IndicatorFunctionOfClosedConvexSet`.

        **Returns**

        - (:class:`~typing.List`\[:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`bool`, :class:`~autolyap.problemclass.indices._InterpolationIndices`\]\]): A list containing two
          tuples with the interpolation data.

        """
        interp_idx_ineq = _InterpolationIndices("r1!=r2")
        matrix_ineq = 0.5 * np.array([
            [0, 0, 0, 1],
            [0, 0, 0, -1],
            [0, 0, 0, 0],
            [1, -1, 0, 0]
        ])
        vector_ineq = np.array([0, 0])
        interp_idx_eq = _InterpolationIndices("r1")
        matrix_eq = np.array([[0, 0], [0, 0]])
        vector_eq = np.array([1])
        return [
            (matrix_ineq, vector_ineq, False, interp_idx_ineq),
            (matrix_eq, vector_eq, True, interp_idx_eq)
        ]

class SupportFunctionOfClosedConvexSet(_FunctionInterpolationCondition):
    r"""
    Function interpolation condition for support functions of nonempty, closed, and convex sets.

    Let :math:`C \subseteq \calH` be nonempty, closed, and convex, and define
    its support function :math:`\sigma_C : \calH \to \reals \cup \{\pm\infty\}`
    by

    .. math::
        \sigma_C(x) = \sup_{c \in C} \langle x, c \rangle.

    - Interpolation inequalities used:

      For any :math:`x_{r_1}, x_{r_2} \in \calH` with
      :math:`g_{r_1} \in \partial \sigma_C(x_{r_1})`, :math:`g_{r_2} \in \partial \sigma_C(x_{r_2})`,
      and :math:`F_{r_1} = \sigma_C(x_{r_1})`, :math:`F_{r_2} = \sigma_C(x_{r_2})`,

      .. math::
        F_{r_1} = \langle x_{r_1}, g_{r_1} \rangle,

      and

      .. math::
        \langle x_{r_2}, g_{r_1} - g_{r_2} \rangle \le 0.

    - Matrix/vector form used in :doc:`Interpolation conditions </theory/interpolation_conditions>`:

      With :math:`F = (F_{r_1}, F_{r_2})` and
      :math:`z = (x_{r_1}, x_{r_2}, g_{r_1}, g_{r_2})`, the inequality
      constraint has

      .. math::
          a_1 = (0, 0), \qquad
          M_1 = \frac{1}{2}
          \begin{bmatrix}
              0 & 0 & 0 & 0 \\
              0 & 0 & 1 & -1 \\
              0 & 1 & 0 & 0 \\
              0 & -1 & 0 & 0
          \end{bmatrix}.

      For the equality :math:`F_{r_1}=\langle x_{r_1},g_{r_1}\rangle`,
      with :math:`\hat{z}=(x_{r_1}, g_{r_1})`, the coefficients are

      .. math::
          a_2 = (-1), \qquad
          M_2 = \frac{1}{2}
          \begin{bmatrix}
              0 & 1 \\
              1 & 0
          \end{bmatrix}.

    This condition has no parameters.

    """
    def get_data(self) -> List[Tuple[np.ndarray, np.ndarray, bool, _InterpolationIndices]]:
        r"""
        Return interpolation data for the support function.

        The tuple format and notation follow the class-level reference in
        :class:`~autolyap.problemclass.SupportFunctionOfClosedConvexSet`.

        **Returns**

        - (:class:`~typing.List`\[:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`bool`, :class:`~autolyap.problemclass.indices._InterpolationIndices`\]\]): A list containing two
          tuples with the interpolation data.

        """
        interp_idx_ineq = _InterpolationIndices("r1!=r2")
        matrix_ineq = 0.5 * np.array([
            [0, 0, 0, 0],
            [0, 0, 1, -1],
            [0, 1, 0, 0],
            [0, -1, 0, 0]
        ])
        vector_ineq = np.array([0, 0])
        interp_idx_eq = _InterpolationIndices("r1")
        matrix_eq = 0.5 * np.array([[0, 1], [1, 0]])
        vector_eq = np.array([-1])
        return [
            (matrix_ineq, vector_ineq, False, interp_idx_ineq),
            (matrix_eq, vector_eq, True, interp_idx_eq)
        ]

class GradientDominated(_FunctionInterpolationCondition):
    r"""
    Function interpolation condition for gradient-dominated functions.

    Let :math:`\mu_{\textup{gd}} \in \mathbb{R}_{++}` and
    :math:`f: \calH \to \mathbb{R} \cup \{\pm\infty\}` be Fréchet differentiable and
    :math:`\mu_{\textup{gd}}`-gradient dominated.

    - Interpolation inequalities used:

      Let :math:`x_\star` be a minimizer with :math:`F_\star = f(x_\star)`.
      For any :math:`x_{r_1} \in \calH` with :math:`g_{r_1} = \nabla f(x_{r_1})` and
      :math:`F_{r_1} = f(x_{r_1})`,

      .. math::
          F_{r_1} - F_\star \le \frac{1}{2\mu_{\textup{gd}}}\|g_{r_1}\|^2
          \quad \text{and} \quad
          F_{r_1} \ge F_\star.

    - Matrix/vector form used in :doc:`Interpolation conditions </theory/interpolation_conditions>`:

      With :math:`F=(F_{r_1},F_\star)` and
      :math:`z=(x_{r_1},x_\star,g_{r_1},0)`, the two inequalities are
      encoded by

      .. math::
          a_1 = (-1, 1), \qquad
          M_1 =
          \begin{bmatrix}
              0 & 0 & 0 & 0 \\
              0 & 0 & 0 & 0 \\
              0 & 0 & 0 & 0 \\
              0 & 0 & 0 & 0
          \end{bmatrix},

      and

      .. math::
          a_2 = (1, -1), \qquad
          M_2 =
          \begin{bmatrix}
              0 & 0 & 0 & 0 \\
              0 & 0 & 0 & 0 \\
              0 & 0 & -\frac{1}{2\mu_{\textup{gd}}} & 0 \\
              0 & 0 & 0 & 0
          \end{bmatrix}.

    **Parameters**

    - `mu_gd` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The gradient-dominated parameter corresponding
      to :math:`\mu_{\textup{gd}}` (must be :math:`> 0` and finite).

    **Raises**

    - `ValueError`: If `mu_gd` is not a number, :math:`\le 0`, or infinite.

    **Note**

    - This condition is only sufficient for AutoLyap analyses; tightness of
      the resulting bound is not guaranteed.
    - When used inside :class:`~autolyap.problemclass.InclusionProblem`,
      the problem must have exactly one component. In the notation of
      :doc:`3. Algorithm representation </theory/algorithm_representation>`,
      :math:`m = 1`, :math:`m_{\textup{func}} = 1`, and
      :math:`m_{\textup{op}} = 0`. This single component may still
      contain a list of function conditions (an intersection).
    """
    def __init__(self, mu_gd: Union[int, float]) -> None:
        mu_gd = _ensure_positive_finite(
            mu_gd,
            "Gradient-dominated parameter",
            "Gradient-dominated parameter (mu_gd) must be greater than 0 and finite.",
        )
        self.mu_gd = mu_gd

    def get_data(self) -> List[Tuple[np.ndarray, np.ndarray, bool, _InterpolationIndices]]:
        r"""
        Return interpolation data for gradient-dominated functions.

        The tuple format and notation follow the class-level reference in
        :class:`~autolyap.problemclass.GradientDominated`.

        **Returns**

        - (:class:`~typing.List`\[:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`bool`, :class:`~autolyap.problemclass.indices._InterpolationIndices`\]\]): A list containing two
          tuples with the interpolation data.

        """
        a1 = np.array([-1, 1])
        M1 = np.zeros((4, 4))
        
        a2 = np.array([1, -1])
        M2 = np.zeros((4, 4))
        M2[2, 2] = -1 / (2 * self.mu_gd)
        
        interp_idx = _InterpolationIndices("r1!=star")
        eq_flag = False
        return [
            (M1, a1, eq_flag, interp_idx),
            (M2, a2, eq_flag, interp_idx)
        ]
