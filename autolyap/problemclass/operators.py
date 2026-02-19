"""Concrete operator interpolation-condition classes."""

import numpy as np
from typing import List, Union, Tuple

from autolyap.problemclass.base import _OperatorInterpolationCondition
from autolyap.problemclass.indices import _InterpolationIndices
from autolyap.utils.validation import ensure_real_number


def _ensure_positive_finite(value: Union[int, float], parameter_name: str, error_message: str) -> float:
    r"""Return `value` as float after enforcing strict positivity and finiteness."""
    numeric_value = ensure_real_number(value, parameter_name)
    if not np.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(error_message)
    return numeric_value


def _ensure_finite(value: Union[int, float], parameter_name: str, error_message: str) -> float:
    r"""Return `value` as float after enforcing finiteness."""
    numeric_value = ensure_real_number(value, parameter_name)
    if not np.isfinite(numeric_value):
        raise ValueError(error_message)
    return numeric_value


class MaximallyMonotone(_OperatorInterpolationCondition):
    r"""
    Operator interpolation condition for maximally monotone operators.

    Let :math:`G: \calH \rightrightarrows \calH` be maximally monotone.

    - Interpolation inequality:

      For any :math:`(x_{r_1},u_{r_1}),(x_{r_2},u_{r_2}) \in \operatorname{gra} G`,

    .. math::
        \langle u_{r_1} - u_{r_2}, x_{r_1} - x_{r_2} \rangle \ge 0.

    - Matrix/vector form used in :doc:`Interpolation conditions </theory/interpolation_conditions>`:

      With an interpolation vector :math:`z` built from the stacked
      variables in the inequality above, the same condition is encoded as

      .. math::
          \mathcal{Q}\p{M, z} \le 0,

      with

      .. math::
          M = \frac{1}{2}
          \begin{bmatrix}
              0 & 0 & -1 & 1 \\
              0 & 0 & 1 & -1 \\
              -1 & 1 & 0 & 0 \\
              1 & -1 & 0 & 0
          \end{bmatrix}.

    This condition has no parameters.

    **References**

    - :cite:`bauschke2017convexanalysismonotone{Theorem 20.21}`.

    """
    def get_data(self) -> List[Tuple[np.ndarray, _InterpolationIndices]]:
        r"""
        Return interpolation data for a maximally monotone operator.

        The tuple format and notation follow the class-level reference in
        :class:`~autolyap.problemclass.MaximallyMonotone`.

        **Returns**

        - (:class:`~typing.List`\[:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`~autolyap.problemclass.indices._InterpolationIndices`\]\]): A list containing one tuple with the
          interpolation matrix and indices.

        """
        matrix = 0.5 * np.array([
            [0, 0, -1, 1],
            [0, 0,  1, -1],
            [-1, 1,  0, 0],
            [1, -1,  0, 0]
        ])
        interp_idx = _InterpolationIndices("r1<r2")
        return [(matrix, interp_idx)]

class StronglyMonotone(_OperatorInterpolationCondition):
    r"""
    Operator interpolation condition for strongly and maximally monotone operators.

    Let :math:`\mu \in \mathbb{R}_{++}` and
    :math:`G: \calH \rightrightarrows \calH` be :math:`\mu`-strongly monotone.

    - Interpolation inequality:

      For any :math:`(x_{r_1},u_{r_1}),(x_{r_2},u_{r_2}) \in \operatorname{gra} G`,

      .. math::
          \langle u_{r_1} - u_{r_2}, x_{r_1} - x_{r_2} \rangle \ge \mu \|x_{r_1} - x_{r_2}\|^2.

    - Matrix/vector form used in :doc:`Interpolation conditions </theory/interpolation_conditions>`:

      With an interpolation vector :math:`z` built from the stacked
      variables in the inequality above, the same condition is encoded as

      .. math::
          \mathcal{Q}\p{M, z} \le 0,

      with

      .. math::
          M = \frac{1}{2}
          \begin{bmatrix}
              2\mu & -2\mu & -1 & 1 \\
              -2\mu & 2\mu & 1 & -1 \\
              -1 & 1 & 0 & 0 \\
              1 & -1 & 0 & 0
          \end{bmatrix}.

    **Parameters**

    - `mu` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The strong monotonicity parameter corresponding
      to :math:`\mu` (must be :math:`> 0` and finite).

    **Raises**

    - `ValueError`: If `mu` is not a number, :math:`\le 0`, or infinite.

    **References**

    - :cite:`ryu2020operatorsplittingperformance{Proposition 1}`.
    """
    def __init__(self, mu: Union[int, float]) -> None:
        mu = _ensure_positive_finite(
            mu,
            "Strong monotonicity parameter",
            "Strong monotonicity parameter (mu) must be greater than 0 and finite.",
        )
        self.mu = mu

    def get_data(self) -> List[Tuple[np.ndarray, _InterpolationIndices]]:
        r"""
        Return interpolation data for the strongly monotone operator.

        The tuple format and notation follow the class-level reference in
        :class:`~autolyap.problemclass.StronglyMonotone`.

        **Returns**

        - (:class:`~typing.List`\[:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`~autolyap.problemclass.indices._InterpolationIndices`\]\]): A list containing one tuple with the
          interpolation matrix and indices.

        """
        matrix = 0.5 * np.array([
            [2 * self.mu, -2 * self.mu, -1, 1],
            [-2 * self.mu, 2 * self.mu,  1, -1],
            [-1, 1, 0, 0],
            [1, -1, 0, 0]
        ])
        interp_idx = _InterpolationIndices("r1<r2")
        return [(matrix, interp_idx)]

class LipschitzOperator(_OperatorInterpolationCondition):
    r"""
    Operator interpolation condition for Lipschitz operators.

    Let :math:`L \in \mathbb{R}_{++}` and
    :math:`G: \calH \to \calH` be :math:`L`-Lipschitz continuous.

    - Interpolation inequality:

      For any :math:`x_{r_1}, x_{r_2} \in \calH` with
      :math:`u_{r_1} = G(x_{r_1})` and :math:`u_{r_2} = G(x_{r_2})`,

      .. math::
          \|u_{r_1} - u_{r_2}\|^2 \le L^2 \|x_{r_1} - x_{r_2}\|^2.

    - Matrix/vector form used in :doc:`Interpolation conditions </theory/interpolation_conditions>`:

      With an interpolation vector :math:`z` built from the stacked
      variables in the inequality above, the same condition is encoded as

      .. math::
          \mathcal{Q}\p{M, z} \le 0,

      with

      .. math::
          M =
          \begin{bmatrix}
              -L^2 & L^2 & 0 & 0 \\
              L^2 & -L^2 & 0 & 0 \\
              0 & 0 & 1 & -1 \\
              0 & 0 & -1 & 1
          \end{bmatrix}.

    **Parameters**

    - `L` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The Lipschitz parameter corresponding to
      :math:`L` (must be :math:`> 0` and finite).

    **Raises**

    - `ValueError`: If `L` is not a number, :math:`\le 0`, or infinite.

    **References**

    - Kirszbraun--Valentine theorem:
      :cite:`kirszbraun1934lipschitz`,
      :cite:`valentine1943extension`,
      :cite:`valentine1945lipschitzconditionpreserving`.
    """
    def __init__(self, L: Union[int, float]) -> None:
        L = _ensure_positive_finite(
            L,
            "Lipschitz parameter",
            "Lipschitz parameter (L) must be greater than 0 and finite.",
        )
        self.L = L

    def get_data(self) -> List[Tuple[np.ndarray, _InterpolationIndices]]:
        r"""
        Return interpolation data for the Lipschitz operator.

        The tuple format and notation follow the class-level reference in
        :class:`~autolyap.problemclass.LipschitzOperator`.

        **Returns**

        - (:class:`~typing.List`\[:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`~autolyap.problemclass.indices._InterpolationIndices`\]\]): A list containing one tuple with the
          interpolation matrix and indices.

        """
        matrix = np.array([
            [-self.L**2, self.L**2, 0, 0],
            [self.L**2, -self.L**2, 0, 0],
            [0, 0, 1, -1],
            [0, 0, -1, 1]
        ])
        interp_idx = _InterpolationIndices("r1<r2")
        return [(matrix, interp_idx)]

class Cocoercive(_OperatorInterpolationCondition):
    r"""
    Operator interpolation condition for cocoercive operators.

    Let :math:`\beta \in \mathbb{R}_{++}` and
    :math:`G: \calH \to \calH` be :math:`\beta`-cocoercive.

    - Interpolation inequality:

      For any :math:`x_{r_1}, x_{r_2} \in \calH` with
      :math:`u_{r_1} = G(x_{r_1})` and :math:`u_{r_2} = G(x_{r_2})`,

      .. math::
          \langle u_{r_1} - u_{r_2}, x_{r_1} - x_{r_2} \rangle \ge \beta \|u_{r_1} - u_{r_2}\|^2.

    - Matrix/vector form used in :doc:`Interpolation conditions </theory/interpolation_conditions>`:

      With an interpolation vector :math:`z` built from the stacked
      variables in the inequality above, the same condition is encoded as

      .. math::
          \mathcal{Q}\p{M, z} \le 0,

      with

      .. math::
          M = \frac{1}{2}
          \begin{bmatrix}
              0 & 0 & -1 & 1 \\
              0 & 0 & 1 & -1 \\
              -1 & 1 & 2\beta & -2\beta \\
              1 & -1 & -2\beta & 2\beta
          \end{bmatrix}.

    **Parameters**

    - `beta` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The cocoercivity parameter corresponding
      to :math:`\beta` (must be :math:`> 0` and finite).

    **Raises**

    - `ValueError`: If `beta` is not a number, :math:`\le 0`, or infinite.

    **References**

    - :cite:`ryu2020operatorsplittingperformance{Proposition 2}`.
    """
    def __init__(self, beta: Union[int, float]) -> None:
        beta = _ensure_positive_finite(
            beta,
            "Cocoercivity parameter",
            "Cocoercivity parameter (beta) must be greater than 0 and finite.",
        )
        self.beta = beta

    def get_data(self) -> List[Tuple[np.ndarray, _InterpolationIndices]]:
        r"""
        Return interpolation data for the cocoercive operator.

        The tuple format and notation follow the class-level reference in
        :class:`~autolyap.problemclass.Cocoercive`.

        **Returns**

        - (:class:`~typing.List`\[:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`~autolyap.problemclass.indices._InterpolationIndices`\]\]): A list containing one tuple with the
          interpolation matrix and indices.

        """
        matrix = 0.5 * np.array([
            [0, 0, -1, 1],
            [0, 0,  1, -1],
            [-1, 1, 2 * self.beta, -2 * self.beta],
            [1, -1, -2 * self.beta, 2 * self.beta]
        ])
        interp_idx = _InterpolationIndices("r1<r2")
        return [(matrix, interp_idx)]

class WeakMintyVariationalInequality(_OperatorInterpolationCondition):
    r"""
    Operator interpolation condition for operators that fulfill the weak Minty variational inequality.

    Let :math:`\rho_{\textup{minty}} \in \mathbb{R}` and
    :math:`G: \calH \rightrightarrows \calH` satisfy a weak Minty variational inequality.

    - Interpolation inequality:

      There exists :math:`x_\star \in \calH` with :math:`0 \in G(x_\star)` such that for any
      :math:`(x_{r_1},u_{r_1}) \in \operatorname{gra} G`,

      .. math::
          \langle u_{r_1}, x_{r_1} - x_\star \rangle \ge \rho_{\textup{minty}} \|u_{r_1}\|^2.

    - Matrix/vector form used in :doc:`Interpolation conditions </theory/interpolation_conditions>`:

      With an interpolation vector :math:`z` built from the stacked
      variables in the inequality above, the same condition is encoded as

      .. math::
          \mathcal{Q}\p{M, z} \le 0,

      with

      .. math::
          M = \frac{1}{2}
          \begin{bmatrix}
              0 & 0 & -1 & 0 \\
              0 & 0 & 1 & 0 \\
              -1 & 1 & 2\rho_{\textup{minty}} & 0 \\
              0 & 0 & 0 & 0
          \end{bmatrix}.

    **Parameters**

    - `rho_minty` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The weak MVI parameter corresponding to
      :math:`\rho_{\textup{minty}}` (must be finite).

    **Raises**

    - `ValueError`: If `rho_minty` is not a number or not finite.

    **Note**

    - When used inside :class:`~autolyap.problemclass.InclusionProblem`,
      the problem must have exactly one component. In the notation of
      :doc:`3. Algorithm representation </theory/algorithm_representation>`,
      :math:`m = 1`, :math:`m_{\textup{op}} = 1`, and
      :math:`m_{\textup{func}} = 0`. This single component may still
      contain a list of operator conditions (an intersection).
    """
    def __init__(self, rho_minty: Union[int, float]) -> None:
        rho_minty = _ensure_finite(
            rho_minty,
            "Weak MVI parameter",
            "Weak MVI parameter (rho_minty) must be finite.",
        )
        self.rho_minty = rho_minty

    def get_data(self) -> List[Tuple[np.ndarray, _InterpolationIndices]]:
        r"""
        Return interpolation data for the weak Minty variational inequality condition.

        The tuple format and notation follow the class-level reference in
        :class:`~autolyap.problemclass.WeakMintyVariationalInequality`.

        **Returns**

        - (:class:`~typing.List`\[:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`~autolyap.problemclass.indices._InterpolationIndices`\]\]): A list containing one tuple with the
          interpolation matrix and indices.

        """
        matrix = 0.5 * np.array([
            [0, 0, -1, 0],
            [0, 0,  1, 0],
            [-1, 1, 2 * self.rho_minty, 0],
            [0, 0, 0, 0]
        ])
        interp_idx = _InterpolationIndices("r1!=star")
        return [(matrix, interp_idx)]
