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

    Let :math:`(\calH,\langle\cdot,\cdot\rangle)` be a real Hilbert space and
    :math:`G: \calH \rightrightarrows \calH`.

    - Graph:

      :math:`\operatorname{gra} G = \{(x,u) \in \calH \times \calH \mid u \in G(x)\}`.

    - Monotone:

      .. math::
          \langle u - v, x - y \rangle \geq 0
          \quad \text{for each } (x,u),(y,v) \in \operatorname{gra} G.

    - Maximally monotone:
      It is monotone and there is no monotone operator
      :math:`\widetilde{G}: \calH \rightrightarrows \calH` whose graph
      strictly contains :math:`\operatorname{gra} G`.

    - Interpolation inequality:

      For any :math:`(x_{r_1},u_{r_1}),(x_{r_2},u_{r_2}) \in \operatorname{gra} G`,

    .. math::
        \langle u_{r_1} - u_{r_2}, x_{r_1} - x_{r_2} \rangle \ge 0.

    This condition has no parameters.

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

    Let :math:`(\calH,\langle\cdot,\cdot\rangle)` be a real Hilbert space,
    :math:`G: \calH \rightrightarrows \calH`, and
    :math:`\mu \in \mathbb{R}_{++}`.

    - Graph:

      :math:`\operatorname{gra} G = \{(x,u) \in \calH \times \calH \mid u \in G(x)\}`.

    - :math:`\mu`-strongly monotone:

      .. math::
          \langle u - v, x - y \rangle \geq \mu \|x - y\|^2
          \quad \text{for each } (x,u),(y,v) \in \operatorname{gra} G.

    - Interpolation inequality:

      For any :math:`(x_{r_1},u_{r_1}),(x_{r_2},u_{r_2}) \in \operatorname{gra} G`,

      .. math::
          \langle u_{r_1} - u_{r_2}, x_{r_1} - x_{r_2} \rangle \ge \mu \|x_{r_1} - x_{r_2}\|^2.

    **Parameters**

    - `mu` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The strong monotonicity parameter corresponding
      to :math:`\mu` (must be :math:`> 0` and finite).

    **Raises**

    - `ValueError`: If `mu` is not a number, :math:`\le 0`, or infinite.
    """
    def __init__(self, mu: Union[int, float]):
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

    Let :math:`(\calH,\langle\cdot,\cdot\rangle)` be a real Hilbert space,
    :math:`G: \calH \to \calH`, and :math:`L \in \mathbb{R}_{++}`.

    - :math:`L`-Lipschitz continuous:

      .. math::
          \|G(x) - G(y)\| \leq L \|x - y\|
          \quad \text{for each } x,y \in \calH.

    - Interpolation inequality:

      For any :math:`x_{r_1}, x_{r_2} \in \calH` with
      :math:`u_{r_1} = G(x_{r_1})` and :math:`u_{r_2} = G(x_{r_2})`,

      .. math::
          \|u_{r_1} - u_{r_2}\|^2 \le L^2 \|x_{r_1} - x_{r_2}\|^2.

    **Parameters**

    - `L` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The Lipschitz parameter corresponding to
      :math:`L` (must be :math:`> 0` and finite).

    **Raises**

    - `ValueError`: If `L` is not a number, :math:`\le 0`, or infinite.
    """
    def __init__(self, L: Union[int, float]):
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

    Let :math:`(\calH,\langle\cdot,\cdot\rangle)` be a real Hilbert space,
    :math:`G: \calH \to \calH`, and :math:`\beta \in \mathbb{R}_{++}`.

    - :math:`\beta`-cocoercive:

      .. math::
          \langle G(x) - G(y), x - y \rangle \geq \beta \|G(x) - G(y)\|^2
          \quad \text{for each } x,y \in \calH.

    - Interpolation inequality:

      For any :math:`x_{r_1}, x_{r_2} \in \calH` with
      :math:`u_{r_1} = G(x_{r_1})` and :math:`u_{r_2} = G(x_{r_2})`,

      .. math::
          \langle u_{r_1} - u_{r_2}, x_{r_1} - x_{r_2} \rangle \ge \beta \|u_{r_1} - u_{r_2}\|^2.

    **Parameters**

    - `beta` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The cocoercivity parameter corresponding
      to :math:`\beta` (must be :math:`> 0` and finite).

    **Raises**

    - `ValueError`: If `beta` is not a number, :math:`\le 0`, or infinite.
    """
    def __init__(self, beta: Union[int, float]):
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

    Let :math:`(\calH,\langle\cdot,\cdot\rangle)` be a real Hilbert space,
    :math:`G: \calH \rightrightarrows \calH`, and
    :math:`\rho_{\textup{minty}} \in \mathbb{R}`.

    - Weak Minty variational inequality:
      There exists :math:`x_\star \in \calH` with :math:`0 \in G(x_\star)` such that
      for each :math:`(x,u) \in \operatorname{gra} G`,

      .. math::
          \langle u, x - x_\star \rangle \geq \rho_{\textup{minty}} \|u\|^2.

    - Interpolation inequality:

      There exists :math:`x_\star \in \calH` with :math:`0 \in G(x_\star)` such that for any
      :math:`(x_{r_1},u_{r_1}) \in \operatorname{gra} G`,

      .. math::
          \langle u_{r_1}, x_{r_1} - x_\star \rangle \ge \rho_{\textup{minty}} \|u_{r_1}\|^2.

    Note: When used inside :class:`~autolyap.problemclass.InclusionProblem`, AutoLyap enforces that the total number
    of components is exactly one (i.e., :math:`m = 1`) if any component uses this condition.

    **Parameters**

    - `rho_minty` (:class:`~typing.Union`\[:class:`int`, :class:`float`\]): The weak MVI parameter corresponding to
      :math:`\rho_{\textup{minty}}` (must be finite).

    **Raises**

    - `ValueError`: If `rho_minty` is not a number or not finite.
    """
    def __init__(self, rho_minty: Union[int, float]):
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
