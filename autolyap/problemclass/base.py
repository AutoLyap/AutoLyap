"""Abstract interpolation-condition interfaces."""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple

from autolyap.problemclass.indices import InterpolationIndices

class InterpolationCondition(ABC):
    r"""
    Abstract base class for an interpolation condition.

    Class-level reference
    =====================

    This class-level docstring centralizes the data contract implemented by
    interpolation-condition subclasses.

    Derived classes must implement :meth:`get_data`, which returns interpolation data as a list
    of tuples encoding the relevant vectors, matrices, and interpolation indices.
    See :class:`~autolyap.problemclass.OperatorInterpolationCondition` and
    :class:`~autolyap.problemclass.FunctionInterpolationCondition` for
    the detailed notation and the corresponding interpolation constraints.

    """
    @abstractmethod
    def get_data(self):
        r"""
        Return interpolation data.

        Shared tuple conventions follow the class-level reference in
        :class:`~autolyap.problemclass.InterpolationCondition`.

        **Returns**

        - (:class:`~typing.List`\[:class:`~typing.Union`\[:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`~autolyap.problemclass.problemclass.InterpolationIndices`\], :class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`bool`, :class:`~autolyap.problemclass.problemclass.InterpolationIndices`\]\]\]):
          A list of tuples representing the interpolation data.

        """
        pass

class OperatorInterpolationCondition(InterpolationCondition):
    r"""
    Base class for operator interpolation conditions.

    Class-level reference
    =====================

    This class-level docstring centralizes notation and tuple conventions for
    operator interpolation constraints.

    Return a list of tuples of the form:
    
    `(matrix, interpolation_indices)`

    where:

    - **matrix** is a square, symmetric 2D numpy array.
    - **interpolation_indices** is an instance of :class:`InterpolationIndices`.

    The returned **matrix** corresponds to the quadratic term
    :math:`M_{(i,o)}^{\textup{op}}`, where :math:`i \in \IndexOp` indexes the operator component
    and :math:`o` selects one of its interpolation constraints. These enter inequalities of the form

    .. math::
        \langle z, (M_{(i,o)}^{\textup{op}} \kron \Id) z \rangle \le 0,

    where :math:`z` stacks the relevant :math:`y_{i,j}^{k}` and :math:`u_{i,j}^{k}` entries selected
    by the interpolation indices. Concretely, if the indices select pairs
    :math:`(p_1,k_1),\ldots,(p_p,k_p)`, then

    .. math::
        z = (y_{i,p_1}^{k_1},\ldots,y_{i,p_p}^{k_p},u_{i,p_1}^{k_1},\ldots,u_{i,p_p}^{k_p}).

    For operator components, :math:`u_{i,j}^{k} \in G_i(y_{i,j}^{k})`.
    No linear term appears for operator conditions.

    For any matrix :math:`M \in \mathbb{R}^{q \times p}`, the tensor product
    :math:`M \kron \Id` denotes the linear map :math:`\calH^{p} \to \calH^{q}` defined by

    .. math::
        (M \kron \Id)z = \Big(\sum_{j=1}^{p}[M]_{1,j} z_j,\ldots,\sum_{j=1}^{p}[M]_{q,j} z_j\Big),

    for :math:`z = (z_1,\ldots,z_p) \in \calH^{p}`.

    """
    @abstractmethod
    def get_data(self) -> List[Tuple[np.ndarray, InterpolationIndices]]:
        r"""
        Return operator interpolation data.

        The tuple format and notation follow the class-level reference in
        :class:`~autolyap.problemclass.OperatorInterpolationCondition`.

        **Returns**

        - (:class:`~typing.List`\[:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`~autolyap.problemclass.problemclass.InterpolationIndices`\]\]): A list of tuples, each containing a square
          symmetric matrix and an instance of :class:`InterpolationIndices`.

        """
        pass

class FunctionInterpolationCondition(InterpolationCondition):
    r"""
    Base class for function interpolation conditions.

    Class-level reference
    =====================

    This class-level docstring centralizes notation and tuple conventions for
    function interpolation constraints.

    Return a list of tuples of the form:
    
    `(matrix, vector, eq, interpolation_indices)`

    where:

    - **matrix** is a square, symmetric 2D numpy array.
    - **vector** is a 1D numpy array.
    - **eq** is a boolean flag (True for equality, False for inequality).
    - **interpolation_indices** is an instance of :class:`InterpolationIndices`.
    - **matrix** has shape :math:`(2p, 2p)` where :math:`p` is the length of **vector**.

    The returned **vector** corresponds to the linear term
    :math:`a_{(i,o)}^{\textup{func-ineq}}` or :math:`a_{(i,o)}^{\textup{func-eq}}`, where
    :math:`i \in \IndexFunc` indexes the functional component and :math:`o` selects one of its
    interpolation constraints, in function values,
    and the **matrix** corresponds to the quadratic term
    :math:`M_{(i,o)}^{\textup{func-ineq}}` or :math:`M_{(i,o)}^{\textup{func-eq}}` in constraints of the form

    .. math::
        \big(a_{(i,o)}^{\textup{func-ineq}}\big)^{\top}\begin{bmatrix}
            F_{i,p_1}^{k_1} \\
            \vdots \\
            F_{i,p_p}^{k_p}
        \end{bmatrix}
        + \langle z, (M_{(i,o)}^{\textup{func-ineq}} \kron \Id) z \rangle \le 0,

    and

    .. math::
        \big(a_{(i,o)}^{\textup{func-eq}}\big)^{\top}\begin{bmatrix}
            F_{i,p_1}^{k_1} \\
            \vdots \\
            F_{i,p_p}^{k_p}
        \end{bmatrix}
        + \langle z, (M_{(i,o)}^{\textup{func-eq}} \kron \Id) z \rangle = 0.

    Here :math:`z` stacks the corresponding :math:`y_{i,p_\ell}^{k_\ell}` and :math:`u_{i,p_\ell}^{k_\ell}`
    terms. Concretely, if the indices select pairs :math:`(p_1,k_1),\ldots,(p_p,k_p)`, then

    .. math::
        z = (y_{i,p_1}^{k_1},\ldots,y_{i,p_p}^{k_p},u_{i,p_1}^{k_1},\ldots,u_{i,p_p}^{k_p}).

    For functional components, :math:`u_{i,j}^{k} \in \partial f_i(y_{i,j}^{k})`, and
    :math:`F_{i,j}^{k} = f_i(y_{i,j}^{k})`. The flag **eq** selects equality vs. inequality.

    For any matrix :math:`M \in \mathbb{R}^{q \times p}`, the tensor product
    :math:`M \kron \Id` denotes the linear map :math:`\calH^{p} \to \calH^{q}` defined by

    .. math::
        (M \kron \Id)z = \Big(\sum_{j=1}^{p}[M]_{1,j} z_j,\ldots,\sum_{j=1}^{p}[M]_{q,j} z_j\Big),

    for :math:`z = (z_1,\ldots,z_p) \in \calH^{p}`.

    """
    @abstractmethod
    def get_data(self) -> List[Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]]:
        r"""
        Return function interpolation data.

        The tuple format and notation follow the class-level reference in
        :class:`~autolyap.problemclass.FunctionInterpolationCondition`.

        **Returns**

        - (:class:`~typing.List`\[:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`bool`, :class:`~autolyap.problemclass.problemclass.InterpolationIndices`\]\]): A list of tuples, each
          containing a square symmetric matrix, a 1D vector, a boolean flag, and an instance of
          :class:`InterpolationIndices`.

        """
        pass