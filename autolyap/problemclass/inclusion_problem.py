"""Inclusion-problem container and validation logic."""

import numpy as np
from typing import List, Union, Tuple, Dict

from autolyap.problemclass.base import (
    InterpolationCondition,
    OperatorInterpolationCondition,
    FunctionInterpolationCondition,
)
from autolyap.problemclass.indices import InterpolationIndices
from autolyap.problemclass.functions import GradientDominated
from autolyap.problemclass.operators import WeakMintyVariationalInequality
from autolyap.utils.validation import ensure_finite_array, ensure_integral

class InclusionProblem:
    r"""
    Representation of an inclusion problem defined by interpolation conditions.

    Class-level reference
    =====================

    This class-level docstring centralizes notation and component conventions
    used by methods that validate and expose interpolation data.

    .. math::
       \text{find } y \in \calH \text{ such that }
       0 \in \sum_{i \in \IndexFunc} \partial f_i(y) + \sum_{i \in \IndexOp} G_i(y),

    where :math:`(\calH,\langle\cdot,\cdot\rangle)` is a real Hilbert space,
    :math:`f_i: \calH \to \mathbb{R} \cup \{\pm\infty\}` are functions,
    :math:`\partial f_i: \calH \rightrightarrows \calH` are their subdifferentials, and
    :math:`G_i: \calH \rightrightarrows \calH` are operators. The index sets
    :math:`\IndexFunc` and :math:`\IndexOp` are derived from the component types in `components`
    (see **Parameters** below).

    **Parameters**

    - `components` (:class:`~typing.List`\[:class:`~typing.Union`\[:class:`~autolyap.problemclass.problemclass.InterpolationCondition`, :class:`~typing.List`\[:class:`~autolyap.problemclass.problemclass.InterpolationCondition`\]\]\]): A list of
      components indexed by :math:`\llbracket 1, m\rrbracket`, where :math:`m` equals `len(components)`.
      Entry `components[i-1]` defines component :math:`i`. Each entry is either a single
      interpolation condition instance or a list
      of such instances. All conditions for a given component must be of the same type
      (either all operator conditions or all function conditions).

    **Raises**

    - `ValueError`: If `components` is empty or if any entry contains invalid conditions.

    Note: If any component uses :class:`GradientDominated` or
    :class:`WeakMintyVariationalInequality`, the total number of components must satisfy
    :math:`m = 1`.
    """
    def __init__(self, components: List[Union[InterpolationCondition, List[InterpolationCondition]]]):
        if not components or len(components) < 1:
            raise ValueError("Error in InclusionProblem __init__: At least one component is required.")
        self.m = len(components)
        self.components: Dict[int, List[InterpolationCondition]] = {}
        self._component_data_cache: Dict[int, Tuple[Union[
            Tuple[np.ndarray, InterpolationIndices],
            Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]
        ], ...]] = {}
        # Convert the list to a dictionary with 1-indexed keys.
        for i, comp in enumerate(components, start=1):
            if isinstance(comp, list):
                if not comp:
                    raise ValueError(
                        f"Error in InclusionProblem __init__: Component {i} must contain "
                        "at least one InterpolationCondition instance."
                    )
                for v in comp:
                    if not isinstance(v, InterpolationCondition):
                        raise ValueError(
                            f"Error in InclusionProblem __init__: Component {i} must contain only "
                            f"InterpolationCondition instances. Got {type(v)}."
                        )
                self.components[i] = list(comp)
            else:
                if not isinstance(comp, InterpolationCondition):
                    raise ValueError(
                        f"Error in InclusionProblem __init__: Component {i} must be an "
                        f"InterpolationCondition instance. Got {type(comp)}."
                    )
                self.components[i] = [comp]
            self._validate_component_uniformity(i, self.components[i])
            self._validate_component_data(i, self.components[i])
        
        # These conditions require m = 1 for the analysis used in AutoLyap.
        requires_single_component = (GradientDominated, WeakMintyVariationalInequality)
        for conditions in self.components.values():
            if any(isinstance(cond, requires_single_component) for cond in conditions):
                if self.m != 1:
                    raise ValueError(
                        "Error: If any component contains a GradientDominated or "
                        "WeakMintyVariationalInequality instance, the total number of "
                        "components (m) must be exactly 1."
                    )

        self._refresh_index_sets()

    @staticmethod
    def _readonly_view(array: np.ndarray) -> np.ndarray:
        view = array.view()
        view.setflags(write=False)
        return view

    def _freeze_component_data(self, data: List[Union[
        Tuple[np.ndarray, InterpolationIndices],
        Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]
    ]]) -> Tuple[Union[
        Tuple[np.ndarray, InterpolationIndices],
        Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]
    ], ...]:
        frozen = []
        for item in data:
            if len(item) == 2:
                matrix, interp_idx = item
                frozen.append((self._readonly_view(matrix), interp_idx))
            else:
                matrix, vector, eq, interp_idx = item
                frozen.append((self._readonly_view(matrix), self._readonly_view(vector), eq, interp_idx))
        return tuple(frozen)

    def _refresh_index_sets(self) -> None:
        r"""
        Recompute the operator and function index sets from `components`.

        Component conventions follow the class-level reference in
        :class:`~autolyap.problemclass.InclusionProblem`.

        **Returns**

        - `None`: This method updates :attr:`I_op` and :attr:`I_func` in place.

        """
        self.I_op = [k for k, conds in self.components.items()
                     if isinstance(conds[0], OperatorInterpolationCondition)]
        self.I_func = [k for k, conds in self.components.items()
                       if isinstance(conds[0], FunctionInterpolationCondition)]

    def _validate_component_index(self, index: int) -> int:
        index = ensure_integral(
            index,
            "Index",
            minimum=1,
        )
        if index > self.m:
            raise ValueError(
                f"Error: Index must be in 1,...,{self.m}. Component {index} is not defined."
            )
        return index
    
    def _validate_component_uniformity(self, index: int, conditions: List[InterpolationCondition]) -> None:
        r"""
        Ensure all conditions for a component are of the same type.

        Component conventions follow the class-level reference in
        :class:`~autolyap.problemclass.InclusionProblem`.

        **Parameters**

        - `index` (:class:`int`): The component index (1-indexed).
        - `conditions` (:class:`~typing.List`\[:class:`~autolyap.problemclass.problemclass.InterpolationCondition`\]): A list of interpolation conditions.

        **Raises**

        - `ValueError`: If the conditions contain a mix of operator and function conditions.
        """
        is_operator = [isinstance(v, OperatorInterpolationCondition) for v in conditions]
        is_function = [isinstance(v, FunctionInterpolationCondition) for v in conditions]
        if any(is_operator) and any(is_function):
            raise ValueError(
                f"Error: Component {index} contains a mix of operator and function "
                "interpolation conditions."
            )
    
    def _validate_condition_data(self, cond: InterpolationCondition) -> None:
        r"""
        Validate the data returned by an interpolation condition.

        Tuple conventions follow the class-level reference in
        :class:`~autolyap.problemclass.InclusionProblem`.

        **Parameters**

        - `cond` (:class:`~autolyap.problemclass.problemclass.InterpolationCondition`): An interpolation condition instance.

        **Raises**

        - `ValueError`: If the condition data does not conform to the expected structure.
        """
        data = cond.get_data()
        if not data:
            raise ValueError("Error: Interpolation condition data must be non-empty.")
        if isinstance(cond, OperatorInterpolationCondition):
            for tup in data:
                if not (isinstance(tup, tuple) and len(tup) == 2):
                    raise ValueError(
                        f"Error: Operator condition data must be a tuple of 2 elements. Received: {tup}"
                    )
                matrix, interp_idx = tup
                if not isinstance(matrix, np.ndarray):
                    raise ValueError("Error: Operator interpolation matrix must be a numpy array.")
                if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                    raise ValueError("Error: Operator interpolation matrix must be square.")
                ensure_finite_array(matrix, "Operator interpolation matrix")
                # Symmetry is required so the quadratic form is well-defined.
                if not np.allclose(matrix, matrix.T, atol=1e-8):
                    raise ValueError("Error: Operator interpolation matrix must be symmetric.")
                if not isinstance(interp_idx, InterpolationIndices):
                    raise ValueError("Error: Operator interpolation indices must be an instance of InterpolationIndices.")
        elif isinstance(cond, FunctionInterpolationCondition):
            for tup in data:
                if not (isinstance(tup, tuple) and len(tup) == 4):
                    raise ValueError(
                        f"Error: Function condition data must be a tuple of 4 elements. Received: {tup}"
                    )
                matrix, vector, eq, interp_idx = tup
                if not isinstance(matrix, np.ndarray):
                    raise ValueError("Error: Function interpolation matrix must be a numpy array.")
                if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                    raise ValueError("Error: Function interpolation matrix must be square.")
                ensure_finite_array(matrix, "Function interpolation matrix")
                # Symmetry ensures the quadratic form matches interpolation constraints.
                if not np.allclose(matrix, matrix.T, atol=1e-8):
                    raise ValueError("Error: Function interpolation matrix must be symmetric.")
                if not isinstance(vector, np.ndarray):
                    raise ValueError("Error: Function interpolation vector must be a numpy array.")
                if vector.ndim != 1:
                    raise ValueError("Error: Function interpolation vector must be 1-dimensional.")
                ensure_finite_array(vector, "Function interpolation vector")
                if matrix.shape[0] != 2 * vector.shape[0]:
                    raise ValueError(
                        "Error: Function interpolation matrix rows must equal 2 times the length of the vector."
                    )
                if not isinstance(eq, (bool, np.bool_)):
                    raise ValueError("Error: Function interpolation eq flag must be a boolean.")
                if not isinstance(interp_idx, InterpolationIndices):
                    raise ValueError("Error: Function interpolation indices must be an instance of InterpolationIndices.")
        else:
            raise ValueError("Error: Unknown interpolation condition type.")
    
    def _validate_component_data(self, index: int, conditions: List[InterpolationCondition]) -> None:
        r"""
        Validate the data of all conditions for a component.

        Component conventions follow the class-level reference in
        :class:`~autolyap.problemclass.InclusionProblem`.

        **Parameters**

        - `index` (:class:`int`): The component index (1-indexed).
        - `conditions` (:class:`~typing.List`\[:class:`~autolyap.problemclass.problemclass.InterpolationCondition`\]): A list of interpolation conditions.

        **Raises**

        - `ValueError`: If any condition returns invalid interpolation data.
        """
        for cond in conditions:
            self._validate_condition_data(cond)
    
    def get_component_data(self, index: int) -> List[Union[
        Tuple[np.ndarray, InterpolationIndices],
        Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]
    ]]:
        r"""
        Return the raw interpolation data for a component.

        Tuple conventions follow the class-level reference in
        :class:`~autolyap.problemclass.InclusionProblem`.

        **Parameters**

        - `index` (:class:`int`): The component index (1-indexed).

        **Returns**

        - (:class:`~typing.List`\[:class:`~typing.Union`\[:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`~autolyap.problemclass.problemclass.InterpolationIndices`\], :class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`bool`, :class:`~autolyap.problemclass.problemclass.InterpolationIndices`\]\]\]):
          A list of tuples containing the interpolation data. For operator conditions, each tuple is
          `(matrix, interpolation_indices)`; for function conditions, each tuple is
          `(matrix, vector, eq, interpolation_indices)`.

        **Raises**

        - `ValueError`: If the index is not defined.
        """
        index = self._validate_component_index(index)
        cached = self._component_data_cache.get(index)
        if cached is not None:
            return list(cached)
        data = []
        for cond in self.components[index]:
            data.extend(cond.get_data())
        frozen = self._freeze_component_data(data)
        self._component_data_cache[index] = frozen
        return list(frozen)
    
    def get_component(self, index: int) -> List[InterpolationCondition]:
        r"""
        Return the interpolation condition instances for a component.

        Component conventions follow the class-level reference in
        :class:`~autolyap.problemclass.InclusionProblem`.

        **Parameters**

        - `index` (:class:`int`): The component index (1-indexed).

        **Returns**

        - (:class:`~typing.List`\[:class:`~autolyap.problemclass.problemclass.InterpolationCondition`\]): A list of interpolation condition instances.

        **Raises**

        - `ValueError`: If the index is not defined.
        """
        index = self._validate_component_index(index)
        return list(self.components[index])
    
    def update_component_instances(self, 
                                   index: int, 
                                   new_instances: Union[InterpolationCondition, List[InterpolationCondition]]):
        r"""
        Update the interpolation condition instances for a component.

        Component conventions follow the class-level reference in
        :class:`~autolyap.problemclass.InclusionProblem`.

        This replaces the full list of instances for that component.

        **Parameters**

        - `index` (:class:`int`): The component index (1-indexed).
        - `new_instances` (:class:`~typing.Union`\[:class:`~autolyap.problemclass.problemclass.InterpolationCondition`, :class:`~typing.List`\[:class:`~autolyap.problemclass.problemclass.InterpolationCondition`\]\]): A single
          interpolation condition instance or a list of them.

        **Raises**

        - `ValueError`: If the index is not defined or if `new_instances` are invalid.
        """
        index = self._validate_component_index(index)
        self._component_data_cache.pop(index, None)
        if isinstance(new_instances, list):
            if not new_instances:
                raise ValueError(
                    f"Error: New instances for component {index} must contain at least one "
                    "InterpolationCondition instance."
                )
            for inst in new_instances:
                if not isinstance(inst, InterpolationCondition):
                    raise ValueError(
                        f"Error: All new instances for component {index} must be "
                        "InterpolationCondition instances."
                    )
            op_flags = [isinstance(inst, OperatorInterpolationCondition) for inst in new_instances]
            func_flags = [isinstance(inst, FunctionInterpolationCondition) for inst in new_instances]
            if any(op_flags) and any(func_flags):
                raise ValueError(
                    f"Error: New instances for component {index} must be all operator "
                    "or all function conditions; mixing is not allowed."
                )
            self.components[index] = list(new_instances)
        else:
            if not isinstance(new_instances, InterpolationCondition):
                raise ValueError(
                    f"Error: New instance for component {index} must be an "
                    "InterpolationCondition instance."
                )
            self.components[index] = [new_instances]
        self._validate_component_uniformity(index, self.components[index])
        self._validate_component_data(index, self.components[index])
        self._refresh_index_sets()
