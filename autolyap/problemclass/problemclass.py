import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Dict

# ---------------------------------------------------------------------------
# InterpolationIndices Class
# ---------------------------------------------------------------------------
class InterpolationIndices:
    r"""
    A wrapper for interpolation indices that ensures only allowed values are used.

    :param value: The interpolation index string.
    :type value: str
    :raises ValueError: If the provided value is not in the allowed set.
    """
    ALLOWED_VALUES = {"i<j", "i!=j", "i", "i!=star"}

    def __init__(self, value: str):
        if value not in self.ALLOWED_VALUES:
            raise ValueError(f"Invalid interpolation index: {value}. Allowed values: {self.ALLOWED_VALUES}")
        self.value = value

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, InterpolationIndices):
            return self.value == other.value
        elif isinstance(other, str):
            return self.value == other
        return False

    def __repr__(self):
        return f"InterpolationIndices({self.value})"


# ---------------------------------------------------------------------------
# Abstract Base Classes for Interpolation Conditions
# ---------------------------------------------------------------------------
class InterpolationCondition(ABC):
    r"""
    Abstract base class for an interpolation condition.

    Derived classes must implement the :meth:`get_data` method, which returns a list of tuples 
    representing the interpolation data.
    """
    @abstractmethod
    def get_data(self):
        r"""
        Return the interpolation data.

        :returns: A list of tuples representing the interpolation data.
        :rtype: list
        """
        pass

class OperatorInterpolationCondition(InterpolationCondition):
    r"""
    Base class for operator interpolation conditions.

    Must return a list of tuples of the form:
    
    ``(matrix, interpolation_indices)``

    where:

    - **matrix** is a square, symmetric 2D numpy array.
    - **interpolation_indices** is an instance of :class:`InterpolationIndices`.
    """
    @abstractmethod
    def get_data(self) -> List[Tuple[np.ndarray, InterpolationIndices]]:
        r"""
        Return the operator interpolation data.

        :returns: A list of tuples, each containing a square symmetric matrix and an instance of :class:`InterpolationIndices`.
        :rtype: List[Tuple[np.ndarray, InterpolationIndices]]
        """
        pass

class FunctionInterpolationCondition(InterpolationCondition):
    r"""
    Base class for function interpolation conditions.

    Must return a list of tuples of the form:
    
    ``(matrix, vector, eq, interpolation_indices)``

    where:

    - **matrix** is a square, symmetric 2D numpy array.
    - **vector** is a 1D numpy array.
    - **eq** is a boolean flag (True for equality, False for inequality).
    - **interpolation_indices** is an instance of :class:`InterpolationIndices`.
    """
    @abstractmethod
    def get_data(self) -> List[Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]]:
        r"""
        Return the function interpolation data.

        :returns: A list of tuples, each containing a square symmetric matrix, a 1D vector, a boolean flag,
                  and an instance of :class:`InterpolationIndices`.
        :rtype: List[Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]]
        """
        pass


# ---------------------------------------------------------------------------
# Concrete Operator Interpolation Condition Classes
# ---------------------------------------------------------------------------
class MaximallyMonotone(OperatorInterpolationCondition):
    r"""
    Operator interpolation condition for a maximally monotone operator.

    This condition does not require any parameters.
    """
    def get_data(self) -> List[Tuple[np.ndarray, InterpolationIndices]]:
        r"""
        Return the interpolation data for a maximally monotone operator.

        :returns: A list containing one tuple with the interpolation matrix and indices.
        :rtype: List[Tuple[np.ndarray, InterpolationIndices]]
        """
        matrix = 0.5 * np.array([
            [0, 0, -1, 1],
            [0, 0,  1, -1],
            [-1, 1,  0, 0],
            [1, -1,  0, 0]
        ])
        interp_idx = InterpolationIndices("i<j")
        return [(matrix, interp_idx)]

class StronglyMonotone(OperatorInterpolationCondition):
    r"""
    Operator interpolation condition for a strongly monotone operator.

    :param mu: The strong monotonicity parameter (must be > 0 and finite).
    :type mu: float
    :raises ValueError: If mu is not a number, <= 0, or infinite.
    """
    def __init__(self, mu: Union[int, float]):
        if not isinstance(mu, (int, float)):
            raise ValueError("Strong monotonicity parameter must be a number.")
        mu = float(mu)
        if mu <= 0:
            raise ValueError("Strong monotonicity parameter (mu) must be greater than 0.")
        if mu == float('inf'):
            raise ValueError("Strong monotonicity parameter (mu) must be finite.")
        self.mu = mu

    def get_data(self) -> List[Tuple[np.ndarray, InterpolationIndices]]:
        r"""
        Return the interpolation data for the strongly monotone operator.

        :returns: A list containing one tuple with the interpolation matrix and indices.
        :rtype: List[Tuple[np.ndarray, InterpolationIndices]]
        """
        matrix = 0.5 * np.array([
            [2 * self.mu, -2 * self.mu, -1, 1],
            [-2 * self.mu, 2 * self.mu,  1, -1],
            [-1, 1, 0, 0],
            [1, -1, 0, 0]
        ])
        interp_idx = InterpolationIndices("i<j")
        return [(matrix, interp_idx)]

class LipschitzOperator(OperatorInterpolationCondition):
    r"""
    Operator interpolation condition for a Lipschitz operator.

    :param L: The Lipschitz parameter (must be > 0 and finite).
    :type L: float
    :raises ValueError: If L is not a number, <= 0, or infinite.
    """
    def __init__(self, L: Union[int, float]):
        if not isinstance(L, (int, float)):
            raise ValueError("Lipschitz parameter must be a number.")
        L = float(L)
        if L <= 0 or L == float('inf'):
            raise ValueError("Lipschitz parameter (L) must be greater than 0 and finite.")
        self.L = L

    def get_data(self) -> List[Tuple[np.ndarray, InterpolationIndices]]:
        r"""
        Return the interpolation data for the Lipschitz operator.

        :returns: A list containing one tuple with the interpolation matrix and indices.
        :rtype: List[Tuple[np.ndarray, InterpolationIndices]]
        """
        matrix = np.array([
            [-self.L**2, self.L**2, 0, 0],
            [self.L**2, -self.L**2, 0, 0],
            [0, 0, 1, -1],
            [0, 0, -1, 1]
        ])
        interp_idx = InterpolationIndices("i<j")
        return [(matrix, interp_idx)]

class Cocoercive(OperatorInterpolationCondition):
    r"""
    Operator interpolation condition for a cocoercive operator.

    :param beta: The cocoercivity parameter (must be > 0 and finite).
    :type beta: float
    :raises ValueError: If beta is not a number, <= 0, or infinite.
    """
    def __init__(self, beta: Union[int, float]):
        if not isinstance(beta, (int, float)):
            raise ValueError("Cocoercivity parameter must be a number.")
        beta = float(beta)
        if beta <= 0 or beta == float('inf'):
            raise ValueError("Cocoercivity parameter (beta) must be greater than 0 and finite.")
        self.beta = beta

    def get_data(self) -> List[Tuple[np.ndarray, InterpolationIndices]]:
        r"""
        Return the interpolation data for the cocoercive operator.

        :returns: A list containing one tuple with the interpolation matrix and indices.
        :rtype: List[Tuple[np.ndarray, InterpolationIndices]]
        """
        matrix = 0.5 * np.array([
            [0, 0, -1, 1],
            [0, 0,  1, -1],
            [-1, 1, 2 * self.beta, -2 * self.beta],
            [1, -1, -2 * self.beta, 2 * self.beta]
        ])
        interp_idx = InterpolationIndices("i<j")
        return [(matrix, interp_idx)]

class WeakMintyVariationalInequality(OperatorInterpolationCondition):
    r"""
    Operator interpolation condition for an operator that fulfills the weak Minty variational inequality.

    :param rho_minty: The weak MVI parameter (must be >= 0 and finite).
    :type rho_minty: float
    :raises ValueError: If rho_minty is not a number, negative, or infinite.
    """
    def __init__(self, rho_minty: Union[int, float]):
        if not isinstance(rho_minty, (int, float)):
            raise ValueError("Weak MVI parameter must be a number.")
        rho_minty = float(rho_minty)
        if rho_minty < 0 or rho_minty == float('inf'):
            raise ValueError("Weak MVI parameter (rho_minty) must be nonnegative and finite.")
        self.rho_minty = rho_minty

    def get_data(self) -> List[Tuple[np.ndarray, InterpolationIndices]]:
        r"""
        Return the interpolation data for the weak Minty variational inequality condition.

        :returns: A list containing one tuple with the interpolation matrix and indices.
        :rtype: List[Tuple[np.ndarray, InterpolationIndices]]
        """
        matrix = 0.5 * np.array([
            [0, 0, -1, 0],
            [0, 0,  1, 0],
            [-1, 1, -self.rho_minty, 0],
            [0, 0, 0, 0]
        ])
        interp_idx = InterpolationIndices("i!=star")
        return [(matrix, interp_idx)]


# ---------------------------------------------------------------------------
# Concrete Function Interpolation Condition Classes
# ---------------------------------------------------------------------------
class ParametrizedFunctionInterpolationCondition(FunctionInterpolationCondition):
    r"""
    Base class for function interpolation conditions that depend on parameters mu and L.

    Provides a helper function to compute the interpolation data based on mu and L.
    This base class is used for both smooth and nonsmooth conditions by setting L appropriately.

    :param mu: The convexity parameter. For strongly convex functions, mu > 0; for convex functions, mu = 0;
               for weakly convex functions, mu < 0.
    :type mu: float
    :param L: The smoothness parameter. For nonsmooth functions, L is set to infinity.
    :type L: float
    :raises ValueError: If mu is not a number or if mu is not less than L.
    """
    def __init__(self, mu: Union[int, float], L: Union[int, float]):
        if not isinstance(mu, (int, float)):
            raise ValueError("Parameter mu must be a number.")
        if not isinstance(L, (int, float)):
            raise ValueError("Parameter L must be a number.")
        mu = float(mu)
        L = float(L)
        if mu == float('-inf'):
            raise ValueError("ParametrizedFunctionInterpolationCondition: mu cannot be -inf.")
        if not (mu < L):
            raise ValueError("ParametrizedFunctionInterpolationCondition requires that -inf < mu < L <= +inf.")
        self.mu = mu
        self.L = L

    @staticmethod
    def _compute_interpolation_data(mu: float, L: float) -> Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]:
        r"""
        Compute the interpolation data based on mu and L.

        When L is infinite, the condition is nonsmooth and a simpler interpolation matrix is used.
        When L is finite, the interpolation data follows the formula for smooth functions.

        :param mu: The convexity parameter.
        :type mu: float
        :param L: The smoothness parameter.
        :type L: float
        :returns: A tuple containing the matrix, vector, eq flag, and interpolation indices.
        :rtype: Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]
        """
        if L == float('inf'):
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
        interp_idx = InterpolationIndices("i!=j")
        return matrix, vector, eq, interp_idx

    def get_data(self) -> List[Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]]:
        r"""
        Return the interpolation data for the function condition.

        :returns: A list containing one tuple with the matrix, vector, eq flag, and interpolation indices.
        :rtype: List[Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]]
        """
        return [self._compute_interpolation_data(self.mu, self.L)]

class Convex(ParametrizedFunctionInterpolationCondition):
    r"""
    Interpolation condition for convex functions.
    """
    def __init__(self):
        super().__init__(mu=0.0, L=float('inf'))

class StronglyConvex(ParametrizedFunctionInterpolationCondition):
    r"""
    Function interpolation condition for a strongly convex function.

    :param mu: Strong convexity parameter (must be > 0 and finite).
    :type mu: float
    :raises ValueError: If mu is not valid.
    """
    def __init__(self, mu: Union[int, float]):
        if not isinstance(mu, (int, float)):
            raise ValueError("Parameter mu must be a number.")
        mu = float(mu)
        if mu <= 0:
            raise ValueError("For StronglyConvex, mu must be > 0.")
        if mu == float('inf'):
            raise ValueError("For StronglyConvex, mu must be finite.")
        super().__init__(mu=mu, L=float('inf'))

class WeaklyConvex(ParametrizedFunctionInterpolationCondition):
    r"""
    Function interpolation condition for a weakly convex function.

    :param mu_tilde: Weak convexity parameter (must be > 0).
    :type mu_tilde: float
    :raises ValueError: If mu is not valid.
    """
    def __init__(self, mu_tilde: Union[int, float]):
        mu = - mu_tilde
        if not isinstance(mu, (int, float)):
            raise ValueError("Parameter mu must be a number.")
        mu = float(mu)
        if mu >= 0:
            raise ValueError("For WeaklyConvex, mu_tilde must be > 0.")
        super().__init__(mu=mu, L=float('inf'))

class Smooth(ParametrizedFunctionInterpolationCondition):
    r"""
    Function interpolation condition for a smooth function.

    :param L: Smoothness parameter (must be > 0 and finite).
    :type L: float
    :raises ValueError: If L is not valid.
    """
    def __init__(self, L: Union[int, float]):
        if not isinstance(L, (int, float)):
            raise ValueError("Parameter L must be a number.")
        L = float(L)
        if L <= 0 or L == float('inf'):
            raise ValueError("For Smooth, L must be > 0 and finite.")
        super().__init__(mu=-L, L=L)

class SmoothConvex(ParametrizedFunctionInterpolationCondition):
    r"""
    Function interpolation condition for a smooth convex function.

    :param L: Smoothness parameter (must be > 0 and finite).
    :type L: float
    :raises ValueError: If L is not valid.
    """
    def __init__(self, L: Union[int, float]):
        if not isinstance(L, (int, float)):
            raise ValueError("Parameter L must be a number.")
        L = float(L)
        if L <= 0 or L == float('inf'):
            raise ValueError("For SmoothConvex, L must be > 0 and finite.")
        super().__init__(mu=0.0, L=L)

class SmoothStronglyConvex(ParametrizedFunctionInterpolationCondition):
    r"""
    Function interpolation condition for a smooth strongly convex function.

    :param mu: Strong convexity parameter (must be > 0 and finite).
    :type mu: float
    :param L: Smoothness parameter (must be > 0 and finite) with mu < L.
    :type L: float
    :raises ValueError: If parameters are not valid.
    """
    def __init__(self, mu: Union[int, float], L: Union[int, float]):
        if not isinstance(mu, (int, float)):
            raise ValueError("Parameter mu must be a number.")
        if not isinstance(L, (int, float)):
            raise ValueError("Parameter L must be a number.")
        mu = float(mu)
        L = float(L)
        if mu <= 0:
            raise ValueError("For SmoothStronglyConvex, mu must be > 0.")
        if mu == float('inf'):
            raise ValueError("For SmoothStronglyConvex, mu must be finite.")
        if L <= 0 or L == float('inf'):
            raise ValueError("For SmoothStronglyConvex, L must be > 0 and finite.")
        if mu >= L:
            raise ValueError("For SmoothStronglyConvex, mu must be less than L.")
        super().__init__(mu=mu, L=L)

class SmoothWeaklyConvex(ParametrizedFunctionInterpolationCondition):
    r"""
    Function interpolation condition for a smooth weakly convex function.

    :param mu_tilde: Weak convexity parameter (must be > 0).
    :type mu_tilde: float
    :param L: Smoothness parameter (must be > 0 and finite).
    :type L: float
    :raises ValueError: If parameters are not valid.
    """
    def __init__(self, mu_tilde: Union[int, float], L: Union[int, float]):
        mu = - mu_tilde
        if not isinstance(mu, (int, float)):
            raise ValueError("Parameter mu_tilde must be a number.")
        if not isinstance(L, (int, float)):
            raise ValueError("Parameter L must be a number.")
        mu = float(mu)
        L = float(L)
        if mu >= 0:
            raise ValueError("For SmoothWeaklyConvex, mu_tilde must be > 0.")
        if L <= 0 or L == float('inf'):
            raise ValueError("For SmoothWeaklyConvex, L must be > 0 and finite.")
        super().__init__(mu=mu, L=L)

class IndicatorFunctionOfClosedConvexSet(FunctionInterpolationCondition):
    r"""
    Function interpolation condition for the indicator function of a closed convex set.

    This condition does not require any parameters.
    """
    def get_data(self) -> List[Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]]:
        r"""
        Return the interpolation data for the indicator function.

        :returns: A list containing two tuples with the interpolation data.
        :rtype: List[Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]]
        """
        interp_idx_ineq = InterpolationIndices("i!=j")
        matrix_ineq = 0.5 * np.array([
            [0, 0, 0, 1],
            [0, 0, 0, -1],
            [0, 0, 0, 0],
            [1, -1, 0, 0]
        ])
        vector_ineq = np.array([0, 0])
        interp_idx_eq = InterpolationIndices("i")
        matrix_eq = np.array([[0, 0], [0, 0]])
        vector_eq = np.array([1])
        return [
            (matrix_ineq, vector_ineq, False, interp_idx_ineq),
            (matrix_eq, vector_eq, True, interp_idx_eq)
        ]

class SupportFunctionOfClosedConvexSet(FunctionInterpolationCondition):
    r"""
    Function interpolation condition for the support function of a closed convex set.

    This condition does not require any parameters.
    """
    def get_data(self) -> List[Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]]:
        r"""
        Return the interpolation data for the support function.

        :returns: A list containing two tuples with the interpolation data.
        :rtype: List[Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]]
        """
        interp_idx_ineq = InterpolationIndices("i!=j")
        matrix_ineq = 0.5 * np.array([
            [0, 0, 0, 0],
            [0, 0, 1, -1],
            [0, 1, 0, 0],
            [0, -1, 0, 0]
        ])
        vector_ineq = np.array([0, 0])
        interp_idx_eq = InterpolationIndices("i")
        matrix_eq = 0.5 * np.array([[0, 1], [1, 0]])
        vector_eq = np.array([-1])
        return [
            (matrix_ineq, vector_ineq, False, interp_idx_ineq),
            (matrix_eq, vector_eq, True, interp_idx_eq)
        ]

class GradientDominated(FunctionInterpolationCondition):
    r"""
    Function interpolation condition for gradient-dominated functions.

    :param mu_gd: The gradient-dominated parameter (must be > 0 and finite).
    :type mu_gd: float
    :raises ValueError: If mu_gd is not a number, <= 0, or infinite.
    """
    def __init__(self, mu_gd: Union[int, float]):
        if not isinstance(mu_gd, (int, float)):
            raise ValueError("Gradient-dominated parameter must be a number.")
        mu_gd = float(mu_gd)
        if mu_gd <= 0 or mu_gd == float('inf'):
            raise ValueError("Gradient-dominated parameter (mu_gd) must be greater than 0 and finite.")
        self.mu_gd = mu_gd

    def get_data(self) -> List[Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]]:
        r"""
        Return the interpolation data for gradient-dominated functions.

        :returns: A list containing two tuples with the interpolation data.
        :rtype: List[Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]]
        """
        a1 = np.array([-1, 1])
        M1 = np.zeros((4, 4))
        
        a2 = np.array([1, -1])
        M2 = np.zeros((4, 4))
        M2[2, 2] = -1 / (2 * self.mu_gd)
        
        interp_idx = InterpolationIndices("i!=star")
        eq_flag = False
        return [
            (M1, a1, eq_flag, interp_idx),
            (M2, a2, eq_flag, interp_idx)
        ]


# ---------------------------------------------------------------------------
# InclusionProblem Class
# ---------------------------------------------------------------------------
class InclusionProblem:
    r"""
    Encapsulates an inclusion problem of the form:

    .. math::
       \text{find } y \in H \text{ such that } 0 \in \sum_{i \in I_{\text{func}}} \partial f_i(y) + \sum_{i \in I_{\text{op}}} G_i(y)

    :param components: A list where each element represents a component (1-indexed).
                       Each element is either a single interpolation condition instance or a list of such instances.
                       All conditions for a given component must be of the same type (either all operator conditions or all function conditions).
    :type components: List[Union[InterpolationCondition, List[InterpolationCondition]]]
    :raises ValueError: If components is empty or if components contain invalid conditions.
    """
    def __init__(self, components: List[Union[InterpolationCondition, List[InterpolationCondition]]]):
        if not components or len(components) < 1:
            raise ValueError("Error in InclusionProblem __init__: At least one component is required.")
        self.m = len(components)
        self.components: Dict[int, List[InterpolationCondition]] = {}
        # Convert the list to a dictionary with 1-indexed keys.
        for i, comp in enumerate(components, start=1):
            if isinstance(comp, list):
                for v in comp:
                    if not isinstance(v, InterpolationCondition):
                        raise ValueError(
                            f"Error in InclusionProblem __init__: Component {i} must contain only "
                            f"InterpolationCondition instances. Got {type(v)}."
                        )
                self.components[i] = comp
            else:
                if not isinstance(comp, InterpolationCondition):
                    raise ValueError(
                        f"Error in InclusionProblem __init__: Component {i} must be an "
                        f"InterpolationCondition instance. Got {type(comp)}."
                    )
                self.components[i] = [comp]
            self._validate_component_uniformity(i, self.components[i])
            self._validate_component_data(i, self.components[i])
        
        for conditions in self.components.values():
            if any(isinstance(cond, GradientDominated) for cond in conditions):
                if self.m != 1:
                    raise ValueError(
                        "Error: If any component contains a GradientDominated instance, "
                        "the total number of components (m) must be exactly 1."
                    )

        for conditions in self.components.values():
            if any(isinstance(cond, WeakMintyVariationalInequality) for cond in conditions):
                if self.m != 1:
                    raise ValueError(
                        "Error: If any component contains a WeakMintyVariationalInequality instance, "
                        "the total number of components (m) must be exactly 1."
                    )

        self.I_op = [k for k, conds in self.components.items() 
                     if isinstance(conds[0], OperatorInterpolationCondition)]
        self.I_func = [k for k, conds in self.components.items() 
                       if isinstance(conds[0], FunctionInterpolationCondition)]
    
    def _validate_component_uniformity(self, index: int, conditions: List[InterpolationCondition]) -> None:
        r"""
        Ensure that all conditions for a given component are of the same type.

        :param index: The component index.
        :type index: int
        :param conditions: A list of interpolation conditions.
        :type conditions: List[InterpolationCondition]
        :raises ValueError: If the conditions contain a mix of operator and function conditions.
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

        :param cond: An interpolation condition instance.
        :type cond: InterpolationCondition
        :raises ValueError: If the condition data does not conform to the expected structure.
        """
        data = cond.get_data()
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
                if not np.allclose(matrix, matrix.T, atol=1e-8):
                    raise ValueError("Error: Function interpolation matrix must be symmetric.")
                if not isinstance(vector, np.ndarray):
                    raise ValueError("Error: Function interpolation vector must be a numpy array.")
                if vector.ndim != 1:
                    raise ValueError("Error: Function interpolation vector must be 1-dimensional.")
                if matrix.shape[0] != 2 * vector.shape[0]:
                    raise ValueError(
                        "Error: Function interpolation matrix rows must equal 2 times the length of the vector."
                    )
                if not isinstance(interp_idx, InterpolationIndices):
                    raise ValueError("Error: Function interpolation indices must be an instance of InterpolationIndices.")
        else:
            raise ValueError("Error: Unknown interpolation condition type.")
    
    def _validate_component_data(self, index: int, conditions: List[InterpolationCondition]) -> None:
        r"""
        Validate the data of all conditions for a given component.

        :param index: The component index.
        :type index: int
        :param conditions: A list of interpolation conditions.
        :type conditions: List[InterpolationCondition]
        """
        for cond in conditions:
            self._validate_condition_data(cond)
    
    def get_component_data(self, index: int) -> List[Union[
        Tuple[np.ndarray, InterpolationIndices],
        Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]
    ]]:
        r"""
        Retrieve the raw interpolation data for a given component.

        :param index: The component index (1-indexed).
        :type index: int
        :returns: A list of tuples containing the interpolation data. For operator conditions, each tuple is
                  ``(matrix, interpolation_indices)``; for function conditions, each tuple is 
                  ``(matrix, vector, eq, interpolation_indices)``.
        :rtype: List[Union[Tuple[np.ndarray, InterpolationIndices], Tuple[np.ndarray, np.ndarray, bool, InterpolationIndices]]]
        :raises ValueError: If the index is not defined.
        """
        if index < 1 or index > self.m:
            raise ValueError(
                f"Error: Index must be in 1,...,{self.m}. Component {index} is not defined."
            )
        data = []
        for cond in self.components[index]:
            data.extend(cond.get_data())
        return data
    
    def get_component(self, index: int) -> List[InterpolationCondition]:
        r"""
        Retrieve the interpolation condition instances for a given component.

        :param index: The component index (1-indexed).
        :type index: int
        :returns: A list of interpolation condition instances.
        :rtype: List[InterpolationCondition]
        :raises ValueError: If the index is not defined.
        """
        if index < 1 or index > self.m:
            raise ValueError(
                f"Error: Index must be in 1,...,{self.m}. Component {index} is not defined."
            )
        return self.components[index]
    
    def update_component_instances(self, 
                                   index: int, 
                                   new_instances: Union[InterpolationCondition, List[InterpolationCondition]]):
        r"""
        Update the interpolation condition instances for a given component.

        This replaces the full list of instances for that component.

        :param index: The component index (1-indexed).
        :type index: int
        :param new_instances: A single interpolation condition instance or a list of them.
        :type new_instances: Union[InterpolationCondition, List[InterpolationCondition]]
        :raises ValueError: If the index is not defined or if new_instances are invalid.
        """
        if index < 1 or index > self.m:
            raise ValueError(
                f"Error: Index must be in 1,...,{self.m}. Component {index} is not defined."
            )
        if isinstance(new_instances, list):
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
            self.components[index] = new_instances
        else:
            if not isinstance(new_instances, InterpolationCondition):
                raise ValueError(
                    f"Error: New instance for component {index} must be an "
                    "InterpolationCondition instance."
                )
            self.components[index] = [new_instances]
        self._validate_component_uniformity(index, self.components[index])
        self._validate_component_data(index, self.components[index])
