import numpy as np
from abc import ABC, abstractmethod
from collections import OrderedDict
from numbers import Integral
from typing import Any, List, Tuple, Dict, Optional, TypeVar, Union

from autolyap.utils.validation import (
    ensure_finite_array,
    ensure_index_list,
    ensure_integral,
    ensure_m_bar_list,
    ensure_real_number,
)

Pair = Union[Tuple[int, int], Tuple[str, str]]
PairList = List[Pair]
MatrixTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
HorizonKey = Tuple[int, int]
StepKey = Union[int, str]
ProjectionKey = Tuple[int, Union[int, str]]
FKey = Tuple[int, Union[int, str], Union[int, str]]
AsBsCsDsDict = Dict[int, MatrixTuple]
UsDict = Dict[StepKey, np.ndarray]
YsDict = Dict[StepKey, np.ndarray]
XsDict = Dict[int, np.ndarray]
PsDict = Dict[ProjectionKey, np.ndarray]
FsDict = Dict[FKey, np.ndarray]
CacheValueT = TypeVar("CacheValueT")

class Algorithm(ABC):
    r"""
    Abstract base class (ABC) for algorithms expressed in the state-space
    representation used by AutoLyap.

    See :doc:`3. Algorithm representation </theory/algorithm_representation>`
    for notation and representation conventions.

    **Note** 
    
    - For a given method, the state-space representation is not unique.
    - Concrete subclasses must implement :meth:`~autolyap.algorithms.Algorithm.get_ABCD`; all other methods are implemented by this base class.

    """

    # Clear cached matrices when algorithm parameters change to avoid stale results.
    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if getattr(self, "_cache_enabled", False):
            if name in {
                "_cache_enabled",
                "_cache_maxsize",
                "_cache_AsBsCsDs",
                "_cache_Us",
                "_cache_Ys",
                "_cache_Xs",
                "_cache_Fs",
                "_Ps_cache",
                "_m_bar_offsets",
                "_func_offsets",
            } or name.startswith("_cache_"):
                return
            # Structural fields affect dimensions; invalidate all caches when they change.
            structural_fields = {
                "n",
                "m",
                "m_bar_is",
                "m_bar",
                "I_func",
                "I_op",
                "m_func",
                "m_op",
                "m_bar_func",
                "m_bar_op",
                "kappa",
            }
            if name in structural_fields:
                self._clear_all_caches()
            else:
                # Other attributes (e.g., step sizes) only affect dynamic matrices.
                self._clear_dynamic_caches()

    def __init__(self, n: int, m: int, m_bar_is: List[int],
                 I_func: List[int], I_op: List[int]) -> None:
        r"""
        Initialize an :class:`~autolyap.algorithms.Algorithm` with the
        structural parameters of the representation.
        
        **Parameters**

        - `n` (:class:`int`): the state dimension, i.e., :math:`\bx^{k} \in \calH^{n}`.
        - `m` (:class:`int`): the number of components :math:`m` in the inclusion problem.
        - `m_bar_is` (:class:`~typing.List`\[:class:`int`\]): the list :math:`(\bar{m}_i)_{i=1}^{m}` of evaluation counts per component,
          with :math:`\bar{m}_i \in \mathbb{N}` for each :math:`i`.
        - `I_func` (:class:`~typing.List`\[:class:`int`\]): the index set :math:`\IndexFunc` for functional components.
        - `I_op` (:class:`~typing.List`\[:class:`int`\]): the index set :math:`\IndexOp` for operator components.

        **Raises**

        - `ValueError`: if :math:`n < 1`, :math:`m < 1`, :math:`m \neq \text{len}(m\_bar\_is)`,
          if any :math:`\bar{m}_i \le 0`,
          :math:`\IndexFunc` and :math:`\IndexOp` are not disjoint, do not cover
          :math:`\llbracket 1, m\rrbracket`, or are not strictly increasing sequences.
        """
        super().__setattr__("_cache_enabled", False)
        # Basic validations
        n = ensure_integral(n, "n", minimum=1)
        m = ensure_integral(m, "m", minimum=1)
        m_bar_is = ensure_m_bar_list(m_bar_is, m)
        I_func = ensure_index_list(I_func, "I_func", m)
        I_op = ensure_index_list(I_op, "I_op", m)
        if not set(I_func).isdisjoint(I_op):
            raise ValueError("I_func and I_op must be disjoint")
        if set(I_func).union(I_op) != set(range(1, m + 1)):
            raise ValueError("I_func and I_op must cover {1,…, m}")

        self.n = n                      # Dimension of x.
        self.m = m                      # Total components.
        self.m_bar_is = list(m_bar_is)  # Evaluations per component (\bar{m}_i).
        self.m_bar = sum(m_bar_is)      # Total evaluations (\bar{m}).
        self.I_func = list(I_func)      # Functional indices (I_{\text{func}}).
        self.I_op = list(I_op)          # Operator indices (I_{\text{op}}).
        self.m_func = len(I_func)       # Count of functional components.
        self.m_op = len(I_op)           # Count of operator components.
        self.m_bar_func = sum(m_bar_is[i - 1] for i in I_func) if I_func else 0  # \bar{m}_{\text{func}}
        self.m_bar_op = sum(m_bar_is[i - 1] for i in I_op) if I_op else 0          # \bar{m}_{\text{op}}
        # Mapping for functional indices (for F matrices)
        self.kappa: Dict[int, int] = {I_func[i]: i + 1 for i in range(self.m_func)} if I_func else {}
        # Precompute offsets for faster projections and F rows.
        self._m_bar_offsets = np.concatenate(([0], np.cumsum(self.m_bar_is)))
        self._func_offsets: Dict[int, int] = {}
        func_offset = 0
        for i in self.I_func:
            self._func_offsets[i] = func_offset
            func_offset += self.m_bar_is[i - 1]
        self._Ps_cache: Optional[PsDict] = None
        # Bounded LRU caches keyed by (k_min, k_max) to reuse horizon-specific matrices.
        self._cache_maxsize = 8
        self._cache_AsBsCsDs: OrderedDict[HorizonKey, AsBsCsDsDict] = OrderedDict()
        self._cache_Us: OrderedDict[HorizonKey, UsDict] = OrderedDict()
        self._cache_Ys: OrderedDict[HorizonKey, YsDict] = OrderedDict()
        self._cache_Xs: OrderedDict[HorizonKey, XsDict] = OrderedDict()
        self._cache_Fs: OrderedDict[HorizonKey, FsDict] = OrderedDict()
        super().__setattr__("_cache_enabled", True)

    def _clear_dynamic_caches(self) -> None:
        r"""
        Clear caches tied to dynamic (non-structural) algorithm parameters.

        These horizon-indexed caches are rebuilt when attributes such as step
        sizes change: ``_cache_AsBsCsDs``, ``_cache_Ys``, and ``_cache_Xs``.
        """
        if hasattr(self, "_cache_AsBsCsDs"):
            self._cache_AsBsCsDs.clear()
        if hasattr(self, "_cache_Ys"):
            self._cache_Ys.clear()
        if hasattr(self, "_cache_Xs"):
            self._cache_Xs.clear()

    def _clear_all_caches(self) -> None:
        r"""
        Clear every cached artifact after structural updates.

        Structural fields (for example ``n``, ``m``, and index partitions) may
        change matrix dimensions and projection maps. This clears dynamic
        horizon caches, static horizon caches (``_cache_Us``, ``_cache_Fs``),
        and the projection cache ``_Ps_cache``.
        """
        self._clear_dynamic_caches()
        if hasattr(self, "_cache_Us"):
            self._cache_Us.clear()
        if hasattr(self, "_cache_Fs"):
            self._cache_Fs.clear()
        super().__setattr__("_Ps_cache", None)

    def _cache_get(self, cache: OrderedDict[HorizonKey, CacheValueT], key: HorizonKey) -> Optional[CacheValueT]:
        r"""
        Return one cached entry and refresh its LRU recency on hit.

        ``key`` is a horizon key ``(k_min, k_max)``. On a cache hit, the entry
        is moved to the end of the :class:`collections.OrderedDict` so it is
        treated as most recently used. On miss, return ``None``.
        """
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        return None

    def _cache_set(
            self,
            cache: OrderedDict[HorizonKey, CacheValueT],
            key: HorizonKey,
            value: CacheValueT,
    ) -> CacheValueT:
        r"""
        Store one horizon-keyed cache entry with bounded LRU eviction.

        The inserted/updated entry becomes most recently used. If the cache
        exceeds ``_cache_maxsize``, the least recently used entry is evicted.
        Returns ``value`` so call sites can assign and return in one step.
        """
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > self._cache_maxsize:
            cache.popitem(last=False)
        return value

    @staticmethod
    def _readonly_view(array: np.ndarray) -> np.ndarray:
        r"""
        Return a read-only view to discourage accidental mutation of cached matrices.

        **Parameters**

        - `array` (:class:`numpy.ndarray`): Input array to view.

        **Returns**

        - (:class:`numpy.ndarray`): A read-only view of the input array.

        """
        view = array.view()
        view.setflags(write=False)
        return view

    def _readonly_tuple(self, mats: MatrixTuple) -> MatrixTuple:
        r"""Return a tuple of read-only views for `(A, B, C, D)` matrices."""
        A, B, C, D = mats
        return (
            self._readonly_view(A),
            self._readonly_view(B),
            self._readonly_view(C),
            self._readonly_view(D),
        )

    @staticmethod
    def _validate_k_bounds(k_min: int, k_max: int) -> Tuple[int, int]:
        r"""Validate and normalize iteration bounds."""
        k_min = ensure_integral(k_min, "k_min", minimum=0)
        k_max = ensure_integral(k_max, "k_max", minimum=0)
        if k_max < k_min:
            raise ValueError("Require 0 <= k_min <= k_max")
        return k_min, k_max

    @staticmethod
    def _validate_finite_real(value: Any, name: str) -> float:
        r"""Validate a finite real scalar parameter."""
        return ensure_real_number(value, name, finite=True)

    @classmethod
    def _validate_positive_finite_real(cls, value: object, name: str) -> float:
        r"""Validate a strictly positive finite real scalar parameter."""
        value = cls._validate_finite_real(value, name)
        if value <= 0:
            raise ValueError(f"{name} must be > 0.")
        return value

    @staticmethod
    def _validate_nonnegative_integral(value: Any, name: str) -> int:
        r"""Validate a nonnegative integer parameter."""
        return ensure_integral(value, name, minimum=0)

    def _set_dynamic_parameter(self, name: str, value: object) -> None:
        r"""
        Update a dynamic scalar parameter only when its value changes.

        This avoids unnecessary cache invalidation for repeated assignments
        of the same parameter value.
        """
        if hasattr(self, name) and getattr(self, name) == value:
            return
        setattr(self, name, value)

    @staticmethod
    def _validate_pairs_container(pairs: PairList) -> None:
        r"""Validate that `pairs` is a non-empty list/tuple of 2-tuples."""
        if pairs is None or not isinstance(pairs, (list, tuple)) or len(pairs) == 0:
            raise ValueError("Pairs must be a nonempty list of 2-tuples.")
        for pair in pairs:
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise ValueError("Each element in pairs must be a tuple of two elements (j, k).")

    @staticmethod
    def _is_star_pair(j: object, k: object) -> bool:
        r"""Return True when a pair corresponds to the star entry."""
        return j == 'star' and k == 'star'

    @staticmethod
    def _validate_nonstar_pair(
            j: object,
            k: object,
            m_bar_i: int,
            k_min: int,
            k_max: int,
            i: int,
    ) -> Tuple[int, int]:
        r"""Validate and normalize a non-star pair `(j, k)`."""
        if isinstance(j, bool) or isinstance(k, bool) or not isinstance(j, Integral) or not isinstance(k, Integral):
            raise ValueError("For non-star pairs, (j, k) must be integers.")
        j = int(j)
        k = int(k)
        if not (1 <= j <= m_bar_i):
            raise ValueError(f"For i = {i}, j must be in [1, {m_bar_i}].")
        if not (k_min <= k <= k_max):
            raise ValueError(f"k must be in [{k_min}, {k_max}].")
        return j, k

    @abstractmethod
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        See :doc:`3. Algorithm representation
        </theory/algorithm_representation>` for notation and representation
        conventions.

        Return the system matrices :math:`(A_k, B_k, C_k, D_k)` at iteration :math:`k`.
        
        **Parameters**

        - `k` (:class:`int`): iteration index :math:`k`.

        **Returns**

        - (:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`\]): A tuple
          :math:`(A_k, B_k, C_k, D_k)` of numpy arrays with

          .. math::
             A_k \in \mathbb{R}^{n \times n}, \quad
             B_k \in \mathbb{R}^{n \times \bar{m}}, \quad
             C_k \in \mathbb{R}^{\bar{m} \times n}, \quad
             D_k \in \mathbb{R}^{\bar{m} \times \bar{m}}.

        """
        pass

    def _get_AsBsCsDs(self, k_min: int, k_max: int) -> AsBsCsDsDict:
        r"""
        Return a dictionary mapping each iteration index :math:`k` to
        :math:`(A_k, B_k, C_k, D_k)` for :math:`k \in \llbracket k_{\textup{min}}, k_{\textup{max}}\rrbracket`.
        In compact form:

        .. math::
           \{(A_k, B_k, C_k, D_k)\}_{k=k_{\textup{min}}}^{k_{\textup{max}}}.

        Each entry is the tuple returned by :meth:`~autolyap.algorithms.Algorithm.get_ABCD`.

        These system matrices are used to build output and state matrices via
        :meth:`~autolyap.algorithms.Algorithm._get_Ys` and :meth:`~autolyap.algorithms.Algorithm._get_Xs`.

        **Parameters**

        - `k_min` (:class:`int`): minimum iteration index :math:`k_{\textup{min}}`.
        - `k_max` (:class:`int`): maximum iteration index :math:`k_{\textup{max}}`.

        **Returns**

        - (:class:`~typing.Dict`\[:class:`int`, :class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`\]\]): A dictionary whose keys
          are iteration indices and values are tuples :math:`(A_k, B_k, C_k, D_k)`, where

          .. math::
             A_k \in \mathbb{R}^{n \times n}, \quad
             B_k \in \mathbb{R}^{n \times \bar{m}}, \quad
             C_k \in \mathbb{R}^{\bar{m} \times n}, \quad
             D_k \in \mathbb{R}^{\bar{m} \times \bar{m}}.

        **Raises**

        - `ValueError`: if :math:`k_{\textup{min}} < 0` or :math:`k_{\textup{max}} < k_{\textup{min}}`.
        """
        k_min, k_max = self._validate_k_bounds(k_min, k_max)
        # Cache by horizon to reuse repeated calls with the same (k_min, k_max).
        key = (k_min, k_max)
        cached = self._cache_get(self._cache_AsBsCsDs, key)
        if cached is not None:
            return dict(cached)
        sys_mats: AsBsCsDsDict = {}
        for k in range(k_min, k_max + 1):
            mats = self.get_ABCD(k)
            if not (isinstance(mats, tuple) and len(mats) == 4):
                raise ValueError("get_ABCD must return a tuple (A, B, C, D).")
            A, B, C, D = mats
            for name, mat in zip(("A", "B", "C", "D"), (A, B, C, D)):
                if not isinstance(mat, np.ndarray):
                    raise ValueError(f"{name} must be a numpy array.")
                if mat.ndim != 2:
                    raise ValueError(f"{name} must be a 2D numpy array.")
                ensure_finite_array(mat, f"{name}")
            if A.shape != (self.n, self.n):
                raise ValueError("A must have shape (n, n).")
            if B.shape != (self.n, self.m_bar):
                raise ValueError("B must have shape (n, m_bar).")
            if C.shape != (self.m_bar, self.n):
                raise ValueError("C must have shape (m_bar, n).")
            if D.shape != (self.m_bar, self.m_bar):
                raise ValueError("D must have shape (m_bar, m_bar).")
            sys_mats[k] = self._readonly_tuple((A, B, C, D))
        self._cache_set(self._cache_AsBsCsDs, key, sys_mats)
        return dict(sys_mats)

    # --- U MATRICES ---
    def _generate_U(self, k_min: int, k_max: int, k: Optional[int] = None, star: bool = False) -> np.ndarray:
        r"""
        Generate one lifted input matrix.

        See :doc:`3. Algorithm representation </theory/algorithm_representation>`
        for the state-space model :eq:`eq:linear_system_with_nonlinearity`.

        This method implements :eq:`eq:u_mats` and :eq:`eq:u_star` for the
        non-star and star cases, respectively, from
        :doc:`5.1. Performance estimation via SDPs </theory/performance_estimation_via_sdps>`.
        These matrices are then used in
        :doc:`5.2. Iteration-independent analyses </theory/iteration_independent_analyses>`
        and :doc:`5.3. Iteration-dependent analyses </theory/iteration_dependent_analyses>`
        through :meth:`~autolyap.algorithms.Algorithm._compute_E`.

        **Parameters**

        - `k_min` (:class:`int`): minimum iteration index :math:`k_{\textup{min}}`.
        - `k_max` (:class:`int`): maximum iteration index :math:`k_{\textup{max}}`.
        - `k` (:class:`int`): current iteration index :math:`k` (required when `star` is False).
        - `star` (:class:`bool`): if True, generate :math:`U_{\star}`.

        **Returns**

        - (:class:`numpy.ndarray`): The generated matrix
          :math:`U_k^{k_{\textup{min}},k_{\textup{max}}}` or
          :math:`U_{\star}^{k_{\textup{min}},k_{\textup{max}}}` for
          ``star=False`` and ``star=True``, respectively, each with

          .. math::
             n + (k_{\textup{max}} - k_{\textup{min}} + 1)\bar{m} + m

          columns.

        **Raises**

        - `ValueError`: if `k` is missing when `star` is False or if
          :math:`k \notin \llbracket k_{\textup{min}}, k_{\textup{max}}\rrbracket`.
        """
        k_min = ensure_integral(k_min, "k_min", minimum=0)
        k_max = ensure_integral(k_max, "k_max", minimum=0)
        if k_max < k_min:
            raise ValueError("k_max must be >= k_min.")
        total_cols = self.n + ((k_max - k_min + 1) * self.m_bar + self.m)
        if star:
            if self.m > 1:
                N = np.vstack([np.eye(self.m - 1), -np.ones((1, self.m - 1))])
                return np.hstack([
                    np.zeros((self.m, self.n + (k_max - k_min + 1) * self.m_bar)),
                    N,
                    np.zeros((self.m, 1))
                ])
            else:
                return np.zeros((self.m, total_cols))
        else:
            if k is None:
                raise ValueError("When star is False, k must be provided")
            k = ensure_integral(k, "k", minimum=0)
            if k < k_min or k > k_max:
                raise ValueError("k must be between k_min and k_max")
            left = np.zeros((self.m_bar, self.n + (k - k_min) * self.m_bar))
            ident = np.eye(self.m_bar)
            right = np.zeros((self.m_bar, (k_max - k) * self.m_bar + self.m))
            return np.hstack([left, ident, right])

    def _get_Us(self, k_min: int, k_max: int) -> UsDict:
        r"""
        Return a dictionary of U matrices for iterations :math:`k \in \llbracket k_{\textup{min}}, k_{\textup{max}}\rrbracket`,
        including the star matrix.
        In compact form:

        .. math::
           \{U_k^{k_{\textup{min}},k_{\textup{max}}}\}_{k=k_{\textup{min}}}^{k_{\textup{max}}}
           \cup \{U_{\star}^{k_{\textup{min}},k_{\textup{max}}}\}.

        Explicitly,

        .. math::
           U_k^{k_{\textup{min}},k_{\textup{max}}} = \begin{bmatrix}
           \mathbf{0}_{\bar{m} \times \left(n + (k - k_{\textup{min}})\bar{m}\right)} &
           I_{\bar{m}} &
           \mathbf{0}_{\bar{m} \times \left((k_{\textup{max}} - k)\bar{m} + m\right)}
           \end{bmatrix},

        and

        .. math::
           U_{\star}^{k_{\textup{min}},k_{\textup{max}}} = \begin{bmatrix}
           \mathbf{0}_{m \times \left(n + ((k_{\textup{max}} - k_{\textup{min}} + 1)\bar{m})\right)} &
           N &
           \mathbf{0}_{m \times 1}
           \end{bmatrix}, \quad
           N = \begin{bmatrix} I_{m-1} \\ -\mathbf{1}_{1 \times (m-1)} \end{bmatrix},

        with the block column containing :math:`N` removed when :math:`m=1`.

        These matrices are used by :meth:`~autolyap.algorithms.Algorithm._compute_E` (and therefore
        :meth:`~autolyap.algorithms.Algorithm._compute_W`) to assemble lifted constraints.

        **Parameters**

        - `k_min` (:class:`int`): minimum iteration index :math:`k_{\textup{min}}`.
        - `k_max` (:class:`int`): maximum iteration index :math:`k_{\textup{max}}`.

        **Returns**

        - (:class:`~typing.Dict`\[:class:`~typing.Any`, :class:`numpy.ndarray`\]): A dictionary whose keys are iteration indices
          :math:`k \in \llbracket k_{\textup{min}}, k_{\textup{max}}\rrbracket` and `'star'`, with
          values :math:`U_k^{k_{\textup{min}},k_{\textup{max}}}` and
          :math:`U_{\star}^{k_{\textup{min}},k_{\textup{max}}}`, respectively, where

          .. math::
             U_k^{k_{\textup{min}},k_{\textup{max}}}
             \in \mathbb{R}^{\bar{m} \times \left(n + (k_{\textup{max}} - k_{\textup{min}} + 1)\bar{m} + m\right)}, \quad
             U_{\star}^{k_{\textup{min}},k_{\textup{max}}}
             \in \mathbb{R}^{m \times \left(n + (k_{\textup{max}} - k_{\textup{min}} + 1)\bar{m} + m\right)}.

        **Raises**

        - `ValueError`: if :math:`k_{\textup{min}} < 0` or :math:`k_{\textup{max}} < k_{\textup{min}}`.
        """
        k_min, k_max = self._validate_k_bounds(k_min, k_max)
        key = (k_min, k_max)
        cached = self._cache_get(self._cache_Us, key)
        if cached is not None:
            return dict(cached)
        Us: UsDict = {}
        for k in range(k_min, k_max + 1):
            Us[k] = self._readonly_view(self._generate_U(k_min, k_max, k=k, star=False))
        Us['star'] = self._readonly_view(self._generate_U(k_min, k_max, star=True))
        self._cache_set(self._cache_Us, key, Us)
        return dict(Us)

    # --- Y MATRICES ---
    def _generate_Y(self,
                    sys_mats: AsBsCsDsDict,
                    k_min: int, k_max: int, k: Optional[int] = None, star: bool = False) -> np.ndarray:
        r"""
        Generate one lifted output matrix.

        See :doc:`3. Algorithm representation </theory/algorithm_representation>`
        for :eq:`eq:linear_system_with_nonlinearity` and :eq:`eq:abcd`.

        This method implements :eq:`eq:y_mats` and :eq:`eq:y_star` for the
        non-star and star cases, respectively, from
        :doc:`5.1. Performance estimation via SDPs </theory/performance_estimation_via_sdps>`.
        These matrices are then used in
        :doc:`5.2. Iteration-independent analyses </theory/iteration_independent_analyses>`
        and :doc:`5.3. Iteration-dependent analyses </theory/iteration_dependent_analyses>`
        through :meth:`~autolyap.algorithms.Algorithm._compute_E`.

        **Parameters**

        - `sys_mats` (:class:`~typing.Dict`\[:class:`int`, :class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`\]\]): dictionary
          mapping each iteration index :math:`k` to :math:`(A_k, B_k, C_k, D_k)`.
        - `k_min` (:class:`int`): minimum iteration index :math:`k_{\textup{min}}`.
        - `k_max` (:class:`int`): maximum iteration index :math:`k_{\textup{max}}`.
        - `k` (:class:`int`): current iteration index :math:`k` (required when `star` is False).
        - `star` (:class:`bool`): if True, generate :math:`Y_{\star}`.

        **Returns**

        - (:class:`numpy.ndarray`): The generated matrix
          :math:`Y_k^{k_{\textup{min}},k_{\textup{max}}}` or
          :math:`Y_{\star}^{k_{\textup{min}},k_{\textup{max}}}` for
          ``star=False`` and ``star=True``, respectively, each with

          .. math::
             n + (k_{\textup{max}} - k_{\textup{min}} + 1)\bar{m} + m

          columns.

        **Raises**

        - `ValueError`: if `k` is missing when `star` is False.
        """
        k_min = ensure_integral(k_min, "k_min", minimum=0)
        k_max = ensure_integral(k_max, "k_max", minimum=0)
        if k_max < k_min:
            raise ValueError("k_max must be >= k_min.")
        total_cols = self.n + (k_max - k_min + 1) * self.m_bar + self.m

        if star:
            return np.hstack([
                np.zeros((self.m, total_cols - 1)),
                np.ones((self.m, 1))
            ])
        if k is None:
            raise ValueError("When star is False, k must be provided")
        k = ensure_integral(k, "k", minimum=0)
        if k < k_min or k > k_max:
            raise ValueError("k must be between k_min and k_max")

        if k == k_min:
            C_kmin = sys_mats[k_min][2]
            D_kmin = sys_mats[k_min][3]
            zeros_blk = np.zeros((self.m_bar, (k_max - k_min) * self.m_bar + self.m))
            return np.hstack([C_kmin, D_kmin, zeros_blk])
        elif k == k_min + 1:
            C_next = sys_mats[k][2]
            A_kmin = sys_mats[k_min][0]
            B_kmin = sys_mats[k_min][1]
            D_next = sys_mats[k][3]
            zeros_blk = np.zeros((self.m_bar, (k_max - k_min - 1) * self.m_bar + self.m))
            # Order matches the definition: C_{k}A_{k_min}, C_{k}B_{k_min}, then D_k.
            block1 = C_next @ A_kmin
            block2 = C_next @ B_kmin
            return np.hstack([block1, block2, D_next, zeros_blk])
        else:
            blocks = []
            C_k = sys_mats[k][2]
            # First block: C_k A_{k-1} ... A_{k_min}.
            prod = C_k.copy()
            for i in reversed(range(k_min, k)):
                prod = prod @ sys_mats[i][0]
            blocks.append(prod)
            for j in range(k_min, k - 1):
                prod_B = np.eye(self.n)
                for i in reversed(range(j + 1, k)):
                    prod_B = prod_B @ sys_mats[i][0]
                # Each block aligns with a past input B_j propagated to time k.
                blocks.append(C_k @ prod_B @ sys_mats[j][1])
            blocks.append(C_k @ sys_mats[k - 1][1])
            D_k = sys_mats[k][3]
            blocks.append(D_k)
            zeros_blk = np.zeros((self.m_bar, (k_max - k) * self.m_bar + self.m))
            blocks.append(zeros_blk)
            return np.hstack(blocks)

    def _get_Ys(self, k_min: int, k_max: int) -> YsDict:
        r"""
        Return a dictionary of Y matrices for iterations :math:`k \in \llbracket k_{\textup{min}}, k_{\textup{max}}\rrbracket`,
        including :math:`Y_{\star}`.
        In compact form:

        .. math::
           \{Y_k^{k_{\textup{min}},k_{\textup{max}}}\}_{k=k_{\textup{min}}}^{k_{\textup{max}}}
           \cup \{Y_{\star}^{k_{\textup{min}},k_{\textup{max}}}\}.

        Explicitly,

        .. math::
           Y_{k}^{k_{\textup{min}},k_{\textup{max}}} =
           \begin{cases}
               \begin{bmatrix}
                   C_{k_{\textup{min}}} & D_{k_{\textup{min}}} &
                   0_{\bar{m}\times\left((k_{\textup{max}}-k_{\textup{min}})\bar{m}+m\right)}
               \end{bmatrix}
               & k = k_{\textup{min}}, \\[0.5em]
               \begin{bmatrix}
                   \left(C_{k_{\textup{min}}+1}A_{k_{\textup{min}}}\right)^{\top} \\
                   \left(C_{k_{\textup{min}}+1}B_{k_{\textup{min}}}\right)^{\top} \\
                   D_{k_{\textup{min}}+1}^{\top} \\
                   0_{\bar{m}\times\left((k_{\textup{max}}-k_{\textup{min}}-1)\bar{m}+m\right)}^{\top}
               \end{bmatrix}^{\top}
               & k = k_{\textup{min}} + 1,\quad k_{\textup{min}} + 1 \le k_{\textup{max}},
               \\[0.5em]
               \begin{bmatrix}
                   \left(C_{k}A_{k-1}\cdots A_{k_{\textup{min}}}\right)^{\top} \\
                   \left(C_{k}A_{k-1}\cdots A_{k_{\textup{min}}+1}B_{k_{\textup{min}}}\right)^{\top} \\
                   \left(C_{k}A_{k-1}\cdots A_{k_{\textup{min}}+2}B_{k_{\textup{min}}+1}\right)^{\top} \\
                   \vdots \\
                   \left(C_{k}A_{k-1}B_{k-2}\right)^{\top} \\
                   \left(C_{k}B_{k-1}\right)^{\top} \\
                   D_{k}^{\top} \\
                   0_{\bar{m}\times\left((k_{\textup{max}}-k)\bar{m}+m\right)}^{\top}
               \end{bmatrix}^{\top}
               & k \in \llbracket k_{\textup{min}}+2, k_{\textup{max}}\rrbracket,\quad
                 k_{\textup{min}} + 2 \le k_{\textup{max}},
           \end{cases}

        and

        .. math::
           Y_{\star}^{k_{\textup{min}},k_{\textup{max}}}
           =
           \begin{bmatrix}
               0_{m\times\left(n+(k_{\textup{max}}-k_{\textup{min}}+1)\bar{m}+m-1\right)} & \mathbf{1}_m
           \end{bmatrix}.

        The :math:`Y` matrices are constructed from system matrices returned by
        :meth:`~autolyap.algorithms.Algorithm._get_AsBsCsDs` (which calls :meth:`~autolyap.algorithms.Algorithm.get_ABCD`) and are used by
        :meth:`~autolyap.algorithms.Algorithm._compute_E`.

        **Parameters**

        - `k_min` (:class:`int`): minimum iteration index :math:`k_{\textup{min}}`.
        - `k_max` (:class:`int`): maximum iteration index :math:`k_{\textup{max}}`.

        **Returns**

        - (:class:`~typing.Dict`\[:class:`~typing.Any`, :class:`numpy.ndarray`\]): A dictionary whose keys are iteration indices
          :math:`k \in \llbracket k_{\textup{min}}, k_{\textup{max}}\rrbracket` and `'star'`, with
          values :math:`Y_k^{k_{\textup{min}},k_{\textup{max}}}` and
          :math:`Y_{\star}^{k_{\textup{min}},k_{\textup{max}}}`, respectively, where

          .. math::
             Y_k^{k_{\textup{min}},k_{\textup{max}}}
             \in \mathbb{R}^{\bar{m} \times \left(n + (k_{\textup{max}} - k_{\textup{min}} + 1)\bar{m} + m\right)}, \quad
             Y_{\star}^{k_{\textup{min}},k_{\textup{max}}}
             \in \mathbb{R}^{m \times \left(n + (k_{\textup{max}} - k_{\textup{min}} + 1)\bar{m} + m\right)}.

        **Raises**

        - `ValueError`: if :math:`k_{\textup{min}} < 0` or :math:`k_{\textup{max}} < k_{\textup{min}}`.
        """
        k_min, k_max = self._validate_k_bounds(k_min, k_max)
        key = (k_min, k_max)
        cached = self._cache_get(self._cache_Ys, key)
        if cached is not None:
            return dict(cached)
        sys_mats = self._get_AsBsCsDs(k_min, k_max)
        Ys: YsDict = {}
        for k in range(k_min, k_max + 1):
            Ys[k] = self._readonly_view(self._generate_Y(sys_mats, k_min, k_max, k=k, star=False))
        Ys['star'] = self._readonly_view(self._generate_Y(sys_mats, k_min, k_max, star=True))
        self._cache_set(self._cache_Ys, key, Ys)
        return dict(Ys)

    # --- X MATRICES ---
    def _generate_X_k(self, sys_mats: AsBsCsDsDict,
                        k: int, k_min: int, k_max: int) -> np.ndarray:
        r"""
        Generate one lifted state matrix.

        See :doc:`3. Algorithm representation </theory/algorithm_representation>`
        for :eq:`eq:linear_system_with_nonlinearity` and :eq:`eq:abcd`.

        This method implements :eq:`eq:x_mats` in
        :doc:`5.1. Performance estimation via SDPs </theory/performance_estimation_via_sdps>`
        for

        .. math::
           k \in \llbracket k_{\textup{min}}, k_{\textup{max}} + 1\rrbracket.

        The resulting matrices are used in
        :eq:`eq:iteration_independent_lyapunov:theta1:c1_mat`,
        :eq:`eq:iteration_independent_lyapunov:theta1:c4_mat`, and
        :eq:`eq:iteration_dependent_lyapunov:theta1_mat`, i.e., in
        :doc:`5.2. Iteration-independent analyses </theory/iteration_independent_analyses>`
        and :doc:`5.3. Iteration-dependent analyses </theory/iteration_dependent_analyses>`.

        **Parameters**

        - `sys_mats` (:class:`~typing.Dict`\[:class:`int`, :class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`\]\]): dictionary
          mapping each iteration index :math:`k` to :math:`(A_k, B_k, C_k, D_k)`.
        - `k` (:class:`int`): current iteration index :math:`k`.
        - `k_min` (:class:`int`): minimum iteration index :math:`k_{\textup{min}}`.
        - `k_max` (:class:`int`): maximum iteration index :math:`k_{\textup{max}}`.

        **Returns**

        - (:class:`numpy.ndarray`): The generated matrix
          :math:`X_k^{k_{\textup{min}},k_{\textup{max}}}`.

        **Raises**

        - `ValueError`: if :math:`k \notin \llbracket k_{\textup{min}}, k_{\textup{max}}+1\rrbracket`.
        """
        k_min = ensure_integral(k_min, "k_min", minimum=0)
        k_max = ensure_integral(k_max, "k_max", minimum=0)
        if k_max < k_min:
            raise ValueError("k_max must be >= k_min.")
        k = ensure_integral(k, "k", minimum=0)
        total_cols = self.n + (k_max - k_min + 1) * self.m_bar + self.m

        if k < k_min or k > k_max + 1:
            raise ValueError("k must be in [k_min, k_max+1]")

        if k == k_min:
            return np.hstack([np.eye(self.n), np.zeros((self.n, total_cols - self.n))])
        if k == k_min + 1:
            A, B, _, _ = sys_mats[k_min]
            return np.hstack([A, B, np.zeros((self.n, total_cols - self.n - self.m_bar))])
        
        parts = []
        # First block: A_{k-1} ... A_{k_min}.
        prod = np.eye(self.n)
        for i in reversed(range(k_min, k)):
            prod = prod @ sys_mats[i][0]
        parts.append(prod)
        for j in range(k_min, k - 1):
            prod_B = np.eye(self.n)
            for i in reversed(range(j + 1, k)):
                prod_B = prod_B @ sys_mats[i][0]
            # Each term maps an earlier input B_j to state k.
            parts.append(prod_B @ sys_mats[j][1])
        parts.append(sys_mats[k - 1][1])
        zeros_blk = np.zeros((self.n, (k_max + 1 - k) * self.m_bar + self.m))
        parts.append(zeros_blk)
        return np.hstack(parts)

    def _get_Xs(self, k_min: int, k_max: int) -> XsDict:
        r"""
        Return a dictionary mapping each iteration index :math:`k`
        (for :math:`k \in \llbracket k_{\textup{min}}, k_{\textup{max}} + 1\rrbracket`) to the corresponding
        :math:`X_k` matrix.
        In compact form:

        .. math::
           \{X_k^{k_{\textup{min}},k_{\textup{max}}}\}_{k=k_{\textup{min}}}^{k_{\textup{max}}+1}.

        Explicitly,

        .. math::
           X_{k}^{k_{\textup{min}},k_{\textup{max}}} =
           \begin{cases}
               \begin{bmatrix}
                   I_{n} &
                   0_{n\times\left((k_{\textup{max}}-k_{\textup{min}}+1)\bar{m}+m\right)}
               \end{bmatrix}
               & k = k_{\textup{min}}, \\[0.5em]
               \begin{bmatrix}
                   A_{k_{\textup{min}}} &
                   B_{k_{\textup{min}}} &
                   0_{n\times\left((k_{\textup{max}}-k_{\textup{min}})\bar{m}+m\right)}
               \end{bmatrix}
               & k = k_{\textup{min}} + 1, \\[0.5em]
               \begin{bmatrix}
                   \left(A_{k-1}\cdots A_{k_{\textup{min}}}\right)^{\top} \\
                   \left(A_{k-1}\cdots A_{k_{\textup{min}}+1}B_{k_{\textup{min}}}\right)^{\top} \\
                   \left(A_{k-1}\cdots A_{k_{\textup{min}}+2}B_{k_{\textup{min}}+1}\right)^{\top} \\
                   \vdots \\
                   \left(A_{k-1}A_{k-2}B_{k-3}\right)^{\top} \\
                   \left(A_{k-1}B_{k-2}\right)^{\top} \\
                   B_{k-1}^{\top} \\
                   0_{n\times\left((k_{\textup{max}}+1-k)\bar{m}+m\right)}^{\top}
               \end{bmatrix}^{\top}
               & k \in \llbracket k_{\textup{min}}+2, k_{\textup{max}}+1\rrbracket,\quad
                 k_{\textup{min}} + 1 \le k_{\textup{max}},
           \end{cases}.

        The :math:`X_k` matrices are constructed from system matrices returned by
        :meth:`~autolyap.algorithms.Algorithm._get_AsBsCsDs` (and therefore :meth:`~autolyap.algorithms.Algorithm.get_ABCD`).

        These matrices are used by the fixed-point residual helpers in
        :class:`~autolyap.iteration_independent._SublinearConvergence` and
        :class:`~autolyap.iteration_dependent.IterationDependent` (see
        :meth:`~autolyap.iteration_independent._SublinearConvergence.get_parameters_fixed_point_residual`
        and :meth:`~autolyap.IterationDependent.get_parameters_fixed_point_residual`,
        respectively).

        **Parameters**

        - `k_min` (:class:`int`): minimum iteration index :math:`k_{\textup{min}}`.
        - `k_max` (:class:`int`): maximum iteration index :math:`k_{\textup{max}}`.

        **Returns**

        - (:class:`~typing.Dict`\[:class:`int`, :class:`numpy.ndarray`\]): A dictionary mapping
          :math:`k \in \llbracket k_{\textup{min}}, k_{\textup{max}}+1\rrbracket` to
          :math:`X_k^{k_{\textup{min}},k_{\textup{max}}}`, where

          .. math::
             X_k^{k_{\textup{min}},k_{\textup{max}}}
             \in \mathbb{R}^{n \times \left(n + (k_{\textup{max}} - k_{\textup{min}} + 1)\bar{m} + m\right)}.

        **Raises**

        - `ValueError`: if :math:`k_{\textup{min}} < 0` or :math:`k_{\textup{max}} < k_{\textup{min}}`.
        """
        k_min, k_max = self._validate_k_bounds(k_min, k_max)
        key = (k_min, k_max)
        cached = self._cache_get(self._cache_Xs, key)
        if cached is not None:
            return dict(cached)
        sys_mats = self._get_AsBsCsDs(k_min, k_max)
        Xs: XsDict = {}
        for k in range(k_min, k_max + 2):
            Xs[k] = self._readonly_view(self._generate_X_k(sys_mats, k, k_min, k_max))
        self._cache_set(self._cache_Xs, key, Xs)
        return dict(Xs)

    # --- PROJECTION MATRICES (P) ---
    def _get_Ps(self) -> PsDict:
        r"""
        Return a dictionary of projection matrices :math:`P`.
        In compact form:

        .. math::
           \{P_{(i,j)}\}_{j \in \llbracket 1,\NumEval_i\rrbracket,i \in \llbracket 1,m\rrbracket}
           \cup \{P_{(i,\star)}\}_{i \in \llbracket 1,m\rrbracket}.

        Explicitly, for each component index :math:`i \in \llbracket 1, m\rrbracket` and each
        evaluation index :math:`j \in \llbracket 1, \NumEval_i\rrbracket`, the projection matrix
        :math:`P_{(i,j)}` is a :math:`1 \times \NumEval` row vector with a 1 at the (offset+j)-th
        position, where

        .. math::
           \text{offset} = \sum_{l=1}^{i-1} \NumEval_l.

        Equivalently,

        .. math::
           P_{(i,j)} = \begin{bmatrix}
           \mathbf{0}_{1 \times \sum_{r=1}^{i-1} \NumEval_r} &
           (e_j^{\NumEval_i})^\top &
           \mathbf{0}_{1 \times \sum_{r=i+1}^{m} \NumEval_r}
           \end{bmatrix}.

        Here :math:`e_i^{p}` denotes the :math:`i`-th standard basis vector in :math:`\mathbb{R}^{p}`.

        Additionally, :math:`P_{(i,\star)}` is a :math:`1 \times m` row vector with a 1 in the i-th position.

        .. math::
           P_{(i,\star)} = (e_i^{m})^\top.

        These projection matrices are used by :meth:`~autolyap.algorithms.Algorithm._compute_E` and by the metric
        builders in :class:`~autolyap.iteration_independent._LinearConvergence`,
        :class:`~autolyap.iteration_independent._SublinearConvergence`, and
        :class:`~autolyap.iteration_dependent.IterationDependent`.

        **Returns**

        - (:class:`~typing.Dict`\[:class:`~typing.Tuple`\[:class:`~typing.Any`, :class:`~typing.Any`\], :class:`numpy.ndarray`\]): A dictionary whose keys are tuples
          :math:`(i, j)` with :math:`i \in \llbracket 1,m\rrbracket`,
          :math:`j \in \llbracket 1,\NumEval_i\rrbracket`, and :math:`(i,\star)`, with values
          :math:`P_{(i,j)}` and :math:`P_{(i,\star)}`, where

          .. math::
             P_{(i,j)} \in \mathbb{R}^{1 \times \NumEval}, \quad
             P_{(i,\star)} \in \mathbb{R}^{1 \times m}.

        """
        # Cache projection matrices since they are structural (independent of horizons and step sizes).
        if self._Ps_cache is not None:
            return dict(self._Ps_cache)
        Ps: PsDict = {}
        for i in range(1, self.m + 1):
            offset = int(self._m_bar_offsets[i - 1])
            for j in range(1, self.m_bar_is[i - 1] + 1):
                vec = np.zeros((1, self.m_bar))
                vec[0, offset + j - 1] = 1
                Ps[(i, j)] = self._readonly_view(vec)
            star_vec = np.zeros((1, self.m))
            star_vec[0, i - 1] = 1
            Ps[(i, 'star')] = self._readonly_view(star_vec)
        self._Ps_cache = Ps
        return dict(Ps)

    # --- F MATRICES (for functional components) ---
    def _generate_F(self, i: int, j: Optional[int] = None, k: Optional[int] = None,
                    star: bool = False, k_min: int = 0, k_max: int = 0) -> np.ndarray:
        r"""
        Generate one lifted function-value selector row.

        See :doc:`3. Algorithm representation </theory/algorithm_representation>`
        for notation.

        This method implements :eq:`eq:f_mats` from
        :doc:`5.1. Performance estimation via SDPs </theory/performance_estimation_via_sdps>`.
        Its outputs are combined into :eq:`eq:f_func_ineq` and :eq:`eq:f_func_eq`.
        These appear in both
        :ref:`(5.28) <eq:iteration_independent_lyapunov:condition>` and
        :ref:`(5.51) <eq:iteration_dependent_lyapunov:condition>` from
        :doc:`5.2. Iteration-independent analyses </theory/iteration_independent_analyses>`
        and :doc:`5.3. Iteration-dependent analyses </theory/iteration_dependent_analyses>`,
        respectively.

        **Parameters**

        - `i` (:class:`int`): functional component index :math:`i` (must satisfy :math:`i \in \IndexFunc`).
        - `j` (:class:`~typing.Optional`\[:class:`int`\]): evaluation index :math:`j` for the non-star case.
        - `k` (:class:`~typing.Optional`\[:class:`int`\]): iteration index :math:`k` for the non-star case (must satisfy
          :math:`k \in \llbracket k_{\textup{min}}, k_{\textup{max}}\rrbracket`).
        - `star` (:class:`bool`): if True, generate the star row.
        - `k_min` (:class:`int`): minimum iteration index :math:`k_{\textup{min}}`.
        - `k_max` (:class:`int`): maximum iteration index :math:`k_{\textup{max}}`.

        **Returns**

        - (:class:`numpy.ndarray`): A :math:`1 \times \text{total_dim}` numpy array representing the generated F row.

        **Raises**

        - `ValueError`: if the indices are invalid or inconsistent with the star/non-star case.
        """
        k_min = ensure_integral(k_min, "k_min", minimum=0)
        k_max = ensure_integral(k_max, "k_max", minimum=0)
        if k_max < k_min:
            raise ValueError("k_max must be >= k_min.")
        total_dim = ((k_max - k_min + 1) * self.m_bar_func + self.m_func)

        if not self.I_func or i not in self.I_func:
            raise ValueError(f"i must be in I_func. Got i = {i}")
        idx = self.kappa[i]  # 1-indexed

        if star:
            if j is not None or k is not None:
                raise ValueError("For star F matrices, do not supply j or k")
            F_star = np.zeros((1, total_dim))
            F_star[0, (k_max - k_min + 1) * self.m_bar_func + idx - 1] = 1
            return F_star
        else:
            if j is None or k is None:
                raise ValueError("For non-star F matrices, both j and k must be provided")
            k = ensure_integral(k, "k", minimum=0)
            if not (k_min <= k <= k_max):
                raise ValueError(f"k must be in [{k_min}, {k_max}]. Got k = {k}")
            max_j = self.m_bar_is[i - 1]
            j = ensure_integral(j, "j", minimum=1)
            if not (1 <= j <= max_j):
                raise ValueError(f"For i = {i}, j must be in [1, {max_j}]. Got j = {j}")
            F_nonstar = np.zeros((1, total_dim))
            offset = self._func_offsets[i]
            start_idx = self.m_bar_func * (k - k_min) + offset
            F_nonstar[0, start_idx + j - 1] = 1
            return F_nonstar

    def _get_Fs(self, k_min: int, k_max: int) -> FsDict:
        r"""
        Return a dictionary of F matrices for all functional components.
        In compact form:

        .. math::
           \{F_{(i,j,k)}^{k_{\textup{min}},k_{\textup{max}}}\}_{i\in\IndexFunc,\, j\in\llbracket 1,\bar{m}_i\rrbracket,\, k\in\llbracket k_{\textup{min}}, k_{\textup{max}}\rrbracket}
           \cup \{F_{(i,\star,\star)}^{k_{\textup{min}},k_{\textup{max}}}\}_{i\in\IndexFunc}.

        Explicitly, letting

        .. math::
           d = (k_{\textup{max}} - k_{\textup{min}} + 1)\bar{m}_{\text{func}} + m_{\text{func}},

        and

        .. math::
           \textup{offset}_i = \sum_{\substack{l \in \IndexFunc\\ l < i}} \bar{m}_l,

        we have

        .. math::
           F_{(i,j,k)}^{k_{\textup{min}},k_{\textup{max}}}
           = \left(e_{(k-k_{\textup{min}})\bar{m}_{\text{func}} + \textup{offset}_i + j}^{d}\right)^\top,
           \quad
           F_{(i,\star,\star)}^{k_{\textup{min}},k_{\textup{max}}}
           = \left(e_{(k_{\textup{max}}-k_{\textup{min}}+1)\bar{m}_{\text{func}} + \kappa(i)}^{d}\right)^\top.

        Here :math:`e_i^{p}` denotes the :math:`i`-th standard basis vector in :math:`\mathbb{R}^{p}`, and
        :math:`\kappa:\IndexFunc\to\llbracket 1,m_{\text{func}}\rrbracket` is the increasing bijection used
        to order functional components.

        The dictionary keys are defined as follows. For non-star F matrices, keys are of the form
        :math:`(i, j, k)` with :math:`i \in \IndexFunc`, :math:`j \in \llbracket 1,\bar{m}_i\rrbracket`, and
        :math:`k \in \llbracket k_{\textup{min}}, k_{\textup{max}}\rrbracket`. For star F matrices, keys are of the form
        :math:`(i,\star,\star)` for each :math:`i \in \IndexFunc`.

        These F matrices are used by :meth:`~autolyap.algorithms.Algorithm._compute_F_aggregated` and by the
        function-value suboptimality helpers in
        :class:`~autolyap.iteration_independent._LinearConvergence`,
        :class:`~autolyap.iteration_independent._SublinearConvergence`, and
        :class:`~autolyap.iteration_dependent.IterationDependent`.

        **Parameters**

        - `k_min` (:class:`int`): minimum iteration index :math:`k_{\textup{min}}`.
        - `k_max` (:class:`int`): maximum iteration index :math:`k_{\textup{max}}`.

        **Returns**

        - (:class:`~typing.Dict`\[:class:`~typing.Tuple`\[:class:`~typing.Any`, :class:`~typing.Any`, :class:`~typing.Any`\], :class:`numpy.ndarray`\]): A dictionary mapping keys to the corresponding
          F row matrices :math:`F_{(i,j,k)}^{k_{\textup{min}},k_{\textup{max}}}` and
          :math:`F_{(i,\star,\star)}^{k_{\textup{min}},k_{\textup{max}}}`, where

          .. math::
             F_{(i,j,k)}^{k_{\textup{min}},k_{\textup{max}}},
             F_{(i,\star,\star)}^{k_{\textup{min}},k_{\textup{max}}}
             \in \mathbb{R}^{1 \times d}, \quad
             d = (k_{\textup{max}} - k_{\textup{min}} + 1)\bar{m}_{\text{func}} + m_{\text{func}}.

        **Raises**

        - `ValueError`: if :math:`k_{\textup{min}} < 0` or :math:`k_{\textup{max}} < k_{\textup{min}}`.
        """
        k_min = ensure_integral(k_min, "k_min", minimum=0)
        k_max = ensure_integral(k_max, "k_max", minimum=0)
        if k_max < k_min:
            raise ValueError("Require 0 <= k_min <= k_max")
        key = (k_min, k_max)
        cached = self._cache_get(self._cache_Fs, key)
        if cached is not None:
            return dict(cached)
        Fs: FsDict = {}
        for k in range(k_min, k_max + 1):
            for i in self.I_func:
                for j in range(1, self.m_bar_is[i - 1] + 1):
                    Fs[(i, j, k)] = self._readonly_view(self._generate_F(i, j, k, star=False,
                                                      k_min=k_min, k_max=k_max))
        for i in self.I_func:
            Fs[(i, 'star', 'star')] = self._readonly_view(self._generate_F(i, star=True,
                                                         k_min=k_min, k_max=k_max))
        self._cache_set(self._cache_Fs, key, Fs)
        return dict(Fs)

    # --- LIFTED CONSTRAINT MATRICES ---
    def _compute_E(self, i: int, pairs: PairList,
                  k_min: int, k_max: int, validate: bool = True) -> np.ndarray:
        r"""
        Compute the lifted matrix :math:`E` for one component and a list of pairs.

        See :doc:`3. Algorithm representation </theory/algorithm_representation>`
        for notation.

        This method implements :eq:`eq:e` from
        :doc:`5.1. Performance estimation via SDPs </theory/performance_estimation_via_sdps>`,
        using :eq:`eq:p_mats` together with :eq:`eq:y_mats`/ :eq:`eq:y_star`
        and :eq:`eq:u_mats`/ :eq:`eq:u_star` for non-star/star pairs,
        respectively.
        The resulting :math:`E` is used in :eq:`eq:w_func_ineq`,
        :eq:`eq:w_func_eq`, and :eq:`eq:w_op` for the func-ineq, func-eq,
        and op cases, respectively, and therefore in
        :ref:`(5.28) <eq:iteration_independent_lyapunov:condition>` and
        :ref:`(5.51) <eq:iteration_dependent_lyapunov:condition>` from
        :doc:`5.2. Iteration-independent analyses </theory/iteration_independent_analyses>`
        and :doc:`5.3. Iteration-dependent analyses </theory/iteration_dependent_analyses>`,
        respectively.

        The pair ordering in ``pairs`` is preserved in the row stacking.

        **Parameters**

        - `i` (:class:`int`): component index :math:`i`.
        - `pairs` (:class:`~typing.List`\[:class:`~typing.Union`\[:class:`~typing.Tuple`\[:class:`int`, :class:`int`\], :class:`~typing.Tuple`\[:class:`str`, :class:`str`\]\]\]): list of :math:`(j, k)` pairs. For non-star pairs,
          :math:`j \in \llbracket 1,\bar{m}_i\rrbracket` and :math:`k \in \llbracket k_{\textup{min}}, k_{\textup{max}}\rrbracket`.
          For the star case, use `('star', 'star')`.
        - `k_min` (:class:`int`): minimum iteration index :math:`k_{\textup{min}}`.
        - `k_max` (:class:`int`): maximum iteration index :math:`k_{\textup{max}}`.
        - `validate` (:class:`bool`): Whether to validate inputs.

        **Returns**

        - (:class:`numpy.ndarray`): The computed E matrix.

        **Raises**

        - `ValueError`: if the pairs are malformed or the iteration bounds are invalid.
        """
        if validate:
            i = ensure_integral(i, "i", minimum=1)
            if i > self.m:
                raise ValueError(f"Component index i must be in [1, {self.m}]. Got i = {i}.")
            self._validate_pairs_container(pairs)
            k_min, k_max = self._validate_k_bounds(k_min, k_max)

        Ys = self._get_Ys(k_min, k_max)
        Us = self._get_Us(k_min, k_max)
        total_cols = self.n + (k_max - k_min + 1) * self.m_bar + self.m
        num_pairs = len(pairs)
        E = np.empty((2 * num_pairs, total_cols))

        m_bar_i = self.m_bar_is[i - 1]
        offset = int(self._m_bar_offsets[i - 1])
        star_row = i - 1
        for idx, (j, k) in enumerate(pairs):
            if self._is_star_pair(j, k):
                E[idx] = Ys['star'][star_row]
                E[idx + num_pairs] = Us['star'][star_row]
                continue
            if validate:
                j, k = self._validate_nonstar_pair(j, k, m_bar_i, k_min, k_max, i)
            row_idx = offset + int(j) - 1
            E[idx] = Ys[int(k)][row_idx]
            E[idx + num_pairs] = Us[int(k)][row_idx]

        return E

    def _compute_W(self, i: int, pairs: PairList,
                  k_min: int, k_max: int, M: np.ndarray, validate: bool = True) -> np.ndarray:
        r"""
        Compute one lifted quadratic constraint matrix :math:`W`.

        See :doc:`3. Algorithm representation </theory/algorithm_representation>`
        for notation.

        This method implements the matrix definitions
        :eq:`eq:w_func_ineq`, :eq:`eq:w_func_eq`, and :eq:`eq:w_op` for the
        func-ineq, func-eq, and op cases, respectively, from
        :doc:`5.1. Performance estimation via SDPs </theory/performance_estimation_via_sdps>`
        using :math:`W = E^\top M E`.

        The resulting matrices are the building blocks in
        :ref:`(5.28) <eq:iteration_independent_lyapunov:condition>` and
        :ref:`(5.51) <eq:iteration_dependent_lyapunov:condition>` from
        :doc:`5.2. Iteration-independent analyses </theory/iteration_independent_analyses>`
        and :doc:`5.3. Iteration-dependent analyses </theory/iteration_dependent_analyses>`,
        respectively.

        **Parameters**

        - `i` (:class:`int`): component index :math:`i`.
        - `pairs` (:class:`~typing.List`\[:class:`~typing.Union`\[:class:`~typing.Tuple`\[:class:`int`, :class:`int`\], :class:`~typing.Tuple`\[:class:`str`, :class:`str`\]\]\]): list of :math:`(j, k)` pairs. For non-star pairs,
          :math:`j \in \llbracket 1,\bar{m}_i\rrbracket` and :math:`k \in \llbracket k_{\textup{min}}, k_{\textup{max}}\rrbracket`.
          For the star case, use `('star', 'star')`.
        - `k_min` (:class:`int`): minimum iteration index :math:`k_{\textup{min}}`.
        - `k_max` (:class:`int`): maximum iteration index :math:`k_{\textup{max}}`.
        - `M` (:class:`numpy.ndarray`): symmetric matrix :math:`M` with shape
          :math:`[2 \cdot (\text{number of pairs}) \times 2 \cdot (\text{number of pairs})]`.
          This comes from the interpolation data of the function/operator class
          (e.g., :math:`M_{(i,o)}^{\textup{func-ineq}}`, :math:`M_{(i,o)}^{\textup{func-eq}}`, or
          :math:`M_{(i,o)}^{\textup{op}}`) for ``func-ineq``, ``func-eq``, and
          ``op`` constraints, respectively. See the
          interpolation-condition
          :meth:`~autolyap.problemclass.base._InterpolationCondition.get_data`
          contract or
          :meth:`~autolyap.problemclass.InclusionProblem._get_component_data`.

        - `validate` (:class:`bool`): Whether to validate inputs.

        **Returns**

        - (:class:`numpy.ndarray`): The computed W matrix.

        **Raises**

        - `ValueError`: if any input conditions are not met.
        """
        if validate:
            i = ensure_integral(i, "i", minimum=1)
            if i > self.m:
                raise ValueError(f"Component index i must be in [1, {self.m}]. Got i = {i}.")
            self._validate_pairs_container(pairs)
            k_min, k_max = self._validate_k_bounds(k_min, k_max)
            if not isinstance(M, np.ndarray):
                raise ValueError("M must be a numpy array.")
            exp_dim = 2 * len(pairs)
            if M.ndim != 2 or M.shape != (exp_dim, exp_dim):
                raise ValueError(f"M must be a square matrix of dimension [{exp_dim} x {exp_dim}].")
            ensure_finite_array(M, "M")
            if not np.allclose(M, M.T, atol=1e-8):
                raise ValueError("M must be symmetric.")

        E = self._compute_E(i, pairs, k_min, k_max, validate=validate)
        return E.T @ M @ E

    def _compute_F_aggregated(self, i: int, pairs: PairList,
                             k_min: int, k_max: int, a: np.ndarray, validate: bool = True) -> np.ndarray:
        r"""
        Compute one aggregated lifted vector :math:`F`.

        See :doc:`3. Algorithm representation </theory/algorithm_representation>`
        for notation.

        This method implements :eq:`eq:f_func_ineq` and :eq:`eq:f_func_eq`
        for the func-ineq and func-eq cases, respectively,
        from :doc:`5.1. Performance estimation via SDPs </theory/performance_estimation_via_sdps>`
        by stacking selected rows from :meth:`~autolyap.algorithms.Algorithm._get_Fs`
        and weighting them with ``a``.

        The resulting vectors are the linear terms used in
        :ref:`(5.28) <eq:iteration_independent_lyapunov:condition>` and
        :ref:`(5.51) <eq:iteration_dependent_lyapunov:condition>` from
        :doc:`5.2. Iteration-independent analyses </theory/iteration_independent_analyses>`
        and :doc:`5.3. Iteration-dependent analyses </theory/iteration_dependent_analyses>`,
        respectively.

        **Parameters**

        - `i` (:class:`int`): component index :math:`i` (must satisfy :math:`i \in \IndexFunc`).
        - `pairs` (:class:`~typing.List`\[:class:`~typing.Union`\[:class:`~typing.Tuple`\[:class:`int`, :class:`int`\], :class:`~typing.Tuple`\[:class:`str`, :class:`str`\]\]\]): list of :math:`(j, k)` pairs. For non-star pairs,
          :math:`j \in \llbracket 1,\bar{m}_i\rrbracket` and :math:`k \in \llbracket k_{\textup{min}}, k_{\textup{max}}\rrbracket`.
          For the star case, use `('star', 'star')`.
        - `k_min` (:class:`int`): minimum iteration index :math:`k_{\textup{min}}`.
        - `k_max` (:class:`int`): maximum iteration index :math:`k_{\textup{max}}`.
        - `a` (:class:`numpy.ndarray`): weight vector :math:`a \in \mathbb{R}^{\text{number of pairs}}`.
          This comes from the functional interpolation data
          (e.g., :math:`a_{(i,o)}^{\textup{func-ineq}}` or :math:`a_{(i,o)}^{\textup{func-eq}}`)
          for ``func-ineq`` and ``func-eq`` constraints, respectively. See the
          functional interpolation-condition
          :meth:`~autolyap.problemclass.base._FunctionInterpolationCondition.get_data`
          contract or
          :meth:`~autolyap.problemclass.InclusionProblem._get_component_data`.

        - `validate` (:class:`bool`): Whether to validate inputs.

        **Returns**

        - (:class:`numpy.ndarray`): The aggregated F vector as a column vector.

        **Raises**

        - `ValueError`: if any input conditions are not met.
        """
        if validate:
            i = ensure_integral(i, "i", minimum=1)
            if i not in self.I_func:
                raise ValueError(f"Component index i must be in I_func. Got i = {i}.")
            k_min, k_max = self._validate_k_bounds(k_min, k_max)
        total_dim = (k_max - k_min + 1) * self.m_bar_func + self.m_func

        if validate:
            self._validate_pairs_container(pairs)
            m_bar_i = self.m_bar_is[i - 1]
            for pair in pairs:
                if not (isinstance(pair, tuple) and len(pair) == 2):
                    raise ValueError("Each element in pairs must be a tuple of two elements (j, k).")
                j, k = pair
                if (j == 'star' or k == 'star') and not self._is_star_pair(j, k):
                    raise ValueError("For star F matrices, both j and k must be 'star'.")
                if j != 'star' and k != 'star':
                    if isinstance(j, bool) or isinstance(k, bool) or not isinstance(j, Integral) or not isinstance(k, Integral):
                        raise ValueError("For non-star F matrices, both j and k must be integers.")
                    j = int(j)
                    k = int(k)
                    if not (k_min <= k <= k_max):
                        raise ValueError(f"k must be in [{k_min}, {k_max}]. Got k = {k}.")
                    if not (1 <= j <= m_bar_i):
                        raise ValueError(f"For component i = {i}, j must be in [1, {m_bar_i}]. Got j = {j}.")
            if not (isinstance(a, np.ndarray) and a.ndim == 1 and len(a) == len(pairs)):
                raise ValueError("a must be a 1D numpy array with length equal to the number of pairs.")
            ensure_finite_array(a, "a")

        Fs_dict = self._get_Fs(k_min, k_max)
        aggregated_F = np.zeros((total_dim, 1))
        for weight, (j, k) in zip(a, pairs):
            key = (i, 'star', 'star') if self._is_star_pair(j, k) else (i, j, k)
            if key not in Fs_dict:
                raise ValueError(f"Key {key} not found in F matrices.")
            aggregated_F += Fs_dict[key].T * weight
        return aggregated_F
