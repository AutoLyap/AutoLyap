import numpy as np
from typing import Type, Optional, Tuple, Union, List, Dict
from itertools import combinations
from mosek.fusion import Model, Domain, OptimizeError, Expr
import mosek.fusion.pythonic
from autolyap.utils.helper_functions import create_symmetric_matrix_expression
from autolyap.utils.validation import (
    ensure_finite_array,
    ensure_integral,
    ensure_real_number,
)
from autolyap.problemclass import InclusionProblem
from autolyap.algorithms import Algorithm

class LinearConvergence:
    r"""
    Class-level namespace for linear-convergence tools in iteration-independent analysis.

    **Scope**
    
    - Static constructors for linear-convergence metrics.
    - Bisection utility
      :meth:`~autolyap.iteration_independent.LinearConvergence.bisection_search_rho`.

    Method-level docstrings provide full API details.
    """
    @staticmethod
    def get_parameters_distance_to_solution(
            algo: Type[Algorithm], 
            h: int = 0, 
            alpha: int = 0,
            i: int = 1, 
            j: int = 1, 
            tau: int = 0
        ) -> Union[Tuple[np.ndarray, np.ndarray],
                   Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
                   ]:
        r"""
        Compute matrices for the distance-to-solution metric.

        Mathematical notation and shared definitions follow the class-level
        reference in :class:`~autolyap.iteration_independent.IterationIndependent`.

        This method computes the matrix

        .. math::
            P = \left( P_{(i,j)}\, Y_\tau^{0,h} - P_{(i,\star)}\, Y_\star^{0,h} \right)^{\top}
                \left( P_{(i,j)}\, Y_\tau^{0,h} - P_{(i,\star)}\, Y_\star^{0,h} \right),

        and sets :math:`T` (and, if functional components exist, the vectors :math:`p` and :math:`t`) to zero.

        **Definitions**

        - :math:`Y_\tau^{0,h}` is the :math:`Y` matrix at iteration :math:`\tau` over :math:`\llbracket 0, h\rrbracket`,
          retrieved via :meth:`~autolyap.algorithms.Algorithm.get_Ys` with `k_min = 0` and `k_max = h`.
        - :math:`Y_\star^{0,h}` is the “star” :math:`Y` matrix over :math:`\llbracket 0, h\rrbracket`,
          also returned by :meth:`~autolyap.algorithms.Algorithm.get_Ys`.
        - :math:`P_{(i,j)}` and :math:`P_{(i,\star)}` are the projection matrices for component :math:`i`,
          returned by :meth:`~autolyap.algorithms.Algorithm.get_Ps`.

        **Resulting lower bounds**

        With this choice of :math:`(P,p,T,t)`,

        .. math::
            \begin{aligned}
            \mathcal{V}(P,p,k) &= \|y_{i,j}^{k+\tau} - y^\star\|^2,\\
            \mathcal{R}(T,t,k) &= 0.
            \end{aligned}

        **Parameters**
        
        - `algo` (:class:`~typing.Type`\[:class:`~autolyap.algorithms.Algorithm`\]): An instance of :class:`~autolyap.algorithms.Algorithm`. It must
          provide `algo.m`, `algo.m_bar_is`, and the methods
          :meth:`~autolyap.algorithms.Algorithm.get_Ys` and
          :meth:`~autolyap.algorithms.Algorithm.get_Ps`.
        - `h` (:class:`int`): A nonnegative integer corresponding to :math:`h` defining the time horizon
          :math:`\llbracket 0, h\rrbracket`
          for :math:`Y` matrices.
        - `alpha` (:class:`int`): A nonnegative integer corresponding to :math:`\alpha` for extending the horizon
          for :math:`T` (and :math:`t`).
        - `i` (:class:`int`): Component index (1-indexed) corresponding to :math:`i`. Default is 1; must satisfy
          :math:`i \in \llbracket 1, m\rrbracket`, where `m = algo.m`.
        - `j` (:class:`int`): Evaluation index for component `i` corresponding to :math:`j`. Default is 1; must satisfy
          :math:`j \in \llbracket 1, \NumEval_i\rrbracket`, where :math:`\NumEval_i` is given by
          `algo.m_bar_is[i-1]`.
        - `tau` (:class:`int`): Iteration index corresponding to :math:`\tau`. Default is 0; must satisfy
          :math:`\tau \in \llbracket 0, h\rrbracket`.

        **Returns**
        
        - (:class:`~typing.Union`\[:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`\], :class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`\]\]):
          If `algo.m_func == 0`, returns `(P, T)` with

          .. math::
              \begin{aligned}
              P &\in \sym^{n + (h+1)\NumEval + m},\\
              T &\in \sym^{n + (h+\alpha+2)\NumEval + m}.
              \end{aligned}

          Otherwise, returns `(P, p, T, t)`, where :math:`P` is computed as above and
          :math:`T`, :math:`p`, and :math:`t` are zero arrays with

          .. math::
              \begin{aligned}
              P &\in \sym^{n + (h+1)\NumEval + m},\\
              T &\in \sym^{n + (h+\alpha+2)\NumEval + m},\\
              p &\in \mathbb{R}^{(h+1)\NumEvalFunc + \NumFunc},\\
              t &\in \mathbb{R}^{(h+\alpha+2)\NumEvalFunc + \NumFunc}.
              \end{aligned}

        **Raises**
        
        - `ValueError`: If any input is out of its valid range or if required matrices are missing.
        """
        # ----- Input Checking -----
        h = ensure_integral(h, "h", minimum=0)
        alpha = ensure_integral(alpha, "alpha", minimum=0)

        i = ensure_integral(i, "i", minimum=1)
        if i > algo.m:
            raise ValueError(f"Component index i must be in [1, {algo.m}]. Got {i}.")

        num_eval = algo.m_bar_is[i - 1]
        j = ensure_integral(j, "j", minimum=1)
        if j > num_eval:
            raise ValueError(f"For component {i}, evaluation index j must be in [1, {num_eval}]. Got {j}.")

        tau = ensure_integral(tau, "tau", minimum=0)
        if tau > h:
            raise ValueError(f"Iteration index tau must be in [0, {h}]. Got {tau}.")

        # ----- Dimensions for P and T -----
        n = algo.n            # State dimension.
        m = algo.m            # Total number of components.
        m_bar = algo.m_bar    # Total evaluations per iteration.
        
        # Dimension of P: n + (h+1)*m_bar + m.
        dim_P = n + (h + 1) * m_bar + m
        
        # Dimension of T: n + (h+alpha+2)*m_bar + m.
        dim_T = n + (h + alpha + 2) * m_bar + m

        # ----- Compute P (nonzero) -----
        # Retrieve Y matrices for the horizon [0, h].
        Ys = algo.get_Ys(0, h)
        if tau not in Ys:
            raise ValueError(f"Y matrix for iteration tau = {tau} not found.")
        if 'star' not in Ys:
            raise ValueError("Y star matrix ('star') not found.")
        
        # Retrieve projection matrices.
        Ps = algo.get_Ps()
        if (i, j) not in Ps:
            raise ValueError(f"Projection matrix for component {i}, evaluation {j} not found.")
        if (i, 'star') not in Ps:
            raise ValueError(f"Projection matrix for component {i} star not found.")
        
        # Compute the difference:
        diff = Ps[(i, j)] @ Ys[tau] - Ps[(i, 'star')] @ Ys['star']
        # Compute the outer product:
        P_mat = diff.T @ diff
        
        # ----- Construct T, p, and t as zeros with appropriate dimensions -----
        T_mat = np.zeros((dim_T, dim_T))
        if algo.m_func > 0:
            m_bar_func = algo.m_bar_func    # Total evaluations for functional components.
            m_func = algo.m_func            # Number of functional components.
            dim_p = (h + 1) * m_bar_func + m_func
            p_vec = np.zeros(dim_p)
            dim_t = (h + alpha + 2) * m_bar_func + m_func
            t_vec = np.zeros(dim_t)
            return P_mat, p_vec, T_mat, t_vec
        else:
            return P_mat, T_mat

    @staticmethod
    def get_parameters_function_value_suboptimality(
            algo: Type[Algorithm],
            h: int = 0,
            alpha: int = 0,
            j: int = 1,
            tau: int = 0
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Compute matrices and vectors for function-value suboptimality.

        Mathematical notation and shared definitions follow the class-level
        reference in :class:`~autolyap.iteration_independent.IterationIndependent`.

        This method is only applicable when :math:`m = \NumFunc = 1`.

        It returns a tuple :math:`(P, p, T, t)` where:

        - :math:`p` is computed as

          .. math::
              p = \left( F_{(1,j,\tau)}^{0,h} - F_{(1,\star,\star)}^{0,h} \right)^{\top},

          with :math:`p` returned as a one-dimensional NumPy array.

        The matrices :math:`F_{(1,j,\tau)}^{0,h}` and :math:`F_{(1,\star,\star)}^{0,h}` are retrieved via
        :meth:`~autolyap.algorithms.Algorithm.get_Fs` with `k_min = 0` and `k_max = h`.

        **Resulting lower bounds**

        With this choice of :math:`(P,p,T,t)`,

        .. math::
            \begin{aligned}
            \mathcal{V}(P,p,k) &= f_1(y_{1,j}^{k+\tau}) - f_1(y^\star),\\
            \mathcal{R}(T,t,k) &= 0.
            \end{aligned}

        **Parameters**

        - `algo` (:class:`~typing.Type`\[:class:`~autolyap.algorithms.Algorithm`\]): An instance of :class:`~autolyap.algorithms.Algorithm`. It must
          satisfy `algo.m == 1`, `algo.m_func == 1`, and provide
          :meth:`~autolyap.algorithms.Algorithm.get_Fs`.
        - `h` (:class:`int`): A nonnegative integer corresponding to :math:`h` defining the horizon
          :math:`\llbracket 0, h\rrbracket` for
          :math:`F` matrices.
        - `alpha` (:class:`int`): A nonnegative integer corresponding to :math:`\alpha` for extending the horizon
          for :math:`T` and :math:`t`.
        - `j` (:class:`int`): Evaluation index for component 1 corresponding to :math:`j`. Default is 1; must satisfy
          :math:`j \in \llbracket 1, \NumEval_1\rrbracket`, where :math:`\NumEval_1` is given by
          `algo.m_bar_is[0]`.
        - `tau` (:class:`int`): Iteration index corresponding to :math:`\tau`. Default is 0; must satisfy
          :math:`\tau \in \llbracket 0, h\rrbracket`.

        **Returns**

        - (:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`\]): A tuple :math:`(P, p, T, t)`,
          where :math:`p` is computed as above (a one-dimensional NumPy array), and :math:`P`, :math:`T`, and
          :math:`t` are zero arrays, with

          .. math::
              \begin{aligned}
              P &\in \sym^{n + (h+1)\NumEval + m},\\
              T &\in \sym^{n + (h+\alpha+2)\NumEval + m},\\
              p &\in \mathbb{R}^{(h+1)\NumEvalFunc + \NumFunc},\\
              t &\in \mathbb{R}^{(h+\alpha+2)\NumEvalFunc + \NumFunc}.
              \end{aligned}

        **Raises**

        - `ValueError`: If `algo.m != 1` or `algo.m_func != 1`, if any input parameter is out of range,
          or if the required :math:`F` matrices are not found.
        """
        # ----- Check that m and m_func equal 1 -----
        if algo.m != 1 or algo.m_func != 1:
            raise ValueError("get_parameters_function_value_suboptimality is only applicable when m = m_func = 1.")
        
        # ----- Validate inputs -----
        h = ensure_integral(h, "h", minimum=0)
        alpha = ensure_integral(alpha, "alpha", minimum=0)
        
        num_eval = algo.m_bar_is[0]
        j = ensure_integral(j, "j", minimum=1)
        if j > num_eval:
            raise ValueError(f"For component 1, evaluation index j must be in [1, {num_eval}]. Got {j}.")

        tau = ensure_integral(tau, "tau", minimum=0)
        if tau > h:
            raise ValueError(f"Iteration index tau must be in [0, {h}]. Got {tau}.")
        
        # ----- Retrieve F matrices for the horizon [0, h] -----
        Fs = algo.get_Fs(0, h)
        key_nonstar = (1, j, tau)
        key_star = (1, 'star', 'star')
        if key_nonstar not in Fs:
            raise ValueError(f"F matrix for key {key_nonstar} not found.")
        if key_star not in Fs:
            raise ValueError("F star matrix (1, 'star', 'star') not found.")
        
        # Compute p as the difference between F matrices, then convert to 1D.
        p_vec = (Fs[key_nonstar] - Fs[key_star]).T
        p_vec = np.ravel(p_vec)  # Ensure p is a 1D numpy array.
        
        # ----- Determine dimensions -----
        n = algo.n                    # State dimension.
        m = algo.m                    # Total number of components (should be 1).
        m_bar = algo.m_bar            # Total evaluations per iteration.
        m_bar_func = algo.m_bar_func  # Evaluations for functional components.
        m_func = algo.m_func          # Number of functional components (should be 1).
        
        dim_P = n + (h + 1) * m_bar + m
        dim_T = n + (h + alpha + 2) * m_bar + m
        dim_p = (h + 1) * m_bar_func + m_func
        dim_t = (h + alpha + 2) * m_bar_func + m_func
        
        # ----- Construct zero matrices/vectors for the remaining outputs -----
        P_mat = np.zeros((dim_P, dim_P))
        T_mat = np.zeros((dim_T, dim_T))
        t_vec = np.zeros(dim_t)
        
        return P_mat, p_vec, T_mat, t_vec
    
    @staticmethod
    def bisection_search_rho(
            prob: Type[InclusionProblem],
            algo: Type[Algorithm],
            P: np.ndarray,
            T: np.ndarray,
            p: Optional[np.ndarray] = None,
            t: Optional[np.ndarray] = None,
            h: int = 0,
            alpha: int = 0,
            Q_equals_P: bool = False,
            S_equals_T: bool = False,
            q_equals_p: bool = False,
            s_equals_t: bool = False,
            remove_C2: bool = False,
            remove_C3: bool = False,
            remove_C4: bool = True,
            lower_bound: float = 0.0,
            upper_bound: float = 1.0,
            tol: float = 1e-12
        ) -> Optional[float]:
        r"""
        Perform a bisection search to find the minimum contraction parameter :math:`\rho`.

        Mathematical notation and shared definitions follow the class-level
        reference in :class:`~autolyap.iteration_independent.IterationIndependent`.

        This method performs a bisection search over :math:`\rho` in the interval 
        :math:`[{\text{lower_bound}}, {\text{upper_bound}}]` to find the minimal value for which the 
        iteration-independent Lyapunov inequality holds. At each step it re-solves the same
        model with an updated :math:`\rho` until the interval size is below :math:`{\text{tol}}`.

        Each feasibility check is performed by
        :meth:`~autolyap.iteration_independent.IterationIndependent.verify_iteration_independent_Lyapunov`;
        see its documentation for the enforced SDP feasibility checks.
        The Lyapunov conditions and convergence conclusions are documented in the
        class-level reference of :class:`~autolyap.iteration_independent.IterationIndependent`.

        **Parameters**

        - `prob` (:class:`~typing.Type`\[:class:`~autolyap.problemclass.InclusionProblem`\]): An :class:`~autolyap.problemclass.InclusionProblem`
          instance containing interpolation conditions.
        - `algo` (:class:`~typing.Type`\[:class:`~autolyap.algorithms.Algorithm`\]): An :class:`~autolyap.algorithms.Algorithm` instance providing
          dimensions and methods.
        - `P` (:class:`numpy.ndarray`): A symmetric matrix corresponding to :math:`P \in \sym^{n + (h+1)\NumEval + m}`.
        - `T` (:class:`numpy.ndarray`): A symmetric matrix corresponding to :math:`T \in \sym^{n + (h+\alpha+2)\NumEval + m}`.
        - `p` (:class:`~typing.Optional`\[:class:`numpy.ndarray`\]): A vector corresponding to :math:`p \in \mathbb{R}^{(h+1)\NumEvalFunc + \NumFunc}` for functional components (if applicable).
        - `t` (:class:`~typing.Optional`\[:class:`numpy.ndarray`\]): A vector corresponding to :math:`t \in \mathbb{R}^{(h+\alpha+2)\NumEvalFunc + \NumFunc}` for functional components (if applicable).
        - `h` (:class:`int`): Nonnegative integer corresponding to :math:`h` defining the history for the matrices.
        - `alpha` (:class:`int`): Nonnegative integer corresponding to :math:`\alpha` for extending the horizon.
        - `Q_equals_P` (:class:`bool`): If True, set Q equal to P.
        - `S_equals_T` (:class:`bool`): If True, set S equal to T.
        - `q_equals_p` (:class:`bool`): For functional components, if True, set q equal to p.
        - `s_equals_t` (:class:`bool`): For functional components, if True, set s equal to t.
        - `remove_C2` (:class:`bool`): Flag to remove constraint C2.
        - `remove_C3` (:class:`bool`): Flag to remove constraint C3.
        - `remove_C4` (:class:`bool`): Flag to remove constraint C4.
        - `lower_bound` (:class:`float`): Lower bound for :math:`\rho`.
        - `upper_bound` (:class:`float`): Upper bound for :math:`\rho`.
        - `tol` (:class:`float`): Tolerance for the bisection search stopping criterion.

        **Returns**

        - (:class:`~typing.Optional`\[:class:`float`\]): The minimal :math:`\rho` in
          :math:`[{\text{lower_bound}}, {\text{upper_bound}}]` that verifies the Lyapunov inequality within
          tolerance :math:`{\text{tol}}`, or `None` if the inequality does not hold at the upper bound.

        **Raises**

        - `ValueError`: If any input is out of range or the bounds are inconsistent.
        - `mosek.fusion.OptimizeError`: If MOSEK raises a license-related error during optimization.
        """
        h = ensure_integral(h, "h", minimum=0)
        alpha = ensure_integral(alpha, "alpha", minimum=0)
        lower_bound = ensure_real_number(lower_bound, "lower_bound", finite=True, minimum=0.0)
        upper_bound = ensure_real_number(upper_bound, "upper_bound", finite=True, minimum=0.0)
        if upper_bound < lower_bound:
            raise ValueError("upper_bound must be >= lower_bound.")
        tol = ensure_real_number(tol, "tol", finite=True, minimum=0.0)
        if tol <= 0:
            raise ValueError("tol must be > 0.")
        h, alpha, _, _, _, _, _, _, _ = IterationIndependent._validate_iteration_independent_inputs(
            prob, algo, P, T, p, t, h, alpha
        )

        Mod = Model()
        rho_param = Mod.parameter(1)
        rho_scalar = rho_param.index(0)
        Mod = IterationIndependent._build_iteration_independent_model(
            prob,
            algo,
            P,
            T,
            p,
            t,
            h,
            alpha,
            Q_equals_P,
            S_equals_T,
            q_equals_p,
            s_equals_t,
            remove_C2,
            remove_C3,
            remove_C4,
            rho_term=rho_scalar,
            model=Mod,
        )

        licence_markers = (
            "err_license_max",         # 1016 – all floating tokens in use
            "err_license_server",      # 1015 – server unreachable / down
            "err_missing_license_file" # 1008 – no licence file / server path
        )

        def _check_rho(rho_value: float) -> bool:
            rho_param.setValue([rho_value])
            try:
                Mod.solve()
                Mod.primalObjValue()
            except OptimizeError as e:
                if any(mark in str(e) for mark in licence_markers):
                    raise
                return False
            except Exception:
                return False
            return True

        try:
            # Ensure that the inequality holds at the initial upper bound.
            if not _check_rho(upper_bound):
                return None

            l = lower_bound
            u = upper_bound
            while (u - l) > tol:
                mid = (l + u) / 2.0
                if _check_rho(mid):
                    u = mid
                else:
                    l = mid
            return u
        finally:
            Mod.dispose()

class SublinearConvergence:
    r"""
    Class-level namespace for sublinear-convergence tools in iteration-independent analysis.

    **Scope**

    - Static constructors for sublinear-convergence metrics.

    Method-level docstrings provide full API details.
    """
    @staticmethod
    def get_parameters_fixed_point_residual(
            algo: Type[Algorithm],
            h: int = 0,
            alpha: int = 0,
            tau: int = 0
        ) -> Union[Tuple[np.ndarray, np.ndarray],
                   Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
                   ]:
        r"""
        Compute matrices for the fixed-point residual.

        For iteration index :math:`\tau` (with :math:`\tau \in \llbracket 0, h+\alpha+1\rrbracket`), define

        .. math::
            T = \left( X_{\tau+1}^{0, h+\alpha+1} - X_{\tau}^{0, h+\alpha+1} \right)^{\top}
                \left( X_{\tau+1}^{0, h+\alpha+1} - X_{\tau}^{0, h+\alpha+1} \right),

        where :math:`X_{\tau}^{0, h+\alpha+1}` is the :math:`X` matrix over the horizon
        :math:`\llbracket 0, h+\alpha+1\rrbracket`, retrieved via
        :meth:`~autolyap.algorithms.Algorithm.get_Xs`.

        **Resulting lower bounds**

        With this choice of :math:`(P,p,T,t)`,

        .. math::
            \begin{aligned}
            \mathcal{V}(P,p,k) &= 0,\\
            \mathcal{R}(T,t,k) &= \|x^{k+\tau+1} - x^{k+\tau}\|^2.
            \end{aligned}

        **Parameters**

        - `algo` (:class:`~typing.Type`\[:class:`~autolyap.algorithms.Algorithm`\]): An instance of :class:`~autolyap.algorithms.Algorithm`.
        - `h` (:class:`int`): A nonnegative integer corresponding to :math:`h` defining the time horizon
          :math:`\llbracket 0, h\rrbracket` for
          :math:`P`.
        - `alpha` (:class:`int`): A nonnegative integer corresponding to :math:`\alpha` for extending the horizon
          for :math:`T` (and :math:`t`).
        - `tau` (:class:`int`): Iteration index corresponding to :math:`\tau` for computing the fixed-point residual.
          Must satisfy
          :math:`\tau \in \llbracket 0, h+\alpha+1\rrbracket`.

        **Returns**

        - (:class:`~typing.Union`\[:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`\], :class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`\]\]):
          If `algo.m_func == 0`, returns `(P, T)` with

          .. math::
              \begin{aligned}
              P &\in \sym^{n + (h+1)\NumEval + m},\\
              T &\in \sym^{n + (h+\alpha+2)\NumEval + m}.
              \end{aligned}

          Otherwise, returns `(P, p, T, t)` with

          .. math::
              \begin{aligned}
              P &\in \sym^{n + (h+1)\NumEval + m},\\
              T &\in \sym^{n + (h+\alpha+2)\NumEval + m},\\
              p &\in \mathbb{R}^{(h+1)\NumEvalFunc + \NumFunc},\\
              t &\in \mathbb{R}^{(h+\alpha+2)\NumEvalFunc + \NumFunc}.
              \end{aligned}

        **Raises**

        - `ValueError`: If any input parameter is out of its valid range or if the required :math:`X` matrices are missing.
        """
        # ----- Input Checking -----
        h = ensure_integral(h, "h", minimum=0)
        alpha = ensure_integral(alpha, "alpha", minimum=0)
        tau = ensure_integral(tau, "tau", minimum=0)
        if tau > h + alpha + 1:
            raise ValueError(f"Iteration index tau must be in [0, {h+alpha+1}]. Got {tau}.")

        # ----- Dimensions for P and T -----
        n = algo.n         # State dimension.
        m = algo.m         # Total number of components.
        m_bar = algo.m_bar # Total evaluations per iteration.
        
        # Dimension of P: n + (h+1)*m_bar + m.
        dim_P = n + (h + 1) * m_bar + m
        # Dimension of T: n + (h+alpha+2)*m_bar + m.
        dim_T = n + (h + alpha + 2) * m_bar + m

        # ----- Compute T -----
        # Retrieve X matrices for the horizon [0, h+alpha+1].
        # Note: get_Xs returns X_tau for tau in [0, (h+alpha+1)+1] = [0, h+alpha+2].
        Xs = algo.get_Xs(0, h + alpha + 1)
        if tau not in Xs or (tau + 1) not in Xs:
            raise ValueError(f"X matrices for iterations tau = {tau} and tau+1 = {tau+1} not found.")
        
        diff = Xs[tau + 1] - Xs[tau]
        T_mat = diff.T @ diff

        # ----- Construct P, p, and t as zeros with appropriate dimensions -----
        P_mat = np.zeros((dim_P, dim_P))
        if algo.m_func > 0:
            m_bar_func = algo.m_bar_func    # Evaluations for functional components.
            m_func = algo.m_func            # Number of functional components.
            dim_p = (h + 1) * m_bar_func + m_func
            p_vec = np.zeros(dim_p)
            dim_t = (h + alpha + 2) * m_bar_func + m_func
            t_vec = np.zeros(dim_t)
            return P_mat, p_vec, T_mat, t_vec
        else:
            return P_mat, T_mat

    @staticmethod
    def get_parameters_duality_gap(
            algo: Type[Algorithm],
            h: int = 0,
            alpha: int = 0,
            tau: int = 0
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Compute matrices for the duality gap.

        For iteration index :math:`\tau` (with :math:`\tau \in \llbracket 0, h+\alpha+1\rrbracket`), define

        .. math::
            T = -\frac{1}{2} \sum_{i=1}^{m} \begin{bmatrix}
            P_{(i,\star)}\, U_\star^{0,h+\alpha+1} \\
            P_{(i,1)}\, Y_\tau^{0,h+\alpha+1}
            \end{bmatrix}^{\top}
            \begin{bmatrix}
            0 & 1 \\
            1 & 0 
            \end{bmatrix}
            \begin{bmatrix}
            P_{(i,\star)}\, U_\star^{0,h+\alpha+1} \\
            P_{(i,1)}\, Y_\tau^{0,h+\alpha+1}
            \end{bmatrix},

        and

        .. math::
            t = \sum_{i=1}^{m} \left( F_{(i,1,\tau)}^{0,h+\alpha+1} - F_{(i,\star,\star)}^{0,h+\alpha+1} \right)^{\top}.

        All other matrices are set to zero.

        Here, :math:`U_\star^{0,h+\alpha+1}` and :math:`Y_\tau^{0,h+\alpha+1}` are retrieved via
        :meth:`~autolyap.algorithms.Algorithm.get_Us` and
        :meth:`~autolyap.algorithms.Algorithm.get_Ys`, :math:`F_{(i,1,\tau)}^{0,h+\alpha+1}` and
        :math:`F_{(i,\star,\star)}^{0,h+\alpha+1}` are retrieved via
        :meth:`~autolyap.algorithms.Algorithm.get_Fs`, and projection matrices
        :math:`P_{(i,\star)}` and :math:`P_{(i,1)}` come from :meth:`~autolyap.algorithms.Algorithm.get_Ps`.

        **Resulting lower bounds**

        With this choice of :math:`(P,p,T,t)`,

        .. math::
            \mathcal{V}(P,p,k) = 0,

        and

        .. math::
            \mathcal{R}(T,t,k)
            = -\sum_{i=1}^{m}
            \left\langle
            P_{(i,\star)}U_{\star}^{0,h+\alpha+1}\boldsymbol{z}_{\mathcal{R}}^{k},
            P_{(i,1)}Y_{\tau}^{0,h+\alpha+1}\boldsymbol{z}_{\mathcal{R}}^{k}
            \right\rangle
            + \sum_{i=1}^{m}
            \left(F_{(i,1,\tau)}^{0,h+\alpha+1} - F_{(i,\star,\star)}^{0,h+\alpha+1}\right)^{\top}
            \boldsymbol{f}_{\mathcal{R}}^{k}.

        **Requirements**

        Requires :math:`m = \NumFunc` (i.e., all components are functional).

        **Parameters**

        - `algo` (:class:`~typing.Type`\[:class:`~autolyap.algorithms.Algorithm`\]): An instance of :class:`~autolyap.algorithms.Algorithm` (with
          `algo.m == algo.m_func`).
        - `h` (:class:`int`): A nonnegative integer corresponding to :math:`h` defining the time horizon
          :math:`\llbracket 0, h\rrbracket` for
          :math:`P`.
        - `alpha` (:class:`int`): A nonnegative integer corresponding to :math:`\alpha` for extending the horizon
          for :math:`T` and :math:`t`.
        - `tau` (:class:`int`): Iteration index corresponding to :math:`\tau` for computing the duality gap. Must satisfy
          :math:`\tau \in \llbracket 0, h+\alpha+1\rrbracket`.

        **Returns**

        - (:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`\]): A tuple :math:`(P, p, T, t)`, where
          :math:`t` is a one-dimensional NumPy array, with

          .. math::
              \begin{aligned}
              P &\in \sym^{n + (h+1)\NumEval + m},\\
              T &\in \sym^{n + (h+\alpha+2)\NumEval + m},\\
              p &\in \mathbb{R}^{(h+1)\NumEvalFunc + \NumFunc},\\
              t &\in \mathbb{R}^{(h+\alpha+2)\NumEvalFunc + \NumFunc}.
              \end{aligned}

        **Raises**

        - `ValueError`: If any input parameter is out of its valid range, if required matrices
          are missing, or if :math:`m \ne \NumFunc`.
        """
        # ----- Check that m = m_func -----
        if algo.m != algo.m_func:
            raise ValueError("get_parameters_duality_gap is only applicable when m = m_func.")
        
        # ----- Input Checking -----
        h = ensure_integral(h, "h", minimum=0)
        alpha = ensure_integral(alpha, "alpha", minimum=0)
        tau = ensure_integral(tau, "tau", minimum=0)
        if tau > h + alpha + 1:
            raise ValueError(f"Iteration index tau must be in [0, {h+alpha+1}]. Got {tau}.")
        
        # ----- Dimensions for P, T, p, and t -----
        n = algo.n         # State dimension.
        m = algo.m         # Total number of components (also equals m_func here).
        m_bar = algo.m_bar # Total evaluations per iteration.
        
        # Dimension of P: n + (h+1)*m_bar + m.
        dim_P = n + (h + 1) * m_bar + m
        # Dimension of T: n + (h+alpha+2)*m_bar + m.
        dim_T = n + (h + alpha + 2) * m_bar + m
        
        # Functional dimensions:
        m_bar_func = algo.m_bar_func
        m_func = algo.m_func
        dim_p = (h + 1) * m_bar_func + m_func
        dim_t = (h + alpha + 2) * m_bar_func + m_func
        
        # ----- Compute T -----
        # Retrieve U and Y matrices over the horizon [0, h+alpha+1]
        U_dict = algo.get_Us(0, h + alpha + 1)
        Y_dict = algo.get_Ys(0, h + alpha + 1)
        if 'star' not in U_dict:
            raise ValueError("U_star matrix ('star') not found.")
        if tau not in Y_dict:
            raise ValueError(f"Y matrix for iteration tau = {tau} not found.")
        U_star = U_dict['star']
        Y_tau = Y_dict[tau]
        
        # Retrieve projection matrices.
        Ps = algo.get_Ps()
        
        # Define the 2x2 swap matrix (renamed to mid).
        mid = np.array([[0, 1],
                        [1, 0]])
        
        # Initialize the accumulator for T.
        T_sum = np.zeros((dim_T, dim_T))
        for i in range(1, m + 1):
            # Retrieve P_{(i,star)} and P_{(i,1)}.
            if (i, 'star') not in Ps:
                raise ValueError(f"Projection matrix for component {i} star not found.")
            if (i, 1) not in Ps:
                raise ValueError(f"Projection matrix for component {i}, evaluation 1 not found.")
            P_i_star = Ps[(i, 'star')]
            P_i_1 = Ps[(i, 1)]
            
            # Compute the two blocks.
            block1 = P_i_star @ U_star  # 1 x dim_T
            block2 = P_i_1 @ Y_tau        # 1 x dim_T
            
            # Stack to form a 2 x dim_T matrix.
            block = np.vstack([block1, block2])
            # Contribution from component i: block^{\top} mid block.
            T_sum += block.T @ mid @ block
        T_mat = -0.5 * T_sum
        
        # ----- Compute t -----
        # Retrieve F matrices over the horizon [0, h+alpha+1].
        Fs = algo.get_Fs(0, h + alpha + 1)
        t_sum = np.zeros((dim_t, 1))
        for i in range(1, m + 1):
            key_nonstar = (i, 1, tau)
            key_star = (i, 'star', 'star')
            if key_nonstar not in Fs:
                raise ValueError(f"F matrix for key {key_nonstar} not found.")
            if key_star not in Fs:
                raise ValueError(f"F star matrix for key {key_star} not found.")
            diff_F = Fs[key_nonstar] - Fs[key_star]  # This is a row vector.
            t_sum += diff_F.T  # Sum the transposed (column) vectors.
        # Flatten to obtain a one-dimensional array.
        t_vec = t_sum.ravel()
        
        # ----- Construct P and p as zeros -----
        P_mat = np.zeros((dim_P, dim_P))
        p_vec = np.zeros(dim_p)
        
        return P_mat, p_vec, T_mat, t_vec
    
    @staticmethod
    def get_parameters_function_value_suboptimality(
            algo: Type[Algorithm],
            h: int = 0,
            alpha: int = 0,
            j: int = 1,
            tau: int = 0
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Compute matrices and vectors for function-value suboptimality.

        This method is only applicable when :math:`m = \NumFunc = 1`.

        It returns a tuple :math:`(P, p, T, t)` where:

        - :math:`t` is computed as

          .. math::
              t = \left( F_{(1,j,\tau)}^{0,h+\alpha+1} - F_{(1,\star,\star)}^{0,h+\alpha+1} \right)^{\top},

          with :math:`t` returned as a one-dimensional NumPy array.

        The matrices :math:`F_{(1,j,\tau)}^{0,h+\alpha+1}` and :math:`F_{(1,\star,\star)}^{0,h+\alpha+1}` are
        retrieved via :meth:`~autolyap.algorithms.Algorithm.get_Fs` with `k_min = 0` and
        `k_max = h+\alpha+1`.

        **Resulting lower bounds**

        With this choice of :math:`(P,p,T,t)`,

        .. math::
            \begin{aligned}
            \mathcal{V}(P,p,k) &= 0,\\
            \mathcal{R}(T,t,k) &= f_1(y_{1,j}^{k+\tau}) - f_1(y^\star).
            \end{aligned}

        **Parameters**

        - `algo` (:class:`~typing.Type`\[:class:`~autolyap.algorithms.Algorithm`\]): An instance of :class:`~autolyap.algorithms.Algorithm`. It must
          satisfy `algo.m == 1`, `algo.m_func == 1`, and provide
          :meth:`~autolyap.algorithms.Algorithm.get_Fs`.
        - `h` (:class:`int`): A nonnegative integer corresponding to :math:`h` defining the horizon
          :math:`\llbracket 0, h + \alpha + 1\rrbracket` for :math:`F` matrices.
        - `alpha` (:class:`int`): A nonnegative integer corresponding to :math:`\alpha` for extending the horizon
          for :math:`T` and :math:`t`.
        - `j` (:class:`int`): Evaluation index for component 1 corresponding to :math:`j`. Default is 1; must satisfy
          :math:`j \in \llbracket 1, \NumEval_1\rrbracket`, where :math:`\NumEval_1` is given by
          `algo.m_bar_is[0]`.
        - `tau` (:class:`int`): Iteration index corresponding to :math:`\tau`. Default is 0; must satisfy
          :math:`\tau \in \llbracket 0, h + \alpha + 1\rrbracket`.

        **Returns**

        - (:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`\]): A tuple :math:`(P, p, T, t)`, where
          :math:`t` is computed as above (a one-dimensional NumPy array), and :math:`P`, :math:`T`, and :math:`p`
          are zero arrays, with

          .. math::
              \begin{aligned}
              P &\in \sym^{n + (h+1)\NumEval + m},\\
              T &\in \sym^{n + (h+\alpha+2)\NumEval + m},\\
              p &\in \mathbb{R}^{(h+1)\NumEvalFunc + \NumFunc},\\
              t &\in \mathbb{R}^{(h+\alpha+2)\NumEvalFunc + \NumFunc}.
              \end{aligned}

        **Raises**

        - `ValueError`: If `algo.m != 1` or `algo.m_func != 1`, if any input parameter is out of range,
          or if the required :math:`F` matrices are not found.
        """
        # ----- Check that m and m_func equal 1 -----
        if algo.m != 1 or algo.m_func != 1:
            raise ValueError("get_parameters_function_value_suboptimality is only applicable when m = m_func = 1.")
        
        # ----- Validate inputs -----
        h = ensure_integral(h, "h", minimum=0)
        alpha = ensure_integral(alpha, "alpha", minimum=0)
        
        num_eval = algo.m_bar_is[0]
        j = ensure_integral(j, "j", minimum=1)
        if j > num_eval:
            raise ValueError(f"For component 1, evaluation index j must be in [1, {num_eval}]. Got {j}.")

        tau = ensure_integral(tau, "tau", minimum=0)
        if tau > h + alpha + 1:
            raise ValueError(f"Iteration index tau must be in [0, {h+alpha+1}]. Got {tau}.")
        
        # ----- Dimensions for P, T, p, and t -----
        n = algo.n         # State dimension.
        m = algo.m         # Total number of components (also equals m_func here).
        m_bar = algo.m_bar # Total evaluations per iteration.
        
        # Dimension of P: n + (h+1)*m_bar + m.
        dim_P = n + (h + 1) * m_bar + m
        # Dimension of T: n + (h+alpha+2)*m_bar + m.
        dim_T = n + (h + alpha + 2) * m_bar + m

        # Functional dimensions:
        m_bar_func = algo.m_bar_func
        m_func = algo.m_func
        dim_p = (h + 1) * m_bar_func + m_func
        dim_t = (h + alpha + 2) * m_bar_func + m_func

        T = np.zeros((dim_T, dim_T))
        P = np.zeros((dim_P, dim_P))
        p = np.zeros(dim_p)
        
        # ----- Compute t -----
        # Retrieve F matrices over the horizon [0, h+alpha+1].
        Fs = algo.get_Fs(0, h + alpha + 1)
        key_nonstar = (1, j, tau)
        key_star = (1, 'star', 'star')
        if key_nonstar not in Fs:
            raise ValueError(f"F matrix for key {key_nonstar} not found.")
        if key_star not in Fs:
            raise ValueError(f"F star matrix for key {key_star} not found.")
        t = Fs[key_nonstar] - Fs[key_star]  # This is a row vector.
        # Flatten to obtain a one-dimensional array.
        t = t.ravel()

        return P, p, T, t
        
    @staticmethod
    def get_parameters_optimality_measure(
                algo: Type[Algorithm],
                h: int = 0,
                alpha: int = 0,
                tau: int = 0
        ) -> Union[Tuple[np.ndarray, np.ndarray],
                   Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
                   ]:
        r"""
        Compute matrices for the optimality measure.

        For iteration index :math:`\tau` (with :math:`\tau \in \llbracket 0, h+\alpha+1\rrbracket`), define

        .. math::
            T =
            \begin{cases}
              \left( P_{(1,1)}\, U_\tau^{0,h+\alpha+1} \right)^{\top} \left( P_{(1,1)}\, U_\tau^{0,h+\alpha+1} \right)
              & \text{if } m = 1, \\[1em]
              \left( \left( \sum_{i=1}^{m} P_{(i,1)}\, U_\tau^{0,h+\alpha+1} \right)^{\top} \left( \sum_{i=1}^{m} P_{(i,1)}\, U_\tau^{0,h+\alpha+1} \right)
              + \sum_{i=2}^{m} \left( \left( P_{(1,1)} - P_{(i,1)} \right) Y_\tau^{0,h+\alpha+1} \right)^{\top} \left( \left( P_{(1,1)} - P_{(i,1)} \right) Y_\tau^{0,h+\alpha+1} \right) \right)
              & \text{if } m > 1.
            \end{cases}

        All other matrices are set to zero.

        Here:

        - :math:`U_\tau^{0,h+\alpha+1}` is retrieved via :meth:`~autolyap.algorithms.Algorithm.get_Us`
          with `k_min = 0` and `k_max = h+\alpha+1`.
        - :math:`Y_\tau^{0,h+\alpha+1}` is retrieved via :meth:`~autolyap.algorithms.Algorithm.get_Ys`
          with `k_min = 0` and `k_max = h+\alpha+1`.
        - :math:`P_{(i,1)}` are the projection matrices returned by
          :meth:`~autolyap.algorithms.Algorithm.get_Ps`.

        **Resulting lower bounds**

        With this choice of :math:`(P,p,T,t)`,

        .. math::
            \mathcal{V}(P,p,k) = 0,

        and

        .. math::
            \mathcal{R}(T,t,k)
            =
            \begin{cases}
                \|u_{1,1}^{k+\tau}\|^2 & \text{if } m = 1, \\
                \left\|\sum_{i=1}^{m} u_{i,1}^{k+\tau}\right\|^2
                + \sum_{i=2}^{m} \|y_{1,1}^{k+\tau} - y_{i,1}^{k+\tau}\|^2
                & \text{if } m > 1.
            \end{cases}

        **Parameters**

        - `algo` (:class:`~typing.Type`\[:class:`~autolyap.algorithms.Algorithm`\]): An instance of :class:`~autolyap.algorithms.Algorithm`.
        - `h` (:class:`int`): A nonnegative integer corresponding to :math:`h` defining the time horizon
          :math:`\llbracket 0, h\rrbracket` for
          :math:`P`.
        - `alpha` (:class:`int`): A nonnegative integer corresponding to :math:`\alpha` for extending the horizon
          for :math:`T` (and :math:`t`).
        - `tau` (:class:`int`): Iteration index corresponding to :math:`\tau` for computing the optimality measure. Must satisfy
          :math:`\tau \in \llbracket 0, h+\alpha+1\rrbracket`.

        **Returns**

        - (:class:`~typing.Union`\[:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`\], :class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`\]\]):
          If `algo.m_func == 0`, returns `(P, T)` with

          .. math::
              \begin{aligned}
              P &\in \sym^{n + (h+1)\NumEval + m},\\
              T &\in \sym^{n + (h+\alpha+2)\NumEval + m}.
              \end{aligned}

          Otherwise, returns `(P, p, T, t)` with

          .. math::
              \begin{aligned}
              P &\in \sym^{n + (h+1)\NumEval + m},\\
              T &\in \sym^{n + (h+\alpha+2)\NumEval + m},\\
              p &\in \mathbb{R}^{(h+1)\NumEvalFunc + \NumFunc},\\
              t &\in \mathbb{R}^{(h+\alpha+2)\NumEvalFunc + \NumFunc}.
              \end{aligned}

        **Raises**

        - `ValueError`: If any input parameter is out of its valid range or if required matrices are missing.
        """
        # ----- Input Checking -----
        h = ensure_integral(h, "h", minimum=0)
        alpha = ensure_integral(alpha, "alpha", minimum=0)
        tau = ensure_integral(tau, "tau", minimum=0)
        if tau > h + alpha + 1:
            raise ValueError(f"Iteration index tau must be in [0, {h+alpha+1}]. Got {tau}.")
        
        # ----- Dimensions for P and T -----
        n = algo.n         # State dimension.
        m = algo.m         # Total number of components.
        m_bar = algo.m_bar # Total evaluations per iteration.
        
        # Dimension of P: n + (h+1)*m_bar + m.
        dim_P = n + (h + 1) * m_bar + m
        # Dimension of T: n + (h+alpha+2)*m_bar + m.
        dim_T = n + (h + alpha + 2) * m_bar + m
        
        # ----- Retrieve U and Y matrices over [0, h+alpha+1] -----
        U_dict = algo.get_Us(0, h + alpha + 1)
        Y_dict = algo.get_Ys(0, h + alpha + 1)
        if tau not in U_dict:
            raise ValueError(f"U matrix for iteration tau = {tau} not found.")
        if tau not in Y_dict:
            raise ValueError(f"Y matrix for iteration tau = {tau} not found.")
        U_tau = U_dict[tau]
        Y_tau = Y_dict[tau]
        
        # ----- Retrieve projection matrices -----
        Ps = algo.get_Ps()
        
        # ----- Compute T -----
        if m == 1:
            # Case: m = 1.
            if (1, 1) not in Ps:
                raise ValueError("Projection matrix for component 1, evaluation 1 not found.")
            P_11 = Ps[(1, 1)]
            block = P_11 @ U_tau
            T_mat = block.T @ block
        else:
            # Case: m > 1.
            # First term: sum_{i=1}^{m} P_{(i,1)} U_tau.
            sum_U = None
            for i in range(1, m + 1):
                if (i, 1) not in Ps:
                    raise ValueError(f"Projection matrix for component {i}, evaluation 1 not found.")
                P_i1 = Ps[(i, 1)]
                term = P_i1 @ U_tau
                sum_U = term if sum_U is None else sum_U + term
            first_term = sum_U.T @ sum_U
            
            # Second term: sum_{i=2}^{m} ((P_{(1,1)} - P_{(i,1)}) Y_k).T ((P_{(1,1)} - P_{(i,1)}) Y_k).
            if (1, 1) not in Ps:
                raise ValueError("Projection matrix for component 1, evaluation 1 not found.")
            P_11 = Ps[(1, 1)]
            second_term = np.zeros((dim_T, dim_T))
            for i in range(2, m + 1):
                if (i, 1) not in Ps:
                    raise ValueError(f"Projection matrix for component {i}, evaluation 1 not found.")
                diff = (P_11 - Ps[(i, 1)]) @ Y_tau
                second_term += diff.T @ diff
            T_mat = first_term + second_term
        
        # ----- Construct zero matrices for the remaining outputs -----
        P_mat = np.zeros((dim_P, dim_P))
        
        if algo.m_func > 0:
            m_bar_func = algo.m_bar_func    # Evaluations for functional components.
            m_func = algo.m_func            # Number of functional components.
            dim_p = (h + 1) * m_bar_func + m_func
            dim_t = (h + alpha + 2) * m_bar_func + m_func
            p_vec = np.zeros(dim_p)
            t_vec = np.zeros(dim_t)
            return P_mat, p_vec, T_mat, t_vec
        else:
            return P_mat, T_mat


class IterationIndependent:
    r"""
    Iteration-independent Lyapunov analysis utilities.

    Class-level reference
    =====================

    This class-level docstring centralizes notation shared across methods.
    Method-level docstrings focus on API details: inputs, outputs, and validation.

    **Problem and algorithm context**

    Notation follows :class:`~autolyap.problemclass.InclusionProblem` and
    :class:`~autolyap.algorithms.Algorithm`.
    The sets :math:`\IndexFunc`, :math:`\IndexOp` and counts
    :math:`\NumEval`, :math:`\NumEvalFunc`, :math:`\NumFunc` are inherited from those classes.
    We consider sequences

    .. math::
        (\bx^{k},\bu^{k},\by^{k},\bFcn^{k})_{k\in\naturals}

    generated by the state-space representation, together with a solution triplet
    :math:`(y^{\star},\hat{\bu}^{\star},\bFcn^{\star})`.

    **Star variables**

    A solution triplet means there exists a vector

    .. math::
        \bu^{\star} = (u_1^{\star},\ldots,u_m^{\star}),

    such that

    .. math::
        \forall i \in \IndexFunc, \quad u_i^{\star}\in\partial f_i(y^{\star}), \qquad
        \forall i \in \IndexOp, \quad u_i^{\star}\in G_i(y^{\star}), \qquad
        \sum_{i=1}^{m} u_i^{\star}=0.

    Hence, each :math:`u_i^{\star}` is the component output at :math:`y^{\star}`.

    The reduced vector :math:`\hat{\bu}^{\star}` collects the first :math:`m-1` components of
    :math:`\bu^{\star}`:

    .. math::
        \hat{\bu}^{\star} = (u_1^{\star},\ldots,u_{m-1}^{\star}), \qquad
        u_m^{\star} = -\sum_{i=1}^{m-1} u_i^{\star}.

    The function-value vector is

    .. math::
        \bFcn^{\star}=(f_i(y^{\star}))_{i\in\IndexFunc}.

    **Lyapunov ansatzes**

    Let :math:`h \in \naturals` be the history parameter and
    :math:`\alpha \in \naturals` the overlap parameter.
    For each :math:`(W,w) \in \{(Q,q),(P,p)\}` and :math:`k \in \naturals`, define

    .. math::
        \boldsymbol{z}_{\mathcal{V}}^{k}
        =
        \begin{bmatrix}
        \bx^{k} \\ \bu^{k} \\ \vdots \\ \bu^{k+h} \\ \hat{\bu}^{\star} \\ y^{\star}
        \end{bmatrix}, \qquad
        \boldsymbol{f}_{\mathcal{V}}^{k}
        =
        \begin{bmatrix}
        \bFcn^{k} \\ \vdots \\ \bFcn^{k+h} \\ \bFcn^{\star}
        \end{bmatrix},

    .. math::
        \mathcal{V}(W,w,k)
        =
        \inner{
        \boldsymbol{z}_{\mathcal{V}}^{k}
        }{
        (W \kron \Id)\boldsymbol{z}_{\mathcal{V}}^{k}
        }
        + w^{\top}\boldsymbol{f}_{\mathcal{V}}^{k}.

    Similarly, for each :math:`(W,w) \in \{(S,s),(T,t)\}`, define

    .. math::
        \boldsymbol{z}_{\mathcal{R}}^{k}
        =
        \begin{bmatrix}
        \bx^{k} \\ \bu^{k} \\ \vdots \\ \bu^{k+h+\alpha+1} \\ \hat{\bu}^{\star} \\ y^{\star}
        \end{bmatrix}, \qquad
        \boldsymbol{f}_{\mathcal{R}}^{k}
        =
        \begin{bmatrix}
        \bFcn^{k} \\ \vdots \\ \bFcn^{k+h+\alpha+1} \\ \bFcn^{\star}
        \end{bmatrix},

    .. math::
        \mathcal{R}(W,w,k)
        =
        \inner{
        \boldsymbol{z}_{\mathcal{R}}^{k}
        }{
        (W \kron \Id)\boldsymbol{z}_{\mathcal{R}}^{k}
        }
        + w^{\top}\boldsymbol{f}_{\mathcal{R}}^{k}.

    **Dimensions**

    .. math::
        P,Q \in \sym^{n + (h+1)\NumEval + m}, \qquad
        T,S \in \sym^{n + (h+\alpha+2)\NumEval + m},

    .. math::
        p,q \in \mathbb{R}^{(h+1)\NumEvalFunc + \NumFunc}, \qquad
        t,s \in \mathbb{R}^{(h+\alpha+2)\NumEvalFunc + \NumFunc}.

    Here :math:`n` is the state dimension (as in :class:`~autolyap.algorithms.Algorithm`).
    When :math:`\NumFunc = 0`, the vectors :math:`p,q,s,t` are omitted.

    **Quadratic Lyapunov inequality**

    For all :math:`k \in \naturals`, the quadratic Lyapunov inequalities are

    .. math::
        \mathcal{V}(Q,q,k+\alpha+1) \le \rho\,\mathcal{V}(Q,q,k) - \mathcal{R}(S,s,k), \tag{C1}

    .. math::
        \mathcal{V}(Q,q,k) \ge \mathcal{V}(P,p,k) \ge 0, \tag{C2}

    .. math::
        \mathcal{R}(S,s,k) \ge \mathcal{R}(T,t,k) \ge 0, \tag{C3}

    .. math::
        \mathcal{R}(S,s,k+1) \le \mathcal{R}(S,s,k). \tag{C4}

    Condition (C4) is optional.

    **Role of h and alpha**

    - :math:`h` is the history parameter in :math:`\mathcal{V}`.
    - :math:`\alpha` is the overlap parameter in :math:`\mathcal{R}` and in the
      shift :math:`k \mapsto k+\alpha+1` appearing in the Lyapunov inequality.

    **Convergence conclusions**

    When :math:`(Q,q,S,s)` satisfy the enabled Lyapunov inequalities (in particular (C1)–(C3)) used in
    :meth:`~autolyap.iteration_independent.IterationIndependent.verify_iteration_independent_Lyapunov`,
    the following conclusions hold.

    - Linear setting (:math:`\rho \in [0,1)`):

      .. math::
          0 \le \mathcal{V}(P,p,k) \le \rho^{\lfloor k/(\alpha+1)\rfloor}
          \max_{i \in \llbracket 0,\alpha\rrbracket} \mathcal{V}(Q,q,i).

      Thus, :math:`\mathcal{V}(P,p,k)` converges to zero

      .. math::
          \sqrt[\alpha+1]{\rho}\text{-linearly}.

    - Sublinear setting (:math:`\rho = 1`):

      .. math::
          \sum_{i=0}^{k}\mathcal{R}(T,t,i)
          \le \sum_{i=0}^{k}\mathcal{R}(S,s,i)
          \le \sum_{j=0}^{\alpha}\mathcal{V}(Q,q,j),

      so :math:`\mathcal{R}(T,t,k)` is summable and

      .. math::
          \min_{i \in \llbracket 0,k \rrbracket}\mathcal{R}(T,t,i) \in \mathcal{O}(1/k),

      with :math:`\mathcal{R}(T,t,k) \in o(1/k)` if (C4) holds.
    """
    
    LinearConvergence = LinearConvergence
    SublinearConvergence = SublinearConvergence

    @staticmethod
    def _scale_by_rho(rho_term, expr):
        r"""Scale a matrix/vector expression by :math:`\rho` for scalar and Fusion-parameter cases."""
        if isinstance(rho_term, (int, float, np.floating)):
            return rho_term * expr
        return Expr.mul(rho_term, expr)

    @staticmethod
    def _validate_iteration_independent_inputs(
            prob: Type[InclusionProblem],
            algo: Type[Algorithm],
            P: np.ndarray,
            T: np.ndarray,
            p: Optional[np.ndarray],
            t: Optional[np.ndarray],
            h: int,
            alpha: int,
        ) -> Tuple[int, int, int, int, int, int, int, int, int]:
        r"""Validate problem/algorithm consistency and array shapes, then return normalized dimensions."""
        # -------------------------------------------------------------------------
        # Validate consistency between the problem and the algorithm.
        # -------------------------------------------------------------------------
        if prob.m != algo.m:
            raise ValueError("Mismatch in number of components: prob.m and algo.m must be the same")
        
        # Check that the functional and operator component indices are identical.
        if set(prob.I_func) != set(algo.I_func):
            raise ValueError("Mismatch in functional component indices between prob and algo")
        if set(prob.I_op) != set(algo.I_op):
            raise ValueError("Mismatch in operator component indices between prob and algo")
        
        # Ensure h and alpha are nonnegative.
        h = ensure_integral(h, "h", minimum=0)
        alpha = ensure_integral(alpha, "alpha", minimum=0)

        # -------------------------------------------------------------------------
        # Retrieve dimensions from the algorithm instance.
        # -------------------------------------------------------------------------
        n = algo.n                      # State dimension.
        m_bar = algo.m_bar              # Total evaluations per iteration.
        m = algo.m                      # Total number of components.
        m_bar_func = algo.m_bar_func    # Total evaluations for functional components.
        m_func = algo.m_func            # Number of functional components.
        m_op = algo.m_op                # Number of operator components.
        m_bar_op = algo.m_bar_op        # Total evaluations for operator components.
        
        # Expected dimension for matrix P: [n + (h+1)*m_bar + m] x [n + (h+1)*m_bar + m].
        dim_P = n + (h + 1) * m_bar + m
        if not (isinstance(P, np.ndarray) and P.ndim == 2 and P.shape[0] == P.shape[1] == dim_P):
            raise ValueError(
                f"P must be a symmetric matrix of dimension {dim_P}x{dim_P}. "
                f"Got shape {getattr(P, 'shape', None)}."
            )
        ensure_finite_array(P, "P")
        if not np.allclose(P, P.T, atol=1e-8):
            raise ValueError("P must be symmetric.")
        
        # Expected dimension for matrix T: [n + (h+alpha+2)*m_bar + m] x [n + (h+alpha+2)*m_bar + m].
        dim_T = n + (h + alpha + 2) * m_bar + m
        if not (isinstance(T, np.ndarray) and T.ndim == 2 and T.shape[0] == T.shape[1] == dim_T):
            raise ValueError(
                f"T must be a symmetric matrix of dimension {dim_T}x{dim_T}. "
                f"Got shape {getattr(T, 'shape', None)}."
            )
        ensure_finite_array(T, "T")
        if not np.allclose(T, T.T, atol=1e-8):
            raise ValueError("T must be symmetric.")
        
        # For functional components, p and t must have proper dimensions.
        if m_func > 0:
            # Compute required dimensions
            dim_p = (h + 1) * m_bar_func + m_func
            dim_t = (h + alpha + 2) * m_bar_func + m_func

            # Check p
            if p is None:
                raise ValueError(
                    f"p must be a 1D numpy array of length {dim_p}, but got None."
                )
            if not (isinstance(p, np.ndarray) and p.ndim == 1 and p.shape[0] == dim_p):
                raise ValueError(
                    f"p must be a 1D numpy array of length {dim_p}. Got shape "
                    f"{getattr(p, 'shape', None)}."
                )
            ensure_finite_array(p, "p")

            # Check t
            if t is None:
                raise ValueError(
                    f"t must be a 1D numpy array of length {dim_t}, but got None."
                )
            if not (isinstance(t, np.ndarray) and t.ndim == 1 and t.shape[0] == dim_t):
                raise ValueError(
                    f"t must be a 1D numpy array of length {dim_t}. Got shape "
                    f"{getattr(t, 'shape', None)}."
                )
            ensure_finite_array(t, "t")
        else:
            if p is not None or t is not None:
                raise ValueError("p and t must be None when there are no functional components.")

        return h, alpha, n, m_bar, m, m_bar_func, m_func, m_op, m_bar_op

    @staticmethod
    def _build_iteration_independent_model(
            prob: Type[InclusionProblem],
            algo: Type[Algorithm],
            P: np.ndarray,
            T: np.ndarray,
            p: Optional[np.ndarray],
            t: Optional[np.ndarray],
            h: int,
            alpha: int,
            Q_equals_P: bool,
            S_equals_T: bool,
            q_equals_p: bool,
            s_equals_t: bool,
            remove_C2: bool,
            remove_C3: bool,
            remove_C4: bool,
            rho_term,
            model: Optional[Model] = None,
        ) -> Model:
        r"""Assemble and return the MOSEK Fusion model for the selected Lyapunov conditions."""
        # Dimensions and indices have already been validated by callers.
        n = algo.n
        m_bar = algo.m_bar
        m = algo.m
        m_bar_func = algo.m_bar_func
        m_func = algo.m_func
        m_op = algo.m_op
        m_bar_op = algo.m_bar_op

        dim_P = n + (h + 1) * m_bar + m
        dim_T = n + (h + alpha + 2) * m_bar + m
        dim_p = (h + 1) * m_bar_func + m_func
        dim_t = (h + alpha + 2) * m_bar_func + m_func

        Mod = model if model is not None else Model()

        # Q variable: either set equal to P or defined as a new symmetric variable.
        if Q_equals_P:
            Q = P
        else:
            Qij = Mod.variable("Q_upper_triangle_vars", dim_P * (dim_P + 1) // 2, Domain.unbounded())
            Q = create_symmetric_matrix_expression(Qij, dim_P)

        # S variable: either set equal to T or defined as a new symmetric variable.
        if S_equals_T:
            S = T
        else:
            Sij = Mod.variable("S_upper_triangle_vars", dim_T * (dim_T + 1) // 2, Domain.unbounded())
            S = create_symmetric_matrix_expression(Sij, dim_T)
        
        # For functional components, create variables q and s (or set them equal to p and t).
        if m_func > 0:
            if q_equals_p:
                q = p
            else:
                q = Mod.variable("q", dim_p, Domain.unbounded())
            if s_equals_t:
                s = t
            else:
                s = Mod.variable("s", dim_t, Domain.unbounded())
        
        # ---------------------------------------------------------------------
        # Build the main PSD (positive semidefinite) and equality constraint sums.
        # These will later be constrained to be in the PSD cone or equal zero, respectively.
        # ---------------------------------------------------------------------
        Ws = {}       # Dictionary for matrix constraint sums.
        k_maxs = {}   # Dictionary to store the maximum iteration index for each condition.

        # For condition "C1": use _compute_Thetas with k_max = h + alpha + 1.
        Theta0_C1, Theta1_C1 = IterationIndependent._compute_Thetas(algo, h, alpha, condition='C1')
        Ws["C1"] = Expr.add(
            Expr.sub(
                Theta1_C1.T @ Q @ Theta1_C1,
                IterationIndependent._scale_by_rho(rho_term, Theta0_C1.T @ Q @ Theta0_C1),
            ),
            S,
        )
        k_maxs["C1"] = h + alpha + 1

        # Condition "C2": if not removed, enforce P - Q.
        if not remove_C2:
            Ws["C2"] = P - Q
            k_maxs["C2"] = h

        # Condition "C3": if not removed, enforce T - S.
        if not remove_C3:
            Ws["C3"] = T - S
            k_maxs["C3"] = h + alpha + 1

        # Condition "C4": if not removed, use _compute_Thetas with k_max = h + alpha + 2.
        if not remove_C4:
            Theta0_C4, Theta1_C4 = IterationIndependent._compute_Thetas(algo, h, alpha, condition='C4')
            Ws["C4"] = Theta1_C4.T @ S @ Theta1_C4 - Theta0_C4.T @ S @ Theta0_C4
            k_maxs["C4"] = h + alpha + 2

        # For functional components, build the analogous vector constraints.
        if m_func > 0:
            ws = {}
            theta0_C1, theta1_C1 = IterationIndependent._compute_thetas(algo, h, alpha, condition='C1')
            term1 = theta1_C1.T @ q
            term2 = IterationIndependent._scale_by_rho(rho_term, theta0_C1.T @ q)
            ws["C1"] = Expr.add(Expr.sub(term1, term2), s)

            if not remove_C2:
                ws["C2"] = p - q

            if not remove_C3:
                ws["C3"] = t - s

            if not remove_C4:
                theta0_C4, theta1_C4 = IterationIndependent._compute_thetas(algo, h, alpha, condition='C4')
                ws["C4"] = (theta1_C4.T - theta0_C4.T) @ s

        # Initialize lists of active conditions.
        conds = ["C1"]
        if not remove_C2:
            conds.append("C2")
        if not remove_C3:
            conds.append("C3")
        if not remove_C4:
            conds.append("C4")

        # Initialize dictionaries to sum up the PSD and equality constraints.
        PSD_constraint_sums = {}
        eq_constraint_sums = {}
        for cond in conds:
            PSD_constraint_sums[cond] = -Ws[cond]
            if m_func > 0:
                eq_constraint_sums[cond] = -ws[cond]

        # Dictionaries to hold multipliers for interpolation conditions.
        if m_op > 0:
            lambdas_op = {}
        if m_func > 0:
            lambdas_func = {}
            nus_func = {}

        # ---------------------------------------------------------------------
        # Define inner helper functions for processing interpolation data.
        # These functions handle both operator and function conditions.
        # ---------------------------------------------------------------------
        op_components = set(algo.I_op)
        m_bar_is = algo.m_bar_is
        compute_W = algo.compute_W
        compute_F_aggregated = algo.compute_F_aggregated
        mod_variable = Mod.variable
        domain_ge0 = Domain.greaterThan(0.0)
        domain_unbounded = Domain.unbounded()
        star_pair = ('star', 'star')

        def process_pairs(cond: str,
                        i: int,
                        o: int,
                        interpolation_data: Union[Tuple[np.ndarray, str],
                                                    Tuple[np.ndarray, np.ndarray, bool, str]],
                        pairs: Tuple[Union[Tuple[int, int], Tuple[str, str]], ...],
                        comp_type: str,
                        has_quadratic: bool,
                        has_linear: bool) -> None:
            r"""
            Internal helper for a single interpolation-pair pattern.

            It creates the appropriate multiplier variables and accumulates contributions
            to PSD/equality constraints for the active Lyapunov condition.
            """
            key = (cond, i, pairs, o)

            if comp_type == 'op':
                if not has_quadratic:
                    return
                M, _ = interpolation_data
            else:
                M, a, eq, _ = interpolation_data
                if not has_quadratic and not has_linear:
                    return

            # Compute the lifted matrix W for the given pairs.
            W_matrix = None
            if has_quadratic:
                W_matrix = compute_W(i, pairs, 0, k_maxs[cond], M, validate=False)

            if comp_type == 'op':
                lambda_var = mod_variable(1, domain_ge0)
                lambdas_op[key] = lambda_var
                PSD_constraint_sums[cond] = PSD_constraint_sums[cond] + lambda_var[0] * W_matrix
            else:
                # For functional components, compute the aggregated F vector.
                F_vector = None
                if has_linear:
                    F_vector = compute_F_aggregated(i, pairs, 0, k_maxs[cond], a, validate=False)
                if eq:
                    nu_var = mod_variable(1, domain_unbounded)
                    nus_func[key] = nu_var
                    if has_quadratic:
                        PSD_constraint_sums[cond] = PSD_constraint_sums[cond] + nu_var[0] * W_matrix
                    if has_linear:
                        eq_constraint_sums[cond] = eq_constraint_sums[cond] + nu_var[0] * F_vector
                else:
                    lambda_var = mod_variable(1, domain_ge0)
                    lambdas_func[key] = lambda_var
                    if has_quadratic:
                        PSD_constraint_sums[cond] = PSD_constraint_sums[cond] + lambda_var[0] * W_matrix
                    if has_linear:
                        eq_constraint_sums[cond] = eq_constraint_sums[cond] + lambda_var[0] * F_vector

        def _handle_j1_lt_j2(cond: str,
                             i: int,
                             o: int,
                             interp_data: Union[Tuple[np.ndarray, str],
                                                Tuple[np.ndarray, np.ndarray, bool, str]],
                             pairs_with_star: List[Union[Tuple[int, int], Tuple[str, str]]],
                             _pairs_no_star: List[Tuple[int, int]],
                             comp_type: str,
                             has_quadratic: bool,
                             has_linear: bool) -> None:
            for pair1, pair2 in combinations(pairs_with_star, 2):
                process_pairs(cond, i, o, interp_data, (pair1, pair2), comp_type, has_quadratic, has_linear)

        def _handle_j1_ne_j2(cond: str,
                             i: int,
                             o: int,
                             interp_data: Union[Tuple[np.ndarray, str],
                                                Tuple[np.ndarray, np.ndarray, bool, str]],
                             pairs_with_star: List[Union[Tuple[int, int], Tuple[str, str]]],
                             _pairs_no_star: List[Tuple[int, int]],
                             comp_type: str,
                             has_quadratic: bool,
                             has_linear: bool) -> None:
            n_pairs = len(pairs_with_star)
            for idx1 in range(n_pairs):
                pair1 = pairs_with_star[idx1]
                for idx2 in range(n_pairs):
                    if idx1 == idx2:
                        continue
                    pair2 = pairs_with_star[idx2]
                    process_pairs(cond, i, o, interp_data, (pair1, pair2), comp_type, has_quadratic, has_linear)

        def _handle_j1(cond: str,
                       i: int,
                       o: int,
                       interp_data: Union[Tuple[np.ndarray, str],
                                          Tuple[np.ndarray, np.ndarray, bool, str]],
                       pairs_with_star: List[Union[Tuple[int, int], Tuple[str, str]]],
                       _pairs_no_star: List[Tuple[int, int]],
                       comp_type: str,
                       has_quadratic: bool,
                       has_linear: bool) -> None:
            for pair in pairs_with_star:
                process_pairs(cond, i, o, interp_data, (pair,), comp_type, has_quadratic, has_linear)

        def _handle_j1_ne_star(cond: str,
                               i: int,
                               o: int,
                               interp_data: Union[Tuple[np.ndarray, str],
                                                  Tuple[np.ndarray, np.ndarray, bool, str]],
                               _pairs_with_star: List[Union[Tuple[int, int], Tuple[str, str]]],
                               pairs_no_star: List[Tuple[int, int]],
                               comp_type: str,
                               has_quadratic: bool,
                               has_linear: bool) -> None:
            for pair in pairs_no_star:
                process_pairs(cond, i, o, interp_data, (pair, star_pair), comp_type, has_quadratic, has_linear)

        handlers = {
            'j1<j2': _handle_j1_lt_j2,
            'j1!=j2': _handle_j1_ne_j2,
            'j1': _handle_j1,
            'j1!=star': _handle_j1_ne_star,
        }

        def _expected_pairs_len(interp_key: str) -> int:
            if interp_key == 'j1':
                return 1
            if interp_key in ('j1<j2', 'j1!=j2', 'j1!=star'):
                return 2
            raise ValueError(f"Error: Invalid interpolation indices: {interp_key}.")

        def process_interpolation(cond: str,
                                i: int,
                                o: int,
                                interp_data: Union[Tuple[np.ndarray, str],
                                                    Tuple[np.ndarray, np.ndarray, bool, str]],
                                pairs_with_star: List[Union[Tuple[int, int], Tuple[str, str]]],
                                pairs_no_star: List[Tuple[int, int]],
                                comp_type: str,
                                interpolation_indices: str,
                                has_quadratic: bool,
                                has_linear: bool,
                                ) -> None:
            r"""
            Internal dispatcher for interpolation-index patterns.

            Selects the handler matching `interpolation_indices` and applies it to the
            provided pair lists.
            """
            interp_key = str(interpolation_indices)
            handler = handlers.get(interp_key)
            if handler is None:
                raise ValueError(f"Error: Invalid interpolation indices: {interpolation_indices}.")
            handler(cond, i, o, interp_data, pairs_with_star, pairs_no_star, comp_type, has_quadratic, has_linear)

        # Cache interpolation data and validate expected pair dimensions once.
        component_data: Dict[int, List[Tuple[Union[Tuple[np.ndarray, str],
                                                  Tuple[np.ndarray, np.ndarray, bool, str]], str, bool, bool]]] = {}
        for i in range(1, m + 1):
            is_op = i in op_components
            data = prob.get_component_data(i)
            validated: List[Tuple[Union[Tuple[np.ndarray, str],
                                        Tuple[np.ndarray, np.ndarray, bool, str]], str, bool, bool]] = []
            for o, interp_data in enumerate(data):
                interp_idx = interp_data[1] if is_op else interp_data[3]
                interp_key = str(interp_idx)
                expected_len = _expected_pairs_len(interp_key)
                expected_dim = 2 * expected_len
                if is_op:
                    M, _ = interp_data
                    if getattr(M, 'shape', None) != (expected_dim, expected_dim):
                        raise ValueError(
                            f"Interpolation matrix for component {i}, condition {o} must have "
                            f"shape ({expected_dim}, {expected_dim}) for indices {interp_key}. "
                            f"Got {getattr(M, 'shape', None)}."
                        )
                    has_quadratic = bool(np.any(M))
                    validated.append((interp_data, interp_key, has_quadratic, False))
                else:
                    M, a, _eq, _ = interp_data
                    if getattr(a, 'shape', None) != (expected_len,):
                        raise ValueError(
                            f"Interpolation vector for component {i}, condition {o} must have "
                            f"length {expected_len} for indices {interp_key}. Got {getattr(a, 'shape', None)}."
                        )
                    if getattr(M, 'shape', None) != (expected_dim, expected_dim):
                        raise ValueError(
                            f"Interpolation matrix for component {i}, condition {o} must have "
                            f"shape ({expected_dim}, {expected_dim}) for indices {interp_key}. "
                            f"Got {getattr(M, 'shape', None)}."
                        )
                    has_quadratic = bool(np.any(M))
                    has_linear = bool(np.any(a))
                    validated.append((interp_data, interp_key, has_quadratic, has_linear))
            component_data[i] = validated

        # Precompute (j, k) pairs per (condition, component) to reduce inner-loop overhead.
        pairs_cache: Dict[str, Dict[int, Tuple[List[Union[Tuple[int, int], Tuple[str, str]]], List[Tuple[int, int]]]]] = {}
        for cond in conds:
            pairs_cache[cond] = {}
            k_max = k_maxs[cond]
            k_range = range(k_max + 1)
            for i in range(1, m + 1):
                # Include star once; it represents the fixed-point reference.
                m_bar_i = m_bar_is[i - 1]
                pairs_no_star = [(j, k) for j in range(1, m_bar_i + 1) for k in k_range]
                pairs_with_star = pairs_no_star + [star_pair]
                pairs_cache[cond][i] = (pairs_with_star, pairs_no_star)

        # ---------------------------------------------------------------------
        # Loop over all active conditions and components to process interpolation data.
        # ---------------------------------------------------------------------
        for cond in conds:
            for i in range(1, m + 1):
                is_op = i in op_components
                comp_type = 'op' if is_op else 'func'
                pairs_with_star, pairs_no_star = pairs_cache[cond][i]
                # Retrieve the interpolation data for component i.
                for o, (interp_data, interp_key, has_quadratic, has_linear) in enumerate(component_data[i]):
                    process_interpolation(
                        cond,
                        i,
                        o,
                        interp_data,
                        pairs_with_star,
                        pairs_no_star,
                        comp_type,
                        interp_key,
                        has_quadratic,
                        has_linear,
                    )

        # ---------------------------------------------------------------------
        # Add final constraints to the model.
        # ---------------------------------------------------------------------
        for cond in conds:
            # Enforce that the PSD constraint sums belong to the PSD cone.
            Mod.constraint(PSD_constraint_sums[cond], Domain.inPSDCone(n + (k_maxs[cond] + 1) * m_bar + m))
            # For functional components, enforce the equality constraint.
            if m_func > 0:
                Mod.constraint(eq_constraint_sums[cond] == 0)

        return Mod

    @staticmethod
    def verify_iteration_independent_Lyapunov(
            prob: Type[InclusionProblem],
            algo: Type[Algorithm],
            P: np.ndarray,
            T: np.ndarray,
            p: Optional[np.ndarray] = None,
            t: Optional[np.ndarray] = None,
            rho: float = 1.0,
            h: int = 0,
            alpha: int = 0,
            Q_equals_P: bool = False,
            S_equals_T: bool = False,
            q_equals_p: bool = False,
            s_equals_t: bool = False,
            remove_C2: bool = False,
            remove_C3: bool = False,
            remove_C4: bool = True
    ) -> bool:
        r"""
        Verify feasibility of an iteration-independent Lyapunov inequality via an SDP.

        This method formulates and solves a semidefinite program (SDP) in MOSEK Fusion
        for a given inclusion problem, algorithm, and user-specified targets
        :math:`(P,p,T,t,\rho,h,\alpha)`.

        Mathematical notation, star-variable definitions, Lyapunov ansatzes, and the roles
        of :math:`h` and :math:`\alpha` follow the class-level reference in
        :class:`~autolyap.iteration_independent.IterationIndependent`.

        **Enforced inequalities**

        The quadratic Lyapunov inequalities (C1)–(C4) are defined in the class-level
        **Quadratic Lyapunov inequality** section.
        This method enforces the enabled subset of those inequalities.
        The flags `remove_C2`, `remove_C3`, and `remove_C4` disable (C2), (C3), and (C4),
        respectively (`remove_C4 = True` by default).

        **User-specified targets**

        The tuple :math:`(P,p,T,t)` defines the target lower bounds through
        :math:`\mathcal{V}(P,p,k)` and :math:`\mathcal{R}(T,t,k)`.
        The user is responsible for ensuring these target functions are nonnegative
        for the relevant iterates and problem class.

        The SDP searches for a certificate :math:`(Q,q,S,s)` consistent with those targets.

        **Parameters**

        - `prob` (:class:`~typing.Type`\[:class:`~autolyap.problemclass.InclusionProblem`\]): An
          :class:`~autolyap.problemclass.InclusionProblem`
          instance containing interpolation conditions.
        - `algo` (:class:`~typing.Type`\[:class:`~autolyap.algorithms.Algorithm`\]): An
          :class:`~autolyap.algorithms.Algorithm` instance providing
          dimensions and methods to compute matrices.
        - `P` (:class:`numpy.ndarray`): Candidate symmetric matrix corresponding to
          :math:`P \in \sym^{n + (h+1)\NumEval + m}`.
        - `T` (:class:`numpy.ndarray`): Candidate symmetric matrix corresponding to
          :math:`T \in \sym^{n + (h+\alpha+2)\NumEval + m}`.
        - `p` (:class:`~typing.Optional`\[:class:`numpy.ndarray`\]): Candidate vector corresponding to
          :math:`p \in \mathbb{R}^{(h+1)\NumEvalFunc + \NumFunc}` for functional components
          (required if :math:`\NumFunc > 0`).
        - `t` (:class:`~typing.Optional`\[:class:`numpy.ndarray`\]): Candidate vector corresponding to
          :math:`t \in \mathbb{R}^{(h+\alpha+2)\NumEvalFunc + \NumFunc}` for functional components
          (required if :math:`\NumFunc > 0`).
        - `rho` (:class:`float`): A scalar contraction parameter corresponding to :math:`\rho` used in forming the Lyapunov inequality
          (typically :math:`\rho \in [0,1]`).
        - `h` (:class:`int`): Nonnegative integer corresponding to :math:`h` defining history.
        - `alpha` (:class:`int`): Nonnegative integer corresponding to :math:`\alpha` defining overlap.
        - `Q_equals_P` (:class:`bool`): If True, sets Q equal to P.
        - `S_equals_T` (:class:`bool`): If True, sets S equal to T.
        - `q_equals_p` (:class:`bool`): For functional components, if True, sets q equal to p.
        - `s_equals_t` (:class:`bool`): For functional components, if True, sets s equal to t.
        - `remove_C2` (:class:`bool`): Flag to remove constraint C2.
        - `remove_C3` (:class:`bool`): Flag to remove constraint C3.
        - `remove_C4` (:class:`bool`): Flag to remove constraint C4.

        **Returns**

        - (:class:`bool`): Returns `True` if the SDP is solved successfully. In that case, there exist
          :math:`(Q,q,S,s)` such that the enabled constraints among (C1)–(C4) hold, with
          (C2)–(C4) possibly removed. Returns `False` otherwise.

        **Raises**

        - `ValueError`: If input dimensions or other conditions are violated.

        **Notes**

        - Requires a working MOSEK installation and license.
        - The inputs `(P, p, T, t)` are typically built with
          :class:`~autolyap.iteration_independent.LinearConvergence` and
          :class:`~autolyap.iteration_independent.SublinearConvergence` helper methods.
        """
        h, alpha, _, _, _, _, _, _, _ = IterationIndependent._validate_iteration_independent_inputs(
            prob, algo, P, T, p, t, h, alpha
        )
        rho = ensure_real_number(rho, "rho", finite=True, minimum=0.0)

        Mod = IterationIndependent._build_iteration_independent_model(
            prob,
            algo,
            P,
            T,
            p,
            t,
            h,
            alpha,
            Q_equals_P,
            S_equals_T,
            q_equals_p,
            s_equals_t,
            remove_C2,
            remove_C3,
            remove_C4,
            rho_term=rho,
        )
        try:
            Mod.solve()
            Mod.primalObjValue()
        except OptimizeError as e:
            licence_markers = (
                "err_license_max",         # 1016 – all floating tokens in use
                "err_license_server",      # 1015 – server unreachable / down
                "err_missing_license_file" # 1008 – no licence file / server path
            )
            if any(mark in str(e) for mark in licence_markers):
                raise
            return False
        except Exception as e:
            # Uncomment the following line for debugging if needed.
            # print("Error during solve: {0}".format(e))
            return False
        finally:
            Mod.dispose()
        return True
    
    @staticmethod
    def _compute_Thetas(algo: Type[Algorithm], h: int, alpha: int, condition: str = 'C1') -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute the Theta matrices (capital :math:`\Theta`) using the :math:`X` matrices.

        For **condition "C1"**:
        
        - Set :math:`k_{\textup{min}} = 0` and :math:`k_{\textup{max}} = h+\alpha+1`.
        - Retrieve :math:`X = X_{\alpha+1}` from :meth:`~autolyap.algorithms.Algorithm.get_Xs`
          with `k_min = 0` and `k_max = h+\alpha+1`.
        - :math:`\Theta_0` is of size :math:`[n + (h+1)\NumEval + m] \times [n + (h+\alpha+2)\NumEval + m]`.
        - :math:`\Theta_1` is formed by vertically stacking :math:`X` with a block row 
          consisting of a zero block and an identity matrix.

        For **condition "C4"**:
        
        - Set :math:`k_{\textup{min}} = 0` and :math:`k_{\textup{max}} = h+\alpha+2`.
        - Retrieve :math:`X = X_1` from :meth:`~autolyap.algorithms.Algorithm.get_Xs`
          with `k_min = 0` and `k_max = h+\alpha+2`.
        - :math:`\Theta_0` is of size :math:`[n + (h+\alpha+2)\NumEval + m] \times [n + (h+\alpha+3)\NumEval + m]`.
        - :math:`\Theta_1` is formed similarly by stacking :math:`X` with an appropriate block row.

        **Parameters**

        - `algo` (:class:`~typing.Type`\[:class:`~autolyap.algorithms.Algorithm`\]): An instance of :class:`~autolyap.algorithms.Algorithm`
          (providing `algo.n`, `algo.m_bar`, `algo.m`).
        - `h` (:class:`int`): A nonnegative integer corresponding to :math:`h`.
        - `alpha` (:class:`int`): A nonnegative integer corresponding to :math:`\alpha`.
        - `condition` (:class:`str`): Either "C1" or "C4".
        
        **Returns**

        - (:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`\]): A tuple :math:`(\Theta_0, \Theta_1)`.

        **Raises**

        - `ValueError`: If :math:`h` or :math:`\alpha` is negative or if condition is invalid.
        """
        h = ensure_integral(h, "h", minimum=0)
        alpha = ensure_integral(alpha, "alpha", minimum=0)
        if condition not in ('C1', 'C4'):
            raise ValueError("Condition must be either 'C1' or 'C4'.")
        
        n = algo.n
        m_bar = algo.m_bar
        m = algo.m

        if condition == 'C1':
            k_min, k_max = 0, h + alpha + 1
            Xs = algo.get_Xs(k_min, k_max)
            key = alpha + 1
            if key not in Xs:
                raise ValueError(f"Expected key {key} in X matrices, but it was not found.")
            X_mat = Xs[key]
            Theta0 = np.block([
                [np.eye(n + (h + 1) * m_bar), np.zeros((n + (h + 1) * m_bar, (alpha + 1) * m_bar)), np.zeros((n + (h + 1) * m_bar, m))],
                [np.zeros((m, n + (h + 1) * m_bar)), np.zeros((m, (alpha + 1) * m_bar)), np.eye(m)]
            ])
            lower_block = np.hstack([
                np.zeros(((h + 1) * m_bar + m, n + (alpha + 1) * m_bar)),
                np.eye((h + 1) * m_bar + m)
            ])
            Theta1 = np.vstack([X_mat, lower_block])
            return Theta0, Theta1

        elif condition == 'C4':
            k_min, k_max = 0, h + alpha + 2
            Xs = algo.get_Xs(k_min, k_max)
            key = 1
            if key not in Xs:
                raise ValueError(f"Expected key {key} in X matrices, but it was not found.")
            X_mat = Xs[key]
            Theta0 = np.block([
                [np.eye(n + (h + alpha + 2) * m_bar), np.zeros((n + (h + alpha + 2) * m_bar, m_bar)), np.zeros((n + (h + alpha + 2) * m_bar, m))],
                [np.zeros((m, n + (h + alpha + 2) * m_bar)), np.zeros((m, m_bar)), np.eye(m)]
            ])
            lower_block = np.hstack([
                np.zeros(((h + alpha + 2) * m_bar + m, n + m_bar)),
                np.eye((h + alpha + 2) * m_bar + m)
            ])
            Theta1 = np.vstack([X_mat, lower_block])
            return Theta0, Theta1

        # Should never reach here.
        raise ValueError("Unexpected error in _compute_Thetas.")

    @staticmethod
    def _compute_thetas(algo: Type[Algorithm], h: int, alpha: int, condition: str = 'C1') -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute the theta matrices (lowercase :math:`\theta`) for functional evaluations.

        For **condition "C1"**:
        
        - :math:`\theta_0 \in \mathbb{R}^{((h+1)\NumEvalFunc+\NumFunc) \times ((h+\alpha+2)\NumEvalFunc+\NumFunc)}` 
          is given by a block matrix with an identity in the upper left and lower right.
        - :math:`\theta_1` is formed as a horizontal block consisting of a zero block and an identity matrix.

        For **condition "C4"**:
        
        - :math:`\theta_0 \in \mathbb{R}^{((h+\alpha+2)\NumEvalFunc+\NumFunc) \times ((h+\alpha+3)\NumEvalFunc+\NumFunc)}` 
          is defined similarly.
        - :math:`\theta_1` is a horizontal block with a zero block and an identity matrix.

        **Parameters**

        - `algo` (:class:`~typing.Type`\[:class:`~autolyap.algorithms.Algorithm`\]): An instance of :class:`~autolyap.algorithms.Algorithm`
          (providing `algo.m_bar_func` and `algo.m_func`).
        - `h` (:class:`int`): A nonnegative integer corresponding to :math:`h`.
        - `alpha` (:class:`int`): A nonnegative integer corresponding to :math:`\alpha`.
        - `condition` (:class:`str`): Either "C1" or "C4".

        **Returns**

        - (:class:`~typing.Tuple`\[:class:`numpy.ndarray`, :class:`numpy.ndarray`\]): A tuple :math:`(\theta_0, \theta_1)`.

        **Raises**

        - `ValueError`: If :math:`h` or :math:`\alpha` is negative, if `condition` is invalid,
          or if there are no functional components (i.e., :math:`\NumFunc \leq 0`).
        """
        h = ensure_integral(h, "h", minimum=0)
        alpha = ensure_integral(alpha, "alpha", minimum=0)
        if condition not in ('C1', 'C4'):
            raise ValueError("Condition must be either 'C1' or 'C4'.")
        
        m_bar_func = algo.m_bar_func
        m_func = algo.m_func
        
        # Theta matrices are only defined when there is at least one functional component.
        if m_func <= 0:
            raise ValueError("Theta matrices require at least one functional component (m_func > 0).")
        
        if condition == 'C1':
            theta0 = np.block([
                [np.eye((h + 1) * m_bar_func), np.zeros(((h + 1) * m_bar_func, (alpha + 1) * m_bar_func)), 
                np.zeros(((h + 1) * m_bar_func, m_func))],
                [np.zeros((m_func, (h + 1) * m_bar_func)), np.zeros((m_func, (alpha + 1) * m_bar_func)), 
                np.eye(m_func)]
            ])
            theta1 = np.hstack([
                np.zeros(((h + 1) * m_bar_func + m_func, (alpha + 1) * m_bar_func)),
                np.eye((h + 1) * m_bar_func + m_func)
            ])
            return theta0, theta1

        elif condition == 'C4':
            theta0 = np.block([
                [np.eye((h + alpha + 2) * m_bar_func), np.zeros(((h + alpha + 2) * m_bar_func, m_bar_func)), 
                np.zeros(((h + alpha + 2) * m_bar_func, m_func))],
                [np.zeros((m_func, (h + alpha + 2) * m_bar_func)), np.zeros((m_func, m_bar_func)), 
                np.eye(m_func)]
            ])
            theta1 = np.hstack([
                np.zeros(((h + alpha + 2) * m_bar_func + m_func, m_bar_func)),
                np.eye((h + alpha + 2) * m_bar_func + m_func)
            ])
            return theta0, theta1

        # Should never reach here.
        raise ValueError("Unexpected error in _compute_thetas.")
