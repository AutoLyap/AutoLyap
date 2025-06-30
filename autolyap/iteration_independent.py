import numpy as np
from typing import Type, Optional, Tuple, Union, List
from itertools import combinations, product
from mosek.fusion import Model, Domain, OptimizeError
import mosek.fusion.pythonic
from autolyap.utils.helper_functions import create_symmetric_matrix_expression
from autolyap.problemclass import InclusionProblem
from autolyap.algorithms import Algorithm

class LinearConvergence:
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
        Compute the matrices for the distance to solution.

        This method computes the following matrix

        .. math::
            P = \left( P_{(i,j)}\, Y_\tau^{0,h} - P_{(i,\star)}\, Y_\star^{0,h} \right)^T
                \left( P_{(i,j)}\, Y_\tau^{0,h} - P_{(i,\star)}\, Y_\star^{0,h} \right),

        and constructs :math:`T` (and, if functional components exist, the vectors :math:`p` and :math:`t`) as zero.

        **Definitions**

        - :math:`Y_\tau^{0,h}` is the Y matrix at iteration :math:`\tau` over the horizon :math:`[0, h]`.
        - :math:`Y_\star^{0,h}` is the “star” Y matrix over :math:`[0, h]`.
        - :math:`P_{(i,j)}` and :math:`P_{(i,\star)}` are the projection matrices for component :math:`i`.

        **Dimensions**

        - :math:`\dim(P) = n + (h+1)\bar{m} + m,`
        - :math:`\dim(T) = n + (h+\alpha+2)\bar{m} + m,`
        - If :math:`algo.m_func > 0`:
        
          - :math:`\mathrm{len}(p) = (h+1)\bar{m}_{\text{func}} + m\_func,`
          - :math:`\mathrm{len}(t) = (h+\alpha+2)\bar{m}_{\text{func}} + m\_func.`

        **Parameters**
        
        :param algo: An instance of Algorithm. It must provide:
                   - ``algo.m`` (total number of components),
                   - ``algo.m_bar_is`` (a list where the *i*-th entry gives the number of evaluations for component *i*),
                   - Methods ``get_Ys(k_min, k_max)`` and ``get_Ps()``.
        :param h: A nonnegative integer defining the time horizon :math:`[0, h]` for Y matrices.
        :param alpha: A nonnegative integer for extending the horizon for :math:`T` (and :math:`t`).
        :param i: Component index (1-indexed). Default is 1; must satisfy :math:`1 \le i \le algo.m`.
        :param j: Evaluation index for component :math:`i`. Default is 1; must satisfy :math:`1 \le j \le algo.m_bar_is[i-1]`.
        :param tau: Iteration index. Default is 0; must satisfy :math:`0 \le \tau \le h`.

        **Returns**
        
        - If :math:`algo.m_func == 0`:
          
          :return: A tuple ``(P, T)``
        
        - Otherwise:
        
          :return: A tuple ``(P, p, T, t)``, where :math:`P` is computed as above and :math:`T`, :math:`p`, and :math:`t` are zero arrays with the appropriate dimensions.

        **Raises**
        
        :raises ValueError: If any input is out of its valid range or if required matrices are missing.
        """
        # ----- Input Checking -----
        if not isinstance(h, int) or h < 0:
            raise ValueError("Parameter h must be a nonnegative integer.")
        if not isinstance(alpha, int) or alpha < 0:
            raise ValueError("Parameter alpha must be a nonnegative integer.")
        
        if not isinstance(i, int) or i < 1 or i > algo.m:
            raise ValueError(f"Component index i must be in [1, {algo.m}]. Got {i}.")
        
        num_eval = algo.m_bar_is[i - 1]
        if not isinstance(j, int) or j < 1 or j > num_eval:
            raise ValueError(f"For component {i}, evaluation index j must be in [1, {num_eval}]. Got {j}.")
        
        if not isinstance(tau, int) or tau < 0 or tau > h:
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
        Compute the matrices/vectors for function-value suboptimality.

        This function is only applicable when :math:`m = m\_func = 1`.

        It returns a tuple :math:`(P, p, T, t)` where:

        - :math:`p` is computed as

          .. math::
              p = \left( F_{(1,j,tau)}^{0,h} - F_{(1,\star,\star)}^{0,h} \right)^T,

          with :math:`p` returned as a 1D numpy array of length 
          :math:`(h+1)\bar{m}_{\text{func}} + m\_func`.
        - :math:`P` is a zero matrix of dimension

          .. math::
              n + (h+1)\bar{m} + m,
          
        - :math:`T` is a zero matrix of dimension

          .. math::
              n + (h+\alpha+2)\bar{m} + m,
          
        - :math:`t` is a zero vector of dimension

          .. math::
              (h+\alpha+2)\bar{m}_{\text{func}} + m\_func.

        **Parameters**

        :param algo: An instance of Algorithm. It must satisfy :math:`algo.m = 1` and :math:`algo.m\_func = 1`.
        :param h: A nonnegative integer defining the horizon :math:`[0, h]` for F matrices.
        :param alpha: A nonnegative integer for extending the horizon for :math:`T` and :math:`t`.
        :param j: Evaluation index for component 1. Default is 1; must satisfy :math:`1 \le j \le algo.m_bar_is[0]`.
        :param tau: Iteration index. Default is 0; must satisfy :math:`0 \le \tau \le h`.

        **Returns**

        :return: A tuple :math:`(P, p, T, t)`, where :math:`p` is computed as above (a 1D numpy array)
                 and :math:`P`, :math:`T`, and :math:`t` are zero arrays with appropriate dimensions.

        **Raises**

        :raises ValueError: If :math:`algo.m \ne 1` or :math:`algo.m\_func \ne 1`, if any input parameter is out of range,
                            or if the required F matrices are not found.
        """
        # ----- Check that m and m_func equal 1 -----
        if algo.m != 1 or algo.m_func != 1:
            raise ValueError("get_parameters_function_value_suboptimality is only applicable when m = m_func = 1.")
        
        # ----- Validate inputs -----
        if not isinstance(h, int) or h < 0:
            raise ValueError("Parameter h must be a nonnegative integer.")
        if not isinstance(alpha, int) or alpha < 0:
            raise ValueError("Parameter alpha must be a nonnegative integer.")
        
        num_eval = algo.m_bar_is[0]
        if not isinstance(j, int) or j < 1 or j > num_eval:
            raise ValueError(f"For component 1, evaluation index j must be in [1, {num_eval}]. Got {j}.")
        
        if not isinstance(tau, int) or tau < 0 or tau > h:
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
        Perform a bisection search to find the minimal contraction parameter :math:`\rho`.

        This method performs a bisection search over :math:`\rho` in the interval 
        :math:`[{\text{lower\_bound}}, {\text{upper\_bound}}]` to find the minimal value for which the 
        iteration-independent Lyapunov inequality holds. At each iteration it calls
        ``verify_iteration_independent_Lyapunov`` until the interval size is below :math:`{\text{tol}}`.

        **Parameters**

        :param prob: An InclusionProblem instance containing interpolation conditions.
        :param algo: An Algorithm instance providing dimensions and methods.
        :param P: A symmetric matrix of dimension 
                  :math:`n + (h+1)\bar{m} + m` by :math:`n + (h+1)\bar{m} + m`.
        :param T: A symmetric matrix of dimension 
                  :math:`n + (h+\alpha+2)\bar{m} + m` by :math:`n + (h+\alpha+2)\bar{m} + m`.
        :param p: A vector for functional components (if applicable).
        :param t: A vector for functional components (if applicable).
        :param h: Nonnegative integer defining the history for the matrices.
        :param alpha: Nonnegative integer for extending the horizon.
        :param Q_equals_P: If True, set Q equal to P.
        :param S_equals_T: If True, set S equal to T.
        :param q_equals_p: For functional components, if True, set q equal to p.
        :param s_equals_t: For functional components, if True, set s equal to t.
        :param remove_C2: Flag to remove constraint C2.
        :param remove_C3: Flag to remove constraint C3.
        :param remove_C4: Flag to remove constraint C4.
        :param lower_bound: Lower bound for :math:`\rho`.
        :param upper_bound: Upper bound for :math:`\rho`.
        :param tol: Tolerance for the bisection search stopping criterion.

        **Returns**

        :return: The minimal :math:`\rho` in :math:`[{\text{lower\_bound}}, {\text{upper\_bound}}]` that verifies the 
                 Lyapunov inequality within tolerance :math:`{\text{tol}}`, or ``None`` if the inequality does not hold 
                 at the upper bound.
        """
        # Ensure that the inequality holds at the initial upper bound.
        if not IterationIndependent.verify_iteration_independent_Lyapunov(
                prob, algo, P, T, p, t, rho=upper_bound, h=h, alpha=alpha,
                Q_equals_P=Q_equals_P, S_equals_T=S_equals_T,
                q_equals_p=q_equals_p, s_equals_t=s_equals_t,
                remove_C2=remove_C2, remove_C3=remove_C3, remove_C4=remove_C4):
            return None

        l = lower_bound
        u = upper_bound
        while (u - l) > tol:
            mid = (l + u) / 2.0
            if IterationIndependent.verify_iteration_independent_Lyapunov(
                    prob, algo, P, T, p, t, rho=mid, h=h, alpha=alpha,
                    Q_equals_P=Q_equals_P, S_equals_T=S_equals_T,
                    q_equals_p=q_equals_p, s_equals_t=s_equals_t,
                    remove_C2=remove_C2, remove_C3=remove_C3, remove_C4=remove_C4):
                u = mid
            else:
                l = mid

        return u

class SublinearConvergence:
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
        Compute the matrices for the fixed-point residual.

        For a given iteration index :math:`\tau` (with :math:`0 \le \tau \le h+\alpha+1`), define

        .. math::
            T = \left( X_{\tau+1}^{0, h+\alpha+1} - X_{\tau}^{0, h+\alpha+1} \right)^T
                \left( X_{\tau+1}^{0, h+\alpha+1} - X_{\tau}^{0, h+\alpha+1} \right),

        where :math:`X_{\tau}^{0, h+\alpha+1}` is the X matrix computed over the horizon :math:`[0, h+\alpha+1]` 
        (via ``algo.get_Xs``).

        **Dimensions**

        - :math:`P` is a zero matrix of dimension :math:`n + (h+1)\bar{m} + m`.
        - :math:`T` is computed as above and has dimension :math:`n + (h+\alpha+2)\bar{m} + m`.
        - If :math:`algo.m_func > 0`:
        
          - :math:`p` is a zero vector of length :math:`(h+1)\bar{m}_{\text{func}} + m\_func,`
          - :math:`t` is a zero vector of length :math:`(h+\alpha+2)\bar{m}_{\text{func}} + m\_func.`

        **Parameters**

        :param algo: An instance of Algorithm.
        :param h: A nonnegative integer defining the time horizon :math:`[0, h]` for :math:`P`.
        :param alpha: A nonnegative integer for extending the horizon for :math:`T` (and :math:`t`).
        :param tau: Iteration index for computing the fixed-point residual.
                  Must satisfy :math:`0 \le \tau \le h+\alpha+1`.

        **Returns**

        - If :math:`algo.m_func == 0`:
          
          :return: A tuple ``(P, T)``.
        
        - Otherwise:
          
          :return: A tuple ``(P, p, T, t)``.

        **Raises**

        :raises ValueError: If any input parameter is out of its valid range or if the required X matrices are missing.
        """
        # ----- Input Checking -----
        if not isinstance(h, int) or h < 0:
            raise ValueError("Parameter h must be a nonnegative integer.")
        if not isinstance(alpha, int) or alpha < 0:
            raise ValueError("Parameter alpha must be a nonnegative integer.")
        if not isinstance(tau, int) or tau < 0 or tau > h + alpha + 1:
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
        Compute the matrices for the duality gap.

        For a given iteration index :math:`\tau` (with :math:`0 \le \tau \le h+\alpha+1`), define

        .. math::
            T = -\frac{1}{2} \sum_{i=1}^{m} \begin{bmatrix}
            P_{(i,\star)}\, U_\star^{0,h+\alpha+1} \\
            P_{(i,1)}\, Y_\tau^{0,h+\alpha+1}
            \end{bmatrix}^T
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
            t = \sum_{i=1}^{m} \left( F_{(i,1,\tau)}^{0,h+\alpha+1} - F_{(i,\star,\star)}^{0,h+\alpha+1} \right)^T.

        All other matrices are set to zero.

        **Requirements**

        It is required that :math:`m = m\_func` (i.e. all components are functional).

        **Dimensions**

        - :math:`P` is a zero matrix of dimension :math:`n + (h+1)\bar{m} + m`.
        - :math:`p` is a zero vector of length :math:`(h+1)\bar{m}_{\text{func}} + m\_func`.
        - :math:`T` is computed as above and has dimension :math:`n + (h+\alpha+2)\bar{m} + m`.
        - :math:`t` is computed as above and has length :math:`(h+\alpha+2)\bar{m}_{\text{func}} + m\_func`.

        **Parameters**

        :param algo: An instance of Algorithm (with :math:`m = m\_func`).
        :param h: A nonnegative integer defining the time horizon :math:`[0, h]` for :math:`P`.
        :param alpha: A nonnegative integer for extending the horizon for :math:`T` and :math:`t`.
        :param tau: Iteration index for computing the duality gap.
                  Must satisfy :math:`0 \le \tau \le h+\alpha+1`.

        **Returns**

        :return: A tuple :math:`(P, p, T, t)`, where :math:`t` is a one-dimensional array.

        **Raises**

        :raises ValueError: If any input parameter is out of its valid range, if the required matrices
                            are missing, or if :math:`m \ne m\_func`.
        """
        # ----- Check that m = m_func -----
        if algo.m != algo.m_func:
            raise ValueError("get_parameters_duality_gap is only applicable when m = m_func.")
        
        # ----- Input Checking -----
        if not isinstance(h, int) or h < 0:
            raise ValueError("Parameter h must be a nonnegative integer.")
        if not isinstance(alpha, int) or alpha < 0:
            raise ValueError("Parameter alpha must be a nonnegative integer.")
        if not isinstance(tau, int) or tau < 0 or tau > h + alpha + 1:
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
            # Contribution from component i: block^T mid block.
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
        Compute the matrices/vectors for function value suboptimality.

        This function is only applicable when :math:`m = m\_func = 1`.

        It returns a tuple :math:`(P, p, T, t)` where:

        - :math:`t` is computed as

          .. math::
              t = \left( F_{(1,j,\tau)}^{0,h+\alpha+1} - F_{(1,\star,\star)}^{0,h+\alpha+1} \right)^T,

          with :math:`t` returned as a 1D numpy array of length 
          :math:`(h+\alpha+2)\bar{m}_{\text{func}} + m\_func`.
        - :math:`P` is a zero matrix of dimension

          .. math::
              n + (h+1)\bar{m} + m,
          
        - :math:`T` is a zero matrix of dimension

          .. math::
              n + (h+\alpha+2)\bar{m} + m,
          
        - :math:`t` is a zero vector of dimension

          .. math::
              (h+1)\bar{m}_{\text{func}} + m\_func.

        **Parameters**

        :param algo: An instance of Algorithm. It must satisfy :math:`algo.m = 1` and :math:`algo.m\_func = 1`.
        :param h: A nonnegative integer defining the horizon :math:`[0, h]` for F matrices.
        :param alpha: A nonnegative integer for extending the horizon for :math:`T` and :math:`t`.
        :param j: Evaluation index for component 1. Default is 1; must satisfy :math:`1 \le j \le algo.m_bar_is[0]`.
        :param tau: Iteration index. Default is 0; must satisfy :math:`0 \le \tau \le h+\alpha+1`.

        **Returns**

        :return: A tuple :math:`(P, p, T, t)`, where :math:`t` is computed as above (a 1D numpy array)
                 and :math:`P`, :math:`T`, and :math:`p` are zero arrays with appropriate dimensions.

        **Raises**

        :raises ValueError: If :math:`algo.m \ne 1` or :math:`algo.m\_func \ne 1`, if any input parameter is out of range,
                            or if the required F matrices are not found.
        """
        # ----- Check that m and m_func equal 1 -----
        if algo.m != 1 or algo.m_func != 1:
            raise ValueError("get_parameters_function_value_suboptimality is only applicable when m = m_func = 1.")
        
        # ----- Validate inputs -----
        if not isinstance(h, int) or h < 0:
            raise ValueError("Parameter h must be a nonnegative integer.")
        if not isinstance(alpha, int) or alpha < 0:
            raise ValueError("Parameter alpha must be a nonnegative integer.")
        
        num_eval = algo.m_bar_is[0]
        if not isinstance(j, int) or j < 1 or j > num_eval:
            raise ValueError(f"For component 1, evaluation index j must be in [1, {num_eval}]. Got {j}.")
        
        if not isinstance(tau, int) or tau < 0 or tau > h:
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
        Compute the matrices for the optimality measure.

        For a given iteration index :math:`\tau` (with :math:`0 \le \tau \le h+\alpha+1`), define

        .. math::
            T =
            \begin{cases}
              \left( P_{(1,1)}\, U_\tau^{0,h+\alpha+1} \right)^T \left( P_{(1,1)}\, U_\tau^{0,h+\alpha+1} \right)
              & \text{if } m = 1, \\[1em]
              \left( \left( \sum_{i=1}^{m} P_{(i,1)}\, U_\tau^{0,h+\alpha+1} \right)^T \left( \sum_{i=1}^{m} P_{(i,1)}\, U_\tau^{0,h+\alpha+1} \right)
              + \sum_{i=2}^{m} \left( \left( P_{(1,1)} - P_{(i,1)} \right) Y_\tau^{0,h+\alpha+1} \right)^T \left( \left( P_{(1,1)} - P_{(i,1)} \right) Y_\tau^{0,h+\alpha+1} \right) \right)
              & \text{if } m > 1.
            \end{cases}

        All other matrices are set to zero.

        **Dimensions**

        - :math:`P` is a zero matrix of dimension :math:`n + (h+1)\bar{m} + m`.
        - If :math:`algo.m_func > 0`:
        
          - :math:`p` is a zero vector of length :math:`(h+1)\bar{m}_{\text{func}} + m\_func,`
          - :math:`t` is a zero vector of length :math:`(h+\alpha+2)\bar{m}_{\text{func}} + m\_func.`

        **Parameters**

        :param algo: An instance of Algorithm.
        :param h: A nonnegative integer defining the time horizon :math:`[0, h]` for :math:`P`.
        :param alpha: A nonnegative integer for extending the horizon for :math:`T` (and :math:`t`).
        :param tau: Iteration index for computing the optimality measure.
                  Must satisfy :math:`0 \le \tau \le h+\alpha+1`.

        **Returns**

        - If :math:`algo.m_func == 0`:
          
          :return: A tuple ``(P, T)``.
        
        - Otherwise:
          
          :return: A tuple ``(P, p, T, t)``.

        **Raises**

        :raises ValueError: If any input parameter is out of its valid range or if required matrices are missing.
        """
        # ----- Input Checking -----
        if not isinstance(h, int) or h < 0:
            raise ValueError("Parameter h must be a nonnegative integer.")
        if not isinstance(alpha, int) or alpha < 0:
            raise ValueError("Parameter alpha must be a nonnegative integer.")
        if not isinstance(tau, int) or tau < 0 or tau > h + alpha + 1:
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
    
    LinearConvergence = LinearConvergence
    SublinearConvergence = SublinearConvergence

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
        Verify an iteration-independent Lyapunov inequality via an SDP.

        This method sets up and solves a semidefinite program (SDP) using Mosek Fusion to 
        verify a Lyapunov inequality for a given inclusion problem and algorithm.

        **Parameters**

        :param prob: An InclusionProblem instance containing interpolation conditions.
        :param algo: An Algorithm instance providing dimensions and methods to compute matrices.
        :param P: A symmetric numpy array of dimension 
                  :math:`n + (h+1)\bar{m} + m` by :math:`n + (h+1)\bar{m} + m`.
        :param T: A symmetric numpy array of dimension 
                  :math:`n + (h+\alpha+2)\bar{m} + m` by :math:`n + (h+\alpha+2)\bar{m} + m`.
        :param p: A vector for functional components (if any) with appropriate dimensions.
        :param t: A vector for functional components (if any) with appropriate dimensions.
        :param rho: A scalar contraction parameter used in forming the Lyapunov inequality.
        :param h: Nonnegative integer defining history.
        :param alpha: Nonnegative integer defining overlap.
        :param Q_equals_P: If True, sets Q equal to P.
        :param S_equals_T: If True, sets S equal to T.
        :param q_equals_p: For functional components, if True, sets q equal to p.
        :param s_equals_t: For functional components, if True, sets s equal to t.
        :param remove_C2: Flag to remove constraint C2.
        :param remove_C3: Flag to remove constraint C3.
        :param remove_C4: Flag to remove constraint C4.

        **Returns**

        :return: True if the SDP is solved successfully (i.e. a Lyapunov inequality exists), False otherwise.

        **Raises**

        :raises ValueError: If input dimensions or other conditions are violated.
        """

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
        if h < 0:
            raise ValueError("Parameter 'h' must be >= 0.")
        if alpha < 0:
            raise ValueError("Parameter 'alpha' must be >= 0.")
        
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
            raise ValueError(f"P must be a symmetric matrix of dimension {dim_P}x{dim_P}. Got shape {P.shape}.")
        if not np.allclose(P, P.T, atol=1e-8):
            raise ValueError("P must be symmetric.")
        
        # Expected dimension for matrix T: [n + (h+alpha+2)*m_bar + m] x [n + (h+alpha+2)*m_bar + m].
        dim_T = n + (h + alpha + 2) * m_bar + m
        if not (isinstance(T, np.ndarray) and T.ndim == 2 and T.shape[0] == T.shape[1] == dim_T):
            raise ValueError(f"T must be a symmetric matrix of dimension {dim_T}x{dim_T}. Got shape {T.shape}.")
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


        # -------------------------------------------------------------------------
        # Set up the Mosek Fusion model and decision variables.
        # -------------------------------------------------------------------------
        with Model() as Mod:
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
            Ws["C1"] = Theta1_C1.T @ Q @ Theta1_C1 - rho * Theta0_C1.T @ Q @ Theta0_C1 + S
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
                ws["C1"] = (theta1_C1.T - rho * theta0_C1.T) @ q + s

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
            def process_pairs(cond: str, 
                            i: int, 
                            o: int, 
                            interpolation_data: Union[Tuple[np.ndarray, str],
                                                        Tuple[np.ndarray, np.ndarray, bool, str]], 
                            pairs: List[Union[Tuple[int, int], Tuple[str, str]]], 
                            comp_type: str) -> None:
                r"""
                Process a given set of interpolation pairs.

                Constructs a unique key based on condition, component, pairs, and interpolation index.
                For operator conditions, creates a nonnegative multiplier.
                For functional conditions, if the constraint is an equality, creates an unrestricted multiplier;
                otherwise, creates a nonnegative multiplier.
                Updates the PSD and equality constraint sums accordingly.

                :param cond: The condition identifier (e.g. "C1", "C2", etc.).
                :param i: The component index.
                :param o: The interpolation index.
                :param interpolation_data: Interpolation data tuple.
                :param pairs: A list of (j, k) pairs.
                :param comp_type: A string, either 'op' for operator or 'func' for functional.
                """
                key_parts = [cond, f"i:{i}"]
                for (j, k) in pairs:
                    key_parts.append(f"(j:{j},k:{k})")
                key_parts.append(f"o:{o}")
                key = tuple(key_parts)

                if comp_type == 'op':
                    M, _ = interpolation_data
                else:
                    M, a, eq, _ = interpolation_data

                # Compute the lifted matrix W for the given pairs.
                W_matrix = algo.compute_W(i, pairs, 0, k_maxs[cond], M)

                if comp_type == 'op':
                    lambda_var = Mod.variable(1, Domain.greaterThan(0.0))
                    lambdas_op[key] = lambda_var
                    PSD_constraint_sums[cond] = PSD_constraint_sums[cond] + lambda_var[0] * W_matrix
                else:
                    # For functional components, compute the aggregated F vector.
                    F_vector = algo.compute_F_aggregated(i, pairs, 0, k_maxs[cond], a)
                    if eq:
                        nu_var = Mod.variable(1, Domain.unbounded())
                        nus_func[key] = nu_var
                        PSD_constraint_sums[cond] = PSD_constraint_sums[cond] + nu_var[0] * W_matrix
                        eq_constraint_sums[cond] = eq_constraint_sums[cond] + nu_var[0] * F_vector
                    else:
                        lambda_var = Mod.variable(1, Domain.greaterThan(0.0))
                        lambdas_func[key] = lambda_var
                        PSD_constraint_sums[cond] = PSD_constraint_sums[cond] + lambda_var[0] * W_matrix
                        eq_constraint_sums[cond] = eq_constraint_sums[cond] + lambda_var[0] * F_vector

            def process_interpolation(cond: str,
                                    i: int,
                                    o: int,
                                    interp_data: Union[Tuple[np.ndarray, str],
                                                        Tuple[np.ndarray, np.ndarray, bool, str]]
                                    ) -> None:
                r"""
                Process the interpolation condition for a given component.

                Generates all (j, k) pairs (including the special ('star', 'star') case) and 
                processes them according to the type of interpolation indices.

                :param cond: The condition identifier.
                :param i: The component index.
                :param o: The interpolation index.
                :param interp_data: Interpolation data tuple.
                """
                # Build the complete list of (j, k) pairs, including the star case.
                j_k_pairs_with_star = [(j, k) for j in range(1, algo.m_bar_is[i - 1] + 1) 
                                            for k in range(k_maxs[cond] + 1)]
                j_k_pairs_with_star.append(('star', 'star'))
                
                # Determine the type and extract the appropriate interpolation indices.
                if i in algo.I_op:
                    comp_type = 'op'
                    interpolation_indices = interp_data[1]
                else: 
                    comp_type = 'func'
                    interpolation_indices = interp_data[3]

                if interpolation_indices == 'i<j':
                    for pair1, pair2 in combinations(j_k_pairs_with_star, 2):
                        process_pairs(cond, i, o, interp_data, [pair1, pair2], comp_type)
                elif interpolation_indices == 'i!=j':
                    for pair1, pair2 in product(j_k_pairs_with_star, repeat=2):
                        if pair1 == pair2:
                            continue
                        process_pairs(cond, i, o, interp_data, [pair1, pair2], comp_type)
                elif interpolation_indices == 'i':
                    for pair in j_k_pairs_with_star:
                        process_pairs(cond, i, o, interp_data, [pair], comp_type)
                elif interpolation_indices == 'i!=star':
                    j_k_pairs = [(j, k) for j in range(1, algo.m_bar_is[i - 1] + 1) 
                                        for k in range(k_maxs[cond] + 1)]
                    for pair in j_k_pairs:
                        process_pairs(cond, i, o, interp_data, [pair, ('star', 'star')], comp_type)
                else:
                    raise ValueError(f"Error: Invalid interpolation indices: {interpolation_indices}.")

            # ---------------------------------------------------------------------
            # Loop over all active conditions and components to process interpolation data.
            # ---------------------------------------------------------------------
            for cond in conds:
                for i in range(1, m + 1):
                    # Retrieve the interpolation data for component i.
                    for o, interp_data in enumerate(prob.get_component_data(i)):
                        process_interpolation(cond, i, o, interp_data)

            # ---------------------------------------------------------------------
            # Add final constraints to the model.
            # ---------------------------------------------------------------------
            for cond in conds:
                # Enforce that the PSD constraint sums belong to the PSD cone.
                Mod.constraint(PSD_constraint_sums[cond], Domain.inPSDCone(n + (k_maxs[cond] + 1) * m_bar + m))
                # For functional components, enforce the equality constraint.
                if m_func > 0:
                    Mod.constraint(eq_constraint_sums[cond] == 0)
                
            # ---------------------------------------------------------------------
            # Attempt to solve the model.
            # ---------------------------------------------------------------------
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
            return True
    
    @staticmethod
    def _compute_Thetas(algo: Type[Algorithm], h: int, alpha: int, condition: str = 'C1') -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute the Theta matrices (capital :math:`\Theta`) using the X matrices.

        For **condition "C1"**:
        
        - Set :math:`k_{\min} = 0` and :math:`k_{\max} = h+\alpha+1`.
        - Retrieve :math:`X = X_{\alpha+1}` from ``algo.get_Xs(k_min, k_max)``.
        - :math:`\Theta_0` is of size :math:`[n + (h+1)\bar{m} + m] \times [n + (h+\alpha+2)\bar{m} + m]`.
        - :math:`\Theta_1` is formed by vertically stacking :math:`X` with a block row 
          consisting of a zero block and an identity matrix.

        For **condition "C4"**:
        
        - Set :math:`k_{\min} = 0` and :math:`k_{\max} = h+\alpha+2`.
        - Retrieve :math:`X = X_1` from ``algo.get_Xs(k_min, k_max)``.
        - :math:`\Theta_0` is of size :math:`[n + (h+\alpha+2)\bar{m} + m] \times [n + (h+\alpha+3)\bar{m} + m]`.
        - :math:`\Theta_1` is formed similarly by stacking :math:`X` with an appropriate block row.

        **Parameters**

        :param algo: An instance of Algorithm (providing n, m_bar, m).
        :param h: A nonnegative integer.
        :param alpha: A nonnegative integer.
        :param condition: Either "C1" or "C4".
        
        **Returns**

        :return: A tuple :math:`(\Theta_0, \Theta_1)`.

        **Raises**

        :raises ValueError: If :math:`h` or :math:`\alpha` is negative or if condition is invalid.
        """
        if h < 0:
            raise ValueError("Parameter h must be >= 0.")
        if alpha < 0:
            raise ValueError("Parameter alpha must be >= 0.")
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
        
        - :math:`\theta_0 \in \mathbb{R}^{((h+1)\bar{m}_{\text{func}}+m\_func) \times ((h+\alpha+2)\bar{m}_{\text{func}}+m\_func)}` 
          is given by a block matrix with an identity in the upper left and lower right.
        - :math:`\theta_1` is formed as a horizontal block consisting of a zero block and an identity matrix.

        For **condition "C4"**:
        
        - :math:`\theta_0 \in \mathbb{R}^{((h+\alpha+2)\bar{m}_{\text{func}}+m\_func) \times ((h+\alpha+3)\bar{m}_{\text{func}}+m\_func)}` 
          is defined similarly.
        - :math:`\theta_1` is a horizontal block with a zero block and an identity matrix.

        **Parameters**

        :param algo: An instance of Algorithm (providing :math:`\bar{m}_{\text{func}}` and :math:`m\_func`).
        :param h: A nonnegative integer.
        :param alpha: A nonnegative integer.
        :param condition: Either "C1" or "C4".

        **Returns**

        :return: A tuple :math:`(\theta_0, \theta_1)`.

        **Raises**

        :raises ValueError: If :math:`h` or :math:`\alpha` is negative, if condition is invalid,
                            or if there are no functional components (i.e. :math:`m\_func \le 0`).
        """
        if h < 0:
            raise ValueError("Parameter h must be >= 0.")
        if alpha < 0:
            raise ValueError("Parameter alpha must be >= 0.")
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