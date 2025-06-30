import numpy as np
from typing import Type, Optional, Tuple, Union, List
from itertools import combinations, product
from mosek.fusion import Model, Domain, ObjectiveSense, OptimizeError
import mosek.fusion.pythonic
from autolyap.utils.helper_functions import create_symmetric_matrix_expression
from autolyap.problemclass import InclusionProblem
from autolyap.algorithms import Algorithm

class IterationDependent:
    @staticmethod
    def verify_iteration_dependent_Lyapunov(
            prob: Type[InclusionProblem],
            algo: Type[Algorithm],
            K: int,
            Q_0: np.ndarray,
            Q_K: np.ndarray,
            q_0: Optional[np.ndarray] = None,
            q_K: Optional[np.ndarray] = None
        ) -> Tuple[bool, Optional[float]]:
        r"""
        Verifies a chain of iteration-dependent Lyapunov inequalities for a given inclusion problem 
        and algorithm by setting up and solving an appropriate semidefinite program (SDP) using Mosek Fusion.

        **Parameters**

        :param prob: An :class:`InclusionProblem` instance containing interpolation conditions.
        :param algo: An :class:`Algorithm` instance providing dimensions and methods to compute matrices.
        :param K: A positive integer defining the iteration budget.
        :param Q_0: A symmetric numpy array of dimension :math:`[n + \bar{m} + m] \times [n + \bar{m} + m]`.
        :param Q_K: A symmetric numpy array of dimension :math:`[n + \bar{m} + m] \times [n + \bar{m} + m]`.
        :param q_0: A vector for functional components (if any), with appropriate dimensions.
        :param q_K: A vector for functional components (if any), with appropriate dimensions.

        **Returns**

        :returns: A tuple :math:`(True, c)` if the SDP is solved successfully, or :math:`(False, None)` otherwise.

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
        
        # Ensure K is positive.
        if K <= 0:
            raise ValueError("Parameter 'K' must be > 0.")
        
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
        
        # Expected dimension for matrices Q: [n + m_bar + m] x [n + m_bar + m].
        dim_Q = n + m_bar + m
        if not (isinstance(Q_0, np.ndarray) and Q_0.ndim == 2 and Q_0.shape[0] == Q_0.shape[1] == dim_Q):
            raise ValueError(f"Q_0 must be a symmetric matrix of dimension {dim_Q}x{dim_Q}. Got shape {Q_0.shape}.")
        if not np.allclose(Q_0, Q_0.T, atol=1e-8):
            raise ValueError("Q_0 must be symmetric.")
        if not (isinstance(Q_K, np.ndarray) and Q_K.ndim == 2 and Q_K.shape[0] == Q_K.shape[1] == dim_Q):
            raise ValueError(f"Q_K must be a symmetric matrix of dimension {dim_Q}x{dim_Q}. Got shape {Q_K.shape}.")
        if not np.allclose(Q_K, Q_K.T, atol=1e-8):
            raise ValueError("Q_K must be symmetric.")
        
        # For functional components, q_0 and q_K must have proper dimensions.
        if m_func > 0:
            dim_q = m_bar_func + m_func

            # Check q_0
            if q_0 is None:
                raise ValueError(f"q_0 must be a 1D numpy array of length {dim_q}, but got None.")
            if not (isinstance(q_0, np.ndarray) and q_0.ndim == 1 and q_0.shape[0] == dim_q):
                raise ValueError(
                    f"q_0 must be a 1D numpy array of length {dim_q}. Got shape {getattr(q_0, 'shape', None)}."
                )

            # Check q_K
            if q_K is None:
                raise ValueError(f"q_K must be a 1D numpy array of length {dim_q}, but got None.")
            if not (isinstance(q_K, np.ndarray) and q_K.ndim == 1 and q_K.shape[0] == dim_q):
                raise ValueError(
                    f"q_K must be a 1D numpy array of length {dim_q}. Got shape {getattr(q_K, 'shape', None)}."
        )


        # -------------------------------------------------------------------------
        # Set up the Mosek Fusion model and decision variables.
        # -------------------------------------------------------------------------
        with Model() as Mod:

            c = Mod.variable(1, Domain.greaterThan(0.0))
            
            Qs = {}
            Qs[0] = Q_0
            Qs[K] = Q_K
            for k in range(1, K):
                Qij = Mod.variable(f"Q_{k}", dim_Q * (dim_Q + 1) // 2, Domain.unbounded())
                Q_k = create_symmetric_matrix_expression(Qij, dim_Q)
                Qs[k] = Q_k
            
            if m_func > 0:
                qs = {}
                qs[0] = q_0
                qs[K] = q_K
                for k in range(1, K):
                    q_k = Mod.variable(f"q_{k}", dim_q, Domain.unbounded())
                    qs[k] = q_k
            
            # ---------------------------------------------------------------------
            # Build the main PSD (positive semidefinite) and equality constraint sums.
            # These will later be constrained to be in the PSD cone or equal zero, respectively.
            # ---------------------------------------------------------------------
            Ws = {}
            (Theta0, Theta1) = IterationDependent._compute_Thetas(algo, 0)
            W_0 = Theta1.T @ Qs[1] @ Theta1 - c[0] * Theta0.T @ Qs[0] @ Theta0
            Ws[0] = W_0
            for k in range(1, K):
                (Theta0, Theta1) = IterationDependent._compute_Thetas(algo, k)
                W_k = Theta1.T @ Qs[k+1] @ Theta1 - Theta0.T @ Qs[k] @ Theta0
                Ws[k] = W_k

            if m_func > 0:
                ws = {}
                (theta0, theta1) = IterationDependent._compute_thetas(algo)
                w_0 = theta1.T @ qs[1] - c[0] * theta0.T @ qs[0]
                ws[0] = w_0
                for k in range(1, K):
                    w_k = theta1.T @ qs[k+1] - theta0.T @ qs[k]
                    ws[k] = w_k

            # Initialize dictionaries to sum up the PSD and equality constraints.
            PSD_constraint_sums = {}
            eq_constraint_sums = {}
            for k in range(0, K):
                PSD_constraint_sums[k] = -Ws[k]
                if m_func > 0:
                    eq_constraint_sums[k] = -ws[k]

            # Dictionaries to hold multipliers for interpolation conditions.
            if m_op > 0:
                lambdas_op = {}
            if m_func > 0:
                lambdas_func = {}
                nus_func = {}

            # ---------------------------------------------------------------------
            # Define inner helper functions for processing interpolation data.
            # These functions handle both operator and function interpolation.
            # ---------------------------------------------------------------------
            def process_pairs(k: int, 
                            i: int, 
                            o: int, 
                            interpolation_data: Union[Tuple[np.ndarray, str],
                                                        Tuple[np.ndarray, np.ndarray, bool, str]], 
                            pairs: List[Union[Tuple[int, int], Tuple[str, str]]], 
                            comp_type: str) -> None:
                """
                Process a given set of interpolation pairs.
                """
                key_parts = [f"k:{k}", f"i:{i}"]
                for (j, k_) in pairs:
                    key_parts.append(f"(j:{j},k:{k_})")
                key_parts.append(f"o:{o}")
                key = tuple(key_parts)

                if comp_type == 'op':
                    M, _ = interpolation_data
                else:
                    M, a, eq, _ = interpolation_data

                # Compute the lifted matrix W for the given pairs.
                W_matrix = algo.compute_W(i, pairs, k, k+1, M)

                if comp_type == 'op':
                    lambda_var = Mod.variable(1, Domain.greaterThan(0.0))
                    lambdas_op[key] = lambda_var
                    PSD_constraint_sums[k] = PSD_constraint_sums[k] + lambda_var[0] * W_matrix
                else:
                    # For functional components, compute the aggregated F vector.
                    F_vector = algo.compute_F_aggregated(i, pairs, k, k+1, a)
                    if eq:
                        nu_var = Mod.variable(1, Domain.unbounded())
                        nus_func[key] = nu_var
                        PSD_constraint_sums[k] = PSD_constraint_sums[k] + nu_var[0] * W_matrix
                        eq_constraint_sums[k] = eq_constraint_sums[k] + nu_var[0] * F_vector
                    else:
                        lambda_var = Mod.variable(1, Domain.greaterThan(0.0))
                        lambdas_func[key] = lambda_var
                        PSD_constraint_sums[k] = PSD_constraint_sums[k] + lambda_var[0] * W_matrix
                        eq_constraint_sums[k] = eq_constraint_sums[k] + lambda_var[0] * F_vector

            def process_interpolation(k: int,
                                      i: int,
                                      o: int,
                                      interp_data: Union[Tuple[np.ndarray, str],
                                                         Tuple[np.ndarray, np.ndarray, bool, str]]) -> None:
                """
                Processes the interpolation condition for component i and interpolation index o.
                """
                # Build the complete list of (j, k) pairs, including the star case.
                j_k_pairs_with_star = [(j, k_) for j in range(1, algo.m_bar_is[i - 1] + 1) 
                                            for k_ in range(k,k+2)]
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
                        process_pairs(k, i, o, interp_data, [pair1, pair2], comp_type)
                elif interpolation_indices == 'i!=j':
                    for pair1, pair2 in product(j_k_pairs_with_star, repeat=2):
                        if pair1 == pair2:
                            continue
                        process_pairs(k, i, o, interp_data, [pair1, pair2], comp_type)
                elif interpolation_indices == 'i':
                    for pair in j_k_pairs_with_star:
                        process_pairs(k, i, o, interp_data, [pair], comp_type)
                elif interpolation_indices == 'i!=star':
                    j_k_pairs = [(j, k_) for j in range(1, algo.m_bar_is[i - 1] + 1) 
                                        for k_ in range(k, k+2)]
                    for pair in j_k_pairs:
                        process_pairs(k, i, o, interp_data, [pair, ('star', 'star')], comp_type)
                else:
                    raise ValueError(f"Error: Invalid interpolation indices: {interpolation_indices}.")

            # ---------------------------------------------------------------------
            # Loop over all iterations and components to process interpolation data.
            # ---------------------------------------------------------------------
            for k in range(0, K):
                for i in range(1, m + 1):
                    # Retrieve the interpolation data for component i.
                    for o, interp_data in enumerate(prob.get_component_data(i)):
                        process_interpolation(k, i, o, interp_data)

            # ---------------------------------------------------------------------
            # Add final constraints to the model.
            # ---------------------------------------------------------------------
            for k in range(0, K):
                # Enforce that the PSD constraint sums belong to the PSD cone.
                Mod.constraint(PSD_constraint_sums[k], Domain.inPSDCone(n + 2 * m_bar + m))
                # For functional components, enforce the equality constraint.
                if m_func > 0:
                    Mod.constraint(eq_constraint_sums[k] == 0)
            
            # ---------------------------------------------------------------------
            # Add the objective function to the model.
            # ---------------------------------------------------------------------
            Mod.objective("obj", ObjectiveSense.Minimize, c)

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
                return (False, None)
            return (True, c.level()[0])

    @staticmethod
    def _compute_Thetas(algo: Type[Algorithm], k: int) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute the capital Theta matrices for the iteration-dependent Lyapunov context.

        The matrices are defined as follows:

        .. math::
            
            \Theta_{0} =
            \begin{bmatrix}
            I_{n+\bar{m}} & 0_{(n+\bar{m})\times\bar{m}} & 0_{(n+\bar{m})\times m} \\
            0_{m\times(n+\bar{m})} & 0_{m\times\bar{m}} & I_{m}
            \end{bmatrix}

        and

        .. math::
            
            \Theta_{1}^{(k)} =
            \begin{bmatrix}
            X_{k+1}^{k,k+1} \\
            0_{(\bar{m}+m)\times(n+\bar{m})} \quad I_{(\bar{m}+m)}
            \end{bmatrix}

        where:

        - :math:`n = algo.n`
        - :math:`\bar{m} = algo.m_bar`
        - :math:`m = algo.m`
        - :math:`X_{k+1}^{k,k+1}` is retrieved via ``algo.get_Xs(k, k+1)`` using key :math:`(k+1)`.

        **Parameters**

        :param algo: An instance of :class:`Algorithm`.
        :param k: A non-negative integer iteration index used to select the appropriate X matrix.

        **Returns**

        :returns: A tuple :math:`(\Theta_0, \Theta_1)` of numpy arrays.
        """

        if not isinstance(k, int) or k < 0:
            raise ValueError("Parameter k must be a non-negative integer.")
        
        n = algo.n
        m_bar = algo.m_bar
        m = algo.m

        # Construct Theta_{0}
        Theta0 = np.block([
            [np.eye(n + m_bar), np.zeros((n + m_bar, m_bar)), np.zeros((n + m_bar, m))],
            [np.zeros((m, n + m_bar)), np.zeros((m, m_bar)), np.eye(m)]
        ])

        # Retrieve X_{k+1}^{k,k+1} using algo.get_Xs(k, k+1)
        Xs = algo.get_Xs(k, k+1)
        if (k + 1) not in Xs:
            raise ValueError(f"Expected key {k+1} in X matrices, but it was not found.")
        X_block = Xs[k + 1]

        # Construct the lower block for Theta_{1}^{(k)}
        lower_block = np.hstack([
            np.zeros((m_bar + m, n + m_bar)),
            np.eye(m_bar + m)
        ])

        # Form Theta_{1}^{(k)} by vertically stacking X_block and the lower block
        Theta1 = np.vstack([X_block, lower_block])
        return Theta0, Theta1

    @staticmethod
    def _compute_thetas(algo: Type[Algorithm]) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute the lowercase theta matrices for the iteration-dependent Lyapunov context.

        The matrices are defined as follows:

        .. math::

            \theta_{0} =
            \begin{bmatrix}
            I_{\bar{m}_{\text{func}}} & 0_{\bar{m}_{\text{func}} \times \bar{m}_{\text{func}}} & 0_{\bar{m}_{\text{func}} \times m_{\text{func}}} \\
            0_{m_{\text{func}} \times \bar{m}_{\text{func}}} & 0_{m_{\text{func}} \times \bar{m}_{\text{func}}} & I_{m_{\text{func}}}
            \end{bmatrix}

        and

        .. math::

            \theta_{1} =
            \begin{bmatrix}
            0_{(\bar{m}_{\text{func}}+m_{\text{func}}) \times \bar{m}_{\text{func}}} & I_{(\bar{m}_{\text{func}}+m_{\text{func}})}
            \end{bmatrix}

        where:

        - :math:`\bar{m}_{\text{func}}` is given by ``algo.m_bar_func``,
        - :math:`m_{\text{func}}` is given by ``algo.m_func``.

        **Note:** The theta matrices are only defined when there is at least one functional component.

        **Parameters**

        :param algo: An instance of :class:`Algorithm`.

        **Returns**

        :returns: A tuple :math:`(\theta_{0}, \theta_{1})` of numpy arrays.
        """
        m_bar_func = algo.m_bar_func  
        m_func = algo.m_func          
        if m_func <= 0:
            raise ValueError("Theta matrices require at least one functional component (m_func > 0).")

        theta0 = np.block([
            [np.eye(m_bar_func), np.zeros((m_bar_func, m_bar_func)), np.zeros((m_bar_func, m_func))],
            [np.zeros((m_func, m_bar_func)), np.zeros((m_func, m_bar_func)), np.eye(m_func)]
        ])

        theta1 = np.hstack([
            np.zeros((m_bar_func + m_func, m_bar_func)),
            np.eye(m_bar_func + m_func)
        ])

        return theta0, theta1

    @staticmethod
    def get_parameters_distance_to_solution(
            algo: Type[Algorithm],
            k: int,
            i: int = 1,
            j: int = 1
        ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""
        Compute the following matrices for the given :class:`Algorithm` instance:

        .. math::
            
            Q_{k} = \left( P_{(i,j)}\, Y_{k}^{k,k} - P_{(i,\star)}\, Y_{\star}^{k,k} \right)^T
                    \left( P_{(i,j)}\, Y_{k}^{k,k} - P_{(i,\star)}\, Y_{\star}^{k,k} \right),

        and

        .. math::
            
            q_{k} = 0 \quad \text{(if functional components exist)}.

        Here:
            
        - :math:`Y_{k}^{k,k}` is the Y matrix at iteration :math:`k` over the horizon :math:`[k, k]`.
        - :math:`Y_{\star}^{k,k}` is the “star” Y matrix over :math:`[k, k]`.
        - :math:`P_{(i,j)}` and :math:`P_{(i,\star)}` are the projection matrices for component :math:`i`.

        **Dimensions**

        - :math:`\dim(Q_{k}) = n + \bar{m} + m`.
        - If :math:`algo.m\_func > 0`, then :math:`\mathrm{len}(q_{k}) = \bar{m}_{\text{func}} + m\_func`.

        **Parameters**

        :param algo: An instance of :class:`Algorithm`. It must provide:
                    - :math:`algo.m` (total number of components),
                    - :math:`algo.m\_bar\_is` (a list where the *i*-th entry gives the number of evaluations for component :math:`i`),
                    - Methods ``get_Ys(k_min, k_max)`` and ``get_Ps()``.
        :param k: Iteration index. Must satisfy :math:`0 \le k`.
        :param i: Component index (1-indexed). Default is 1; must satisfy :math:`1 \le i \le algo.m`.
        :param j: Evaluation index for component :math:`i`. Default is 1; must satisfy :math:`1 \le j \le algo.m\_bar\_is[i-1]`.

        **Returns**

        - If :math:`algo.m\_func == 0`:

        :return: :math:`Q_{k}`.
        
        - Otherwise:

        :return: A tuple :math:`(Q_{k}, q_{k})`.

        **Raises**

        :raises ValueError: If any input is out of its valid range or if required matrices are missing.
        """
        # ----- Input Checking -----
        if not isinstance(k, int) or k < 0 :
            raise ValueError(f"Iteration index k must be nonnegative. Got {k}.")
        if not isinstance(i, int) or i < 1 or i > algo.m:
            raise ValueError(f"Component index i must be in [1, {algo.m}]. Got {i}.")
        if not isinstance(j, int) or j < 1 or j > algo.m_bar_is[i - 1]:
            raise ValueError(f"For component {i}, evaluation index j must be in [1, {algo.m_bar_is[i - 1]}]. Got {j}.")

        # ----- Compute Q_k -----
        Ys = algo.get_Ys(k, k)
        if k not in Ys:
            raise ValueError(f"Y matrix for iteration k = {k} not found.")
        if 'star' not in Ys:
            raise ValueError("Y star matrix ('star') not found.")
        
        Ps = algo.get_Ps()
        if (i, j) not in Ps:
            raise ValueError(f"Projection matrix for component {i}, evaluation {j} not found.")
        if (i, 'star') not in Ps:
            raise ValueError(f"Projection matrix for component {i} star not found.")
        
        diff = Ps[(i, j)] @ Ys[k] - Ps[(i, 'star')] @ Ys['star']
        Q_k = diff.T @ diff
        
        # ----- Construct T, p, and t as zeros with appropriate dimensions -----
        if algo.m_func > 0:
            m_bar_func = algo.m_bar_func    # Total evaluations for functional components.
            m_func = algo.m_func            # Number of functional components.
            dim_q_k = m_bar_func + m_func
            q_k_vec = np.zeros(dim_q_k)
            return Q_k, q_k_vec
        else:
            return Q_k
    
    @staticmethod
    def get_parameters_function_value_suboptimality(
            algo: Type[Algorithm], 
            k: int, 
            j: int = 1
        ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute the following matrices for the given :class:`Algorithm` instance:

        .. math::
            
            Q_{k} = 0, \quad
            q_{k} = \left( F_{(1,j,k)}^{k,k} - F_{(1,\star,\star)}^{k,k} \right)^T.

        Here:

        - :math:`F_{(1,j,k)}^{k,k}` is the F matrix for functional component 1 corresponding to evaluation :math:`j` at iteration :math:`k` over the horizon :math:`[k, k]`.
        - :math:`F_{(1,\star,\star)}^{k,k}` is the star F matrix for functional component 1 over the horizon :math:`[k, k]`.

        **Dimensions**

        - :math:`\dim(Q_{k}) = n + \bar{m} + m,`
        - :math:`\mathrm{len}(q_{k}) = \bar{m}_{\text{func}} + m_{\text{func}}.`

        **Note:** This function is only defined for problems with a single functional component, i.e.,
        :math:`m = m_{\text{func}} = 1`.

        **Parameters**

        :param algo: An instance of :class:`Algorithm`. It must provide:

                    - :math:`algo.m` (total number of components),
                    - :math:`algo.m_{\text{func}}` (number of functional components),
                    - :math:`algo.m_{\text{bar\_is}}` (a list where the first entry gives the number of evaluations for component 1),
                    - the method ``get_Fs(k_min, k_max)`` to obtain the F matrices.

        :param k: Iteration index. Must satisfy :math:`0 \le k \le K`.
        :param j: Evaluation index for component 1. Must satisfy :math:`1 \le j \le algo.m_{\text{bar\_is}}[0]`.

        **Returns**

        :returns: A tuple :math:`(Q_{k}, q_{k})` where:

                - :math:`Q_{k}` is a zero matrix of dimensions 
                    :math:`(n + \bar{m} + m) \times (n + \bar{m} + m)`,
                - :math:`q_{k}` is a column vector given by 
                    :math:`\left( F_{(1,j,k)}^{k,k} - F_{(1,\star,\star)}^{k,k} \right)^T`.

        **Raises**

        :raises ValueError: If any input is out of its valid range, if required matrices are missing,
                            or if :math:`m \ne 1` or :math:`m_{\text{func}} \ne 1`.
        """
        # ----- Input Checking -----
        if not isinstance(k, int) or k < 0:
            raise ValueError(f"Iteration index k must be nonnegative. Got {k}.")
        if not isinstance(j, int) or j < 1 or j > algo.m_bar_is[0]:
            raise ValueError(f"For component 1, evaluation index j must be in [1, {algo.m_bar_is[0]}]. Got {j}.")
        if algo.m != 1 or algo.m_func != 1:
            raise ValueError("Function value suboptimality is defined only for problems with a single functional component (m = m_func = 1).")

        # ----- Dimensions for Q_k -----
        dim_Q_k = algo.n + algo.m_bar + algo.m
        Q_k = np.zeros((dim_Q_k, dim_Q_k))

        # ----- Compute q_k -----
        Fs = algo.get_Fs(k, k)
        F_nonstar = Fs[(1, j, k)]
        F_star = Fs[(1, 'star', 'star')]
        q_k = (F_nonstar - F_star).T
        q_k = np.ravel(q_k)

        return Q_k, q_k

    @staticmethod
    def get_parameters_fixed_point_residual(
            algo: Type[Algorithm], 
            k: int
        ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""
        Compute the following matrices for the given :class:`Algorithm` instance:

        .. math::
            
            Q_{k} = \left( X_{k+1}^{k,k} - X_{k}^{k,k} \right)^T 
                    \left( X_{k+1}^{k,k} - X_{k}^{k,k} \right)

        and

        .. math::
            
            q_{k} = 0.

        Here:

        - :math:`X_{k}^{k,k}` is the X matrix at iteration :math:`k` over the horizon :math:`[k, k]`.
        - :math:`X_{k+1}^{k,k}` is the X matrix at iteration :math:`k+1` over the horizon :math:`[k, k]`.

        **Dimensions**

        - :math:`\dim(Q_{k}) = n + \bar{m} + m`.
        - :math:`\dim(q_{k}) = \bar{m}_{\text{func}} + m_{\text{func}}`.

        **Parameters**

        :param algo: An instance of :class:`Algorithm`. It must provide:
                    - :math:`algo.n` (state dimension),
                    - :math:`algo.m` (total number of components),
                    - :math:`algo.m_bar` (total number of evaluations per iteration),
                    - the method ``get_Xs(k_min, k_max)`` to obtain the X matrices.
        :param k: Iteration index. Must satisfy :math:`0 \le k`.

        **Returns**

        - If :math:`algo.m\_func > 0`:

        :returns: A tuple :math:`(Q_{k}, q_{k})` where:
                    - :math:`Q_{k}` is a matrix of dimensions 
                    :math:`(n + \bar{m} + m) \times (n + \bar{m} + m)`,
                    - :math:`q_{k}` is a zero vector of length 
                    :math:`(\bar{m}_{\text{func}} + m_{\text{func}})`.
                    
        - Otherwise:

        :returns: :math:`Q_{k}`.

        **Raises**

        :raises ValueError: If any input is out of its valid range or if required matrices are missing.
        """
        # ----- Input Checking -----
        if not isinstance(k, int) or k < 0:
            raise ValueError(f"Iteration index k must be nonnegative. Got {k}.")

        # ----- Retrieve X matrices -----
        Xs = algo.get_Xs(k, k)
        if k not in Xs or (k + 1) not in Xs:
            raise ValueError(f"X matrices for iterations {k} and {k+1} not found.")

        # ----- Compute Q_k -----
        diff = Xs[k + 1] - Xs[k]
        Q_k = diff.T @ diff

        # ----- Set q_k to zero -----
        if algo.m_func > 0:
            q_dim = algo.m_bar_func + algo.m_func
            q_k = np.zeros(q_dim)
            return Q_k, q_k
        else:
            return Q_k

    @staticmethod
    def get_parameters_optimality_measure(
            algo: Type[Algorithm], 
            k: int
        ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""
        Compute the following matrices for the given :class:`Algorithm` instance:

        .. math::
            
            Q_{k} = \begin{cases}
            \left( P_{(1,1)}\, U_{k}^{k,k} \right)^T \, P_{(1,1)}\, U_{k}^{k,k}
            & \text{if } m = 1, \\[1em]
            \left( \sum_{i=1}^{m} P_{(i,1)}\, U_{k}^{k,k} \right)^T 
            \left( \sum_{i=1}^{m} P_{(i,1)}\, U_{k}^{k,k} \right)
            + \sum_{i=2}^{m} \left( \left( P_{(1,1)} - P_{(i,1)} \right) Y_{k}^{k,k} \right)^T 
            \left( \left( P_{(1,1)} - P_{(i,1)} \right) Y_{k}^{k,k} \right)
            & \text{if } m > 1.
            \end{cases}

        .. math::
            
            q_{k} = 0.

        Here:

        - :math:`U_{k}^{k,k}` is the U matrix at iteration :math:`k` over the horizon :math:`[k,k]`.
        - :math:`Y_{k}^{k,k}` is the Y matrix at iteration :math:`k` over the horizon :math:`[k,k]`.
        - :math:`P_{(i,1)}` are the projection matrices for component :math:`i`.

        **Dimensions**

        - :math:`\dim(Q_{k}) = n + \bar{m} + m`.
        - :math:`\dim(q_{k}) = \bar{m}_{\text{func}} + m_{\text{func}}`.

        **Parameters**

        :param algo: An instance of :class:`Algorithm`. It must provide:

                    - :math:`algo.m` (total number of components),
                    - :math:`algo.m_bar` (total number of evaluations per iteration),
                    - :math:`algo.m_bar_is` (a list where the *i*-th entry gives the number of evaluations for component :math:`i`),
                    - the methods ``get_Us(k_min, k_max)`` and ``get_Ys(k_min, k_max)`` to obtain the U and Y matrices,
                    - the method ``get_Ps()`` to obtain the projection matrices.

        :param k: Iteration index. Must satisfy :math:`0 \le k`.

        **Returns**

        - If :math:`algo.m_{\text{func}} > 0`:

        :returns: A tuple :math:`(Q_{k}, q_{k})` where:

                    - :math:`Q_{k}` is a matrix of dimensions :math:`(n + \bar{m} + m) \times (n + \bar{m} + m)`,
                    - :math:`q_{k}` is a zero vector of length :math:`(\bar{m}_{\text{func}} + m_{\text{func}})`.

        - Otherwise:

        :returns: :math:`Q_{k}`.

        **Raises**

        :raises ValueError: If any input is out of its valid range or if required matrices are missing.
        """
        # ----- Input Checking -----
        if not isinstance(k, int) or k < 0:
            raise ValueError(f"Iteration index k must be nonnegative. Got {k}.")

        # ----- Retrieve U and Y matrices -----
        Us = algo.get_Us(k, k)
        Ys = algo.get_Ys(k, k)
        if k not in Us:
            raise ValueError(f"U matrix for iteration {k} not found.")
        if k not in Ys:
            raise ValueError(f"Y matrix for iteration {k} not found.")
        
        # ----- Retrieve Projection matrices -----
        Ps = algo.get_Ps()

        # ----- Compute Q_k -----
        if algo.m == 1:
            # Q_k = (P_{(1,1)}U_{k}^{k,k})^T P_{(1,1)}U_{k}^{k,k}
            P = Ps[(1, 1)]
            U = Us[k]
            term = P @ U
            Q_k = term.T @ term
        else:
            # m > 1:
            # term1 = (sum_{i=1}^{m} P_{(i,1)}U_{k}^{k,k})^T (sum_{i=1}^{m} P_{(i,1)}U_{k}^{k,k})
            S = None
            for i in range(1, algo.m + 1):
                P_i = Ps[(i, 1)]
                if S is None:
                    S = P_i @ Us[k]
                else:
                    S = S + P_i @ Us[k]
            term1 = S.T @ S
            
            # term2 = sum_{i=2}^{m} ((P_{(1,1)} - P_{(i,1)})Y_{k}^{k,k})^T ((P_{(1,1)} - P_{(i,1)})Y_{k}^{k,k})
            Y = Ys[k]
            term2 = 0
            for i in range(2, algo.m + 1):
                diff_P = Ps[(1, 1)] - Ps[(i, 1)]
                temp = diff_P @ Y
                term2 = term2 + (temp.T @ temp)
            Q_k = term1 + term2

        # ----- Set q_k to zero -----
        if algo.m_func > 0:
            q_dim = algo.m_bar_func + algo.m_func
            q_k = np.zeros(q_dim)
            return Q_k, q_k
        else:
            return Q_k
