Iteration-dependent analyses
============================

Here, we focus on finite-horizon analyses, which apply to
algorithms that run indefinitely, as well as algorithms with finite
iteration budgets matching or exceeding the horizon. For notational
simplicity, we assume that

.. math::

   K\in\mathbb{N}

in
:eq:`eq:linear_system_with_nonlinearity`,
representing the horizon.

Following :cite:`pep-taylor2019stochasticfirstorder`, we adopt an ansatz
consisting of a sequence of iteration-dependent and quadratic Lyapunov
functions, also often referred to as *potential functions*
:cite:`pep-bansal2019`.

.. container:: definition

   .. _def:iteration_dependent_Lyapunov:chain:

   **Definition 5.3.1 (Chained Lyapunov inequalities).**

   Suppose that
   :ref:`Assumption 3.1 (Well-posedness) <ass:well-posedness>` holds. Let

   .. math::

      \bx_{0}\in\calH^{n}, \qquad
      \p{ ( \bx^{k}, \bu^{k}, \by^{k}, \bFcn^{k} ) }_{k=0}^{K},

   be an initial point and a sequence of iterates satisfying
   :eq:`eq:linear_system_with_nonlinearity`, and let

   .. math::

      \p{y^{\star}, \hat{\bu}^{\star}, \bFcn^{\star}}

   be a point satisfying :eq:`eq:solution`. Also let

   .. math::

      c_K\in\reals_{+}.

   Define

   .. _eq:iteration_dependent_lyapunov:chain:v:

   .. math::
      :label: eq:iteration_dependent_lyapunov:chain:v

      \begin{aligned}
              \p{\forall k \in \llbracket0,K\rrbracket} \quad
              \left[
              \begin{aligned}
                  &\mathcal{V}(Q_k,q_k,k) {}={} \quadform{Q_{k}}{\p{\bx^{k},\bu^{k},\hat{\bu}^{\star},y^{\star}}} + q_{k}^{\top}\p{\bFcn^{k},\bFcn^{\star}}, \\
                  &Q_{k} \in \sym^{n+\NumEval+m}, \\
                  &q_{k} \in \reals^{\NumEvalFunc + \NumFunc}.
              \end{aligned}
              \right.
          \end{aligned}

   We say that :math:`(\p{Q_{k},q_{k}})_{k=0}^{K}` and :math:`c_K` satisfy
   a *length* :math:`K` *sequence of chained Lyapunov inequalities for
   algorithm*
   :eq:`eq:linear_system_with_nonlinearity`
   *over the problem class defined by*
   :math:`(\mathcal{F}_i)_{i\in\IndexFunc}` *and*
   :math:`(\mathcal{G}_i)_{i\in\IndexOp}` if

   .. _eq:iteration_dependent_lyapunov:chain:

   .. math::
      :label: eq:iteration_dependent_lyapunov:chain

      \begin{aligned}
              \mathcal{V}\p{Q_K,q_K,K} &\leq \mathcal{V}\p{Q_{K-1},q_{K-1},K-1} \\
              &\leq \ldots \\
              &\leq \mathcal{V}\p{Q_1,q_1,1} \\
              &\leq c_K \mathcal{V}\p{Q_0,q_0,0}
          \end{aligned}

   holds for each

   .. math::

      \bx_{0}, \qquad
      \p{ ( \bx^{k}, \bu^{k}, \by^{k}, \bFcn^{k} ) }_{k=0}^{K}, \qquad
      \p{y^{\star}, \hat{\bu}^{\star},\bFcn^{\star}},

   and for each

   .. math::

      \p{f_{i}}_{i\in\IndexFunc} \in \prod_{i\in\IndexFunc} \mathcal{F}_{i},
      \qquad
      \p{G_i}_{i\in\IndexOp} \in \prod_{i\in\IndexOp}\mathcal{G}_i.

In the proposed methodology, the user specifies the boundary parameters
:math:`\p{Q_{0},q_{0},Q_{K},q_{K}}`, which define the initial and final
Lyapunov expressions :math:`\mathcal{V}\p{Q_0,q_0,0}` and
:math:`\mathcal{V}\p{Q_K,q_K,K}`. The SDP then searches for
:math:`(\p{Q_{k},q_{k}})_{k=1}^{K-1}` and the smallest :math:`c_K` such that
:eq:`eq:iteration_dependent_lyapunov:chain` holds. When feasible, this yields

.. math::

   \begin{aligned}
       \mathcal{V}\p{Q_K,q_K,K} \leq c_K \mathcal{V}\p{Q_0,q_0,0}.\end{aligned}

For built-in choices of these boundary terms:

- Distance to solution:
  :meth:`Distance-to-solution helper <autolyap.IterationDependent.get_parameters_distance_to_solution>`.
- State-component distance to solution:
  :meth:`State-component-distance helper <autolyap.IterationDependent.get_parameters_state_component_distance_to_solution>`.
- State-component same-iteration difference:
  :meth:`State-component-difference helper <autolyap.IterationDependent.get_parameters_state_component_difference>`.
- State-component cross-iteration difference:
  :meth:`State-component-cross-iteration-difference helper <autolyap.IterationDependent.get_parameters_state_component_cross_iteration_difference>`.
- Function-value suboptimality:
  :meth:`Function-value-suboptimality helper <autolyap.IterationDependent.get_parameters_function_value_suboptimality>`.
- Fixed-point residual:
  :meth:`Fixed-point-residual helper <autolyap.IterationDependent.get_parameters_fixed_point_residual>`.
- Optimality measure:
  :meth:`Optimality-measure helper <autolyap.IterationDependent.get_parameters_optimality_measure>`.

The corresponding API entry point is
:meth:`~autolyap.IterationDependent.search_lyapunov`.

The SDP characterization is given in
:ref:`Theorem 5.3.2 (Chained Lyapunov inequalities via SDP) <thm:iteration_dependent_lyapunov>`. Computationally,
each inequality in :eq:`eq:iteration_dependent_lyapunov:chain` is enforced by
a one-step analysis with positive semidefinite constraints of constant
dimension. As a result, the number of variables and constraints grows
linearly with :math:`K`.

.. container:: theorem

   .. _thm:iteration_dependent_lyapunov:

   **Theorem 5.3.2 (Chained Lyapunov inequalities via SDP).**

   Suppose that
   :ref:`Assumption 3.1 (Well-posedness) <ass:well-posedness>` and
   :ref:`Assumption 4.1 (Interpolation conditions) <ass:interpolation>`
   hold, and

   .. math::

      K\in\mathbb{N}, \qquad
      Q_{0},Q_{K} \in \sym^{n+\NumEval+m}, \qquad
      q_{0},q_{K} \in \reals^{\NumEvalFunc + \NumFunc}.

   Then a sufficient condition for there to exist

   .. math::

      \p{ (Q_{k},q_{k}) }_{k=1}^{K-1}, \qquad c_K,

   such that :math:`\p{ (Q_{k},q_{k}) }_{k=0}^{K}` and :math:`c_K`
   satisfy a length :math:`K` sequence of chained Lyapunov
   inequalities for algorithm
   :eq:`eq:linear_system_with_nonlinearity`
   over the problem class defined by
   :math:`\p{\mathcal{F}_i}_{i\in\IndexFunc}` and
   :math:`\p{\mathcal{G}_i}_{i\in\IndexOp}` is that the following
   system of constraints, which reuses the lifted
   matrices/vectors from :doc:`5.1. Performance estimation via SDPs
   </theory/performance_estimation_via_sdps>`, including the lifted state
   matrices :eq:`eq:x_mats`, and the interpolation terms
   :eq:`eq:w_func_ineq`, :eq:`eq:w_func_eq`, :eq:`eq:w_op`,
   :eq:`eq:f_func_ineq`, and :eq:`eq:f_func_eq`,


   .. container:: subequations

      .. _eq:iteration_dependent_lyapunov:condition:

      .. _eq:iteration_dependent_lyapunov:condition:func-eq:

      .. _eq:iteration_dependent_lyapunov:condition:op:

      .. _eq:iteration_dependent_lyapunov:condition:psd:

      .. _eq:iteration_dependent_lyapunov:condition:zero:

      .. _eq:iteration_dependent_lyapunov:condition:q_mat:

      .. _eq:iteration_dependent_lyapunov:condition:q:

      .. _eq:iteration_dependent_lyapunov:condition:a_k:

      .. _eq:iteration_dependent_lyapunov:condition:func-ineq:

      .. math::
         :no-wrap:

         \begin{align}
                     &\textbf{for each}\ k \in \llbracket 0, K-1 \rrbracket \notag \\
                     &\quad
                     \left(
                     \begin{array}{@{}c@{}}
                         \forall i \in\IndexFunc \\
                         \forall o \in \mathcal{O}^{\textup{func-ineq}}_{i} \\
                         \forall \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{k,k+1}
                     \end{array}
                     \right) \quad \lambda_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{\textup{func-ineq},\,k} \geq 0, \tag{5.51a}\\
                     &\quad
                     \left(
                     \begin{array}{@{}c@{}}
                         \forall i \in\IndexFunc \\
                         \forall o \in \mathcal{O}^{\textup{func-eq}}_{i} \\
                         \forall \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{k,k+1}
                     \end{array}
                     \right) \quad
                     \nu_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{\textup{func-eq},\,k} \in \reals, \tag{5.51b}\\
                     &\quad
                     \left(
                     \begin{array}{@{}c@{}}
                         \forall i \in\IndexFunc \\
                         \forall o \in \mathcal{O}^{\textup{op}}_{i} \\
                         \forall \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{k,k+1}
                     \end{array}
                     \right) \quad
                     \lambda_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{\textup{op},\,k} \geq 0, \tag{5.51c}\\
                     &\quad - W_{k}
                     + \sum_{\substack{i\in\IndexFunc \\ o \in \mathcal{O}^{\textup{func-ineq}}_{i} \\ \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{k,k+1} }}
                     \lambda_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{\textup{func-ineq},\,k}
                     W_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{k,\,k+1,\,\textup{func-ineq}} \notag \\
                     &\quad \quad + \sum_{\substack{i\in\IndexFunc  \\ o \in \mathcal{O}^{\textup{func-eq}}_{i} \\ \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{k,k+1} }}
                     \nu_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{\textup{func-eq},\,k}
                     W_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{k,\,k+1,\,\textup{func-eq}} \notag \\
                     &\quad \quad + \sum_{\substack{i\in\IndexOp \\ o \in \mathcal{O}^{\textup{op}}_{i} \\ \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{k,k+1} }}
                     \lambda_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{\textup{op},\,k}
                     W_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{k,\,k+1,\,\textup{op}}
                     \succeq 0, \tag{5.51d}\\
                     &\quad
                     - w_{k}
                     + \sum_{\substack{i\in\IndexFunc \\ o \in \mathcal{O}^{\textup{func-ineq}}_{i} \\ \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{k,k+1} }}
                     \lambda_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{\textup{func-ineq},\,k}
                     F_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{k,\,k+1,\,\textup{func-ineq}} \notag \\
                     &\quad \quad + \sum_{\substack{i\in\IndexFunc  \\ o \in \mathcal{O}^{\textup{func-eq}}_{i} \\ \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{k,k+1} }}
                     \nu_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{\textup{func-eq},\,k}
                     F_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{k,\,k+1,\,\textup{func-eq}}
                     = 0, \tag{5.51e}\\
                     &\mathbf{end} \notag \\
                     & \p{\forall k\in \llbracket1,K-1\rrbracket} \quad Q_{k} \in \sym^{n+\NumEval+m}, \tag{5.51f}\\
                     & \p{\forall k\in \llbracket1,K-1\rrbracket} \quad q_{k} \in \mathbb{R}^{\NumEvalFunc + \NumFunc}, \tag{5.51g}\\
                     & c_K \in \reals_{+}. \tag{5.51h}
                 \end{align}

   is feasible for the scalars

   .. math::

      \begin{aligned}
              \lambda_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{\textup{func-ineq},\,k}, \\
              \nu_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{\textup{func-eq},\,k}, \\
              \lambda_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{\textup{op},\,k},    
          \end{aligned}

   the matrices and vectors :math:`\p{\p{Q_{k},q_{k}}}_{k=1}^{K-1}`, and
   the scalar :math:`c_K`, where we have introduced

   .. math::
      :label: eq:iteration_dependent_lyapunov:w_0_mat
      :class: eq-align-dep-w

      \begin{aligned}
              W_{0} &= \p{\Theta_{1}^{\p{0}}}^{\top}Q_{1}\Theta_{1}^{\p{0}}
              - c_K \Theta_{0}^{\top}Q_{0}\Theta_{0}.
          \end{aligned}

   .. math::
      :label: eq:iteration_dependent_lyapunov:w_0_vec
      :class: eq-align-dep-w

      \begin{aligned}
              w_{0} &= \theta_{1}^{\top}q_{1} - c_K\theta_{0}^{\top}q_{0}.
          \end{aligned}

   .. math::
      :label: eq:iteration_dependent_lyapunov:w_k_mat
      :class: eq-align-dep-w

      \begin{aligned}
              \p{\forall k\in\llbracket1,K-1\rrbracket}\quad
              W_{k} &= \p{\Theta_{1}^{\p{k}}}^{\top}Q_{k+1}\Theta_{1}^{\p{k}}
              - \Theta_{0}^{\top}Q_{k}\Theta_{0}.
          \end{aligned}

   .. math::
      :label: eq:iteration_dependent_lyapunov:w_k_vec
      :class: eq-align-dep-w

      \begin{aligned}
              \p{\forall k\in\llbracket1,K-1\rrbracket}\quad
              w_{k} &= \theta_{1}^{\top}q_{k+1} - \theta_{0}^{\top}q_{k}.
          \end{aligned}

   and

   .. math::
      :label: eq:iteration_dependent_lyapunov:theta0_mat
      :class: eq-align-dep-theta

      \begin{aligned}
              \Theta_{0} &=
              \underbracket{
              \begin{bmatrix}
                  I_{n+\NumEval } & 0_{\p{n+\NumEval}\times\NumEval } & 0_{\p{n+\NumEval}\times m} \\
                  0_{m \times \p{n+\NumEval}} & 0_{m \times\NumEval } & I_{m}
              \end{bmatrix}
              }_{
              \in \reals^{ \p{n + \NumEval + m} \times \p{n + 2\NumEval + m} }
              }.
          \end{aligned}

   .. math::
      :label: eq:iteration_dependent_lyapunov:theta0_vec
      :class: eq-align-dep-theta

      \begin{aligned}
              \theta_{0} &=
              \underbracket{
              \begin{bmatrix}
                  I_{\NumEvalFunc} & 0_{\NumEvalFunc \times \NumEvalFunc } & 0_{\NumEvalFunc \times \NumFunc } \\
                  0_{\NumFunc\times \NumEvalFunc} &0_{\NumFunc\times \NumEvalFunc} & I_{\NumFunc}
              \end{bmatrix}
              }_{
              \in \reals^{\p{\NumEvalFunc + \NumFunc}\times\p{2\NumEvalFunc + \NumFunc}}
              }.
          \end{aligned}

   .. math::
      :label: eq:iteration_dependent_lyapunov:theta1_mat
      :class: eq-align-dep-theta

      \begin{aligned}
              \p{\forall k\in\llbracket0,K-1\rrbracket}\quad
              \Theta_{1}^{\p{k}} &=
              \underbracket{
              \begin{bmatrix}
                  X_{k+1}^{k,k+1} \\
                  0_{\p{\NumEval + m} \times \p{n + \NumEval } } \quad I_{\NumEval + m}
              \end{bmatrix}
              }_{
              \in \reals^{ \p{n + \NumEval + m} \times \p{n + 2\NumEval + m} }
              }.
          \end{aligned}

   .. math::
      :label: eq:iteration_dependent_lyapunov:theta1_vec
      :class: eq-align-dep-theta

      \begin{aligned}
              \theta_{1} &=
              \underbracket{
              \begin{bmatrix}
                  0_{\p{\NumEvalFunc + \NumFunc}\times\NumEvalFunc} & I_{\NumEvalFunc + \NumFunc}
              \end{bmatrix}
              }_{
              \in \reals^{\p{\NumEvalFunc + \NumFunc}\times\p{2\NumEvalFunc + \NumFunc}}
              }.
          \end{aligned}

   Furthermore, if the interpolation conditions for
   :math:`\p{\mathcal{F}_{i}}_{i\in\IndexFunc}` and
   :math:`\p{\mathcal{G}_{i}}_{i\in\IndexOp}` are tight,
   :math:`\dim \calH \geq n + 2\NumEval + m`, and there exists

   .. math::

      \begin{aligned}
              \p{\forall k\in \llbracket0,K-1\rrbracket}
              \quad
              \left[
              \begin{aligned}
                  &G_{k}\in\sym_{++}^{n + 2\NumEval + m}, \\
                  &\bchi_{k}\in\reals^{2\NumEvalFunc + \NumFunc}, 
              \end{aligned}
              \right.
          \end{aligned}

   such that


   .. math::

      \begin{aligned}
              & \textbf{for each}\ k\in \llbracket0,K-1\rrbracket \\
              & \quad
              \left(
              \begin{array}{@{}c@{}}
                  \forall i \in\IndexFunc \\
                  \forall o \in \mathcal{O}^{\textup{func-ineq}}_{i} \\
                  \forall \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{k,k+1}
              \end{array}
              \right) \quad 
              \begin{aligned}
                  & \bm{\chi}^{\top}_{k} F_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{k,\,k+1,\,\textup{func-ineq}} \\    
                  & + \trace\p{W_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{k,\,k+1,\,\textup{func-ineq}}G_{k}} \leq 0,
              \end{aligned}
              \\ 
              & \quad
              \left(
              \begin{array}{@{}c@{}}
                  \forall i \in\IndexFunc \\
                  \forall o \in \mathcal{O}^{\textup{func-eq}}_{i} \\
                  \forall \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{k,k+1}
              \end{array}
              \right) \quad
              \begin{aligned}
                  & \bm{\chi}^{\top}_{k} F_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{k,\,k+1,\,\textup{func-eq}} \\
                  & + \trace\p{W_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{k,\,k+1,\,\textup{func-eq}}G_{k}} = 0,
              \end{aligned}
              \\
              & \quad
              \left(
              \begin{array}{@{}c@{}}
                  \forall i \in\IndexFunc \\
                  \forall o \in \mathcal{O}^{\textup{op}}_{i} \\
                  \forall \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{k,k+1}
              \end{array}
              \right) \quad
              \trace\p{W_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{k,\,k+1,\,\textup{op}}G_{k}} \leq 0,
              \\
              & \mathbf{end}
          \end{aligned}

   then :ref:`(5.51) <eq:iteration_dependent_lyapunov:condition>` is
   also a necessary condition.

.. container:: proof

   *Proof.* Note that


   .. math::

      \begin{aligned}
              \p{\bx^{k},\bu^{k},\hat{\bu}^{\star},y^{\star}} &= \p{\Theta_{0} \kron \Id }\p{\bx^{k},\bu^{k},\bu^{k+1},\hat{\bu}^{\star},y^{\star}}, \\
              \p{\bFcn^{k},\bFcn^{\star}} &= \theta_{0}\p{\bFcn^{k},\bFcn^{k+1},\bFcn^{\star}}, \\
              \p{\bx^{k+1},\bu^{k+1},\hat{\bu}^{\star},y^{\star}} &= \p{ \Theta_{1}^{\p{k}} \kron \Id }\p{\bx^{k},\bu^{k},\bu^{k+1},\hat{\bu}^{\star},y^{\star}}, \\
              \p{\bFcn^{k+1},\bFcn^{\star}} &= \theta_{1}\p{\bFcn^{k},\bFcn^{k+1},\bFcn^{\star}},
          \end{aligned}

   where we have used
   in :eq:`eq:iteration_dependent_lyapunov:theta0_mat`,
   :eq:`eq:iteration_dependent_lyapunov:theta0_vec`,
   :eq:`eq:iteration_dependent_lyapunov:theta1_mat`, and
   :eq:`eq:iteration_dependent_lyapunov:theta1_vec`, respectively.

   First, suppose that the parameters
   :math:`\p{\p{Q_{k},q_{k}}}_{k=1}^{K-1}` and :math:`c_K` are fixed and
   consider :math:`\p{\mathcal{V}\p{Q_k,q_k,k}}_{k=0}^{K}` in
   :eq:`eq:iteration_dependent_lyapunov:chain:v`.
   Note that

   .. _eq:iteration_independent_lyapunov:obj_0:

   .. math::
      :label: eq:iteration_independent_lyapunov:obj_0

      \begin{aligned}
              &\mathcal{V}\p{Q_1,q_1,1} - c_K\mathcal{V}\p{Q_0,q_0,0} \notag\\
              & = \quadform{\p{\Theta_{1}^{\p{0}}}^{\top}Q_{1}\Theta_{1}^{\p{0}}  - c_K\Theta_{0}^{\top}Q_{0}\Theta_{0}}{\p{\bx^{0},\bu^{0},\bu^{1},\hat{\bu}^{\star},y^{\star}}}  \notag \\
              & \quad + \p{q_{1}^{\top}\theta_{1} - c_K q_{0}^{\top}\theta_{0} } \p{\bFcn^{0},\bFcn^{1},\bFcn^{\star}} \notag \\
              &= \quadform{W_{0}}{\p{\bx^{0},\bu^{0},\bu^{1},\hat{\bu}^{\star},y^{\star}}} + w_{0}^{\top} \p{\bFcn^{0},\bFcn^{1},\bFcn^{\star}} 
          \end{aligned}

   where :eq:`eq:iteration_dependent_lyapunov:w_0_mat`
   and :eq:`eq:iteration_dependent_lyapunov:w_0_vec`
   is used in the last equality. Therefore, using the
   :eq:`eq:iteration_independent_lyapunov:obj_0`
   as the objective function in :ref:`(PEP) <eq:pep>`,
   :ref:`Theorem 5.1.1 (Performance estimation via SDP) <thm:pep_leq_zero>`
   gives that :ref:`(5.51) <eq:iteration_dependent_lyapunov:condition>`,
   with :math:`k=0`, is a sufficient condition for the inequality
   :math:`\mathcal{V}\p{Q_1,q_1,1} \leq c_K\mathcal{V}\p{Q_0,q_0,0}`.

   For :math:`k\in \llbracket1,K-1\rrbracket`, note that

   .. _eq:iteration_independent_lyapunov:obj_k:

   .. math::
      :label: eq:iteration_independent_lyapunov:obj_k

      \begin{aligned}
              &\mathcal{V}\p{Q_{k+1},q_{k+1},k+1} - \mathcal{V}\p{Q_k,q_k,k} \notag \\
              & =\quadform{Q_{k+1}}{\p{\bx^{k+1},\bu^{k+1},\hat{\bu}^{\star},y^{\star}}}  + q_{k+1}^{\top}\p{\bFcn^{k+1},\bFcn^{\star}} \notag\\
              &\quad -\quadform{Q_{k}}{\p{\bx^{k},\bu^{k},\hat{\bu}^{\star},y^{\star}}}  - q_{k}^{\top} \p{\bFcn^{k},\bFcn^{\star}} \notag \\
              & = \quadform{\p{\Theta_{1}^{\p{k}}}^{\top}Q_{k+1}\Theta_{1}^{\p{k}} - \Theta_{0}^{\top}Q_{k}\Theta_{0}}{\p{\bx^{k},\bu^{k},\bu^{k+1},\hat{\bu}^{\star},y^{\star}}} \notag\\
              &\quad + \p{q_{k+1}^{\top}\theta_{1} - q_{k}^{\top}\theta_{0} } \p{\bFcn^{k},\bFcn^{k+1},\bFcn^{\star}} \notag \\
              & = \quadform{W_k}{\p{\bx^{k},\bu^{k},\bu^{k+1},\hat{\bu}^{\star},y^{\star}}} + w_k^{\top}\p{\bFcn^{k},\bFcn^{k+1},\bFcn^{\star}} 
          \end{aligned}

   where :eq:`eq:iteration_dependent_lyapunov:w_k_mat`
   and :eq:`eq:iteration_dependent_lyapunov:w_k_vec`
   is used in the last equality. Therefore, using the
   :eq:`eq:iteration_independent_lyapunov:obj_k`
   as the objective function in :ref:`(PEP) <eq:pep>`,
   :ref:`Theorem 5.1.1 (Performance estimation via SDP) <thm:pep_leq_zero>`
   gives that :ref:`(5.51) <eq:iteration_dependent_lyapunov:condition>`,
   constrained to this particular :math:`k`, is a sufficient condition
   for the inequality :math:`\mathcal{V}\p{Q_{k+1},q_{k+1},k+1} \leq \mathcal{V}\p{Q_k,q_k,k}`.

   Second, note that the proof is complete if we let the parameters
   :math:`\p{\p{Q_{k},q_{k}}}_{k=1}^{K-1}` and :math:`c_K` free, as
   in :ref:`(5.51f) <eq:iteration_dependent_lyapunov:condition:q_mat>`,
   :ref:`(5.51g) <eq:iteration_dependent_lyapunov:condition:q>`, and
   :ref:`(5.51h) <eq:iteration_dependent_lyapunov:condition:a_k>`. :math:`\square`

.. rubric:: References

.. bibliography::
   :filter: docname in docnames
   :keyprefix: pep-
