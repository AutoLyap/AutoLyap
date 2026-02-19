Iteration-independent analyses
==============================

For iteration-independent analyses, we consider algorithms that continue to iterate
indefinitely and have iteration-independent parameters. I.e.,
in :eq:`eq:linear_system_with_nonlinearity`,
we assume that :math:`K=\infty` and that there exist fixed matrices

.. math::

   \begin{aligned}
       \p{A,B,C,D}\in\reals^{n\times n}\times\reals^{n\times \NumEval}\times\reals^{\NumEval\times n}\times\reals^{\NumEval \times \NumEval}\end{aligned}

such that

.. _eq:constant_abcd:

.. math::
   :label: eq:constant_abcd

   \begin{aligned}
       \p{\forall k \in \naturals }\quad \p{A_{k},B_{k},C_{k},D_{k}} = \p{A,B,C,D}.\end{aligned}

.. container:: definition

   .. _def:iteration_independent_Lyapunov:

   **Definition 5.2.1 (Iteration-independent quadratic Lyapunov inequality).**

   Suppose that
   :ref:`Assumption 3.1 (Well-posedness) <ass:well-posedness>`
   and :eq:`eq:constant_abcd` hold. Let

   .. math::

      \bx_{0}\in\calH^{n}, \qquad
      \p{ ( \bx^{k}, \bu^{k}, \by^{k}, \bFcn^{k} ) }_{k\in\naturals},

   be an initial point and a sequence of iterates satisfying
   :eq:`eq:linear_system_with_nonlinearity`, and let

   .. math::

      \p{y^{\star}, \hat{\bu}^{\star}, \bFcn^{\star}}

   be a point satisfying :eq:`eq:solution`. Also let

   .. math::

      \rho \in [0,1], \qquad h\in\naturals, \qquad \alpha\in\naturals.

   Define

   .. _eq:iteration_independent_lyapunov:v:

   .. math::
      :label: eq:iteration_independent_lyapunov:v

      \begin{aligned}
              \p{\forall k \in \naturals}
              \quad
                  \mathcal{V}(W,w,k) {}={}&{} \quadform{W}{\p{\bx^{k},\bu^{k},\ldots,\bu^{k+h},\hat{\bu}^{\star},y^{\star}}} \\ 
                  & + w^{\top}\p{\bFcn^{k},\ldots,\bFcn^{k+h},\bFcn^{\star}},
          \end{aligned}

   for each :math:`\p{W,w}\in\set{\p{Q,q},\p{P,p}}`, where

   .. math::

      Q,P \in \sym^{n+\p{h+1}\NumEval+m}, \qquad
      q,p\in\mathbb{R}^{\p{h+1}\NumEvalFunc + \NumFunc},

   and define

   .. _eq:iteration_independent_lyapunov:r:

   .. math::
      :label: eq:iteration_independent_lyapunov:r

      \begin{aligned}
              \p{\forall k \in \naturals}
              \quad
                  \mathcal{R}(W,w,k) {}={}&{} \quadform{W}{\p{\bx^{k},\bu^{k},\ldots,\bu^{k+h + \alpha + 1},\hat{\bu}^{\star},y^{\star}}} \\ 
                  &{} + w^{\top}\p{\bFcn^{k},\ldots,\bFcn^{k+h + \alpha + 1},\bFcn^{\star}},
          \end{aligned}

   for each :math:`\p{W,w}\in\set{\p{S,s},\p{T,t}}`, where

   .. math::

      S,T\in\sym^{n+\p{h+\alpha+2}\NumEval+m}, \qquad
      s, t \in\mathbb{R}^{\p{h+\alpha+2}\NumEvalFunc + \NumFunc}.

   We say that :math:`\p{Q,q,S,s}` satisfies a
   :math:`\p{P,p,T,t,\rho, h, \alpha}`-*quadratic Lyapunov inequality*
   for algorithm
   :eq:`eq:linear_system_with_nonlinearity`
   over the problem class defined by
   :math:`(\mathcal{F}_i)_{i\in\IndexFunc}` and
   :math:`(\mathcal{G}_i)_{i\in\IndexOp}` if

   .. _eq:c2:

   .. _eq:c3:

   .. _eq:c4:

   .. _eq:c1:

   .. math::
      :nowrap:

      \begin{align}
      \p{\forall k \in \naturals} &\quad \mathcal{V}\p{Q,q,k+\alpha+1} \leq \rho \mathcal{V}\p{Q,q,k} - \mathcal{R}\p{S,s,k}, \tag{C1} \\
      \p{\forall k \in \naturals} &\quad \mathcal{V}\p{Q,q,k} \geq \mathcal{V}\p{P,p,k} \geq 0, \tag{C2} \\
      \p{\forall k \in \naturals} &\quad \mathcal{R}\p{S,s,k} \geq \mathcal{R}\p{T,t,k}\geq 0, \tag{C3} \\
      \p{\forall k \in \naturals} &\quad \mathcal{R}\p{S,s,k+1} \leq \mathcal{R}\p{S,s,k}, \tag{C4}
      \end{align}

   hold for each

   .. math::

      \bx_{0}, \qquad
      \p{ ( \bx^{k}, \bu^{k}, \by^{k}, \bFcn^{k} ) }_{k\in\naturals}, \qquad
      \p{y^{\star}, \hat{\bu}^{\star},\bFcn^{\star}},

   and for each

   .. math::

      \p{f_{i}}_{i\in\IndexFunc} \in \prod_{i\in\IndexFunc} \mathcal{F}_{i},
      \qquad
      \p{G_i}_{i\in\IndexOp} \in \prod_{i\in\IndexOp} \mathcal{G}_i,

   where :ref:`(C4) <eq:c4>` is an optional requirement that may be
   removed.

Definition 5.2.1 is a trajectory-level condition:
:ref:`(C1) <eq:c1>`-:ref:`(C3) <eq:c3>` (and optionally
:ref:`(C4) <eq:c4>`) must hold for all iterates and all admissible
functions/operators in the class. This statement is not directly
computational. In practice, the user specifies
:math:`\p{P,p,T,t,\rho,h,\alpha}` and searches for
:math:`\p{Q,q,S,s}`. The SDP sufficient condition that makes this search
computable is given in :ref:`Theorem 5.2.2 (Iteration-independent Lyapunov inequality via SDP) <thm:iteration_independent_lyapunov>`.
When such :math:`\p{Q,q,S,s}` exists, the choice of
:math:`\p{P,p,T,t,\rho,h,\alpha}` determines the convergence guarantee.
In particular, we obtain:

- Linear setting: if :math:`\rho \in [0,1[`, then

  .. math::

    \begin{aligned}
            0 \leq \mathcal{V}\p{P,p,k} \leq \mathcal{V}\p{Q,q,k} \leq \rho^{\lfloor k/\p{\alpha+1}\rfloor }\max_{i\in\llbracket0,\alpha\rrbracket}\mathcal{V}\p{Q,q,i} \xrightarrow[k\rightarrow \infty]{} 0.
        \end{aligned}

  Thus, :math:`(\mathcal{V}\p{P,p,k})_{k\in\naturals}` converges to zero

  .. math::
    \sqrt[\alpha + 1]{\rho}\textup{-linearly}.

- Sublinear setting: if :math:`\rho = 1`, then

  .. math::

    \begin{aligned}
            \p{\forall k \in \naturals}\quad \sum_{i=0}^{k} \mathcal{R}\p{T,t,i} \leq \sum_{i=0}^{k} \mathcal{R}\p{S,s,i} \leq \sum_{j=0}^{\alpha} \mathcal{V}\p{Q,q,j},
        \end{aligned}

  by a telescoping summation argument. In particular,

  .. math::
    (\mathcal{R}\p{T,t,k})_{k\in\naturals}

  is summable, converges to zero, and, e.g.,

  .. math::
    \min_{i\in \llbracket 0,k \rrbracket } \mathcal{R}\p{T,t,i} \in \mathcal{O}\p{1/k} \quad \textup{ as } \quad  k\to\infty.

  If the optional requirement :ref:`(C4) <eq:c4>` holds, we obtain the
  stronger last-iterate convergence result

  .. math::
    \mathcal{R}\p{T,t,k} \in o\p{1/k} \quad \textup{ as } \quad  k\to\infty.

The user-chosen :math:`\p{P,p,T,t}` fixes the particular Lyapunov and
residual expressions :math:`\mathcal{V}\p{P,p,\cdot}` and
:math:`\mathcal{R}\p{T,t,\cdot}` to be analyzed. For built-in choices:

.. _theory-iter-independent-linear-builtins:

- Linear convergence:
  :ref:`Linear-convergence helpers <iter-independent-linear-helpers>`.

.. _theory-iter-independent-sublinear-builtins:

- Sublinear convergence:
  :ref:`Sublinear-convergence helpers <iter-independent-sublinear-helpers>`.

For the corresponding API entry points, use
:meth:`~autolyap.IterationIndependent.search_lyapunov` for fixed
:math:`\rho`, and
:meth:`~autolyap.IterationIndependent.LinearConvergence.bisection_search_rho`
to estimate the smallest feasible contraction factor in the
linear-convergence setting.

Role of :math:`h` and :math:`\alpha`:

- :math:`h` is a history parameter for :math:`\mathcal{V}`
  (see :eq:`eq:iteration_independent_lyapunov:v`).

  .. math::

     \text{the } \mathcal{V}\text{-window length is } h+1.
- :math:`\alpha` is an overlap parameter that induces the shift
  :math:`k \mapsto k+\alpha+1` in :ref:`(C1) <eq:c1>`.
- :math:`h` and :math:`\alpha` determine the :math:`\mathcal{R}` window
  (see :eq:`eq:iteration_independent_lyapunov:r`).

  .. math::

     \text{the } \mathcal{R}\text{-window length is } h+\alpha+2.

We now state the finite-dimensional SDP condition that certifies the
existence of such :math:`\p{Q,q,S,s}`.

.. container:: theorem

   .. _thm:iteration_independent_lyapunov:

   **Theorem 5.2.2 (Iteration-independent Lyapunov inequality via SDP).**

   Suppose that
   :ref:`Assumption 3.1 (Well-posedness) <ass:well-posedness>`, :ref:`Assumption 4.1 (Interpolation conditions) <ass:interpolation>`
   and :eq:`eq:constant_abcd` hold. Let

   .. math::

      \rho \in [0,1], \qquad h\in\naturals, \qquad \alpha\in\naturals,

   and let

   .. math::

      P \in \sym^{n+\p{h+1}\NumEval+m}, \qquad
      p\in\mathbb{R}^{\p{h+1}\NumEvalFunc + \NumFunc},

   such that

   .. math::

      \begin{aligned}
              \p{\forall k \in \naturals} \qquad \mathcal{V}\p{P,p,k} \geq 0,
          \end{aligned}

   where :math:`\mathcal{V}` is defined in
   :eq:`eq:iteration_independent_lyapunov:v`,
   and let

   .. math::

      T\in\sym^{n+\p{h+\alpha+2}\NumEval+m}, \qquad
      t\in\mathbb{R}^{\p{h+\alpha+2}\NumEvalFunc + \NumFunc},

   such that

   .. math::

      \begin{aligned}
              \p{\forall k \in \naturals} \qquad \mathcal{R}\p{T,t,k} \geq 0,
          \end{aligned}

   where :math:`\mathcal{R}` is defined in
   :eq:`eq:iteration_independent_lyapunov:r`.
   Then a sufficient condition for there to exist :math:`\p{Q,q,S,s}`
   that satisfies a :math:`\p{P,p,T,t,\rho, h, \alpha}`-quadratic
   Lyapunov inequality for algorithm
   :eq:`eq:linear_system_with_nonlinearity`
   over the problem class defined by
   :math:`\p{\mathcal{F}_i}_{i\in\IndexFunc}` and
   :math:`\p{\mathcal{G}_i}_{i\in\IndexOp}` (recall that
   :ref:`(C4) <eq:c4>` is optional and may be omitted) is that the
   following system of constraints, which reuses the lifted
   matrices/vectors from :doc:`5.1. Performance estimation via SDPs
   </theory/performance_estimation_via_sdps>`, including the lifted state
   matrices :eq:`eq:x_mats`, and the interpolation terms
   :eq:`eq:w_func_ineq`, :eq:`eq:w_func_eq`, :eq:`eq:w_op`,
   :eq:`eq:f_func_ineq`, and :eq:`eq:f_func_eq`,

   .. container:: subequations

      .. _eq:iteration_independent_lyapunov:condition:
      .. _eq:iteration_independent_lyapunov:condition:func-ineq:
      .. _eq:iteration_independent_lyapunov:condition:func-eq:
      .. _eq:iteration_independent_lyapunov:condition:op:
      .. _eq:iteration_independent_lyapunov:condition:psd:
      .. _eq:iteration_independent_lyapunov:condition:zero:
      .. _eq:iteration_independent_lyapunov:condition:q_mat:
      .. _eq:iteration_independent_lyapunov:condition:q:
      .. _eq:iteration_independent_lyapunov:condition:s_mat:
      .. _eq:iteration_independent_lyapunov:condition:s:

      .. math::
         :no-wrap:

         \begin{align}
                 & \textbf{for each}\ \textup{cond} \in\set{\href{#eq-c1}{\textup{C1}},\href{#eq-c2}{\textup{C2}},\href{#eq-c3}{\textup{C3}},\href{#eq-c4}{\textup{C4}}} \notag \\
                 &\qquad
                 \left(
                 \begin{array}{@{}c@{}}
                     \forall i \in\IndexFunc \\
                     \forall o \in \mathcal{O}^{\textup{func-ineq}}_{i} \\
                     \forall \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{0,\PEPMaxIter_{\textup{cond}}}
                 \end{array}
                 \right) \qquad \lambda_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{\textup{func-ineq},\,\textup{cond}} \geq 0, \tag{5.28a}\\
                 &\qquad
                 \left(
                 \begin{array}{@{}c@{}}
                     \forall i \in\IndexFunc \\
                     \forall o \in \mathcal{O}^{\textup{func-eq}}_{i} \\
                     \forall \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{0,\PEPMaxIter_{\textup{cond}}}
                 \end{array}
                 \right) \qquad
                 \nu_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{\textup{func-eq},\,\textup{cond}} \in \reals, \tag{5.28b}\\
                 &\qquad
                 \left(
                 \begin{array}{@{}c@{}}
                     \forall i \in\IndexFunc \\
                     \forall o \in \mathcal{O}^{\textup{op}}_{i} \\
                     \forall \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{0,\PEPMaxIter_{\textup{cond}}}
                 \end{array}
                 \right) \qquad
                 \lambda_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{\textup{op},\,\textup{cond}} \geq 0, \tag{5.28c}\\
                 &\qquad - W^{\textup{cond}}
                 + \sum_{\substack{i\in\IndexFunc \\ o \in \mathcal{O}^{\textup{func-ineq}}_{i} \\ \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{0,\PEPMaxIter_{\textup{cond}}} }}
                 \lambda_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{\textup{func-ineq},\,\textup{cond}}
                 W_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{0,\,\PEPMaxIter_{\textup{cond}},\,\textup{func-ineq}} \notag \\
                 &\qquad \qquad + \sum_{\substack{i\in\IndexFunc  \\ o \in \mathcal{O}^{\textup{func-eq}}_{i} \\ \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{0,\PEPMaxIter_{\textup{cond}}} }}
                 \nu_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{\textup{func-eq},\,\textup{cond}}
                 W_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{0,\,\PEPMaxIter_{\textup{cond}},\,\textup{func-eq}} \notag \\
                 &\qquad \qquad + \sum_{\substack{i\in\IndexOp \\ o \in \mathcal{O}^{\textup{op}}_{i} \\ \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{0,\PEPMaxIter_{\textup{cond}}} }}
                 \lambda_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{\textup{op},\,\textup{cond}}
                 W_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{0,\,\PEPMaxIter_{\textup{cond}},\,\textup{op}}
                 \succeq 0, \tag{5.28d}\\
                 &\qquad
                 - w^{\textup{cond}}
                 + \sum_{\substack{i\in\IndexFunc \\ o \in \mathcal{O}^{\textup{func-ineq}}_{i} \\ \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{0,\PEPMaxIter_{\textup{cond}}} }}
                 \lambda_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{\textup{func-ineq},\,\textup{cond}}
                 F_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{0,\,\PEPMaxIter_{\textup{cond}},\,\textup{func-ineq}} \notag \\
                 &\qquad \qquad + \sum_{\substack{i\in\IndexFunc  \\ o \in \mathcal{O}^{\textup{func-eq}}_{i} \\ \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{0,\PEPMaxIter_{\textup{cond}}} }}
                 \nu_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{\textup{func-eq},\,\textup{cond}}
                 F_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{0,\,\PEPMaxIter_{\textup{cond}},\,\textup{func-eq}}
                 = 0, \tag{5.28e}\\
                 & \mathbf{end} \notag \\
                 & Q \in \sym^{n+\p{h+1}\NumEval+m}, \tag{5.28f}\\
                 & q \in \mathbb{R}^{\p{h+1}\NumEvalFunc + \NumFunc}, \tag{5.28g}\\
                 & S \in \sym^{n+\p{h+\alpha+2}\NumEval+m}, \tag{5.28h}\\
                 & s \in \reals^{\p{h+\alpha+2}\NumEvalFunc + \NumFunc}. \tag{5.28i}
         \end{align}

   is feasible for the scalars

   .. math::

      \begin{aligned}
              &\lambda_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{\textup{func-ineq},\,\textup{cond}}, \\
              &\nu_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{\textup{func-eq},\,\textup{cond}}, \\
              &\lambda_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{\textup{op},\,\textup{cond}},
          \end{aligned}

   the matrices :math:`Q` and :math:`S`, and the vectors :math:`q` and
   :math:`s`, where


   .. math::
      :label: eq:iteration_independent_lyapunov:w_c1_mat
      :class: eq-align-w

      \begin{aligned}
              W^{\href{#eq-c1}{\textup{C1}}} &= \p{\Theta_{1}^{\href{#eq-c1}{\textup{C1}}}}^{\top}Q\Theta_{1}^{\href{#eq-c1}{\textup{C1}}} - \rho \p{\Theta_{0}^{\href{#eq-c1}{\textup{C1}}}}^{\top}Q\Theta_{0}^{\href{#eq-c1}{\textup{C1}}} + S,
          \end{aligned}

   .. math::
      :label: eq:iteration_independent_lyapunov:w_c1
      :class: eq-align-w

      \begin{aligned}
              w^{\href{#eq-c1}{\textup{C1}}} &= \p{\theta_{1}^{\href{#eq-c1}{\textup{C1}}} - \rho \theta_{0}^{\href{#eq-c1}{\textup{C1}}} }^{\top}q + s,
          \end{aligned}

   .. math::
      :label: eq:iteration_independent_lyapunov:w_c2_mat
      :class: eq-align-w

      \begin{aligned}
              W^{\href{#eq-c2}{\textup{C2}}} &= P-Q,
          \end{aligned}

   .. math::
      :label: eq:iteration_independent_lyapunov:w_c2
      :class: eq-align-w

      \begin{aligned}
              w^{\href{#eq-c2}{\textup{C2}}} &= p-q,
          \end{aligned}

   .. math::
      :label: eq:iteration_independent_lyapunov:w_c3_mat
      :class: eq-align-w

      \begin{aligned}
              W^{\href{#eq-c3}{\textup{C3}}} &= T - S,
          \end{aligned}

   .. math::
      :label: eq:iteration_independent_lyapunov:w_c3
      :class: eq-align-w

      \begin{aligned}
              w^{\href{#eq-c3}{\textup{C3}}} &= t - s,
          \end{aligned}

   .. math::
      :label: eq:iteration_independent_lyapunov:w_c4_mat
      :class: eq-align-w

      \begin{aligned}
              W^{\href{#eq-c4}{\textup{C4}}} &= \p{\Theta_{1}^{\href{#eq-c4}{\textup{C4}}}}^{\top}S\Theta_{1}^{\href{#eq-c4}{\textup{C4}}} - \p{\Theta_{0}^{\href{#eq-c4}{\textup{C4}}}}^{\top}S\Theta_{0}^{\href{#eq-c4}{\textup{C4}}},
          \end{aligned}

   .. math::
      :label: eq:iteration_independent_lyapunov:w_c4
      :class: eq-align-w

      \begin{aligned}
              w^{\href{#eq-c4}{\textup{C4}}} &= \p{\theta_{1}^{\href{#eq-c4}{\textup{C4}}} - \theta_{0}^{\href{#eq-c4}{\textup{C4}}} }^{\top}s,
          \end{aligned}

   .. math::

      \begin{aligned}
              \PEPMaxIter_{\textup{cond}} &=
              \begin{cases}
                  h + \alpha + 1& \text{ if } \textup{cond} \in \set{\href{#eq-c1}{\textup{C1}},\href{#eq-c3}{\textup{C3}}},\\
                  h & \text{ if } \textup{cond} \in \set{\href{#eq-c2}{\textup{C2}}}, \\
                  h + \alpha + 2 & \text{ if } \textup{cond} \in \set{\href{#eq-c4}{\textup{C4}}},
              \end{cases} \nonumber
          \end{aligned}

   and


   .. math::
      :label: eq:iteration_independent_lyapunov:theta0:c1_mat
      :class: eq-align-theta

      \begin{aligned}
              \Theta_{0}^{\href{#eq-c1}{\textup{C1}}} &= 
              \underbracket{
              \begin{bmatrix}
                  I_{n+\p{h+1}\NumEval } & 0_{\p{n+\p{h+1}\NumEval}\times\p{\alpha + 1}\NumEval } & 0_{\p{n+\p{h+1}\NumEval}\times m} \\
                  0_{m \times \p{n+\p{h+1}\NumEval}} & 0_{m \times\p{\alpha + 1}\NumEval } & I_{m}
              \end{bmatrix}
              }_{
              \in \reals^{\p{n+\p{h+1}\NumEval+m}\times\p{n+\p{h + \alpha +2}\NumEval+m}}
              }, 
          \end{aligned}

   .. math::
      :label: eq:iteration_independent_lyapunov:theta0:c1
      :class: eq-align-theta

      \begin{aligned}
              \theta_{0}^{\href{#eq-c1}{\textup{C1}}} &=
              \underbracket{
              \begin{bmatrix}
                  I_{\p{h+1}\NumEvalFunc} & 0_{\p{h+1}\NumEvalFunc \times \p{\alpha + 1}\NumEvalFunc } & 0_{\p{h+1}\NumEvalFunc \times \NumFunc } \\
                  0_{\NumFunc\times \p{h+1}\NumEvalFunc} &0_{\NumFunc\times \p{\alpha + 1}\NumEvalFunc} & I_{\NumFunc}
              \end{bmatrix}
              }_{
              \in \reals^{\p{\p{h+1}\NumEvalFunc + \NumFunc}\times\p{\p{h+\alpha+2}\NumEvalFunc + \NumFunc}}
              },
          \end{aligned}

   .. math::
      :label: eq:iteration_independent_lyapunov:theta1:c1_mat
      :class: eq-align-theta

      \begin{aligned}
              \Theta_{1}^{\href{#eq-c1}{\textup{C1}}} &=
              \underbracket{
              \begin{bmatrix}
                X_{\alpha + 1}^{0,h + \alpha + 1} \\
                0_{\p{\p{h + 1}\NumEval + m} \times \p{n + \p{\alpha + 1}\NumEval } } \quad I_{\p{h + 1}\NumEval + m}
              \end{bmatrix}
              }_{
              \in \reals^{\p{n+\p{h+1}\NumEval+m} \times \p{n+\p{h + \alpha +2}\NumEval+m}}
              }
              ,
          \end{aligned}

   .. math::
      :label: eq:iteration_independent_lyapunov:theta1:c1
      :class: eq-align-theta

      \begin{aligned}
              \theta_{1}^{\href{#eq-c1}{\textup{C1}}} &=
              \underbracket{
              \begin{bmatrix}
                  0_{\p{\p{h+1}\NumEvalFunc + \NumFunc} \times \p{\alpha+1}\NumEvalFunc } & I_{\p{h+1}\NumEvalFunc + \NumFunc }
              \end{bmatrix}
              }_{
              \in \reals^{\p{\p{h+1}\NumEvalFunc + \NumFunc}\times\p{\p{h+\alpha+2}\NumEvalFunc + \NumFunc}}
              },
          \end{aligned}

   .. math::
      :label: eq:iteration_independent_lyapunov:theta0:c4_mat
      :class: eq-align-theta

      \begin{aligned}
              \Theta_{0}^{\href{#eq-c4}{\textup{C4}}} &=
              \underbracket{
              \begin{bmatrix}
                  I_{n+\p{h+\alpha+2}\NumEval } & 0_{\p{n+\p{h+\alpha+2}\NumEval}\times \NumEval } & 0_{\p{n+\p{h+\alpha+2}\NumEval}\times m} \\
                  0_{m \times \p{n+\p{h+\alpha+2}\NumEval}} & 0_{m \times \NumEval } & I_{m}
              \end{bmatrix}
              }_{
              \in \reals^{\p{n+\p{h+\alpha+2}\NumEval+m}\times\p{n+\p{h + \alpha +3}\NumEval+m}}
              }, 
          \end{aligned}

   .. math::
      :label: eq:iteration_independent_lyapunov:theta0:c4
      :class: eq-align-theta

      \begin{aligned}
              \theta_{0}^{\href{#eq-c4}{\textup{C4}}} &=
              \underbracket{
              \begin{bmatrix}
                  I_{\p{h+\alpha+2}\NumEvalFunc} & 0_{\p{h+\alpha+2}\NumEvalFunc \times \NumEvalFunc } & 0_{\p{h+\alpha+2}\NumEvalFunc \times \NumFunc } \\
                  0_{\NumFunc\times \p{h+\alpha+2}\NumEvalFunc} &0_{\NumFunc\times \NumEvalFunc} & I_{\NumFunc}
              \end{bmatrix}
              }_{
              \in \reals^{\p{\p{h+\alpha+2}\NumEvalFunc + \NumFunc}\times\p{\p{h+\alpha+3}\NumEvalFunc + \NumFunc}}
              },
          \end{aligned}

   .. math::
      :label: eq:iteration_independent_lyapunov:theta1:c4_mat
      :class: eq-align-theta

      \begin{aligned}
              \Theta_{1}^{\href{#eq-c4}{\textup{C4}}} &=
              \underbracket{
              \begin{bmatrix}
                X_{1}^{0,h + \alpha + 2} \\
                0_{\p{\p{h + \alpha + 2}\NumEval + m} \times \p{n + \NumEval } } \quad I_{\p{h + \alpha + 2}\NumEval + m}
              \end{bmatrix}
              }_{
              \in \reals^{\p{n+\p{h+\alpha+2}\NumEval+m} \times \p{n+\p{h + \alpha +3}\NumEval+m}}
              }
              ,
          \end{aligned}

   .. math::
      :label: eq:iteration_independent_lyapunov:theta1:c4
      :class: eq-align-theta

      \begin{aligned}
              \theta_{1}^{\href{#eq-c4}{\textup{C4}}} &=
              \underbracket{
              \begin{bmatrix}
                  0_{\p{\p{h+\alpha+2}\NumEvalFunc + \NumFunc} \times \NumEvalFunc } & I_{\p{h+\alpha+2}\NumEvalFunc + \NumFunc }
              \end{bmatrix}
              }_{
              \in \reals^{\p{\p{h+\alpha+2}\NumEvalFunc + \NumFunc}\times\p{\p{h+\alpha+3}\NumEvalFunc + \NumFunc}}
              }.
          \end{aligned}

   Furthermore, if the interpolation conditions for
   :math:`\p{\mathcal{F}_{i}}_{i\in\IndexFunc}` and
   :math:`\p{\mathcal{G}_{i}}_{i\in\IndexOp}` are tight,


   .. math::

      \begin{aligned}
              & \textbf{for each}\ \textup{cond} \in\set{\href{#eq-c1}{\textup{C1}},\href{#eq-c2}{\textup{C2}},\href{#eq-c3}{\textup{C3}},\href{#eq-c4}{\textup{C4}}}   \\
              & \quad \dim \calH \geq n + \p{\PEPMaxIter_{\textup{cond}}+1}\NumEval + m, \\
              & \mathbf{end}
          \end{aligned}

   and there exists


   .. math::

      \begin{aligned}
              & \textbf{for each}\ \textup{cond} \in\set{\href{#eq-c1}{\textup{C1}},\href{#eq-c2}{\textup{C2}},\href{#eq-c3}{\textup{C3}},\href{#eq-c4}{\textup{C4}}}   \\
              & \quad G_{\textup{cond}}\in\sym_{++}^{n + \p{\PEPMaxIter_{\textup{cond}}+1}\NumEval + m}, \\
              & \quad \bchi_{\textup{cond}}\in\reals^{\p{\PEPMaxIter_{\textup{cond}}+1}\NumEvalFunc + \NumFunc}, \\
              & \mathbf{end}
          \end{aligned}

   such that


   .. math::

      \begin{aligned}
              & \textbf{for each}\ \textup{cond} \in\set{\href{#eq-c1}{\textup{C1}},\href{#eq-c2}{\textup{C2}},\href{#eq-c3}{\textup{C3}},\href{#eq-c4}{\textup{C4}}} \\
              & \quad
              \left(
              \begin{array}{@{}c@{}}
                  \forall i \in\IndexFunc \\
                  \forall o \in \mathcal{O}^{\textup{func-ineq}}_{i} \\
                  \forall \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{0,\PEPMaxIter_{\textup{cond}}}
              \end{array}
              \right) \quad 
              \begin{aligned}
                  & \bm{\chi}^{\top}_{\textup{cond}} F_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{0,\,\PEPMaxIter_{\textup{cond}},\,\textup{func-ineq}} \\    
                  & + \trace\p{W_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{0,\,\PEPMaxIter_{\textup{cond}},\,\textup{func-ineq}}G_{\textup{cond}}} \leq 0,
              \end{aligned}
              \\ 
              & \quad
              \left(
              \begin{array}{@{}c@{}}
                  \forall i \in\IndexFunc \\
                  \forall o \in \mathcal{O}^{\textup{func-eq}}_{i} \\
                  \forall \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{0,\PEPMaxIter_{\textup{cond}}}
              \end{array}
              \right) \quad
              \begin{aligned}
                  & \bm{\chi}^{\top}_{\textup{cond}} F_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{0,\,\PEPMaxIter_{\textup{cond}},\,\textup{func-eq}} \\
                  & + \trace\p{W_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{0,\,\PEPMaxIter_{\textup{cond}},\,\textup{func-eq}}G_{\textup{cond}}} = 0,
              \end{aligned}
              \\
              & \quad
              \left(
              \begin{array}{@{}c@{}}
                  \forall i \in\IndexFunc \\
                  \forall o \in \mathcal{O}^{\textup{op}}_{i} \\
                  \forall \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{0,\PEPMaxIter_{\textup{cond}}}
              \end{array}
              \right) \quad
              \trace\p{W_{\p{i,j_{1},k_{1},\ldots,j_{n_{i,o}},k_{n_{i,o}},o}}^{0,\,\PEPMaxIter_{\textup{cond}},\,\textup{op}}G_{\textup{cond}}} \leq 0,
              \\
              & \mathbf{end}
          \end{aligned}

   then :ref:`(5.28) <eq:iteration_independent_lyapunov:condition>` is
   also a necessary condition.

.. container:: proof

   *Proof.* By induction, we only need to consider the case :math:`k=0`
   in :ref:`(C1) <eq:c1>` to :ref:`(C4) <eq:c4>`. Additionally, note
   that


   .. math::

      \begin{aligned}
              \p{\bx^{0},\bu^{0},\ldots,\bu^{h},\hat{\bu}^{\star},y^{\star}} &= \p{\Theta_{0}^{\href{#eq-c1}{\textup{C1}}} \kron \Id }\p{\bx^{0},\bu^{0},\ldots,\bu^{h + \alpha + 1},\hat{\bu}^{\star},y^{\star}}, \\
              \p{\bFcn^{0},\ldots,\bFcn^{h},\bFcn^{\star}} &= \theta_{0}^{\href{#eq-c1}{\textup{C1}}}\p{\bFcn^{0},\ldots,\bFcn^{h + \alpha + 1},\bFcn^{\star}}, \\
              \p{\bx^{\alpha + 1},\bu^{\alpha + 1},\ldots,\bu^{h + \alpha + 1},\hat{\bu}^{\star},y^{\star}} &= \p{\Theta_{1}^{\href{#eq-c1}{\textup{C1}}} \kron \Id }\p{\bx^{0},\bu^{0},\ldots,\bu^{h + \alpha + 1},\hat{\bu}^{\star},y^{\star}}, \\
              \p{\bFcn^{\alpha + 1},\ldots,\bFcn^{h + \alpha + 1},\bFcn^{\star}} &= \theta_{1}^{\href{#eq-c1}{\textup{C1}}}\p{\bFcn^{0},\ldots,\bFcn^{h + \alpha + 1},\bFcn^{\star}}, \\
              \p{\bx^{0},\bu^{0},\ldots,\bu^{h+\alpha + 1},\hat{\bu}^{\star},y^{\star}} &= \p{\Theta_{0}^{\href{#eq-c4}{\textup{C4}}} \kron \Id }\p{\bx^{0},\bu^{0},\ldots,\bu^{h + \alpha + 2},\hat{\bu}^{\star},y^{\star}}, \\
              \p{\bFcn^{0},\ldots,\bFcn^{h + \alpha + 1},\bFcn^{\star}} &= \theta_{0}^{\href{#eq-c4}{\textup{C4}}} \p{\bFcn^{0},\ldots,\bFcn^{h + \alpha + 2},\bFcn^{\star}}, \\
              \p{\bx^{1},\bu^{1},\ldots,\bu^{h + \alpha + 2},\hat{\bu}^{\star},y^{\star}} &= \p{\Theta_{1}^{\href{#eq-c4}{\textup{C4}}} \kron \Id }\p{\bx^{0},\bu^{0},\ldots,\bu^{h + \alpha + 2},\hat{\bu}^{\star},y^{\star}}, \\
              \p{\bFcn^{1},\ldots,\bFcn^{h + \alpha + 2},\bFcn^{\star}} &= \theta_{1}^{\href{#eq-c4}{\textup{C4}}} \p{\bFcn^{0},\ldots,\bFcn^{h + \alpha + 2},\bFcn^{\star}}, 
          \end{aligned}

   where we have used
   :eq:`eq:iteration_independent_lyapunov:theta0:c1_mat`,
   :eq:`eq:iteration_independent_lyapunov:theta0:c1`,
   :eq:`eq:iteration_independent_lyapunov:theta1:c1_mat`,
   :eq:`eq:iteration_independent_lyapunov:theta1:c1`,
   :eq:`eq:iteration_independent_lyapunov:theta0:c4_mat`,
   :eq:`eq:iteration_independent_lyapunov:theta0:c4`,
   :eq:`eq:iteration_independent_lyapunov:theta1:c4_mat`, and
   :eq:`eq:iteration_independent_lyapunov:theta1:c4`, respectively.

   First, suppose that the parameters :math:`\p{Q,q,S,s}` are fixed.
   Note that


   .. _eq:iteration_independent_lyapunov:obj_c1:

   .. math::
      :label: eq:iteration_independent_lyapunov:obj_c1

      \begin{aligned}
              &\mathcal{V}\p{Q,q,\alpha+1} - \rho \mathcal{V}\p{Q,q,0} + \mathcal{R}\p{S,s,0} \nonumber \\
              & = \quadform{Q}{\p{\bx^{\alpha + 1},\bu^{\alpha + 1},\ldots,\bu^{h + \alpha + 1},\hat{\bu}^{\star},y^{\star}}} + q^{\top}\p{\bFcn^{\alpha + 1},\ldots,\bFcn^{h + \alpha + 1},\bFcn^{\star}} \nonumber \\
              &\quad - \rho 
              \quadform{Q}{\p{\bx^{0},\bu^{0},\ldots,\bu^{h},\hat{\bu}^{\star},y^{\star}}} - \rho q^{\top}\p{\bFcn^{0},\ldots,\bFcn^{h},\bFcn^{\star}}
               \nonumber \\
              &\quad + \quadform{S}{\p{\bx^{0},\bu^{0},\ldots,\bu^{h + \alpha + 1},\hat{\bu}^{\star},y^{\star}}} + s^{\top}\p{\bFcn^{0},\ldots,\bFcn^{h + \alpha + 1},\bFcn^{\star}} \nonumber \\
              & = \quadform{\p{\Theta_{1}^{\href{#eq-c1}{\textup{C1}}}}^{\top}Q\Theta_{1}^{\href{#eq-c1}{\textup{C1}}} - \rho \p{\Theta_{0}^{\href{#eq-c1}{\textup{C1}}}}^{\top}Q\Theta_{0}^{\href{#eq-c1}{\textup{C1}}} + S}{\p{\bx^{0},\bu^{0},\ldots,\bu^{h + \alpha + 1},\hat{\bu}^{\star},y^{\star}}} \nonumber \\
              &\quad + \p{q^{\top}\p{\theta_{1}^{\href{#eq-c1}{\textup{C1}}} - \rho \theta_{0}^{\href{#eq-c1}{\textup{C1}}} } + s^{\top}} \p{\bFcn^{0},\ldots,\bFcn^{h + \alpha + 1},\bFcn^{\star}} \nonumber \\
              &  = \quadform{W^{\href{#eq-c1}{\textup{C1}}}}{\p{\bx^{0},\bu^{0},\ldots,\bu^{h + \alpha + 1},\hat{\bu}^{\star},y^{\star}}}
              + \p{w^{\href{#eq-c1}{\textup{C1}}}}^{\top}\p{\bFcn^{0},\ldots,\bFcn^{h + \alpha + 1},\bFcn^{\star}} \nonumber
          \end{aligned}

   where :eq:`eq:iteration_independent_lyapunov:w_c1_mat`
   and :eq:`eq:iteration_independent_lyapunov:w_c1`
   are used in the last equality. Therefore, using
   :eq:`eq:iteration_independent_lyapunov:obj_c1`
   as the objective function in :ref:`(PEP) <eq:pep>`,
   :ref:`Theorem 5.1.1 (Performance estimation via SDP) <thm:pep_leq_zero>`
   gives that :ref:`(5.28) <eq:iteration_independent_lyapunov:condition>`,
   with :math:`\textup{cond} = \href{#eq-c1}{\textup{C1}}`, is a sufficient
   condition for :ref:`(C1) <eq:c1>`. Note that

   .. _eq:iteration_independent_lyapunov:obj_c2:

   .. math::
      :label: eq:iteration_independent_lyapunov:obj_c2

      \begin{aligned}
              &\mathcal{V}\p{P,p,0} - \mathcal{V}\p{Q,q,0} \notag \\
              & = \quadform{P-Q}{\p{\bx^{0},\bu^{0},\ldots,\bu^{h},\hat{\bu}^{\star},y^{\star}}}
              + \p{p-q}^{\top}\p{\bFcn^{0},\ldots,\bFcn^{h},\bFcn^{\star}} \notag \\
              & = \quadform{W^{\href{#eq-c2}{\textup{C2}}}}{\p{\bx^{0},\bu^{0},\ldots,\bu^{h},\hat{\bu}^{\star},y^{\star}}}
              + \p{w^{\href{#eq-c2}{\textup{C2}}}}^{\top}\p{\bFcn^{0},\ldots,\bFcn^{h},\bFcn^{\star}}
          \end{aligned}

   where :eq:`eq:iteration_independent_lyapunov:w_c2_mat`
   and :eq:`eq:iteration_independent_lyapunov:w_c2`
   are used in the last equality. Therefore, using the
   :eq:`eq:iteration_independent_lyapunov:obj_c2`
   as the objective function in :ref:`(PEP) <eq:pep>`,
   :ref:`Theorem 5.1.1 (Performance estimation via SDP) <thm:pep_leq_zero>`
   gives that :ref:`(5.28) <eq:iteration_independent_lyapunov:condition>`,
   with :math:`\textup{cond} = \href{#eq-c2}{\textup{C2}}`, is a sufficient
   condition for :ref:`(C2) <eq:c2>`. Note that

   .. _eq:iteration_independent_lyapunov:obj_c3:

   .. math::
      :label: eq:iteration_independent_lyapunov:obj_c3

      \begin{aligned}
              & \mathcal{R}\p{T,t,0} - \mathcal{R}\p{S,s,0} \notag \\
              & = \quadform{T-S}{\p{\bx^{0},\bu^{0},\ldots,\bu^{h + \alpha + 1},\hat{\bu}^{\star},y^{\star}}}
              + \p{t-s}^{\top}\p{\bFcn^{0},\ldots,\bFcn^{h + \alpha + 1},\bFcn^{\star}} \notag \\
              & = \quadform{W^{\href{#eq-c3}{\textup{C3}}}}{\p{\bx^{0},\bu^{0},\ldots,\bu^{h + \alpha + 1},\hat{\bu}^{\star},y^{\star}}}
              + \p{w^{\href{#eq-c3}{\textup{C3}}}}^{\top} \p{\bFcn^{0},\ldots,\bFcn^{h + \alpha + 1},\bFcn^{\star}}
          \end{aligned}

   where :eq:`eq:iteration_independent_lyapunov:w_c3_mat`
   and :eq:`eq:iteration_independent_lyapunov:w_c3`
   are used in the last equality. Therefore, using the
   :eq:`eq:iteration_independent_lyapunov:obj_c3`
   as the objective function in :ref:`(PEP) <eq:pep>`,
   :ref:`Theorem 5.1.1 (Performance estimation via SDP) <thm:pep_leq_zero>`
   gives that :ref:`(5.28) <eq:iteration_independent_lyapunov:condition>`,
   with :math:`\textup{cond} = \href{#eq-c3}{\textup{C3}}`, is a sufficient
   condition for :ref:`(C3) <eq:c3>`. Note that

   .. _eq:iteration_independent_lyapunov:obj_c4:

   .. math::
      :label: eq:iteration_independent_lyapunov:obj_c4

      \begin{aligned}
              &\mathcal{R}\p{S,s,1} - \mathcal{R}\p{S,s,0} \notag \\
              & 
              = \quadform{S}{\p{\bx^{1},\bu^{1},\ldots,\bu^{h + \alpha + 2},\hat{\bu}^{\star},y^{\star}}} + s^{\top}\p{\bFcn^{1},\ldots,\bFcn^{h + \alpha + 2},\bFcn^{\star}} \notag \\
              & \quad -
              \quadform{S}{\p{\bx^{0},\bu^{0},\ldots,\bu^{h + \alpha + 1},\hat{\bu}^{\star},y^{\star}}} - s^{\top} \p{\bFcn^{0},\ldots,\bFcn^{h + \alpha + 1},\bFcn^{\star}} \notag \\
              &  = \quadform{\p{\Theta_{1}^{\href{#eq-c4}{\textup{C4}}}}^{\top}S\Theta_{1}^{\href{#eq-c4}{\textup{C4}}} - \p{\Theta_{0}^{\href{#eq-c4}{\textup{C4}}}}^{\top}S\Theta_{0}^{\href{#eq-c4}{\textup{C4}}}}{\p{\bx^{0},\bu^{0},\ldots,\bu^{h + \alpha + 2},\hat{\bu}^{\star},y^{\star}}} \notag \\
              & \quad + \p{s^{\top}\p{\theta_{1}^{\href{#eq-c4}{\textup{C4}}} - \theta_{0}^{\href{#eq-c4}{\textup{C4}}} } } \p{\bFcn^{0},\ldots,\bFcn^{h + \alpha + 2},\bFcn^{\star}} \nonumber \\
              &  = \quadform{W^{\href{#eq-c4}{\textup{C4}}}}{\p{\bx^{0},\bu^{0},\ldots,\bu^{h + \alpha + 2},\hat{\bu}^{\star},y^{\star}}} +  \p{w^{\href{#eq-c4}{\textup{C4}}}}^{\top} \p{\bFcn^{0},\ldots,\bFcn^{h + \alpha + 2},\bFcn^{\star}} 
          \end{aligned}

   where :eq:`eq:iteration_independent_lyapunov:w_c4_mat`
   and :eq:`eq:iteration_independent_lyapunov:w_c4`
   are used in the last equality. Therefore, using the
   :eq:`eq:iteration_independent_lyapunov:obj_c4`
   as the objective function in :ref:`(PEP) <eq:pep>`,
   :ref:`Theorem 5.1.1 (Performance estimation via SDP) <thm:pep_leq_zero>`
   gives that :ref:`(5.28) <eq:iteration_independent_lyapunov:condition>`,
   with :math:`\textup{cond} = \href{#eq-c4}{\textup{C4}}`, is a sufficient
   condition for :ref:`(C4) <eq:c4>`.

   Second, note that the proof is complete if we let the parameters
   :math:`\p{Q,q,S,s}` free, as
   in :ref:`(5.28f) <eq:iteration_independent_lyapunov:condition:q_mat>`,
   :ref:`(5.28g) <eq:iteration_independent_lyapunov:condition:q>`,
   :ref:`(5.28h) <eq:iteration_independent_lyapunov:condition:s_mat>`, and
   :ref:`(5.28i) <eq:iteration_independent_lyapunov:condition:s>`. :math:`\square`
