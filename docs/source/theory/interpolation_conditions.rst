Interpolation conditions
========================

This page is technical. It introduces a key assumption used to
formulate the Lyapunov search as a semidefinite program (SDP).

.. container:: assumption

   .. _ass:interpolation:

   .. _ass:interplotation:

   **Assumption 4.1 (Interpolation conditions).**

   Consider
   :eq:`eq:linear_system_with_nonlinearity`
   and, for notational convenience, let

   .. container:: allowdisplaybreaks

      .. math::

         \begin{aligned}
             \left(
                 \begin{array}{@{}c@{}}
                     \forall i \in \IndexFunc\cup\IndexOp \\
                     \forall k \in \llbracket0,K\rrbracket
                 \end{array}
             \right) 
             &\quad \bu^{k}_{i} = \p{u^{k}_{i,1}, \ldots, u^{k}_{i,\NumEval_{i}}}\in \calH^{\NumEval_i},  \\
             \left(
                 \begin{array}{@{}c@{}}
                     \forall i \in \IndexFunc\cup\IndexOp \\
                     \forall k \in \llbracket0,K\rrbracket
                 \end{array}
             \right) 
             &\quad \by^{k}_{i} = \p{y^{k}_{i,1}, \ldots, y^{k}_{i,\NumEval_{i}}}\in \calH^{\NumEval_i}, \\
             \left(
                 \begin{array}{@{}c@{}}
                     \forall i \in \IndexFunc \\
                     \forall k \in \llbracket0,K\rrbracket
                 \end{array}
             \right) 
             &\quad \bfcn_{i}\p{\by_{i}^{k}} = \p{F^{k}_{i,1}, \ldots, F^{k}_{i,\NumEval_{i}}} \in \reals^{\NumEval_{i}}\end{aligned}

      and

      .. math::

         \begin{aligned}
             \bu^{\star} &= \p{u^{\star}_{1},\ldots,u^{\star}_{m}} = \p{u^{\star}_{1,\star},\ldots,u^{\star}_{m,\star}} \in \calH^{m} \quad \text{ such that }\quad \sum_{i=1}^{m} u^{\star}_{i} = 0, \\
             y^{\star} &= y^{\star}_{1,\star}=\cdots=y^{\star}_{m,\star} \in \calH, \\
             \bFcn^{\star} &=\p{F^{\star}_{i,\star}}_{i\in\IndexFunc}\in\reals^{\NumFunc }.\end{aligned}

   (a) For each :math:`i\in\IndexFunc`, suppose that there exist finite
       and disjoint sets :math:`\mathcal{O}_{i}^{\textup{func-ineq}}`
       and :math:`\mathcal{O}_{i}^{\textup{func-eq}}`, vectors and
       matrices

       .. math::

          \begin{aligned}
                  \p{\forall o \in \mathcal{O}_{i}^{\textup{func-ineq}}} 
                  &\quad \p{ a_{\p{i,o}}^{\textup{func-ineq}}, M_{\p{i,o}}^{\textup{func-ineq}}} \in  \reals^{n_{i,o}} \times \sym^{2n_{i,o}}, \\
                  \p{\forall o \in \mathcal{O}_{i}^{\textup{func-eq}}} 
                  &\quad \p{a_{\p{i,o}}^{\textup{func-eq}}, M_{\p{i,o}}^{\textup{func-eq}}} \in  \reals^{n_{i,o}} \times \sym^{2n_{i,o}}, 
              \end{aligned}

       and, depending on
       :math:`\PEPMinIter,\PEPMaxIter\in\llbracket0,K\rrbracket` such
       that :math:`\PEPMinIter\leq\PEPMaxIter`, index sets

       .. math::

          \begin{gathered}
                  \p{\forall o \in \mathcal{O}_{i}^{\textup{func-ineq}}\cup \mathcal{O}_{i}^{\textup{func-eq}} } \\
                  \mathcal{J}_{i,o}^{\PEPMinIter,\PEPMaxIter}\subseteq \p{ \p{\llbracket 1,\NumEval_{i} \rrbracket \times \llbracket\PEPMinIter,\PEPMaxIter\rrbracket } \cup \set{ \p{\star,\star} }   }^{n_{i,o}},
              \end{gathered}

       such that *(i)* implies *(ii)* below:

       (i)  There exists a function :math:`f_{i}\in\mathcal{F}_{i}` such
            that

            .. math::

               \begin{aligned}
                           \p{\forall (j,k)\in\p{\llbracket 1,\NumEval_{i} \rrbracket \times \llbracket\PEPMinIter,\PEPMaxIter\rrbracket } \cup \set{ \p{\star,\star} } } \quad
                           \left[
                           \begin{aligned}
                               f_{i}\p{y_{i,j}^{l}} = F_{i,j}^{k}, \\    
                               u_{i,j}^{k} \in \partial f_{i} \p{y_{i,j}^{k} }.
                           \end{aligned}
                           \right.
                       \end{aligned}

       (ii) It holds that

            .. math::

               \begin{aligned}
                   \left(
                       \begin{array}{@{}c@{}}
                           \forall o \in \mathcal{O}_{i}^{\textup{func-ineq}} \\
                           \forall \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{\PEPMinIter,\PEPMaxIter}
                       \end{array}
                   \right)
                   \quad
                   \p{a_{\p{i,o}}^{\textup{func-ineq}}}^{\top} F
                   + \quadform{M_{\p{i,o}}^{\textup{func-ineq}}}{z}
                   \leq 0,
                   \\
                   \left(
                       \begin{array}{@{}c@{}}
                           \forall o \in \mathcal{O}_{i}^{\textup{func-eq}} \\
                           \forall \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{\PEPMinIter,\PEPMaxIter}
                       \end{array}
                   \right)
                   \quad
                   \p{a_{\p{i,o}}^{\textup{func-eq}}}^{\top} F
                   + \quadform{M_{\p{i,o}}^{\textup{func-eq}}}{z}
                   = 0.
               \end{aligned}

            where

            .. math::

               z = \p{y_{i,j_{1}}^{k_{1}},\ldots,y_{i,j_{n_{i,o}}}^{k_{n_{i,o}}},u_{i,j_{1}}^{k_{1}},\ldots,u_{i,j_{n_{i,o}}}^{k_{n_{i,o}}}}
               \quad \text{and} \quad
               F = \begin{bmatrix}
                   F_{i,j_{1}}^{k_{1}} \\
                   \vdots \\
                   F_{i,j_{n_{i,o}}}^{k_{n_{i,o}}}
               \end{bmatrix}.

       Moreover, if the converse holds, i.e., *(ii)* implies *(i)*, then
       we say that the function class :math:`\mathcal{F}_i` has a *tight
       interpolation condition*.

   (b) Similarly, for each :math:`i\in\IndexOp`, suppose that there
       exists a finite set :math:`\mathcal{O}_{i}^{\textup{op}}`,
       matrices

       .. math::

          \begin{aligned}
                  \p{\forall o \in \mathcal{O}_{i}^{\textup{op}}} \quad M_{\p{i,o}}^{\textup{op}} \in \sym^{2 n_{i,o}},
              \end{aligned}

       and, depending on
       :math:`\PEPMinIter,\PEPMaxIter\in\llbracket0,K\rrbracket` such
       that :math:`\PEPMinIter\leq\PEPMaxIter`, index sets

       .. math::

          \begin{aligned}
                  \p{\forall o \in \mathcal{O}_{i}^{\textup{op}}} 
                  &\quad \mathcal{J}_{i,o}^{\PEPMinIter,\PEPMaxIter}\subseteq \p{ \p{\llbracket 1,\NumEval_{i} \rrbracket \times \llbracket\PEPMinIter,\PEPMaxIter\rrbracket } \cup \set{ \p{\star,\star} }   }^{n_{i,o}}
              \end{aligned}

       such that *(i)* implies *(ii)* below:

       (i)  There exists an operator :math:`G_{i}\in\mathcal{G}_{i}`
            such that

            .. math::

               \begin{aligned}
                           \p{\forall (j,k)\in\p{\llbracket 1,\NumEval_{i} \rrbracket \times \llbracket\PEPMinIter,\PEPMaxIter\rrbracket } \cup \set{ \p{\star,\star} } } \quad u_{i,j}^{k} \in G_{i} \p{y_{i,j}^{k} }.
                       \end{aligned}

       (ii) It holds that

            .. math::

               \left(
                   \begin{array}{@{}c@{}}
                       \forall o \in \mathcal{O}_{i}^{\textup{op}} \\
                       \forall \p{ \p{j_{1},k_{1}}, \ldots, \p{j_{n_{i,o}},k_{n_{i,o}} }}\in\mathcal{J}_{i,o}^{\PEPMinIter,\PEPMaxIter}
                   \end{array}
               \right)
               \quad
               \quadform{M_{\p{i,o}}^{\textup{op}}}{z} \leq 0,

            where

            .. math::

               z = \p{y_{i,j_{1}}^{k_{1}},\ldots,y_{i,j_{n_{i,o}}}^{k_{n_{i,o}}},u_{i,j_{1}}^{k_{1}},\ldots,u_{i,j_{n_{i,o}}}^{k_{n_{i,o}}}}.

       Moreover, if the converse holds, i.e., *(ii)* implies *(i)*, then
       we say that the operator class :math:`\mathcal{G}_i` has a *tight
       interpolation condition*.

For concrete interpolation-condition examples in API docstrings, see
:doc:`Function classes </function_classes>` and
:doc:`Operator classes </operator_classes>`. In the setting of
:ref:`Assumption 4.1 (Interpolation conditions) <ass:interpolation>`,
the following shipped classes have tight interpolation
conditions:

- Function classes:
  
  - :py:class:`Convex <autolyap.problemclass.Convex>`
  - :py:class:`StronglyConvex <autolyap.problemclass.StronglyConvex>`
  - :py:class:`WeaklyConvex <autolyap.problemclass.WeaklyConvex>`
  - :py:class:`Smooth <autolyap.problemclass.Smooth>`
  - :py:class:`SmoothConvex <autolyap.problemclass.SmoothConvex>`
  - :py:class:`SmoothStronglyConvex <autolyap.problemclass.SmoothStronglyConvex>`
  - :py:class:`SmoothWeaklyConvex <autolyap.problemclass.SmoothWeaklyConvex>`
  - :py:class:`IndicatorFunctionOfClosedConvexSet <autolyap.problemclass.IndicatorFunctionOfClosedConvexSet>`
  - :py:class:`SupportFunctionOfClosedConvexSet <autolyap.problemclass.SupportFunctionOfClosedConvexSet>`
- Operator classes:
  
  - :py:class:`MaximallyMonotone <autolyap.problemclass.MaximallyMonotone>`
  - :py:class:`StronglyMonotone <autolyap.problemclass.StronglyMonotone>`
  - :py:class:`LipschitzOperator <autolyap.problemclass.LipschitzOperator>`
  - :py:class:`Cocoercive <autolyap.problemclass.Cocoercive>`

Intersections of function classes are supported, and intersections of
operator classes are also supported. However, after taking such
intersections, tightness of the resulting class is not guaranteed in
general.
