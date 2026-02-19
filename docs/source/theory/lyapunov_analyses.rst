Lyapunov analyses
=================

This page explains how AutoLyap uses the algorithm representation and
interpolation assumptions to cast the existence of a Lyapunov analysis
as the feasibility of a particular SDP. 
Moreover, this formulation is constructive, in the sense that any feasible 
solution directly yields an explicit Lyapunov function and associated 
convergence certificate.

For later convenience across pages
:doc:`5.1 </theory/performance_estimation_via_sdps>`,
:doc:`5.2 </theory/iteration_independent_analyses>`, and
:doc:`5.3 </theory/iteration_dependent_analyses>`, we introduce the
solution variables used in the Lyapunov analyses.

We are interested in analyses that may depend on a solution to the inclusion
problem :eq:`eq:the_problem_inclusion`. Without loss of generality and for
computational efficiency, we use

.. _eq:solution:

.. math::
   :label: eq:solution

   \begin{aligned}
       \p{y^{\star}, \hat{\bu}^{\star}, \bFcn^{\star}} \in
       \Bigset{
       \underbracket{\p{y,\hat{\bu}, \bFcn}}_{ \in \calH \times\calH^{m-1} \times \reals^{\NumFunc}  } \xmiddle|
       \begin{aligned}
           &\p{u_{i}}_{i\in\IndexFunc} \in \prod_{i\in\IndexFunc}\partial f_{i}\p{y}, \\
           &\p{u_{i}}_{i\in\IndexOp} \in \prod_{i\in\IndexOp}G_{i}\p{y}, \\
           &\sum_{i=1}^{m} u_{i} = 0, \\
           &\hat{\bu} = \p{u_{1},\ldots,u_{m-1}}, \\
           &\bFcn=\p{\bfcn_{i}\p{y}}_{i\in\IndexFunc}
       \end{aligned}
       },\end{aligned}

where :math:`\hat{\bu}^{\star}` is void when :math:`m=1`. In particular,
:math:`y^{\star}` in :eq:`eq:solution` is a solution to
:eq:`eq:the_problem_inclusion`.

The three subpages below are organized as follows. Page
:doc:`5.1 </theory/performance_estimation_via_sdps>` introduces a technical
SDP primitive. Pages :doc:`5.2 </theory/iteration_independent_analyses>` and
:doc:`5.3 </theory/iteration_dependent_analyses>` then build on that
primitive. The actual Lyapunov analyses, together with their corresponding
convergence conclusions, are presented in pages
:doc:`5.2 </theory/iteration_independent_analyses>` and
:doc:`5.3 </theory/iteration_dependent_analyses>`.

.. toctree::
   :maxdepth: 1

   performance_estimation_via_sdps
   iteration_independent_analyses
   iteration_dependent_analyses
