Lyapunov analyses
=================

This page explains how AutoLyap uses the algorithm representation and
interpolation assumptions to cast the existence of a Lyapunov analysis
as the feasibility of a particular SDP. 
Moreover, this formulation is constructive, in the sense that any feasible 
solution directly yields an explicit Lyapunov function and associated 
convergence certificate.

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
