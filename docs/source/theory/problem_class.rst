Problem classes
===============

To cover both structured optimization and inclusion problems, we
introduce two disjoint index sets
:math:`\IndexFunc, \IndexOp \subseteq \llbracket1,m\rrbracket`, where
:math:`m\in\mathbb{N}`, such that
:math:`\IndexFunc\cup \IndexOp = \llbracket1,m\rrbracket`, and consider
inclusion problems of the form

.. _eq:the_problem_inclusion:

.. math::
   :label: eq:the_problem_inclusion

   \begin{aligned}
       \text{find}\  y\in\calH\ \text{ such that }\ 0\in\sum_{i\in\IndexFunc} \partial f_{i}\p{y} + \sum_{i\in\IndexOp} G_{i}\p{y},\end{aligned}

where the functions :math:`f_i:\calH \to \reals \cup \set{\pm \infty}`
and operators :math:`G_i : \calH \rightrightarrows \calH` are chosen
from some user-specified function class :math:`\mathcal{F}_i` and
operator class :math:`\mathcal{G}_{i}`, respectively, i.e.,

.. math::

   \begin{aligned}
       \p{\forall i \in \IndexFunc}& \quad f_{i} \in \mathcal{F}_{i} \subseteq \set{ f: \calH \to \reals\cup \set{\pm \infty}}, \\
       \p{\forall i \in \IndexOp}& \quad G_{i} \in \mathcal{G}_{i} \subseteq \set{ G: \calH \rightrightarrows \calH }.\end{aligned}

For example, if :math:`\IndexOp=\emptyset`, then
:eq:`eq:the_problem_inclusion` is a
first-order optimality condition for minimizing

.. math::

   \sum_{i\in\IndexFunc} f_{i}.

Moreover,
:eq:`eq:the_problem_inclusion` provides a
formalism that covers monotone inclusion problems
,
certain equilibrium problems,
so-called (mixed) variational inequalities,
and beyond.

The API for these objects is documented in
:doc:`API reference </api_reference>`. In particular,
:doc:`Problem classes </problem_class>` documents the
:py:class:`InclusionProblem <autolyap.problemclass.InclusionProblem>`
interface for formulating
:eq:`eq:the_problem_inclusion`, while
:doc:`Function classes </function_classes>` and
:doc:`Operator classes </operator_classes>` document the shipped
function- and operator-class interfaces corresponding to the
classes :math:`(\mathcal{F}_i)_{i\in\IndexFunc}` and
:math:`(\mathcal{G}_i)_{i\in\IndexOp}`, respectively, in
:eq:`eq:the_problem_inclusion`. The API also supports intersections of
function classes and intersections of operator classes (but not mixed
function-operator intersections).
