Algorithm representation
========================

We consider first-order algorithms that solve
:eq:`eq:the_problem_inclusion` that can be
represented as a discrete-time linear time‐varying system in state-space
form in feedback interconnection with the potentially nonlinear and
set-valued operators :math:`(\partial f_{i})_{i\in\IndexFunc}` and
:math:`(G_{i})_{i\in\IndexOp}` that define the problem.

Before presenting the algorithm representation, we introduce some
notation:

(i)   :math:`\NumFunc = \abs{\IndexFunc}` denotes the number of
      functional components in
      :eq:`eq:the_problem_inclusion`;

(ii)  :math:`\NumOp = \abs{\IndexOp}` denotes the number of operator
      components in
      :eq:`eq:the_problem_inclusion`;

(iii) :math:`\NumEval_{i}\in\mathbb{N}` denotes the number of
      evaluations of :math:`\partial f_{i}` and :math:`G_{i}` per
      iteration, for :math:`i\in\IndexFunc` and :math:`i\in\IndexOp`,
      respectively;

(iv)  :math:`\NumEvalFunc = \sum_{i\in\IndexFunc} \NumEval_{i}` denotes
      the total number of subdifferential evaluations per iteration;

(v)   :math:`\NumEvalOp = \sum_{i\in\IndexOp} \NumEval_{i}` denotes the
      total number of operator evaluations per iteration; and

(vi)  :math:`\NumEval = \NumEvalFunc + \NumEvalOp` denotes the combined
      total number of evaluations per iteration.

Since we consider algorithms that allow for multiple evaluation of
:math:`(\partial f_{i})_{i\in\IndexFunc}` and
:math:`(G_{i})_{i\in\IndexOp}` per iteration, we define
:math:`\bfcn_{i}:\calH^{\NumEval_{i}}\to\p{\reals\cup\{\pm\infty\}}^{\NumEval_{i}}`
such that

.. math::

   \begin{aligned}
       \Bigp{
       \begin{array}{@{}c@{}}
           \forall i\in\IndexFunc \\
           \forall \by_{i}=\p{y_{i,1},\ldots,y_{i,\NumEval_{i}}}\in \calH^{\NumEval_{i}}
       \end{array}
       }
       \quad
       \bfcn_{i}(\by_{i})&=\p{f_i\p{y_{i,1}},\ldots,f_{i}\p{y_{i,\NumEval_{i}}}},\end{aligned}

:math:`\bm{\partial}\bfcn_{i}:\calH^{\NumEval_{i}}\rightrightarrows\calH^{\NumEval_{i}}`
such that 

.. math::

   \begin{aligned}
       \Bigp{
       \begin{array}{@{}c@{}}
           \forall i\in\IndexFunc \\
           \forall \by_{i}=\p{y_{i,1},\ldots,y_{i,\NumEval_{i}}}\in \calH^{\NumEval_{i}}
       \end{array}
       }
       \quad
       \bm{\partial}\bfcn_{i}(\by_{i})&= \prod_{j=1}^{\NumEval_{i}} \partial f_{i}\p{y_{i,j}},\end{aligned}

and
:math:`\bm{G}_{i}: \calH^{\NumEval_{i}} \rightrightarrows \calH^{\NumEval_{i}}`
such that

.. math::

   \begin{aligned}
       \Bigp{
       \begin{array}{@{}c@{}}
           \forall i\in\IndexOp \\
           \forall \by_{i}=\p{y_{i,1},\ldots,y_{i,\NumEval_{i}}}\in \calH^{\NumEval_{i}}
       \end{array}
       }
       \quad
       \bm{G}_{i}\p{\by_{i}}=\prod_{j=1}^{\NumEval_{i}} G_{i}\p{y_{i,j}}.\end{aligned}

We are now ready to give the algorithm representation: Pick an initial
:math:`\bx_{0}\in\calH^{n}`, an iteration horizon
:math:`K\in\naturals\cup\set{\infty}`, and let

.. _eq:linear_system_with_nonlinearity:

.. math::
   :label: eq:linear_system_with_nonlinearity

       \p{\forall k \in\llbracket0,K\rrbracket} \quad
       \left[
       \begin{aligned}
           &\bx^{k+1} = \p{A_{k} \kron \Id} \bx^{k} + \p{B_{k} \kron \Id} \bu^{k}, \\
           &\by^{k} =  \p{C_{k} \kron \Id} \bx^{k} + \p{D_{k} \kron \Id} \bu^{k}, \\
           &\p{\bu^{k}_{i}}_{i\in\IndexFunc} \in \prod_{i\in\IndexFunc}\bm{\partial}\bfcn_{i}\p{\by_{i}^{k}}, \\
           &\p{\bu^{k}_{i}}_{i\in\IndexOp} \in \prod_{i\in\IndexOp}\bm{G}_{i}\p{\by_{i}^{k}}, \\
           &\bFcn^{k} =\p{\bfcn_{i}\p{\by_{i}^{k}} }_{i\in\IndexFunc},
       \end{aligned} 
       \right.

where

.. math::

   \begin{aligned}
      \bx^{k} &\in \calH^n, \\
      \bu^{k} &= \p{\bu^{k}_{1},\ldots,\bu^{k}_{m}} \in \prod_{i=1}^{m}\calH^{\NumEval_{i}}, \\
      \by^{k} &= \p{\by^{k}_{1},\ldots,\by^{k}_{m}} \in \prod_{i=1}^{m}\calH^{\NumEval_{i}}, \\
      \bFcn^{k} &\in \reals^{\NumEvalFunc}.
   \end{aligned}

are the algorithm variables, and

.. _eq:abcd:

.. math::
   :label: eq:abcd

   \begin{aligned}
       A_{k}&\in\reals^{n\times n},& B_{k}&\in\reals^{n\times {\NumEval}},& C_{k}&\in\reals^{{\NumEval}\times n},& D_{k}&\in\reals^{{\NumEval}\times {\NumEval}}\end{aligned}

are matrices containing the parameters of the algorithm at hand.

We close this page with
:ref:`Assumption 3.1 (Well-posedness) <ass:well-posedness>`. It is satisfied
by practical algorithms and ensures that the subsequent theoretical
results are well-posed.

.. container:: assumption

   .. _ass:well-posedness:

   **Assumption 3.1 (Well-posedness).**

   We assume that the parameter sequence

   .. math::

      \p{A_{k},B_{k},C_{k},D_{k}}_{k=0}^{K}

   is chosen such that, for each

   .. math::

      \bx_{0}\in\calH^{n}, \qquad
      \p{f_{i}}_{i\in\IndexFunc} \in \prod_{i\in\IndexFunc} \mathcal{F}_{i}, \qquad
      \p{G_i}_{i\in\IndexOp} \in \prod_{i\in\IndexOp} \mathcal{G}_i,

   there exists a sequence

   .. math::

      \p{\bx^{k},\bu^{k},\by^{k},\bFcn^{k}}_{k=0}^{K}

   satisfying :eq:`eq:linear_system_with_nonlinearity`.

In this context, the symbol :math:`\Pi` denotes Cartesian products.
For interface documentation in AutoLyap, see
:doc:`Algorithms </algorithms>`. In particular:

- *Base algorithms*: abstract interfaces for users to implement their
  own algorithms; see :doc:`Base algorithms </base_algorithms>`. For an
  example, see :doc:`The proximal gradient
  method </examples/define_your_own_algorithm/proximal_gradient_method>`.
- *Concrete algorithms*: built-in algorithm implementations; see
  :doc:`Concrete algorithms </concrete_algorithms>` for the full list.
