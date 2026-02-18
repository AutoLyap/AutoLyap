Notation and preliminaries
==========================

Let :math:`\naturals` denote the set of nonnegative integers,
:math:`\mathbb{N}` the set of positive integers, and :math:`\mathbb{Z}`
the set of integers. For
:math:`n,m \in \mathbb{Z}\cup\set{\pm\infty}`, we define

.. math::

   \llbracket n,m \rrbracket = \set{l \in \mathbb{Z} \xmiddle\vert n \leq l \leq m}.

We use :math:`\reals`, :math:`\reals_+`, and :math:`\reals_{++}` for
the real, nonnegative real, and positive real numbers, respectively. We
write :math:`\reals^n` for the set of all :math:`n`-tuples in
:math:`\reals`, :math:`\reals^{m\times n}` for real-valued matrices of
size :math:`m\times n`, and :math:`[M]_{i,j}` for the :math:`(i,j)`-th
element of :math:`M\in\reals^{m\times n}`. We denote by
:math:`\sym^{n}` the set of symmetric real-valued matrices of size
:math:`n\times n`, and by :math:`\sym_+^n\subseteq \sym^{n}` the set of
positive semidefinite ones.

The matrix :math:`0_{n\times m} \in \reals^{n\times m}` is the
:math:`n\times m` all-zero matrix, :math:`I_{n} \in \reals^{n\times n}`
is the identity matrix, :math:`e^n_{i}\in \reals^{n}` is the
:math:`i`\ th standard basis vector in :math:`\reals^{n}`, and
:math:`\mathbf{1}_{n}\in \reals^{n}` is the all-ones vector. All
vectors in :math:`\reals^n` are column vectors by convention.

Throughout this documentation, :math:`(\calH,\inner{\cdot}{\cdot})` will denote
a real Hilbert space. All norms :math:`\norm{\cdot}` are canonical norms
where the inner product will be clear from the context. We denote the
identity mapping :math:`x \mapsto x` on :math:`\calH` by :math:`\Id`.

.. container:: definition

   .. _def:func_defs:

   Let
   :math:`f\colon \calH \to \reals \cup \{\pm\infty\}`,
   :math:`L\in\reals_{+}` and
   :math:`\mu,\widetilde{\mu},\mu_{\textup{gd}}\in\reals_{++}`. The
   function :math:`f` is said to be

   (i)   *proper* if :math:`-\infty \notin f\p{\calH}` and
         :math:`\dom f \neq \emptyset`, where

         .. math::

            \dom f = \{x\in\calH \mid f(x) < + \infty\}

         is called the *effective domain* of :math:`f,`

   (ii)  *lower semicontinuous* if

         .. math::

            \liminf_{y\to x} f\p{y} \geq f\p{x}

         for each :math:`x\in\calH`,

   (iii) *convex* if

         .. math::

            f\p{\p{1 - \lambda} x + \lambda y} \leq \p{1 - \lambda} f\p{x} + \lambda f\p{y}

         for each :math:`x, y \in \calH` and
         :math:`0 \leq \lambda \leq 1`,

   (iv)  :math:`\mu`-strongly convex if :math:`f` is proper and
         :math:`f-(\mu/2)\norm{\cdot}^{2}` is convex,

   (v)   :math:`\widetilde{\mu}`-weakly convex if
         :math:`f+(\widetilde{\mu}/2)\norm{\cdot}^{2}` is convex,

   (vi)  :math:`L`-smooth if :math:`f` is Fréchet differentiable and
         the gradient :math:`\nabla f:\calH \to \calH` is
         :math:`L`-Lipschitz continuous, i.e.,

         .. math::

            \norm{\nabla f(x) - \nabla f(y)} \leq L \norm{x - y}

         for each :math:`x,y\in\calH`, and

   (vii) :math:`\mu_{\textup{gd}}`-gradient dominated if :math:`f` is
         Fréchet differentiable and

         .. _eq:pl:

         .. math::
            :label: eq:pl

            \begin{aligned}
                        f\p{x} - \inf_{y\in\calH}f\p{y} \leq \frac{1}{2\mu_{\textup{gd}}}\norm{\nabla f(x)}^2    
                    \end{aligned}

         for each :math:`x \in \calH`. Inequality :eq:`eq:pl`
         is sometimes called the *Polyak–Łojasiewicz* inequality or
         simply the *Łojasiewicz* inequality.

.. container:: definition

   .. _def:calF:

   Let :math:`-\infty < \mu \leq L \leq +\infty` such that
   :math:`L\geq 0`. We let :math:`\mathcal{F}_{\mu,L}\p{\mathcal{H}}`
   denote the class of all proper and lower semicontinuous functions
   :math:`f:\calH\rightarrow\reals\cup\{\pm\infty\}` such that

   (i)  :math:`(L/2)\norm{\cdot}^{2} - f` is convex and :math:`f` is
        Fréchet differentiable if :math:`L < +\infty`, and

   (ii) :math:`f-(\mu/2)\norm{\cdot}^{2}` is convex.

For example, :math:`\mathcal{F}_{-L,L}\p{\mathcal{H}}` is equal to the
class of :math:`L`-smooth functions with domain :math:`\calH`.

The *Fréchet subdifferential* of a function
:math:`f\colon \calH \to \reals \cup \{\pm\infty\}` is the set-valued
operator :math:`\partial f:\calH \rightrightarrows \calH` given by

.. math::

   \begin{aligned}
       \partial f \p{x} =
       \begin{cases}
           \Bigset{u\in\calH \xmiddle| \displaystyle{\liminf_{y\rightarrow x}} \frac{f\p{y}-f\p{x} - \inner{u}{y-x} }{\norm{y-x}}\geq 0 } & \text{if } \abs{f(x)} < + \infty, \\
           \emptyset & \text{otherwise}
       \end{cases}\end{aligned}

for each :math:`x\in\calH`.

(i)   If :math:`f` is Fréchet differentiable at a point
      :math:`x\in\calH`, then
      
      .. math::

         \partial f \p{x} = \set{\nabla f(x)}

      for each :math:`x\in\calH`.

(ii)  If :math:`f` is proper and convex, the Fréchet subdifferential
      becomes the *convex subdifferential*, i.e.,

      .. math::

         \partial f \p{x} = \set{u \in \calH \xmiddle\vert \forall y \in \calH,\, f(y) \geq f(x) + \inner{u}{y-x} }

      for each :math:`x \in \calH`.

(iii) If :math:`f` is proper and :math:`\widetilde{\mu}`-weakly convex
      for some :math:`\widetilde{\mu}\in\reals_{++}`, then

      .. math::

         \partial f(x) = \partial \p{f + (\widetilde{\mu}/2)\norm{\cdot}^{2}}\p{x} - \widetilde{\mu} x

      for each :math:`x\in\mathcal{H}`,
      where :math:`\partial \p{f + (\widetilde{\mu}/2)\norm{\cdot}^{2}}`
      reduces to the convex subdifferential.

.. container:: definition

   .. _def:prox:

   Suppose that
   :math:`f\in\mathcal{F}_{0,\infty}\p{\mathcal{H}}` and
   :math:`\gamma\in\reals_{++}`. Then the *proximal operator* of
   :math:`f` with *step size* :math:`\gamma`, denoted
   :math:`\prox_{\gamma f} : \calH \to \calH`, is defined as the
   single-valued operator given by

   .. math::

      \begin{aligned}
              \prox_{\gamma f}(x) = \argmin_{z\in\calH}\Bigp{f(z) + \frac{1}{2\gamma}\norm{x-z}^2} 
          \end{aligned}

   for each :math:`x\in\calH`.

Suppose that :math:`f\in\mathcal{F}_{0,\infty}\p{\mathcal{H}}` and
:math:`\gamma\in\reals_{++}`. If :math:`x,p\in\calH`, then

.. math::

   p = \prox_{\gamma f}(x) \Leftrightarrow \gamma^{-1}\p{x-p} \in \partial f (p).

Moreover, the *conjugate* of :math:`f`, denoted
:math:`f^{*}:\calH\to \reals \cup \{+\infty\}`, is the proper, lower
semicontinuous and convex function given by

.. math::

   f^{*}(u) = \sup_{x\in\calH}\p{\inner{u}{x} - f(x)}

for each :math:`u\in\calH`. If :math:`x,u\in\calH`, then

.. math::

   u \in \partial f(x) \Leftrightarrow x \in \partial f^{*}(u).

Let :math:`G:\calH\rightrightarrows\calH` be a set-valued operator. The
set of *zeros* of :math:`G` is denoted by

.. math::

   \zer G =\set{x\in\calH \xmiddle| 0 \in G\p{x} }

and the *graph* of :math:`G` is denoted by

.. math::

   \gra G = \set{(x,y)\in\calH\times\calH \xmiddle| y\in G(x)}.

.. container:: definition

   .. _def:set_op:

   Let :math:`G:\calH\rightrightarrows\calH` and
   :math:`\mu \in\reals_{++}`. The operator :math:`G` is said to be

   (i)   *monotone* if

         .. math::

            \inner{u - v}{x-y}\geq 0

         for each :math:`(x,u),(y,v)\in\gra G`,

   (ii)  *maximally monotone* if :math:`G` is monotone and there does
         not exist a monotone operator
         :math:`H:\calH\rightrightarrows\calH` such that
         :math:`\gra G \subsetneq \gra H`, and

   (iii) :math:`\mu`-*strongly monotone* if

         .. math::

            \inner{u - v}{x-y}\geq \mu \norm{x-y}^{2}

         for each :math:`(x,u),(y,v)\in\gra G`.

The *inverse* of a set-valued operator
:math:`G:\calH\rightrightarrows\calH`, denoted by
:math:`G^{-1}:\calH\rightrightarrows\calH`, is defined through its graph

.. math::

   \gra G^{-1} = \set{(y,x)\in\calH\times\calH \xmiddle| \p{x, y} \in \gra G}.

.. container:: definition

   .. _def:resolvent:

   Suppose that :math:`G:\calH \rightrightarrows\calH`
   is maximally monotone and :math:`\gamma \in \reals_{++}`. The
   *resolvent* of :math:`G` with *step size* :math:`\gamma`, denoted
   :math:`J_{\gamma G}:\calH \to \calH`, is defined by

   .. math::

      \begin{aligned}
              \p{\Id + \gamma G}^{-1}\p{x} = \set{J_{\gamma G} \p{x}}   
          \end{aligned}

   for each :math:`x\in \calH`, since :math:`\p{\Id + \gamma G}^{-1}` is
   singleton-valued in this case.

.. container:: definition

   .. _def:single_op:

   Let :math:`G:\calH \to \calH`,
   :math:`L\in\reals_{+}`, and :math:`\beta\in\reals_{++}`. The operator
   :math:`G` is said to be

   (i)  :math:`L`-*Lipschitz continuous* if

        .. math::

           \norm{G\p{x} - G\p{y} } \leq L \norm{x-y}

        for each :math:`x,y\in \calH`, and

   (ii) :math:`\beta`-*cocoercive* if

        .. math::

           \inner{G\p{x} - G\p{y} }{x-y} \geq \beta \norm{G\p{x} - G\p{y}}^{2}

        for each :math:`x,y\in \calH`.

We introduce the following conventions that enable us to treat
single-valued and singleton-valued operators interchangeably.

(i)  For notational convenience (at the expense of a slight abuse of
     notation), we will sometimes identify the operator
     
     .. math::

        G:\calH\to\calH

     with the set-valued mapping

     .. math::

        \calH \ni x \mapsto \set{G\p{x}} \subseteq\calH,

     which will be clear from context. For example, if
     :math:`x,y \in \calH`, the inclusion/equality pair

     .. math::

        y\in G\p{x}
        \quad\Longleftrightarrow\quad
        y = G\p{x}

     is used interchangeably.

(ii) Similarly, if

     .. math::

        G:\calH\rightrightarrows \calH,
        \qquad
        T: \calH \to \calH,
        \qquad
        G\p{x} = \set{T\p{x}}

     for each :math:`x\in\calH`, i.e., :math:`G` is a singleton-valued
     operator, we will sometimes identify :math:`G` with the
     corresponding single-valued operator :math:`T`.

Given any positive integer :math:`n`, we let the inner-product
:math:`\inner{\cdot}{\cdot}` on :math:`\calH^{n}` be given by

.. math::

   \begin{aligned}
       \inner{\bz^{1}}{\bz^{2}}=\sum_{j=1}^{n}\inner{z^{1}_{j}}{z^{2}_{j}}    \end{aligned}

for each :math:`\bz^{i}=\p{z^{i}_{1},\ldots,z^{i}_{n}}\in\calH^{n}` and
:math:`i\in\llbracket 1,2\rrbracket`. If
:math:`M\in \reals^{m\times n}`, we define the tensor product
:math:`M\kron \Id` to be the mapping
:math:`(M\kron \Id):\calH^{n}\rightarrow\calH^{m}` such that

.. math::

   \begin{aligned}
     (M\kron\Id)\bz = \Bigp{\sum_{j=1}^n[M]_{1,j}z_{j},\ldots,\sum_{j=1}^n[M]_{m,j}z_{j}}
   \end{aligned}

for each :math:`\bz=\p{z_{1},\ldots,z_{n}}\in\calH^n`. The adjoint and
composition formulas are

.. math::

   (M\kron\Id)^*=M^{\top}\kron\Id,
   \qquad
   (M\kron\Id)\circ(N\kron\Id)=(MN)\kron\Id

for each :math:`N\in\reals^{n\times l}`.

If we let :math:`M_{1}\in\reals^{m\times n_{1}}` and
:math:`M_{2}\in\reals^{m\times n_{2}}`, the relations above imply that

.. math::

   \inner{(M_1\kron\Id)\bz^{1}}{(M_2\kron\Id)\bz^{2}}
   =
   \inner{\bz^{1}}{\p{\p{M_1^{\top}M_2}\kron\Id}\bz^{2}}

for each :math:`\bz^{1}\in\calH^{n_1}` and :math:`\bz^{2}\in\calH^{n_2}`.
We define the mapping :math:`\mathcal{Q}:\sym^{n}\times \calH^{n}
\rightarrow \reals` by

.. math::

   \quadform{M}{\bz}=\inner{\bz}{(M\kron\Id)\bz}

for each :math:`M\in\sym^{n}` and :math:`\bz\in\calH^{n}`. Note that, if
:math:`M\in\sym^{n}`, :math:`N\in\reals^{n\times m}` and
:math:`\bz\in\calH^{m}`, then

.. math::

   \quadform{M}{(N\kron\Id)\bz}=\quadform{N^{\top}MN}{\bz}.

We define the *Gramian function*
:math:`\gramFunc:\mathcal{H}^n \to \sym^{n}_{+}` such that

.. math::

   [\gramFunc\p{\bz}]_{i,j} = \inner{z_{i}}{z_{j}}

for each :math:`\bz=\p{z_1,\ldots,z_n} \in \mathcal{H}^{n}`. If
:math:`M\in\sym^{n}` and :math:`\bz \in\mathcal{H}^{n}`, it holds that

.. math::

   \mathcal{Q}\p{M,\bz} = \trace\p{M \gramFunc\p{\bz}}.
