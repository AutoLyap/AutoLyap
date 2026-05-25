<!-- SPDX-FileCopyrightText: 2025-2026 AutoLyap contributors -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Iteration-independent analysis

```{eval-rst}
.. autoclass:: autolyap.IterationIndependent
   :members: search_lyapunov
   :show-inheritance:
```

(iter-independent-linear-helpers)=
## Linear-convergence helpers

```{eval-rst}
.. automethod:: autolyap.IterationIndependent.LinearConvergence.get_parameters_distance_to_solution

.. automethod:: autolyap.IterationIndependent.LinearConvergence.get_parameters_function_value_suboptimality

.. automethod:: autolyap.IterationIndependent.LinearConvergence.bisection_search_rho
```

(iter-independent-sublinear-helpers)=
## Sublinear-convergence helpers

```{eval-rst}
.. automethod:: autolyap.IterationIndependent.SublinearConvergence.get_parameters_fixed_point_residual

.. automethod:: autolyap.IterationIndependent.SublinearConvergence.get_parameters_duality_gap

.. automethod:: autolyap.IterationIndependent.SublinearConvergence.get_parameters_function_value_suboptimality

.. automethod:: autolyap.IterationIndependent.SublinearConvergence.get_parameters_optimality_measure
```
