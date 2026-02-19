# Internal analysis and solver modules

This page focuses on internals that support analysis assembly and solver execution.

Public API docs for these topics are in {doc}`/lyapunov_analyses` and {doc}`/solver_backends`.

## Module inventory

1. {py:mod}`autolyap.iteration_independent`: iteration-independent SDP assembly and feasibility checks.
2. {py:mod}`autolyap.iteration_dependent`: finite-horizon SDP assembly and chained-certificate checks.
3. {py:mod}`autolyap.solver_options`: backend normalization and CVXPY solver-argument preparation.

```{eval-rst}
.. automodule:: autolyap.iteration_independent
   :no-members:
```

```{eval-rst}
.. automodule:: autolyap.iteration_dependent
   :no-members:
```

```{eval-rst}
.. automodule:: autolyap.solver_options
   :no-members:
```

## Iteration-independent helper namespaces

```{eval-rst}
.. autoclass:: autolyap.iteration_independent._LinearConvergence
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: autolyap.iteration_independent._SublinearConvergence
   :members:
   :show-inheritance:
```

## Solver option helper functions

```{eval-rst}
.. autofunction:: autolyap.solver_options._normalize_solver_options
```

```{eval-rst}
.. autofunction:: autolyap.solver_options._get_cvxpy_solve_kwargs
```

```{eval-rst}
.. autofunction:: autolyap.solver_options._get_cvxpy_accepted_statuses
```

```{eval-rst}
.. autofunction:: autolyap.solver_options._normalize_mapping
```

```{eval-rst}
.. autofunction:: autolyap.solver_options._normalize_named_mapping
```
