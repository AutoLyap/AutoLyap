# Internal problem-class modules

This page documents problem-class implementation helpers used by contributors.

Public condition class docs are in {doc}`/problem_class`.

## Module map

| Module | Responsibility |
| --- | --- |
| {py:mod}`autolyap.problemclass` | Package-level exports for the problem-class subsystem. |
| {py:mod}`autolyap.problemclass.indices` | Parses and stores interpolation index constraints. |
| {py:mod}`autolyap.problemclass.base` | Abstract interfaces for operator and function interpolation conditions. |
| {py:mod}`autolyap.problemclass.functions` | Concrete function interpolation conditions and parameter checks. |
| {py:mod}`autolyap.problemclass.operators` | Concrete operator interpolation conditions and parameter checks. |
| {py:mod}`autolyap.problemclass.inclusion_problem` | Container that aggregates interpolation conditions into one problem definition. |

## Module docstrings

```{eval-rst}
.. automodule:: autolyap.problemclass
   :no-members:
```

```{eval-rst}
.. automodule:: autolyap.problemclass.base
```

```{eval-rst}
.. automodule:: autolyap.problemclass.functions
```

```{eval-rst}
.. automodule:: autolyap.problemclass.operators
```

```{eval-rst}
.. automodule:: autolyap.problemclass.inclusion_problem
```

## Key internal inclusion-problem method

```{eval-rst}
.. automethod:: autolyap.problemclass.inclusion_problem.InclusionProblem._get_component_data
```

## Internal condition and index types

```{eval-rst}
.. automodule:: autolyap.problemclass.indices
```

```{eval-rst}
.. autoclass:: autolyap.problemclass.indices._InterpolationIndices
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: autolyap.problemclass.base._InterpolationCondition
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: autolyap.problemclass.base._OperatorInterpolationCondition
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: autolyap.problemclass.base._FunctionInterpolationCondition
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: autolyap.problemclass.functions._ParametrizedFunctionInterpolationCondition
   :members:
   :show-inheritance:
```

## Internal parameter-validation helpers

```{eval-rst}
.. autofunction:: autolyap.problemclass.functions._ensure_positive_finite
```

```{eval-rst}
.. autofunction:: autolyap.problemclass.functions._ensure_positive_mu_tilde
```

```{eval-rst}
.. autofunction:: autolyap.problemclass.operators._ensure_positive_finite
```

```{eval-rst}
.. autofunction:: autolyap.problemclass.operators._ensure_finite
```
