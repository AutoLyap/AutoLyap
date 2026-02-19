# Internal problem-class modules

This page documents problem-class implementation helpers used by contributors.

Public condition class docs are in {doc}`/problem_class`.

## Module inventory

1. {py:mod}`autolyap.problemclass`: package-level exports for the problem-class subsystem.
2. {py:mod}`autolyap.problemclass.indices`: parses and stores interpolation index constraints.
3. {py:mod}`autolyap.problemclass.base`: abstract interfaces for operator and function interpolation conditions.
4. {py:mod}`autolyap.problemclass.functions`: concrete function interpolation conditions and parameter checks.
5. {py:mod}`autolyap.problemclass.operators`: concrete operator interpolation conditions and parameter checks.
6. {py:mod}`autolyap.problemclass.inclusion_problem`: container that aggregates interpolation conditions into one problem definition.

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
