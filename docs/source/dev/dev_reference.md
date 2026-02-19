# Developer reference

This page is for contributors working on AutoLyap internals.

Internal implementation APIs may change without deprecation; use the user-facing API reference for stability guarantees.

## Public API map

Use these pages for stable API contracts and normal package usage.

1. {doc}`/problem_class`
2. {doc}`/algorithms`
3. {doc}`/lyapunov_analyses`
4. {doc}`/solver_backends`

## Internal API map

These pages document implementation details that are useful when developing AutoLyap.

```{toctree}
:maxdepth: 1

dev_internal_core
dev_internal_algorithms
dev_internal_problemclass
dev_internal_utils
```

```{toctree}
:hidden:

dev_external_reference_targets
```

## Architecture map

1. {py:mod}`autolyap.problemclass` defines interpolation conditions and index semantics.
2. {py:mod}`autolyap.algorithms` maps method updates into lifted matrices and projection operators.
3. {py:mod}`autolyap.iteration_independent` and {py:mod}`autolyap.iteration_dependent` build SDP certificates on top of problem and algorithm structure.
4. {py:mod}`autolyap.solver_options` normalizes backend configuration for MOSEK Fusion and CVXPY execution paths.
5. {py:mod}`autolyap.utils` provides shared validation, matrix-construction, and backend structural-typing helpers used across modules.

## Extension points

1. Add a new algorithm by subclassing {py:class}`autolyap.algorithms.algorithm.Algorithm` and implementing its required matrix/projection accessors.
2. Add a new interpolation condition by implementing {py:meth}`autolyap.problemclass.base._InterpolationCondition.get_data` in the relevant {py:mod}`autolyap.problemclass` hierarchy.
3. Add or modify analysis workflows in {py:mod}`autolyap.iteration_independent` or {py:mod}`autolyap.iteration_dependent` while preserving solver-backend parity.
4. Keep parameter validation centralized by reusing helpers in {py:mod}`autolyap.utils.validation`, and keep backend protocol types centralized in {py:mod}`autolyap.utils.backend_types`.

## Developer invariants and pitfalls

1. User-facing interpolation indices are 1-based and validated strictly.
2. Matrix dimensions must remain consistent across `Y`, `P`, and interpolation blocks; shape drift usually surfaces as SDP assembly failures.
3. Solver options should be normalized once via {py:func}`autolyap.solver_options._normalize_solver_options` before backend-specific solve code.
4. Shared numeric checks should stay in validation helpers rather than being duplicated per algorithm or condition class.
