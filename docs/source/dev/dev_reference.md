# Developer reference

Use this page as a guide to AutoLyap internals.
It is intended for contributors implementing or refactoring core behavior.

Internal APIs can change without deprecation. For stable contracts, use the public docs first.

## Start here

1. Read {doc}`dev_internal_problemclass` to understand interpolation-condition data contracts.
2. Read {doc}`dev_internal_algorithms` to see how algorithm steps are lifted into matrix objects.
3. Read {doc}`dev_internal_core` to follow SDP assembly and solver execution.
4. Use {doc}`dev_internal_utils` as the shared helper reference across all layers.

## Documentation map

### Public docs (stable API)

| If you need to... | Go to... |
| --- | --- |
| Define interpolation-based inclusion problems | {doc}`/problem_class` |
| Select and configure algorithms | {doc}`/algorithms` |
| Run convergence analyses | {doc}`/lyapunov_analyses` |
| Configure solver backends | {doc}`/solver_backends` |

### Internal docs (implementation details)

| If you need to... | Go to... |
| --- | --- |
| Work on SDP assembly and solver option normalization | {doc}`dev_internal_core` |
| Modify the internal `Algorithm` contract or matrix/projection accessors | {doc}`dev_internal_algorithms` |
| Change interpolation-condition internals, indices, or inclusion-problem assembly | {doc}`dev_internal_problemclass` |
| Reuse validators, matrix helpers, or backend typing protocols | {doc}`dev_internal_utils` |

```{toctree}
:hidden:

dev_internal_core
dev_internal_algorithms
dev_internal_problemclass
dev_internal_utils
dev_external_reference_targets
```

## Architecture at a glance

| Layer | Core modules | What it owns |
| --- | --- | --- |
| Problem definitions | {py:mod}`autolyap.problemclass` | Interpolation conditions, index semantics, and component validation. |
| Algorithm lifting | {py:mod}`autolyap.algorithms` | Conversion of updates into lifted matrix objects (`X`, `Y`, `U`, `P`, `F`, `E`, `W`). |
| Analysis assembly | {py:mod}`autolyap.iteration_independent`, {py:mod}`autolyap.iteration_dependent` | Construction and solving of SDP certificates from problem and algorithm structure. |
| Backend normalization | {py:mod}`autolyap.solver_options` | One-pass normalization of solver options before backend-specific calls. |
| Shared utilities | {py:mod}`autolyap.utils` | Validation helpers, matrix helpers, and backend protocol types. |

## Common extension tasks

| Task | Primary entry point | Keep in mind |
| --- | --- | --- |
| Add a new algorithm | {py:class}`autolyap.algorithms.algorithm.Algorithm` | Implement required matrix/projection accessors consistently. |
| Add a new interpolation condition | {py:meth}`autolyap.problemclass.base._InterpolationCondition.get_data` | Follow the existing {py:mod}`autolyap.problemclass` condition hierarchy. |
| Add or modify analysis workflows | {py:mod}`autolyap.iteration_independent`, {py:mod}`autolyap.iteration_dependent` | Preserve solver-backend parity across supported backends. |
| Add validation or backend typing logic | {py:mod}`autolyap.utils.validation`, {py:mod}`autolyap.utils.backend_types` | Keep shared checks centralized; avoid per-module duplication. |

## Invariants and common pitfalls

- Interpolation indices exposed to users are 1-based and strictly validated.
- Keep matrix dimensions aligned across `Y`, `P`, and interpolation blocks; shape drift usually appears as SDP assembly failures.
- Normalize solver options exactly once via {py:func}`autolyap.solver_options._normalize_solver_options` before backend-specific solve calls.
- Keep numeric checks in shared validation helpers instead of duplicating them inside algorithm or condition modules.
