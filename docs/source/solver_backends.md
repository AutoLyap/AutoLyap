# Solver backends

```{eval-rst}
.. autoclass:: autolyap.SolverOptions
   :members:
   :show-inheritance:
```

## Installation

Install AutoLyap core:

```bash
python -m pip install autolyap
```

Install optional solver extras from PyPI:

```bash
# MOSEK Fusion backend
python -m pip install "autolyap[mosek]"

# CVXPY + SDPA (regular precision)
python -m pip install "autolyap[sdpa]"

# CVXPY + SDPA multiprecision
python -m pip install "autolyap[sdpa_multiprecision]"
```
