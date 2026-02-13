# Internal algorithm modules

This page is for contributors implementing or modifying algorithm internals.

Public class-level API docs are in {doc}`/algorithms`.

## Base contract

Use the base class contract when introducing a new algorithm module.

```{eval-rst}
.. autoclass:: autolyap.algorithms.algorithm.Algorithm
   :members:
   :private-members:
   :special-members: __init__
   :show-inheritance:
   :no-index:
```

## Private matrix/projection accessors

```{eval-rst}
.. automethod:: autolyap.algorithms.algorithm.Algorithm._get_AsBsCsDs
```

```{eval-rst}
.. automethod:: autolyap.algorithms.algorithm.Algorithm._get_Us
```

```{eval-rst}
.. automethod:: autolyap.algorithms.algorithm.Algorithm._get_Ys
```

```{eval-rst}
.. automethod:: autolyap.algorithms.algorithm.Algorithm._get_Xs
```

```{eval-rst}
.. automethod:: autolyap.algorithms.algorithm.Algorithm._get_Ps
```

```{eval-rst}
.. automethod:: autolyap.algorithms.algorithm.Algorithm._get_Fs
```

```{eval-rst}
.. automethod:: autolyap.algorithms.algorithm.Algorithm._compute_E
```

```{eval-rst}
.. automethod:: autolyap.algorithms.algorithm.Algorithm._compute_W
```

```{eval-rst}
.. automethod:: autolyap.algorithms.algorithm.Algorithm._compute_F_aggregated
```

## Module inventory

1. {py:mod}`autolyap.algorithms.accelerated_proximal_point` ({py:class}`autolyap.algorithms.accelerated_proximal_point.AcceleratedProximalPoint`)
2. {py:mod}`autolyap.algorithms.chambolle_pock` ({py:class}`autolyap.algorithms.chambolle_pock.ChambollePock`)
3. {py:mod}`autolyap.algorithms.davis_yin` ({py:class}`autolyap.algorithms.davis_yin.DavisYin`)
4. {py:mod}`autolyap.algorithms.douglas_rachford` ({py:class}`autolyap.algorithms.douglas_rachford.DouglasRachford`)
5. {py:mod}`autolyap.algorithms.extragradient` ({py:class}`autolyap.algorithms.extragradient.Extragradient`)
6. {py:mod}`autolyap.algorithms.gradient` ({py:class}`autolyap.algorithms.gradient.GradientMethod`)
7. {py:mod}`autolyap.algorithms.gradient_with_Nesterov_like_momentum` ({py:class}`autolyap.algorithms.gradient_with_Nesterov_like_momentum.GradientNesterovMomentum`)
8. {py:mod}`autolyap.algorithms.heavy_ball` ({py:class}`autolyap.algorithms.heavy_ball.HeavyBallMethod`)
9. {py:mod}`autolyap.algorithms.information_theoretic_exact_method` ({py:class}`autolyap.algorithms.information_theoretic_exact_method.ITEM`)
10. {py:mod}`autolyap.algorithms.nesterov_constant` ({py:class}`autolyap.algorithms.nesterov_constant.NesterovConstant`)
11. {py:mod}`autolyap.algorithms.nesterov_fast_gradient_method` ({py:class}`autolyap.algorithms.nesterov_fast_gradient_method.NesterovFastGradientMethod`)
12. {py:mod}`autolyap.algorithms.optimized_gradient_method` ({py:class}`autolyap.algorithms.optimized_gradient_method.OptimizedGradientMethod`)
13. {py:mod}`autolyap.algorithms.proximal_point` ({py:class}`autolyap.algorithms.proximal_point.ProximalPoint`)
14. {py:mod}`autolyap.algorithms.triple_momentum` ({py:class}`autolyap.algorithms.triple_momentum.TripleMomentum`)
15. {py:mod}`autolyap.algorithms.tseng_fbf` ({py:class}`autolyap.algorithms.tseng_fbf.TsengFBF`)

## Module targets

```{eval-rst}
.. automodule:: autolyap.algorithms
   :no-members:
```

```{eval-rst}
.. automodule:: autolyap.algorithms.accelerated_proximal_point
   :no-members:
```

```{eval-rst}
.. automodule:: autolyap.algorithms.chambolle_pock
   :no-members:
```

```{eval-rst}
.. automodule:: autolyap.algorithms.davis_yin
   :no-members:
```

```{eval-rst}
.. automodule:: autolyap.algorithms.douglas_rachford
   :no-members:
```

```{eval-rst}
.. automodule:: autolyap.algorithms.extragradient
   :no-members:
```

```{eval-rst}
.. automodule:: autolyap.algorithms.forward
   :no-members:
```

```{eval-rst}
.. automodule:: autolyap.algorithms.gradient
   :no-members:
```

```{eval-rst}
.. automodule:: autolyap.algorithms.gradient_with_Nesterov_like_momentum
   :no-members:
```

```{eval-rst}
.. automodule:: autolyap.algorithms.heavy_ball
   :no-members:
```

```{eval-rst}
.. automodule:: autolyap.algorithms.information_theoretic_exact_method
   :no-members:
```

```{eval-rst}
.. automodule:: autolyap.algorithms.malitsky_tam_frb
   :no-members:
```

```{eval-rst}
.. automodule:: autolyap.algorithms.nesterov_constant
   :no-members:
```

```{eval-rst}
.. automodule:: autolyap.algorithms.nesterov_fast_gradient_method
   :no-members:
```

```{eval-rst}
.. automodule:: autolyap.algorithms.optimized_gradient_method
   :no-members:
```

```{eval-rst}
.. automodule:: autolyap.algorithms.proximal_point
   :no-members:
```

```{eval-rst}
.. automodule:: autolyap.algorithms.triple_momentum
   :no-members:
```

```{eval-rst}
.. automodule:: autolyap.algorithms.tseng_fbf
   :no-members:
```
