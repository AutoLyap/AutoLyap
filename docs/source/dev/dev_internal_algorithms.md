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

## Module map

| Class | Implementation module |
| --- | --- |
| {py:class}`~autolyap.algorithms.accelerated_proximal_point.AcceleratedProximalPoint` | `autolyap.algorithms.accelerated_proximal_point` |
| {py:class}`~autolyap.algorithms.chambolle_pock.ChambollePock` | `autolyap.algorithms.chambolle_pock` |
| {py:class}`~autolyap.algorithms.davis_yin.DavisYin` | `autolyap.algorithms.davis_yin` |
| {py:class}`~autolyap.algorithms.douglas_rachford.DouglasRachford` | `autolyap.algorithms.douglas_rachford` |
| {py:class}`~autolyap.algorithms.extragradient.Extragradient` | `autolyap.algorithms.extragradient` |
| {py:class}`~autolyap.algorithms.forward.ForwardMethod` | `autolyap.algorithms.forward` |
| {py:class}`~autolyap.algorithms.gradient.GradientMethod` | `autolyap.algorithms.gradient` |
| {py:class}`~autolyap.algorithms.gradient_with_Nesterov_like_momentum.GradientNesterovMomentum` | `autolyap.algorithms.gradient_with_Nesterov_like_momentum` |
| {py:class}`~autolyap.algorithms.heavy_ball.HeavyBallMethod` | `autolyap.algorithms.heavy_ball` |
| {py:class}`~autolyap.algorithms.information_theoretic_exact_method.ITEM` | `autolyap.algorithms.information_theoretic_exact_method` |
| {py:class}`~autolyap.algorithms.malitsky_tam_frb.MalitskyTamFRB` | `autolyap.algorithms.malitsky_tam_frb` |
| {py:class}`~autolyap.algorithms.nesterov_constant.NesterovConstant` | `autolyap.algorithms.nesterov_constant` |
| {py:class}`~autolyap.algorithms.nesterov_fast_gradient_method.NesterovFastGradientMethod` | `autolyap.algorithms.nesterov_fast_gradient_method` |
| {py:class}`~autolyap.algorithms.optimized_gradient_method.OptimizedGradientMethod` | `autolyap.algorithms.optimized_gradient_method` |
| {py:class}`~autolyap.algorithms.proximal_point.ProximalPoint` | `autolyap.algorithms.proximal_point` |
| {py:class}`~autolyap.algorithms.triple_momentum.TripleMomentum` | `autolyap.algorithms.triple_momentum` |
| {py:class}`~autolyap.algorithms.tseng_fbf.TsengFBF` | `autolyap.algorithms.tseng_fbf` |
