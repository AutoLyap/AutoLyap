import numpy as np
import pytest

from autolyap.algorithms import (
    AcceleratedProximalPoint,
    ChambollePock,
    DavisYin,
    DouglasRachford,
    Extragradient,
    ForwardMethod,
    GradientMethod,
    GradientNesterovMomentum,
    HeavyBallMethod,
    ITEM,
    MalitskyTamFRB,
    NesterovConstant,
    NesterovFastGradientMethod,
    OptimizedGradientMethod,
    ProximalPoint,
    TripleMomentum,
    TsengFBF,
)


POSITIVE_REAL_SETTERS = [
    (lambda: ForwardMethod(gamma=0.1), "set_gamma"),
    (lambda: GradientMethod(gamma=0.1), "set_gamma"),
    (lambda: ProximalPoint(gamma=0.1), "set_gamma"),
    (lambda: HeavyBallMethod(gamma=0.1, delta=0.0), "set_gamma"),
    (lambda: GradientNesterovMomentum(gamma=0.1, delta=0.0), "set_gamma"),
    (lambda: NesterovFastGradientMethod(gamma=0.1), "set_gamma"),
    (lambda: AcceleratedProximalPoint(gamma=0.1, type="operator"), "set_gamma"),
    (lambda: Extragradient(gamma=0.1, delta=0.2, type="unconstrained"), "set_gamma"),
    (lambda: Extragradient(gamma=0.1, delta=0.2, type="unconstrained"), "set_delta"),
    (lambda: TsengFBF(gamma=0.1, theta=0.0), "set_gamma"),
    (lambda: DavisYin(gamma=0.1, lambda_value=0.0), "set_gamma"),
    (lambda: DouglasRachford(gamma=0.1, lambda_value=0.0, type="operator"), "set_gamma"),
    (lambda: MalitskyTamFRB(gamma=0.1), "set_gamma"),
    (lambda: NesterovConstant(mu=1.0, L=4.0), "set_mu"),
    (lambda: NesterovConstant(mu=1.0, L=4.0), "set_L"),
    (lambda: TripleMomentum(mu=1.0, L=4.0), "set_mu"),
    (lambda: TripleMomentum(mu=1.0, L=4.0), "set_L"),
    (lambda: ITEM(mu=1.0, L=4.0), "set_mu"),
    (lambda: ITEM(mu=0.1, L=4.0), "set_L"),
    (lambda: OptimizedGradientMethod(L=1.0, K=1), "set_L"),
    (lambda: ChambollePock(tau=0.1, sigma=0.2, theta=0.0), "set_tau"),
    (lambda: ChambollePock(tau=0.1, sigma=0.2, theta=0.0), "set_sigma"),
]


FINITE_REAL_SETTERS = [
    (lambda: HeavyBallMethod(gamma=0.1, delta=0.0), "set_delta"),
    (lambda: GradientNesterovMomentum(gamma=0.1, delta=0.0), "set_delta"),
    (lambda: DavisYin(gamma=0.1, lambda_value=0.0), "set_lambda"),
    (lambda: DouglasRachford(gamma=0.1, lambda_value=0.0, type="operator"), "set_lambda"),
    (lambda: TsengFBF(gamma=0.1, theta=0.0), "set_theta"),
    (lambda: ChambollePock(tau=0.1, sigma=0.2, theta=0.0), "set_theta"),
]


@pytest.mark.parametrize(
    "factory,setter_name",
    POSITIVE_REAL_SETTERS,
    ids=[f"{factory().__class__.__name__}.{name}" for factory, name in POSITIVE_REAL_SETTERS],
)
def test_positive_real_setters_validate_input(factory, setter_name):
    algo = factory()
    setter = getattr(algo, setter_name)

    # Accepts positive finite values.
    setter(0.3)

    # Rejects non-positive, non-finite, and non-real inputs.
    for bad in [0.0, -1.0, np.inf, -np.inf, np.nan, "x", True]:
        with pytest.raises(ValueError):
            setter(bad)


@pytest.mark.parametrize(
    "factory,setter_name",
    FINITE_REAL_SETTERS,
    ids=[f"{factory().__class__.__name__}.{name}" for factory, name in FINITE_REAL_SETTERS],
)
def test_finite_real_setters_validate_input(factory, setter_name):
    algo = factory()
    setter = getattr(algo, setter_name)

    # Accepts finite reals (including negatives/zero for unconstrained parameters).
    setter(0.0)
    setter(-0.25)

    # Rejects non-finite and non-real inputs.
    for bad in [np.inf, -np.inf, np.nan, "x", True]:
        with pytest.raises(ValueError):
            setter(bad)


def test_optimized_gradient_method_set_k_validation():
    algo = OptimizedGradientMethod(L=1.0, K=1)
    algo.set_K(0)
    assert algo.K == 0
    algo.set_K(5)
    assert algo.K == 5

    for bad in [-1, 1.5, "3", True]:
        with pytest.raises(ValueError):
            algo.set_K(bad)


@pytest.mark.public_api
def test_optimized_gradient_method_compute_theta_validation():
    algo = OptimizedGradientMethod(L=1.0, K=5)
    assert algo.compute_theta(0, 5) == pytest.approx(1.0)
    assert algo.compute_theta(1, 5) == pytest.approx((1.0 + np.sqrt(5.0)) / 2.0)
    assert algo.compute_theta(5, 5) > algo.compute_theta(4, 5)

    for bad in [-1, 1.5, "3", True]:
        with pytest.raises(ValueError):
            algo.compute_theta(bad, 5)
        with pytest.raises(ValueError):
            algo.compute_theta(1, bad)
    with pytest.raises(ValueError):
        algo.compute_theta(6, 5)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: NesterovConstant(mu=0.0, L=4.0),
        lambda: NesterovConstant(mu=1.0, L=0.0),
        lambda: TripleMomentum(mu=0.0, L=4.0),
        lambda: TripleMomentum(mu=1.0, L=0.0),
        lambda: ITEM(mu=0.0, L=4.0),
        lambda: ITEM(mu=1.0, L=1.0),
        lambda: ITEM(mu=2.0, L=1.0),
        lambda: OptimizedGradientMethod(L=0.0, K=1),
        lambda: OptimizedGradientMethod(L=1.0, K=-1),
    ],
)
def test_constructors_validate_inputs(factory):
    with pytest.raises(ValueError):
        factory()


def test_item_requires_mu_less_than_l_in_setters():
    algo = ITEM(mu=1.0, L=4.0)
    algo.set_mu(3.0)
    algo.set_L(3.5)

    with pytest.raises(ValueError):
        algo.set_mu(3.5)
    with pytest.raises(ValueError):
        algo.set_L(3.0)


@pytest.mark.parametrize("method_name", ["get_A", "compute_beta", "compute_delta", "get_ABCD"])
def test_item_iteration_index_validation(method_name):
    algo = ITEM(mu=1.0, L=4.0)
    method = getattr(algo, method_name)

    for bad in [-1, 1.5, "3", True]:
        with pytest.raises(ValueError):
            method(bad)


def test_constructor_type_parameter_validation():
    with pytest.raises(ValueError):
        AcceleratedProximalPoint(gamma=0.1, type="bad")
    with pytest.raises(ValueError):
        DouglasRachford(gamma=0.1, lambda_value=0.0, type="bad")
    with pytest.raises(ValueError):
        Extragradient(gamma=0.1, delta=0.2, type="bad")


def test_constructor_type_selects_component_kinds():
    app_op = AcceleratedProximalPoint(gamma=0.1, type="operator")
    app_fn = AcceleratedProximalPoint(gamma=0.1, type="function")
    assert app_op.I_op == [1] and app_op.I_func == []
    assert app_fn.I_op == [] and app_fn.I_func == [1]

    dr_op = DouglasRachford(gamma=0.1, lambda_value=0.0, type="operator")
    dr_fn = DouglasRachford(gamma=0.1, lambda_value=0.0, type="function")
    assert dr_op.I_op == [1, 2] and dr_op.I_func == []
    assert dr_fn.I_op == [] and dr_fn.I_func == [1, 2]

    eg_unc = Extragradient(gamma=0.1, delta=0.2, type="unconstrained")
    eg_con = Extragradient(gamma=0.1, delta=0.2, type="constrained")
    assert eg_unc.I_op == [1] and eg_unc.I_func == []
    assert eg_con.I_op == [1] and eg_con.I_func == [2]
