"""Shared fixtures for CVXPY backend tests."""

import pytest

from tests.shared.cvxpy_test_utils import (
    require_cvxpy_clarabel_solver,
    require_cvxpy_sdpa_solver,
    require_cvxpy_sdpa_multiprecision_solver,
    make_cvxpy_mosek_options,
    make_cvxpy_sdpa_options,
    make_cvxpy_solver_options,
    make_mosek_fusion_options,
    require_cvxpy_module,
    require_cvxpy_mosek_solver,
    require_open_source_cvxpy_solver,
)


@pytest.fixture(scope="module")
def cvxpy_module():
    return require_cvxpy_module()


@pytest.fixture(scope="module")
def cvxpy_open_source_solver_name():
    return require_open_source_cvxpy_solver()


@pytest.fixture(scope="module")
def cvxpy_open_source_solver_options():
    solver_name = require_open_source_cvxpy_solver()
    extra = {"max_iter": 400} if solver_name == "CLARABEL" else None
    return make_cvxpy_solver_options(solver_name, extra_params=extra)


@pytest.fixture(scope="module")
def cvxpy_clarabel_solver_name():
    return require_cvxpy_clarabel_solver()


@pytest.fixture(scope="module")
def cvxpy_clarabel_solver_options():
    require_cvxpy_clarabel_solver()
    return make_cvxpy_solver_options("CLARABEL", extra_params={"max_iter": 400})


@pytest.fixture(
    scope="module",
    params=[
        pytest.param("clarabel", id="clarabel"),
        pytest.param("sdpa", marks=pytest.mark.sdpa, id="sdpa"),
        pytest.param(
            "sdpa_multiprecision",
            marks=pytest.mark.sdpa_multiprecision,
            id="sdpa_multiprecision",
        ),
    ],
)
def cvxpy_convergence_solver_options(request):
    profile = request.param
    if profile == "clarabel":
        require_cvxpy_clarabel_solver()
        return make_cvxpy_solver_options("CLARABEL", extra_params={"max_iter": 400})
    if profile == "sdpa":
        require_cvxpy_sdpa_solver()
        return make_cvxpy_sdpa_options(
            extra_params={
                "maxIteration": 400,
                "epsilonStar": 1e-8,
                "epsilonDash": 1e-8,
            }
        )
    if profile == "sdpa_multiprecision":
        require_cvxpy_sdpa_multiprecision_solver()
        return make_cvxpy_sdpa_options(
            multiprecision=True,
            extra_params={
                "maxIteration": 500,
                "epsilonStar": 1e-30,
                "epsilonDash": 1e-30,
                "mpfPrecision": 512,
            },
        )
    raise RuntimeError(f"Unsupported convergence solver profile: {profile}")


@pytest.fixture(scope="module")
def cvxpy_sdpa_solver_name():
    return require_cvxpy_sdpa_solver()


@pytest.fixture(scope="module")
def cvxpy_sdpa_solver_options():
    require_cvxpy_sdpa_solver()
    return make_cvxpy_sdpa_options()


@pytest.fixture(scope="module")
def cvxpy_sdpa_multiprecision_solver_options():
    require_cvxpy_sdpa_multiprecision_solver()
    return make_cvxpy_sdpa_options(multiprecision=True)


@pytest.fixture(scope="module")
def cvxpy_mosek_solver_options():
    require_cvxpy_mosek_solver()
    return make_cvxpy_mosek_options()


@pytest.fixture(scope="module")
def mosek_fusion_solver_options():
    return make_mosek_fusion_options()
