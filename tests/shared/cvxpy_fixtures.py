"""Shared fixtures for CVXPY backend tests."""

import pytest

from tests.shared.cvxpy_test_utils import (
    make_cvxpy_mosek_options,
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
def cvxpy_mosek_solver_options():
    require_cvxpy_mosek_solver()
    return make_cvxpy_mosek_options()


@pytest.fixture(scope="module")
def mosek_fusion_solver_options():
    return make_mosek_fusion_options()
