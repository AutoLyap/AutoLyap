import os
from typing import NoReturn

import pytest

_STRICT_MOSEK_ENV = "AUTOLYAP_STRICT_MOSEK"


def strict_mosek_mode_enabled() -> bool:
    value = os.environ.get(_STRICT_MOSEK_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def skip_or_fail_mosek(message: str) -> NoReturn:
    if strict_mosek_mode_enabled():
        pytest.fail(message, pytrace=False)
    pytest.skip(message)


# Shared MOSEK license gate used by multiple tests.
def require_mosek_license() -> None:
    try:
        import mosek.fusion as mf
    except ModuleNotFoundError:
        skip_or_fail_mosek("MOSEK Python package is not installed.")

    try:
        with mf.Model() as model:
            x = model.variable(1, mf.Domain.greaterThan(0.0))
            model.objective(mf.ObjectiveSense.Minimize, x)
            model.solve()
    except mf.OptimizeError as exc:
        error_text = str(exc).lower()
        license_markers = (
            "err_license_expired",
            "err_license_max",
            "err_license_server",
            "err_missing_license_file",
        )
        if any(marker in error_text for marker in license_markers):
            skip_or_fail_mosek(f"MOSEK license not available: {exc}")
        raise
