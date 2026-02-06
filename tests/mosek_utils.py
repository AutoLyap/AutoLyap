import pytest


# Shared MOSEK license gate used by multiple tests.
def require_mosek_license() -> None:
    pytest.importorskip("mosek")
    import mosek.fusion as mf

    try:
        with mf.Model() as model:
            x = model.variable(1, mf.Domain.greaterThan(0.0))
            model.objective(mf.ObjectiveSense.Minimize, x)
            model.solve()
    except mf.OptimizeError as exc:
        license_markers = (
            "err_license_max",
            "err_license_server",
            "err_missing_license_file",
        )
        if any(marker in str(exc) for marker in license_markers):
            pytest.skip(f"MOSEK license not available: {exc}")
        raise
