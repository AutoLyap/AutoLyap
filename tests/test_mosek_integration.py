import pytest

from tests.mosek_utils import require_mosek_license

# Smoke test that verifies MOSEK license availability in CI environments.
@pytest.mark.mosek
def test_mosek_license_smoke():
    require_mosek_license()
