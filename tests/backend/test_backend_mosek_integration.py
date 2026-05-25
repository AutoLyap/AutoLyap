# SPDX-FileCopyrightText: 2025-2026 AutoLyap contributors
# SPDX-License-Identifier: GPL-3.0-only

import pytest

from tests.shared.mosek_utils import require_mosek_license

# Smoke test that verifies MOSEK license availability in CI environments.
@pytest.mark.mosek
def test_mosek_license_smoke():
    require_mosek_license()
