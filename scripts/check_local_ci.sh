#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Run local checks that mirror the GitHub Actions PR workflow.
Default mode enforces strict MOSEK validation and fails if MOSEK is unavailable.

Usage:
  bash scripts/check_local_ci.sh [--mosek-only]

Options:
  --mosek-only  Run strict MOSEK-only validation (fails on unavailable/skipped MOSEK checks).
  -h, --help
EOF
}

mosek_only=0
for arg in "$@"; do
  case "$arg" in
    --mosek-only)
      mosek_only=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      usage >&2
      exit 1
      ;;
  esac
done

has_mosek_license() {
  python - <<'PY' >/dev/null 2>&1
import mosek.fusion as mf

with mf.Model() as model:
    x = model.variable(1, mf.Domain.greaterThan(0.0))
    model.objective(mf.ObjectiveSense.Minimize, x)
    model.solve()
PY
}

run_strict_mosek_suite() {
  local report_file
  report_file="$(mktemp "${TMPDIR:-/tmp}/pytest-mosek.XXXXXX")"
  trap 'rm -f "$report_file"' RETURN

  AUTOLYAP_STRICT_MOSEK=1 python -m pytest -m "mosek" --junitxml "$report_file"

  PYTEST_MOSEK_REPORT="$report_file" python - <<'PY'
import os
import sys
import xml.etree.ElementTree as ET

report_path = os.environ["PYTEST_MOSEK_REPORT"]
root = ET.parse(report_path).getroot()
if root.tag == "testsuite":
    skipped = int(root.attrib.get("skipped", 0))
else:
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in root.findall("testsuite"))
if skipped:
    print(f"Strict MOSEK validation requires executed tests; found {skipped} skipped tests.", file=sys.stderr)
    sys.exit(1)
print("Strict MOSEK suite completed with zero skipped tests.")
PY
}

if [ "$mosek_only" -eq 1 ]; then
  echo "[1/1] Pytest MOSEK suite checks (strict)"
  if has_mosek_license; then
    run_strict_mosek_suite
  else
    cat <<'EOF' >&2
MOSEK is not available in this local environment.
Install/configure MOSEK and a valid license, then rerun:
  make check-mosek
EOF
    exit 1
  fi
  echo "All checks passed."
  exit 0
fi

if ! python -m ruff --version >/dev/null 2>&1; then
  echo "Missing dependency: ruff" >&2
  echo "Run: python -m pip install -e '.[test]'" >&2
  exit 1
fi

if ! python -m mypy --version >/dev/null 2>&1; then
  echo "Missing dependency: mypy" >&2
  echo "Run: python -m pip install -e '.[test]'" >&2
  exit 1
fi

echo "[1/6] Validate CITATION.cff version matches VERSION"
python scripts/sync_citation_version.py --check

echo "[2/6] Ruff checks"
python -m ruff check autolyap tests setup.py --select E9,F63,F7,F82

echo "[3/6] Ruff core lint policy checks"
python -m ruff check \
  autolyap/solver_options.py \
  autolyap/iteration_independent.py \
  autolyap/iteration_dependent.py \
  autolyap/algorithms/algorithm.py

echo "[4/6] Mypy checks"
python -m mypy \
  autolyap/solver_options.py \
  autolyap/iteration_independent.py \
  autolyap/iteration_dependent.py \
  autolyap/algorithms/algorithm.py \
  autolyap/problemclass \
  --ignore-missing-imports \
  --follow-imports=silent

echo "[5/6] Pytest core suite checks"
python -m pytest -m "not mosek and not scs and not copt and not sdpa and not sdpa_multiprecision"

echo "[6/6] Pytest MOSEK suite checks (strict)"
if has_mosek_license; then
  run_strict_mosek_suite
else
  cat <<'EOF' >&2
MOSEK is not available in this local environment.
`make check` enforces strict MOSEK validation to mirror required CI checks.
Install/configure MOSEK and a valid license, then rerun:
  make check

To debug MOSEK setup independently, run:
  make check-mosek
EOF
  exit 1
fi

echo "All checks passed."
