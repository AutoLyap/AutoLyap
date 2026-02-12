#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Run local checks that mirror the GitHub Actions PR workflow.

Usage:
  bash scripts/check_local_ci.sh [--docs]

Options:
  --docs    Also build Sphinx docs (equivalent to `make -C docs html`).
  -h, --help
EOF
}

run_docs=0
for arg in "$@"; do
  case "$arg" in
    --docs)
      run_docs=1
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

if ! python -m ruff --version >/dev/null 2>&1; then
  echo "Missing dependency: ruff" >&2
  echo "Run: python -m pip install ruff mypy" >&2
  exit 1
fi

if ! python -m mypy --version >/dev/null 2>&1; then
  echo "Missing dependency: mypy" >&2
  echo "Run: python -m pip install ruff mypy" >&2
  exit 1
fi

echo "[1/3] Ruff checks"
python -m ruff check autolyap tests setup.py --select E9,F63,F7,F82

echo "[2/3] Mypy checks"
python -m mypy \
  autolyap/solver_options.py \
  autolyap/problemclass \
  --ignore-missing-imports \
  --follow-imports=silent

echo "[3/3] Pytest full suite checks"
python -m pytest

if [ "$run_docs" -eq 1 ]; then
  echo "[4/4] Docs build"
  make -C docs html
fi

echo "All checks passed."
