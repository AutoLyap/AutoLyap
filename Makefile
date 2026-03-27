.PHONY: check check-mosek check-clarabel check-scs check-copt check-sdpa check-sdpa-multiprecision docs sync-citation check-citation sync-notebooks check-notebooks

check:
	@bash scripts/check_local_ci.sh

check-mosek:
	@bash scripts/check_local_ci.sh --mosek-only

check-clarabel:
	@python -m pytest tests/convergence/test_convergence_cvxpy_*.py -m "clarabel"

check-scs:
	@python -m pytest tests/convergence/test_convergence_cvxpy_*.py -m "scs"

check-copt:
	@python -m pytest tests/convergence/test_convergence_cvxpy_*.py -m "copt"

check-sdpa:
	@python -m pytest tests/convergence/test_convergence_cvxpy_*.py -m "sdpa and not sdpa_multiprecision"

check-sdpa-multiprecision:
	@python -m pytest tests/convergence/test_convergence_cvxpy_*.py -m "sdpa_multiprecision"

docs:
	@$(MAKE) -C docs dirhtml

sync-citation:
	@python scripts/sync_citation_version.py

check-citation:
	@python scripts/sync_citation_version.py --check

sync-notebooks:
	@python scripts/generate_quick_start_notebook.py

check-notebooks:
	@python scripts/generate_quick_start_notebook.py --check
