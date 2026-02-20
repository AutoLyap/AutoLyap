.PHONY: check check-mosek check-sdpa check-sdpa-multiprecision docs sync-citation check-citation

check:
	@bash scripts/check_local_ci.sh

check-mosek:
	@bash scripts/check_local_ci.sh --mosek-only

check-sdpa:
	@python -m pytest tests/convergence/test_convergence_cvxpy_*.py -m "sdpa and not sdpa_multiprecision"

check-sdpa-multiprecision:
	@python -m pytest tests/convergence/test_convergence_cvxpy_*.py -m "sdpa_multiprecision"

docs:
	@make -C docs dirhtml

sync-citation:
	@python scripts/sync_citation_version.py

check-citation:
	@python scripts/sync_citation_version.py --check
