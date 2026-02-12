.PHONY: check check-mosek docs

check:
	@bash scripts/check_local_ci.sh

check-mosek:
	@bash scripts/check_local_ci.sh --mosek-only

docs:
	@make -C docs html
