.PHONY: check check-docs

check:
	@bash scripts/check_local_ci.sh

check-docs:
	@bash scripts/check_local_ci.sh --docs
