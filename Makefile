.PHONY: check check-mosek docs sync-citation check-citation

check:
	@bash scripts/check_local_ci.sh

check-mosek:
	@bash scripts/check_local_ci.sh --mosek-only

docs:
	@make -C docs dirhtml

sync-citation:
	@python scripts/sync_citation_version.py

check-citation:
	@python scripts/sync_citation_version.py --check
