.PHONY: verify clean

SCRIPTS := 01_dag_dseparation.py 02_dowhy_identification.py 03_variational_mi_inequalities.py 04_algebraic_identity.py 05_consistency_rate.py 06_xor_construction.py

PYTHON ?= python3

verify:
	@set -e; for s in $(SCRIPTS); do \
		echo "=== $$s ==="; \
		$(PYTHON) $$s || exit 1; \
	done
	@echo
	@echo "All verification scripts passed."

clean:
	rm -f results/*.json
