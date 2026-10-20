PYTHON ?= python3
.PHONY: test eval eval-sync eval-check run
test:
	$(PYTHON) -m pytest -q
eval:
	$(PYTHON) -m eval.run_eval
eval-sync:
	$(PYTHON) -m eval.sync_readme
eval-check:
	$(PYTHON) -m eval.sync_readme --check
run:
	$(PYTHON) -m uvicorn finpilot.api:app --reload
